import asyncio
import logging

from tasks import celery_app

from core.database import AsyncSessionLocal
from core.redis_client import get_redis
from models.risk import AlertSettings
from models.trade import Trade
from models.watchlist import WatchlistItem
from services.market_service import get_stock_current_price
from services.risk_service import (
    _get_portfolio_total,
    _get_today_loss,
    get_or_create_settings,
)
from sqlalchemy import select

_logger = logging.getLogger(__name__)


def _get_notification_email(user_id: str) -> str | None:
    """notification_email 설정 or user.email fallback."""
    import uuid
    from core.database import AsyncSessionLocal
    from models.risk import AlertSettings
    from models.user import User
    from sqlalchemy import select

    async def _fetch():
        async with AsyncSessionLocal() as db:
            uid = uuid.UUID(user_id)
            result = await db.execute(select(AlertSettings).where(AlertSettings.user_id == uid))
            alert = result.scalar_one_or_none()
            if alert and alert.notification_email:
                return alert.notification_email
            result2 = await db.execute(select(User).where(User.id == uid))
            user = result2.scalar_one_or_none()
            return user.email if user else None

    return asyncio.run(_fetch())


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_fill_notification(self, user_id: str, trade_id: str) -> None:
    """체결 완료 이메일."""
    import asyncio
    import uuid
    from core.database import AsyncSessionLocal
    from models.trade import Trade
    from sqlalchemy import select

    async def _fetch_trade():
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Trade).where(Trade.id == uuid.UUID(trade_id)))
            trade = result.scalar_one_or_none()
            alert_result = await db.execute(
                select(AlertSettings).where(AlertSettings.user_id == uuid.UUID(user_id))
            )
            return trade, alert_result.scalar_one_or_none()

    trade, alert_settings = asyncio.run(_fetch_trade())
    if not trade:
        return
    if alert_settings and not alert_settings.trade_filled:
        return

    to_email = _get_notification_email(user_id)
    if not to_email:
        return

    order_type_kr = "매수" if trade.order_type == "BUY" else "매도"
    subject = f"[StockSenseAI] {trade.stock_code} {order_type_kr} 체결 완료"
    body = (
        f"<p>{trade.stock_code} {trade.stock_name or ''} {order_type_kr} 체결되었습니다.</p>"
        f"<p>체결가: {trade.executed_price:,}원 | 수량: {trade.filled_quantity}주</p>"
    )

    try:
        from core.config import settings
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        if not settings.SENDGRID_API_KEY:
            return
        msg = Mail(from_email=settings.FROM_EMAIL, to_emails=to_email,
                   subject=subject, html_content=body)
        client = SendGridAPIClient(settings.SENDGRID_API_KEY)
        asyncio.run(asyncio.to_thread(client.send, msg))
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_risk_alert(self, user_id: str, reason: str) -> None:
    """리스크 한도 초과 경고 이메일."""
    to_email = _get_notification_email(user_id)
    if not to_email:
        return
    try:
        import asyncio
        from core.config import settings
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        if not settings.SENDGRID_API_KEY:
            return
        msg = Mail(
            from_email=settings.FROM_EMAIL,
            to_emails=to_email,
            subject="[StockSenseAI] 리스크 한도 초과 경고",
            html_content=f"<p>리스크 한도 초과: {reason}</p>",
        )
        client = SendGridAPIClient(settings.SENDGRID_API_KEY)
        asyncio.run(asyncio.to_thread(client.send, msg))
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task
def check_price_alerts() -> None:
    """관심종목 목표가 도달 시 이메일 — APScheduler 5분 간격."""
    asyncio.run(_check_price_alerts_async())


async def _check_price_alerts_async() -> None:
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(WatchlistItem).where(
                (WatchlistItem.target_price_high.isnot(None))
                | (WatchlistItem.target_price_low.isnot(None))
            )
        )
        items = res.scalars().all()
        if not items:
            return

        alert_res = await db.execute(
            select(AlertSettings).where(
                AlertSettings.user_id.in_([i.user_id for i in items])
            )
        )
        alert_map = {str(a.user_id): a for a in alert_res.scalars().all()}

        price_map: dict[str, int] = {}
        for code in {i.stock_code for i in items}:
            try:
                data = await get_stock_current_price(code)
                price_map[code] = data.get("close", 0)
            except Exception as exc:
                _logger.warning("가격 조회 실패 (code=%s): %s", code, exc)

        redis = await get_redis()

        for item in items:
            uid_str = str(item.user_id)
            cfg = alert_map.get(uid_str)
            if cfg and not cfg.watchlist_price:
                continue

            current = price_map.get(item.stock_code, 0)
            if current == 0:
                continue

            if item.target_price_high and current >= float(item.target_price_high):
                key = f"price_alert:{uid_str}:{item.stock_code}:high"
                if not await redis.exists(key):
                    await redis.setex(key, 86400, "1")
                    send_price_alert_email.delay(
                        user_id=uid_str,
                        stock_code=item.stock_code,
                        stock_name=item.stock_name or item.stock_code,
                        current_price=current,
                        target_price=float(item.target_price_high),
                        alert_type="high",
                    )

            if item.target_price_low and current <= float(item.target_price_low):
                key = f"price_alert:{uid_str}:{item.stock_code}:low"
                if not await redis.exists(key):
                    await redis.setex(key, 86400, "1")
                    send_price_alert_email.delay(
                        user_id=uid_str,
                        stock_code=item.stock_code,
                        stock_name=item.stock_name or item.stock_code,
                        current_price=current,
                        target_price=float(item.target_price_low),
                        alert_type="low",
                    )


@celery_app.task
def check_daily_loss() -> None:
    """일일 손실 한도 체크 — APScheduler 10분 간격."""
    asyncio.run(_check_daily_loss_async())


async def _check_daily_loss_async() -> None:
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    _KST = ZoneInfo("Asia/Seoul")

    async with AsyncSessionLocal() as db:
        kst_now = datetime.now(_KST)
        today_start = kst_now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)

        res = await db.execute(
            select(Trade.user_id, Trade.mode)
            .where(
                Trade.order_type == "SELL",
                Trade.status == "FILLED",
                Trade.filled_at >= today_start,
            )
            .distinct()
        )
        user_mode_pairs = res.all()
        if not user_mode_pairs:
            return

        redis = await get_redis()

        for user_id, mode in user_mode_pairs:
            uid_str = str(user_id)

            settings = await get_or_create_settings(user_id, db)
            if settings.trading_blocked:
                continue

            redis_key = f"daily_loss_alert:{uid_str}:{mode}"
            if await redis.exists(redis_key):
                continue

            alert_res = await db.execute(
                select(AlertSettings).where(AlertSettings.user_id == user_id)
            )
            alert_cfg = alert_res.scalar_one_or_none()
            if alert_cfg and not alert_cfg.daily_loss_limit:
                continue

            today_loss = await _get_today_loss(user_id, mode, db)
            portfolio_total = await _get_portfolio_total(user_id, mode, db)
            if portfolio_total == 0:
                continue

            loss_pct = today_loss / portfolio_total * 100
            limit_pct = float(settings.daily_loss_limit_pct)

            if loss_pct > limit_pct:
                if mode == "real":
                    settings.trading_blocked = True
                    settings.blocked_at = datetime.now(timezone.utc)
                    await db.commit()
                await redis.setex(redis_key, 86400, "1")
                reason = f"일일 손실 {loss_pct:.1f}% > 한도 {limit_pct:.1f}% (mode={mode})"
                send_risk_alert.delay(uid_str, reason)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_price_alert_email(
    self,
    user_id: str,
    stock_code: str,
    stock_name: str,
    current_price: float,
    target_price: float,
    alert_type: str,
) -> None:
    """목표가/손절가 도달 이메일. alert_type: 'high' | 'low'"""
    to_email = _get_notification_email(user_id)
    if not to_email:
        return

    label = "목표가 도달" if alert_type == "high" else "손절가 도달"
    subject = f"[StockSenseAI] {stock_name}({stock_code}) {label}"
    body = (
        f"<p><strong>{stock_name}</strong>({stock_code}) {label}!</p>"
        f"<p>현재가: <strong>{int(current_price):,}원</strong> "
        f"/ 설정가: {int(target_price):,}원</p>"
    )

    try:
        from core.config import settings as app_settings
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        if not app_settings.SENDGRID_API_KEY:
            return
        msg = Mail(
            from_email=app_settings.FROM_EMAIL,
            to_emails=to_email,
            subject=subject,
            html_content=body,
        )
        client = SendGridAPIClient(app_settings.SENDGRID_API_KEY)
        asyncio.run(asyncio.to_thread(client.send, msg))
    except Exception as exc:
        raise self.retry(exc=exc)
