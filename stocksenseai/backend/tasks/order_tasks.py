import asyncio
import uuid

from tasks import celery_app


@celery_app.task(bind=True, max_retries=5, default_retry_delay=10)
def poll_order_fill(self, trade_id: str, user_id: str, kis_order_no: str, mode: str) -> None:
    """KIS 체결 폴링 (10초 간격, 최대 5회). 체결 시 trades/portfolios 업데이트 + 이메일."""
    asyncio.run(_poll_async(self, trade_id, user_id, kis_order_no, mode))


async def _poll_async(task, trade_id: str, user_id: str, kis_order_no: str, mode: str) -> None:
    from core.database import AsyncSessionLocal
    from models.portfolio import Portfolio
    from models.trade import Trade
    from models.user import User
    from services import kis_service
    from sqlalchemy import select
    from tasks.email_tasks import send_fill_notification

    trade_uuid = uuid.UUID(trade_id)
    user_uuid = uuid.UUID(user_id)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Trade).where(Trade.id == trade_uuid).with_for_update()
        )
        trade = result.scalar_one_or_none()
        if not trade or trade.status in {"FILLED", "CANCELLED"}:
            return

        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
        if not user:
            return

        fill = await kis_service.poll_fill(user, kis_order_no, mode=trade.mode)
        if fill is None:
            if task.request.retries < task.max_retries:
                raise task.retry(countdown=10)
            if trade.filled_quantity == 0:
                trade.status = "UNKNOWN"
            await db.commit()
            return

        cumulative_filled = min(fill["filled_qty"], trade.quantity)
        fill_delta = max(0, cumulative_filled - trade.filled_quantity)
        if fill_delta:
            await _update_portfolio(db, trade, fill["executed_price"], fill_delta)
            trade.filled_quantity = cumulative_filled

        trade.executed_price = fill["executed_price"]
        trade.filled_at = fill.get("filled_at")
        trade.status = "FILLED" if cumulative_filled >= trade.quantity else "PARTIALLY_FILLED"
        await db.commit()

    if trade.status == "FILLED":
        send_fill_notification.delay(user_id, trade_id)


async def _update_portfolio(db, trade, executed_price: int, fill_quantity: int) -> None:
    """체결 후 portfolios 테이블 UPSERT."""
    from models.portfolio import Portfolio
    from sqlalchemy import select

    result = await db.execute(
        select(Portfolio).where(
            Portfolio.user_id == trade.user_id,
            Portfolio.stock_code == trade.stock_code,
            Portfolio.mode == trade.mode,
        )
    )
    holding = result.scalar_one_or_none()

    if trade.order_type == "BUY":
        if holding is None:
            db.add(Portfolio(
                user_id=trade.user_id,
                stock_code=trade.stock_code,
                stock_name=trade.stock_name,
                quantity=fill_quantity,
                avg_price=executed_price,
                mode=trade.mode,
            ))
        else:
            total_qty = holding.quantity + fill_quantity
            new_avg = (holding.avg_price * holding.quantity + executed_price * fill_quantity) / total_qty
            holding.quantity = total_qty
            holding.avg_price = round(new_avg, 2)
    elif trade.order_type == "SELL" and holding:
        # 체결 전 평균매수가 기준 실현손익 계산
        realized_delta = int((executed_price - float(holding.avg_price)) * fill_quantity)
        trade.realized_pnl = (trade.realized_pnl or 0) + realized_delta
        holding.quantity -= fill_quantity
        if holding.quantity <= 0:
            await db.delete(holding)
