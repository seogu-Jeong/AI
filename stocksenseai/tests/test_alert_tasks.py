# tests/test_alert_tasks.py
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_item(user_id, stock_code="005930", high=None, low=None):
    item = MagicMock()
    item.user_id = user_id
    item.stock_code = stock_code
    item.stock_name = "삼성전자"
    item.target_price_high = high
    item.target_price_low = low
    return item


def _make_alert_cfg(watchlist_price=True, daily_loss_limit=True):
    cfg = MagicMock()
    cfg.watchlist_price = watchlist_price
    cfg.daily_loss_limit = daily_loss_limit
    return cfg


def _mock_db_for_price_alerts(items, alert_cfgs):
    db = AsyncMock()
    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = items
    alerts_result = MagicMock()
    alerts_result.scalars.return_value.all.return_value = alert_cfgs
    db.execute.side_effect = [items_result, alerts_result]
    return db


async def test_price_alert_high_triggered():
    """현재가 ≥ 목표가 → send_price_alert_email.delay 호출."""
    from tasks.email_tasks import _check_price_alerts_async

    user_id = uuid.uuid4()
    item = _make_item(user_id, high=75000.0)
    alert_cfg = _make_alert_cfg()
    alert_cfg.user_id = user_id

    db = _mock_db_for_price_alerts([item], [alert_cfg])

    mock_redis = AsyncMock()
    mock_redis.exists.return_value = False

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_stock_current_price",
                new_callable=AsyncMock,
                return_value={"close": 76000},
            ):
                with patch("tasks.email_tasks.send_price_alert_email") as mock_send:
                    await _check_price_alerts_async()

    mock_send.delay.assert_called_once()
    kwargs = mock_send.delay.call_args.kwargs
    assert kwargs["alert_type"] == "high"
    assert kwargs["stock_code"] == "005930"
    assert kwargs["current_price"] == 76000


async def test_price_alert_cooldown_skip():
    """Redis 쿨다운 키 있음 → delay 미호출."""
    from tasks.email_tasks import _check_price_alerts_async

    user_id = uuid.uuid4()
    item = _make_item(user_id, high=75000.0)
    alert_cfg = _make_alert_cfg()
    alert_cfg.user_id = user_id

    db = _mock_db_for_price_alerts([item], [alert_cfg])
    mock_redis = AsyncMock()
    mock_redis.exists.return_value = True  # 쿨다운 키 존재

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_stock_current_price",
                new_callable=AsyncMock,
                return_value={"close": 76000},
            ):
                with patch("tasks.email_tasks.send_price_alert_email") as mock_send:
                    await _check_price_alerts_async()

    mock_send.delay.assert_not_called()


async def test_price_alert_setting_disabled():
    """watchlist_price=False → delay 미호출."""
    from tasks.email_tasks import _check_price_alerts_async

    user_id = uuid.uuid4()
    item = _make_item(user_id, high=75000.0)
    alert_cfg = _make_alert_cfg(watchlist_price=False)
    alert_cfg.user_id = user_id

    db = _mock_db_for_price_alerts([item], [alert_cfg])
    mock_redis = AsyncMock()
    mock_redis.exists.return_value = False

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_stock_current_price",
                new_callable=AsyncMock,
                return_value={"close": 76000},
            ):
                with patch("tasks.email_tasks.send_price_alert_email") as mock_send:
                    await _check_price_alerts_async()

    mock_send.delay.assert_not_called()


async def test_daily_loss_blocks_trading():
    """loss_pct > limit → trading_blocked=True + send_risk_alert.delay 호출."""
    from tasks.email_tasks import _check_daily_loss_async

    user_id = uuid.uuid4()

    # Mock trade result (user_id, mode) — real 모드에서만 trading_blocked 설정됨
    trade_result = MagicMock()
    trade_result.all.return_value = [(user_id, "real")]

    # Mock alert settings (daily_loss_limit=True)
    alert_result = MagicMock()
    alert_cfg = _make_alert_cfg(daily_loss_limit=True)
    alert_result.scalar_one_or_none.return_value = alert_cfg

    db = AsyncMock()
    db.execute.side_effect = [trade_result, alert_result]

    mock_settings = MagicMock()
    mock_settings.trading_blocked = False
    mock_settings.daily_loss_limit_pct = 5.0

    mock_redis = AsyncMock()
    mock_redis.exists.return_value = False

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_or_create_settings",
                new_callable=AsyncMock,
                return_value=mock_settings,
            ):
                with patch(
                    "tasks.email_tasks._get_today_loss",
                    new_callable=AsyncMock,
                    return_value=600_000,       # 60만원 손실
                ):
                    with patch(
                        "tasks.email_tasks._get_portfolio_total",
                        new_callable=AsyncMock,
                        return_value=10_000_000,  # 포트폴리오 1천만원
                    ):
                        with patch("tasks.email_tasks.send_risk_alert") as mock_alert:
                            await _check_daily_loss_async()

    assert mock_settings.trading_blocked is True
    mock_alert.delay.assert_called_once()


async def test_daily_loss_within_limit():
    """loss_pct ≤ limit → 차단 없음."""
    from tasks.email_tasks import _check_daily_loss_async

    user_id = uuid.uuid4()

    trade_result = MagicMock()
    trade_result.all.return_value = [(user_id, "paper")]

    alert_result = MagicMock()
    alert_cfg = _make_alert_cfg(daily_loss_limit=True)
    alert_result.scalar_one_or_none.return_value = alert_cfg

    db = AsyncMock()
    db.execute.side_effect = [trade_result, alert_result]

    mock_settings = MagicMock()
    mock_settings.trading_blocked = False
    mock_settings.daily_loss_limit_pct = 5.0

    mock_redis = AsyncMock()
    mock_redis.exists.return_value = False

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_or_create_settings",
                new_callable=AsyncMock,
                return_value=mock_settings,
            ):
                with patch(
                    "tasks.email_tasks._get_today_loss",
                    new_callable=AsyncMock,
                    return_value=100_000,        # 10만원 손실 (1%)
                ):
                    with patch(
                        "tasks.email_tasks._get_portfolio_total",
                        new_callable=AsyncMock,
                        return_value=10_000_000,
                    ):
                        with patch("tasks.email_tasks.send_risk_alert") as mock_alert:
                            await _check_daily_loss_async()

    assert mock_settings.trading_blocked is False
    mock_alert.delay.assert_not_called()
