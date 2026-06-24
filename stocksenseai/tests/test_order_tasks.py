import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tasks.order_tasks import _poll_async, _update_portfolio


async def test_update_portfolio_applies_only_new_fill_quantity():
    holding = SimpleNamespace(quantity=10, avg_price=100)
    trade = SimpleNamespace(
        user_id=uuid.uuid4(),
        stock_code="005930",
        stock_name="삼성전자",
        mode="paper",
        order_type="BUY",
        realized_pnl=None,
    )
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = holding
    db.execute.return_value = result

    await _update_portfolio(db, trade, executed_price=120, fill_quantity=2)

    assert holding.quantity == 12
    assert holding.avg_price == pytest.approx(103.33, abs=0.01)


async def test_poll_uses_trade_mode_and_applies_fill_delta():
    trade = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        stock_code="005930",
        stock_name="삼성전자",
        order_type="BUY",
        quantity=10,
        mode="paper",
        status="PENDING",
        filled_quantity=3,
        executed_price=None,
        filled_at=None,
        realized_pnl=None,
    )
    user = SimpleNamespace(id=trade.user_id, mode="real")
    db = AsyncMock()

    trade_result = MagicMock()
    trade_result.scalar_one_or_none.return_value = trade
    user_result = MagicMock()
    user_result.scalar_one_or_none.return_value = user
    db.execute.side_effect = [trade_result, user_result]

    session_context = AsyncMock()
    session_context.__aenter__.return_value = db
    task = MagicMock()

    with patch("core.database.AsyncSessionLocal", return_value=session_context):
        with patch(
            "services.kis_service.poll_fill",
            new_callable=AsyncMock,
            return_value={"executed_price": 120, "filled_qty": 5, "filled_at": None},
        ) as poll_fill:
            with patch(
                "tasks.order_tasks._update_portfolio", new_callable=AsyncMock
            ) as update_portfolio:
                with patch("tasks.email_tasks.send_fill_notification"):
                    await _poll_async(
                        task,
                        str(trade.id),
                        str(trade.user_id),
                        "12345",
                        "paper",
                    )

    poll_fill.assert_awaited_once_with(user, "12345", mode="paper")
    update_portfolio.assert_awaited_once_with(db, trade, 120, 2)
    assert trade.filled_quantity == 5
    assert trade.status == "PARTIALLY_FILLED"
