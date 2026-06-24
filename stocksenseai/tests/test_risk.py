# tests/test_risk.py
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_risk_settings(
    max_per_stock_pct=20.0,
    daily_loss_limit_pct=5.0,
    enforce_hard_stop=True,
    trading_blocked=False,
):
    rs = MagicMock()
    rs.max_per_stock_pct = Decimal(str(max_per_stock_pct))
    rs.daily_loss_limit_pct = Decimal(str(daily_loss_limit_pct))
    rs.enforce_hard_stop = enforce_hard_stop
    rs.trading_blocked = trading_blocked
    return rs


async def test_check_order_passes_within_limit(client):
    """한도 미초과 주문은 통과."""
    from services.risk_service import check_order

    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    db = AsyncMock()

    with patch("services.risk_service.get_or_create_settings", new_callable=AsyncMock,
               return_value=_make_risk_settings(max_per_stock_pct=50.0)):
        with patch("services.risk_service._get_portfolio_total", new_callable=AsyncMock, return_value=1_000_000):
            with patch("services.risk_service._get_holding_value", new_callable=AsyncMock, return_value=100_000):
                with patch("services.risk_service._get_today_loss", new_callable=AsyncMock, return_value=0):
                    result = await check_order(user, "005930", "BUY", 1, 50_000, db)
                    assert result is None


async def test_check_order_hard_stop_raises(client):
    """enforce_hard_stop=True이고 한도 초과 시 400."""
    from fastapi import HTTPException
    from services.risk_service import check_order

    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    db = AsyncMock()

    with patch("services.risk_service.get_or_create_settings", new_callable=AsyncMock,
               return_value=_make_risk_settings(max_per_stock_pct=10.0, enforce_hard_stop=True)):
        with patch("services.risk_service._get_portfolio_total", new_callable=AsyncMock, return_value=1_000_000):
            with patch("services.risk_service._get_holding_value", new_callable=AsyncMock, return_value=500_000):
                with patch("services.risk_service._get_today_loss", new_callable=AsyncMock, return_value=0):
                    with pytest.raises(HTTPException) as exc_info:
                        await check_order(user, "005930", "BUY", 1, 50_000, db)
                    assert exc_info.value.status_code == 400


async def test_check_order_warning_mode(client):
    """enforce_hard_stop=False이고 한도 초과 시 경고 문자열 반환."""
    from services.risk_service import check_order

    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    db = AsyncMock()

    with patch("services.risk_service.get_or_create_settings", new_callable=AsyncMock,
               return_value=_make_risk_settings(max_per_stock_pct=10.0, enforce_hard_stop=False)):
        with patch("services.risk_service._get_portfolio_total", new_callable=AsyncMock, return_value=1_000_000):
            with patch("services.risk_service._get_holding_value", new_callable=AsyncMock, return_value=500_000):
                with patch("services.risk_service._get_today_loss", new_callable=AsyncMock, return_value=0):
                    result = await check_order(user, "005930", "BUY", 1, 50_000, db)
                    assert result is not None
                    assert "한도" in result


async def test_check_order_sell_allowed_while_trading_blocked(client):
    """리스크 차단 상태에서도 포지션 청산을 위한 매도는 허용."""
    from services.risk_service import check_order

    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    db = AsyncMock()

    with patch(
        "services.risk_service.get_or_create_settings",
        new_callable=AsyncMock,
        return_value=_make_risk_settings(trading_blocked=True),
    ):
        result = await check_order(user, "005930", "SELL", 1, 50_000, db)

    assert result is None


async def test_get_risk_settings_returns_200(client):
    """인증된 유저는 리스크 설정 조회 가능."""
    from main import app
    from api.deps import get_current_user

    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    app.dependency_overrides[get_current_user] = lambda: user

    mock_settings = _make_risk_settings()
    mock_settings.stop_loss_enabled = True
    mock_settings.trading_blocked = False

    with patch("api.routes.risk.get_or_create_settings", new_callable=AsyncMock,
               return_value=mock_settings):
        resp = await client.get("/risk/settings")

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    data = resp.json()
    assert "max_per_stock_pct" in data
    assert "enforce_hard_stop" in data


async def test_put_risk_settings_returns_200(client):
    """리스크 설정 수정."""
    from main import app
    from api.deps import get_current_user

    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    app.dependency_overrides[get_current_user] = lambda: user

    mock_settings = _make_risk_settings()
    mock_settings.max_per_stock_pct = 20.0
    mock_settings.daily_loss_limit_pct = 5.0
    mock_settings.stop_loss_enabled = False
    mock_settings.enforce_hard_stop = True
    mock_settings.trading_blocked = False

    with patch("api.routes.risk.get_or_create_settings", new_callable=AsyncMock,
               return_value=mock_settings):
        resp = await client.put("/risk/settings", json={"max_per_stock_pct": 30.0})

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json()["updated"] is True
