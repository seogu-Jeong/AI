import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException as FastHTTPException

from api.deps import get_current_user


def _mock_user():
    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    return user


async def test_portfolio_empty(client):
    """보유종목 없으면 빈 리스트."""
    from main import app
    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    with patch("api.routes.portfolio.kis_account_service.get_account_balance",
               new_callable=AsyncMock,
               side_effect=FastHTTPException(status_code=503, detail="KIS 키 미설정")):
        with patch("api.routes.portfolio._get_holdings", new_callable=AsyncMock, return_value=[]):
            resp = await client.get("/portfolio")

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json()["holdings"] == []


@pytest.mark.asyncio
async def test_portfolio_paper_mode_uses_kis_account_balance():
    """paper 모드도 KIS 계좌 잔고를 현재 보유 현황의 원본으로 사용한다."""
    from api.routes.portfolio import _get_portfolio_response
    user = _mock_user()

    mock_account_data = {
        "mode": "paper",
        "account_no": "1234****-01",
        "summary": {
            "total_asset": 1_750_000,
            "deposit": 1_000_000,
            "eval_amount": 750_000,
            "buy_amount": 700_000,
            "eval_profit_loss": 50_000,
            "return_pct": 7.14,
        },
        "holdings": [{
            "stock_code": "005930",
            "stock_name": "삼성전자",
            "quantity": 10,
            "avg_price": 70_000,
            "current_price": 75_000,
            "eval_amount": 750_000,
            "profit_loss": 50_000,
            "return_pct": 7.14,
        }],
        "data_source": "KIS 모의투자 계좌",
    }

    with patch("api.routes.portfolio.settings.SYSTEM_KIS_MODE", "paper"), \
         patch("api.routes.portfolio.kis_account_service.get_account_balance",
               new_callable=AsyncMock, return_value=mock_account_data) as get_balance:
        data = await _get_portfolio_response(user=user, db=AsyncMock())

    get_balance.assert_awaited_once_with("paper")
    assert data["holdings"][0]["stock_code"] == "005930"
    assert data["total_eval"] == 750000
    assert data["total_cost"] == 700000
    assert data["total_return_pct"] == 7.14
    assert data["holding_source"] == "KIS 모의투자 계좌"
    assert data["performance_source"] == "앱 거래 기록 기준"


async def test_portfolio_with_holdings(client):
    """보유종목 있으면 수익률 계산 포함."""
    from main import app
    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    mock_holding = MagicMock()
    mock_holding.stock_code = "005930"
    mock_holding.stock_name = "삼성전자"
    mock_holding.quantity = 10
    mock_holding.avg_price = 70000
    mock_holding.mode = "paper"

    with patch("api.routes.portfolio.kis_account_service.get_account_balance",
               new_callable=AsyncMock,
               side_effect=FastHTTPException(status_code=503, detail="KIS 키 미설정")):
        with patch("api.routes.portfolio._get_holdings",
                   new_callable=AsyncMock, return_value=[mock_holding]):
            with patch("api.routes.portfolio.get_stock_current_price",
                       new_callable=AsyncMock, return_value={"close": 75000}):
                resp = await client.get("/portfolio")

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["holdings"]) == 1
    assert data["holdings"][0]["return_pct"] > 0


async def test_portfolio_metrics(client):
    """metrics 엔드포인트는 mdd_pct/win_rate_pct/sharpe_ratio 포함."""
    from main import app
    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    resp = await client.get("/portfolio/metrics")
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert "mdd_pct" in data
    assert "win_rate_pct" in data
    assert "sharpe_ratio" in data


async def test_portfolio_export_csv(client):
    """CSV export는 text/csv Content-Type 반환."""
    from main import app
    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    with patch("api.routes.portfolio._get_holdings", new_callable=AsyncMock, return_value=[]):
        resp = await client.get("/portfolio/export")

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_portfolio_real_mode_uses_kis():
    """real 모드도 공통 KIS 계좌 잔고 서비스를 사용한다."""
    from api.routes.portfolio import _get_portfolio_response
    user = _mock_user()
    user.mode = "real"

    mock_kis_data = {
        "mode": "real",
        "account_no": "1234****-01",
        "summary": {
            "total_asset": 1_750_000,
            "deposit": 1_000_000,
            "eval_amount": 750_000,
            "buy_amount": 700_000,
            "eval_profit_loss": 50_000,
            "return_pct": 7.14,
        },
        "holdings": [{
            "stock_code": "005930",
            "stock_name": "삼성전자",
            "quantity": 10,
            "avg_price": 70000.0,
            "current_price": 75000,
            "eval_amount": 750000,
            "profit_loss": 50000,
            "return_pct": 7.14,
        }],
        "data_source": "KIS 실계좌 계좌",
    }

    with patch("api.routes.portfolio.settings.SYSTEM_KIS_MODE", "real"), \
         patch("api.routes.portfolio.kis_account_service.get_account_balance",
               new_callable=AsyncMock, return_value=mock_kis_data):
        data = await _get_portfolio_response(user=user, db=AsyncMock())

    assert data["holdings"][0]["stock_code"] == "005930"
    assert data["total_eval"] == 750000
    assert data["total_return_pct"] == 7.14


@pytest.mark.asyncio
async def test_portfolio_real_mode_fallback_to_db():
    """real 모드에서 KIS 키 미설정 시 DB fallback."""
    from api.routes.portfolio import _get_portfolio_response
    user = _mock_user()
    user.mode = "real"

    with patch("api.routes.portfolio.kis_account_service.get_account_balance",
               new_callable=AsyncMock,
               side_effect=FastHTTPException(status_code=503, detail="KIS 키 미설정")):
        with patch("api.routes.portfolio._get_holdings",
                   new_callable=AsyncMock, return_value=[]):
            data = await _get_portfolio_response(user=user, db=AsyncMock())

    assert data["holdings"] == []
    assert data["holding_source"] == "앱 DB 포트폴리오 fallback"


async def test_portfolio_performance(client):
    """performance 엔드포인트 200 응답, 리스트 반환."""
    from main import app
    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    resp = await client.get("/portfolio/performance")
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
