# tests/test_simulate.py
import pytest
from datetime import date


# ──── 단위 테스트 (DB/pykrx 없이 실행 가능) ────────────────────────

def _make_prices():
    """2024-01-02 ~ 2024-01-10 삼성전자 가격 픽스처 (주말 제외)."""
    return {
        "2024-01-02": 70000.0,
        "2024-01-03": 71000.0,
        "2024-01-04": 72000.0,
        "2024-01-05": 73000.0,
        "2024-01-08": 74000.0,
        "2024-01-09": 73500.0,
        "2024-01-10": 75000.0,
    }


def test_calc_lumpsum_logic():
    """100만원으로 70000원 주식 14주 매수 → 매도가 기반 수익 계산."""
    from services.simulator_service import calc_lumpsum

    prices = _make_prices()
    result = calc_lumpsum(
        ticker="005930",
        buy_date=date(2024, 1, 2),
        sell_date=date(2024, 1, 10),
        amount_krw=1_000_000,
        prices=prices,
        name="삼성전자",
    )

    assert result["shares"] == 14         # int(1_000_000 / 70000)
    assert result["buy_price"] == 70000
    assert result["sell_price"] == 75000
    assert result["buy_value_krw"] == 980_000   # 14 * 70000
    assert result["sell_value_krw"] == 1_050_000  # 14 * 75000
    assert result["profit_krw"] == 70_000
    assert result["return_pct"] == pytest.approx(7.1429, abs=0.01)
    assert result["buy_date_actual"] == "2024-01-02"
    assert result["sell_date_actual"] == "2024-01-10"
    assert len(result["chart_data"]) == len(prices)
    assert result["chart_data"][0] == {"date": "2024-01-02", "return_pct": 0.0}


def test_calc_lumpsum_weekend_adjustment():
    """토요일 매수일 → 다음 월요일로 조정."""
    from services.simulator_service import calc_lumpsum

    prices = _make_prices()
    result = calc_lumpsum(
        ticker="005930",
        buy_date=date(2024, 1, 6),   # 토요일
        sell_date=date(2024, 1, 10),
        amount_krw=1_000_000,
        prices=prices,
        name="삼성전자",
    )
    assert result["buy_date_actual"] == "2024-01-08"   # 다음 월요일


def test_calc_recurring_logic():
    """3개월 적립: 매월 첫 영업일에 매수."""
    from services.simulator_service import calc_recurring

    prices = {
        "2024-01-02": 70000.0,
        "2024-02-01": 72000.0,
        "2024-03-04": 68000.0,
        "2024-03-31": 69000.0,  # 마지막 날
    }
    result = calc_recurring(
        ticker="005930",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        monthly_amount_krw=300_000,
        prices=prices,
        name="삼성전자",
    )

    # 1월: int(300000/70000)=4주, 2월: int(300000/72000)=4주, 3월: int(300000/68000)=4주
    assert result["total_purchases"] == 3
    assert result["total_shares"] == 12    # 4+4+4
    jan_invested = 4 * 70000               # 280000
    feb_invested = 4 * 72000               # 288000
    mar_invested = 4 * 68000               # 272000
    assert result["total_invested_krw"] == jan_invested + feb_invested + mar_invested
    # 최종가 69000
    assert result["current_value_krw"] == 12 * 69000
    # 3개 매수일 + 최종 평가 시점(2024-03-31) = 4
    assert len(result["chart_data"]) == 4
    assert result["chart_data"][-1]["date"] == "2024-03-31"
    assert result["chart_data"][-1]["value"] == 12 * 69000
    assert result["start_date_actual"] == "2024-01-02"
    assert result["end_date_actual"] == "2024-03-31"

# ──── 통합 테스트 ────────────────────────────────────────────

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from api.deps import get_current_user


def _mock_user():
    u = MagicMock()
    u.id = uuid.uuid4()
    u.mode = "paper"
    return u


def _mock_lumpsum_result():
    return {
        "ticker": "005930", "name": "삼성전자",
        "shares": 13, "buy_price": 76000, "sell_price": 73400,
        "buy_value_krw": 988000, "sell_value_krw": 954200,
        "profit_krw": -33800, "return_pct": -3.42,
        "buy_date_actual": "2022-01-03", "sell_date_actual": "2026-05-30",
        "chart_data": [],
    }


def _mock_recurring_result():
    return {
        "ticker": "005930", "name": "삼성전자",
        "total_invested_krw": 19200000, "total_shares": 252,
        "avg_buy_price": 76190, "current_value_krw": 18496800,
        "return_pct": -3.56, "total_purchases": 64,
        "chart_data": [],
    }


async def test_lumpsum_returns_result(client):
    """정상 lumpsum 요청 → 200 + 필수 필드."""
    from main import app

    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    with patch(
        "api.routes.simulate.simulator_service.run_lumpsum",
        new_callable=AsyncMock,
        return_value=_mock_lumpsum_result(),
    ):
        resp = await client.post(
            "/simulate/lumpsum",
            json={
                "tickers": ["005930"],
                "buy_date": "2022-01-03",
                "sell_date": "2026-05-31",
                "amount_krw": 1000000,
            },
        )

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "buy_date_actual" in data
    assert data["results"][0]["ticker"] == "005930"


async def test_lumpsum_invalid_dates(client):
    """sell_date <= buy_date → 422."""
    from main import app

    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    resp = await client.post(
        "/simulate/lumpsum",
        json={
            "tickers": ["005930"],
            "buy_date": "2026-01-01",
            "sell_date": "2022-01-01",
            "amount_krw": 1000000,
        },
    )
    app.dependency_overrides.clear()
    assert resp.status_code == 422


async def test_recurring_returns_result(client):
    """정상 recurring 요청 → 200 + chart_data."""
    from main import app

    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    with patch(
        "api.routes.simulate.simulator_service.run_recurring",
        new_callable=AsyncMock,
        return_value=_mock_recurring_result(),
    ):
        resp = await client.post(
            "/simulate/recurring",
            json={
                "tickers": ["005930"],
                "start_date": "2020-01-02",
                "end_date": "2026-05-31",
                "monthly_amount_krw": 300000,
            },
        )

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["results"][0]["total_purchases"] == 64


async def test_data_status_not_ready(client):
    """빈 price_cache → ready: false."""
    from main import app

    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    with patch(
        "api.routes.simulate.simulator_service.get_data_status",
        new_callable=AsyncMock,
        return_value={"ready": False, "ticker_count": 0, "last_updated": None},
    ):
        resp = await client.get("/simulate/data-status")

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json()["ready"] is False


async def test_download_sse_content_type(client):
    """SSE 엔드포인트는 text/event-stream 반환."""
    from main import app

    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    async def _mock_gen(_db):
        yield {"current": 1, "total": 80, "ticker": "005930", "name": "삼성전자"}

    with patch("api.routes.simulate.simulator_service.download_tickers", side_effect=_mock_gen):
        resp = await client.get("/simulate/download")

    app.dependency_overrides.clear()
    assert "text/event-stream" in resp.headers.get("content-type", "")
