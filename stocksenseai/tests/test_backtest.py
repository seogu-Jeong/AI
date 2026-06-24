# tests/test_backtest.py
import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.deps import get_current_user
from services.backtest_service import BacktestConfig, _compute_metrics, _simulate


def _mock_user():
    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    return user


def _mock_result():
    r = MagicMock()
    r.id = uuid.uuid4()
    r.stock_code = "005930"
    r.period_start = date(2024, 1, 1)
    r.period_end = date(2025, 1, 1)
    r.total_return_pct = 12.34
    r.mdd_pct = 5.21
    r.sharpe_ratio = 1.45
    r.win_rate_pct = 60.0
    r.total_trades = 6
    r.strategy_config = {"entry_signal_score": 65.0}
    r.result_detail = {"trades": [], "equity_curve": []}
    r.created_at = None
    return r


async def test_run_backtest_returns_result(client):
    """정상 요청은 200 + 필수 필드 반환."""
    from main import app

    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    with patch(
        "api.routes.backtest.backtest_service.run_backtest",
        new_callable=AsyncMock,
        return_value=_mock_result(),
    ):
        resp = await client.post(
            "/backtest/run",
            json={
                "code": "005930",
                "start_date": "2024-01-01",
                "end_date": "2025-01-01",
            },
        )

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    data = resp.json()
    assert "total_return_pct" in data
    assert "mdd_pct" in data
    assert "win_rate_pct" in data
    assert "sharpe_ratio" in data


async def test_run_backtest_invalid_dates(client):
    """end_date <= start_date → 422."""
    from main import app

    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    resp = await client.post(
        "/backtest/run",
        json={
            "code": "005930",
            "start_date": "2025-01-01",
            "end_date": "2024-01-01",
        },
    )

    app.dependency_overrides.clear()
    assert resp.status_code == 422


async def test_get_backtest_not_found(client):
    """존재하지 않는 id → 404."""
    from main import app

    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user

    resp = await client.get(f"/backtest/{uuid.uuid4()}")
    app.dependency_overrides.clear()
    assert resp.status_code == 404


async def test_get_backtest_success(client, db_session):
    """DB에 직접 삽입한 결과를 조회."""
    from main import app
    from models.backtest import BacktestResult
    from models.user import User
    from sqlalchemy import select

    result = await db_session.execute(select(User).limit(1))
    user_row = result.scalar_one_or_none()
    if user_row is None:
        pytest.skip("No user in test DB")

    app.dependency_overrides[get_current_user] = lambda: user_row

    backtest = BacktestResult(
        user_id=user_row.id,
        stock_code="005930",
        strategy_config={"entry_signal_score": 65.0},
        period_start=date(2024, 1, 1),
        period_end=date(2025, 1, 1),
        total_return_pct=10.0,
        mdd_pct=5.0,
        sharpe_ratio=1.2,
        win_rate_pct=55.0,
        total_trades=4,
        result_detail={"trades": [], "equity_curve": []},
    )
    db_session.add(backtest)
    await db_session.commit()
    await db_session.refresh(backtest)

    resp = await client.get(f"/backtest/{backtest.id}")
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["stock_code"] == "005930"


# ──── 엔진 단위 테스트 ────────────────────────────────────────────


def _cfg(**kwargs):
    defaults = dict(
        code="005930",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        initial_cash=10_000_000,
        entry_signal_score=65.0,
        exit_signal_score=35.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        commission_rate=0.00015,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def test_simulate_no_trades():
    """시그널이 없으면 거래 없음 — equity는 초기 자금 유지."""
    # 모든 score=50 → 진입 기준(65) 미달
    daily = [("2024-01-02", 70000.0, 50.0)] * 10
    trades, equity, open_pos = _simulate(daily, _cfg())
    assert trades == []
    assert open_pos is None
    assert all(e == 10_000_000 for e in equity)


def test_simulate_buy_and_signal_exit():
    """매수 후 시그널 청산: PnL에 매수 수수료 반영."""
    daily = [
        ("2024-01-02", 10000.0, 70.0),   # 매수 발생
        ("2024-01-03", 11000.0, 30.0),   # 시그널 청산
    ]
    trades, equity, open_pos = _simulate(daily, _cfg())
    assert len(trades) == 1
    t = trades[0]
    assert t["reason"] == "signal"
    assert t["exit_price"] == 11000
    # PnL = revenue - entry_cost (수수료 포함)
    shares = t["shares"]
    entry_cost = shares * 10000 * (1 + 0.00015)
    revenue = shares * 11000 * (1 - 0.00015)
    assert abs(t["pnl"] - round(revenue - entry_cost)) <= 1


def test_simulate_stop_loss():
    """손절 조건 발동."""
    daily = [
        ("2024-01-02", 10000.0, 70.0),   # 매수
        ("2024-01-03", 9400.0, 50.0),    # 6% 하락 → stop_loss=5% 발동
    ]
    trades, equity, open_pos = _simulate(daily, _cfg())
    assert len(trades) == 1
    assert trades[0]["reason"] == "stop_loss"


def test_simulate_open_position_at_end():
    """기간 종료 시 미청산 포지션 open_position 반환."""
    daily = [
        ("2024-01-02", 10000.0, 70.0),   # 매수
        ("2024-01-03", 10500.0, 50.0),   # 청산 조건 없음 → 미청산 유지
    ]
    trades, equity, open_pos = _simulate(daily, _cfg())
    assert trades == []
    assert open_pos is not None
    assert open_pos["shares"] > 0
    assert open_pos["last_price"] == 10500
    assert open_pos["entry_price"] == 10000


def test_compute_metrics_basic():
    """단순 상승 equity curve: MDD=0, 수익률 계산."""
    equity = [10_000_000, 10_500_000, 11_000_000]
    trades = [{"pnl": 500_000}, {"pnl": 500_000}]
    m = _compute_metrics(equity, trades, 10_000_000)
    assert m["total_return_pct"] == pytest.approx(10.0, abs=0.01)
    assert m["mdd_pct"] == 0.0
    assert m["win_rate_pct"] == 100.0


def test_compute_metrics_mdd():
    """MDD 계산: 고점 대비 낙폭."""
    equity = [10_000_000, 12_000_000, 9_000_000, 11_000_000]
    m = _compute_metrics(equity, [], 10_000_000)
    # 고점 12M → 9M, 낙폭 = 25%
    assert m["mdd_pct"] == pytest.approx(25.0, abs=0.01)


def test_compute_metrics_empty():
    """빈 데이터는 0 반환."""
    m = _compute_metrics([], [], 10_000_000)
    assert m["total_return_pct"] == 0.0
    assert m["mdd_pct"] == 0.0
