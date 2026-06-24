"""자동매매 알고리즘 단위 테스트."""
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from models.auto_trade import AutoTradeConfig, AutoTradeLog
from models.portfolio import Portfolio
from models.trade import Trade
from models.user import User
from services.auto_trade_service import _allocate, _calculate_buying_power, _execute_paper_order


class _FakeRedisLock:
    def __init__(self):
        self.value = None

    async def set(self, key, value, ex=None, nx=False):
        if nx and self.value is not None:
            return False
        self.value = value
        return True

    async def get(self, key):
        return self.value

    async def delete(self, key):
        self.value = None

    async def eval(self, script, numkeys, key, token):
        if self.value == token:
            self.value = None
            return 1
        return 0


def test_allocate_caps_single_stock_at_30_percent():
    """단일 종목 배분이 총예산의 30%를 초과하지 않는다."""
    candidates = [{"code": "005930", "score": 100}]
    result = _allocate(candidates, available=500_000, total_budget=1_000_000)

    assert len(result) == 1
    assert result[0]["alloc"] == 300_000  # capped at 30% of 1_000_000


def test_calculate_buying_power_reserves_cash():
    """_calculate_buying_power 는 10% 현금 보유 후 가용 매수 금액을 반환한다."""
    # 투자된 금액이 없을 때: 1,000,000 * 0.9 = 900,000
    assert _calculate_buying_power(1_000_000, 0) == 900_000

    # 투자된 금액 800,000: 900,000 - 800,000 = 100,000
    assert _calculate_buying_power(1_000_000, 800_000) == 100_000

    # 투자된 금액이 investable_limit 과 같을 때: 0
    assert _calculate_buying_power(1_000_000, 900_000) == 0

    # 투자된 금액이 investable_limit 초과: 음수가 되어선 안 됨 → 0
    assert _calculate_buying_power(1_000_000, 950_000) == 0


@pytest.mark.asyncio
async def test_execute_paper_sell_logs_actual_filled_quantity(db_session):
    """SELL 시 holding 수량보다 많이 요청해도 AutoTradeLog/반환값은 실제 체결 수량 기준이어야 한다."""
    # 1. 사용자 생성
    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    # 2. Portfolio: 005930 보유 수량 5주
    db_session.add(Portfolio(
        user_id=user_id,
        stock_code="005930",
        stock_name="삼성전자",
        quantity=5,
        avg_price=10_000,
        mode="paper",
    ))
    await db_session.flush()

    # 3. SELL 10주 요청 (보유는 5주)
    result = await _execute_paper_order(
        user_id=user_id,
        stock_code="005930",
        stock_name="삼성전자",
        order_type="SELL",
        quantity=10,
        price=11_000,
        reason="손절",
        mode="paper",
        signal_score=0.0,
        db=db_session,
    )

    # 4. Trade 검증
    trade_res = await db_session.execute(
        select(Trade).where(Trade.user_id == user_id, Trade.stock_code == "005930")
    )
    trade = trade_res.scalar_one()
    assert trade.quantity == 5, f"Trade.quantity expected 5, got {trade.quantity}"
    assert trade.filled_quantity == 5, f"Trade.filled_quantity expected 5, got {trade.filled_quantity}"
    assert trade.realized_pnl == 5_000, f"Trade.realized_pnl expected 5000, got {trade.realized_pnl}"

    # 5. AutoTradeLog 검증
    log_res = await db_session.execute(
        select(AutoTradeLog).where(AutoTradeLog.user_id == user_id, AutoTradeLog.stock_code == "005930")
    )
    log = log_res.scalar_one()
    assert log.quantity == 5, f"AutoTradeLog.quantity expected 5, got {log.quantity}"
    assert log.total_amount == 55_000, f"AutoTradeLog.total_amount expected 55000, got {log.total_amount}"

    # 6. 반환값 검증
    assert result["quantity"] == 5, f"return quantity expected 5, got {result['quantity']}"
    assert result["total_amount"] == 55_000, f"return total_amount expected 55000, got {result['total_amount']}"


@pytest.mark.asyncio
async def test_execute_paper_buy_updates_average_price_and_log_amount(db_session):
    """BUY 시 평균단가 갱신이 올바르고 AutoTradeLog/반환값도 executed_qty 기준이어야 한다."""
    # 1. 사용자 생성
    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    # 2. Portfolio: 000660 보유 수량 2주, 평균 50,000원
    db_session.add(Portfolio(
        user_id=user_id,
        stock_code="000660",
        stock_name="SK하이닉스",
        quantity=2,
        avg_price=50_000,
        mode="paper",
    ))
    await db_session.flush()

    # 3. BUY 3주 @ 60,000원
    result = await _execute_paper_order(
        user_id=user_id,
        stock_code="000660",
        stock_name="SK하이닉스",
        order_type="BUY",
        quantity=3,
        price=60_000,
        reason="AI BUY",
        mode="paper",
        signal_score=80.0,
        db=db_session,
    )

    # 4. Portfolio 검증: 수량 5, 평균단가 56,000
    port_res = await db_session.execute(
        select(Portfolio).where(Portfolio.user_id == user_id, Portfolio.stock_code == "000660")
    )
    portfolio = port_res.scalar_one()
    assert portfolio.quantity == 5, f"Portfolio.quantity expected 5, got {portfolio.quantity}"
    assert float(portfolio.avg_price) == 56_000.0, f"Portfolio.avg_price expected 56000, got {portfolio.avg_price}"

    # 5. AutoTradeLog 검증
    log_res = await db_session.execute(
        select(AutoTradeLog).where(AutoTradeLog.user_id == user_id, AutoTradeLog.stock_code == "000660")
    )
    log = log_res.scalar_one()
    assert log.quantity == 3, f"AutoTradeLog.quantity expected 3, got {log.quantity}"
    assert log.total_amount == 180_000, f"AutoTradeLog.total_amount expected 180000, got {log.total_amount}"

    # 6. 반환값 검증
    assert result["quantity"] == 3, f"return quantity expected 3, got {result['quantity']}"
    assert result["total_amount"] == 180_000, f"return total_amount expected 180000, got {result['total_amount']}"


@pytest.mark.asyncio
async def test_execute_paper_sell_raises_when_holding_quantity_is_zero(db_session):
    """SELL 시 보유 수량이 0이면 ValueError를 발생시킨다."""
    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    # Portfolio quantity=0 (dirty data scenario)
    db_session.add(Portfolio(
        user_id=user_id,
        stock_code="005380",
        stock_name="현대차",
        quantity=0,
        avg_price=100_000,
        mode="paper",
    ))
    await db_session.flush()

    with pytest.raises(ValueError, match="데이터 오염"):
        await _execute_paper_order(
            user_id=user_id,
            stock_code="005380",
            stock_name="현대차",
            order_type="SELL",
            quantity=1,
            price=105_000,
            reason="손절",
            mode="paper",
            signal_score=0.0,
            db=db_session,
        )


@pytest.mark.asyncio
async def test_failed_paper_sell_does_not_leave_pending_trade_for_later_commit(db_session):
    """실패한 SELL 주문은 같은 세션의 후속 commit으로 Trade가 저장되면 안 된다."""
    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    db_session.add(Portfolio(
        user_id=user_id,
        stock_code="005380",
        stock_name="현대차",
        quantity=0,
        avg_price=100_000,
        mode="paper",
    ))
    await db_session.flush()

    with pytest.raises(ValueError, match="데이터 오염"):
        await _execute_paper_order(
            user_id=user_id,
            stock_code="005380",
            stock_name="현대차",
            order_type="SELL",
            quantity=1,
            price=105_000,
            reason="손절",
            mode="paper",
            signal_score=0.0,
            db=db_session,
        )

    await db_session.commit()

    trade_res = await db_session.execute(
        select(Trade).where(Trade.user_id == user_id, Trade.stock_code == "005380")
    )
    assert trade_res.scalars().all() == []


@pytest.mark.asyncio
async def test_release_run_lock_does_not_delete_lock_owned_by_newer_run():
    """TTL 만료 후 새 실행이 잡은 lock을 이전 실행의 finally가 삭제하면 안 된다."""
    from services.auto_trade_service import _acquire_run_lock, _release_run_lock

    redis = _FakeRedisLock()
    user_id = uuid.uuid4()

    with patch("core.redis_client.get_redis", new=AsyncMock(return_value=redis)):
        first_token = await _acquire_run_lock(user_id)
        assert first_token

        # TTL 만료 후 다른 실행이 같은 user lock을 다시 획득한 상황을 재현한다.
        redis.value = None
        second_token = await _acquire_run_lock(user_id)
        assert second_token
        assert second_token != first_token

        await _release_run_lock(user_id, first_token)

    assert redis.value == second_token


@pytest.mark.asyncio
async def test_run_cycle_does_not_buy_when_cash_reserve_would_be_broken(db_session):
    """현금 보유 한도 도달 시 BUY 후보가 있어도 매수하지 않는다."""
    from services.auto_trade_service import run_cycle

    # 1. 사용자 생성
    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    # 2. AutoTradeConfig: total_budget=1,000,000, enabled=True, mode="paper", signal_threshold=0
    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    await db_session.flush()

    # 3. Portfolio: 이미 900,000원 투자됨 (1주 × 평균단가 900,000)
    #    _calculate_buying_power(1_000_000, 900_000) = 900_000 - 900_000 = 0 → 매수 불가
    db_session.add(Portfolio(
        user_id=user_id,
        stock_code="000660",
        stock_name="SK하이닉스",
        quantity=1,
        avg_price=900_000,
        mode="paper",
    ))
    await db_session.flush()

    # 4. AI 스크리닝은 BUY 신호(score=90)를 반환하도록 mock
    buy_candidates = [{"code": "005930", "score": 90.0}]

    # 현재가를 평균단가(900_000)와 같게 설정해 손절/익절 조건 미충족
    # → SELL 단계는 실행되지 않고, BUY 단계만 가용 예산이 0이라 건너뜀
    with patch(
        "services.auto_trade_service._get_buy_candidates",
        new=AsyncMock(return_value=buy_candidates),
    ), patch(
        "services.market_service.get_stock_current_price",
        new=AsyncMock(return_value={"close": 900_000, "name": "SK하이닉스"}),
    ), patch(
        "services.ai_service.get_signal",
        new=AsyncMock(return_value={"signal": "HOLD", "signal_score": 50}),
    ):
        result = await run_cycle(user_id=user_id, db=db_session)

    # 5. 매수 실행 없어야 함
    assert result["executed"] == 0, (
        f"expected 0 executions but got {result['executed']}; "
        f"no_trade_reason={result.get('no_trade_reason')}"
    )
    # 6. no_trade_reason 에 현금 보유 한도 관련 문구 포함
    reason = result.get("no_trade_reason") or ""
    assert "가용 예산" in reason or "현금 보유" in reason, (
        f"expected cash-reserve reason but got: {reason!r}"
    )


@pytest.mark.asyncio
async def test_run_cycle_respects_signal_threshold(db_session):
    """signal_threshold 미달 후보는 매수 대상에서 제외된다."""
    from services.auto_trade_service import run_cycle

    # 1. 사용자 생성
    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    # 2. AutoTradeConfig: signal_threshold=80 (기본값 70보다 높음), 예산 충분
    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        signal_threshold=80,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    await db_session.flush()

    # 3. BUY 후보 score=60 (threshold 80 미달)
    buy_candidates = [{"code": "005930", "score": 60.0}]

    with patch(
        "services.auto_trade_service._get_buy_candidates",
        new=AsyncMock(return_value=buy_candidates),
    ), patch(
        "services.market_service.get_stock_current_price",
        new=AsyncMock(return_value={"close": 70_000, "name": "삼성전자"}),
    ), patch(
        "services.ai_service.get_signal",
        new=AsyncMock(return_value={"signal": "HOLD", "signal_score": 50}),
    ):
        result = await run_cycle(user_id=user_id, db=db_session)

    assert result["executed"] == 0, (
        f"expected 0 executions but got {result['executed']}; "
        f"no_trade_reason={result.get('no_trade_reason')}"
    )
    reason = result.get("no_trade_reason") or ""
    assert "미달" in reason, (
        f"expected '미달' in no_trade_reason but got: {reason!r}"
    )


@pytest.mark.asyncio
async def test_run_cycle_respects_max_positions(db_session):
    """max_positions 한도 도달 시 BUY 후보가 있어도 신규 매수하지 않는다."""
    from services.auto_trade_service import run_cycle

    # 1. 사용자 생성
    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    # 2. AutoTradeConfig: max_positions=1, 예산 충분, signal_threshold=0 (모든 후보 통과)
    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        max_positions=1,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    await db_session.flush()

    # 3. 이미 1 포지션 보유 중 → remaining_slots = 1 - 1 = 0
    db_session.add(Portfolio(
        user_id=user_id,
        stock_code="000660",
        stock_name="SK하이닉스",
        quantity=1,
        avg_price=10_000,
        mode="paper",
    ))
    await db_session.flush()

    # 4. 신선한 BUY 후보 (000660이 아닌 종목)
    buy_candidates = [{"code": "005930", "score": 90.0}]

    with patch(
        "services.auto_trade_service._get_buy_candidates",
        new=AsyncMock(return_value=buy_candidates),
    ), patch(
        "services.market_service.get_stock_current_price",
        new=AsyncMock(return_value={"close": 10_000, "name": "SK하이닉스"}),
    ), patch(
        "services.ai_service.get_signal",
        new=AsyncMock(return_value={"signal": "HOLD", "signal_score": 50}),
    ):
        result = await run_cycle(user_id=user_id, db=db_session)

    assert result["executed"] == 0, (
        f"expected 0 executions but got {result['executed']}; "
        f"no_trade_reason={result.get('no_trade_reason')}"
    )
    reason = result.get("no_trade_reason") or ""
    assert "한도" in reason, (
        f"expected '한도' in no_trade_reason but got: {reason!r}"
    )


@pytest.mark.asyncio
async def test_run_cycle_reports_price_fetch_failures(db_session):
    """가격 조회 실패 건수가 diagnostics에 기록된다."""
    from services.auto_trade_service import run_cycle

    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    await db_session.flush()

    with (
        patch("services.auto_trade_service._acquire_run_lock", new=AsyncMock(return_value="lock-token")),
        patch("services.auto_trade_service._release_run_lock", new=AsyncMock()),
        patch("services.auto_trade_service._get_buy_candidates",
              new=AsyncMock(return_value=[
                  {"code": "005930", "score": 90.0},
                  {"code": "000660", "score": 85.0},
              ])),
        patch("services.market_service.get_stock_current_price",
              new=AsyncMock(side_effect=RuntimeError("시세 조회 실패"))),
    ):
        result = await run_cycle(user_id, db_session)

    diag = result.get("diagnostics", {})
    assert diag.get("price_fetch_failed", 0) == 2, (
        f"expected price_fetch_failed=2 but got {diag}"
    )


@pytest.mark.asyncio
async def test_run_cycle_reports_sell_side_price_fetch_failures(db_session):
    """보유 종목 SELL 검사 중 가격 조회 실패도 diagnostics에 기록된다."""
    from services.auto_trade_service import run_cycle

    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    db_session.add(Portfolio(
        user_id=user_id,
        stock_code="005930",
        stock_name="삼성전자",
        quantity=1,
        avg_price=70_000,
        mode="paper",
    ))
    await db_session.flush()

    with (
        patch("services.auto_trade_service._acquire_run_lock", new=AsyncMock(return_value="lock-token")),
        patch("services.auto_trade_service._release_run_lock", new=AsyncMock()),
        patch("services.auto_trade_service._get_buy_candidates", new=AsyncMock(return_value=[])),
        patch("services.market_service.get_stock_current_price",
              new=AsyncMock(side_effect=RuntimeError("시세 조회 실패"))),
    ):
        result = await run_cycle(user_id, db_session)

    diag = result.get("diagnostics", {})
    assert diag.get("price_fetch_failed", 0) == 1, (
        f"expected sell-side price_fetch_failed=1 but got {diag}"
    )


@pytest.mark.asyncio
async def test_run_cycle_allows_sell_even_when_buy_is_blocked(db_session):
    """risk hard stop이 BUY를 막아도 익절 SELL은 실행된다."""
    from fastapi import HTTPException

    from services.auto_trade_service import run_cycle

    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    # avg=60_000, price=70_000 → +16.7% > take_profit_pct(5%) → 익절 SELL 조건
    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=5.0,
    ))
    await db_session.flush()

    db_session.add(Portfolio(
        user_id=user_id,
        stock_code="005930",
        stock_name="삼성전자",
        quantity=5,
        avg_price=60_000,
        mode="paper",
    ))
    await db_session.flush()

    risk_check_mock = AsyncMock(side_effect=HTTPException(status_code=400, detail="거래 차단"))
    with (
        patch("services.auto_trade_service._acquire_run_lock", new=AsyncMock(return_value="lock-token")),
        patch("services.auto_trade_service._release_run_lock", new=AsyncMock()),
        patch("services.auto_trade_service._get_buy_candidates",
              new=AsyncMock(return_value=[{"code": "000660", "score": 90.0}])),
        patch("services.market_service.get_stock_current_price",
              new=AsyncMock(return_value={"close": 70_000, "name": "삼성전자"})),
        patch("services.auto_trade_service.risk_service.check_order", new=risk_check_mock),
    ):
        result = await run_cycle(user_id, db_session)

    sell_actions = [a for a in result.get("actions", []) if a.get("action") == "SELL"]
    assert len(sell_actions) >= 1, f"expected SELL action but got actions: {result.get('actions')}"
    assert result["executed"] >= 1, f"expected executed>=1 but got {result['executed']}"
    # check_order called once (for BUY candidate only, never for SELL)
    assert risk_check_mock.call_count == 1, (
        f"expected check_order called once (for BUY) but got {risk_check_mock.call_count}"
    )


@pytest.mark.asyncio
async def test_run_cycle_skips_when_user_lock_is_held(db_session):
    """lock을 이미 획득한 상태에서 run_cycle을 호출하면 already_running을 반환한다."""
    from services.auto_trade_service import run_cycle

    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    await db_session.flush()

    with patch("services.auto_trade_service._acquire_run_lock", new=AsyncMock(return_value=False)):
        result = await run_cycle(user_id, db_session)

    assert result["skipped"] is True
    assert result["reason"] == "already_running"


@pytest.mark.asyncio
async def test_run_cycle_releases_lock_when_price_fetch_raises(db_session):
    """내부 예외가 발생해도 lock은 반드시 해제된다."""
    from services.auto_trade_service import run_cycle

    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    await db_session.flush()

    release_mock = AsyncMock()

    with (
        patch("services.auto_trade_service._acquire_run_lock", new=AsyncMock(return_value="lock-token")),
        patch("services.auto_trade_service._release_run_lock", new=release_mock),
        patch("services.auto_trade_service._get_buy_candidates", new=AsyncMock(side_effect=RuntimeError("fail"))),
    ):
        with pytest.raises(RuntimeError):
            await run_cycle(user_id, db_session)

    # Lock must have been released even when an exception propagates
    release_mock.assert_called_once_with(user_id, "lock-token")


@pytest.mark.asyncio
async def test_run_cycle_does_not_buy_when_risk_hard_stop_blocks(db_session):
    """risk hard stop이면 BUY가 실행되지 않는다."""
    from fastapi import HTTPException

    from services.auto_trade_service import run_cycle

    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    await db_session.flush()

    with (
        patch("services.auto_trade_service._acquire_run_lock", new=AsyncMock(return_value="lock-token")),
        patch("services.auto_trade_service._release_run_lock", new=AsyncMock()),
        patch("services.auto_trade_service._get_buy_candidates",
              new=AsyncMock(return_value=[{"code": "005930", "score": 90.0}])),
        patch("services.market_service.get_stock_current_price",
              new=AsyncMock(return_value={"close": 50_000, "name": "삼성전자"})),
        patch("services.auto_trade_service.risk_service.check_order",
              new=AsyncMock(side_effect=HTTPException(status_code=400, detail="거래 차단"))),
        patch("services.ai_service.get_signal",
              new=AsyncMock(return_value={"signal": "HOLD", "signal_score": 50})),
    ):
        result = await run_cycle(user_id, db_session)

    assert result["executed"] == 0
    assert any(a.get("action") == "SKIP" for a in result.get("actions", []))
    reason = result.get("no_trade_reason") or ""
    assert "리스크" in reason or "차단" in reason, (
        f"expected risk-blocked message in no_trade_reason but got: {reason!r}"
    )


@pytest.mark.asyncio
async def test_run_cycle_records_warning_when_risk_service_warns(db_session):
    """risk_service가 경고를 반환하면 주문은 실행되지만 log에 warning이 기록된다."""
    from services.auto_trade_service import run_cycle

    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=1_000_000,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    await db_session.flush()

    with (
        patch("services.auto_trade_service._acquire_run_lock", new=AsyncMock(return_value="lock-token")),
        patch("services.auto_trade_service._release_run_lock", new=AsyncMock()),
        patch("services.auto_trade_service._get_buy_candidates",
              new=AsyncMock(return_value=[{"code": "005930", "score": 90.0}])),
        patch("services.market_service.get_stock_current_price",
              new=AsyncMock(return_value={"close": 50_000, "name": "삼성전자"})),
        patch("services.auto_trade_service.risk_service.check_order",
              new=AsyncMock(return_value="종목별 한도 초과: 25.0% > 20.0%")),
        patch("services.ai_service.get_signal",
              new=AsyncMock(return_value={"signal": "HOLD", "signal_score": 50})),
    ):
        result = await run_cycle(user_id, db_session)

    assert result["executed"] == 1
    assert result["actions"][0].get("warning") is not None


@pytest.mark.asyncio
async def test_run_cycle_does_not_buy_when_max_positions_already_reached(db_session):
    """max_positions=2, 2개 보유 → remaining_slots=0 → 신규 매수 없음."""
    from services.auto_trade_service import run_cycle

    # 1. 사용자 생성
    user_id = uuid.uuid4()
    db_session.add(User(
        id=user_id,
        email=f"algo-test-{uuid.uuid4().hex[:6]}@test.com",
        password_hash="x",
        is_verified=True,
    ))
    await db_session.flush()

    # 2. AutoTradeConfig: max_positions=2, signal_threshold=0
    db_session.add(AutoTradeConfig(
        user_id=user_id,
        enabled=True,
        mode="paper",
        total_budget=2_000_000,
        max_positions=2,
        signal_threshold=0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ))
    await db_session.flush()

    # 3. 2개 종목 이미 보유
    for code, name in [("000660", "SK하이닉스"), ("035720", "카카오")]:
        db_session.add(Portfolio(
            user_id=user_id,
            stock_code=code,
            stock_name=name,
            quantity=1,
            avg_price=10_000,
            mode="paper",
        ))
    await db_session.flush()

    # 4. 신선한 BUY 후보
    buy_candidates = [{"code": "005930", "score": 90.0}]

    with patch(
        "services.auto_trade_service._get_buy_candidates",
        new=AsyncMock(return_value=buy_candidates),
    ), patch(
        "services.market_service.get_stock_current_price",
        new=AsyncMock(return_value={"close": 10_000, "name": "삼성전자"}),
    ), patch(
        "services.ai_service.get_signal",
        new=AsyncMock(return_value={"signal": "HOLD", "signal_score": 50}),
    ):
        result = await run_cycle(user_id=user_id, db=db_session)

    assert result["executed"] == 0, (
        f"expected 0 executions but got {result['executed']}; "
        f"no_trade_reason={result.get('no_trade_reason')}"
    )
    reason = result.get("no_trade_reason") or ""
    assert "한도" in reason, (
        f"expected '한도' in no_trade_reason but got: {reason!r}"
    )
