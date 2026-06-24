import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.deps import get_current_user, get_db
from main import app
from models.trade import Trade
from models.user import User


def _mock_user(mode="paper"):
    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = mode
    user.kis_paper_key_enc = "enc_key"
    return user


async def _seed_user(db_session, user_id):
    """users 테이블에 FK용 더미 User 행 삽입."""
    u = User(id=user_id, email=f"test-{user_id}@test.com", mode="paper")
    db_session.add(u)
    await db_session.flush()
    return u


async def _seed_trades(db_session, user_id, trades_data):
    """Trade 행을 DB에 직접 삽입하는 헬퍼."""
    trades = [
        Trade(
            user_id=user_id,
            stock_code=d.get("stock_code", "005930"),
            order_type=d.get("order_type", "BUY"),
            price_type=d.get("price_type", "LIMIT"),
            quantity=d.get("quantity", 1),
            order_price=d.get("order_price", 70000),
            status=d.get("status", "PENDING"),
            mode=d.get("mode", "paper"),
        )
        for d in trades_data
    ]
    for t in trades:
        db_session.add(t)
    await db_session.flush()
    return trades


def _mock_db():
    """DB 세션 mock — 실제 DB 쓰기 없이 Trade 저장 시뮬레이션."""
    session = AsyncMock()
    trade_mock = MagicMock()
    trade_mock.id = uuid.uuid4()
    trade_mock.status = "PENDING"
    trade_mock.kis_order_no = "0000123456"

    async def fake_refresh(obj):
        # Trade 객체에 id 주입
        obj.id = trade_mock.id

    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = fake_refresh
    session.execute = AsyncMock()
    return session


async def test_order_uses_system_kis_mode(client):
    """주문 기록과 체결 폴링은 실제 시스템 KIS 모드를 사용한다."""
    user = _mock_user(mode="demo")
    db = _mock_db()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    try:
        with patch("api.routes.trades.settings.SYSTEM_KIS_MODE", "real"), \
             patch("api.routes.trades.risk_service.check_order", new_callable=AsyncMock, return_value=None), \
             patch("api.routes.trades.kis_service.place_order",
                   new_callable=AsyncMock, return_value={"kis_order_no": "0000123456"}), \
             patch("api.routes.trades.poll_order_fill") as mock_task:
            mock_task.delay = MagicMock()
            resp = await client.post("/trades/order", json={
                "stock_code": "005930", "order_type": "BUY",
                "price_type": "LIMIT", "quantity": 1, "price": 70000
            })
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)
    assert resp.status_code == 200
    trade = db.add.call_args.args[0]
    assert trade.mode == "real"
    mock_task.delay.assert_called_once_with(
        str(trade.id), str(user.id), "0000123456", "real"
    )


async def test_order_returns_pending(client):
    """정상 주문은 PENDING 응답."""
    user = _mock_user()
    db = _mock_db()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    try:
        with patch("api.routes.trades.risk_service.check_order", new_callable=AsyncMock, return_value=None):
            with patch("api.routes.trades.kis_service.place_order",
                       new_callable=AsyncMock, return_value={"kis_order_no": "0000123456"}):
                with patch("api.routes.trades.poll_order_fill") as mock_task:
                    mock_task.delay = MagicMock()
                    resp = await client.post("/trades/order", json={
                        "stock_code": "005930", "order_type": "BUY",
                        "price_type": "LIMIT", "quantity": 1, "price": 70000
                    })
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)
    assert resp.status_code == 200
    assert resp.json()["status"] == "PENDING"


async def test_order_with_warning(client):
    """경고 모드에서 주문 통과 + warning 포함."""
    user = _mock_user()
    db = _mock_db()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    try:
        with patch("api.routes.trades.risk_service.check_order",
                   new_callable=AsyncMock, return_value="종목별 한도 초과: 25.0% > 20.0%"):
            with patch("api.routes.trades.kis_service.place_order",
                       new_callable=AsyncMock, return_value={"kis_order_no": "0000123456"}):
                with patch("api.routes.trades.poll_order_fill") as mock_task:
                    mock_task.delay = MagicMock()
                    resp = await client.post("/trades/order", json={
                        "stock_code": "005930", "order_type": "BUY",
                        "price_type": "LIMIT", "quantity": 1, "price": 70000
                    })
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)
    assert resp.status_code == 200
    assert "warning" in resp.json()


async def test_get_trades_list(client):
    """주문 목록 조회."""
    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        resp = await client.get("/trades")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_cancel_trade_not_found(client):
    """존재하지 않는 주문 취소 시 404."""
    user = _mock_user()
    fake_id = str(uuid.uuid4())
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        resp = await client.delete(f"/trades/{fake_id}")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 404


async def test_order_risk_hard_stop(client):
    """hard_stop 리스크 차단 시 400."""
    from fastapi import HTTPException
    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        with patch("api.routes.trades.risk_service.check_order",
                   new_callable=AsyncMock, side_effect=HTTPException(400, "한도 초과")):
            resp = await client.post("/trades/order", json={
                "stock_code": "005930", "order_type": "BUY",
                "price_type": "LIMIT", "quantity": 1, "price": 70000
            })
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 400


async def test_list_trades_limit_param(client):
    """limit query param이 100 기본값을 대체해야 한다."""
    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        resp = await client.get("/trades?limit=10")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_list_trades_limit_capped(client, db_session):
    """limit은 500을 초과할 수 없다 — 600개 seed 후 반환값이 500개를 넘지 않아야 한다."""
    user = _mock_user()
    await _seed_user(db_session, user.id)
    # 600개 trade 삽입 (limit=9999 요청 시 최대 500개만 반환돼야 한다)
    await _seed_trades(db_session, user.id, [{"status": "FILLED"} for _ in range(600)])
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        resp = await client.get("/trades?limit=9999")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 200
    result = resp.json()
    assert isinstance(result, list)
    assert len(result) <= 500, f"limit clamp 미작동: {len(result)}개 반환됨"


async def test_list_trades_status_filter(client, db_session):
    """status query param이 필터로 작동해야 한다 — PENDING/FILLED 혼합 seed 후 검증."""
    user = _mock_user()
    await _seed_user(db_session, user.id)
    # PENDING 3개 + FILLED 2개 삽입
    await _seed_trades(db_session, user.id, [
        {"status": "PENDING"},
        {"status": "PENDING"},
        {"status": "PENDING"},
        {"status": "FILLED"},
        {"status": "FILLED"},
    ])
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        resp = await client.get("/trades?status=PENDING")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 200
    result = resp.json()
    assert isinstance(result, list)
    # PENDING 3개가 반환돼야 하며 FILLED는 포함되면 안 된다
    assert len(result) >= 3, f"PENDING 거래가 {len(result)}개만 반환됨 (최소 3개 기대)"
    for trade in result:
        assert trade["status"] == "PENDING", f"status 필터 미작동: {trade['status']} 반환됨"


async def test_list_trades_limit_and_status_combined(client):
    """limit + status 조합 필터."""
    user = _mock_user()
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        resp = await client.get("/trades?limit=5&status=FILLED")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
