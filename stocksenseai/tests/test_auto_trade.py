"""자동매매 API 테스트 (7개 케이스)."""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from api.deps import get_current_user
from main import app


# ── 헬퍼 ────────────────────────────────────────────────────────────────────


def _mock_user(mode: str = "paper") -> MagicMock:
    """테스트용 더미 User 객체 생성."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = mode
    return user


async def _register_and_verify(client, email: str, password: str = "pass1234") -> None:
    """회원가입 + 이메일 인증 헬퍼."""
    with patch("api.routes.auth.send_verification_email"):
        await client.post("/auth/register", json={"email": email, "password": password})
    from core.security import create_email_token
    token = create_email_token(email)
    await client.post("/auth/verify-email", json={"token": token})


async def _get_token(client, email: str, password: str = "pass1234") -> str:
    """로그인 후 access_token 반환."""
    resp = await client.post("/auth/login", json={"email": email, "password": password})
    return resp.json()["access_token"]


# ── 테스트 케이스 ─────────────────────────────────────────────────────────────


async def test_config_requires_auth(client):
    """1. 토큰 없이 GET /auto-trade/config 요청 → 401."""
    resp = await client.get("/auto-trade/config")
    assert resp.status_code == 401


async def test_config_default_created(client):
    """2. 인증 후 GET /auto-trade/config → 200, 기본값 enabled=False, mode='paper'."""
    email = f"autotrade-default-{uuid.uuid4().hex[:8]}@test.com"
    await _register_and_verify(client, email)
    token = await _get_token(client, email)

    resp = await client.get(
        "/auto-trade/config",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False
    assert data["mode"] == "paper"


async def test_config_update_budget(client):
    """3. PUT /auto-trade/config → 200, 전달한 값이 응답에 반영된다."""
    email = f"autotrade-budget-{uuid.uuid4().hex[:8]}@test.com"
    await _register_and_verify(client, email)
    token = await _get_token(client, email)
    headers = {"Authorization": f"Bearer {token}"}

    resp = await client.put(
        "/auto-trade/config",
        json={"total_budget": 2_000_000, "stop_loss_pct": 3.0, "take_profit_pct": 15.0},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_budget"] == 2_000_000
    assert data["stop_loss_pct"] == 3.0
    assert data["take_profit_pct"] == 15.0


async def test_run_skipped_when_disabled(client):
    """4. enabled=False(기본값) 상태에서 POST /auto-trade/run → 200, skipped=True."""
    email = f"autotrade-run-{uuid.uuid4().hex[:8]}@test.com"
    await _register_and_verify(client, email)
    token = await _get_token(client, email)

    # enabled 기본값 False이므로 바로 run → skipped
    resp = await client.post(
        "/auto-trade/run",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("skipped") is True
    assert data.get("reason") == "not_enabled"


async def test_stop_disables(client):
    """5. enabled=True 로 PUT 후 POST /auto-trade/stop → 200, enabled=False가 된다."""
    email = f"autotrade-stop-{uuid.uuid4().hex[:8]}@test.com"
    await _register_and_verify(client, email)
    token = await _get_token(client, email)
    headers = {"Authorization": f"Bearer {token}"}

    # 먼저 enabled=True로 업데이트
    put_resp = await client.put(
        "/auto-trade/config",
        json={"enabled": True},
        headers=headers,
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["enabled"] is True

    # kill_switch 호출
    stop_resp = await client.post("/auto-trade/stop", headers=headers)
    assert stop_resp.status_code == 200
    assert stop_resp.json().get("stopped") is True

    # config 재조회 → enabled=False 확인
    cfg_resp = await client.get("/auto-trade/config", headers=headers)
    assert cfg_resp.status_code == 200
    assert cfg_resp.json()["enabled"] is False


async def test_invalid_mode_422(client):
    """6. PUT /auto-trade/config {mode: 'invalid_mode'} → 422 (Pydantic 검증 실패)."""
    email = f"autotrade-mode-{uuid.uuid4().hex[:8]}@test.com"
    await _register_and_verify(client, email)
    token = await _get_token(client, email)

    resp = await client.put(
        "/auto-trade/config",
        json={"mode": "invalid_mode"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


async def test_unused_budget_per_trade_is_rejected(client):
    """미구현 예약 필드 budget_per_trade는 조용히 무시하지 않고 422로 거절한다."""
    email = f"autotrade-budget-field-{uuid.uuid4().hex[:8]}@test.com"
    await _register_and_verify(client, email)
    token = await _get_token(client, email)

    resp = await client.put(
        "/auto-trade/config",
        json={"budget_per_trade": 100_000},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


async def test_logs_limit_clamp(client):
    """7. GET /auto-trade/logs?limit=999 → 200, 반환 로그는 최대 200개를 초과하지 않는다."""
    email = f"autotrade-logs-{uuid.uuid4().hex[:8]}@test.com"
    await _register_and_verify(client, email)
    token = await _get_token(client, email)

    resp = await client.get(
        "/auto-trade/logs?limit=999",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # 응답 형태: {"logs": [...], "count": N}
    assert "logs" in data
    assert len(data["logs"]) <= 200
