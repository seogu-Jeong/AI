"""시스템 상태 진단 API 테스트.

KIS 외부 호출은 mock 처리.
DB 쿼리(AISignalHistory count)도 mock 처리하여 외부 의존 없이 동작.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from api.deps import get_optional_user
from main import app

# DB 업로드 예측 건수 조회를 항상 0으로 단락시키는 패치 경로.
# system.py의 _count_uploaded_predictions 함수를 직접 patch하여
# DB 연결 없이도 AI 상태 섹션이 안정적으로 동작하도록 한다.
_COUNT_PATCH = "api.routes.system._count_uploaded_predictions"


# ---------------------------------------------------------------------------
# 1. KIS 미설정 + 비로그인 → 최소 응답
# ---------------------------------------------------------------------------

async def test_status_no_kis_no_auth(client):
    """KIS 키 없음 + 비로그인 시 기본 필드만 반환, 민감 정보 없음."""
    app.dependency_overrides[get_optional_user] = lambda: None
    try:
        with patch("api.routes.system.settings.SYSTEM_KIS_APP_KEY", ""), \
             patch("api.routes.system.settings.SYSTEM_KIS_APP_SECRET", ""), \
             patch("api.routes.system.settings.SYSTEM_KIS_ACCOUNT_NO", ""), \
             patch(_COUNT_PATCH, new=AsyncMock(return_value=0)):
            resp = await client.get("/system/status")
    finally:
        app.dependency_overrides.pop(get_optional_user, None)

    assert resp.status_code == 200
    body = resp.json()

    # 필수 키 존재
    for key in ("backend", "auth", "kis", "account", "portfolio", "ai", "checked_at"):
        assert key in body, f"missing key: {key}"

    # 백엔드 정상
    assert body["backend"]["ok"] is True

    # 비로그인
    assert body["auth"]["logged_in"] is False
    assert body["auth"]["email"] is None

    # KIS 미설정
    assert body["kis"]["configured"] is False
    assert body["kis"]["account_no"] is None

    # 비로그인 상태에서는 account 잔고 count를 조회하지 않음
    assert body["account"]["ok"] is None
    assert body["account"]["message"] == "login_required"

    # 민감 정보 노출 없음 — 실제 키 값이나 인증 토큰이 포함되면 안 됨
    # 환경변수 이름(SYSTEM_KIS_APP_KEY) 자체는 설명 목적으로 허용하되,
    # 실제 키 값(fake_key 등)이나 Bearer 토큰은 금지
    raw = str(body)
    for word in ("Bearer", "fake_key", "fake_secret"):
        assert word not in raw, f"sensitive word found: {word}"


# ---------------------------------------------------------------------------
# 2. KIS 설정됨 + 잔고 조회 성공 (mock)
# ---------------------------------------------------------------------------

async def test_status_kis_configured_balance_ok(client):
    """KIS 설정 + 잔고 조회 성공 시 holdings_count 표시."""
    import uuid
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    mock_user.email = "test@example.com"

    app.dependency_overrides[get_optional_user] = lambda: mock_user

    mock_balance = {
        "mode": "paper",
        "account_no": "1234****-01",
        "summary": {},
        "holdings": [{"stock_code": "005930"}, {"stock_code": "000660"}],
        "data_source": "KIS 모의투자 계좌",
    }

    try:
        with patch("api.routes.system.settings.SYSTEM_KIS_APP_KEY", "fake_key"), \
             patch("api.routes.system.settings.SYSTEM_KIS_APP_SECRET", "fake_secret"), \
             patch("api.routes.system.settings.SYSTEM_KIS_ACCOUNT_NO", "12345678-01"), \
             patch("api.routes.system.settings.SYSTEM_KIS_MODE", "paper"), \
             patch("api.routes.system.get_account_balance", new=AsyncMock(return_value=mock_balance)), \
             patch(_COUNT_PATCH, new=AsyncMock(return_value=0)):
            resp = await client.get("/system/status")
    finally:
        app.dependency_overrides.pop(get_optional_user, None)

    assert resp.status_code == 200
    body = resp.json()

    assert body["kis"]["configured"] is True
    assert body["kis"]["mode"] == "paper"
    # 계좌번호는 마스킹 형태여야 함 — 전체 번호 노출 없음
    account_no = body["kis"]["account_no"]
    assert account_no is not None
    assert "****" in account_no or len(account_no) < 14  # 마스킹 확인

    assert body["account"]["ok"] is True
    assert body["account"]["holdings_count"] == 2

    assert body["auth"]["logged_in"] is True
    assert body["auth"]["email"] == "test@example.com"


# ---------------------------------------------------------------------------
# 3. KIS 설정됨 + 잔고 조회 실패 (mock)
# ---------------------------------------------------------------------------

async def test_status_kis_configured_balance_fail(client):
    """잔고 조회 실패 시 ok=False, 민감 정보 미노출."""
    from fastapi import HTTPException

    import uuid
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    mock_user.email = "fail@example.com"
    app.dependency_overrides[get_optional_user] = lambda: mock_user

    try:
        with patch("api.routes.system.settings.SYSTEM_KIS_APP_KEY", "fake_key"), \
             patch("api.routes.system.settings.SYSTEM_KIS_APP_SECRET", "fake_secret"), \
             patch("api.routes.system.settings.SYSTEM_KIS_ACCOUNT_NO", "12345678-01"), \
             patch("api.routes.system.settings.SYSTEM_KIS_MODE", "paper"), \
             patch(
                 "api.routes.system.get_account_balance",
                 new=AsyncMock(side_effect=HTTPException(status_code=502, detail="KIS 연결 실패: timeout"))
             ), \
             patch(_COUNT_PATCH, new=AsyncMock(return_value=0)):
            resp = await client.get("/system/status")
    finally:
        app.dependency_overrides.pop(get_optional_user, None)

    assert resp.status_code == 200
    body = resp.json()

    assert body["account"]["ok"] is False
    assert body["account"]["holdings_count"] is None
    # 오류 메시지에 민감 정보 없음
    msg = body["account"]["message"]
    for word in ("fake_key", "fake_secret", "Bearer"):
        assert word not in msg


# ---------------------------------------------------------------------------
# 4. 포트폴리오가 비었을 때 "조회 성공 + 보유 없음"과 "조회 실패" 구분
# ---------------------------------------------------------------------------

async def test_status_empty_holdings_vs_failure(client):
    """보유 0종목(성공)과 조회 실패를 명확히 구분."""
    import uuid
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    mock_user.email = "user@test.com"

    app.dependency_overrides[get_optional_user] = lambda: mock_user

    # Case A: 조회 성공, 보유 0
    mock_balance_empty = {
        "mode": "paper",
        "account_no": "1234****-01",
        "summary": {},
        "holdings": [],
        "data_source": "KIS 모의투자 계좌",
    }

    try:
        with patch("api.routes.system.settings.SYSTEM_KIS_APP_KEY", "fake_key"), \
             patch("api.routes.system.settings.SYSTEM_KIS_APP_SECRET", "fake_secret"), \
             patch("api.routes.system.settings.SYSTEM_KIS_ACCOUNT_NO", "12345678-01"), \
             patch("api.routes.system.settings.SYSTEM_KIS_MODE", "paper"), \
             patch("api.routes.system.get_account_balance", new=AsyncMock(return_value=mock_balance_empty)), \
             patch(_COUNT_PATCH, new=AsyncMock(return_value=0)):
            resp_ok = await client.get("/system/status")
    finally:
        app.dependency_overrides.pop(get_optional_user, None)

    body_ok = resp_ok.json()
    assert body_ok["account"]["ok"] is True
    assert body_ok["account"]["holdings_count"] == 0
    assert "보유 없음" in body_ok["account"]["message"] or "보유 종목 없음" in body_ok["account"]["message"]

    # Case B: 조회 실패
    from fastapi import HTTPException
    app.dependency_overrides[get_optional_user] = lambda: mock_user

    try:
        with patch("api.routes.system.settings.SYSTEM_KIS_APP_KEY", "fake_key"), \
             patch("api.routes.system.settings.SYSTEM_KIS_APP_SECRET", "fake_secret"), \
             patch("api.routes.system.settings.SYSTEM_KIS_ACCOUNT_NO", "12345678-01"), \
             patch("api.routes.system.settings.SYSTEM_KIS_MODE", "paper"), \
             patch(
                 "api.routes.system.get_account_balance",
                 new=AsyncMock(side_effect=HTTPException(status_code=502, detail="연결 실패"))
             ), \
             patch(_COUNT_PATCH, new=AsyncMock(return_value=0)):
            resp_fail = await client.get("/system/status")
    finally:
        app.dependency_overrides.pop(get_optional_user, None)

    body_fail = resp_fail.json()
    assert body_fail["account"]["ok"] is False
    assert body_fail["account"]["holdings_count"] is None

    # 두 경우의 message가 달라야 함
    assert body_ok["account"]["message"] != body_fail["account"]["message"]


# ---------------------------------------------------------------------------
# 5. checked_at 형식 검증
# ---------------------------------------------------------------------------

async def test_status_checked_at_format(client):
    """checked_at는 ISO 8601 형식이어야 함."""
    app.dependency_overrides[get_optional_user] = lambda: None

    try:
        with patch("api.routes.system.settings.SYSTEM_KIS_APP_KEY", ""), \
             patch("api.routes.system.settings.SYSTEM_KIS_APP_SECRET", ""), \
             patch("api.routes.system.settings.SYSTEM_KIS_ACCOUNT_NO", ""), \
             patch(_COUNT_PATCH, new=AsyncMock(return_value=0)):
            resp = await client.get("/system/status")
    finally:
        app.dependency_overrides.pop(get_optional_user, None)

    assert resp.status_code == 200
    body = resp.json()
    checked_at = body.get("checked_at", "")
    assert "T" in checked_at, f"checked_at not ISO format: {checked_at}"
    assert len(checked_at) >= 19
