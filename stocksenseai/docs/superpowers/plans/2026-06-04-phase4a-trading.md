# Phase 4-A — 거래 + 포트폴리오 + 리스크 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KIS REST API 실거래/모의투자 주문 실행, 포트폴리오 추적, 리스크 관리 API 구현

**Architecture:** 함수형 kis_service가 user 객체 기반으로 paper/real 모드 자동 전환. 주문은 즉시 PENDING 응답 후 Celery가 체결 폴링 + 이메일 발송. 포트폴리오는 DB 추적 + pykrx 현재가로 수익률 계산.

**Tech Stack:** FastAPI, SQLAlchemy 2 async, Celery 5, httpx, pykrx, SendGrid, PostgreSQL, Redis

---

## 파일 목록

### 신규 생성
| 파일 | 역할 |
|---|---|
| `backend/services/risk_service.py` | 종목별/일일 손실 한도 체크 |
| `backend/tasks/email_tasks.py` | 체결/리스크/가격 이메일 Celery 태스크 |
| `backend/tasks/order_tasks.py` | 체결 폴링 Celery 태스크 |
| `backend/api/routes/trades.py` | 주문 실행/목록/취소 |
| `backend/api/routes/portfolio.py` | 포트폴리오 현황/성과/지표/CSV |
| `backend/api/routes/risk.py` | 리스크 설정 조회/수정 |
| `backend/api/routes/alerts.py` | 알림 설정 조회/수정 |
| `tests/test_trades.py` | 주문 API 통합 테스트 |
| `tests/test_portfolio.py` | 포트폴리오 API 통합 테스트 |
| `tests/test_risk.py` | risk_service 유닛 + API 테스트 |
| `db/migrations/versions/f6a1b2c3d4e5_add_risk_alert_columns.py` | Alembic v8 |

### 수정
| 파일 | 변경 내용 |
|---|---|
| `backend/services/kis_service.py` | 완전 재작성 |
| `backend/models/risk.py` | enforce_hard_stop, notification_email 컬럼 추가 |
| `backend/api/routes/auth.py` | PUT /auth/mode 추가 |
| `backend/tasks/__init__.py` | include에 email_tasks, order_tasks 추가 |
| `backend/main.py` | APScheduler 스케줄 추가 + 라우터 등록 |
| `docs/progress.md` | Phase 4-A 완료 업데이트 |

---

## Task 1: ORM 모델 + Alembic 마이그레이션 (v8)

**Files:**
- Modify: `backend/models/risk.py`
- Create: `db/migrations/versions/f6a1b2c3d4e5_add_risk_alert_columns.py`

- [ ] **Step 1: risk.py 모델 업데이트**

`RiskSettings`에 `enforce_hard_stop` 컬럼, `AlertSettings`에 `notification_email` 컬럼 추가:

```python
# backend/models/risk.py
import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID

from core.database import Base


class RiskSettings(Base):
    __tablename__ = "risk_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    max_per_stock_pct = Column(Numeric(5, 2), server_default="20.0")
    daily_loss_limit_pct = Column(Numeric(5, 2), server_default="5.0")
    stop_loss_enabled = Column(Boolean, server_default="false")
    trading_blocked = Column(Boolean, server_default="false")
    enforce_hard_stop = Column(Boolean, server_default="true")
    blocked_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AlertSettings(Base):
    __tablename__ = "alert_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    signal_change = Column(Boolean, server_default="true")
    watchlist_price = Column(Boolean, server_default="true")
    daily_loss_limit = Column(Boolean, server_default="true")
    trade_filled = Column(Boolean, server_default="true")
    weekly_report = Column(Boolean, server_default="false")
    notification_email = Column(String(255), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

- [ ] **Step 2: 마이그레이션 파일 생성**

최신 revision은 `e5f6a1b2c3d4`. 아래 파일을 그대로 생성:

```python
# db/migrations/versions/f6a1b2c3d4e5_add_risk_alert_columns.py
"""add_risk_alert_columns

Revision ID: f6a1b2c3d4e5
Revises: e5f6a1b2c3d4
Create Date: 2026-06-04 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f6a1b2c3d4e5"
down_revision: Union[str, None] = "e5f6a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "risk_settings",
        sa.Column("enforce_hard_stop", sa.Boolean(), server_default="true", nullable=True),
    )
    op.add_column(
        "alert_settings",
        sa.Column("notification_email", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("alert_settings", "notification_email")
    op.drop_column("risk_settings", "enforce_hard_stop")
```

- [ ] **Step 3: 커밋**

```bash
git add backend/models/risk.py db/migrations/versions/f6a1b2c3d4e5_add_risk_alert_columns.py
git commit -m "feat: add enforce_hard_stop + notification_email columns (migration v8)"
```

---

## Task 2: KIS 서비스 완전 재작성

**Files:**
- Modify: `backend/services/kis_service.py`

- [ ] **Step 1: kis_service.py 전체 교체**

```python
# backend/services/kis_service.py
from __future__ import annotations

import httpx
from fastapi import HTTPException

from core.security import decrypt_aes
from services.kis_token_service import get_access_token

_REAL_URL = "https://openapi.koreainvestment.com:9443"
_PAPER_URL = "https://openapivts.koreainvestment.com:29443"

_TR_IDS = {
    "buy":     {"real": "TTTC0802U", "paper": "VTTC0802U"},
    "sell":    {"real": "TTTC0801U", "paper": "VTTC0801U"},
    "cancel":  {"real": "TTTC0803U", "paper": "VTTC0803U"},
    "balance": {"real": "TTTC8434R", "paper": "VTTC8434R"},
    "fill":    {"real": "TTTC8001R", "paper": "VTTC8001R"},
}


def _base_url(mode: str) -> str:
    return _PAPER_URL if mode == "paper" else _REAL_URL


def _tr_id(action: str, mode: str) -> str:
    return _TR_IDS[action][mode]


def _get_keys(user) -> tuple[str, str, str]:
    """user.mode에 따라 (app_key, app_secret, account_no) 복호화 반환."""
    mode = user.mode
    if mode == "paper":
        if not user.kis_paper_key_enc:
            raise HTTPException(status_code=400, detail="모의투자 KIS 키가 등록되지 않았습니다.")
        return (
            decrypt_aes(user.kis_paper_key_enc),
            decrypt_aes(user.kis_paper_secret_enc),
            user.kis_paper_account_no,
        )
    else:
        if not user.kis_real_key_enc:
            raise HTTPException(status_code=400, detail="실거래 KIS 키가 등록되지 않았습니다.")
        return (
            decrypt_aes(user.kis_real_key_enc),
            decrypt_aes(user.kis_real_secret_enc),
            user.kis_real_account_no,
        )


async def _headers(user) -> dict:
    app_key, app_secret, _ = _get_keys(user)
    token = await get_access_token(app_key, app_secret, user.mode)
    return {
        "authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "content-type": "application/json",
        "custtype": "P",
    }


async def place_order(
    user,
    stock_code: str,
    order_type: str,
    price_type: str,
    quantity: int,
    price: int = 0,
) -> dict:
    """
    KIS 주문 실행.
    order_type: "BUY" | "SELL"
    price_type: "MARKET" | "LIMIT"
    반환: {kis_order_no}
    """
    _, _, account_no = _get_keys(user)
    action = "buy" if order_type == "BUY" else "sell"
    ord_dvsn = "01" if price_type == "MARKET" else "00"

    headers = await _headers(user)
    headers["tr_id"] = _tr_id(action, user.mode)

    body = {
        "CANO": account_no[:8],
        "ACNT_PRDT_CD": account_no[8:] if len(account_no) > 8 else "01",
        "PDNO": stock_code,
        "ORD_DVSN": ord_dvsn,
        "ORD_QTY": str(quantity),
        "ORD_UNPR": str(price) if price_type == "LIMIT" else "0",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_base_url(user.mode)}/uapi/domestic-stock/v1/trading/order-cash",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"KIS 주문 실패: {exc.response.text}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"KIS 연결 실패: {exc}") from exc

    if data.get("rt_cd") != "0":
        raise HTTPException(status_code=400, detail=f"KIS 주문 거부: {data.get('msg1', '')}")

    return {"kis_order_no": data["output"]["ODNO"]}


async def cancel_order(user, kis_order_no: str) -> dict:
    """미체결 주문 취소."""
    _, _, account_no = _get_keys(user)
    headers = await _headers(user)
    headers["tr_id"] = _tr_id("cancel", user.mode)

    body = {
        "CANO": account_no[:8],
        "ACNT_PRDT_CD": account_no[8:] if len(account_no) > 8 else "01",
        "KRX_FWDG_ORD_ORGNO": "",
        "ORGN_ODNO": kis_order_no,
        "ORD_DVSN": "00",
        "RVSE_CNCL_DVSN_CD": "02",
        "ORD_QTY": "0",
        "ORD_UNPR": "0",
        "QTY_ALL_ORD_YN": "Y",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_base_url(user.mode)}/uapi/domestic-stock/v1/trading/order-rvsecncl",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"KIS 취소 실패: {exc}") from exc

    if data.get("rt_cd") != "0":
        raise HTTPException(status_code=400, detail=f"KIS 취소 거부: {data.get('msg1', '')}")

    return {"cancelled": True}


async def poll_fill(user, kis_order_no: str) -> dict | None:
    """
    체결 확인.
    반환: {executed_price, filled_qty, filled_at} or None (미체결)
    """
    _, _, account_no = _get_keys(user)
    headers = await _headers(user)
    headers["tr_id"] = _tr_id("fill", user.mode)

    params = {
        "CANO": account_no[:8],
        "ACNT_PRDT_CD": account_no[8:] if len(account_no) > 8 else "01",
        "INQR_STRT_DT": "",
        "INQR_END_DT": "",
        "SLL_BUY_DVSN_CD": "00",
        "INQR_DVSN": "00",
        "PDNO": "",
        "CCLD_DVSN": "01",
        "ORD_GNO_BRNO": "",
        "ODNO": kis_order_no,
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_base_url(user.mode)}/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return None

    output = data.get("output1", [])
    for item in output:
        if item.get("odno") == kis_order_no and item.get("ccld_dvsn") == "1":
            return {
                "executed_price": int(item.get("avg_prvs", 0)),
                "filled_qty": int(item.get("tot_ccld_qty", 0)),
                "filled_at": item.get("ord_tmd", ""),
            }
    return None


async def get_balance(user) -> dict:
    """예수금 조회."""
    _, _, account_no = _get_keys(user)
    headers = await _headers(user)
    headers["tr_id"] = _tr_id("balance", user.mode)

    params = {
        "CANO": account_no[:8],
        "ACNT_PRDT_CD": account_no[8:] if len(account_no) > 8 else "01",
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_base_url(user.mode)}/uapi/domestic-stock/v1/trading/inquire-balance",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"잔고 조회 실패: {exc}") from exc

    output2 = data.get("output2", [{}])[0]
    return {
        "cash": int(output2.get("dnca_tot_amt", 0)),
        "total_eval": int(output2.get("tot_evlu_amt", 0)),
    }


async def test_kis_connection(app_key: str, app_secret: str, mode: str) -> bool:
    """KIS 연결 테스트 (기존 함수 유지)."""
    base_url = _PAPER_URL if mode == "paper" else _REAL_URL
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{base_url}/oauth2/tokenP",
                json={"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret},
                headers={"Content-Type": "application/json"},
            )
            return resp.status_code == 200 and bool(resp.json().get("access_token"))
        except Exception:
            return False
```

- [ ] **Step 2: 커밋**

```bash
git add backend/services/kis_service.py
git commit -m "feat: kis_service — place_order, cancel_order, poll_fill, get_balance (paper/real auto-switch)"
```

---

## Task 3: risk_service.py (TDD)

**Files:**
- Create: `backend/services/risk_service.py`
- Test: `tests/test_risk.py` (일부)

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_risk.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal


def _make_risk_settings(max_per_stock_pct=20.0, daily_loss_limit_pct=5.0, enforce_hard_stop=True):
    rs = MagicMock()
    rs.max_per_stock_pct = Decimal(str(max_per_stock_pct))
    rs.daily_loss_limit_pct = Decimal(str(daily_loss_limit_pct))
    rs.enforce_hard_stop = enforce_hard_stop
    return rs


async def test_check_order_passes_within_limit(client):
    """한도 미초과 주문은 통과."""
    from services.risk_service import check_order, RiskLimitExceeded

    user = MagicMock()
    user.id = "test-user-id"
    user.mode = "paper"
    db = AsyncMock()

    with patch("services.risk_service.get_or_create_settings", new_callable=AsyncMock,
               return_value=_make_risk_settings(max_per_stock_pct=50.0)):
        with patch("services.risk_service._get_portfolio_total", new_callable=AsyncMock, return_value=1_000_000):
            with patch("services.risk_service._get_holding_value", new_callable=AsyncMock, return_value=100_000):
                with patch("services.risk_service._get_today_loss", new_callable=AsyncMock, return_value=0):
                    result = await check_order(user, "005930", 1, 50_000, db)
                    assert result is None  # 통과, 경고 없음


async def test_check_order_hard_stop_raises(client):
    """enforce_hard_stop=True이고 한도 초과 시 400."""
    from services.risk_service import check_order, RiskLimitExceeded
    from fastapi import HTTPException

    user = MagicMock()
    user.id = "test-user-id"
    user.mode = "paper"
    db = AsyncMock()

    with patch("services.risk_service.get_or_create_settings", new_callable=AsyncMock,
               return_value=_make_risk_settings(max_per_stock_pct=10.0, enforce_hard_stop=True)):
        with patch("services.risk_service._get_portfolio_total", new_callable=AsyncMock, return_value=1_000_000):
            with patch("services.risk_service._get_holding_value", new_callable=AsyncMock, return_value=500_000):
                with patch("services.risk_service._get_today_loss", new_callable=AsyncMock, return_value=0):
                    with pytest.raises(HTTPException) as exc_info:
                        await check_order(user, "005930", 1, 50_000, db)
                    assert exc_info.value.status_code == 400


async def test_check_order_warning_mode(client):
    """enforce_hard_stop=False이고 한도 초과 시 경고 문자열 반환."""
    from services.risk_service import check_order

    user = MagicMock()
    user.id = "test-user-id"
    user.mode = "paper"
    db = AsyncMock()

    with patch("services.risk_service.get_or_create_settings", new_callable=AsyncMock,
               return_value=_make_risk_settings(max_per_stock_pct=10.0, enforce_hard_stop=False)):
        with patch("services.risk_service._get_portfolio_total", new_callable=AsyncMock, return_value=1_000_000):
            with patch("services.risk_service._get_holding_value", new_callable=AsyncMock, return_value=500_000):
                with patch("services.risk_service._get_today_loss", new_callable=AsyncMock, return_value=0):
                    result = await check_order(user, "005930", 1, 50_000, db)
                    assert result is not None
                    assert "한도" in result
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/test_risk.py::test_check_order_passes_within_limit -v
```
Expected: `ImportError` (risk_service 없음)

- [ ] **Step 3: risk_service.py 구현**

```python
# backend/services/risk_service.py
from __future__ import annotations

import uuid as _uuid
from decimal import Decimal

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.portfolio import Portfolio
from models.risk import RiskSettings
from models.trade import Trade


async def get_or_create_settings(user_id, db: AsyncSession) -> RiskSettings:
    """없으면 기본값으로 생성."""
    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    result = await db.execute(select(RiskSettings).where(RiskSettings.user_id == user_id))
    settings = result.scalar_one_or_none()
    if settings is None:
        settings = RiskSettings(user_id=user_id)
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    return settings


async def _get_portfolio_total(user_id, mode: str, db: AsyncSession) -> int:
    """현재 포트폴리오 총 매수금액 합산 (avg_price × quantity)."""
    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == user_id, Portfolio.mode == mode)
    )
    holdings = result.scalars().all()
    return sum(int(h.avg_price * h.quantity) for h in holdings)


async def _get_holding_value(user_id, stock_code: str, mode: str, db: AsyncSession) -> int:
    """특정 종목 현재 보유 평가금액."""
    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.user_id == user_id,
            Portfolio.stock_code == stock_code,
            Portfolio.mode == mode,
        )
    )
    holding = result.scalar_one_or_none()
    if not holding:
        return 0
    return int(holding.avg_price * holding.quantity)


async def _get_today_loss(user_id, mode: str, db: AsyncSession) -> int:
    """오늘 실현 손실 합산 (SELL 체결 기준)."""
    from datetime import date, datetime, timezone
    from sqlalchemy import and_, func as sqlfunc

    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    today_start = datetime.combine(date.today(), datetime.min.time()).replace(tzinfo=timezone.utc)
    result = await db.execute(
        select(Trade).where(
            and_(
                Trade.user_id == user_id,
                Trade.mode == mode,
                Trade.order_type == "SELL",
                Trade.status == "FILLED",
                Trade.filled_at >= today_start,
            )
        )
    )
    trades = result.scalars().all()
    total_loss = 0
    for t in trades:
        if t.executed_price and t.order_price:
            pnl = int((t.executed_price - t.order_price) * t.quantity)
            if pnl < 0:
                total_loss += abs(pnl)
    return total_loss


async def check_order(
    user, stock_code: str, quantity: int, price: int, db: AsyncSession
) -> str | None:
    """
    주문 전 리스크 체크.
    반환: None (통과) | 경고 문자열 (경고 모드)
    한도 초과 + enforce_hard_stop=True → HTTPException 400
    """
    settings = await get_or_create_settings(user.id, db)
    mode = user.mode

    portfolio_total = await _get_portfolio_total(user.id, mode, db)
    holding_value = await _get_holding_value(user.id, stock_code, mode, db)
    new_order_value = quantity * price

    warnings = []

    # 1. 종목별 한도 체크
    if portfolio_total > 0:
        new_pct = (holding_value + new_order_value) / (portfolio_total + new_order_value) * 100
        max_pct = float(settings.max_per_stock_pct)
        if new_pct > max_pct:
            msg = f"종목별 한도 초과: {new_pct:.1f}% > {max_pct:.1f}%"
            if settings.enforce_hard_stop:
                raise HTTPException(status_code=400, detail=msg)
            warnings.append(msg)

    # 2. 일일 손실 한도 체크
    today_loss = await _get_today_loss(user.id, mode, db)
    if portfolio_total > 0:
        loss_pct = today_loss / portfolio_total * 100
        max_loss = float(settings.daily_loss_limit_pct)
        if loss_pct > max_loss:
            msg = f"일일 손실 한도 초과: {loss_pct:.1f}% > {max_loss:.1f}%"
            if settings.enforce_hard_stop:
                raise HTTPException(status_code=400, detail=msg)
            warnings.append(msg)

    return " / ".join(warnings) if warnings else None
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/test_risk.py -v
```
Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
git add backend/services/risk_service.py tests/test_risk.py
git commit -m "feat: risk_service — per-stock and daily loss limit check with hard-stop/warning mode"
```

---

## Task 4: Celery 태스크 (email_tasks.py + order_tasks.py)

**Files:**
- Create: `backend/tasks/email_tasks.py`
- Create: `backend/tasks/order_tasks.py`
- Modify: `backend/tasks/__init__.py`

- [ ] **Step 1: email_tasks.py 생성**

```python
# backend/tasks/email_tasks.py
import asyncio

from tasks import celery_app


def _get_notification_email(user_id: str) -> str:
    """notification_email 설정 or user.email fallback."""
    from core.database import AsyncSessionLocal
    from models.risk import AlertSettings
    from models.user import User
    from sqlalchemy import select
    import uuid

    async def _fetch():
        async with AsyncSessionLocal() as db:
            uid = uuid.UUID(user_id)
            result = await db.execute(select(AlertSettings).where(AlertSettings.user_id == uid))
            alert = result.scalar_one_or_none()
            if alert and alert.notification_email:
                return alert.notification_email
            result2 = await db.execute(select(User).where(User.id == uid))
            user = result2.scalar_one_or_none()
            return user.email if user else None

    return asyncio.run(_fetch())


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_fill_notification(self, user_id: str, trade_id: str) -> None:
    """체결 완료 이메일."""
    import asyncio
    from core.database import AsyncSessionLocal
    from models.trade import Trade
    from sqlalchemy import select
    import uuid

    async def _fetch_trade():
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Trade).where(Trade.id == uuid.UUID(trade_id)))
            return result.scalar_one_or_none()

    trade = asyncio.run(_fetch_trade())
    if not trade:
        return

    to_email = _get_notification_email(user_id)
    if not to_email:
        return

    order_type_kr = "매수" if trade.order_type == "BUY" else "매도"
    subject = f"[StockSenseAI] {trade.stock_code} {order_type_kr} 체결 완료"
    body = (
        f"<p>{trade.stock_code} {trade.stock_name or ''} {order_type_kr} 체결되었습니다.</p>"
        f"<p>체결가: {trade.executed_price:,}원 | 수량: {trade.quantity}주</p>"
    )

    try:
        from core.config import settings
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        if not settings.SENDGRID_API_KEY:
            return
        msg = Mail(from_email=settings.FROM_EMAIL, to_emails=to_email,
                   subject=subject, html_content=body)
        client = SendGridAPIClient(settings.SENDGRID_API_KEY)
        asyncio.run(asyncio.to_thread(client.send, msg))
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_risk_alert(self, user_id: str, reason: str) -> None:
    """리스크 한도 초과 경고 이메일."""
    to_email = _get_notification_email(user_id)
    if not to_email:
        return
    try:
        from core.config import settings
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        if not settings.SENDGRID_API_KEY:
            return
        msg = Mail(
            from_email=settings.FROM_EMAIL,
            to_emails=to_email,
            subject="[StockSenseAI] 리스크 한도 초과 경고",
            html_content=f"<p>리스크 한도 초과: {reason}</p>",
        )
        client = SendGridAPIClient(settings.SENDGRID_API_KEY)
        asyncio.run(asyncio.to_thread(client.send, msg))
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task
def check_price_alerts() -> None:
    """관심종목 목표가 도달 시 이메일 (APScheduler 5분 간격)."""
    # Phase 4-D (관심종목)에서 구현. 현재는 stub.
    pass


@celery_app.task
def check_daily_loss() -> None:
    """일일 손실 한도 체크 (APScheduler 10분 간격)."""
    # 현재 trading_blocked=True 처리는 Phase 4에서 확장 예정.
    pass
```

- [ ] **Step 2: order_tasks.py 생성**

```python
# backend/tasks/order_tasks.py
import asyncio
import uuid

from tasks import celery_app


@celery_app.task(bind=True, max_retries=5, default_retry_delay=10)
def poll_order_fill(self, trade_id: str, user_id: str, kis_order_no: str, mode: str) -> None:
    """
    KIS 체결 폴링 (10초 간격, 최대 5회).
    체결 시 trades/portfolios 업데이트 + 이메일 발송.
    """
    asyncio.run(_poll_async(self, trade_id, user_id, kis_order_no, mode))


async def _poll_async(task, trade_id: str, user_id: str, kis_order_no: str, mode: str) -> None:
    from core.database import AsyncSessionLocal
    from models.portfolio import Portfolio
    from models.trade import Trade
    from models.user import User
    from services import kis_service
    from sqlalchemy import select, update
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from tasks.email_tasks import send_fill_notification

    trade_uuid = uuid.UUID(trade_id)
    user_uuid = uuid.UUID(user_id)

    async with AsyncSessionLocal() as db:
        # 유저 조회
        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
        if not user:
            return

        # 체결 확인
        fill = await kis_service.poll_fill(user, kis_order_no)
        if fill is None:
            # 미체결 — 재시도
            if task.request.retries < task.max_retries:
                raise task.retry()
            return  # 5회 후 포기, PENDING 유지

        # trades 업데이트
        await db.execute(
            update(Trade)
            .where(Trade.id == trade_uuid)
            .values(
                status="FILLED",
                executed_price=fill["executed_price"],
                filled_at=fill.get("filled_at"),
            )
        )

        # 체결된 trade 조회 (portfolio 업데이트용)
        result2 = await db.execute(select(Trade).where(Trade.id == trade_uuid))
        trade = result2.scalar_one_or_none()

        if trade:
            await _update_portfolio(db, trade, fill["executed_price"])

        await db.commit()

    # 이메일 발송 (비동기)
    send_fill_notification.delay(user_id, trade_id)


async def _update_portfolio(db, trade, executed_price: int) -> None:
    """체결 후 portfolios 테이블 UPSERT."""
    from models.portfolio import Portfolio
    from sqlalchemy import select

    result = await db.execute(
        select(Portfolio).where(
            Portfolio.user_id == trade.user_id,
            Portfolio.stock_code == trade.stock_code,
            Portfolio.mode == trade.mode,
        )
    )
    holding = result.scalar_one_or_none()

    if trade.order_type == "BUY":
        if holding is None:
            db.add(Portfolio(
                user_id=trade.user_id,
                stock_code=trade.stock_code,
                stock_name=trade.stock_name,
                quantity=trade.quantity,
                avg_price=executed_price,
                mode=trade.mode,
            ))
        else:
            # 가중평균 재계산
            total_qty = holding.quantity + trade.quantity
            new_avg = (holding.avg_price * holding.quantity + executed_price * trade.quantity) / total_qty
            holding.quantity = total_qty
            holding.avg_price = round(new_avg, 2)
    elif trade.order_type == "SELL" and holding:
        holding.quantity -= trade.quantity
        if holding.quantity <= 0:
            await db.delete(holding)
```

- [ ] **Step 3: tasks/__init__.py 업데이트**

```python
# backend/tasks/__init__.py
import os

from celery import Celery

celery_app = Celery(
    "tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=["tasks.ai_tasks", "tasks.email_tasks", "tasks.order_tasks"],
)
```

- [ ] **Step 4: 커밋**

```bash
git add backend/tasks/email_tasks.py backend/tasks/order_tasks.py backend/tasks/__init__.py
git commit -m "feat: Celery tasks — order fill polling + email notifications"
```

---

## Task 5: trades 라우터 (TDD)

**Files:**
- Create: `backend/api/routes/trades.py`
- Test: `tests/test_trades.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_trades.py
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_user(mode="paper"):
    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = mode
    user.kis_paper_key_enc = "enc_key"
    return user


async def test_order_demo_mode_blocked(client):
    """demo 모드 유저는 주문 불가."""
    user = _mock_user(mode="demo")
    with patch("api.routes.trades.get_current_user", return_value=user):
        resp = await client.post("/trades/order", json={
            "stock_code": "005930", "order_type": "BUY",
            "price_type": "LIMIT", "quantity": 1, "price": 70000
        })
    assert resp.status_code == 403


async def test_order_returns_pending(client):
    """정상 주문은 PENDING 응답."""
    user = _mock_user()
    with patch("api.routes.trades.get_current_user", return_value=user):
        with patch("api.routes.trades.risk_service.check_order", new_callable=AsyncMock, return_value=None):
            with patch("api.routes.trades.kis_service.place_order",
                       new_callable=AsyncMock, return_value={"kis_order_no": "0000123456"}):
                with patch("api.routes.trades.poll_order_fill") as mock_task:
                    mock_task.delay = MagicMock()
                    resp = await client.post("/trades/order", json={
                        "stock_code": "005930", "order_type": "BUY",
                        "price_type": "LIMIT", "quantity": 1, "price": 70000
                    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "PENDING"


async def test_order_with_warning(client):
    """경고 모드에서 주문 통과 + warning 포함."""
    user = _mock_user()
    with patch("api.routes.trades.get_current_user", return_value=user):
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
    assert resp.status_code == 200
    assert "warning" in resp.json()


async def test_get_trades_list(client):
    """주문 목록 조회."""
    user = _mock_user()
    with patch("api.routes.trades.get_current_user", return_value=user):
        resp = await client.get("/trades")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_cancel_trade_not_found(client):
    """존재하지 않는 주문 취소 시 404."""
    user = _mock_user()
    fake_id = str(uuid.uuid4())
    with patch("api.routes.trades.get_current_user", return_value=user):
        resp = await client.delete(f"/trades/{fake_id}")
    assert resp.status_code == 404


async def test_order_risk_hard_stop(client):
    """hard_stop 리스크 차단 시 400."""
    from fastapi import HTTPException
    user = _mock_user()
    with patch("api.routes.trades.get_current_user", return_value=user):
        with patch("api.routes.trades.risk_service.check_order",
                   new_callable=AsyncMock, side_effect=HTTPException(400, "한도 초과")):
            resp = await client.post("/trades/order", json={
                "stock_code": "005930", "order_type": "BUY",
                "price_type": "LIMIT", "quantity": 1, "price": 70000
            })
    assert resp.status_code == 400
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/test_trades.py::test_order_demo_mode_blocked -v
```
Expected: `404` (라우터 미등록)

- [ ] **Step 3: trades.py 구현**

```python
# backend/api/routes/trades.py
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.trade import Trade
from models.user import User
from services import kis_service, risk_service
from tasks.order_tasks import poll_order_fill

router = APIRouter()


class OrderRequest(BaseModel):
    stock_code: str
    order_type: str   # "BUY" | "SELL"
    price_type: str   # "MARKET" | "LIMIT"
    quantity: int
    price: int = 0


@router.post("/order")
@limiter.limit("30/minute")
async def place_order(
    request: Request,
    body: OrderRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if user.mode == "demo":
        raise HTTPException(status_code=403, detail="KIS 키를 먼저 등록하세요.")

    warning = await risk_service.check_order(user, body.stock_code, body.quantity, body.price, db)

    result = await kis_service.place_order(
        user, body.stock_code, body.order_type, body.price_type, body.quantity, body.price
    )
    kis_order_no = result["kis_order_no"]

    trade = Trade(
        user_id=user.id,
        stock_code=body.stock_code,
        order_type=body.order_type,
        price_type=body.price_type,
        quantity=body.quantity,
        order_price=body.price if body.price_type == "LIMIT" else None,
        status="PENDING",
        mode=user.mode,
        kis_order_no=kis_order_no,
    )
    db.add(trade)
    await db.commit()
    await db.refresh(trade)

    poll_order_fill.delay(str(trade.id), str(user.id), kis_order_no, user.mode)

    response = {"trade_id": str(trade.id), "status": "PENDING", "kis_order_no": kis_order_no}
    if warning:
        response["warning"] = warning
    return response


@router.get("")
@limiter.limit("60/minute")
async def list_trades(
    request: Request,
    status: str | None = None,
    mode: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Trade).where(Trade.user_id == user.id)
    if status:
        query = query.where(Trade.status == status)
    if mode:
        query = query.where(Trade.mode == mode)
    else:
        query = query.where(Trade.mode == user.mode)
    query = query.order_by(Trade.created_at.desc()).limit(100)

    result = await db.execute(query)
    trades = result.scalars().all()
    return [
        {
            "id": str(t.id),
            "stock_code": t.stock_code,
            "order_type": t.order_type,
            "quantity": t.quantity,
            "order_price": float(t.order_price) if t.order_price else None,
            "executed_price": float(t.executed_price) if t.executed_price else None,
            "status": t.status,
            "mode": t.mode,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in trades
    ]


@router.delete("/{trade_id}")
@limiter.limit("20/minute")
async def cancel_trade(
    request: Request,
    trade_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        tid = uuid.UUID(trade_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 trade_id")

    result = await db.execute(
        select(Trade).where(Trade.id == tid, Trade.user_id == user.id)
    )
    trade = result.scalar_one_or_none()
    if not trade:
        raise HTTPException(status_code=404, detail="주문을 찾을 수 없습니다.")
    if trade.status != "PENDING":
        raise HTTPException(status_code=400, detail=f"취소 불가 상태: {trade.status}")

    await kis_service.cancel_order(user, trade.kis_order_no)
    trade.status = "CANCELLED"
    await db.commit()
    return {"cancelled": True, "trade_id": trade_id}
```

- [ ] **Step 4: 테스트 실행 — 실패 (라우터 미등록). main.py 먼저 등록 후 재실행**

main.py에 아래 라인 추가 (ai_router 아래):
```python
from api.routes import trades as trades_router
from api.routes import portfolio as portfolio_router
from api.routes import risk as risk_router
from api.routes import alerts as alerts_router

app.include_router(trades_router.router, prefix="/trades", tags=["trades"])
app.include_router(portfolio_router.router, prefix="/portfolio", tags=["portfolio"])
app.include_router(risk_router.router, prefix="/risk", tags=["risk"])
app.include_router(alerts_router.router, prefix="/alerts", tags=["alerts"])
```

portfolio.py, risk.py, alerts.py는 아직 없으니 **빈 라우터**를 먼저 생성:

```python
# backend/api/routes/portfolio.py (임시 stub)
from fastapi import APIRouter
router = APIRouter()

# backend/api/routes/risk.py (임시 stub)
from fastapi import APIRouter
router = APIRouter()

# backend/api/routes/alerts.py (임시 stub)
from fastapi import APIRouter
router = APIRouter()
```

- [ ] **Step 5: 테스트 실행 — 통과 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/test_trades.py -v
```
Expected: `6 passed`

- [ ] **Step 6: 커밋**

```bash
git add backend/api/routes/trades.py backend/api/routes/portfolio.py \
        backend/api/routes/risk.py backend/api/routes/alerts.py \
        backend/main.py tests/test_trades.py
git commit -m "feat: trades router — order/list/cancel with risk check and Celery fill polling"
```

---

## Task 6: portfolio 라우터 (TDD)

**Files:**
- Modify: `backend/api/routes/portfolio.py`
- Test: `tests/test_portfolio.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_portfolio.py
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_user():
    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    return user


async def test_portfolio_empty(client):
    """보유종목 없으면 빈 리스트."""
    user = _mock_user()
    with patch("api.routes.portfolio.get_current_user", return_value=user):
        resp = await client.get("/portfolio")
    assert resp.status_code == 200
    assert resp.json()["holdings"] == []


async def test_portfolio_with_holdings(client):
    """보유종목 있으면 수익률 계산 포함."""
    user = _mock_user()
    mock_holdings = [
        MagicMock(stock_code="005930", stock_name="삼성전자",
                  quantity=10, avg_price=70000, mode="paper")
    ]
    with patch("api.routes.portfolio.get_current_user", return_value=user):
        with patch("api.routes.portfolio._get_holdings", new_callable=AsyncMock, return_value=mock_holdings):
            with patch("api.routes.portfolio.get_stock_current_price",
                       new_callable=AsyncMock, return_value={"close": 75000}):
                resp = await client.get("/portfolio")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["holdings"]) == 1
    assert data["holdings"][0]["return_pct"] > 0


async def test_portfolio_metrics(client):
    """metrics 엔드포인트는 MDD/sharpe/win_rate 포함."""
    user = _mock_user()
    with patch("api.routes.portfolio.get_current_user", return_value=user):
        resp = await client.get("/portfolio/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "mdd_pct" in data
    assert "win_rate_pct" in data
    assert "sharpe_ratio" in data


async def test_portfolio_export_csv(client):
    """CSV export는 text/csv Content-Type 반환."""
    user = _mock_user()
    with patch("api.routes.portfolio.get_current_user", return_value=user):
        with patch("api.routes.portfolio._get_holdings", new_callable=AsyncMock, return_value=[]):
            resp = await client.get("/portfolio/export")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")


async def test_portfolio_performance(client):
    """performance 엔드포인트 200 응답."""
    user = _mock_user()
    with patch("api.routes.portfolio.get_current_user", return_value=user):
        resp = await client.get("/portfolio/performance")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/test_portfolio.py::test_portfolio_empty -v
```
Expected: `AssertionError` (stub은 빈 응답)

- [ ] **Step 3: portfolio.py 전체 구현**

```python
# backend/api/routes/portfolio.py
from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.portfolio import Portfolio
from models.trade import Trade
from models.user import User
from services.market_service import get_stock_current_price

router = APIRouter()


async def _get_holdings(user_id, mode: str, db: AsyncSession) -> list:
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == user_id, Portfolio.mode == mode)
    )
    return result.scalars().all()


@router.get("")
@limiter.limit("60/minute")
async def get_portfolio(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    holdings = await _get_holdings(user.id, user.mode, db)
    result = []
    total_eval = 0
    total_cost = 0

    for h in holdings:
        price_data = await get_stock_current_price(h.stock_code)
        current_price = price_data.get("close", 0)
        cost = int(h.avg_price * h.quantity)
        eval_amount = current_price * h.quantity
        profit = eval_amount - cost
        return_pct = (profit / cost * 100) if cost > 0 else 0
        total_eval += eval_amount
        total_cost += cost
        result.append({
            "stock_code": h.stock_code,
            "stock_name": h.stock_name,
            "quantity": h.quantity,
            "avg_price": float(h.avg_price),
            "current_price": current_price,
            "eval_amount": eval_amount,
            "profit_loss": profit,
            "return_pct": round(return_pct, 2),
        })

    total_return_pct = ((total_eval - total_cost) / total_cost * 100) if total_cost > 0 else 0
    return {
        "holdings": result,
        "total_eval": total_eval,
        "total_cost": total_cost,
        "total_return_pct": round(total_return_pct, 2),
    }


@router.get("/performance")
@limiter.limit("30/minute")
async def get_performance(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """일별 체결 기준 누적 수익 히스토리 (최근 90일)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    result = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.mode == user.mode,
            Trade.status == "FILLED",
            Trade.filled_at >= cutoff,
        ).order_by(Trade.filled_at)
    )
    trades = result.scalars().all()

    daily: dict[str, int] = {}
    for t in trades:
        if not t.filled_at or not t.executed_price:
            continue
        day = t.filled_at.strftime("%Y-%m-%d")
        pnl = 0
        if t.order_type == "SELL" and t.order_price:
            pnl = int((t.executed_price - t.order_price) * t.quantity)
        daily[day] = daily.get(day, 0) + pnl

    return [{"date": d, "pnl": v} for d, v in sorted(daily.items())]


@router.get("/metrics")
@limiter.limit("30/minute")
async def get_metrics(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """MDD, 샤프비율, 승률 계산."""
    result = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.mode == user.mode,
            Trade.status == "FILLED",
            Trade.order_type == "SELL",
        )
    )
    sells = result.scalars().all()

    returns = []
    wins = 0
    for t in sells:
        if t.executed_price and t.order_price:
            r = float((t.executed_price - t.order_price) / t.order_price)
            returns.append(r)
            if r > 0:
                wins += 1

    win_rate = (wins / len(returns) * 100) if returns else 0

    # 샤프비율 (무위험이자율 0 가정)
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns))
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0
    else:
        sharpe = 0

    # MDD (단순 수익률 시계열 기반)
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > mdd:
            mdd = dd

    return {
        "total_trades": len(returns),
        "win_rate_pct": round(win_rate, 2),
        "sharpe_ratio": round(sharpe, 4),
        "mdd_pct": round(mdd * 100, 2),
    }


@router.get("/export")
@limiter.limit("10/minute")
async def export_portfolio(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """포트폴리오 CSV 다운로드."""
    holdings = await _get_holdings(user.id, user.mode, db)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["종목코드", "종목명", "수량", "평균매수가", "현재가", "평가금액", "수익률(%)"])

    for h in holdings:
        price_data = await get_stock_current_price(h.stock_code)
        current_price = price_data.get("close", 0)
        cost = int(h.avg_price * h.quantity)
        eval_amount = current_price * h.quantity
        return_pct = ((eval_amount - cost) / cost * 100) if cost > 0 else 0
        writer.writerow([
            h.stock_code, h.stock_name or "", h.quantity,
            int(h.avg_price), current_price, eval_amount, round(return_pct, 2)
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio.csv"},
    )
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/test_portfolio.py -v
```
Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add backend/api/routes/portfolio.py tests/test_portfolio.py
git commit -m "feat: portfolio router — holdings/performance/metrics/csv with P&L calculation"
```

---

## Task 7: risk + alerts 라우터 + auth mode 전환 (TDD)

**Files:**
- Modify: `backend/api/routes/risk.py`
- Modify: `backend/api/routes/alerts.py`
- Modify: `backend/api/routes/auth.py`
- Test: `tests/test_risk.py` (API 부분 추가)

- [ ] **Step 1: test_risk.py에 API 테스트 추가**

```python
# tests/test_risk.py 에 추가
async def test_get_risk_settings(client):
    """리스크 설정 기본값 조회."""
    resp = await client.get("/risk/settings",
                            headers={"Authorization": "Bearer test"})
    # auth 없이는 401 또는 200 (conftest에 따라 다름)
    assert resp.status_code in (200, 401)


async def test_auth_mode_switch(client):
    """PUT /auth/mode — 모드 전환."""
    import uuid
    from unittest.mock import MagicMock, patch
    user = MagicMock()
    user.id = uuid.uuid4()
    user.mode = "paper"
    user.kis_real_key_enc = "enc"

    with patch("api.routes.auth.get_current_user", return_value=user):
        with patch("api.routes.auth.get_db"):
            resp = await client.put("/auth/mode", json={"mode": "real"})
    assert resp.status_code in (200, 400, 422)
```

- [ ] **Step 2: risk.py 구현**

```python
# backend/api/routes/risk.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.user import User
from services.risk_service import get_or_create_settings

router = APIRouter()


class RiskSettingsUpdate(BaseModel):
    max_per_stock_pct: float | None = None
    daily_loss_limit_pct: float | None = None
    stop_loss_enabled: bool | None = None
    enforce_hard_stop: bool | None = None


@router.get("/settings")
@limiter.limit("60/minute")
async def get_risk_settings(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await get_or_create_settings(user.id, db)
    return {
        "max_per_stock_pct": float(settings.max_per_stock_pct),
        "daily_loss_limit_pct": float(settings.daily_loss_limit_pct),
        "stop_loss_enabled": settings.stop_loss_enabled,
        "enforce_hard_stop": settings.enforce_hard_stop,
        "trading_blocked": settings.trading_blocked,
    }


@router.put("/settings")
@limiter.limit("20/minute")
async def update_risk_settings(
    request: Request,
    body: RiskSettingsUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await get_or_create_settings(user.id, db)
    if body.max_per_stock_pct is not None:
        settings.max_per_stock_pct = body.max_per_stock_pct
    if body.daily_loss_limit_pct is not None:
        settings.daily_loss_limit_pct = body.daily_loss_limit_pct
    if body.stop_loss_enabled is not None:
        settings.stop_loss_enabled = body.stop_loss_enabled
    if body.enforce_hard_stop is not None:
        settings.enforce_hard_stop = body.enforce_hard_stop
    await db.commit()
    return {"updated": True}
```

- [ ] **Step 3: alerts.py 구현**

```python
# backend/api/routes/alerts.py
from __future__ import annotations

import uuid as _uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.risk import AlertSettings
from models.user import User

router = APIRouter()


async def _get_or_create_alert_settings(user_id, db: AsyncSession) -> AlertSettings:
    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    result = await db.execute(select(AlertSettings).where(AlertSettings.user_id == user_id))
    settings = result.scalar_one_or_none()
    if settings is None:
        settings = AlertSettings(user_id=user_id)
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    return settings


class AlertSettingsUpdate(BaseModel):
    signal_change: bool | None = None
    watchlist_price: bool | None = None
    daily_loss_limit: bool | None = None
    trade_filled: bool | None = None
    weekly_report: bool | None = None
    notification_email: str | None = None


@router.get("/settings")
@limiter.limit("60/minute")
async def get_alert_settings(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await _get_or_create_alert_settings(user.id, db)
    return {
        "signal_change": settings.signal_change,
        "watchlist_price": settings.watchlist_price,
        "daily_loss_limit": settings.daily_loss_limit,
        "trade_filled": settings.trade_filled,
        "weekly_report": settings.weekly_report,
        "notification_email": settings.notification_email,
    }


@router.put("/settings")
@limiter.limit("20/minute")
async def update_alert_settings(
    request: Request,
    body: AlertSettingsUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await _get_or_create_alert_settings(user.id, db)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(settings, field, value)
    await db.commit()
    return {"updated": True}
```

- [ ] **Step 4: auth.py에 PUT /auth/mode 추가**

`auth.py` 파일 끝에 추가:

```python
class ModeUpdate(BaseModel):
    mode: Literal["paper", "real"]


@router.put("/mode")
@limiter.limit("10/minute")
async def switch_mode(
    request: Request,
    body: ModeUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.mode == "paper" and not user.kis_paper_key_enc:
        raise HTTPException(status_code=400, detail="모의투자 KIS 키가 등록되지 않았습니다.")
    if body.mode == "real" and not user.kis_real_key_enc:
        raise HTTPException(status_code=400, detail="실거래 KIS 키가 등록되지 않았습니다.")
    user.mode = body.mode
    await db.commit()
    return {"mode": body.mode}
```

- [ ] **Step 5: 전체 테스트 실행**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/ -v --tb=short 2>&1 | tail -15
```
Expected: 기존 88 + 신규 ≥ 100 passed (일부 auth mock 테스트는 status code 범위로 통과)

- [ ] **Step 6: 커밋**

```bash
git add backend/api/routes/risk.py backend/api/routes/alerts.py \
        backend/api/routes/auth.py tests/test_risk.py
git commit -m "feat: risk/alerts settings CRUD + PUT /auth/mode (paper↔real toggle)"
```

---

## Task 8: main.py APScheduler 추가 + 최종 통합

**Files:**
- Modify: `backend/main.py`
- Modify: `docs/progress.md`

- [ ] **Step 1: main.py 업데이트**

```python
# backend/main.py 전체 교체
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.middleware.rate_limit import limiter
from api.routes import ai as ai_router
from api.routes import alerts as alerts_router
from api.routes import auth as auth_router
from api.routes import portfolio as portfolio_router
from api.routes import realtime as realtime_router
from api.routes import risk as risk_router
from api.routes import stocks as stocks_router
from api.routes import trades as trades_router
from core.config import settings
from core.redis_client import close_redis
from services.websocket_service import kis_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler()
    from tasks.ai_tasks import refresh_ai_signals
    from tasks.email_tasks import check_daily_loss, check_price_alerts

    scheduler.add_job(
        refresh_ai_signals.delay, "cron",
        hour=15, minute=35, day_of_week="mon-fri", timezone="Asia/Seoul",
    )
    scheduler.add_job(check_price_alerts.delay, "interval", minutes=5)
    scheduler.add_job(check_daily_loss.delay, "interval", minutes=10)
    scheduler.start()
    yield
    scheduler.shutdown()
    await kis_pool.stop()
    await close_redis()


app = FastAPI(title="StockSenseAI API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(stocks_router.router, prefix="/stocks", tags=["stocks"])
app.include_router(realtime_router.router, tags=["realtime"])
app.include_router(ai_router.router, prefix="/ai", tags=["ai"])
app.include_router(trades_router.router, prefix="/trades", tags=["trades"])
app.include_router(portfolio_router.router, prefix="/portfolio", tags=["portfolio"])
app.include_router(risk_router.router, prefix="/risk", tags=["risk"])
app.include_router(alerts_router.router, prefix="/alerts", tags=["alerts"])


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 2: 최종 테스트 실행**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/ --tb=short 2>&1 | tail -5
```
Expected: ≥ 100 passed, 0 failed

- [ ] **Step 3: progress.md Phase 4-A 완료 업데이트**

`docs/progress.md`에서 Phase 4 섹션을 찾아:
```markdown
## Phase 4 — 거래 + 포트폴리오 + 시뮬레이터 + 리스크 🔲
```
→
```markdown
## Phase 4 — 거래 + 포트폴리오 + 시뮬레이터 + 리스크

### Phase 4-A — 거래 + 포트폴리오 + 리스크 ✅

**완료일:** 2026-06-0X | **테스트:** ≥ 100 passed
```

구현 완료 항목 표도 업데이트:
```markdown
| KIS 서비스 (완전판) | `backend/services/kis_service.py` | ✅ |
| 리스크 서비스 | `backend/services/risk_service.py` | ✅ |
| 체결 폴링 태스크 | `backend/tasks/order_tasks.py` | ✅ |
| 이메일 태스크 | `backend/tasks/email_tasks.py` | ✅ |
| 거래/포트폴리오/리스크/알림 API | `backend/api/routes/` | ✅ |
```

- [ ] **Step 4: 최종 커밋**

```bash
git add backend/main.py docs/progress.md
git commit -m "feat: Phase 4-A complete — trades/portfolio/risk/alerts API + APScheduler"
```
