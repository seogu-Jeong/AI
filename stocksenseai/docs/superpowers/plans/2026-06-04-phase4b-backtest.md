# Phase 4-B — 백테스팅 엔진 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OHLCV 기반 기술적 지표 시그널로 과거 매매를 시뮬레이션하는 백테스팅 엔진 + API 구현

**Architecture:** `backtest_service.run_backtest()`가 pykrx OHLCV → `build_features()` → `_calc_tech_score()` → 매매 시뮬레이션 → DB 저장을 동기로 처리. 라우터는 thin wrapper.

**Tech Stack:** FastAPI, SQLAlchemy 2 async, pykrx, pandas-ta (기존), PostgreSQL

---

## 파일 목록

| 파일 | 역할 |
|---|---|
| `backend/services/backtest_service.py` | 백테스팅 엔진 (신규) |
| `backend/api/routes/backtest.py` | POST /backtest/run, GET /backtest/{id} (신규) |
| `tests/test_backtest.py` | 통합 테스트 4개 (신규) |
| `backend/main.py` | backtest 라우터 등록 (수정) |
| `docs/progress.md` | Phase 4-B 완료 업데이트 (수정) |

---

## Task 1: backtest_service.py

**Files:**
- Create: `backend/services/backtest_service.py`

- [ ] **Step 1: `backend/services/backtest_service.py` 생성**

```python
# backend/services/backtest_service.py
from __future__ import annotations

import math
import uuid as _uuid
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from pykrx import stock as pykrx_stock
from sqlalchemy.ext.asyncio import AsyncSession

from ml.features import build_features, FEATURE_COLS
from models.backtest import BacktestResult
from services.ai_service import _calc_tech_score


@dataclass
class BacktestConfig:
    code: str
    start_date: date
    end_date: date
    initial_cash: int = 10_000_000
    entry_signal_score: float = 65.0
    exit_signal_score: float = 35.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    commission_rate: float = 0.00015


def _fetch_ohlcv(code: str, start_date: date, end_date: date) -> pd.DataFrame:
    """pykrx로 OHLCV 다운로드 후 DataFrame 반환."""
    df = pykrx_stock.get_market_ohlcv_by_date(
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
        code,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
    )
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _compute_daily_scores(df: pd.DataFrame) -> list[tuple[str, float, float]]:
    """
    날짜별 (date_str, price, tech_score) 리스트 반환.
    충분한 데이터가 없으면 빈 리스트.
    """
    feat_df = build_features(df)
    if feat_df.empty:
        return []

    results = []
    for idx, row in feat_df.iterrows():
        indicators = {
            "rsi_14": float(row.get("rsi_14", 50)),
            "macd_hist": float(row.get("macd_hist", 0)),
            "close": float(row.get("close", 0)),
            "bb_upper": float(row.get("bb_upper", 0)),
            "bb_lower": float(row.get("bb_lower", 0)),
        }
        score = _calc_tech_score(indicators)
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        results.append((date_str, float(row["close"]), score))
    return results


def _simulate(
    daily: list[tuple[str, float, float]],
    config: BacktestConfig,
) -> tuple[list[dict], list[float]]:
    """
    매매 시뮬레이션.
    반환: (trades_log, equity_curve)
    trades_log: [{date, entry_price, exit_price, pnl, reason}]
    equity_curve: 날짜별 평가금액 리스트
    """
    cash = float(config.initial_cash)
    position = 0
    entry_price = 0.0
    trades_log: list[dict] = []
    equity_curve: list[float] = []
    entry_date = ""

    for date_str, price, score in daily:
        if price == 0:
            equity_curve.append(cash + position * (price or entry_price))
            continue

        # 매수 조건
        if score >= config.entry_signal_score and position == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cost = shares * price * (1 + config.commission_rate)
                cash -= cost
                position = shares
                entry_price = price
                entry_date = date_str

        # 청산 조건
        elif position > 0:
            change = (price - entry_price) / entry_price
            reason = None
            if score <= config.exit_signal_score:
                reason = "signal"
            elif change <= -config.stop_loss_pct:
                reason = "stop_loss"
            elif change >= config.take_profit_pct:
                reason = "take_profit"

            if reason:
                revenue = position * price * (1 - config.commission_rate)
                pnl = revenue - position * entry_price
                trades_log.append({
                    "entry_date": entry_date,
                    "exit_date": date_str,
                    "entry_price": round(entry_price),
                    "exit_price": round(price),
                    "shares": position,
                    "pnl": round(pnl),
                    "reason": reason,
                })
                cash += revenue
                position = 0
                entry_price = 0.0

        equity_curve.append(cash + position * price)

    return trades_log, equity_curve


def _compute_metrics(
    equity_curve: list[float],
    trades_log: list[dict],
    initial_cash: float,
) -> dict:
    """MDD, 샤프비율, 승률, 총 수익률 계산."""
    if not equity_curve:
        return {
            "total_return_pct": 0.0,
            "mdd_pct": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate_pct": 0.0,
        }

    # MDD
    peak = equity_curve[0]
    mdd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > mdd:
            mdd = dd

    # 샤프비율
    if len(equity_curve) > 1:
        daily_returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))
            if equity_curve[i - 1] > 0
        ]
        if daily_returns:
            mean_r = sum(daily_returns) / len(daily_returns)
            std_r = math.sqrt(
                sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns)
            )
            sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # 승률
    wins = sum(1 for t in trades_log if t["pnl"] > 0)
    win_rate = (wins / len(trades_log) * 100) if trades_log else 0.0

    # 총 수익률
    total_return = (equity_curve[-1] - initial_cash) / initial_cash * 100

    return {
        "total_return_pct": round(total_return, 4),
        "mdd_pct": round(mdd * 100, 4),
        "sharpe_ratio": round(sharpe, 4),
        "win_rate_pct": round(win_rate, 2),
    }


async def run_backtest(
    config: BacktestConfig,
    user_id,
    db: AsyncSession,
) -> BacktestResult:
    """
    백테스팅 실행 → BacktestResult DB 저장 → 반환.
    pykrx 호출은 동기라 asyncio.to_thread 사용.
    """
    import asyncio

    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)

    # pykrx 동기 호출
    df = await asyncio.to_thread(_fetch_ohlcv, config.code, config.start_date, config.end_date)
    if df.empty:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"{config.code} 데이터를 가져올 수 없습니다.")

    daily = _compute_daily_scores(df)
    if not daily:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="지표 계산에 충분한 데이터가 없습니다.")

    trades_log, equity_curve = _simulate(daily, config)
    metrics = _compute_metrics(equity_curve, trades_log, config.initial_cash)

    result = BacktestResult(
        user_id=user_id,
        stock_code=config.code,
        strategy_config={
            "entry_signal_score": config.entry_signal_score,
            "exit_signal_score": config.exit_signal_score,
            "stop_loss_pct": config.stop_loss_pct,
            "take_profit_pct": config.take_profit_pct,
            "commission_rate": config.commission_rate,
            "initial_cash": config.initial_cash,
        },
        period_start=config.start_date,
        period_end=config.end_date,
        total_return_pct=metrics["total_return_pct"],
        mdd_pct=metrics["mdd_pct"],
        sharpe_ratio=metrics["sharpe_ratio"],
        win_rate_pct=metrics["win_rate_pct"],
        total_trades=len(trades_log),
        result_detail={
            "trades": trades_log,
            "equity_curve": equity_curve[::5],  # 5일 간격으로 축약
        },
    )
    db.add(result)
    await db.commit()
    await db.refresh(result)
    return result
```

- [ ] **Step 2: 커밋**

```bash
git add backend/services/backtest_service.py
git commit -m "feat: backtest_service — OHLCV-based technical indicator backtesting engine"
```

---

## Task 2: backtest 라우터 (TDD)

**Files:**
- Create: `tests/test_backtest.py`
- Create: `backend/api/routes/backtest.py`
- Modify: `backend/main.py`

- [ ] **Step 1: `tests/test_backtest.py` 작성**

```python
# tests/test_backtest.py
import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.deps import get_current_user


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

    # 테스트 유저 조회
    result = await db_session.execute(select(User).limit(1))
    user_row = result.scalar_one_or_none()
    if user_row is None:
        pytest.skip("No user in test DB")

    app.dependency_overrides[get_current_user] = lambda: user_row

    # BacktestResult 직접 삽입
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/test_backtest.py::test_run_backtest_returns_result -v 2>&1 | tail -5
```

Expected: `404` (라우터 미등록)

- [ ] **Step 3: `backend/api/routes/backtest.py` 생성**

```python
# backend/api/routes/backtest.py
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.backtest import BacktestResult
from models.user import User
from services import backtest_service
from services.backtest_service import BacktestConfig

router = APIRouter()


class BacktestRequest(BaseModel):
    code: str
    start_date: date
    end_date: date
    initial_cash: int = 10_000_000
    entry_signal_score: float = 65.0
    exit_signal_score: float = 35.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    commission_rate: float = 0.00015

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v, info):
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v


def _serialize_result(r: BacktestResult) -> dict:
    return {
        "id": str(r.id),
        "stock_code": r.stock_code,
        "period_start": str(r.period_start),
        "period_end": str(r.period_end),
        "total_return_pct": float(r.total_return_pct or 0),
        "mdd_pct": float(r.mdd_pct or 0),
        "sharpe_ratio": float(r.sharpe_ratio or 0),
        "win_rate_pct": float(r.win_rate_pct or 0),
        "total_trades": r.total_trades or 0,
        "strategy_config": r.strategy_config,
        "result_detail": r.result_detail,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@router.post("/run")
@limiter.limit("5/minute")
async def run_backtest(
    request: Request,
    body: BacktestRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    config = BacktestConfig(
        code=body.code,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_cash=body.initial_cash,
        entry_signal_score=body.entry_signal_score,
        exit_signal_score=body.exit_signal_score,
        stop_loss_pct=body.stop_loss_pct,
        take_profit_pct=body.take_profit_pct,
        commission_rate=body.commission_rate,
    )
    result = await backtest_service.run_backtest(config, user.id, db)
    return _serialize_result(result)


@router.get("/{backtest_id}")
@limiter.limit("30/minute")
async def get_backtest(
    request: Request,
    backtest_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        bid = uuid.UUID(backtest_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 backtest_id")

    result = await db.execute(
        select(BacktestResult).where(
            BacktestResult.id == bid,
            BacktestResult.user_id == user.id,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="백테스팅 결과를 찾을 수 없습니다.")
    return _serialize_result(row)
```

- [ ] **Step 4: `backend/main.py`에 backtest 라우터 등록**

`main.py`에서 alerts 라우터 import 아래에 추가:
```python
from api.routes import backtest as backtest_router
```

`app.include_router(alerts_router.router, ...)` 아래에 추가:
```python
app.include_router(backtest_router.router, prefix="/backtest", tags=["backtest"])
```

- [ ] **Step 5: 테스트 실행 — 통과 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/test_backtest.py -v 2>&1 | tail -15
```

Expected: `4 passed` (test_get_backtest_success는 유저 없으면 skip)

- [ ] **Step 6: 전체 테스트 회귀 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject && \
source .venv/bin/activate && \
python -m pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: `≥ 84 passed, 0 failed`

- [ ] **Step 7: 커밋**

```bash
git add backend/api/routes/backtest.py tests/test_backtest.py backend/main.py
git commit -m "feat: backtest router — POST /backtest/run + GET /backtest/{id} (TDD)"
```

---

## Task 3: progress.md 업데이트

**Files:**
- Modify: `docs/progress.md`

- [ ] **Step 1: Phase 4-B 완료 표기**

`docs/progress.md`에서 아래를 찾아:
```
### Phase 4-B — 백테스팅 🔲
```
→ 교체:
```
### Phase 4-B — 백테스팅 ✅

**완료일:** 2026-06-04

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| 백테스팅 엔진 | `backend/services/backtest_service.py` | ✅ |
| 백테스팅 API | `backend/api/routes/backtest.py` | ✅ |
```

- [ ] **Step 2: 커밋**

```bash
git add docs/progress.md
git commit -m "docs: Phase 4-B complete — backtesting engine and API"
```
