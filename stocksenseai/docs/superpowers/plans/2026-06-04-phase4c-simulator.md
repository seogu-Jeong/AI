# Phase 4-C — 투자 시뮬레이터 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pykrx 기반 KOSPI 대형주 80개 가격 캐시 + 일시불/적립식 수익률 계산 API 구현

**Architecture:** `simulator_service.py`가 pykrx 조회 → PostgreSQL `price_cache` 저장 → 계산 로직 처리. 라우터는 thin wrapper. SSE 스트리밍은 sse_starlette `EventSourceResponse` 사용.

**Tech Stack:** FastAPI, SQLAlchemy 2 async, pykrx, sse_starlette (기존), PostgreSQL

---

## 파일 목록

| 파일 | 역할 |
|---|---|
| `db/migrations/versions/a7b8c9d0e1f2_add_price_cache.py` | price_cache 테이블 생성 (신규) |
| `backend/models/price_cache.py` | SQLAlchemy 모델 (신규) |
| `backend/services/simulator_service.py` | 가격 로더 + 계산 엔진 (신규) |
| `backend/api/routes/simulate.py` | 4개 엔드포인트 (신규) |
| `tests/test_simulate.py` | 통합 5개 + 단위 3개 (신규) |
| `backend/main.py` | simulate 라우터 등록 (수정) |
| `docs/progress.md` | Phase 4-C 완료 표기 (수정) |

---

## Task 1: DB 마이그레이션 + 모델

**Files:**
- Create: `db/migrations/versions/a7b8c9d0e1f2_add_price_cache.py`
- Create: `backend/models/price_cache.py`

- [ ] **Step 1: 마이그레이션 파일 생성**

```python
# db/migrations/versions/a7b8c9d0e1f2_add_price_cache.py
"""add_price_cache

Revision ID: a7b8c9d0e1f2
Revises: f6a1b2c3d4e5
Create Date: 2026-06-04 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "f6a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "price_cache",
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("close_price", sa.Numeric(12, 2), nullable=False),
        sa.Column("market", sa.String(10), nullable=False, server_default="KR"),
        sa.PrimaryKeyConstraint("ticker", "trade_date", "market"),
    )
    op.create_index("idx_price_cache_ticker_market", "price_cache", ["ticker", "market"])


def downgrade() -> None:
    op.drop_index("idx_price_cache_ticker_market", table_name="price_cache")
    op.drop_table("price_cache")
```

- [ ] **Step 2: SQLAlchemy 모델 생성**

```python
# backend/models/price_cache.py
from sqlalchemy import Column, Date, Index, Numeric, String

from core.database import Base


class PriceCache(Base):
    __tablename__ = "price_cache"

    ticker = Column(String(20), primary_key=True, nullable=False)
    trade_date = Column(Date, primary_key=True, nullable=False)
    close_price = Column(Numeric(12, 2), nullable=False)
    market = Column(String(10), primary_key=True, nullable=False, default="KR")

    __table_args__ = (
        Index("idx_price_cache_ticker_market", "ticker", "market"),
    )
```

- [ ] **Step 3: 커밋**

```bash
git add db/migrations/versions/a7b8c9d0e1f2_add_price_cache.py backend/models/price_cache.py
git commit -m "feat: price_cache 테이블 마이그레이션 + 모델"
```

---

## Task 2: simulator_service.py (TDD)

**Files:**
- Create: `tests/test_simulate.py` (단위 테스트 먼저)
- Create: `backend/services/simulator_service.py`

- [ ] **Step 1: 단위 테스트 작성 (엔진 로직만 — DB/pykrx 불필요)**

```python
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
    assert len(result["chart_data"]) == 3
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/test_simulate.py::test_calc_lumpsum_logic -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'calc_lumpsum'`

- [ ] **Step 3: simulator_service.py 생성**

```python
# backend/services/simulator_service.py
from __future__ import annotations

import asyncio
from datetime import date
from typing import AsyncGenerator

from fastapi import HTTPException
from pykrx import stock as pykrx_stock
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.price_cache import PriceCache

SIMULATOR_TICKERS = [
    {"code": "005930", "name": "삼성전자"},
    {"code": "000660", "name": "SK하이닉스"},
    {"code": "373220", "name": "LG에너지솔루션"},
    {"code": "207940", "name": "삼성바이오로직스"},
    {"code": "005380", "name": "현대차"},
    {"code": "000270", "name": "기아"},
    {"code": "005490", "name": "POSCO홀딩스"},
    {"code": "006400", "name": "삼성SDI"},
    {"code": "051910", "name": "LG화학"},
    {"code": "012330", "name": "현대모비스"},
    {"code": "068270", "name": "셀트리온"},
    {"code": "015760", "name": "한국전력"},
    {"code": "017670", "name": "SK텔레콤"},
    {"code": "105560", "name": "KB금융"},
    {"code": "055550", "name": "신한지주"},
    {"code": "086790", "name": "하나금융지주"},
    {"code": "316140", "name": "우리금융지주"},
    {"code": "032830", "name": "삼성생명"},
    {"code": "066570", "name": "LG전자"},
    {"code": "096770", "name": "SK이노베이션"},
    {"code": "034020", "name": "두산에너빌리티"},
    {"code": "247540", "name": "에코프로비엠"},
    {"code": "086520", "name": "에코프로"},
    {"code": "003670", "name": "포스코퓨처엠"},
    {"code": "034220", "name": "LG디스플레이"},
    {"code": "035720", "name": "카카오"},
    {"code": "035420", "name": "NAVER"},
    {"code": "323410", "name": "카카오뱅크"},
    {"code": "009150", "name": "삼성전기"},
    {"code": "028260", "name": "삼성물산"},
    {"code": "329180", "name": "HD현대중공업"},
    {"code": "009540", "name": "한국조선해양"},
    {"code": "011070", "name": "LG이노텍"},
    {"code": "034730", "name": "SK"},
    {"code": "000720", "name": "현대건설"},
    {"code": "010140", "name": "삼성중공업"},
    {"code": "023530", "name": "롯데쇼핑"},
    {"code": "139480", "name": "이마트"},
    {"code": "004170", "name": "신세계"},
    {"code": "097950", "name": "CJ제일제당"},
    {"code": "128940", "name": "한미약품"},
    {"code": "000100", "name": "유한양행"},
    {"code": "018260", "name": "삼성에스디에스"},
    {"code": "086280", "name": "현대글로비스"},
    {"code": "003490", "name": "대한항공"},
    {"code": "004020", "name": "현대제철"},
    {"code": "010130", "name": "고려아연"},
    {"code": "071050", "name": "한국금융지주"},
    {"code": "006800", "name": "미래에셋증권"},
    {"code": "016360", "name": "삼성증권"},
    {"code": "005940", "name": "NH투자증권"},
    {"code": "039490", "name": "키움증권"},
    {"code": "138040", "name": "메리츠금융지주"},
    {"code": "259960", "name": "크래프톤"},
    {"code": "036570", "name": "엔씨소프트"},
    {"code": "251270", "name": "넷마블"},
    {"code": "068760", "name": "셀트리온헬스케어"},
    {"code": "302440", "name": "SK바이오사이언스"},
    {"code": "018880", "name": "한온시스템"},
    {"code": "010120", "name": "LS ELECTRIC"},
    {"code": "004800", "name": "효성"},
    {"code": "241560", "name": "두산밥캣"},
    {"code": "009830", "name": "한화솔루션"},
    {"code": "011780", "name": "금호석유"},
    {"code": "020560", "name": "아시아나항공"},
    {"code": "006360", "name": "GS건설"},
    {"code": "001040", "name": "CJ"},
    {"code": "004370", "name": "농심"},
    {"code": "271560", "name": "오리온"},
    {"code": "030200", "name": "KT"},
    {"code": "032640", "name": "LG유플러스"},
    {"code": "047810", "name": "한국항공우주"},
    {"code": "003550", "name": "LG"},
    {"code": "000810", "name": "삼성화재"},
    {"code": "090430", "name": "아모레퍼시픽"},
    {"code": "021240", "name": "코웨이"},
    {"code": "008770", "name": "호텔신라"},
    {"code": "011200", "name": "HMM"},
    {"code": "010950", "name": "S-Oil"},
]


def _get_ticker_name(ticker: str) -> str:
    for t in SIMULATOR_TICKERS:
        if t["code"] == ticker:
            return t["name"]
    return ticker


async def get_prices(
    ticker: str,
    start: date,
    end: date,
    db: AsyncSession,
    market: str = "KR",
) -> dict[str, float]:
    """
    price_cache 조회 후 누락 시 pykrx로 다운로드.
    반환: {date_str: close_price} (영업일만 포함, 결측치 없음)
    """
    result = await db.execute(
        select(PriceCache)
        .where(
            PriceCache.ticker == ticker,
            PriceCache.market == market,
            PriceCache.trade_date >= start,
            PriceCache.trade_date <= end,
        )
        .order_by(PriceCache.trade_date)
    )
    rows = result.scalars().all()
    cached = {str(r.trade_date): float(r.close_price) for r in rows}

    if not cached:
        try:
            df = await asyncio.to_thread(
                pykrx_stock.get_market_ohlcv_by_date,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                ticker,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"시세 데이터 조회 실패: {exc}") from exc

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"{ticker} 해당 기간 데이터가 없습니다.")

        df = df.rename(columns={"종가": "close"})
        new_rows = [
            PriceCache(
                ticker=ticker,
                trade_date=idx.date() if hasattr(idx, "date") else idx,
                close_price=float(row["close"]),
                market=market,
            )
            for idx, row in df.iterrows()
            if float(row["close"]) > 0
        ]
        db.add_all(new_rows)
        await db.commit()

        cached = {
            str(idx.date() if hasattr(idx, "date") else idx): float(row["close"])
            for idx, row in df.iterrows()
            if float(row["close"]) > 0
        }

    if not cached:
        raise HTTPException(status_code=404, detail=f"{ticker} 해당 기간 데이터가 없습니다.")

    return cached


def _find_nearest(prices: dict[str, float], target: date, direction: str) -> str | None:
    """
    direction="forward" : target 이후 가장 빠른 영업일
    direction="backward": target 이전 가장 늦은 영업일
    """
    target_str = str(target)
    sorted_dates = sorted(prices.keys())
    if direction == "forward":
        for d in sorted_dates:
            if d >= target_str:
                return d
    else:
        for d in reversed(sorted_dates):
            if d <= target_str:
                return d
    return None


def _get_first_trading_days(
    sorted_dates: list[str],
    start: date,
    end: date,
) -> list[str]:
    """prices 날짜 목록에서 start~end 범위의 매월 첫 번째 영업일 추출."""
    start_str, end_str = str(start), str(end)
    seen: set[str] = set()
    result: list[str] = []
    for d in sorted_dates:
        if d < start_str or d > end_str:
            continue
        month_key = d[:7]
        if month_key not in seen:
            seen.add(month_key)
            result.append(d)
    return result


def calc_lumpsum(
    ticker: str,
    buy_date: date,
    sell_date: date,
    amount_krw: int,
    prices: dict[str, float],
    name: str = "",
) -> dict:
    """prices dict를 받아 일시불 수익률 계산. get_prices() 결과를 넘겨받음."""
    buy_date_actual = _find_nearest(prices, buy_date, "forward")
    sell_date_actual = _find_nearest(prices, sell_date, "backward")

    if not buy_date_actual or not sell_date_actual or buy_date_actual > sell_date_actual:
        raise HTTPException(status_code=422, detail="거래일이 없습니다.")

    buy_price = prices[buy_date_actual]
    sell_price = prices[sell_date_actual]
    shares = int(amount_krw / buy_price)

    if shares == 0:
        raise HTTPException(status_code=422, detail="매수 금액이 주가보다 작아 매수가 불가능합니다.")

    buy_value = shares * buy_price
    sell_value = shares * sell_price
    profit_krw = sell_value - buy_value
    return_pct = profit_krw / buy_value * 100

    chart_data = [
        {"date": d, "return_pct": round((p / buy_price - 1) * 100, 4)}
        for d, p in sorted(prices.items())
        if buy_date_actual <= d <= sell_date_actual
    ]

    return {
        "ticker": ticker,
        "name": name,
        "shares": shares,
        "buy_price": round(buy_price),
        "sell_price": round(sell_price),
        "buy_value_krw": round(buy_value),
        "sell_value_krw": round(sell_value),
        "profit_krw": round(profit_krw),
        "return_pct": round(return_pct, 4),
        "buy_date_actual": buy_date_actual,
        "sell_date_actual": sell_date_actual,
        "chart_data": chart_data,
    }


def calc_recurring(
    ticker: str,
    start_date: date,
    end_date: date,
    monthly_amount_krw: int,
    prices: dict[str, float],
    name: str = "",
) -> dict:
    """prices dict를 받아 적립식 수익률 계산. get_prices() 결과를 넘겨받음."""
    sorted_dates = sorted(prices.keys())
    trading_days = _get_first_trading_days(sorted_dates, start_date, end_date)

    if not trading_days:
        raise HTTPException(status_code=422, detail="해당 기간에 거래일이 없습니다.")

    total_shares = 0
    total_invested = 0.0
    chart_data: list[dict] = []

    for trade_date in trading_days:
        price = prices[trade_date]
        shares_this_month = int(monthly_amount_krw / price)
        if shares_this_month == 0:
            continue
        total_shares += shares_this_month
        total_invested += shares_this_month * price
        chart_data.append({
            "date": trade_date,
            "invested": round(total_invested),
            "value": round(total_shares * price),
        })

    if total_shares == 0:
        raise HTTPException(status_code=422, detail="매수 가능한 주식이 없습니다.")

    end_date_actual = _find_nearest(prices, end_date, "backward")
    final_price = prices[end_date_actual]
    current_value = total_shares * final_price
    avg_buy_price = total_invested / total_shares
    return_pct = (current_value - total_invested) / total_invested * 100

    return {
        "ticker": ticker,
        "name": name,
        "total_invested_krw": round(total_invested),
        "total_shares": total_shares,
        "avg_buy_price": round(avg_buy_price),
        "current_value_krw": round(current_value),
        "return_pct": round(return_pct, 4),
        "total_purchases": len(chart_data),
        "chart_data": chart_data,
    }


async def run_lumpsum(
    ticker: str,
    buy_date: date,
    sell_date: date,
    amount_krw: int,
    db: AsyncSession,
) -> dict:
    name = _get_ticker_name(ticker)
    prices = await get_prices(ticker, buy_date, sell_date, db)
    return calc_lumpsum(ticker, buy_date, sell_date, amount_krw, prices, name)


async def run_recurring(
    ticker: str,
    start_date: date,
    end_date: date,
    monthly_krw: int,
    db: AsyncSession,
) -> dict:
    name = _get_ticker_name(ticker)
    prices = await get_prices(ticker, start_date, end_date, db)
    return calc_recurring(ticker, start_date, end_date, monthly_krw, prices, name)


async def get_data_status(db: AsyncSession) -> dict:
    result = await db.execute(
        select(
            func.count(func.distinct(PriceCache.ticker)),
            func.max(PriceCache.trade_date),
        ).where(PriceCache.market == "KR")
    )
    count, last_updated = result.one()
    ready = (count or 0) >= len(SIMULATOR_TICKERS)
    return {
        "ready": ready,
        "ticker_count": count or 0,
        "last_updated": str(last_updated) if last_updated else None,
    }


async def download_tickers(db: AsyncSession) -> AsyncGenerator[dict, None]:
    """SSE 스트리밍용 async generator. 80개 종목 순서대로 가격 다운로드."""
    today = date.today()
    try:
        five_years_ago = today.replace(year=today.year - 5)
    except ValueError:
        five_years_ago = today.replace(year=today.year - 5, day=28)

    for i, t in enumerate(SIMULATOR_TICKERS):
        await get_prices(t["code"], five_years_ago, today, db)
        yield {
            "current": i + 1,
            "total": len(SIMULATOR_TICKERS),
            "ticker": t["code"],
            "name": t["name"],
        }
```

- [ ] **Step 4: 단위 테스트 실행 — 통과 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/test_simulate.py::test_calc_lumpsum_logic \
  ../tests/test_simulate.py::test_calc_lumpsum_weekend_adjustment \
  ../tests/test_simulate.py::test_calc_recurring_logic -v 2>&1 | tail -10
```

Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
git add backend/services/simulator_service.py tests/test_simulate.py
git commit -m "feat: simulator_service — price cache loader + lumpsum/recurring engine (TDD)"
```

---

## Task 3: API 라우터 (TDD)

**Files:**
- Modify: `tests/test_simulate.py` (통합 테스트 추가)
- Create: `backend/api/routes/simulate.py`
- Modify: `backend/main.py`

- [ ] **Step 1: 통합 테스트 추가 (`tests/test_simulate.py` 하단에 append)**

```python
# tests/test_simulate.py 하단에 추가

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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/test_simulate.py::test_lumpsum_returns_result -v 2>&1 | tail -5
```

Expected: `404` 또는 `ImportError` (라우터 미등록)

- [ ] **Step 3: `backend/api/routes/simulate.py` 생성**

```python
# backend/api/routes/simulate.py
import json
import re
from datetime import date
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.user import User
from services import simulator_service

router = APIRouter()


class LumpsumRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=10)
    buy_date: date
    sell_date: date
    amount_krw: int = Field(gt=0)

    @field_validator("tickers", mode="before")
    @classmethod
    def validate_tickers(cls, v: list) -> list:
        for t in v:
            if not re.fullmatch(r"\d{6}", str(t)):
                raise ValueError(f"종목코드는 6자리 숫자여야 합니다: {t}")
        return v

    @field_validator("sell_date")
    @classmethod
    def sell_after_buy(cls, v: date, info) -> date:
        buy = info.data.get("buy_date")
        if buy and v <= buy:
            raise ValueError("sell_date는 buy_date 이후여야 합니다.")
        return v


class RecurringRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=5)
    start_date: date
    end_date: date
    monthly_amount_krw: int = Field(gt=0)

    @field_validator("tickers", mode="before")
    @classmethod
    def validate_tickers(cls, v: list) -> list:
        for t in v:
            if not re.fullmatch(r"\d{6}", str(t)):
                raise ValueError(f"종목코드는 6자리 숫자여야 합니다: {t}")
        return v

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date는 start_date 이후여야 합니다.")
        return v


@router.post("/lumpsum")
@limiter.limit("20/minute")
async def lumpsum(
    request: Request,
    body: LumpsumRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = []
    buy_date_actual = sell_date_actual = None
    for ticker in body.tickers:
        r = await simulator_service.run_lumpsum(
            ticker, body.buy_date, body.sell_date, body.amount_krw, db
        )
        results.append(r)
        if buy_date_actual is None:
            buy_date_actual = r.get("buy_date_actual")
            sell_date_actual = r.get("sell_date_actual")
    return {"buy_date_actual": buy_date_actual, "sell_date_actual": sell_date_actual, "results": results}


@router.post("/recurring")
@limiter.limit("20/minute")
async def recurring(
    request: Request,
    body: RecurringRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = []
    for ticker in body.tickers:
        r = await simulator_service.run_recurring(
            ticker, body.start_date, body.end_date, body.monthly_amount_krw, db
        )
        results.append(r)
    return {"results": results}


@router.get("/data-status")
@limiter.limit("60/minute")
async def data_status(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await simulator_service.get_data_status(db)


@router.get("/download")
@limiter.limit("3/minute")
async def download(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    async def event_generator() -> AsyncGenerator[dict, None]:
        async for progress in simulator_service.download_tickers(db):
            yield {"event": "progress", "data": json.dumps(progress, ensure_ascii=False)}
        yield {
            "event": "complete",
            "data": json.dumps(
                {"message": "다운로드 완료", "total": len(simulator_service.SIMULATOR_TICKERS)},
                ensure_ascii=False,
            ),
        }

    return EventSourceResponse(event_generator())
```

- [ ] **Step 4: `backend/main.py`에 라우터 등록**

`backend/main.py`에서 `from api.routes import backtest as backtest_router` 아래에 추가:
```python
from api.routes import simulate as simulate_router
```

`app.include_router(backtest_router.router, ...)` 아래에 추가:
```python
app.include_router(simulate_router.router, prefix="/simulate", tags=["simulate"])
```

- [ ] **Step 5: 통합 테스트 실행 — 통과 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/test_simulate.py -v 2>&1 | tail -20
```

Expected: `8 passed` (단위 3 + 통합 5)

- [ ] **Step 6: 전체 회귀 테스트**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/ --tb=short -q 2>&1 | tail -5
```

Expected: `≥ 86 passed, 0 new failures` (기존 bcrypt 실패 13개 제외)

- [ ] **Step 7: 커밋**

```bash
git add backend/api/routes/simulate.py backend/main.py tests/test_simulate.py
git commit -m "feat: simulate router — POST /lumpsum, /recurring, GET /data-status, /download (TDD)"
```

---

## Task 4: progress.md 업데이트

**Files:**
- Modify: `docs/progress.md`

- [ ] **Step 1: Phase 4-C 완료 표기**

`docs/progress.md`에서:
```
### Phase 4-C — 투자 시뮬레이터 🔲
```
→ 교체:
```
### Phase 4-C — 투자 시뮬레이터 ✅

**완료일:** 2026-06-04

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| price_cache 마이그레이션 | `db/migrations/versions/a7b8c9d0e1f2_add_price_cache.py` | ✅ |
| PriceCache 모델 | `backend/models/price_cache.py` | ✅ |
| 시뮬레이터 엔진 | `backend/services/simulator_service.py` | ✅ |
| 시뮬레이터 API | `backend/api/routes/simulate.py` | ✅ |
```

- [ ] **Step 2: 커밋**

```bash
git add docs/progress.md
git commit -m "docs: Phase 4-C complete — investment simulator API"
```
