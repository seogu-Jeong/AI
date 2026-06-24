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
    {"code": "011170", "name": "롯데케미칼"},
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
    price_cache 조회 후 범위가 부족하면 pykrx로 보충.
    반환: {date_str: close_price} (영업일만 포함)
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

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

    # 캐시 커버리지 확인: 5일 이상 범위 이탈 시 pykrx 재조회
    needs_fetch = True
    if cached:
        cache_start = date.fromisoformat(min(cached.keys()))
        cache_end = date.fromisoformat(max(cached.keys()))
        needs_fetch = (
            (cache_start - start).days > 5
            or (end - cache_end).days > 5
        )

    if needs_fetch:
        try:
            df = await asyncio.to_thread(
                pykrx_stock.get_market_ohlcv_by_date,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                ticker,
            )
        except Exception as exc:
            if cached:
                return cached  # 부분 캐시라도 반환
            raise HTTPException(status_code=502, detail=f"시세 데이터 조회 실패: {exc}") from exc

        if df is not None and not df.empty:
            df = df.rename(columns={"종가": "close"})
            rows_to_insert = [
                {
                    "ticker": ticker,
                    "trade_date": idx.date() if hasattr(idx, "date") else idx,
                    "close_price": float(row["close"]),
                    "market": market,
                }
                for idx, row in df.iterrows()
                if float(row["close"]) > 0
            ]
            if rows_to_insert:
                stmt = pg_insert(PriceCache).values(rows_to_insert).on_conflict_do_nothing()
                await db.execute(stmt)
                await db.commit()

        # 재조회로 최신 캐시 반영
        result2 = await db.execute(
            select(PriceCache)
            .where(
                PriceCache.ticker == ticker,
                PriceCache.market == market,
                PriceCache.trade_date >= start,
                PriceCache.trade_date <= end,
            )
            .order_by(PriceCache.trade_date)
        )
        rows2 = result2.scalars().all()
        cached = {str(r.trade_date): float(r.close_price) for r in rows2}

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
    """prices dict를 받아 일시불 수익률 계산."""
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
        "cash_left_krw": round(amount_krw - buy_value),
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
    """prices dict를 받아 적립식 수익률 계산."""
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

    start_date_actual = trading_days[0]
    end_date_actual = _find_nearest(prices, end_date, "backward")
    final_price = prices[end_date_actual]
    current_value = total_shares * final_price
    avg_buy_price = total_invested / total_shares
    return_pct = (current_value - total_invested) / total_invested * 100

    # 최종 평가 시점이 마지막 매수일과 다르면 종료 포인트 추가
    if chart_data and chart_data[-1]["date"] != end_date_actual:
        chart_data.append({
            "date": end_date_actual,
            "invested": round(total_invested),
            "value": round(current_value),
        })

    return {
        "ticker": ticker,
        "name": name,
        "start_date_actual": start_date_actual,
        "end_date_actual": end_date_actual,
        "total_invested_krw": round(total_invested),
        "total_shares": total_shares,
        "avg_buy_price": round(avg_buy_price),
        "current_value_krw": round(current_value),
        "return_pct": round(return_pct, 4),
        "total_purchases": len([c for c in chart_data if c["date"] != end_date_actual]),
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
        try:
            await get_prices(t["code"], five_years_ago, today, db)
            yield {
                "current": i + 1,
                "total": len(SIMULATOR_TICKERS),
                "ticker": t["code"],
                "name": t["name"],
            }
        except Exception as exc:
            yield {
                "current": i + 1,
                "total": len(SIMULATOR_TICKERS),
                "ticker": t["code"],
                "name": t["name"],
                "status": "error",
                "error": str(exc),
            }
