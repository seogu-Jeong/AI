import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import HTTPException
from pykrx import stock as pykrx_stock

from core.redis_client import get_redis

_KST = ZoneInfo("Asia/Seoul")


def _is_market_open() -> bool:
    now = datetime.now(_KST)
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 900 <= t <= 1530


def _last_trading_day() -> str:
    """Return the most recent weekday date string (YYYYMMDD)."""
    now = datetime.now(_KST)
    day = now
    # Step back until weekday (Mon=0 … Fri=4)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.strftime("%Y%m%d")


async def get_ohlcv_from_pykrx(code: str, period: str, interval: str) -> list[dict]:
    period_days = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
    # pykrx는 d/m만 지원. week는 일봉 fetch 후 resample.
    pykrx_freq = "m" if interval == "month" else "d"

    end = datetime.now(_KST)
    start = end - timedelta(days=period_days.get(period, 30))
    try:
        df = await asyncio.to_thread(
            pykrx_stock.get_market_ohlcv_by_date,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            code,
            pykrx_freq,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"시세 데이터 조회 실패: {exc}") from exc

    if df is None or df.empty:
        return []

    df.index = pd.to_datetime(df.index)

    if interval == "week":
        df = df.rename(columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})
        df = (
            df.resample("W")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["close"])
        )
        return [
            {
                "date": date.strftime("%Y%m%d"),
                "open": int(row["open"]),
                "high": int(row["high"]),
                "low": int(row["low"]),
                "close": int(row["close"]),
                "volume": int(row["volume"]),
            }
            for date, row in df.iterrows()
        ]

    return [
        {
            "date": date.strftime("%Y%m%d"),
            "open": int(row.get("시가", 0)),
            "high": int(row.get("고가", 0)),
            "low": int(row.get("저가", 0)),
            "close": int(row.get("종가", 0)),
            "volume": int(row.get("거래량", 0)),
        }
        for date, row in df.iterrows()
    ]


async def get_ohlcv_cached(code: str, period: str, interval: str) -> list[dict]:
    redis = await get_redis()
    cache_key = f"ohlcv:{code}:{period}:{interval}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    data = await get_ohlcv_from_pykrx(code, period, interval)
    ttl = 30 if _is_market_open() else 86400
    await redis.setex(cache_key, ttl, json.dumps(data))
    return data


async def get_stock_current_price(code: str) -> dict:
    redis = await get_redis()
    cache_key = f"price:{code}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    day = datetime.now(_KST)
    df = None
    try:
        for _ in range(10):
            date_str = day.strftime("%Y%m%d")
            df = await asyncio.to_thread(
                pykrx_stock.get_market_ohlcv_by_date, date_str, date_str, code
            )
            if df is not None and not df.empty:
                break
            day -= timedelta(days=1)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"현재가 조회 실패: {exc}") from exc

    if df is None or df.empty:
        raise HTTPException(status_code=502, detail="최근 거래일 현재가를 찾을 수 없습니다.")

    row = df.iloc[-1]
    data = {
        "code": code,
        "close": int(row.get("종가", 0)),
        "open": int(row.get("시가", 0)),
        "high": int(row.get("고가", 0)),
        "low": int(row.get("저가", 0)),
        "volume": int(row.get("거래량", 0)),
    }
    ttl = 30 if _is_market_open() else 86400
    await redis.setex(cache_key, ttl, json.dumps(data))
    return data


def _build_ticker_cache(market_str: str) -> list[dict]:
    """top100_codes.txt + stock_names.json 기반 종목 목록 반환."""
    base = Path(__file__).parent.parent / "ml"
    codes_path = base / "top100_codes.txt"
    names_path = base / "stock_names.json"

    codes = codes_path.read_text().strip().splitlines() if codes_path.exists() else []
    names: dict = json.loads(names_path.read_text()) if names_path.exists() else {}

    return [{"code": c.strip(), "name": names.get(c.strip(), c.strip())} for c in codes if c.strip()]


async def _get_ticker_list(market: str) -> list[dict]:
    """Return cached [{code, name}] list for the given market key ('kospi'/'kosdaq')."""
    redis = await get_redis()
    cache_key = f"stocklist:{market.lower()}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    market_str = "KOSPI" if market.lower() == "kospi" else "KOSDAQ"
    tickers = await asyncio.to_thread(_build_ticker_cache, market_str)
    await redis.setex(cache_key, 86400, json.dumps(tickers))
    return tickers


async def get_stock_list(market: str, limit: int, page: int) -> list[dict]:
    tickers = await _get_ticker_list(market)
    start = (page - 1) * limit
    return tickers[start : start + limit]


async def search_stocks(query: str) -> list[dict]:
    all_tickers: list[dict] = []
    for market in ["kospi", "kosdaq"]:
        all_tickers.extend(await _get_ticker_list(market))

    q = query.lower()
    results = []
    for item in all_tickers:
        if q in item["code"].lower() or q in item["name"].lower():
            results.append(item)
        if len(results) >= 20:
            break
    return results


async def get_indices() -> list[dict]:
    redis = await get_redis()
    cache_key = "indices"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    date_str = _last_trading_day()
    result = []
    for name, code in [("KOSPI", "1"), ("KOSDAQ", "2")]:
        try:
            df = await asyncio.to_thread(
                pykrx_stock.get_index_ohlcv_by_date, date_str, date_str, code
            )
            if df is not None and not df.empty:
                row = df.iloc[-1]
                result.append({
                    "name": name,
                    "value": float(row.get("종가", 0)),
                    "change_rate": float(row.get("등락률", 0)),
                })
        except Exception:
            pass

    ttl = 30 if _is_market_open() else 3600
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result
