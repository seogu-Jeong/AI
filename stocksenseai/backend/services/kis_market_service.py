# backend/services/kis_market_service.py
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from fastapi import HTTPException

from core.config import settings
from core.redis_client import get_redis
from services.kis_token_service import get_access_token
from services.market_service import _is_market_open

_KST = ZoneInfo("Asia/Seoul")


def _intraday_unix(date_str: str, time_str: str) -> int:
    """YYYYMMDD + HHMMSS (KST) → UTC Unix timestamp (seconds)."""
    try:
        dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S").replace(tzinfo=_KST)
        return int(dt.timestamp())
    except ValueError:
        return 0


def _intraday_query_time(now_kst: datetime) -> str:
    """KIS 당일분봉 조회 기준 시간을 정규장 범위로 보정한다."""
    hhmmss = now_kst.strftime("%H%M%S")
    if hhmmss < "090000":
        return "090000"
    if hhmmss > "153000":
        return "153000"
    return hhmmss


_REAL_BASE = "https://openapi.koreainvestment.com:9443"
_PAPER_BASE = "https://openapivts.koreainvestment.com:29443"


def _base(mode: str) -> str:
    return _PAPER_BASE if mode == "paper" else _REAL_BASE


def _market_base() -> str:
    """시세 조회(FHKST)는 모의투자 서버 미지원 → 항상 실전 서버 사용."""
    return _REAL_BASE


def _ensure_kis_ok(body: dict) -> None:
    if body.get("rt_cd") != "0":
        raise HTTPException(status_code=502, detail=f"KIS API 오류: {body.get('msg1', '')}")


def _kis_headers(access_token: str, tr_id: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "appkey": settings.SYSTEM_KIS_APP_KEY,
        "appsecret": settings.SYSTEM_KIS_APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",
        "Content-Type": "application/json; charset=utf-8",
    }


async def get_orderbook(code: str) -> dict:
    """10단 호가. system KIS 키 없으면 빈 결과."""
    if not settings.SYSTEM_KIS_APP_KEY:
        return {"code": code, "asks": [], "bids": []}

    redis = await get_redis()
    cache_key = f"orderbook:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    token = await get_access_token(
        settings.SYSTEM_KIS_APP_KEY, settings.SYSTEM_KIS_APP_SECRET, settings.SYSTEM_KIS_MODE
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{_market_base()}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
                headers=_kis_headers(token, "FHKST01010200"),
                params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
            )
            resp.raise_for_status()
            body = resp.json()
            _ensure_kis_ok(body)
            out = body.get("output1", {})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"호가 조회 실패: {exc}") from exc

    asks = [
        {"price": int(out.get(f"askp{i}", 0)), "qty": int(out.get(f"askp_rsqn{i}", 0))}
        for i in range(1, 11)
    ]
    bids = [
        {"price": int(out.get(f"bidp{i}", 0)), "qty": int(out.get(f"bidp_rsqn{i}", 0))}
        for i in range(1, 11)
    ]
    data = {"code": code, "asks": asks, "bids": bids}
    ttl = 5 if _is_market_open() else 60
    await redis.setex(cache_key, ttl, json.dumps(data))
    return data


async def get_recent_trades(code: str) -> list[dict]:
    """최근 체결 20건. system KIS 키 없으면 빈 결과."""
    if not settings.SYSTEM_KIS_APP_KEY:
        return []

    redis = await get_redis()
    cache_key = f"trades:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    token = await get_access_token(
        settings.SYSTEM_KIS_APP_KEY, settings.SYSTEM_KIS_APP_SECRET, settings.SYSTEM_KIS_MODE
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{_market_base()}/uapi/domestic-stock/v1/quotations/inquire-time-itemconclusion",
                headers=_kis_headers(token, "FHKST01010300"),
                params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
            )
            resp.raise_for_status()
            body = resp.json()
            _ensure_kis_ok(body)
            output2 = body.get("output2", [])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"체결 조회 실패: {exc}") from exc

    trades = [
        {
            "time": row.get("stck_cntg_hour", ""),
            "price": int(row.get("stck_prpr", 0)),
            "volume": int(row.get("cntg_vol", 0)),
            "change": row.get("prdy_vrss_sign", "3"),
        }
        for row in output2[:20]
    ]
    ttl = 3 if _is_market_open() else 60
    await redis.setex(cache_key, ttl, json.dumps(trades))
    return trades


async def _get_1min_records_cached(code: str) -> list[dict]:
    """당일 1분봉 raw records를 캐시에서 읽거나 KIS에서 가져옴. 다른 interval이 재사용."""
    redis = await get_redis()
    raw_key = f"intraday_raw:{code}"
    cached = await redis.get(raw_key)
    if cached:
        return json.loads(cached)

    token = await get_access_token(
        settings.SYSTEM_KIS_APP_KEY, settings.SYSTEM_KIS_APP_SECRET, settings.SYSTEM_KIS_MODE
    )

    now_kst = datetime.now(_KST)
    today_str = now_kst.strftime("%Y%m%d")
    all_rows: list[dict] = []
    seen_times: set[str] = set()
    query_time = _intraday_query_time(now_kst)

    import asyncio as _asyncio
    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(15):  # 최대 15회 × 30건 = 450분 (하루 전체 커버)
            resp = await client.get(
                f"{_market_base()}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                headers=_kis_headers(token, "FHKST03010200"),
                params={
                    "FID_ETC_CLS_CODE": "",
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": code,
                    "FID_INPUT_HOUR_1": query_time,
                    "FID_PW_DATA_INCU_YN": "Y",
                },
            )
            if resp.status_code != 200:
                break
            body = resp.json()
            if body.get("rt_cd") != "0":
                break
            rows = body.get("output2", [])
            if not rows:
                break
            # 오늘 날짜 + 미수집 시각만 필터
            new_rows = [
                r for r in rows
                if r.get("stck_bsop_date") == today_str
                and r.get("stck_cntg_hour") not in seen_times
            ]
            if not new_rows:
                break
            all_rows.extend(new_rows)
            for r in new_rows:
                seen_times.add(r.get("stck_cntg_hour", ""))
            earliest = min(r.get("stck_cntg_hour", "235959") for r in new_rows)
            if earliest <= "090100":
                break
            from datetime import timedelta
            earliest_dt = datetime.strptime(earliest, "%H%M%S") - timedelta(minutes=1)
            query_time = earliest_dt.strftime("%H%M%S")
            await _asyncio.sleep(1.0)  # KIS rate limit

    records = []
    for row in all_rows:
        ts = _intraday_unix(row.get("stck_bsop_date", ""), row.get("stck_cntg_hour", "000000"))
        if ts <= 0:
            continue
        records.append({
            "ts": ts,
            "open": int(row.get("stck_oprc", 0)),
            "high": int(row.get("stck_hgpr", 0)),
            "low": int(row.get("stck_lwpr", 0)),
            "close": int(row.get("stck_prpr", 0)),
            "volume": int(row.get("cntg_vol", 0)),
        })
    records.sort(key=lambda r: r["ts"])

    ttl = 60 if _is_market_open() else 3600
    await redis.setex(raw_key, ttl, json.dumps(records))
    return records


async def get_intraday_ohlcv(code: str, interval: str) -> list[dict]:
    """분봉 OHLCV. KIS API는 1분봉만 지원 → 5분/1시간은 1분봉 캐시에서 resample."""
    if not settings.SYSTEM_KIS_APP_KEY:
        return []

    _VALID_INTERVALS = {"1min", "5min", "15min", "1h"}
    if interval not in _VALID_INTERVALS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 interval: {interval}")

    redis = await get_redis()
    cache_key = f"intraday:{code}:{interval}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        records = await _get_1min_records_cached(code)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"분봉 조회 실패: {exc}") from exc

    if not records:
        return []

    import pandas as pd
    resample_map = {"5min": "5min", "15min": "15min", "1h": "h"}
    if interval in resample_map:
        df = pd.DataFrame(records).set_index(
            pd.to_datetime([r["ts"] for r in records], unit="s", utc=True)
        )
        df = df.resample(resample_map[interval]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["close"])
        candles = [
            {"date": int(idx.timestamp()), "open": int(row["open"]), "high": int(row["high"]),
             "low": int(row["low"]), "close": int(row["close"]), "volume": int(row["volume"])}
            for idx, row in df.iterrows()
        ]
    else:
        candles = [
            {"date": r["ts"], "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"], "volume": r["volume"]}
            for r in records
        ]

    ttl = 60 if _is_market_open() else 3600
    await redis.setex(cache_key, ttl, json.dumps(candles))
    return candles
