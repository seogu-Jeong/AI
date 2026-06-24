"""각종 지수 서비스 (키 불필요, 네이버 금융 실데이터).

KOSPI / KOSDAQ / KOSPI200 지수의 현재가·등락률을 받아 시장 추세를 산출한다.
AI 종합 판단의 '보조 필터'로 쓰인다 — 시장이 하락 추세면 BUY 추천에 주의 신호를
붙인다.
"""
from __future__ import annotations

import json

import httpx

from core.redis_client import get_redis

_NAVER_INDEX = "https://m.stock.naver.com/api/index/{code}/basic"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# 네이버 지수 코드
_INDICES = [("KOSPI", "KOSPI"), ("KOSDAQ", "KOSDAQ"), ("KOSPI200", "KPI200")]


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


async def _fetch_index(code: str) -> dict | None:
    url = _NAVER_INDEX.format(code=code)
    try:
        async with httpx.AsyncClient(timeout=8.0, headers=_HEADERS) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


async def get_index_context() -> dict:
    """KOSPI/KOSDAQ/KOSPI200 + 시장 추세(상승/중립/하락) 반환. 캐시 5분."""
    redis = await get_redis()
    cache_key = "index_context"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    indices = []
    for name, code in _INDICES:
        data = await _fetch_index(code)
        if not data:
            continue
        indices.append({
            "name": name,
            "value": _to_float(data.get("closePrice")),
            "change_rate": _to_float(data.get("fluctuationsRatio")),
        })

    # 시장 추세: KOSPI·KOSDAQ 등락률 평균으로 판단.
    rates = [i["change_rate"] for i in indices if i["name"] in ("KOSPI", "KOSDAQ") and i["change_rate"] is not None]
    avg = sum(rates) / len(rates) if rates else 0.0
    if avg >= 0.3:
        trend = "up"
    elif avg <= -0.7:
        trend = "down"
    else:
        trend = "neutral"

    result = {"indices": indices, "trend": trend, "avg_change": round(avg, 2)}
    await redis.setex(cache_key, 300, json.dumps(result))
    return result
