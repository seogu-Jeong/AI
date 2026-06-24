"""공유 종목 데이터 캐시 서비스.

100개 종목의 AI 시그널 + 재무 데이터를 한 번만 스캔하고
Redis에 5분간 캐시한다. 스크리너·AI랭킹·추천 서비스가 모두
이 캐시를 재활용하므로 중복 pykrx/네이버 호출을 없앤다.

첫 호출: 100종목 병렬 스캔 (~10-20초)
이후:    Redis hit → 즉시 반환
"""
from __future__ import annotations

import asyncio
import json

from core.redis_client import get_redis
from services import ai_service, fundamental_service
from services.market_service import _build_ticker_cache, get_ohlcv_cached

_CACHE_KEY = "stock_data_cache:v4"
_CACHE_TTL = 300   # 5분
_CONCURRENCY = 20  # pykrx 병렬 한도

# 급등 판정 임계값
_SURGE_PRICE_PCT   = 5.0   # 전일 대비 가격 변동 %
_SURGE_VOL_RATIO   = 2.0   # 20일 평균 거래량 대비 배율
_SURGE_VOL_WINDOW  = 20    # 거래량 평균 계산 구간 (일)


def _calc_52w(raw_ohlcv: list[dict]) -> dict:
    """1년치 OHLCV로 52주 신고가/신저가 브레이크아웃 감지."""
    default = {
        "w52_high": None,
        "w52_low": None,
        "high_breakout": False,
        "near_high": False,
        "w52_from_high_pct": None,
    }
    if len(raw_ohlcv) < 20:
        return default

    closes = [d["close"] for d in raw_ohlcv]
    current = closes[-1]
    hist = closes[:-1]  # 오늘 제외한 과거 52주 데이터

    w52_high = max(hist)
    w52_low  = min(hist)

    from_high_pct = (current - w52_high) / w52_high * 100

    return {
        "w52_high": round(w52_high),
        "w52_low":  round(w52_low),
        "high_breakout": current >= w52_high,       # 52주 최고가 갱신
        "near_high":     current >= w52_high * 0.97, # 3% 이내 근접
        "w52_from_high_pct": round(from_high_pct, 2),
    }


def _calc_surge(raw_ohlcv: list[dict]) -> dict:
    """OHLCV 리스트로 급등 지표 계산. 데이터 부족 시 기본값 반환."""
    default = {
        "price_change_pct": None,
        "volume_ratio": None,
        "surge_detected": False,
        "surge_reason": None,
    }
    if len(raw_ohlcv) < _SURGE_VOL_WINDOW + 1:
        return default

    closes  = [d["close"]  for d in raw_ohlcv]
    volumes = [d["volume"] for d in raw_ohlcv]

    last_close = closes[-1]
    prev_close = closes[-2]
    if not prev_close:
        return default

    price_change_pct = (last_close - prev_close) / prev_close * 100

    avg_vol = sum(volumes[-(_SURGE_VOL_WINDOW + 1):-1]) / _SURGE_VOL_WINDOW
    volume_ratio = (volumes[-1] / avg_vol) if avg_vol > 0 else None

    ma5 = sum(closes[-5:]) / 5
    ma5_breakout = last_close > ma5

    surge_detected = (
        price_change_pct >= _SURGE_PRICE_PCT
        and volume_ratio is not None and volume_ratio >= _SURGE_VOL_RATIO
        and ma5_breakout
    )

    surge_reason = None
    if surge_detected:
        parts = [
            f"+{price_change_pct:.1f}%",
            f"거래량 {volume_ratio:.1f}배",
            "5일선 돌파",
        ]
        surge_reason = " + ".join(parts)

    return {
        "price_change_pct": round(price_change_pct, 2),
        "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
        "surge_detected": surge_detected,
        "surge_reason": surge_reason,
    }


async def _fetch_one(item: dict, sem: asyncio.Semaphore) -> dict | None:
    code = item["code"]
    async with sem:
        try:
            signal = await ai_service.get_signal(code)
        except Exception:
            return None
        # 1y OHLCV: 52주 신고가용 — semaphore 안에서 pykrx 호출 제어
        try:
            raw_ohlcv_1y = await get_ohlcv_cached(code, "1y", "day")
        except Exception:
            raw_ohlcv_1y = []

    try:
        fund = await fundamental_service.get_fundamental(code)
    except Exception:
        fund = {"available": False}

    # 3m OHLCV: ai_service.get_signal() 이 이미 캐시했으므로 Redis hit
    try:
        raw_ohlcv_3m = await get_ohlcv_cached(code, "3m", "day")
        surge = _calc_surge(raw_ohlcv_3m)
    except Exception:
        surge = {
            "price_change_pct": None,
            "volume_ratio": None,
            "surge_detected": False,
            "surge_reason": None,
        }

    w52 = _calc_52w(raw_ohlcv_1y)

    breakdown = signal.get("signal_breakdown") or {}
    metrics = (fund.get("metrics") or {}) if fund.get("available") else {}

    return {
        "code": code,
        "name": item.get("name", code),
        # AI
        "signal": signal.get("signal"),
        "signal_score": signal.get("signal_score", 0),
        "tech_score": breakdown.get("technical_score"),
        "lstm_score": breakdown.get("lstm_score"),
        "lstm_available": signal.get("lstm_available", False),
        # 재무
        "financial_score": fund.get("score"),
        "financial_grade": fund.get("grade"),
        "financial_risk": fund.get("risk"),
        "per": metrics.get("per"),
        "pbr": metrics.get("pbr"),
        "roe": metrics.get("roe"),
        "eps": metrics.get("eps"),
        "dividend_yield": metrics.get("dividend_yield"),
        # 급등 감지
        **surge,
        # 52주 신고가
        **w52,
    }


async def get_all_stock_data(force_refresh: bool = False) -> list[dict]:
    """전체 종목 데이터 반환 (캐시 우선)."""
    redis = await get_redis()

    if not force_refresh:
        cached = await redis.get(_CACHE_KEY)
        if cached:
            return json.loads(cached)

    tickers = await asyncio.to_thread(_build_ticker_cache, "KOSPI")
    sem = asyncio.Semaphore(_CONCURRENCY)
    raw = await asyncio.gather(*[_fetch_one(t, sem) for t in tickers])
    data = [r for r in raw if r]

    await redis.setex(_CACHE_KEY, _CACHE_TTL, json.dumps(data))
    return data
