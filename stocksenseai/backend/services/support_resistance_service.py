"""지지/저항 레벨 자동 감지 서비스.

K-means 비지도학습으로 6개월 OHLCV 가격대를 클러스터링하여
현재가 기준 지지(support) / 저항(resistance) 레벨을 산출한다.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import numpy as np
from sklearn.cluster import KMeans

from core.redis_client import get_redis
from services.market_service import _last_trading_day

_CACHE_TTL = 3600  # 1시간


def _fetch_ohlcv(code: str, days: int = 180) -> list[dict]:
    from pykrx import stock as pykrx_stock

    end = datetime.strptime(_last_trading_day(), "%Y%m%d")
    start = end - timedelta(days=days)
    df = pykrx_stock.get_market_ohlcv_by_date(
        start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code
    )
    if df is None or df.empty:
        return []

    result = []
    for date_idx, row in df.iterrows():
        try:
            result.append({
                "date":  str(date_idx)[:10].replace("-", ""),
                "open":  float(row.get("시가",  0) or 0),
                "high":  float(row.get("고가",  0) or 0),
                "low":   float(row.get("저가",  0) or 0),
                "close": float(row.get("종가",  0) or 0),
            })
        except Exception:
            continue
    return result


def _calc_levels(ohlcv: list[dict], n_clusters: int = 6) -> dict:
    """K-means 클러스터링으로 지지/저항 레벨 산출."""
    if len(ohlcv) < n_clusters * 3:
        return {"support": [], "resistance": [], "current_price": 0}

    # 고가·저가·종가를 모두 가격 샘플로 사용
    prices = np.array(
        [[r["high"]] for r in ohlcv]
        + [[r["low"]] for r in ohlcv]
        + [[r["close"]] for r in ohlcv],
        dtype=float,
    )

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(prices)
    centers = sorted([float(c[0]) for c in km.cluster_centers_])

    current_price = ohlcv[-1]["close"]
    if current_price == 0:
        current_price = ohlcv[-1]["high"]

    support    = [p for p in centers if p < current_price]
    resistance = [p for p in centers if p >= current_price]

    # 현재가에서 가까운 순서로 정렬
    support    = sorted(support,    reverse=True)   # 가장 가까운 지지가 먼저
    resistance = sorted(resistance)                  # 가장 가까운 저항이 먼저

    # 각 레벨에 강도(strength) 부여: 해당 클러스터에 속한 샘플 수
    labels = km.labels_
    label_counts = np.bincount(labels, minlength=n_clusters)

    def strength(price: float) -> int:
        idx = int(np.argmin(np.abs(km.cluster_centers_ - price)))
        return int(label_counts[idx])

    def fmt(price: float) -> dict:
        return {"price": round(price), "strength": strength(price)}

    return {
        "support":       [fmt(p) for p in support[:3]],
        "resistance":    [fmt(p) for p in resistance[:3]],
        "current_price": round(current_price),
    }


async def get_support_resistance(code: str) -> dict:
    redis = await get_redis()
    cache_key = f"sr_levels:v1:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        ohlcv = await asyncio.to_thread(_fetch_ohlcv, code)
        result = _calc_levels(ohlcv)
    except Exception:
        result = {"support": [], "resistance": [], "current_price": 0}

    result["code"] = code
    has_data = bool(result.get("support") or result.get("resistance"))
    ttl = _CACHE_TTL if has_data else 300
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result
