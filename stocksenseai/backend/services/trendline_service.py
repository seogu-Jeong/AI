"""추세선 자동 감지 서비스.

scipy 피크 탐지 → 고점/저점 선형 회귀로
상승 채널 / 하락 채널 / 수평 지지·저항선을 산출한다.
결과는 LightweightCharts LineSeries 에 바로 쓸 수 있는 포인트 배열로 반환.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import numpy as np
from scipy.signal import argrelextrema

from core.redis_client import get_redis
from services.market_service import _last_trading_day

_CACHE_TTL = 3600
_ORDER = 5   # 극값 탐지 윈도우 (N일 내 최고/최저)


def _fetch_ohlcv(code: str, days: int = 180) -> list[dict]:
    from pykrx import stock as pykrx_stock

    end = datetime.strptime(_last_trading_day(), "%Y%m%d")
    start = end - timedelta(days=days)
    df = pykrx_stock.get_market_ohlcv_by_date(
        start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code
    )
    if df is None or df.empty:
        return []

    rows = []
    for date_idx, row in df.iterrows():
        try:
            rows.append({
                "date":  str(date_idx)[:10],
                "high":  float(row.get("고가", 0) or 0),
                "low":   float(row.get("저가", 0) or 0),
                "close": float(row.get("종가", 0) or 0),
            })
        except Exception:
            continue
    return rows


def _linear_reg(x: np.ndarray, y: np.ndarray):
    """단순 선형 회귀 → (기울기, 절편, R²)"""
    if len(x) < 2:
        return 0.0, float(y.mean()) if len(y) else 0.0, 0.0
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def _calc_trendlines(ohlcv: list[dict]) -> dict:
    if len(ohlcv) < _ORDER * 4:
        return {"lines": [], "available": False}

    highs  = np.array([r["high"]  for r in ohlcv], dtype=float)
    lows   = np.array([r["low"]   for r in ohlcv], dtype=float)
    dates  = [r["date"] for r in ohlcv]
    xs     = np.arange(len(ohlcv), dtype=float)

    # 극대(고점) / 극소(저점) 인덱스
    peak_idx   = argrelextrema(highs, np.greater, order=_ORDER)[0]
    trough_idx = argrelextrema(lows,  np.less,   order=_ORDER)[0]

    lines = []

    def _make_line(idx_arr: np.ndarray, prices: np.ndarray,
                   color: str, label: str):
        if len(idx_arr) < 2:
            return
        x_pts = xs[idx_arr]
        y_pts = prices[idx_arr]
        slope, intercept, r2 = _linear_reg(x_pts, y_pts)
        if r2 < 0.5:   # 적합도 낮으면 제외
            return
        # 전체 기간 시작~끝 두 포인트만 반환
        x0, x1 = float(xs[0]), float(xs[-1])
        lines.append({
            "label":  label,
            "color":  color,
            "slope":  round(slope, 4),
            "r2":     round(r2, 3),
            "points": [
                {"date": dates[0],        "price": round(slope * x0 + intercept)},
                {"date": dates[-1],       "price": round(slope * x1 + intercept)},
            ],
            # 최근 극값 표시용
            "pivots": [
                {"date": dates[int(i)], "price": round(float(prices[int(i)]))}
                for i in idx_arr[-5:]
            ],
        })

    # 상단 저항 추세선 (고점 연결)
    _make_line(peak_idx,   highs, "#ef4444", "저항 추세선")
    # 하단 지지 추세선 (저점 연결)
    _make_line(trough_idx, lows,  "#22c55e", "지지 추세선")

    return {"lines": lines, "available": len(lines) > 0}


async def get_trendline(code: str) -> dict:
    redis = await get_redis()
    cache_key = f"trendline:v1:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        ohlcv = await asyncio.to_thread(_fetch_ohlcv, code)
        result = await asyncio.to_thread(_calc_trendlines, ohlcv)
    except Exception:
        result = {"lines": [], "available": False}

    result["code"] = code
    ttl = _CACHE_TTL if result.get("available") else 300
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result
