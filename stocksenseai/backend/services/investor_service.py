"""투자자별 매매동향 서비스.

pykrx get_market_trading_value_by_date (on="순매수") 를 사용해
외국인·기관·개인 순매수 데이터를 반환한다.
KRX 로그인 없이는 빈 DataFrame이 반환될 수 있으며,
그 경우 available=False로 안전하게 처리한다.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta

from core.redis_client import get_redis
from services.market_service import _is_market_open

_TTL_OPEN   = 300    # 장중 5분
_TTL_CLOSED = 86400  # 장외 24시간


def _fetch_investor_raw(code: str, fromdate: str, todate: str) -> list[dict]:
    """pykrx 동기 호출 — asyncio.to_thread 안에서 실행."""
    from pykrx import stock as pykrx_stock

    try:
        df = pykrx_stock.get_market_trading_value_by_date(
            fromdate, todate, code, on="순매수"
        )
    except Exception:
        return []

    if df is None or df.empty:
        return []

    rows = []
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]

        def _get(col: str) -> int:
            try:
                v = row.get(col, 0)
                return int(v) if v == v else 0  # NaN → 0
            except (KeyError, TypeError, ValueError):
                return 0

        rows.append({
            "date": date_str,
            "foreign_net":     _get("외국인합계"),
            "institution_net": _get("기관합계"),
            "individual_net":  _get("개인"),
        })

    return rows


async def get_investor_trend(code: str, days: int = 10) -> dict:
    """종목별 최근 N일 외국인·기관·개인 순매수 동향 (캐시 포함)."""
    redis = await get_redis()
    cache_key = f"investor_trend:{code}:{days}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    end   = date.today()
    # 거래일 기준 N일 확보를 위해 캘린더 2배 여유
    start = end - timedelta(days=days * 2)
    fromdate = start.strftime("%Y%m%d")
    todate   = end.strftime("%Y%m%d")

    try:
        rows = await asyncio.to_thread(_fetch_investor_raw, code, fromdate, todate)
    except Exception:
        rows = []

    # 최근 N 거래일만 슬라이스
    rows = rows[-days:] if len(rows) > days else rows

    # 5일 누적 합계
    recent = rows[-5:] if len(rows) >= 5 else rows
    foreign_5d     = sum(r["foreign_net"]     for r in recent)
    institution_5d = sum(r["institution_net"] for r in recent)

    result = {
        "code": code,
        "available": len(rows) > 0,
        "trend": rows,
        "summary": {
            "foreign_5d":     foreign_5d,
            "institution_5d": institution_5d,
            "foreign_net_buy":     foreign_5d > 0,
            "institution_net_buy": institution_5d > 0,
        },
    }

    ttl = _TTL_OPEN if _is_market_open() else _TTL_CLOSED
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result
