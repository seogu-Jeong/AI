"""추천 종목 서비스 — stock_data_service 공유 캐시 기반.

동료 피드백 반영: BUY 추천 + SELL 경고 종목 모두 반환.
"""
from __future__ import annotations

import json

from core.redis_client import get_redis
from services import market_index_service
from services.stock_data_service import get_all_stock_data


async def get_ai_ranking(limit: int = 50) -> dict:
    """top100 전체를 AI 점수 순 정렬 (재무 필터 없음, 공유 캐시 활용)."""
    all_data = await get_all_stock_data()
    ranked = sorted(all_data, key=lambda x: x["signal_score"], reverse=True)
    return {
        "ranking": ranked[:limit],
        "scanned": len(all_data),
        "total": len(ranked),
    }


async def get_recommendations(limit: int = 20) -> dict:
    """BUY 추천 + SELL 경고 종목 반환 (공유 캐시 활용, 5분 캐시)."""
    redis = await get_redis()
    cache_key = f"recommendations_v4:{limit}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    all_data = await get_all_stock_data()
    index_ctx = await market_index_service.get_index_context()
    caution = index_ctx.get("trend") == "down"

    buy_picks = []
    sell_picks = []
    surge_alerts = []
    high_breakouts = []

    for item in all_data:
        sig = item.get("signal")
        risk = item.get("financial_risk")

        if sig == "BUY":
            # 재무 위험 종목은 BUY 추천에서 제외
            if risk:
                continue
            buy_picks.append({**item, "market_caution": caution})

        elif sig == "SELL":
            # SELL 신호는 재무 무관하게 경고 표시
            sell_picks.append({**item, "market_caution": caution})

        if item.get("surge_detected"):
            surge_alerts.append({
                "code": item["code"],
                "name": item["name"],
                "surge_reason": item.get("surge_reason"),
                "price_change_pct": item.get("price_change_pct"),
                "volume_ratio": item.get("volume_ratio"),
                "signal": sig,
            })

        if item.get("near_high") or item.get("high_breakout"):
            high_breakouts.append({
                "code": item["code"],
                "name": item["name"],
                "w52_high": item.get("w52_high"),
                "w52_from_high_pct": item.get("w52_from_high_pct"),
                "high_breakout": item.get("high_breakout", False),
                "signal": sig,
            })

    buy_picks.sort(key=lambda x: (x["signal_score"], x.get("financial_score") or 0), reverse=True)
    sell_picks.sort(key=lambda x: x["signal_score"])  # 점수 낮은 순 (가장 강한 SELL 먼저)
    surge_alerts.sort(key=lambda x: x.get("price_change_pct") or 0, reverse=True)
    # 신고가 갱신 우선, 그 다음 고점 근접도 내림차순
    high_breakouts.sort(key=lambda x: x.get("w52_from_high_pct") or -999, reverse=True)

    result = {
        "picks": buy_picks[:limit],
        "sell_warnings": sell_picks[:limit],
        "surge_alerts": surge_alerts[:10],
        "high_breakouts": high_breakouts[:10],
        "scanned": len(all_data),
        "buy_count": len(buy_picks),
        "sell_count": len(sell_picks),
        "surge_count": len(surge_alerts),
        "high_breakout_count": len(high_breakouts),
        "market_trend": index_ctx.get("trend"),
        "market_caution": caution,
    }
    await redis.setex(cache_key, 300, json.dumps(result))
    return result
