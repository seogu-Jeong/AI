"""종합 판단 서비스 — AI 시그널 + 재무 평가 + 시장 지수를 합산.

가중치:
  AI 시그널   40% (BUY/HOLD/SELL + score 0-100)
  재무 평가   35% (PER/PBR/ROE/EPS 기반 0-5점)
  시장 지수   25% (KOSPI/KOSDAQ 추세)

결과 0-10점 → 강력매수(≥8) / 매수(≥6.5) / 중립(≥5) / 매도주의(≥3.5) / 매도(<3.5)
"""
from __future__ import annotations

import asyncio
import json

from core.redis_client import get_redis
from services import ai_service, fundamental_service, market_index_service

_CACHE_TTL = 300  # 5분


def _ai_component(signal: dict) -> tuple[float, str]:
    """signal_score(0–100) → 0–1 컴포넌트."""
    sig = signal.get("signal", "HOLD")
    score = float(signal.get("signal_score", 50))
    if sig == "BUY":
        comp = min(1.0, (score - 50) / 50)
        label = "강한 매수 신호" if score >= 80 else "매수 신호"
        return comp, f"{label} (점수: {score:.0f})"
    if sig == "SELL":
        comp = max(0.0, 1.0 - (50 - score) / 50) if score < 50 else 0.1
        return comp, f"매도 신호 (점수: {score:.0f})"
    return 0.3, f"중립 신호 (점수: {score:.0f})"


def _financial_component(fund: dict) -> tuple[float, str]:
    """financial score(0–5) → 0–1 컴포넌트."""
    if not fund.get("available"):
        return 0.5, "재무 데이터 없음 (중립 처리)"
    raw = fund.get("score") or 0.0
    comp = float(raw) / 5.0
    grade = fund.get("grade", "N/A")
    return comp, f"재무 {grade} ({raw:.1f}/5.0)"


def _market_component(ctx: dict) -> tuple[float, str]:
    """market trend → 0–1 컴포넌트."""
    trend = ctx.get("trend", "neutral")
    avg = ctx.get("avg_change", 0.0) or 0.0
    if trend == "up":
        return 0.8, f"시장 상승세 (KOSPI/KOSDAQ 평균 {avg:+.2f}%)"
    if trend == "down":
        return 0.2, f"시장 하락세 (KOSPI/KOSDAQ 평균 {avg:+.2f}%)"
    return 0.5, f"시장 중립 (KOSPI/KOSDAQ 평균 {avg:+.2f}%)"


async def _get_ai_safe(code: str) -> dict:
    try:
        return await ai_service.get_signal(code)
    except Exception:
        return {"signal": "HOLD", "signal_score": 50}


async def get_comprehensive(code: str) -> dict:
    """AI + 재무 + 시장 → 종합 판단 (0–10점, 캐시 5분)."""
    redis = await get_redis()
    cache_key = f"comprehensive:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    ai_data, fund_data, market_ctx = await asyncio.gather(
        _get_ai_safe(code),
        fundamental_service.get_fundamental(code),
        market_index_service.get_index_context(),
    )

    ai_comp, ai_reason = _ai_component(ai_data)
    fund_comp, fund_reason = _financial_component(fund_data)
    market_comp, market_reason = _market_component(market_ctx)

    # 가중 평균: AI 40%, 재무 35%, 시장 25%
    weighted = 0.40 * ai_comp + 0.35 * fund_comp + 0.25 * market_comp
    overall_score = round(weighted * 10, 1)

    if overall_score >= 8.0:
        grade = "강력매수"
    elif overall_score >= 6.5:
        grade = "매수"
    elif overall_score >= 5.0:
        grade = "중립"
    elif overall_score >= 3.5:
        grade = "매도주의"
    else:
        grade = "매도"

    trend_kr = {"up": "상승", "neutral": "중립", "down": "하락"}.get(
        market_ctx.get("trend", "neutral"), "중립"
    )
    recommendation = (
        f"AI {ai_data.get('signal', 'HOLD')} · "
        f"재무 {fund_data.get('grade') or 'N/A'} · "
        f"시장 {trend_kr}"
    )

    result = {
        "code": code,
        "overall_score": overall_score,
        "overall_grade": grade,
        "recommendation": recommendation,
        "components": {
            "ai": {
                "signal": ai_data.get("signal"),
                "signal_score": ai_data.get("signal_score"),
                "component_score": round(ai_comp, 3),
                "reason": ai_reason,
            },
            "financial": {
                "score": fund_data.get("score"),
                "grade": fund_data.get("grade"),
                "component_score": round(fund_comp, 3),
                "reason": fund_reason,
                "risk": fund_data.get("risk"),
            },
            "market": {
                "trend": market_ctx.get("trend"),
                "avg_change": market_ctx.get("avg_change"),
                "component_score": round(market_comp, 3),
                "reason": market_reason,
                "indices": market_ctx.get("indices", []),
            },
        },
        "factors": [ai_reason, fund_reason, market_reason],
    }
    await redis.setex(cache_key, _CACHE_TTL, json.dumps(result))
    return result
