"""재무 평가 서비스 (키 불필요, 네이버 금융 실데이터).

pykrx의 재무/지수 엔드포인트가 동작하지 않는 환경을 위해 네이버 금융 공개
API를 사용한다. 차트(OHLCV)는 기존대로 pykrx를 쓰고, 여기서는 PER/PBR/EPS/BPS
기반으로 재무 건전성을 5.0점 만점(소수 1자리)으로 평가한다.

평가 기준(한국 시장 기준):
- PER 적정선 10~15(만점), 저평가(<=10) 우수, 고평가(>30)/적자 위험
- PBR <=1 저평가 우수, >5 고평가
- EPS>0(흑자) 가점, 적자 시 위험
- 배당수익률 가점
- 종합 점수 2.5 미만이면 '위험'으로 표시
"""
from __future__ import annotations

import json
import re

import httpx

from core.redis_client import get_redis

_NAVER_INTEGRATION = "https://m.stock.naver.com/api/stock/{code}/integration"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

RISK_THRESHOLD = 2.5  # 이 점수 미만이면 '위험'


def _to_float(value: str | None) -> float | None:
    """'26.03배', '12,372원', '-' 같은 문자열을 float로. 변환 불가 시 None."""
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


async def _fetch_naver(code: str) -> dict:
    url = _NAVER_INTEGRATION.format(code=code)
    async with httpx.AsyncClient(timeout=8.0, headers=_HEADERS) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def _extract(raw: dict) -> dict:
    infos = {i.get("code"): i.get("value") for i in raw.get("totalInfos", [])}
    return {
        "per": _to_float(infos.get("per")),
        "pbr": _to_float(infos.get("pbr")),
        "eps": _to_float(infos.get("eps")),
        "bps": _to_float(infos.get("bps")),
        "dividend_yield": _to_float(infos.get("dividendYield")),
        "roe": _to_float(infos.get("roe")),
    }


def _per_component(per: float | None, eps: float | None) -> tuple[float, str]:
    if eps is not None and eps <= 0:
        return 0.0, "적자(EPS<=0)"
    if per is None or per <= 0:
        return 0.1, "PER 산출 불가"
    if per <= 10:
        return 1.0, f"PER {per:.1f} 저평가"
    if per <= 15:
        return 0.85, f"PER {per:.1f} 적정"
    if per <= 20:
        return 0.6, f"PER {per:.1f} 다소 높음"
    if per <= 30:
        return 0.35, f"PER {per:.1f} 고평가"
    return 0.15, f"PER {per:.1f} 과열"


def _pbr_component(pbr: float | None) -> tuple[float, str]:
    if pbr is None or pbr <= 0:
        return 0.3, "PBR 산출 불가"
    if pbr <= 1:
        return 1.0, f"PBR {pbr:.1f} 자산 대비 저평가"
    if pbr <= 2:
        return 0.75, f"PBR {pbr:.1f} 적정"
    if pbr <= 3:
        return 0.5, f"PBR {pbr:.1f} 다소 높음"
    if pbr <= 5:
        return 0.3, f"PBR {pbr:.1f} 고평가"
    return 0.15, f"PBR {pbr:.1f} 과열"


def _eps_component(eps: float | None) -> float:
    if eps is None:
        return 0.5
    return 1.0 if eps > 0 else 0.0


def _div_component(div: float | None) -> float:
    if not div or div <= 0:
        return 0.1
    if div >= 3:
        return 1.0
    if div >= 1:
        return 0.6
    return 0.3


def _roe_component(roe: float | None) -> tuple[float, str]:
    """ROE(자기자본이익률, %) → 0-1 컴포넌트."""
    if roe is None:
        return 0.5, "ROE 데이터 없음"
    if roe >= 20:
        return 1.0, f"ROE {roe:.1f}% 우수 (자본 효율 매우 높음)"
    if roe >= 15:
        return 0.8, f"ROE {roe:.1f}% 양호"
    if roe >= 10:
        return 0.6, f"ROE {roe:.1f}% 보통"
    if roe >= 5:
        return 0.4, f"ROE {roe:.1f}% 다소 낮음"
    if roe >= 0:
        return 0.2, f"ROE {roe:.1f}% 낮음"
    return 0.0, f"ROE {roe:.1f}% 마이너스 — 자본 잠식 주의"


def score_financials(fund: dict) -> dict:
    """PER/PBR/EPS/배당/ROE → 5.0 만점(소수 1자리) 재무 점수 + 위험 여부."""
    per_c, per_reason = _per_component(fund.get("per"), fund.get("eps"))
    pbr_c, pbr_reason = _pbr_component(fund.get("pbr"))
    eps_c = _eps_component(fund.get("eps"))
    div_c = _div_component(fund.get("dividend_yield"))
    roe_c, roe_reason = _roe_component(fund.get("roe"))

    roe = fund.get("roe")
    if roe is not None:
        # ROE 있으면 5지표 가중: PER 35%, PBR 25%, ROE 20%, EPS 12%, 배당 8%
        weighted = 0.35 * per_c + 0.25 * pbr_c + 0.20 * roe_c + 0.12 * eps_c + 0.08 * div_c
    else:
        # ROE 없으면 4지표 가중 유지: PER 45%, PBR 30%, EPS 15%, 배당 10%
        weighted = 0.45 * per_c + 0.30 * pbr_c + 0.15 * eps_c + 0.10 * div_c
    score = round(weighted * 5.0, 1)

    is_loss = fund.get("eps") is not None and fund["eps"] <= 0
    risk = score < RISK_THRESHOLD or is_loss

    if score >= 4.0:
        grade = "우수"
    elif score >= 3.0:
        grade = "양호"
    elif score >= RISK_THRESHOLD:
        grade = "보통"
    else:
        grade = "위험"
    if is_loss:
        grade = "위험"

    reasons = [per_reason, pbr_reason]
    if roe is not None:
        reasons.append(roe_reason)
    if is_loss:
        reasons.append("적자 기업 — 투자 주의")

    return {
        "score": score,
        "max_score": 5.0,
        "grade": grade,
        "risk": risk,
        "risk_threshold": RISK_THRESHOLD,
        "reasons": reasons,
        "metrics": {
            "per": fund.get("per"),
            "pbr": fund.get("pbr"),
            "eps": fund.get("eps"),
            "bps": fund.get("bps"),
            "dividend_yield": fund.get("dividend_yield"),
            "roe": fund.get("roe"),
        },
    }


async def get_fundamental(code: str) -> dict:
    """종목 재무 평가(캐시 포함). 데이터 없으면 available=False."""
    redis = await get_redis()
    cache_key = f"fundamental:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        raw = await _fetch_naver(code)
        fund = _extract(raw)
    except Exception:
        result = {"code": code, "available": False, "score": None, "grade": None, "risk": None}
        await redis.setex(cache_key, 300, json.dumps(result))
        return result

    evaluated = score_financials(fund)
    result = {"code": code, "available": True, **evaluated}
    # 재무는 자주 안 바뀌므로 1시간 캐시.
    await redis.setex(cache_key, 3600, json.dumps(result))
    return result
