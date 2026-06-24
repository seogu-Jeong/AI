"""스크리너 서비스 — 사용자 정의 조건으로 종목 필터링.

stock_data_service의 공유 캐시를 재활용하므로
첫 스캔 이후에는 필터 적용만 하여 즉시 응답한다.

필터 조건:
  signals              - BUY/HOLD/SELL 포함 여부 (미지정 시 전체)
  min_score            - AI 점수 최솟값 (0-100)
  grades               - 재무 등급 포함 여부 (우수/양호/보통/위험)
  max_per              - PER 최댓값
  max_pbr              - PBR 최댓값
  min_roe              - ROE 최솟값 (%)
  exclude_risk         - 재무 위험 종목 제외
  foreign_net_buy      - 외국인 5일 순매수 종목만
  institution_net_buy  - 기관 5일 순매수 종목만
  sort_by              - signal_score | financial_score | per | pbr
"""
from __future__ import annotations

import asyncio

from services.investor_service import get_investor_trend
from services.stock_data_service import get_all_stock_data


def _passes(
    item: dict,
    signals: list[str] | None,
    min_score: float | None,
    grades: list[str] | None,
    max_per: float | None,
    max_pbr: float | None,
    min_roe: float | None,
    exclude_risk: bool,
) -> bool:
    if signals and item.get("signal") not in signals:
        return False
    if min_score is not None and (item.get("signal_score") or 0) < min_score:
        return False
    if grades and item.get("financial_grade") not in grades:
        return False
    if max_per is not None:
        per = item.get("per")
        if per is None or per > max_per:
            return False
    if max_pbr is not None:
        pbr = item.get("pbr")
        if pbr is None or pbr > max_pbr:
            return False
    if min_roe is not None:
        roe = item.get("roe")
        if roe is None or roe < min_roe:
            return False
    if exclude_risk and item.get("financial_risk"):
        return False
    return True


async def _apply_investor_filter(
    results: list[dict],
    foreign_net_buy: bool,
    institution_net_buy: bool,
) -> list[dict]:
    """외국인/기관 순매수 필터 — 개별 종목 investor_service 호출."""
    if not foreign_net_buy and not institution_net_buy:
        return results

    sem = asyncio.Semaphore(10)

    async def _check(item: dict) -> dict | None:
        async with sem:
            trend = await get_investor_trend(item["code"], days=5)
        summary = trend.get("summary", {})
        if foreign_net_buy and not summary.get("foreign_net_buy"):
            return None
        if institution_net_buy and not summary.get("institution_net_buy"):
            return None
        return {
            **item,
            "foreign_5d":     summary.get("foreign_5d"),
            "institution_5d": summary.get("institution_5d"),
        }

    checked = await asyncio.gather(*[_check(r) for r in results])
    return [r for r in checked if r is not None]


_SORT_KEY = {
    "signal_score":    lambda x: x.get("signal_score") or 0,
    "financial_score": lambda x: x.get("financial_score") or 0,
    "per":             lambda x: x.get("per") or 9999,
    "pbr":             lambda x: x.get("pbr") or 9999,
}


async def run_screener(
    signals: list[str] | None = None,
    min_score: float | None = None,
    grades: list[str] | None = None,
    max_per: float | None = None,
    max_pbr: float | None = None,
    min_roe: float | None = None,
    exclude_risk: bool = False,
    foreign_net_buy: bool = False,
    institution_net_buy: bool = False,
    sort_by: str = "signal_score",
    limit: int = 50,
) -> dict:
    raw = await get_all_stock_data()

    results = [
        r for r in raw
        if _passes(r, signals, min_score, grades, max_per, max_pbr, min_roe, exclude_risk)
    ]

    # 투자자 필터는 추가 API 호출이 필요하므로 다른 필터 이후에 적용
    investor_filtered = foreign_net_buy or institution_net_buy
    if investor_filtered:
        results = await _apply_investor_filter(results, foreign_net_buy, institution_net_buy)

    key_fn = _SORT_KEY.get(sort_by, _SORT_KEY["signal_score"])
    reverse = sort_by not in ("per", "pbr")
    results.sort(key=key_fn, reverse=reverse)

    return {
        "results": results[:limit],
        "matched": len(results),
        "scanned": len(raw),
        "investor_filtered": investor_filtered,
    }
