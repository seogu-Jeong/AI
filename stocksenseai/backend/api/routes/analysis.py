"""종합 분석 라우트 — 재무 평가 / 추천 종목 / 지수 컨텍스트.

기존 라우트를 수정하지 않고 추가하는 신규 엔드포인트.
데이터 소스는 키 불필요한 네이버 금융(실데이터).
"""
from fastapi import APIRouter, Query, Request

from api.middleware.rate_limit import limiter
from services import comprehensive_service, fundamental_service, investor_service, market_index_service, recommend_service, screener_service, sector_service, stock_data_service, support_resistance_service, anomaly_service, trendline_service

import asyncio

router = APIRouter()

# 백그라운드 태스크 강한 참조 — GC 소멸 방지
_bg_tasks: set = set()


@router.get("/fundamental/{code}")
@limiter.limit("60/minute")
async def get_fundamental(request: Request, code: str):
    """종목 재무 평가 (5.0 만점, 소수 1자리, 2.5 미만이면 위험)."""
    return await fundamental_service.get_fundamental(code)


@router.get("/recommendations")
@limiter.limit("20/minute")
async def get_recommendations(request: Request, limit: int = Query(20, ge=1, le=50)):
    """top100 전체 스캔 → AI BUY + 재무·지수 보조 필터로 추천 종목."""
    return await recommend_service.get_recommendations(limit)


@router.get("/indices")
@limiter.limit("60/minute")
async def get_indices(request: Request):
    """KOSPI/KOSDAQ/KOSPI200 + 시장 추세."""
    return await market_index_service.get_index_context()


@router.get("/comprehensive/{code}")
@limiter.limit("30/minute")
async def get_comprehensive(request: Request, code: str):
    """AI 시그널 + 재무 평가 + 시장 지수 → 종합 판단 (0-10점)."""
    return await comprehensive_service.get_comprehensive(code)


@router.get("/ai-ranking")
@limiter.limit("10/minute")
async def get_ai_ranking(request: Request, limit: int = Query(50, ge=1, le=100)):
    """top100 전체를 AI 점수만으로 정렬한 순위 (재무 필터 없음)."""
    return await recommend_service.get_ai_ranking(limit)


@router.post("/warmup")
@limiter.limit("2/minute")
async def warmup_cache(request: Request):
    """스크리너/랭킹 공유 캐시를 강제 갱신 (백그라운드)."""
    task = asyncio.create_task(stock_data_service.get_all_stock_data(force_refresh=True))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return {"status": "warming up"}


@router.get("/sector")
@limiter.limit("10/minute")
async def get_sector_heatmap(request: Request):
    """KOSPI 업종별 등락률·시가총액·상위 종목 (섹터 히트맵용)."""
    return await sector_service.get_sector_heatmap()


@router.get("/investor/{code}")
@limiter.limit("60/minute")
async def get_investor_trend(request: Request, code: str, days: int = Query(default=10, ge=5, le=30)):
    """종목별 외국인·기관·개인 순매수 동향 (최근 N거래일)."""
    return await investor_service.get_investor_trend(code, days)


@router.get("/support-resistance/{code}")
@limiter.limit("30/minute")
async def get_support_resistance(request: Request, code: str):
    """K-means 클러스터링 기반 지지/저항 레벨."""
    return await support_resistance_service.get_support_resistance(code)


@router.get("/anomaly/{code}")
@limiter.limit("20/minute")
async def get_anomaly(request: Request, code: str):
    """Autoencoder 기반 이상 패턴 감지."""
    return await anomaly_service.get_anomaly(code)


@router.get("/trendline/{code}")
@limiter.limit("30/minute")
async def get_trendline(request: Request, code: str):
    """피크·저점 선형회귀 기반 추세선 자동 감지."""
    return await trendline_service.get_trendline(code)


@router.get("/screener")
@limiter.limit("5/minute")
async def run_screener(
    request: Request,
    signals: str = Query(default="", description="BUY,HOLD,SELL 쉼표 구분 (빈값=전체)"),
    min_score: float | None = Query(default=None, ge=0, le=100),
    grades: str = Query(default="", description="우수,양호,보통,위험 쉼표 구분 (빈값=전체)"),
    max_per: float | None = Query(default=None, ge=0),
    max_pbr: float | None = Query(default=None, ge=0),
    min_roe: float | None = Query(default=None),
    exclude_risk: bool = Query(default=False),
    foreign_net_buy: bool = Query(default=False, description="외국인 5일 순매수 종목만"),
    institution_net_buy: bool = Query(default=False, description="기관 5일 순매수 종목만"),
    sort_by: str = Query(default="signal_score", pattern="^(signal_score|financial_score|per|pbr)$"),
    limit: int = Query(default=50, ge=1, le=100),
):
    """사용자 조건으로 top100 필터링."""
    sig_list = [s.strip() for s in signals.split(",") if s.strip()] or None
    grade_list = [g.strip() for g in grades.split(",") if g.strip()] or None
    return await screener_service.run_screener(
        signals=sig_list,
        min_score=min_score,
        grades=grade_list,
        max_per=max_per,
        max_pbr=max_pbr,
        min_roe=min_roe,
        exclude_risk=exclude_risk,
        foreign_net_buy=foreign_net_buy,
        institution_net_buy=institution_net_buy,
        sort_by=sort_by,
        limit=limit,
    )
