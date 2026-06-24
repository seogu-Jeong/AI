import asyncio

from fastapi import APIRouter, Query, Request

from api.middleware.rate_limit import limiter
from core.config import settings
from services import kis_market_service, market_service

router = APIRouter()

_INTRADAY_INTERVALS = {"1min", "5min", "15min", "1h"}
_intraday_tasks: dict[str, asyncio.Task] = {}


@router.get("")
@limiter.limit("100/minute")
async def list_stocks(
    request: Request,
    market: str = Query("kospi", pattern="^(kospi|kosdaq)$"),
    limit: int = Query(50, ge=1, le=200),
    page: int = Query(1, ge=1),
):
    return await market_service.get_stock_list(market, limit, page)


@router.get("/search")
@limiter.limit("100/minute")
async def search_stocks(request: Request, q: str = Query(..., min_length=1)):
    return await market_service.search_stocks(q)


@router.get("/indices")
@limiter.limit("100/minute")
async def get_indices(request: Request):
    return await market_service.get_indices()


@router.get("/{code}/chart")
@limiter.limit("100/minute")
async def get_stock_chart(
    request: Request,
    code: str,
    period: str = Query("1y", pattern="^(1d|1w|1m|3m|1y|2y|3y|5y)$"),
    interval: str = Query("day", pattern="^(1min|5min|15min|1h|day|week|month)$"),
):
    if interval in _INTRADAY_INTERVALS:
        if not settings.SYSTEM_KIS_APP_KEY:
            data = await market_service.get_ohlcv_cached(code, "1m", "day")
            return {
                "code": code,
                "period": "1m",
                "interval": interval,
                "requested_interval": interval,
                "actual_interval": "day",
                "status": "fallback_only",
                "data": data,
            }

        task_key = f"{code}:{interval}"
        task = _intraday_tasks.get(task_key)
        if task is None:
            task = asyncio.create_task(kis_market_service.get_intraday_ohlcv(code, interval))
            _intraday_tasks[task_key] = task

        data = []
        if task.done():
            _intraday_tasks.pop(task_key, None)
            try:
                data = task.result()
            except Exception:
                data = []

        if data:
            return {
                "code": code,
                "period": "1d",
                "interval": interval,
                "requested_interval": interval,
                "actual_interval": interval,
                "status": "ready",
                "data": data,
            }

        data = await market_service.get_ohlcv_cached(code, "1m", "day")
        return {
            "code": code,
            "period": "1m",
            "interval": interval,
            "requested_interval": interval,
            "actual_interval": "day",
            "status": "loading_intraday",
            "data": data,
        }
    else:
        data = await market_service.get_ohlcv_cached(code, period, interval)
        effective_period = period
    return {
        "code": code,
        "period": effective_period,
        "interval": interval,
        "requested_interval": interval,
        "actual_interval": interval,
        "status": "ready",
        "data": data,
    }


@router.get("/{code}/orderbook")
@limiter.limit("100/minute")
async def get_orderbook(request: Request, code: str):
    return await kis_market_service.get_orderbook(code)


@router.get("/{code}/trades")
@limiter.limit("100/minute")
async def get_recent_trades(request: Request, code: str):
    return await kis_market_service.get_recent_trades(code)


@router.get("/{code}")
@limiter.limit("100/minute")
async def get_stock_detail(request: Request, code: str):
    return await market_service.get_stock_current_price(code)
