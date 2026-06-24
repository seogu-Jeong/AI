# backend/api/routes/simulate.py
import json
import re
from datetime import date
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.user import User
from services import simulator_service

router = APIRouter()


class LumpsumRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=10)
    buy_date: date
    sell_date: date
    amount_krw: int = Field(gt=0)

    @field_validator("tickers", mode="before")
    @classmethod
    def validate_tickers(cls, v: list) -> list:
        for t in v:
            if not re.fullmatch(r"\d{6}", str(t)):
                raise ValueError(f"종목코드는 6자리 숫자여야 합니다: {t}")
        return v

    @field_validator("sell_date")
    @classmethod
    def sell_after_buy(cls, v: date, info) -> date:
        buy = info.data.get("buy_date")
        if buy and v <= buy:
            raise ValueError("sell_date는 buy_date 이후여야 합니다.")
        if v > date.today():
            raise ValueError("sell_date는 오늘 이후일 수 없습니다.")
        return v


class RecurringRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=5)
    start_date: date
    end_date: date
    monthly_amount_krw: int = Field(gt=0)

    @field_validator("tickers", mode="before")
    @classmethod
    def validate_tickers(cls, v: list) -> list:
        for t in v:
            if not re.fullmatch(r"\d{6}", str(t)):
                raise ValueError(f"종목코드는 6자리 숫자여야 합니다: {t}")
        return v

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date는 start_date 이후여야 합니다.")
        if v > date.today():
            raise ValueError("end_date는 오늘 이후일 수 없습니다.")
        return v


@router.post("/lumpsum")
@limiter.limit("20/minute")
async def lumpsum(
    request: Request,
    body: LumpsumRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = []
    buy_date_actual = sell_date_actual = None
    for ticker in body.tickers:
        r = await simulator_service.run_lumpsum(
            ticker, body.buy_date, body.sell_date, body.amount_krw, db
        )
        results.append(r)
        if buy_date_actual is None:
            buy_date_actual = r.get("buy_date_actual")
            sell_date_actual = r.get("sell_date_actual")
    return {"buy_date_actual": buy_date_actual, "sell_date_actual": sell_date_actual, "results": results}


@router.post("/recurring")
@limiter.limit("20/minute")
async def recurring(
    request: Request,
    body: RecurringRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = []
    for ticker in body.tickers:
        r = await simulator_service.run_recurring(
            ticker, body.start_date, body.end_date, body.monthly_amount_krw, db
        )
        results.append(r)
    return {"results": results}


@router.get("/data-status")
@limiter.limit("60/minute")
async def data_status(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await simulator_service.get_data_status(db)


@router.get("/download")
@limiter.limit("3/minute")
async def download(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    async def event_generator() -> AsyncGenerator[dict, None]:
        async for progress in simulator_service.download_tickers(db):
            yield {"event": "progress", "data": json.dumps(progress, ensure_ascii=False)}
        yield {
            "event": "complete",
            "data": json.dumps(
                {"message": "다운로드 완료", "total": len(simulator_service.SIMULATOR_TICKERS)},
                ensure_ascii=False,
            ),
        }

    return EventSourceResponse(event_generator())
