from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from api.middleware.rate_limit import limiter
from core.config import settings
from core.redis_client import get_redis
from models.ai_signal import AISignalHistory
from services import ai_service, pattern_service
from services.market_service import get_ohlcv_cached

router = APIRouter()


async def _get_ohlcv_df(code: str, period: str = "3m") -> pd.DataFrame:
    raw = await get_ohlcv_cached(code, period, "day")
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").sort_index()


@router.get("/top-picks")
@limiter.limit("20/minute")
async def get_top_picks(request: Request):
    return await ai_service.get_top_picks()


@router.get("/signals/history/{code}")
@limiter.limit("20/minute")
async def get_signals_history(
    request: Request, code: str, db: AsyncSession = Depends(get_db)
):
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    result = await db.execute(
        select(AISignalHistory)
        .where(AISignalHistory.stock_code == code)
        .where(AISignalHistory.recorded_at >= cutoff)
        .where(AISignalHistory.signal.in_(["BUY", "HOLD", "SELL"]))
        .order_by(desc(AISignalHistory.recorded_at))
        .limit(100)
    )
    rows = result.scalars().all()
    return {
        "code": code,
        "history": [
            {
                "signal": r.signal,
                "signal_score": float(r.signal_score or 0),
                "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
            }
            for r in rows
        ],
    }


@router.get("/{code}/signal")
@limiter.limit("20/minute")
async def get_signal(request: Request, code: str, db: AsyncSession = Depends(get_db)):
    return await ai_service.get_signal(code, db)


@router.get("/{code}/predict")
@limiter.limit("20/minute")
async def get_predict(
    request: Request,
    code: str,
    db: AsyncSession = Depends(get_db),
):
    return await ai_service.get_prediction(code, db)


# ── 로컬 ML 스크립트 → 배포 서버 예측 업로드 ──────────────────────

class _PredictionItem(BaseModel):
    code: str
    current_price: float = Field(gt=0)
    bullish: list[float] = Field(min_length=5, max_length=5)
    base: list[float] = Field(min_length=5, max_length=5)
    bearish: list[float] = Field(min_length=5, max_length=5)
    confidence: float = Field(ge=0, le=100)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if not re.fullmatch(r"\d{6}", value):
            raise ValueError("code must be exactly 6 digits")
        return value

    @field_validator("bullish", "base", "bearish")
    @classmethod
    def validate_prices(cls, values: list[float]) -> list[float]:
        if any(value <= 0 for value in values):
            raise ValueError("predicted prices must be positive")
        return values


class PredictionUploadRequest(BaseModel):
    predictions: list[_PredictionItem] = Field(min_length=1, max_length=100)


@router.post("/predictions/upload", status_code=200)
async def upload_predictions(
    request: Request,
    body: PredictionUploadRequest,
    db: AsyncSession = Depends(get_db),
):
    """로컬 LSTM 예측 결과를 DB에 저장. X-Upload-Key 헤더 인증."""
    key = request.headers.get("X-Upload-Key", "")
    expected_key = settings.ML_UPLOAD_KEY
    if not expected_key or not secrets.compare_digest(key, expected_key):
        raise HTTPException(status_code=403, detail="Invalid upload key")

    now = datetime.now(timezone.utc)
    rows = [
        AISignalHistory(
            stock_code=p.code,
            signal="PREDICTION",
            confidence=p.confidence,
            predicted_prices={
                "current_price": p.current_price,
                "bullish": p.bullish,
                "base": p.base,
                "bearish": p.bearish,
                "generated_at": now.isoformat(),
            },
            recorded_at=now,
        )
        for p in body.predictions
    ]
    db.add_all(rows)
    await db.commit()

    redis = await get_redis()
    for prediction in body.predictions:
        await redis.delete(f"ai_predict:{prediction.code}")

    return {"uploaded": len(rows), "generated_at": now.isoformat()}


@router.get("/{code}/indicators")
@limiter.limit("20/minute")
async def get_indicators_endpoint(request: Request, code: str):
    result = await ai_service.get_indicators(code)
    if not result:
        raise HTTPException(status_code=404, detail="지표 계산 불가 (데이터 부족)")
    return result


@router.get("/{code}/patterns")
@limiter.limit("20/minute")
async def get_patterns(request: Request, code: str):
    df = await _get_ohlcv_df(code, "3m")
    patterns = pattern_service.detect_patterns(df)
    return {"code": code, "patterns": patterns}


@router.get("/{code}/similar")
@limiter.limit("20/minute")
async def get_similar(request: Request, code: str):
    return await ai_service.get_similar(code)


@router.get("/{code}/multiframe")
@limiter.limit("20/minute")
async def get_multiframe(request: Request, code: str):
    return await ai_service.get_multiframe(code)
