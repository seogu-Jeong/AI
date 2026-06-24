# backend/api/routes/risk.py
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.user import User
from services.risk_service import get_or_create_settings

router = APIRouter()


class RiskSettingsUpdate(BaseModel):
    max_per_stock_pct: float | None = Field(default=None, ge=0, le=100)
    daily_loss_limit_pct: float | None = Field(default=None, ge=0, le=100)
    stop_loss_enabled: bool | None = None
    enforce_hard_stop: bool | None = None


@router.get("/settings")
@limiter.limit("60/minute")
async def get_risk_settings(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await get_or_create_settings(user.id, db)
    return {
        "max_per_stock_pct": float(settings.max_per_stock_pct),
        "daily_loss_limit_pct": float(settings.daily_loss_limit_pct),
        "stop_loss_enabled": settings.stop_loss_enabled,
        "enforce_hard_stop": settings.enforce_hard_stop,
        "trading_blocked": settings.trading_blocked,
    }


@router.put("/settings")
@limiter.limit("20/minute")
async def update_risk_settings(
    request: Request,
    body: RiskSettingsUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await get_or_create_settings(user.id, db)
    if body.max_per_stock_pct is not None:
        settings.max_per_stock_pct = body.max_per_stock_pct
    if body.daily_loss_limit_pct is not None:
        settings.daily_loss_limit_pct = body.daily_loss_limit_pct
    if body.stop_loss_enabled is not None:
        settings.stop_loss_enabled = body.stop_loss_enabled
    if body.enforce_hard_stop is not None:
        settings.enforce_hard_stop = body.enforce_hard_stop
    await db.commit()
    return {"updated": True}


@router.post("/unblock")
@limiter.limit("5/minute")
async def unblock_trading(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await get_or_create_settings(user.id, db)
    settings.trading_blocked = False
    settings.blocked_at = None
    await db.commit()
    return {"unblocked": True}
