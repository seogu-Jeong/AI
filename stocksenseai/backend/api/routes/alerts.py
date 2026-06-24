# backend/api/routes/alerts.py
import uuid as _uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.risk import AlertSettings
from models.user import User

router = APIRouter()


async def _get_or_create_alert_settings(user_id, db: AsyncSession) -> AlertSettings:
    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    result = await db.execute(select(AlertSettings).where(AlertSettings.user_id == user_id))
    settings = result.scalar_one_or_none()
    if settings is None:
        settings = AlertSettings(user_id=user_id)
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    return settings


class AlertSettingsUpdate(BaseModel):
    signal_change: bool | None = None
    watchlist_price: bool | None = None
    daily_loss_limit: bool | None = None
    trade_filled: bool | None = None
    weekly_report: bool | None = None
    notification_email: EmailStr | None = None


@router.get("/settings")
@limiter.limit("60/minute")
async def get_alert_settings(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await _get_or_create_alert_settings(user.id, db)
    return {
        "signal_change": settings.signal_change,
        "watchlist_price": settings.watchlist_price,
        "daily_loss_limit": settings.daily_loss_limit,
        "trade_filled": settings.trade_filled,
        "weekly_report": settings.weekly_report,
        "notification_email": settings.notification_email,
    }


@router.put("/settings")
@limiter.limit("20/minute")
async def update_alert_settings(
    request: Request,
    body: AlertSettingsUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await _get_or_create_alert_settings(user.id, db)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(settings, field, value)
    await db.commit()
    return {"updated": True}
