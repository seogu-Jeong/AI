# backend/api/routes/account.py
from fastapi import APIRouter, Depends, Request

from api.deps import get_current_user
from api.middleware.rate_limit import limiter
from core.config import settings
from models.user import User
from services import kis_account_service

router = APIRouter()


@router.get("/config")
@limiter.limit("60/minute")
async def get_account_config(
    request: Request,
    user: User = Depends(get_current_user),
):
    return {
        "mode": settings.SYSTEM_KIS_MODE,
        "account_no": kis_account_service.mask_account_no(settings.SYSTEM_KIS_ACCOUNT_NO),
    }


@router.get("/balance")
@limiter.limit("30/minute")
async def get_account_balance(
    request: Request,
    user: User = Depends(get_current_user),
):
    return await kis_account_service.get_account_balance(settings.SYSTEM_KIS_MODE)
