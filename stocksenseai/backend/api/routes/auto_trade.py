from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.user import User
from services import auto_trade_service

router = APIRouter()


class AutoTradeConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = None
    mode: Optional[str] = None
    total_budget: Optional[int] = Field(default=None, gt=0)
    max_positions: Optional[int] = Field(default=None, ge=1, le=20)
    signal_threshold: Optional[int] = Field(default=None, ge=0, le=100)
    stop_loss_pct: Optional[float] = Field(default=None, ge=1.0, le=30.0)
    take_profit_pct: Optional[float] = Field(default=None, ge=1.0, le=100.0)

    @field_validator("mode")
    @classmethod
    def valid_mode(cls, v: str) -> str:
        if v not in ("paper", "real"):
            raise ValueError("mode는 'paper' 또는 'real'이어야 합니다.")
        return v


class RunRequest(BaseModel):
    extra_codes: Optional[List[str]] = None


class ScanRequest(BaseModel):
    codes: Optional[List[str]] = None


def _cfg_to_dict(cfg: Any) -> dict:
    return {
        "id": str(cfg.id),
        "enabled": cfg.enabled,
        "mode": cfg.mode,
        "total_budget": cfg.total_budget,
        "max_positions": cfg.max_positions,
        "signal_threshold": cfg.signal_threshold,
        "stop_loss_pct": cfg.stop_loss_pct,
        "take_profit_pct": cfg.take_profit_pct,
        "created_at": cfg.created_at.isoformat() if cfg.created_at else None,
        "updated_at": cfg.updated_at.isoformat() if cfg.updated_at else None,
    }


@router.get("/config")
@limiter.limit("30/minute")
async def get_config(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    cfg = await auto_trade_service.get_config(user.id, db)
    return _cfg_to_dict(cfg)


@router.put("/config")
@limiter.limit("20/minute")
async def update_config(
    request: Request,
    body: AutoTradeConfigUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    data = body.model_dump(exclude_none=True)
    if data.get("mode") == "real":
        raise HTTPException(status_code=400, detail="자동매매는 모의투자 전용입니다. 실거래 모드는 지원하지 않습니다.")
    cfg = await auto_trade_service.update_config(user.id, data, db)
    return _cfg_to_dict(cfg)


@router.get("/logs")
@limiter.limit("30/minute")
async def get_logs(
    request: Request,
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    effective_limit = max(1, min(limit, 200))
    logs = await auto_trade_service.get_logs(user.id, db, effective_limit)
    return {"logs": logs, "count": len(logs)}


@router.post("/run")
@limiter.limit("30/minute")
async def run_cycle(
    request: Request,
    body: RunRequest = RunRequest(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await auto_trade_service.run_cycle(user.id, db, extra_codes=body.extra_codes)
    return result


@router.post("/scan")
@limiter.limit("20/minute")
async def scan_stocks(
    request: Request,
    body: ScanRequest = ScanRequest(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stocks = await auto_trade_service.scan_stocks(body.codes or [], db)
    return {
        "stocks": stocks,
        "scanned_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }


@router.post("/stop")
@limiter.limit("10/minute")
async def kill_switch(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await auto_trade_service.kill_switch(user.id, db)


@router.post("/reset")
@limiter.limit("5/minute")
async def reset_paper(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    cfg = await auto_trade_service.get_config(user.id, db)
    if cfg.mode != "paper":
        raise HTTPException(status_code=400, detail="초기화는 모의매매 모드에서만 가능합니다.")
    return await auto_trade_service.reset_paper_data(user.id, db)
