# backend/api/routes/backtest.py
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db, get_optional_user
from api.middleware.rate_limit import limiter
from models.backtest import BacktestResult
from models.user import User
from services import backtest_service
from services.backtest_service import BacktestConfig, PortfolioStock

router = APIRouter()


class BacktestRequest(BaseModel):
    code: str
    start_date: date
    end_date: date
    initial_cash: int = Field(default=10_000_000, gt=0)
    entry_signal_score: float = Field(default=65.0, ge=0, le=100)
    exit_signal_score: float = Field(default=35.0, ge=0, le=100)
    stop_loss_pct: float = Field(default=0.05, ge=0, le=1)
    take_profit_pct: float = Field(default=0.15, ge=0, le=1)
    commission_rate: float = Field(default=0.00015, ge=0, le=0.01)

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v, info):
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v

    @model_validator(mode="after")
    def entry_above_exit(self) -> "BacktestRequest":
        if self.entry_signal_score <= self.exit_signal_score:
            raise ValueError("entry_signal_score는 exit_signal_score보다 커야 합니다.")
        return self


def _serialize_result(r: BacktestResult) -> dict:
    return {
        "id": str(r.id),
        "stock_code": r.stock_code,
        "period_start": str(r.period_start),
        "period_end": str(r.period_end),
        "total_return_pct": float(r.total_return_pct or 0),
        "mdd_pct": float(r.mdd_pct or 0),
        "sharpe_ratio": float(r.sharpe_ratio or 0),
        "win_rate_pct": float(r.win_rate_pct or 0),
        "total_trades": r.total_trades or 0,
        "strategy_config": r.strategy_config,
        "result_detail": r.result_detail,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


class PortfolioStockItem(BaseModel):
    code: str
    name: str
    weight_pct: float = Field(gt=0, le=100)


class PortfolioBacktestRequest(BaseModel):
    stocks: list[PortfolioStockItem] = Field(min_length=1, max_length=10)
    start_date: date
    end_date: date
    initial_cash: int = Field(default=10_000_000, gt=0)
    entry_signal_score: float = Field(default=65.0, ge=0, le=100)
    exit_signal_score: float = Field(default=35.0, ge=0, le=100)
    stop_loss_pct: float = Field(default=0.05, ge=0, le=1)
    take_profit_pct: float = Field(default=0.15, ge=0, le=1)
    commission_rate: float = Field(default=0.00015, ge=0, le=0.01)

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v, info):
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v

    @model_validator(mode="after")
    def validate_weights(self) -> "PortfolioBacktestRequest":
        total = sum(s.weight_pct for s in self.stocks)
        if abs(total - 100.0) > 0.1:
            raise ValueError(f"비중 합계는 100%여야 합니다. 현재: {total:.1f}%")
        if self.entry_signal_score <= self.exit_signal_score:
            raise ValueError("entry_signal_score는 exit_signal_score보다 커야 합니다.")
        return self


@router.post("/portfolio-run")
@limiter.limit("3/minute")
async def run_portfolio_backtest(
    request: Request,
    body: PortfolioBacktestRequest,
    user: User | None = Depends(get_optional_user),
):
    ps_list = [PortfolioStock(code=s.code, name=s.name, weight_pct=s.weight_pct) for s in body.stocks]
    return await backtest_service.run_portfolio_backtest(
        stocks=ps_list,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_cash=body.initial_cash,
        entry_signal_score=body.entry_signal_score,
        exit_signal_score=body.exit_signal_score,
        stop_loss_pct=body.stop_loss_pct,
        take_profit_pct=body.take_profit_pct,
        commission_rate=body.commission_rate,
    )


@router.post("/run")
@limiter.limit("5/minute")
async def run_backtest(
    request: Request,
    body: BacktestRequest,
    user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    config = BacktestConfig(
        code=body.code,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_cash=body.initial_cash,
        entry_signal_score=body.entry_signal_score,
        exit_signal_score=body.exit_signal_score,
        stop_loss_pct=body.stop_loss_pct,
        take_profit_pct=body.take_profit_pct,
        commission_rate=body.commission_rate,
    )
    result = await backtest_service.run_backtest(config, user.id if user else None, db)
    return _serialize_result(result)


@router.get("/{backtest_id}")
@limiter.limit("30/minute")
async def get_backtest(
    request: Request,
    backtest_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        bid = uuid.UUID(backtest_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 backtest_id")

    result = await db.execute(
        select(BacktestResult).where(
            BacktestResult.id == bid,
            BacktestResult.user_id == user.id,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="백테스팅 결과를 찾을 수 없습니다.")
    return _serialize_result(row)
