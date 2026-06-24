import csv
import io
import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from core.config import settings
from models.portfolio import Portfolio
from models.trade import Trade
from models.user import User
from services import kis_account_service
from services.market_service import get_stock_current_price

router = APIRouter()


async def _get_holdings(user_id, mode: str, db: AsyncSession) -> list:
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == user_id, Portfolio.mode == mode)
    )
    return result.scalars().all()


def _portfolio_from_kis_balance(kis_data: dict) -> dict:
    summary = kis_data.get("summary", {})
    total_eval = int(summary.get("eval_amount", 0))
    total_cost = int(summary.get("buy_amount", 0))
    return {
        "holdings": kis_data.get("holdings", []),
        "total_eval": total_eval,
        "total_cost": total_cost,
        "total_return_pct": float(summary.get("return_pct", 0)),
        "total_asset": int(summary.get("total_asset", total_eval)),
        "deposit": int(summary.get("deposit", 0)),
        "holding_source": kis_data.get("data_source", "KIS 계좌 잔고"),
        "performance_source": "앱 거래 기록 기준",
    }


@router.get("")
@limiter.limit("60/minute")
async def get_portfolio(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_portfolio_response(user, db)


async def _get_portfolio_response(user: User, db: AsyncSession) -> dict:
    mode = settings.SYSTEM_KIS_MODE
    try:
        kis_data = await kis_account_service.get_account_balance(mode)
        return _portfolio_from_kis_balance(kis_data)
    except HTTPException:
        # KIS 잔고 조회가 불가능한 개발/오프라인 상황에서는 앱 DB 기록으로 fallback한다.
        pass

    # KIS 조회 실패 시 DB 기반 계산
    holdings = await _get_holdings(user.id, mode, db)
    result = []
    total_eval = 0
    total_cost = 0

    for h in holdings:
        price_data = await get_stock_current_price(h.stock_code)
        current_price = price_data.get("close", 0)
        cost = int(h.avg_price * h.quantity)
        eval_amount = current_price * h.quantity
        profit = eval_amount - cost
        return_pct = (profit / cost * 100) if cost > 0 else 0
        total_eval += eval_amount
        total_cost += cost
        result.append({
            "stock_code": h.stock_code,
            "stock_name": h.stock_name,
            "quantity": h.quantity,
            "avg_price": float(h.avg_price),
            "current_price": current_price,
            "eval_amount": eval_amount,
            "profit_loss": profit,
            "return_pct": round(return_pct, 2),
        })

    total_return_pct = ((total_eval - total_cost) / total_cost * 100) if total_cost > 0 else 0
    return {
        "holdings": result,
        "total_eval": total_eval,
        "total_cost": total_cost,
        "total_return_pct": round(total_return_pct, 2),
        "total_asset": total_eval,
        "deposit": 0,
        "holding_source": "앱 DB 포트폴리오 fallback",
        "performance_source": "앱 거래 기록 기준",
    }


@router.get("/performance")
@limiter.limit("30/minute")
async def get_performance(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """일별 체결 기준 누적 수익 히스토리 (최근 90일)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    result = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.mode == settings.SYSTEM_KIS_MODE,
            Trade.status == "FILLED",
            Trade.filled_at >= cutoff,
        ).order_by(Trade.filled_at)
    )
    trades = result.scalars().all()

    daily: dict[str, int] = {}
    for t in trades:
        if not t.filled_at or not t.executed_price:
            continue
        day = t.filled_at.strftime("%Y-%m-%d")
        pnl = 0
        if t.order_type == "SELL":
            if t.realized_pnl is not None:
                pnl = t.realized_pnl
            elif t.order_price:
                pnl = int((t.executed_price - t.order_price) * t.quantity)
        daily[day] = daily.get(day, 0) + pnl

    return [{"date": d, "pnl": v} for d, v in sorted(daily.items())]


@router.get("/metrics")
@limiter.limit("30/minute")
async def get_metrics(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """MDD, 샤프비율, 승률 계산."""
    result = await db.execute(
        select(Trade).where(
            Trade.user_id == user.id,
            Trade.mode == settings.SYSTEM_KIS_MODE,
            Trade.status == "FILLED",
            Trade.order_type == "SELL",
        )
    )
    sells = result.scalars().all()

    returns = []
    wins = 0
    for t in sells:
        if t.realized_pnl is not None and t.executed_price:
            # cost_basis = executed * qty - realized_pnl  (역산)
            cost_basis = int(t.executed_price * t.quantity) - t.realized_pnl
            if cost_basis > 0:
                r = t.realized_pnl / cost_basis
                returns.append(r)
                if r > 0:
                    wins += 1
        elif t.executed_price and t.order_price:
            r = float((t.executed_price - t.order_price) / t.order_price)
            returns.append(r)
            if r > 0:
                wins += 1

    win_rate = (wins / len(returns) * 100) if returns else 0

    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns))
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0
    else:
        sharpe = 0

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > mdd:
            mdd = dd

    return {
        "total_trades": len(returns),
        "win_rate_pct": round(win_rate, 2),
        "sharpe_ratio": round(sharpe, 4),
        "mdd_pct": round(mdd * 100, 2),
    }


@router.get("/export")
@limiter.limit("10/minute")
async def export_portfolio(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """포트폴리오 CSV 다운로드."""
    holdings = await _get_holdings(user.id, settings.SYSTEM_KIS_MODE, db)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["종목코드", "종목명", "수량", "평균매수가", "현재가", "평가금액", "수익률(%)"])

    for h in holdings:
        price_data = await get_stock_current_price(h.stock_code)
        current_price = price_data.get("close", 0)
        cost = int(h.avg_price * h.quantity)
        eval_amount = current_price * h.quantity
        return_pct = ((eval_amount - cost) / cost * 100) if cost > 0 else 0
        writer.writerow([
            h.stock_code, h.stock_name or "", h.quantity,
            int(h.avg_price), current_price, eval_amount, round(return_pct, 2)
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio.csv"},
    )
