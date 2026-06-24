# backend/services/risk_service.py
from __future__ import annotations

import uuid as _uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.portfolio import Portfolio
from models.risk import RiskSettings
from models.trade import Trade


async def get_or_create_settings(user_id, db: AsyncSession) -> RiskSettings:
    """없으면 기본값으로 생성."""
    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    result = await db.execute(select(RiskSettings).where(RiskSettings.user_id == user_id))
    settings = result.scalar_one_or_none()
    if settings is None:
        settings = RiskSettings(user_id=user_id)
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    return settings


async def _get_portfolio_total(user_id, mode: str, db: AsyncSession) -> int:
    """현재 포트폴리오 총 매수금액 합산."""
    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == user_id, Portfolio.mode == mode)
    )
    holdings = result.scalars().all()
    return sum(int(h.avg_price * h.quantity) for h in holdings)


async def _get_holding_value(user_id, stock_code: str, mode: str, db: AsyncSession) -> int:
    """특정 종목 현재 보유 평가금액."""
    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.user_id == user_id,
            Portfolio.stock_code == stock_code,
            Portfolio.mode == mode,
        )
    )
    holding = result.scalar_one_or_none()
    if not holding:
        return 0
    return int(holding.avg_price * holding.quantity)


async def _get_today_loss(user_id, mode: str, db: AsyncSession) -> int:
    """오늘 실현 손실 합산 (SELL 체결 기준, KST 거래일 기준)."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from sqlalchemy import and_

    if isinstance(user_id, str):
        user_id = _uuid.UUID(user_id)
    _KST = ZoneInfo("Asia/Seoul")
    today_start = datetime.now(_KST).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    result = await db.execute(
        select(Trade).where(
            and_(
                Trade.user_id == user_id,
                Trade.mode == mode,
                Trade.order_type == "SELL",
                Trade.status == "FILLED",
                Trade.filled_at >= today_start,
            )
        )
    )
    trades = result.scalars().all()
    total_loss = 0
    for t in trades:
        if t.realized_pnl is not None:
            if t.realized_pnl < 0:
                total_loss += abs(t.realized_pnl)
        elif t.executed_price and t.order_price:
            # 이전 기록 fallback (realized_pnl 컬럼 추가 전 데이터)
            pnl = int((t.executed_price - t.order_price) * t.quantity)
            if pnl < 0:
                total_loss += abs(pnl)
    return total_loss


async def check_order(
    user,
    stock_code: str,
    order_type: str,
    quantity: int,
    price: int,
    db: AsyncSession,
    mode: str | None = None,
) -> str | None:
    """
    주문 전 리스크 체크.
    반환: None (통과) | 경고 문자열 (경고 모드)
    한도 초과 + enforce_hard_stop=True → HTTPException 400
    """
    settings = await get_or_create_settings(user.id, db)
    if order_type == "SELL":
        return None
    if settings.trading_blocked:
        raise HTTPException(status_code=400, detail="거래가 차단된 상태입니다. 리스크 설정에서 확인하세요.")
    mode = mode or user.mode

    portfolio_total = await _get_portfolio_total(user.id, mode, db)
    holding_value = await _get_holding_value(user.id, stock_code, mode, db)
    new_order_value = quantity * price

    warnings = []

    # 1. 종목별 한도 체크
    if portfolio_total > 0:
        new_pct = (holding_value + new_order_value) / (portfolio_total + new_order_value) * 100
        max_pct = float(settings.max_per_stock_pct)
        if new_pct > max_pct:
            msg = f"종목별 한도 초과: {new_pct:.1f}% > {max_pct:.1f}%"
            if settings.enforce_hard_stop:
                raise HTTPException(status_code=400, detail=msg)
            warnings.append(msg)

    # 2. 일일 손실 한도 체크
    today_loss = await _get_today_loss(user.id, mode, db)
    if portfolio_total > 0:
        loss_pct = today_loss / portfolio_total * 100
        max_loss = float(settings.daily_loss_limit_pct)
        if loss_pct > max_loss:
            msg = f"일일 손실 한도 초과: {loss_pct:.1f}% > {max_loss:.1f}%"
            if settings.enforce_hard_stop:
                raise HTTPException(status_code=400, detail=msg)
            warnings.append(msg)

    return " / ".join(warnings) if warnings else None
