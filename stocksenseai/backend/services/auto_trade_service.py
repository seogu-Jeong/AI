"""AI 기반 자동매매 서비스.

사용자는 예산만 설정. AI가 전종목 스크리닝 → 종목 선정 → 비중 결정 → 매매 실행까지 총괄.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

# 종목코드 → 이름 (JSON 파일 기반, 없으면 빈 dict)
_NAMES_PATH = Path(__file__).parent.parent / "ml" / "stock_names.json"
try:
    _STOCK_NAMES: dict[str, str] = json.loads(_NAMES_PATH.read_text())
except Exception:
    _STOCK_NAMES = {}

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import HTTPException
from models.ai_signal import AISignalHistory
from models.auto_trade import AutoTradeConfig, AutoTradeLog
from models.portfolio import Portfolio
from models.trade import Trade
from models.user import User
from services import risk_service

logger = logging.getLogger(__name__)

_MAX_SINGLE_STOCK_PCT = 0.30   # 한 종목에 총예산의 최대 30%
_CASH_RESERVE_PCT    = 0.10   # 총예산의 10%는 현금 보유


async def _acquire_run_lock(user_id: UUID) -> str | None:
    from core.redis_client import get_redis
    redis = await get_redis()
    token = uuid4().hex
    locked = await redis.set(f"auto_trade:lock:{user_id}", token, ex=360, nx=True)
    return token if locked else None


async def _release_run_lock(user_id: UUID, token: str) -> None:
    from core.redis_client import get_redis
    redis = await get_redis()
    key = f"auto_trade:lock:{user_id}"
    script = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    end
    return 0
    """
    await redis.eval(script, 1, key, token)


def _calculate_buying_power(total_budget: int, used_cost: int) -> int:
    """현금 보유 비율을 제외한 실제 매수 가용 금액 계산.

    Args:
        total_budget: 총 투자 예산 (원)
        used_cost:    이미 투자된 금액 (보유 종목 평가액 합산, 원)

    Returns:
        추가 매수 가능 금액 (음수가 되는 경우 0 반환)
    """
    investable_limit = int(total_budget * (1 - _CASH_RESERVE_PCT))
    return max(0, investable_limit - used_cost)


async def get_config(user_id: UUID, db: AsyncSession) -> AutoTradeConfig:
    result = await db.execute(
        select(AutoTradeConfig).where(AutoTradeConfig.user_id == user_id)
    )
    cfg = result.scalar_one_or_none()
    if cfg is None:
        cfg = AutoTradeConfig(user_id=user_id)
        db.add(cfg)
        await db.commit()
        await db.refresh(cfg)
    return cfg


async def update_config(user_id: UUID, data: dict, db: AsyncSession) -> AutoTradeConfig:
    cfg = await get_config(user_id, db)
    allowed = {"enabled", "mode", "total_budget", "stop_loss_pct", "take_profit_pct", "max_positions", "signal_threshold"}
    for key, val in data.items():
        if key in allowed:
            setattr(cfg, key, val)
    await db.commit()
    await db.refresh(cfg)
    return cfg


async def get_logs(user_id: UUID, db: AsyncSession, limit: int = 50) -> list[dict]:
    result = await db.execute(
        select(AutoTradeLog)
        .where(AutoTradeLog.user_id == user_id)
        .order_by(AutoTradeLog.created_at.desc())
        .limit(limit)
    )
    return [
        {
            "id": str(r.id),
            "stock_code": r.stock_code,
            "stock_name": r.stock_name,
            "action": r.action,
            "quantity": r.quantity,
            "price": r.price,
            "total_amount": r.total_amount,
            "reason": r.reason,
            "signal_score": r.signal_score,
            "mode": r.mode,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in result.scalars().all()
    ]


_MAJOR_50 = [
    "005930","000660","035720","005380","051910","006400","068270","207940",
    "035420","105560","055550","086790","032830","028260","066570","017670",
    "003550","012330","011200","096770","034220","000270","015760","009150",
    "018260","010950","011070","047050","024110","000810","033780","030200",
    "003490","036570","251270","316140","323410","402340","259960","293490",
    "352820","035900","036460","180640","011780","009830","004020","010060",
    "000100","007070",
]


async def _get_buy_candidates(db: AsyncSession, extra_codes: list[str] | None = None) -> list[dict]:
    """AI BUY 신호 종목 수집 (점수 내림차순). extra_codes는 사용자 관심종목."""
    from services.ai_service import get_signal, get_top_picks

    candidates: dict[str, dict] = {}

    # 1. LSTM top picks
    try:
        for p in (await get_top_picks()).get("picks", []):
            candidates[p["code"]] = {"code": p["code"], "score": float(p.get("signal_score", 0))}
    except Exception:
        pass

    # 2. 최근 24h DB BUY 신호
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    try:
        result = await db.execute(
            select(AISignalHistory)
            .where(AISignalHistory.signal == "BUY", AISignalHistory.recorded_at >= cutoff)
            .order_by(desc(AISignalHistory.signal_score))
            .limit(50)
        )
        for row in result.scalars().all():
            if row.stock_code not in candidates:
                candidates[row.stock_code] = {
                    "code": row.stock_code,
                    "score": float(row.signal_score or 0),
                }
    except Exception:
        pass

    # 3. 사용자 관심 종목 + 코스피 주요 50종목 스캔
    scan_codes = list(dict.fromkeys((extra_codes or []) + _MAJOR_50))
    for code in scan_codes:
        if code in candidates:
            continue
        try:
            sig = await get_signal(code, db)
            if sig.get("signal") == "BUY":
                candidates[code] = {"code": code, "score": float(sig.get("signal_score", 0))}
        except Exception:
            continue

    return sorted(candidates.values(), key=lambda x: x["score"], reverse=True)


async def scan_stocks(codes: list[str], db: AsyncSession) -> list[dict]:
    """주어진 종목들의 현재 AI 신호를 조회 (매매 없이 분석만)."""
    from services.ai_service import get_signal

    results: list[dict] = []
    seen: set[str] = set()

    # 전달받은 코드 + MAJOR_50 합쳐서 최대 60종목
    all_codes = list(dict.fromkeys(codes + _MAJOR_50))[:60]

    for code in all_codes:
        if code in seen:
            continue
        seen.add(code)
        try:
            sig = await get_signal(code, db)
            results.append({
                "code": code,
                "name": _STOCK_NAMES.get(code, code),
                "signal": sig.get("signal", "HOLD"),
                "score": float(sig.get("signal_score", 0)),
                "rsi": float(sig.get("indicators", {}).get("rsi_14", 0) or 0),
            })
        except Exception:
            results.append({
                "code": code,
                "name": _STOCK_NAMES.get(code, code),
                "signal": "HOLD",
                "score": 0.0,
                "rsi": 0.0,
            })

    # BUY 우선, 점수 내림차순
    priority = {"BUY": 0, "HOLD": 1, "SELL": 2}
    results.sort(key=lambda x: (priority.get(x["signal"], 1), -x["score"]))
    return results


def _allocate(candidates: list[dict], available: int, total_budget: int) -> list[dict]:
    """신호 강도 비례 비중 배분. 단일 종목 최대 30%, 현금 10% 보유."""
    if not candidates or available <= 0:
        return []

    investable = min(available, int(total_budget * (1 - _CASH_RESERVE_PCT)))
    cap = int(total_budget * _MAX_SINGLE_STOCK_PCT)

    total_score = sum(c["score"] for c in candidates)
    if total_score <= 0:
        per = investable // len(candidates)
        return [{"code": c["code"], "score": c["score"], "alloc": min(per, cap)} for c in candidates]

    result = []
    for c in candidates:
        alloc = min(int(investable * c["score"] / total_score), cap)
        if alloc > 0:
            result.append({"code": c["code"], "score": c["score"], "alloc": alloc})
    return result


async def _execute_paper_order(
    user_id: UUID, stock_code: str, stock_name: str,
    order_type: str, quantity: int, price: int,
    reason: str, mode: str, signal_score: float,
    db: AsyncSession,
    warning: str | None = None,
) -> dict[str, Any]:
    if quantity <= 0 or price <= 0:
        raise ValueError(f"invalid quantity={quantity} or price={price}")

    result = await db.execute(
        select(Portfolio).where(
            Portfolio.user_id == user_id,
            Portfolio.stock_code == stock_code,
            Portfolio.mode == mode,
        )
    )
    holding = result.scalar_one_or_none()
    executed_qty = quantity

    if order_type == "SELL":
        if holding is None:
            raise ValueError(f"SELL 실패: {stock_code} 보유 없음")
        executed_qty = min(quantity, holding.quantity)
        if executed_qty <= 0:
            raise ValueError(f"SELL 실패: {stock_code} 보유 수량이 0 (데이터 오염 의심)")

    trade = Trade(
        user_id=user_id, stock_code=stock_code, stock_name=stock_name,
        order_type=order_type, price_type="MARKET",
        quantity=executed_qty, executed_price=price, filled_quantity=executed_qty,
        status="FILLED", mode=mode,
        ai_signal_at_order=reason[:10] if reason else None,
        filled_at=datetime.now(timezone.utc),
    )
    db.add(trade)

    if order_type == "BUY":
        if holding is None:
            db.add(Portfolio(user_id=user_id, stock_code=stock_code, stock_name=stock_name,
                             quantity=executed_qty, avg_price=price, mode=mode))
        else:
            new_qty = holding.quantity + executed_qty
            new_avg = (float(holding.avg_price) * holding.quantity + price * executed_qty) / new_qty
            holding.quantity = new_qty
            holding.avg_price = round(new_avg, 2)
    elif order_type == "SELL":
        trade.realized_pnl = int((price - float(holding.avg_price)) * executed_qty)
        trade.filled_quantity = executed_qty
        trade.quantity = executed_qty
        holding.quantity -= executed_qty
        if holding.quantity <= 0:
            await db.delete(holding)

    if order_type == "BUY" and warning:
        reason_str = f"{reason} [WARN: {warning[:80]}]"
    else:
        reason_str = reason

    db.add(AutoTradeLog(
        user_id=user_id, stock_code=stock_code, stock_name=stock_name,
        action=order_type, quantity=executed_qty, price=price,
        total_amount=executed_qty * price, reason=reason_str,
        signal_score=signal_score, mode=mode,
    ))
    await db.commit()
    ret = {"action": order_type, "stock_code": stock_code, "stock_name": stock_name,
           "quantity": executed_qty, "price": price, "total_amount": executed_qty * price,
           "reason": reason_str}
    if warning:
        ret["warning"] = warning
    return ret


async def run_cycle(user_id: UUID, db: AsyncSession, extra_codes: list[str] | None = None) -> dict[str, Any]:
    from services.ai_service import get_signal
    from services.market_service import get_stock_current_price

    cfg = await get_config(user_id, db)
    if not cfg.enabled:
        return {"skipped": True, "reason": "not_enabled"}

    user = await db.get(User, user_id)
    if user is None:
        return {"skipped": True, "reason": "user_not_found"}

    lock_token = await _acquire_run_lock(user_id)
    if not lock_token:
        return {"skipped": True, "reason": "already_running"}
    try:
        actions: list[dict] = []
        diagnostics: dict[str, Any] = {
            "signal_fetch_failed": 0,
            "price_fetch_failed": 0,
            "risk_blocked": 0,
            "below_threshold": 0,
            "max_positions_reached": False,
        }

        is_real = cfg.mode == "real"

        # ── real 모드: KIS에서 실제 보유 종목 + 예수금 가져오기 ──────────────
        real_holdings: list[dict] = []
        if is_real:
            try:
                from services.kis_service import get_balance, get_balance_full
                bal_full = await get_balance_full(user)
                real_holdings = bal_full.get("holdings", [])
                bal = await get_balance(user)
                available = bal.get("cash", 0)
            except Exception as exc:
                logger.warning("KIS 잔고 조회 실패, 사이클 중단: %s", exc)
                return {"skipped": True, "reason": "kis_balance_failed",
                        "message": f"KIS 잔고 조회 실패: {exc}"}

        # ── 1. 기존 보유 포지션: 손절/익절/AI SELL ────────────────────────
        if is_real:
            iter_holdings = [
                type("H", (), {
                    "stock_code": h["stock_code"],
                    "stock_name": h.get("stock_name", h["stock_code"]),
                    "quantity": h["quantity"],
                    "avg_price": h["avg_price"],
                })()
                for h in real_holdings
            ]
        else:
            holdings_res = await db.execute(
                select(Portfolio).where(Portfolio.user_id == user_id, Portfolio.mode == cfg.mode)
            )
            iter_holdings = holdings_res.scalars().all()

        for holding in iter_holdings:
            try:
                price_data = await get_stock_current_price(holding.stock_code)
                cur = price_data.get("close", 0)
            except Exception:
                diagnostics["price_fetch_failed"] += 1
                continue
            if cur <= 0:
                continue

            avg = float(holding.avg_price)
            pct = (cur - avg) / avg * 100.0
            sell_reason, sell_score = None, 0.0

            if pct <= -cfg.stop_loss_pct:
                sell_reason = f"손절({pct:.1f}%)"
            elif pct >= cfg.take_profit_pct:
                sell_reason = f"익절(+{pct:.1f}%)"
            else:
                try:
                    sig = await get_signal(holding.stock_code, db)
                    sell_score = float(sig.get("signal_score", 0))
                    if sig.get("signal") == "SELL":
                        sell_reason = f"AI SELL({sell_score:.0f}점)"
                except Exception:
                    diagnostics["signal_fetch_failed"] += 1

            if sell_reason:
                try:
                    if is_real:
                        log = await _execute_real_order(
                            user, holding.stock_code, holding.stock_name or "",
                            "SELL", holding.quantity, cur, sell_reason, sell_score, db,
                        )
                    else:
                        log = await _execute_paper_order(
                            user_id, holding.stock_code, holding.stock_name or "",
                            "SELL", holding.quantity, cur,
                            sell_reason, cfg.mode, sell_score, db,
                        )
                    actions.append(log)
                except Exception as exc:
                    logger.error("매도 실패 %s: %s", holding.stock_code, exc)

        # ── 2. AI 스크리닝 → 신규 매수 ────────────────────────────────────
        if not is_real:
            used = sum(int(float(h.avg_price) * h.quantity) for h in (
                (await db.execute(select(Portfolio).where(
                    Portfolio.user_id == user_id, Portfolio.mode == cfg.mode
                ))).scalars().all()
            ))
            available = _calculate_buying_power(cfg.total_budget, used)

        candidates: list[dict] = []
        held: set[str] = set()
        fresh: list[dict] = []
        remaining_slots: int = 0

        if available > 0:
            candidates = await _get_buy_candidates(db, extra_codes=extra_codes)

            if is_real:
                held = {h["stock_code"] for h in real_holdings}
            else:
                held_res = await db.execute(
                    select(Portfolio.stock_code).where(
                        Portfolio.user_id == user_id, Portfolio.mode == cfg.mode
                    )
                )
                held = {r[0] for r in held_res.fetchall()}

            fresh = [
                c for c in candidates
                if c["code"] not in held and c["score"] >= cfg.signal_threshold
            ]
            remaining_slots = max(0, cfg.max_positions - len(held))
            diagnostics["below_threshold"] = len([c for c in candidates if c["code"] not in held and c["score"] < cfg.signal_threshold])
            diagnostics["max_positions_reached"] = remaining_slots == 0
            fresh = fresh[:remaining_slots]

            # real 모드 budget_ref: KIS 예수금 기반 / paper 모드: cfg.total_budget
            budget_ref = available if is_real else cfg.total_budget
            allocations = _allocate(fresh, available, budget_ref)

            for alloc in allocations:
                if available <= 0:
                    break
                try:
                    price_data = await get_stock_current_price(alloc["code"])
                    cur = price_data.get("close", 0)
                    name = price_data.get("name", _STOCK_NAMES.get(alloc["code"], alloc["code"]))
                except Exception:
                    diagnostics["price_fetch_failed"] += 1
                    continue
                if cur <= 0:
                    continue

                qty = alloc["alloc"] // cur
                if qty < 1:
                    continue

                # ── risk check ──────────────────────────────────────────────
                try:
                    warning = await risk_service.check_order(
                        user, alloc["code"], "BUY", qty, cur, db, mode=cfg.mode
                    )
                except HTTPException as exc:
                    actions.append({
                        "action": "SKIP",
                        "stock_code": alloc["code"],
                        "reason": f"risk_blocked:{exc.detail}",
                    })
                    diagnostics["risk_blocked"] += 1
                    continue
                # ─────────────────────────────────────────────────────────────

                try:
                    if is_real:
                        log = await _execute_real_order(
                            user, alloc["code"], name, "BUY", qty, cur,
                            f"AI BUY({alloc['score']:.0f}점)", alloc["score"], db,
                        )
                    else:
                        log = await _execute_paper_order(
                            user_id, alloc["code"], name, "BUY", qty, cur,
                            f"AI BUY({alloc['score']:.0f}점)", cfg.mode, alloc["score"], db,
                            warning=warning,
                        )
                    actions.append(log)
                    available -= qty * cur
                except Exception as exc:
                    logger.error("매수 실패 %s: %s", alloc["code"], exc)

        # 매매 없는 경우 이유 설명
        trade_actions = [a for a in actions if a.get("action") in ("BUY", "SELL")]
        no_trade_reason = None
        if not trade_actions:
            invested = 0 if is_real else used  # real 모드는 KIS 잔고가 이미 반영됨
            invested_str = f"{invested // 100000000}억원" if invested >= 100000000 else (
                f"{invested // 10000}만원" if invested >= 10000 else f"{invested:,}원"
            )
            if available <= 0:
                no_trade_reason = f"가용 예산 부족 또는 현금 보유 한도 도달 (투자됨: {invested_str})"
            elif not candidates:
                no_trade_reason = "분석된 BUY 종목 없음 (신호 데이터 부족)"
            elif all(c["code"] in held for c in candidates):
                no_trade_reason = f"BUY 후보 {len(candidates)}개 모두 이미 보유 중"
            elif remaining_slots == 0:
                no_trade_reason = f"보유 종목 수 한도 도달 ({len(held)}/{cfg.max_positions})"
            elif not fresh:
                no_trade_reason = f"BUY 후보 {len(candidates)}개 모두 신호 점수 미달 (기준: {cfg.signal_threshold}점)"
            elif diagnostics["risk_blocked"] > 0:
                no_trade_reason = f"BUY 후보 {diagnostics['risk_blocked']}개 리스크 규칙으로 차단됨"
            else:
                no_trade_reason = f"BUY 후보 {len(candidates)}개 분석 — 1주 매수 금액 미달"

        return {
            "executed": len(trade_actions),
            "actions": actions,
            "scanned": len(candidates),
            "held_count": len(held),
            "no_trade_reason": no_trade_reason,
            "diagnostics": diagnostics,
        }
    finally:
        await _release_run_lock(user_id, lock_token)


async def reset_paper_data(user_id: UUID, db: AsyncSession) -> dict[str, Any]:
    """모의매매 포지션 및 거래 기록 전체 초기화."""
    await db.execute(
        delete(Portfolio).where(Portfolio.user_id == user_id, Portfolio.mode == "paper")
    )
    await db.execute(
        delete(AutoTradeLog).where(AutoTradeLog.user_id == user_id)
    )
    cfg = await get_config(user_id, db)
    cfg.enabled = False
    await db.commit()
    return {"reset": True, "message": "모의매매가 초기화되었습니다."}


async def _execute_real_order(
    user: Any, stock_code: str, stock_name: str,
    order_type: str, quantity: int, price: int,
    reason: str, signal_score: float, db: AsyncSession,
) -> dict[str, Any]:
    """실거래 모드: KIS API로 실주문 실행 후 로그 기록."""
    from services.kis_service import place_order

    result = await place_order(user, stock_code, order_type, "MARKET", quantity, price=0)

    db.add(AutoTradeLog(
        user_id=user.id, stock_code=stock_code, stock_name=stock_name,
        action=order_type, quantity=quantity, price=price,
        total_amount=quantity * price, reason=reason,
        signal_score=signal_score, mode="real",
    ))
    await db.commit()

    return {
        "action": order_type, "stock_code": stock_code, "stock_name": stock_name,
        "quantity": quantity, "price": price, "total_amount": quantity * price,
        "reason": reason, "kis_order_no": result.get("kis_order_no", ""),
    }


async def kill_switch(user_id: UUID, db: AsyncSession) -> dict[str, Any]:
    cfg = await get_config(user_id, db)
    cfg.enabled = False
    await db.commit()
    return {"stopped": True, "message": "자동매매가 비활성화되었습니다."}
