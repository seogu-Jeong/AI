from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.redis_client import get_redis
from ml.features import FEATURE_COLS, build_features
from ml.pattern_matcher import find_similar_patterns
from ml.predict import WEIGHTS_DIR, get_lstm_direction, predict_scenarios
from models.ai_signal import AISignalHistory
from services.market_service import _is_market_open, get_ohlcv_cached

_KST = ZoneInfo("Asia/Seoul")


def _ohlcv_to_df(data: list[dict]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").sort_index()


def _calc_tech_score(indicators: dict) -> float:
    """기술적 지표 점수 0~100: RSI(40%) + MACD(35%) + BB위치(25%)"""
    rsi = float(indicators.get("rsi_14", 50.0))
    rsi_score = rsi if rsi >= 50 else 100 - rsi

    macd_hist = float(indicators.get("macd_hist", 0.0))
    macd_score = 70.0 if macd_hist > 0 else 30.0

    close = float(indicators.get("close", 0.0))
    bb_upper = float(indicators.get("bb_upper", close))
    bb_lower = float(indicators.get("bb_lower", close))
    bb_range = bb_upper - bb_lower
    bb_score = 50.0
    if bb_range > 0:
        bb_pos = (close - bb_lower) / bb_range
        bb_score = bb_pos * 100

    return rsi_score * 0.4 + macd_score * 0.35 + bb_score * 0.25


def _score_to_signal(score: float) -> str:
    if score >= 65:
        return "BUY"
    if score <= 35:
        return "SELL"
    return "HOLD"


def _extract_indicators(df: pd.DataFrame, feat_df: pd.DataFrame) -> dict:
    """이미 계산된 DataFrame에서 지표 원시값 추출."""
    if feat_df.empty:
        return {}
    last = feat_df.iloc[-1]
    return {
        "rsi_14": round(float(last.get("rsi_14", 0)), 2),
        "macd": round(float(last.get("macd", 0)), 4),
        "macd_hist": round(float(last.get("macd_hist", 0)), 4),
        "bb_upper": round(float(last.get("bb_upper", 0))),
        "bb_lower": round(float(last.get("bb_lower", 0))),
        "ma5": round(float(last.get("ma5", 0))),
        "ma20": round(float(last.get("ma20", 0))),
        "close": round(float(df["close"].iloc[-1])),
    }


async def get_indicators(code: str) -> dict:
    """OHLCV 로드 후 기술적 지표 원시값 반환."""
    raw = await get_ohlcv_cached(code, "3m", "day")
    df = _ohlcv_to_df(raw)
    if df.empty or len(df) < 60:
        return {}
    feat_df = build_features(df)
    return _extract_indicators(df, feat_df)


async def calculate_signal(code: str, db: AsyncSession | None = None) -> dict:
    """AI 시그널 계산. DB 세션 전달 시 ai_signals_history에 저장."""
    raw = await get_ohlcv_cached(code, "3m", "day")
    df = _ohlcv_to_df(raw)
    if df.empty or len(df) < 60:
        return {"code": code, "signal": "HOLD", "signal_score": 50.0, "lstm_available": False}

    feat_df = build_features(df)
    indicators = _extract_indicators(df, feat_df)
    tech_score = _calc_tech_score(indicators)

    lstm_direction = await asyncio.to_thread(get_lstm_direction, code, df)
    lstm_available = lstm_direction is not None

    if lstm_available:
        lstm_score = 50.0 + lstm_direction * 50.0
        final_score = tech_score * 0.4 + lstm_score * 0.6
    else:
        lstm_score = 50.0
        final_score = tech_score

    signal = _score_to_signal(final_score)
    result = {
        "code": code,
        "signal": signal,
        "signal_score": round(final_score, 1),
        "signal_breakdown": {
            "technical_score": round(tech_score, 1),
            "lstm_score": round(lstm_score, 1),
            "technical_weight": 0.4 if lstm_available else 1.0,
            "lstm_weight": 0.6 if lstm_available else 0.0,
        },
        "lstm_available": lstm_available,
        "as_of": datetime.now(_KST).isoformat(),
        "indicators": indicators,
    }

    if db is not None:
        await db.execute(
            insert(AISignalHistory).values(
                stock_code=code,
                signal=signal,
                signal_score=final_score,
                tech_score=tech_score,
                lstm_score=lstm_score if lstm_available else None,
                rsi=indicators.get("rsi_14"),
                macd=indicators.get("macd"),
            )
        )
        await db.commit()

    return result


async def get_signal(code: str, db: AsyncSession | None = None) -> dict:
    """캐시 조회 → 없으면 calculate_signal 호출."""
    redis = await get_redis()
    cache_key = f"ai_signal:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    result = await calculate_signal(code, db)
    ttl = 300 if _is_market_open() else 86400
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result


async def get_prediction(code: str, db: AsyncSession | None = None) -> dict:
    """LSTM 5일 예측.
    우선순위: DB 저장 예측 → 로컬 추론(가중치 있을 때) → 빈 값
    """
    redis = await get_redis()
    cache_key = f"ai_predict:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # 1. DB에서 최근 저장된 예측 조회. 저장값만으로 응답해 외부 시세 장애와 무관하게 동작한다.
    if db is not None:
        res = await db.execute(
            select(AISignalHistory)
            .where(
                AISignalHistory.stock_code == code,
                AISignalHistory.predicted_prices.isnot(None),
            )
            .order_by(AISignalHistory.recorded_at.desc())
            .limit(1)
        )
        row = res.scalar_one_or_none()
        if row and row.predicted_prices:
            p = row.predicted_prices
            recorded_at = row.recorded_at
            if recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=timezone.utc)
            result = {
                "code": code,
                "current_price": int(p.get("current_price", 0)),
                "prediction": {
                    "bullish": p.get("bullish", []),
                    "base": p.get("base", []),
                    "bearish": p.get("bearish", []),
                },
                "lstm_available": True,
                "predicted_at": row.recorded_at.isoformat(),
                "confidence": float(row.confidence or p.get("confidence", 0)),
                "source": "stored",
                "stale": datetime.now(timezone.utc) - recorded_at > timedelta(days=3),
            }
            ttl = 300 if _is_market_open() else 86400
            await redis.setex(cache_key, ttl, json.dumps(result))
            return result

    # 2. DB에 없으면 로컬 추론 시도 (가중치 있을 때만)
    raw = await get_ohlcv_cached(code, "1y", "day")
    df = _ohlcv_to_df(raw)
    scenarios = None
    if not df.empty:
        scenarios = await asyncio.to_thread(predict_scenarios, code, df)

    current = int(df["close"].iloc[-1]) if not df.empty else 0
    result = {
        "code": code,
        "current_price": current,
        "prediction": scenarios or {"bullish": [], "base": [], "bearish": []},
        "lstm_available": scenarios is not None,
        "predicted_at": None,
        "confidence": None,
        "source": "local" if scenarios is not None else "unavailable",
        "stale": False,
    }
    ttl = 300 if _is_market_open() else 86400
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result


async def get_similar(code: str) -> dict:
    """유사 패턴 Top 5 (캐시 포함)."""
    redis = await get_redis()
    cache_key = f"ai_similar:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    raw = await get_ohlcv_cached(code, "2y", "day")
    df = _ohlcv_to_df(raw)
    similar = await asyncio.to_thread(find_similar_patterns, df) if not df.empty else []

    result = {"code": code, "similar": similar}
    ttl = 300 if _is_market_open() else 86400
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result


async def get_multiframe(code: str) -> dict:
    """일봉/주봉/월봉 멀티타임프레임 시그널 (캐시 포함)."""
    redis = await get_redis()
    cache_key = f"ai_multiframe:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # 일봉 2년치를 가져온 뒤 resample로 주봉/월봉을 파생한다.
    # pykrx 는 'w'(주봉) 미지원, 'm'(월봉)도 불안정하므로 일봉 기반으로 처리한다.
    frames: dict[str, dict] = {}
    try:
        raw_daily = await get_ohlcv_cached(code, "3y", "day")
    except Exception:
        raw_daily = []

    daily_df = _ohlcv_to_df(raw_daily)

    resample_cfg: list[tuple[str, str, int]] = [
        ("daily",   "",   30),
        ("weekly",  "W",  20),
        ("monthly", "ME", 12),
    ]
    for label, rule, min_rows in resample_cfg:
        if daily_df.empty:
            continue
        if rule:
            df = (
                daily_df.resample(rule)
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )
        else:
            df = daily_df.copy()
        if len(df) < min_rows:
            continue
        feat = build_features(df)
        if feat.empty:
            continue
        last = feat.iloc[-1]
        indicators = {
            "rsi_14": float(last.get("rsi_14", 50)),
            "macd_hist": float(last.get("macd_hist", 0)),
            "close": float(df["close"].iloc[-1]),
            "bb_upper": float(last.get("bb_upper", 0)),
            "bb_lower": float(last.get("bb_lower", 0)),
        }
        score = _calc_tech_score(indicators)
        frames[label] = {"signal": _score_to_signal(score), "score": round(score, 1)}

    result = {"code": code, "timeframes": frames}
    ttl = 300 if _is_market_open() else 86400
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result


async def get_top_picks() -> dict:
    """학습된 종목 중 BUY 시그널 상위 20개 (캐시 300초)."""
    redis = await get_redis()
    cache_key = "ai_top_picks"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    codes = [p.stem for p in WEIGHTS_DIR.glob("*.pth")]
    picks = []
    for code in codes[:50]:
        try:
            signal = await get_signal(code)
            if signal.get("signal") == "BUY":
                picks.append({"code": code, "signal_score": signal.get("signal_score", 0)})
        except Exception:
            continue

    picks.sort(key=lambda x: x["signal_score"], reverse=True)
    result = {"picks": picks[:20]}
    await redis.setex(cache_key, 300, json.dumps(result))
    return result
