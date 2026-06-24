"""Autoencoder 기반 이상 패턴 감지 서비스.

종목의 최근 1년 일봉 수익률 시퀀스로 경량 Autoencoder를 즉석 학습하고,
재구성 오차(reconstruction error)가 높은 날을 이상 포인트로 반환한다.

별도 모델 파일 없음 — 요청마다 해당 종목 데이터로 학습(~0.5초).
결과는 Redis에 1시간 캐시.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import numpy as np

from core.redis_client import get_redis
from services.market_service import _last_trading_day

_CACHE_TTL = 3600
_WINDOW = 10        # 입력 시퀀스 길이 (10거래일)
_THRESHOLD_SIGMA = 2.0  # 평균 + N*표준편차 초과 시 이상


def _fetch_ohlcv(code: str, days: int = 365) -> list[dict]:
    from pykrx import stock as pykrx_stock

    end = datetime.strptime(_last_trading_day(), "%Y%m%d")
    start = end - timedelta(days=days)
    df = pykrx_stock.get_market_ohlcv_by_date(
        start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code
    )
    if df is None or df.empty:
        return []

    rows = []
    for date_idx, row in df.iterrows():
        try:
            rows.append({
                "date":  str(date_idx)[:10],
                "close": float(row.get("종가", 0) or 0),
                "volume": float(row.get("거래량", 0) or 0),
            })
        except Exception:
            continue
    return rows


def _train_and_detect(ohlcv: list[dict]) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"anomalies": [], "scores": [], "threshold": 0, "available": False}

    class _Autoencoder(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 3),
            )
            self.decoder = nn.Sequential(
                nn.Linear(3, 8),
                nn.ReLU(),
                nn.Linear(8, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    if len(ohlcv) < _WINDOW * 3:
        return {"anomalies": [], "scores": [], "threshold": 0, "available": False}

    closes = np.array([r["close"] for r in ohlcv], dtype=float)
    # 일별 수익률로 정규화 (스케일 불변)
    returns = np.diff(closes) / (closes[:-1] + 1e-9)

    # 슬라이딩 윈도우로 입력 시퀀스 생성 (마지막 윈도우까지 포함)
    X = np.array([returns[i: i + _WINDOW] for i in range(len(returns) - _WINDOW + 1)])
    if len(X) < 20:
        return {"anomalies": [], "scores": [], "threshold": 0, "available": False}

    # 정규화
    mu, sigma = X.mean(), X.std() + 1e-9
    X_norm = (X - mu) / sigma

    torch.set_num_threads(1)  # 동시 요청 시 CPU 대경합 방지
    tensor = torch.tensor(X_norm, dtype=torch.float32)
    model = _Autoencoder(input_dim=_WINDOW)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(60):  # 빠른 수렴 위해 60 에포크
        optimizer.zero_grad()
        out = model(tensor)
        loss = criterion(out, tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        recon = model(tensor)
        errors = ((recon - tensor) ** 2).mean(dim=1).numpy()

    threshold = float(errors.mean() + _THRESHOLD_SIGMA * errors.std())

    # 이상 포인트: 오류가 임계값 초과
    anomaly_indices = np.where(errors > threshold)[0]

    # 인덱스 → 날짜 매핑 (윈도우 끝 날짜 기준, returns[i:i+_WINDOW] → ohlcv[i+_WINDOW])
    anomalies = []
    for idx in anomaly_indices:
        date_idx = int(idx) + _WINDOW   # diff로 한 칸 밀림이 이미 ohlcv 기준으로 보정됨
        if date_idx < len(ohlcv):
            anomalies.append({
                "date":  ohlcv[date_idx]["date"],
                "score": round(float(errors[idx]), 4),
                "close": ohlcv[date_idx]["close"],
            })

    # 점수 리스트 (차트 오버레이용): 전체 날짜 × 오류값
    score_list = []
    for i, err in enumerate(errors):
        date_idx = i + _WINDOW
        if date_idx < len(ohlcv):
            score_list.append({
                "date":  ohlcv[date_idx]["date"],
                "score": round(float(err), 4),
            })

    return {
        "anomalies": anomalies[-10:],   # 최근 10개만
        "scores":    score_list[-60:],  # 최근 60거래일
        "threshold": round(threshold, 4),
        "available": True,
    }


async def get_anomaly(code: str) -> dict:
    redis = await get_redis()
    cache_key = f"anomaly:v1:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        ohlcv = await asyncio.to_thread(_fetch_ohlcv, code)
        result = await asyncio.to_thread(_train_and_detect, ohlcv)
    except Exception:
        result = {"anomalies": [], "scores": [], "threshold": 0, "available": False}

    result["code"] = code
    # 실패(available=False)는 5분 TTL로 짧게 캐싱해 재시도 가능하게
    ttl = _CACHE_TTL if result.get("available") else 300
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result
