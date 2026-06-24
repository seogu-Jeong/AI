import numpy as np
import pandas as pd


def find_similar_patterns(
    df: pd.DataFrame, lookback: int = 20, top_k: int = 5
) -> list[dict]:
    """
    최근 lookback일 종가 패턴과 유사한 과거 구간 Top-k 반환.
    유사도: cosine similarity (정규화 후)
    반환: [{date, similarity, actual_return_5d}]
    """
    closes = df["close"].values.astype(float)
    if len(closes) < lookback + 5:
        return []

    recent = closes[-lookback:]
    r_min, r_max = recent.min(), recent.max()
    recent_norm = (recent - r_min) / (r_max - r_min + 1e-10)

    results = []
    for i in range(len(closes) - lookback - 5):
        window = closes[i : i + lookback]
        w_min, w_max = window.min(), window.max()
        window_norm = (window - w_min) / (w_max - w_min + 1e-10)

        dot = float(np.dot(recent_norm, window_norm))
        norm_r = float(np.linalg.norm(recent_norm))
        norm_w = float(np.linalg.norm(window_norm))
        similarity = dot / (norm_r * norm_w + 1e-10)

        entry_price = closes[i + lookback - 1]
        exit_price = closes[i + lookback + 4]
        future_return = (exit_price - entry_price) / (entry_price + 1e-10) * 100

        idx = df.index[i + lookback - 1]
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]

        results.append({
            "date": date_str,
            "similarity": round(similarity, 4),
            "actual_return_5d": round(float(future_return), 2),
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
