from pathlib import Path

import numpy as np

from ml.features import FEATURE_COLS, SEQ_LEN, build_features

WEIGHTS_DIR = Path(__file__).parent / "weights"


def load_model(code: str):
    """Returns (StockLSTM, MinMaxScaler) or None if weights not found."""
    path = WEIGHTS_DIR / f"{code}.pth"
    if not path.exists():
        return None
    try:
        import torch
        from ml.model import StockLSTM
    except ImportError:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = StockLSTM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["scaler"]


def predict_scenarios(code: str, df) -> dict | None:
    """
    Returns {bullish, base, bearish} — 5-day price predictions — or None if no model.
    Uses Monte Carlo Dropout (50 passes) for uncertainty estimation.
    """
    result = load_model(code)
    if result is None:
        return None
    model, scaler = result
    import torch

    feat_df = build_features(df)
    if len(feat_df) < SEQ_LEN:
        return None

    x = scaler.transform(feat_df[FEATURE_COLS].values[-SEQ_LEN:])
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 60, 13)

    model.train()  # enable MC Dropout
    preds = []
    with torch.no_grad():
        for _ in range(50):
            preds.append(model(x_tensor).cpu().numpy())

    preds = np.array(preds).squeeze()  # (50, 5) — change rates
    base = np.median(preds, axis=0)
    bullish = np.percentile(preds, 75, axis=0)
    bearish = np.percentile(preds, 25, axis=0)

    current = float(df["close"].iloc[-1])
    return {
        "bullish": [round(current * (1 + r)) for r in bullish.tolist()],
        "base": [round(current * (1 + r)) for r in base.tolist()],
        "bearish": [round(current * (1 + r)) for r in bearish.tolist()],
    }


def get_lstm_direction(code: str, df) -> float | None:
    """
    Returns LSTM-based direction in [-1.0, 1.0] (positive = bullish) or None if no model.
    Derived from 5-day base prediction vs current price.
    """
    scenarios = predict_scenarios(code, df)
    if scenarios is None:
        return None
    current = float(df["close"].iloc[-1])
    if current == 0:
        return 0.0
    change = (scenarios["base"][-1] - current) / current
    return float(np.clip(change * 10, -1.0, 1.0))
