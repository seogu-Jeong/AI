import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler

# 13 features: 5 OHLCV + 8 indicators
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi_14",
    "macd", "macd_hist",
    "bb_upper", "bb_lower",
    "ma5", "ma20",
    "stoch_k",
]
SEQ_LEN = 60


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with columns [open, high, low, close, volume] (lowercase)
    Output: DataFrame with exactly FEATURE_COLS, dropna applied
    """
    df = df.copy()

    df["rsi_14"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is None:
        df["macd"] = float("nan")
        df["macd_hist"] = float("nan")
    else:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is None:
        df["bb_upper"] = float("nan")
        df["bb_lower"] = float("nan")
    else:
        # pandas-ta column naming may vary by version (BBU_20_2.0 vs BBU_20_2.0_2.0)
        bb_upper_col = next(c for c in bb.columns if c.startswith("BBU_"))
        bb_lower_col = next(c for c in bb.columns if c.startswith("BBL_"))
        df["bb_upper"] = bb[bb_upper_col]
        df["bb_lower"] = bb[bb_lower_col]

    df["ma5"] = ta.sma(df["close"], length=5)
    df["ma20"] = ta.sma(df["close"], length=20)

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    if stoch is None:
        df["stoch_k"] = float("nan")
    else:
        df["stoch_k"] = stoch["STOCHk_14_3_3"]

    return df[FEATURE_COLS].dropna()


def fit_scaler(feat_df: pd.DataFrame) -> MinMaxScaler:
    """Fit MinMaxScaler on feature columns and return it."""
    scaler = MinMaxScaler()
    scaler.fit(feat_df[FEATURE_COLS].values)
    return scaler
