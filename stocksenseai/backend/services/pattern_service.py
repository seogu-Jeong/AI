import pandas as pd
import pandas_ta_classic as ta

_PATTERNS = [
    "hammer", "invertedhammer", "doji", "engulfing",
    "morningstar", "eveningstar", "shootingstar", "hangingman",
    "3whitesoldiers", "3blackcrows", "piercingline",
    "darkcloudcover", "harami", "haramicross",
]

_FIXED_DIRECTION: dict[str, str] = {
    "hammer": "bullish",
    "invertedhammer": "bullish",
    "morningstar": "bullish",
    "3whitesoldiers": "bullish",
    "piercingline": "bullish",
    "eveningstar": "bearish",
    "shootingstar": "bearish",
    "hangingman": "bearish",
    "3blackcrows": "bearish",
    "darkcloudcover": "bearish",
}


def detect_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Input:  DataFrame with [open, high, low, close] (lowercase)
    Output: [{name, direction, value}] for detected patterns (value != 0)
    """
    if df.empty or len(df) < 3:
        return []

    results = []
    for pattern in _PATTERNS:
        try:
            series = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name=pattern)
            if series is None or series.empty:
                continue
            last_val = int(series.iloc[-1].item())
            if last_val == 0:
                continue
            direction = _FIXED_DIRECTION.get(pattern)
            if direction is None:
                direction = "bullish" if last_val > 0 else "bearish"
            results.append({"name": pattern, "direction": direction, "value": last_val})
        except Exception:
            continue

    return results
