import numpy as np
import pandas as pd
import pytest

from services.pattern_service import detect_patterns


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 200)
    return pd.DataFrame({
        "open": close * 0.995,
        "high": close * 1.01,
        "low": close * 0.985,
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    })


def test_detect_patterns_returns_list():
    df = _make_ohlcv()
    result = detect_patterns(df)
    assert isinstance(result, list)


def test_detect_patterns_structure():
    df = _make_ohlcv()
    result = detect_patterns(df)
    for item in result:
        assert "name" in item
        assert "direction" in item
        assert item["direction"] in ("bullish", "bearish")
        assert "value" in item
        assert item["value"] != 0


def test_detect_patterns_empty_df():
    df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    result = detect_patterns(df)
    assert result == []


def test_detect_patterns_short_df_no_crash():
    df = _make_ohlcv(3)
    result = detect_patterns(df)
    assert isinstance(result, list)
