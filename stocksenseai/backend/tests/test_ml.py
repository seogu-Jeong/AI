import numpy as np
import pandas as pd
import pytest
import torch

from ml.features import FEATURE_COLS, SEQ_LEN, build_features, fit_scaler
from ml.model import StockLSTM
import ml.predict as predict_module


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    })


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 13


def test_build_features_columns():
    df = _make_ohlcv(200)
    result = build_features(df)
    assert list(result.columns) == FEATURE_COLS


def test_build_features_no_nan():
    df = _make_ohlcv(200)
    result = build_features(df)
    assert not result.isnull().any().any()


def test_build_features_short_df_empty():
    df = _make_ohlcv(10)
    result = build_features(df)
    assert result.empty


def test_fit_scaler_range():
    df = _make_ohlcv(200)
    feat = build_features(df)
    scaler = fit_scaler(feat)
    scaled = scaler.transform(feat.values)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0


def test_lstm_forward_output_shape():
    model = StockLSTM()
    x = torch.randn(4, SEQ_LEN, 13)
    out = model(x)
    assert out.shape == (4, 5)


def test_lstm_train_mode_dropout_varies():
    model = StockLSTM()
    model.train()
    x = torch.randn(1, SEQ_LEN, 13)
    with torch.no_grad():
        out1 = model(x).numpy()
        out2 = model(x).numpy()
    assert not (out1 == out2).all()


def test_predict_no_model_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "WEIGHTS_DIR", tmp_path)
    df = _make_ohlcv(200)
    result = predict_module.predict_scenarios("999999", df)
    assert result is None


def test_predict_scenarios_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "WEIGHTS_DIR", tmp_path)
    df = _make_ohlcv(200)
    feat_df = build_features(df)
    scaler = fit_scaler(feat_df)
    model = StockLSTM()
    torch.save({"model_state_dict": model.state_dict(), "scaler": scaler}, tmp_path / "005930.pth")
    result = predict_module.predict_scenarios("005930", df)
    assert result is not None
    assert set(result.keys()) == {"bullish", "base", "bearish"}
    assert len(result["base"]) == 5


def test_get_lstm_direction_range(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "WEIGHTS_DIR", tmp_path)
    df = _make_ohlcv(200)
    feat_df = build_features(df)
    scaler = fit_scaler(feat_df)
    model = StockLSTM()
    torch.save({"model_state_dict": model.state_dict(), "scaler": scaler}, tmp_path / "005930.pth")
    direction = predict_module.get_lstm_direction("005930", df)
    assert direction is not None
    assert -1.0 <= direction <= 1.0


from ml.pattern_matcher import find_similar_patterns


def test_find_similar_returns_list():
    df = _make_ohlcv(200)
    result = find_similar_patterns(df)
    assert isinstance(result, list)


def test_find_similar_top_k():
    df = _make_ohlcv(200)
    result = find_similar_patterns(df, top_k=5)
    assert len(result) <= 5


def test_find_similar_structure():
    df = _make_ohlcv(200)
    result = find_similar_patterns(df, top_k=3)
    for item in result:
        assert "date" in item
        assert "similarity" in item
        assert "actual_return_5d" in item
        assert -1.0 <= item["similarity"] <= 1.0


def test_find_similar_short_df():
    df = _make_ohlcv(10)
    result = find_similar_patterns(df)
    assert result == []
