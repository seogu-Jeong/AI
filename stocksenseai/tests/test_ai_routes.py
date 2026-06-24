import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_MOCK_SIGNAL = {
    "code": "005930",
    "signal": "BUY",
    "signal_score": 72.4,
    "signal_breakdown": {
        "technical_score": 68.0,
        "lstm_score": 75.0,
        "technical_weight": 0.4,
        "lstm_weight": 0.6,
    },
    "lstm_available": True,
    "as_of": "2026-06-04T15:30:00+09:00",
}

_MOCK_PREDICT = {
    "code": "005930",
    "current_price": 73400,
    "prediction": {
        "bullish": [74200, 75100, 75800, 76200, 77000],
        "base": [73800, 74200, 74100, 74500, 74900],
        "bearish": [73100, 72800, 72500, 72000, 71800],
    },
    "lstm_available": True,
}

_MOCK_INDICATORS = {
    "rsi_14": 58.4,
    "macd": 142.3,
    "macd_hist": 43.6,
    "bb_upper": 75200.0,
    "bb_lower": 71600.0,
    "ma5": 73100.0,
    "ma20": 72300.0,
    "close": 73400.0,
}


async def test_signal_returns_200(client):
    with patch("api.routes.ai.ai_service.get_signal", new_callable=AsyncMock, return_value=_MOCK_SIGNAL):
        resp = await client.get("/ai/005930/signal")
    assert resp.status_code == 200
    data = resp.json()
    assert data["signal"] in ("BUY", "HOLD", "SELL")
    assert "signal_score" in data


async def test_signal_fallback_no_lstm(client):
    fallback = {**_MOCK_SIGNAL, "lstm_available": False}
    with patch("api.routes.ai.ai_service.get_signal", new_callable=AsyncMock, return_value=fallback):
        resp = await client.get("/ai/005930/signal")
    assert resp.status_code == 200
    assert resp.json()["lstm_available"] is False


async def test_predict_returns_200(client):
    with patch("api.routes.ai.ai_service.get_prediction", new_callable=AsyncMock, return_value=_MOCK_PREDICT):
        resp = await client.get("/ai/005930/predict")
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "lstm_available" in data


async def test_prediction_upload_rejects_invalid_key(client):
    with patch("api.routes.ai.settings.ML_UPLOAD_KEY", "correct-key"):
        resp = await client.post(
            "/ai/predictions/upload",
            headers={"X-Upload-Key": "wrong-key"},
            json={
                "predictions": [{
                    "code": "005930",
                    "current_price": 73000,
                    "bullish": [74000, 75000, 76000, 77000, 78000],
                    "base": [73500, 74000, 74500, 75000, 75500],
                    "bearish": [72000, 71500, 71000, 70500, 70000],
                    "confidence": 80,
                }]
            },
        )
    assert resp.status_code == 403


async def test_prediction_upload_validates_five_day_scenarios(client):
    with patch("api.routes.ai.settings.ML_UPLOAD_KEY", "correct-key"):
        resp = await client.post(
            "/ai/predictions/upload",
            headers={"X-Upload-Key": "correct-key"},
            json={
                "predictions": [{
                    "code": "005930",
                    "current_price": 73000,
                    "bullish": [74000],
                    "base": [73500],
                    "bearish": [72000],
                    "confidence": 80,
                }]
            },
        )
    assert resp.status_code == 422


async def test_prediction_upload_stores_rows_and_clears_cache(client):
    mock_redis = AsyncMock()
    payload = {
        "predictions": [{
            "code": "005930",
            "current_price": 73000,
            "bullish": [74000, 75000, 76000, 77000, 78000],
            "base": [73500, 74000, 74500, 75000, 75500],
            "bearish": [72000, 71500, 71000, 70500, 70000],
            "confidence": 80,
        }]
    }
    with patch("api.routes.ai.settings.ML_UPLOAD_KEY", "correct-key"), \
         patch("api.routes.ai.get_redis", return_value=mock_redis):
        resp = await client.post(
            "/ai/predictions/upload",
            headers={"X-Upload-Key": "correct-key"},
            json=payload,
        )
    assert resp.status_code == 200
    assert resp.json()["uploaded"] == 1
    mock_redis.delete.assert_awaited_once_with("ai_predict:005930")


async def test_indicators_returns_200(client):
    with patch("api.routes.ai.ai_service.get_indicators", new_callable=AsyncMock, return_value=_MOCK_INDICATORS):
        resp = await client.get("/ai/005930/indicators")
    assert resp.status_code == 200
    assert "rsi_14" in resp.json()


async def test_patterns_returns_200(client):
    mock_patterns = [{"name": "hammer", "direction": "bullish", "value": 100}]
    with patch("api.routes.ai.pattern_service.detect_patterns", return_value=mock_patterns):
        with patch("api.routes.ai.get_ohlcv_cached", new_callable=AsyncMock, return_value=[]):
            resp = await client.get("/ai/005930/patterns")
    assert resp.status_code == 200
    assert "patterns" in resp.json()


async def test_similar_returns_200(client):
    mock_similar = {"code": "005930", "similar": []}
    with patch("api.routes.ai.ai_service.get_similar", new_callable=AsyncMock, return_value=mock_similar):
        resp = await client.get("/ai/005930/similar")
    assert resp.status_code == 200
    assert "similar" in resp.json()


async def test_multiframe_returns_200(client):
    mock_mf = {"code": "005930", "timeframes": {"daily": {"signal": "BUY", "score": 67.0}}}
    with patch("api.routes.ai.ai_service.get_multiframe", new_callable=AsyncMock, return_value=mock_mf):
        resp = await client.get("/ai/005930/multiframe")
    assert resp.status_code == 200
    assert "timeframes" in resp.json()


async def test_top_picks_returns_200(client):
    mock_picks = {"picks": [{"code": "005930", "signal_score": 72.4}]}
    with patch("api.routes.ai.ai_service.get_top_picks", new_callable=AsyncMock, return_value=mock_picks):
        resp = await client.get("/ai/top-picks")
    assert resp.status_code == 200
    assert "picks" in resp.json()


async def test_signals_history_returns_200(client):
    resp = await client.get("/ai/signals/history/005930")
    assert resp.status_code == 200
    data = resp.json()
    assert "history" in data
    assert isinstance(data["history"], list)


async def test_signal_rate_limit(client):
    with patch("api.routes.ai.ai_service.get_signal", new_callable=AsyncMock, return_value=_MOCK_SIGNAL):
        for _ in range(20):
            r = await client.get("/ai/005930/signal")
            assert r.status_code == 200
        r = await client.get("/ai/005930/signal")
        assert r.status_code == 429
