import pytest
from services.ai_service import _calc_tech_score, _score_to_signal


def test_score_to_signal_buy():
    assert _score_to_signal(65.0) == "BUY"
    assert _score_to_signal(80.0) == "BUY"


def test_score_to_signal_sell():
    assert _score_to_signal(35.0) == "SELL"
    assert _score_to_signal(10.0) == "SELL"


def test_score_to_signal_hold():
    assert _score_to_signal(50.0) == "HOLD"
    assert _score_to_signal(64.9) == "HOLD"
    assert _score_to_signal(35.1) == "HOLD"


def test_calc_tech_score_range():
    indicators = {
        "rsi_14": 60.0,
        "macd_hist": 100.0,
        "close": 73000.0,
        "bb_upper": 75000.0,
        "bb_lower": 71000.0,
    }
    score = _calc_tech_score(indicators)
    assert 0.0 <= score <= 100.0


def test_calc_tech_score_oversold_is_bullish():
    indicators = {
        "rsi_14": 25.0,
        "macd_hist": 50.0,
        "close": 71500.0,
        "bb_upper": 75000.0,
        "bb_lower": 71000.0,
    }
    score = _calc_tech_score(indicators)
    assert score > 50.0
