"""재무 평가 점수 로직 단위 테스트 (순수 함수 — 네트워크/DB 불필요)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from services.fundamental_service import RISK_THRESHOLD, score_financials


def test_저평가_우량주는_높은_점수():
    result = score_financials({"per": 9.0, "pbr": 0.9, "eps": 5000, "bps": 70000, "dividend_yield": 3.5})
    assert result["score"] >= 4.0
    assert result["risk"] is False
    assert result["grade"] in ("우수", "양호")


def test_고평가주는_위험으로_분류():
    result = score_financials({"per": 26.0, "pbr": 4.5, "eps": 12000, "bps": 71000, "dividend_yield": None})
    assert result["score"] < RISK_THRESHOLD
    assert result["risk"] is True
    assert result["grade"] == "위험"


def test_적자기업은_무조건_위험():
    result = score_financials({"per": None, "pbr": 1.2, "eps": -500, "bps": 30000, "dividend_yield": 0})
    assert result["risk"] is True
    assert result["grade"] == "위험"


def test_점수는_5점만점_소수1자리():
    result = score_financials({"per": 12.0, "pbr": 1.5, "eps": 5000, "bps": 70000, "dividend_yield": 1.0})
    assert 0.0 <= result["score"] <= 5.0
    # 소수 1자리로 반올림되어 있어야 한다.
    assert round(result["score"], 1) == result["score"]
    assert result["max_score"] == 5.0
