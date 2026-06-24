"""로컬 LSTM 가중치로 예측을 생성하고 배포 API에 업로드한다."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
import numpy as np

from ml.predict import predict_scenarios
from ml.train import _fetch_ohlcv


def _read_codes(codes: str | None, codes_file: str | None) -> list[str]:
    if codes:
        result = codes.split(",")
    else:
        path = Path(codes_file) if codes_file else Path(__file__).parent / "top100_codes.txt"
        result = path.read_text().splitlines()
    return [code.strip() for code in result if code.strip()]


def _confidence(scenarios: dict[str, list[float]]) -> float:
    base = np.asarray(scenarios["base"], dtype=float)
    bullish = np.asarray(scenarios["bullish"], dtype=float)
    bearish = np.asarray(scenarios["bearish"], dtype=float)
    spread = np.mean((bullish - bearish) / np.maximum(base, 1))
    return round(float(np.clip(100 - spread * 100, 0, 100)), 1)


def generate_one(code: str) -> dict | None:
    df = _fetch_ohlcv(code, years=2)
    if df is None or df.empty:
        print(f"[{code}] 시세 데이터 없음")
        return None

    scenarios = predict_scenarios(code, df)
    if scenarios is None:
        print(f"[{code}] 가중치 없음 또는 예측 실패")
        return None

    return {
        "code": code,
        "current_price": float(df["close"].iloc[-1]),
        **scenarios,
        "confidence": _confidence(scenarios),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LSTM 예측 생성 후 배포 API 업로드")
    parser.add_argument("--codes", help="쉼표 구분 종목 코드 (예: 005930,000660)")
    parser.add_argument("--codes-file", help="종목 코드 파일 (줄당 1개)")
    parser.add_argument(
        "--api-url",
        default=os.getenv("ML_UPLOAD_URL", "http://localhost:8000"),
        help="배포 API 기본 URL",
    )
    parser.add_argument("--upload-key", default=os.getenv("ML_UPLOAD_KEY"), help="업로드 인증 키")
    args = parser.parse_args()

    if not args.upload_key:
        print("ML_UPLOAD_KEY 또는 --upload-key가 필요합니다.", file=sys.stderr)
        return 2

    codes = _read_codes(args.codes, args.codes_file)
    predictions = []
    for code in codes:
        try:
            prediction = generate_one(code)
            if prediction:
                predictions.append(prediction)
        except Exception as exc:
            print(f"[{code}] 예측 생성 실패: {exc}", file=sys.stderr)

    if not predictions:
        print("업로드할 예측 결과가 없습니다.", file=sys.stderr)
        return 1

    url = f"{args.api_url.rstrip('/')}/ai/predictions/upload"
    try:
        response = httpx.post(
            url,
            headers={"X-Upload-Key": args.upload_key},
            json={"predictions": predictions},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"예측 업로드 실패: {exc}", file=sys.stderr)
        return 1

    print(f"예측 업로드 완료: {response.json().get('uploaded', 0)}/{len(codes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
