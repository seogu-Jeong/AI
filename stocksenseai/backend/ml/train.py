"""
오프라인 LSTM 학습 스크립트.

Usage:
    python -m ml.train --codes 005930,000660
    python -m ml.train --codes-file ml/top100_codes.txt
    python -m ml.train  # uses top100_codes.txt by default
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.features import FEATURE_COLS, SEQ_LEN, build_features, fit_scaler
from ml.model import StockLSTM

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


def _fetch_ohlcv(code: str, years: int = 2):
    from pykrx import stock as pykrx_stock

    end = datetime.now()
    start = end - timedelta(days=years * 365)
    try:
        df = pykrx_stock.get_market_ohlcv_by_date(
            start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code
        )
    except Exception as e:
        print(f"[{code}] 데이터 조회 실패: {e}")
        return None
    if df is None or df.empty:
        return None
    df = df.rename(
        columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
    )
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _make_sequences(feat_df, n_train_rows: int):
    """n_train_rows: scaler fit에 사용할 train 구간 행 수 (data leakage 방지)."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(feat_df[FEATURE_COLS].values[:n_train_rows])
    data = scaler.transform(feat_df[FEATURE_COLS].values)
    raw_close = feat_df["close"].values

    X, y = [], []
    for i in range(len(data) - SEQ_LEN - 5):
        X.append(data[i : i + SEQ_LEN])
        future = raw_close[i + SEQ_LEN : i + SEQ_LEN + 5]
        base = raw_close[i + SEQ_LEN - 1]
        y.append((future - base) / (base + 1e-8))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler


def train_one(code: str) -> bool:
    print(f"[{code}] 데이터 다운로드 중...")
    df = _fetch_ohlcv(code, years=2)
    if df is None:
        return False

    feat_df = build_features(df)
    if len(feat_df) < SEQ_LEN + 20:
        print(f"[{code}] 데이터 부족 ({len(feat_df)} rows)")
        return False

    n_train_rows = int(len(feat_df) * 0.70)
    X, y, scaler = _make_sequences(feat_df, n_train_rows)
    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)

    X_train = torch.tensor(X[:n_train])
    y_train = torch.tensor(y[:n_train])
    X_val = torch.tensor(X[n_train:n_val])
    y_val = torch.tensor(y[n_train:n_val])

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    model = StockLSTM()
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_loss = float("inf")
    patience, no_improve = 10, 0

    for epoch in range(100):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "scaler": scaler},
                WEIGHTS_DIR / f"{code}.pth",
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{code}] Early stopping at epoch {epoch+1}")
                break

    print(f"[{code}] 완료 — best_val_loss={best_val_loss:.6f}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", help="쉼표 구분 종목 코드 (예: 005930,000660)")
    parser.add_argument("--codes-file", help="종목 코드 파일 (줄당 1개)")
    args = parser.parse_args()

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",")]
    elif args.codes_file:
        codes = Path(args.codes_file).read_text().strip().splitlines()
    else:
        codes_file = Path(__file__).parent / "top100_codes.txt"
        codes = codes_file.read_text().strip().splitlines()

    print(f"총 {len(codes)}개 종목 학습 시작")
    success = sum(1 for code in codes if train_one(code))
    print(f"완료: {success}/{len(codes)}")


if __name__ == "__main__":
    main()
