import os
import argparse
import json
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from models.transformer_model import TransformerFactorModel
from data.feature_store import FeatureStore
from db.database import SessionLocal

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # don't stop
        self.counter += 1
        return self.counter >= self.patience

class TransformerDataset(Dataset):
    def __init__(self, factors, labels):
        self.factors = torch.FloatTensor(factors)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.factors)

    def __getitem__(self, idx):
        return self.factors[idx], self.labels[idx]

def train_transformer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        args.tickers = 10
        args.epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    db = SessionLocal()
    fs = FeatureStore()
    
    from db.models import RawOHLCV
    from sqlalchemy import select
    ticker_stmt = select(RawOHLCV.ticker).distinct().limit(args.tickers)
    tickers = db.execute(ticker_stmt).scalars().all()
    
    data = None
    if tickers:
        data = fs.get_features_for_training(tickers, db)
    db.close()

    if data is None or len(data.get('transformer_factors', [])) == 0:
        print('[WARNING] Using mock data — run seed_data.py first')
        n_samples = 200
        factors = np.random.randn(n_samples, 24).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples).astype(float)
    else:
        factors = pd.DataFrame(data['transformer_factors']).fillna(0).values.astype(np.float32)
        factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
        # Target: direction_5d == 0 (UP)
        labels = (data['labels']['direction_5d'] == 0).astype(float)

    dataset = TransformerDataset(factors, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = TransformerFactorModel(n_factors=24).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    early_stopping = EarlyStopping(patience=7)
    os.makedirs("weights", exist_ok=True)

    epochs_run = 0
    for epoch in range(args.epochs):
        epochs_run += 1
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prob, _ = model(x)
            loss = criterion(prob, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                prob, _ = model(x)
                loss = criterion(prob, y)
                val_loss += loss.item()
                preds = (prob > 0.5).float()
                correct += (preds == y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_dataset)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < early_stopping.best_loss:
            torch.save(model.state_dict(), "weights/transformer_best.pt")
        
        if early_stopping(avg_val_loss):
            print("Early stopping!")
            break

    # Evaluation metrics
    model.load_state_dict(torch.load("weights/transformer_best.pt"))
    model.eval()
    correct = 0
    attn_sums = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            prob, attn = model(x)
            preds = (prob > 0.5).float()
            correct += (preds == y).sum().item()
            # attn shape: [batch, n_factors, 1] or similar. Check models/transformer_model.py
            # If it's from Softmax, it should sum to 1.0 along factors dim.
            if attn is not None:
                # Assuming attn has shape (B, N, 1) or (B, N)
                s = attn.sum(dim=1).cpu().numpy()
                attn_sums.extend(s.flatten())
    
    final_val_acc = correct / len(val_dataset)
    avg_attn_sum = np.mean(attn_sums) if attn_sums else 0.0
    
    print("\nFinal Evaluation Metrics:")
    print(f"Binary Accuracy: {final_val_acc:.4f}")
    print(f"Average Attention Weight Sum: {avg_attn_sum:.4f}")

    # Save metrics.json
    metrics = {
        'model': 'transformer',
        'val_accuracy': round(final_val_acc, 4),
        'avg_attn_sum': round(float(avg_attn_sum), 4),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d"),
        'epochs_trained': epochs_run
    }
    with open("weights/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_transformer()
