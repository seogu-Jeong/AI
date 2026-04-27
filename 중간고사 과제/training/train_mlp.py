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
from models.mlp_model import MLPModel
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
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

class MLPDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_mlp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        args.epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    db = SessionLocal()
    fs = FeatureStore()
    
    from db.models import RawOHLCV
    from sqlalchemy import select
    ticker_stmt = select(RawOHLCV.ticker).distinct().limit(20)
    tickers = db.execute(ticker_stmt).scalars().all()
    
    X, Y = None, None
    if tickers:
        data = fs.get_features_for_training(tickers, db)
        if data and len(data.get('transformer_factors', [])) > 0:
            factors = pd.DataFrame(data['transformer_factors']).fillna(0).values.astype(np.float32)
            factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
            labels = pd.Series(data['labels']['signal']).fillna(1).astype(int).values # (N,)
            
            # Load other models to get their predictions for the 30-dim input
            lstm = LSTMModel().to(device)
            cnn = CNNModel().to(device)
            
            # Try to load weights
            if os.path.exists("weights/lstm_best.pt"):
                lstm.load_state_dict(torch.load("weights/lstm_best.pt", map_location=device))
            if os.path.exists("weights/cnn_best.pt"):
                cnn.load_state_dict(torch.load("weights/cnn_best.pt", map_location=device))
            
            lstm.eval()
            cnn.eval()
            
            N = len(factors)
            lstm_preds = np.zeros((N, 3))
            cnn_preds = np.zeros((N, 3))
            
            with torch.no_grad():
                seq_data = data.get('lstm_sequences', [])
                if len(seq_data) > 0:
                    sequences = torch.FloatTensor(seq_data).to(device)
                    p5, _, _ = lstm(sequences)
                    lstm_preds[:len(p5)] = p5.cpu().numpy()
            
            X = np.hstack([factors, lstm_preds, cnn_preds])
            Y = labels

    if X is None:
        print('[WARNING] Using mock data — run seed_data.py first')
        X = np.random.randn(200, 30).astype(np.float32)
        Y = np.random.randint(0, 3, 200)
    
    db.close()

    dataset = MLPDataset(X, Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0:
        train_dataset, val_dataset = dataset, dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = MLPModel(input_size=30).to(device)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
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
            probs = model(x)
            
            loss_ce = ce_loss(probs, y)
            y_onehot = torch.zeros_like(probs).scatter_(1, y.unsqueeze(1), 1)
            loss_mse = mse_loss(probs, y_onehot)
            
            loss = 0.7 * loss_ce + 0.3 * loss_mse
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                probs = model(x)
                
                loss_ce = ce_loss(probs, y)
                y_onehot = torch.zeros_like(probs).scatter_(1, y.unsqueeze(1), 1)
                loss_mse = mse_loss(probs, y_onehot)
                loss = 0.7 * loss_ce + 0.3 * loss_mse
                
                val_loss += loss.item()
                correct += (probs.argmax(1) == y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = correct / len(val_dataset) if len(val_dataset) > 0 else 0

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < early_stopping.best_loss:
            torch.save(model.state_dict(), "weights/mlp_best.pt")

        if early_stopping(avg_val_loss):
            print("Early stopping!")
            break

    # Evaluation metrics
    model.load_state_dict(torch.load("weights/mlp_best.pt"))
    model.eval()
    all_preds = []
    final_val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            probs = model(x)
            loss_ce = ce_loss(probs, y)
            y_onehot = torch.zeros_like(probs).scatter_(1, y.unsqueeze(1), 1)
            loss_mse = mse_loss(probs, y_onehot)
            final_val_loss += (0.7 * loss_ce + 0.3 * loss_mse).item()
            all_preds.extend(probs.argmax(1).cpu().numpy())
    
    final_val_loss /= len(val_loader) if len(val_loader) > 0 else 1
    all_preds = np.array(all_preds)
    dist = {
        'Buy': int((all_preds == 0).sum()),
        'Hold': int((all_preds == 1).sum()),
        'Sell': int((all_preds == 2).sum())
    }
    total = len(all_preds) if len(all_preds) > 0 else 1
    dist_pct = {k: round(v/total, 4) for k, v in dist.items()}
    
    print("\nFinal Evaluation Metrics:")
    print(f"Class Distribution: {dist}")
    print(f"Class Distribution (%): {dist_pct}")
    print(f"Combined Val Loss: {final_val_loss:.4f}")

    # Save metrics.json
    metrics = {
        'model': 'mlp',
        'val_accuracy': round(float((all_preds == np.array([y for _, y in val_dataset])).mean() if len(val_dataset)>0 else 0), 4),
        'class_distribution': dist_pct,
        'combined_loss': round(final_val_loss, 4),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d"),
        'epochs_trained': epochs_run
    }
    with open("weights/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_mlp()
