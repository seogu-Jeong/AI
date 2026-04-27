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
from models.lstm_model import LSTMModel
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

class LSTMDataset(Dataset):
    def __init__(self, sequences, labels_5d, labels_20d, labels_60d):
        self.sequences = torch.FloatTensor(sequences)
        self.labels_5d = torch.LongTensor(labels_5d)
        self.labels_20d = torch.LongTensor(labels_20d)
        self.labels_60d = torch.LongTensor(labels_60d)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels_5d[idx], self.labels_20d[idx], self.labels_60d[idx]

def train_lstm():
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

    # Load data
    db = SessionLocal()
    fs = FeatureStore()
    
    # Get all unique tickers from DB (up to args.tickers)
    from db.models import RawOHLCV
    from sqlalchemy import select
    ticker_stmt = select(RawOHLCV.ticker).distinct().limit(args.tickers)
    tickers = db.execute(ticker_stmt).scalars().all()
    
    data = None
    if tickers:
        print(f"Loading data for {len(tickers)} tickers...")
        data = fs.get_features_for_training(tickers, db)
    db.close()

    if data is None or len(data.get('lstm_sequences', [])) == 0:
        print('[WARNING] Using mock data — run seed_data.py first')
        n_samples = 200
        sequences = np.random.randn(n_samples, 30, 10).astype(np.float32)
        l5 = np.random.randint(0, 3, n_samples)
        l20 = np.random.randint(0, 3, n_samples)
        l60 = np.random.randint(0, 3, n_samples)
    else:
        sequences = data['lstm_sequences'].astype(np.float32)
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        
        l5 = pd.Series(data['labels']['direction_5d']).fillna(1).astype(int).values
        l20 = pd.Series(data['labels']['direction_20d']).fillna(1).astype(int).values
        l60 = pd.Series(data['labels']['direction_60d']).fillna(1).astype(int).values

    dataset = LSTMDataset(sequences, l5, l20, l60)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = LSTMModel(input_size=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    early_stopping = EarlyStopping(patience=7)
    os.makedirs("weights", exist_ok=True)

    epochs_run = 0
    for epoch in range(args.epochs):
        epochs_run += 1
        model.train()
        train_loss = 0
        for seq, y5, y20, y60 in train_loader:
            seq, y5, y20, y60 = seq.to(device), y5.to(device), y20.to(device), y60.to(device)
            
            optimizer.zero_grad()
            p5, p20, p60 = model(seq)
            
            loss = criterion(p5, y5) + criterion(p20, y20) + criterion(p60, y60)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct5, correct20, correct60 = 0, 0, 0
        with torch.no_grad():
            for seq, y5, y20, y60 in val_loader:
                seq, y5, y20, y60 = seq.to(device), y5.to(device), y20.to(device), y60.to(device)
                p5, p20, p60 = model(seq)
                
                loss = criterion(p5, y5) + criterion(p20, y20) + criterion(p60, y60)
                val_loss += loss.item()
                
                correct5 += (p5.argmax(1) == y5).sum().item()
                correct20 += (p20.argmax(1) == y20).sum().item()
                correct60 += (p60.argmax(1) == y60).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        acc5 = correct5 / len(val_dataset)
        acc20 = correct20 / len(val_dataset)
        acc60 = correct60 / len(val_dataset)
        val_acc = (acc5 + acc20 + acc60) / 3

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < early_stopping.best_loss:
            torch.save(model.state_dict(), "weights/lstm_best.pt")
        
        if early_stopping(avg_val_loss):
            print("Early stopping!")
            break

    # Evaluation metrics
    model.load_state_dict(torch.load("weights/lstm_best.pt"))
    model.eval()
    correct5, correct20, correct60 = 0, 0, 0
    with torch.no_grad():
        for seq, y5, y20, y60 in val_loader:
            seq, y5, y20, y60 = seq.to(device), y5.to(device), y20.to(device), y60.to(device)
            p5, p20, p60 = model(seq)
            correct5 += (p5.argmax(1) == y5).sum().item()
            correct20 += (p20.argmax(1) == y20).sum().item()
            correct60 += (p60.argmax(1) == y60).sum().item()
    
    final_acc5 = correct5 / len(val_dataset)
    final_acc20 = correct20 / len(val_dataset)
    final_acc60 = correct60 / len(val_dataset)
    final_val_acc = (final_acc5 + final_acc20 + final_acc60) / 3
    
    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy 5d: {final_acc5:.4f}")
    print(f"Accuracy 20d: {final_acc20:.4f}")
    print(f"Accuracy 60d: {final_acc60:.4f}")
    print(f"Combined Val Accuracy: {final_val_acc:.4f}")

    # Save metrics.json
    metrics = {
        'model': 'lstm',
        'val_accuracy': round(final_val_acc, 4),
        'val_acc_5d': round(final_acc5, 4),
        'val_acc_20d': round(final_acc20, 4),
        'val_acc_60d': round(final_acc60, 4),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d"),
        'epochs_trained': epochs_run
    }
    with open("weights/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_lstm()
