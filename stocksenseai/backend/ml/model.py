import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Input:  (batch, seq_len=60, features=13)
    Output: (batch, 5) — 다음 5 거래일 종가 변화율 예측
    """

    def __init__(
        self,
        input_size: int = 13,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)            # (batch, seq, hidden)
        last_10 = lstm_out[:, -10:, :]        # (batch, 10, hidden)
        attn_out, _ = self.attention(last_10, last_10, last_10)
        out = attn_out[:, -1, :]              # (batch, hidden)
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)                  # (batch, 5)
