import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # 3 separate heads for 5d, 20d, 60d prediction
        self.head_5d = nn.Linear(64, 3)
        self.head_20d = nn.Linear(64, 3)
        self.head_60d = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_out = lstm_out[:, -1, :]
        out = self.bn(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        pred_5d = self.softmax(self.head_5d(out))
        pred_20d = self.softmax(self.head_20d(out))
        pred_60d = self.softmax(self.head_60d(out))
        
        return pred_5d, pred_20d, pred_60d

    def predict(self, x: np.ndarray, device='cpu') -> dict:
        self.eval()
        with torch.no_grad():
            if x.ndim == 2:
                x = np.expand_dims(x, axis=0)
            tensor_x = torch.FloatTensor(x).to(device)
            p5, p20, p60 = self.forward(tensor_x)
            
            p5, p20, p60 = p5.cpu().numpy()[0], p20.cpu().numpy()[0], p60.cpu().numpy()[0]
            
            directions = ["UP", "FLAT", "DOWN"]
            return {
                'up_5d': float(p5[0]),
                'up_20d': float(p20[0]),
                'up_60d': float(p60[0]),
                'dir_5d': directions[np.argmax(p5)],
                'dir_20d': directions[np.argmax(p20)],
                'dir_60d': directions[np.argmax(p60)]
            }
