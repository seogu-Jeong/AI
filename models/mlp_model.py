import torch
import torch.nn as nn
import numpy as np

class MLPModel(nn.Module):
    def __init__(self, input_size=30):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 3)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, 30)
        logits = self.net(x)
        return self.softmax(logits)

    def predict(self, features: np.ndarray, device='cpu') -> dict:
        self.eval()
        with torch.no_grad():
            if features.ndim == 1:
                features = np.expand_dims(features, axis=0)
            
            tensor_x = torch.FloatTensor(features).to(device)
            out = self.forward(tensor_x)
            out = out.cpu().numpy()[0]
            
            signals = ["Buy", "Hold", "Sell"]
            return {
                'buy': float(out[0]),
                'hold': float(out[1]),
                'sell': float(out[2]),
                'signal': signals[np.argmax(out)]
            }
