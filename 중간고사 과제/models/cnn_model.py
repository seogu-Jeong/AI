import torch
import torch.nn as nn
import numpy as np

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128, 64)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, 3, height, width)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu_fc(self.fc1(x)))
        x = self.softmax(self.fc2(x))
        return x

    def predict(self, image: np.ndarray, device='cpu') -> dict:
        self.eval()
        with torch.no_grad():
            image = np.array(image, dtype=np.float32)
            # Handle (B,C,H,W), (C,H,W), or (H,W,C) inputs
            if image.ndim == 4:
                # Already (B,C,H,W) - use as-is
                tensor_x = torch.FloatTensor(image / 255.0).to(device)
            elif image.ndim == 3:
                if image.shape[0] == 3:
                    # (C,H,W) - add batch dim
                    tensor_x = torch.FloatTensor((image / 255.0)[None]).to(device)
                else:
                    # (H,W,C) - transpose then add batch dim
                    tensor_x = torch.FloatTensor((image.transpose(2, 0, 1) / 255.0)[None]).to(device)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            out = self.forward(tensor_x)
            out = out.cpu().numpy()[0]
            
            patterns = ["Bullish", "Neutral", "Bearish"]
            return {
                'bullish': float(out[0]),
                'neutral': float(out[1]),
                'bearish': float(out[2]),
                'pattern': patterns[np.argmax(out)]
            }
