import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskVisionSystem:
    """
    분류, 회귀, 세그멘테이션을 아우르는 통합 비전 모델 모듈입니다.
    """
    
    # 1. 이미지 분류기 (Image Classifier)
    class Classifier(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return F.log_softmax(x, dim=1)

    # 2. 자율주행 조향각 회귀 모델 (Steering Regressor)
    class SteeringModel(nn.Module):
        def __init__(self):
            super().__init__()
            # NVIDIA Dave-2 아키텍처 영감
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 24, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(24, 36, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(36, 48, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(48, 64, 3),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3),
                nn.ReLU(),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 1 * 18, 100), # 입력 해상도에 따라 가변적
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.Linear(50, 1) # 최종 조향각 (단일값)
            )

        def forward(self, x):
            x = self.backbone(x)
            x = self.regressor(x)
            return x

    # 3. 시맨틱 세그멘테이션 (Simple UNet Style)
    class Segmenter(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            # Encoder
            self.enc1 = self.conv_block(3, 64)
            self.pool = nn.MaxPool2d(2)
            
            # Middle
            self.middle = self.conv_block(64, 128)
            
            # Decoder (Upsampling)
            self.up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec1 = self.conv_block(64, num_classes)

        def conv_block(self, in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x1 = self.enc1(x)
            x_pool = self.pool(x1)
            x_mid = self.middle(x_pool)
            
            x_up = self.up(x_mid)
            # 여기서는 단순화를 위해 skip connection 생략
            out = self.dec1(x_up)
            return out

def run_test():
    print("--- PyTorch Vision Models Shape Test ---")
    
    # 가상 데이터 (Batch=1, RGB=3, H=64, W=64)
    dummy_input = torch.randn(1, 3, 64, 64)
    
    # 1. Classification
    clf = MultiTaskVisionSystem.Classifier(num_classes=10)
    out_clf = clf(dummy_input)
    print(f"Classification Output Shape: {out_clf.shape} (Expected: [1, 10])")
    
    # 2. Regression (Autonomous Driving)
    # 조향각 모델은 특유의 해상도(66x200)가 있으나 여기서는 64x64로 테스트
    # 실제로는 Linear Layer 크기 에러가 날 수 있으므로 dummy를 맞춰줌
    reg = MultiTaskVisionSystem.SteeringModel()
    try:
        # SteeringModel의 Linear 파라미터를 동적으로 맞추지 않았으므로 에러 처리 시연
        # 실제 레이어 계산 과정을 거쳐야 함
        out_reg = reg(torch.randn(1, 3, 66, 200))
        print(f"Regression Output Value: {out_reg.item():.4f}")
    except Exception as e:
        print(f"Regression model requires specific input size (66x200)")

    # 3. Segmentation
    seg = MultiTaskVisionSystem.Segmenter(num_classes=5)
    out_seg = seg(dummy_input)
    print(f"Segmentation Output Shape: {out_seg.shape} (Expected: [1, 5, 64, 64])")

if __name__ == "__main__":
    run_test()
