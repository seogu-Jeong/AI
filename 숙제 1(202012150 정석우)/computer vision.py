import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import ssl

# SSL 인증서 문제 해결
ssl._create_default_https_context = ssl._create_unverified_context

# 1. GPU 세팅
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. MNIST 데이터셋 다운로드 및 로드
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
trainset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫 번째 합성곱 & 풀링 레이어
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 두 번째 합성곱 & 풀링 레이어
        self.conv2 = nn.Conv2d(24, 36, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        # 입력 차원 계산 (28x28 -> 26x26 -> 13x13 -> 11x11 -> 5x5)
        self.fc1 = nn.Linear(36 * 5 * 5, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = CNN().to(device)

# 4. 손실 함수 및 옵티마이저
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=1e-2)

# 5. 모델 훈련 (Training Loop)
epochs = 5
results_log = []
print("훈련 시작!")

for epoch in range(epochs):
    cnn_model.train()
    total_loss, correct_pred, total_pred = 0, 0, 0

    for images, labels in tqdm(trainset_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad() # 기울기 초기화
        logits = cnn_model(images) # 순전파
        loss = loss_function(logits, labels) # 손실 계산
        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트
        
        total_loss += loss.item() * images.size(0)
        predicted = torch.argmax(logits, dim=1)
        correct_pred += (predicted == labels).sum().item()
        total_pred += labels.size(0)

    epoch_loss = total_loss / total_pred
    epoch_acc = correct_pred / total_pred
    log_msg = f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
    print(log_msg)
    results_log.append(log_msg)

# 6. 테스트 데이터 평가 (Evaluation)
cnn_model.eval()
test_loss, correct_pred, total_pred = 0, 0, 0
with torch.no_grad():
    for images, labels in testset_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_model(images)
        loss = loss_function(outputs, labels)
        
        test_loss += loss.item() * images.size(0)
        predicted = torch.argmax(outputs, dim=1)
        correct_pred += (predicted == labels).sum().item()
        total_pred += labels.size(0)

final_accuracy = correct_pred / total_pred
final_msg = f"\n최종 테스트 정확도: {final_accuracy:.4f}"
print(final_msg)
results_log.append(final_msg)

# 결과 리포트 저장
with open("mnist_training_report.txt", "w") as f:
    f.write("\n".join(results_log))

# 모델 가중치 저장
torch.save(cnn_model.state_dict(), "mnist_cnn.pth")
print("\n결과 보고서와 모델 파일이 저장되었습니다.")
