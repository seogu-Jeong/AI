export interface Topic {
  slug: string
  icon: string
  title: string
  subtitle: string
  tags: string[]
  isFree: boolean
  description: string
  content: {
    sections: { heading: string; body: string }[]
    codeFile: string
    code: string
    codeExplanations?: { line: string; title: string; explanation: string }[]
  }
}

export const TOPICS: Topic[] = [
  {
    slug: "regularization",
    icon: "🛡️",
    title: "Regularization (규제)",
    subtitle: "Topic 01",
    tags: ["L1", "L2", "Dropout", "Weight Decay"],
    isFree: true,
    description:
      "모델이 훈련 데이터에 과도하게 적합되는 것을 방지하는 핵심 기법들을 학습합니다. L1/L2 정규화와 Dropout의 수학적 원리를 이해합니다.",
    content: {
      sections: [
        {
          heading: "왜 Regularization이 필요한가?",
          body: "딥러닝 모델은 수백만 개의 파라미터를 가지기 때문에, 훈련 데이터에 지나치게 맞춰지는 과적합(Overfitting)이 자주 발생합니다. 예를 들어 훈련 정확도는 99%인데 테스트 정확도는 70%에 그치는 경우가 전형적입니다. Regularization은 손실 함수(Loss Function)에 '패널티 항(Penalty Term)'을 추가하여, 가중치가 지나치게 커지는 것을 억제합니다. 직관적으로는 '가중치에 세금을 부과해 불필요한 복잡성을 줄인다'고 이해하면 됩니다.",
        },
        {
          heading: "L1 vs L2 정규화: 수학적 차이",
          body: "L1(Lasso) 정규화는 Loss += λ·Σ|wᵢ| 형태로, 가중치 절댓값의 합을 패널티로 씁니다. 이 방식은 중요하지 않은 가중치를 정확히 0으로 만들어 희소(sparse) 모델을 생성합니다. 즉 특성 선택(Feature Selection) 효과가 있습니다. L2(Ridge/Weight Decay)는 Loss += λ·Σwᵢ² 형태로, 가중치 제곱합을 패널티로 씁니다. 모든 가중치를 골고루 0에 가깝게 만들지만 완전히 0으로 만들지는 않습니다. PyTorch에서는 optimizer의 weight_decay 파라미터로 L2를 간편하게 적용할 수 있어, 별도의 코드 없이 사용할 수 있습니다.",
        },
        {
          heading: "Dropout: 무작위 비활성화로 앙상블 효과",
          body: "Dropout은 훈련 중 각 뉴런을 확률 p로 무작위 비활성화(0으로 만들기)하는 기법입니다. 예를 들어 p=0.3이면 매 배치마다 30%의 뉴런이 꺼집니다. 이렇게 하면 각 뉴런이 특정 입력 패턴에 의존하지 않고 독립적인 특성을 학습하게 되어, 마치 여러 모델의 예측을 평균 내는 앙상블(Ensemble) 효과를 얻습니다. 주의할 점은 추론(inference) 시에는 모든 뉴런을 활성화해야 하며, 출력을 (1-p)로 스케일링하거나 훈련 시 1/(1-p)로 스케일 보정하는 inverted dropout 방식을 씁니다. PyTorch의 nn.Dropout은 이를 자동으로 처리합니다.",
        },
      ],
      codeFile: "01_regularization.py",
      code: `import torch
import torch.nn as nn


class RegularizedNet(nn.Module):
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),   # Dropout regularization
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


# L2 regularization via optimizer weight_decay
model = RegularizedNet(dropout_p=0.3)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,  # L2 penalty coefficient λ
)

# Training loop (simplified)
criterion = nn.CrossEntropyLoss()
model.train()
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()`,
      codeExplanations: [
        {
          line: "nn.BatchNorm1d(256)",
          title: "배치 정규화(Batch Normalization)",
          explanation:
            "각 레이어의 출력을 배치 단위로 정규화(평균 0, 분산 1)합니다. 내부 공변량 이동(Internal Covariate Shift) 문제를 해결해 더 높은 학습률 사용이 가능해지고, 자체적으로 약한 regularization 효과도 있습니다. Dropout과 함께 쓰면 일반적으로 Dropout을 BatchNorm 뒤에 배치합니다.",
        },
        {
          line: "nn.Dropout(p=dropout_p)",
          title: "Dropout 레이어",
          explanation:
            "훈련 중에는 p=0.3 확률로 각 뉴런 출력을 0으로 만들고, 나머지는 1/(1-p)로 스케일 업합니다(inverted dropout). model.eval() 호출 후에는 자동으로 비활성화되어 모든 뉴런을 사용합니다. 0.2~0.5 범위를 자주 사용하며, 너무 높으면 underfitting이 발생합니다.",
        },
        {
          line: "torch.optim.AdamW(..., weight_decay=1e-4)",
          title: "AdamW와 weight_decay",
          explanation:
            "AdamW는 기존 Adam에서 L2 정규화를 gradient에 추가하는 방식(Adam+L2)의 문제를 해결하고, weight_decay를 파라미터 업데이트에 직접 적용합니다. weight_decay=1e-4는 λ=0.0001의 L2 패널티를 의미합니다. 딥러닝에서는 Adam 대신 AdamW를 사용하는 것이 현재 모범 사례(best practice)입니다.",
        },
        {
          line: "optimizer.zero_grad()",
          title: "그래디언트 초기화",
          explanation:
            "PyTorch는 기본적으로 backward() 호출 시 그래디언트를 누적합니다. 매 배치 시작 전에 zero_grad()로 이전 배치의 그래디언트를 초기화하지 않으면 그래디언트가 계속 쌓여 잘못된 방향으로 업데이트됩니다. set_to_none=True 옵션을 주면 메모리를 더 효율적으로 사용합니다.",
        },
      ],
    },
  },
  {
    slug: "overfitting-underfitting",
    icon: "⚖️",
    title: "Overfitting vs Underfitting",
    subtitle: "Topic 02",
    tags: ["Bias-Variance", "Learning Curve", "Model Complexity"],
    isFree: true,
    description:
      "편향-분산 트레이드오프를 깊이 이해하고, 학습 곡선을 분석해 모델의 일반화 성능을 진단하는 방법을 배웁니다.",
    content: {
      sections: [
        {
          heading: "Bias-Variance Tradeoff: 핵심 딜레마",
          body: "모든 머신러닝 모델의 오류는 편향(Bias)과 분산(Variance) 두 부분으로 분해됩니다. 편향은 모델이 실제 패턴을 얼마나 단순화하는가를 나타냅니다. 너무 단순한 모델(예: 선형 모델로 비선형 데이터 학습)은 High Bias 상태로, 훈련 데이터와 테스트 데이터 모두 성능이 낮은 Underfitting이 됩니다. 분산은 데이터의 작은 변화에 모델이 얼마나 민감한가를 나타냅니다. 너무 복잡한 모델(파라미터 수 >> 데이터 수)은 High Variance 상태로, 훈련 성능은 높지만 테스트 성능이 크게 떨어지는 Overfitting이 됩니다. 최적의 모델은 두 오류의 합 Bias² + Variance를 최소화하는 복잡도에서 얻어집니다.",
        },
        {
          heading: "학습 곡선으로 상태 진단하기",
          body: "에폭(epoch)별 훈련 손실과 검증 손실을 그래프로 그리면 모델 상태를 즉시 파악할 수 있습니다. Underfitting의 징후: 훈련 손실이 높고(>0.3), 검증 손실도 비슷하게 높습니다. 두 곡선이 거의 겹쳐서 내려오다 높은 곳에서 수렴합니다. Overfitting의 징후: 훈련 손실은 계속 낮아지는데, 검증 손실은 특정 에폭 이후 다시 올라갑니다(validation loss가 U자형). 두 곡선 사이의 gap이 커집니다. 좋은 학습: 두 곡선이 함께 낮아지며 간격이 좁게 수렴합니다.",
        },
        {
          heading: "각 문제의 해결 전략",
          body: "Underfitting 해결: 모델 용량(capacity) 증가 — 레이어 추가, 뉴런 수 증가, 더 복잡한 아키텍처로 교체. 학습률(learning rate) 재조정, 에폭 수 증가, 과도한 Regularization 완화. Overfitting 해결: Regularization 강화(L1/L2/Dropout), 데이터 증강(Augmentation), 조기 종료(Early Stopping) — 검증 손실이 증가하기 시작하는 지점에서 학습을 멈춥니다. 또한 더 많은 훈련 데이터 수집이 가장 확실한 해법입니다. BatchNorm과 학습률 스케줄러(StepLR, CosineAnnealingLR)는 두 문제 모두에 안정적인 훈련을 도와줍니다.",
        },
      ],
      codeFile: "02_overfitting_underfitting.py",
      code: `import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str = "learning_curves.png",
) -> None:
    """학습 곡선으로 Overfitting/Underfitting 진단"""
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4, linewidth=2)
    ax.plot(epochs, val_losses,  "r-o", label="Val Loss",   markersize=4, linewidth=2)
    ax.fill_between(epochs, train_losses, val_losses, alpha=0.08, color="red")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Learning Curves — Bias-Variance Diagnosis", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # 자동 진단
    gap = val_losses[-1] - train_losses[-1]
    final_train = train_losses[-1]

    if final_train > 0.3 and gap < 0.05:
        print(f"⚠️  Underfitting (train={final_train:.3f}) — 모델 복잡도 증가 권장")
    elif gap > 0.15:
        print(f"⚠️  Overfitting (gap={gap:.3f}) — Regularization/Early Stopping 권장")
    else:
        print(f"✅  Good fit (train={final_train:.3f}, gap={gap:.3f})")`,
      codeExplanations: [
        {
          line: "ax.fill_between(epochs, train_losses, val_losses, ...)",
          title: "Generalization Gap 시각화",
          explanation:
            "두 손실 곡선 사이의 영역을 색으로 채웁니다. 이 gap이 넓을수록 Overfitting이 심각한 것입니다. alpha=0.08로 투명도를 낮게 설정해 두 곡선을 가리지 않으면서도 gap을 직관적으로 보여줍니다. 실제 연구 논문에서도 이 시각화 방식을 자주 사용합니다.",
        },
        {
          line: "gap = val_losses[-1] - train_losses[-1]",
          title: "마지막 에폭의 Generalization Gap",
          explanation:
            "훈련이 끝난 시점의 검증 손실과 훈련 손실 차이를 계산합니다. 이 값이 클수록 Overfitting이 심합니다. [-1]은 리스트의 마지막 원소를 가져오는 Python 문법입니다. 주의: 이 값만으로 판단하는 것은 부족하고, 곡선의 추세(trend)도 함께 보아야 합니다.",
        },
        {
          line: "if final_train > 0.3 and gap < 0.05:",
          title: "Underfitting 자동 감지 조건",
          explanation:
            "훈련 손실 자체가 높고(0.3 이상) gap이 작다면, 모델이 훈련 데이터조차 제대로 학습하지 못하는 Underfitting 상태입니다. 임계값 0.3과 0.05는 태스크와 데이터에 따라 조정해야 합니다. 이 자동 진단은 출발점을 제공할 뿐이며, 실제 판단은 곡선의 형태와 도메인 지식을 함께 활용해야 합니다.",
        },
      ],
    },
  },
  {
    slug: "data-augmentation",
    icon: "🖼️",
    title: "Data Augmentation",
    subtitle: "Topic 03",
    tags: ["Flip", "Crop", "Rotation", "Cutout", "Mixup"],
    isFree: false,
    description:
      "제한된 데이터로 모델 성능을 극대화하는 데이터 증강 전략들을 학습합니다. 최신 증강 기법인 Mixup과 CutMix도 다룹니다.",
    content: {
      sections: [
        {
          heading: "왜 Data Augmentation인가?",
          body: "딥러닝 모델은 데이터가 많을수록 성능이 올라가지만, 현실에서는 레이블링된 데이터를 모으는 비용이 매우 큽니다. Data Augmentation은 기존 데이터를 변형해 인위적으로 데이터 다양성을 늘리는 기법입니다. 예를 들어 고양이 사진을 좌우 반전하거나 약간 회전해도 '고양이'라는 사실은 변하지 않습니다. 이 불변성(invariance)을 모델에 주입하면, 같은 양의 데이터로도 훨씬 강건한 모델을 훈련할 수 있습니다. CIFAR-10 기준으로 적절한 augmentation만으로 정확도를 3~5%p 향상시킬 수 있습니다.",
        },
        {
          heading: "기본 증강 기법들",
          body: "RandomHorizontalFlip: 50% 확률로 이미지를 좌우 반전합니다. 자연 이미지에 효과적이지만, 텍스트나 의료 이미지 등 방향이 의미 있는 경우에는 사용하지 않습니다. RandomCrop: 이미지를 무작위로 잘라 다른 위치와 스케일을 학습합니다. 일반적으로 원본보다 약간 작게(예: 224→200 crop) 설정합니다. ColorJitter: 밝기, 대비, 채도, 색조를 무작위로 변환합니다. 조명 변화에 강건하게 만듭니다. RandomRotation: 이미지를 ±15° 정도 회전합니다. 너무 크면 정보 손실이 생깁니다. Normalize: 평균과 표준편차로 픽셀값을 정규화합니다. ImageNet 사전 학습 모델을 쓸 때는 ImageNet 통계값(mean=[0.485,0.456,0.406])을 사용해야 합니다.",
        },
        {
          heading: "고급 증강: Mixup과 CutMix",
          body: "Mixup은 두 이미지를 픽셀 단위로 선형 보간합니다: x_mix = λ·x₁ + (1-λ)·x₂. 레이블도 동일하게 섞어 soft label로 만듭니다. 모델이 카테고리 간의 연속적인 경계를 학습하게 해 일반화 성능을 높입니다. CutMix는 한 이미지의 직사각형 영역을 다른 이미지에서 잘라 붙이는 방식입니다. 레이블은 붙여 넣은 면적 비율에 따라 혼합합니다. 이 두 기법은 특히 ImageNet 및 CIFAR-100 등 대규모 분류 태스크에서 일관된 성능 향상을 보입니다. 두 기법 모두 lambda 파라미터를 Beta 분포 Beta(α, α)에서 샘플링하며, α=1이 기본값입니다.",
        },
      ],
      codeFile: "03_data_augmentation.py",
      code: `import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np


# ── 기본 증강 파이프라인 ──────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])


# ── Mixup 구현 ───────────────────────────────────────
def mixup_data(x: torch.Tensor, y: torch.Tensor,
               alpha: float = 1.0):
    """두 배치를 Beta(alpha, alpha) 비율로 혼합"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup용 손실 함수: 두 레이블을 lambda 비율로 혼합"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── 학습 루프에서 Mixup 적용 ─────────────────────────
criterion = torch.nn.CrossEntropyLoss()

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    images, labels_a, labels_b, lam = mixup_data(images, labels)

    outputs = model(images)
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`,
      codeExplanations: [
        {
          line: "transforms.RandomCrop(32, padding=4)",
          title: "RandomCrop with Padding",
          explanation:
            "먼저 상하좌우로 4픽셀씩 zero-padding을 추가해 40×40 이미지를 만든 뒤, 무작위로 32×32 영역을 잘라냅니다. 이렇게 하면 원본 이미지에서 최대 ±4픽셀 이동한 효과를 냅니다. CIFAR-10(32×32)에서 표준적으로 쓰이는 설정이며, 위치 이동에 대한 불변성을 모델에 주입합니다.",
        },
        {
          line: "transforms.Normalize(mean=[0.4914, ...], std=[0.2470, ...])",
          title: "데이터셋 통계 기반 정규화",
          explanation:
            "각 채널(R, G, B)의 픽셀값을 (픽셀 - 평균) / 표준편차로 변환합니다. 0.4914 등의 값은 CIFAR-10 전체 훈련 데이터로부터 계산된 채널별 평균입니다. 이 정규화는 각 채널의 분포를 평균 0 근처로 맞춰주어 그래디언트 계산이 안정되고 학습이 빨라집니다. 검증/테스트 때도 동일한 값을 사용해야 합니다.",
        },
        {
          line: "lam = np.random.beta(alpha, alpha)",
          title: "Beta 분포에서 혼합 비율 샘플링",
          explanation:
            "alpha=1이면 Beta(1,1) = Uniform(0,1)이 되어 두 이미지가 0~100% 비율로 균등하게 혼합됩니다. alpha가 클수록 0.5에 집중되어 반반 혼합이 자주 발생하고, alpha가 작을수록 극단값(0 또는 1)이 자주 나옵니다. 실험적으로 alpha=0.2~1.0이 좋은 성능을 보입니다.",
        },
        {
          line: "index = torch.randperm(batch_size, device=x.device)",
          title: "배치 내 무작위 쌍 만들기",
          explanation:
            "0부터 batch_size-1까지의 숫자를 무작위로 섞어 인덱스를 만듭니다. x[index]는 같은 배치 내의 이미지를 무작위 순서로 재배열한 것입니다. 즉, 배치의 i번째 샘플은 index[i]번째 샘플과 혼합됩니다. device=x.device를 명시해 GPU와 CPU 불일치 오류를 방지합니다.",
        },
      ],
    },
  },
  {
    slug: "transfer-learning",
    icon: "🧠",
    title: "Transfer Learning",
    subtitle: "Topic 04",
    tags: ["Fine-tuning", "Feature Extraction", "ResNet", "EfficientNet"],
    isFree: false,
    description:
      "사전 학습된 대형 모델의 지식을 내 태스크에 전이하는 방법을 배웁니다. Fine-tuning 전략과 레이어 동결 기법을 실습합니다.",
    content: {
      sections: [
        {
          heading: "Transfer Learning의 핵심 아이디어",
          body: "ImageNet으로 학습된 ResNet이나 EfficientNet 같은 모델들은 수백만 장의 이미지를 처리하면서 '에지 감지 → 텍스처 → 형태 → 객체 부위 → 객체'로 이어지는 계층적 특성 추출기를 내부에 이미 갖추고 있습니다. Transfer Learning은 이 지식을 새로운 태스크에 재사용합니다. 예를 들어 강아지 품종 분류(120종)를 처음부터 학습하려면 수만 장이 필요하지만, ImageNet 사전 학습 모델을 가져와 마지막 분류층만 교체하면 수백 장만으로도 높은 정확도를 달성할 수 있습니다. 데이터가 적거나 컴퓨팅 자원이 제한될 때 특히 효과적입니다.",
        },
        {
          heading: "Feature Extraction vs Fine-tuning",
          body: "Feature Extraction은 사전 학습 모델의 모든 레이어를 동결(freeze)하고, 마지막 분류층만 새로 추가해 훈련합니다. 파라미터 업데이트가 적어 빠르고, 데이터가 매우 적을 때(클래스당 10~50장) 적합합니다. Fine-tuning은 동결한 레이어를 일부 또는 전부 해제하여 전체 모델을 새 데이터로 다시 훈련합니다. 일반적으로 하위 레이어(일반적 특성)는 낮은 학습률로, 상위 레이어(태스크별 특성)는 높은 학습률로 설정하는 Differential Learning Rate 전략을 씁니다. 실무에서는 Feature Extraction으로 먼저 수렴시킨 뒤 Fine-tuning으로 추가 개선하는 단계적 접근이 효과적입니다.",
        },
        {
          heading: "실전 팁: 언제 어떤 전략을?",
          body: "새 데이터가 ImageNet과 유사하고 소규모: Feature Extraction으로 충분합니다. 새 데이터가 ImageNet과 다르고 소규모: 하위 레이어만 동결하고 상위 레이어를 Fine-tuning합니다. 새 데이터가 충분히 크면: 전체 Fine-tuning 또는 처음부터 훈련. 학습률 설정이 매우 중요합니다. Fine-tuning 시 사전 학습 레이어에는 작은 학습률(1e-5)을, 새로 추가한 분류층에는 큰 학습률(1e-3)을 사용합니다. PyTorch의 optimizer parameter groups 기능으로 이를 쉽게 구현할 수 있습니다. 또한 입력 이미지를 모델이 사전 학습된 해상도(ResNet: 224×224)에 맞게 리사이즈하고 ImageNet 통계로 정규화해야 합니다.",
        },
      ],
      codeFile: "04_transfer_learning.py",
      code: `import torch
import torch.nn as nn
from torchvision import models


def build_transfer_model(num_classes: int, strategy: str = "finetune"):
    """
    strategy: 'feature_extract' | 'finetune'
    """
    # ImageNet 사전 학습 가중치 로드
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if strategy == "feature_extract":
        # 전체 레이어 동결 — 분류층만 학습
        for param in model.parameters():
            param.requires_grad = False

    # 마지막 분류층 교체 (1000 → num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Differential Learning Rate 설정 ─────────────────
model = build_transfer_model(num_classes=10, strategy="finetune")

# 새로 추가한 분류층 파라미터
head_params = list(model.fc.parameters())
head_param_ids = {id(p) for p in head_params}

# 사전 학습된 backbone 파라미터 (학습률 1/100)
backbone_params = [
    p for p in model.parameters()
    if id(p) not in head_param_ids
]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},  # 낮은 학습률
    {"params": head_params,     "lr": 1e-3},  # 높은 학습률
], weight_decay=1e-4)

# 학습률 스케줄러 (10 에폭마다 0.1배)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)`,
      codeExplanations: [
        {
          line: "models.ResNet50_Weights.IMAGENET1K_V2",
          title: "사전 학습 가중치 지정",
          explanation:
            "torchvision 0.13+부터 권장되는 새로운 가중치 로드 방식입니다. IMAGENET1K_V2는 더 강력한 데이터 증강과 훈련 기법으로 학습된 ResNet50 가중치로, V1 대비 약 1~2%p 높은 ImageNet 정확도를 보입니다. weights=None으로 설정하면 무작위 초기화가 됩니다.",
        },
        {
          line: "for param in model.parameters(): param.requires_grad = False",
          title: "레이어 동결(Freezing)",
          explanation:
            "requires_grad=False로 설정된 파라미터는 backward() 시 그래디언트가 계산되지 않고, optimizer가 업데이트하지 않습니다. 이 코드 이후 새로 추가되는 model.fc의 파라미터는 기본적으로 requires_grad=True이므로, 분류층만 학습되는 Feature Extraction 모드가 됩니다.",
        },
        {
          line: "model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(...))",
          title: "분류층 교체",
          explanation:
            "ResNet50의 마지막 fc(Fully Connected) 레이어는 1000개 ImageNet 클래스를 출력하도록 설계되어 있습니다. in_features(=2048)는 fc 직전의 특성 벡터 크기입니다. Dropout을 추가하면 작은 데이터셋에서의 Overfitting을 방지할 수 있습니다. 이 새 레이어는 무작위로 초기화됩니다.",
        },
        {
          line: "optimizer = torch.optim.AdamW([{backbone}, {head}], ...)",
          title: "Parameter Groups으로 Differential LR",
          explanation:
            "optimizer에 파라미터 그룹을 리스트로 전달하면 각 그룹에 다른 학습률을 설정할 수 있습니다. backbone(사전 학습 레이어)에는 1e-5를, head(새로 추가한 레이어)에는 1e-3을 사용합니다. backbone 학습률을 너무 높게 설정하면 이미 잘 학습된 특성 표현을 망가뜨릴 수 있습니다.",
        },
      ],
    },
  },
  {
    slug: "mnist-cnn",
    icon: "✍️",
    title: "MNIST CNN 실습",
    subtitle: "Topic 05",
    tags: ["PyTorch", "Conv2d", "BatchNorm", "MNIST", "Accuracy 99%+"],
    isFree: false,
    description:
      "CNN 아키텍처를 직접 설계하고 MNIST 데이터셋으로 99% 이상의 정확도를 달성해봅니다. 전체 훈련 파이프라인을 구현합니다.",
    content: {
      sections: [
        {
          heading: "CNN의 핵심: Convolution이란 무엇인가?",
          body: "Fully Connected(FC) 레이어는 이미지의 공간적 구조를 무시하고 모든 픽셀을 일렬로 펼쳐 처리합니다. 반면 Convolutional Layer는 작은 필터(kernel)를 이미지 위에 슬라이딩하며 국소적인 패턴을 추출합니다. 3×3 필터 하나는 3×3 픽셀 영역의 에지나 코너 같은 패턴을 감지하는 탐지기가 됩니다. 필터의 가중치는 학습을 통해 결정됩니다. 여러 필터를 사용하면 다양한 패턴을 동시에 감지하는 특성 맵(Feature Map)을 얻습니다. 이 특성 맵들이 다음 레이어의 입력이 되어, 하위 레이어의 패턴들이 조합되어 더 복잡한 패턴을 형성합니다. 이것이 CNN이 이미지 인식에 강한 이유입니다.",
        },
        {
          heading: "MNIST CNN 아키텍처 설계 원칙",
          body: "MNIST(28×28, 흑백, 10클래스)는 비교적 단순한 데이터셋으로, 잘 설계된 CNN으로 99% 이상의 정확도를 달성할 수 있습니다. 일반적인 아키텍처 패턴: [Conv→BN→ReLU→MaxPool]을 2~3번 반복해 공간 해상도를 줄이면서 채널(특성) 수를 늘립니다. 이후 Global Average Pooling 또는 Flatten 후 FC 레이어로 최종 분류합니다. MaxPool(2×2)은 특성 맵을 절반으로 줄이며 위치 불변성을 더합니다. 채널 수는 일반적으로 32→64→128처럼 두 배씩 늘립니다. 너무 깊은 네트워크는 MNIST에서 오히려 과적합을 유발하므로 Dropout과 함께 사용하거나 간결하게 유지합니다.",
        },
        {
          heading: "전체 훈련 파이프라인과 99% 달성 전략",
          body: "데이터 준비: MNIST는 torchvision.datasets.MNIST로 쉽게 로드하며, 픽셀을 [0,1]로 정규화 후 평균 0.1307, 표준편차 0.3081로 재정규화합니다. 이 값들은 MNIST 전체 데이터의 통계치입니다. 훈련 전략: 학습률 0.001의 Adam optimizer, CrossEntropyLoss 사용. ReduceLROnPlateau 스케줄러로 검증 손실이 개선되지 않으면 학습률을 자동 감소. 일반적으로 10~15 에폭이면 수렴합니다. 99% 이상을 위한 핵심: BatchNorm으로 안정적 훈련, Dropout(0.25~0.5)으로 과적합 방지, 적절한 데이터 증강(RandomAffine으로 약간의 회전/이동), 테스트 시 model.eval() 필수 호출(Dropout과 BatchNorm 비활성화 위해).",
        },
      ],
      codeFile: "05_mnist_cnn.py",
      code: `import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MnistCNN(nn.Module):
    """MNIST 99%+ 달성 CNN 아키텍처"""

    def __init__(self):
        super().__init__()
        # Block 1: 1ch → 32ch, 28×28 → 14×14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
        )
        # Block 2: 32ch → 64ch, 14×14 → 7×7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
        )
        # Classifier: 64*7*7 → 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n`,
      codeExplanations: [
        {
          line: "nn.Conv2d(1, 32, kernel_size=3, padding=1)",
          title: "Conv2d 파라미터 이해",
          explanation:
            "첫 번째 인자(1)는 입력 채널 수(MNIST는 흑백이므로 1), 두 번째 인자(32)는 출력 채널 수(학습할 필터 수), kernel_size=3은 3×3 필터, padding=1은 출력 크기를 입력과 동일하게 유지하기 위한 zero-padding입니다. 이 레이어의 파라미터 수는 1×32×3×3+32(bias)=320개입니다. padding=1 덕분에 28×28 입력이 28×28 출력을 유지합니다.",
        },
        {
          line: "nn.BatchNorm2d(32)",
          title: "2D Batch Normalization",
          explanation:
            "Conv 레이어 출력(N, C, H, W)을 채널별로 배치 단위 정규화합니다. 각 채널의 H×W 픽셀 전체에 걸쳐 평균과 분산을 계산합니다. BN은 학습 중 각 채널의 평균(γ)과 분산(β)을 학습 가능한 파라미터로 보정합니다. 추론 시에는 훈련 중 누적된 이동 평균(running mean/var)을 사용합니다.",
        },
        {
          line: "nn.Dropout2d(p=0.25)",
          title: "Dropout2d: 채널 단위 Dropout",
          explanation:
            "일반 Dropout과 달리 Dropout2d는 채널(feature map) 전체를 확률 p로 0으로 만듭니다. Conv 레이어에서는 인접 픽셀들의 상관관계가 높아 픽셀 단위 Dropout이 비효율적입니다. 채널 전체를 끄는 방식이 더 강한 regularization 효과를 냅니다. Classifier 부분에는 일반 Dropout(p=0.5)을 사용합니다.",
        },
        {
          line: "@torch.no_grad()",
          title: "그래디언트 계산 비활성화",
          explanation:
            "데코레이터 방식으로 함수 내 모든 연산에서 그래디언트 추적을 비활성화합니다. 검증/테스트 시에는 역전파가 필요 없으므로 그래디언트를 계산할 필요가 없습니다. 이를 통해 메모리 사용량을 줄이고 추론 속도를 높입니다. with torch.no_grad(): 블록 방식과 동일하지만 함수에 적용할 때는 데코레이터가 더 깔끔합니다.",
        },
        {
          line: "outputs.argmax(1) == labels",
          title: "정확도 계산",
          explanation:
            "argmax(1)은 각 샘플의 10개 출력값 중 가장 큰 값의 인덱스(예측 클래스)를 반환합니다. 차원 1(클래스 차원)을 따라 argmax를 취합니다. 이를 실제 레이블과 비교해 맞은 샘플 수를 셉니다. .sum().item()으로 텐서를 Python 정수로 변환합니다. 최종 정확도 = correct / 전체 샘플 수",
        },
      ],
    },
  },
]

export function getTopicBySlug(slug: string): Topic | undefined {
  return TOPICS.find((t) => t.slug === slug)
}
