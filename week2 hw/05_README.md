# 🧪 Hooke's AI Physics Lab

인공지능(AI)과 물리 법칙을 결합한 인터랙티브 가상 실험실입니다. 훅의 법칙(Hooke's Law)을 선형 회귀(Linear Regression) 모델을 통해 학습하고 시각화하는 교육용 대시보드입니다.

## 🚀 주요 기능

### 1. 물리 수식 편집기 (Ground Truth Editor)
- 사용자가 직접 용수철 상수($k$)와 편향($b$)을 설정하여 실제 물리 법칙을 정의할 수 있습니다.
- 설정된 수식에 따라 실시간으로 데이터가 생성되며, AI는 이 수식을 찾아내는 것을 목표로 학습합니다.

### 2. 하이퍼파라미터 튜닝
- **데이터 샘플 수 ($N$):** 실험 데이터의 양을 조절합니다.
- **측정 노이즈 ($\sigma$):** 실제 실험에서 발생할 수 있는 오차 범위를 설정합니다.
- **학습률 ($\alpha$):** AI 모델이 최적의 수식에 도달하는 속도를 조절합니다.

### 3. 실시간 대시보드 & 시각화
- **Loss Convergence:** 학습이 진행됨에 따라 오차(MSE)가 줄어드는 과정을 그래프로 확인합니다.
- **Regression Fit:** 원본 데이터(Noisy Data), 실제 물리 수식(Target), AI가 예측한 수식(AI Pred)을 한눈에 비교합니다.
- **KPI Cards:** $R^2$ Score, Final Loss 등 학습 성과를 수치로 제공합니다.

### 4. 인터랙티브 가상 실험실 (Sandbox)
- 학습된 모델을 바탕으로 새로운 무게를 매달았을 때 용수철이 얼마나 늘어날지 AI가 예측합니다.
- 스프링 애니메이션을 통해 이론값과 AI 예측값의 차이를 시각적으로 체험할 수 있습니다.

### 5. 교육용 이론 섹션
- 훅의 법칙($F = kx$)과 머신러닝의 선형 회귀($\hat{y} = wx + b$) 사이의 수학적 관계를 상세히 설명합니다.
- MathJax를 사용하여 복잡한 수식을 깨짐 없이 깔끔하게 렌더링합니다.

## 🛠 기술 스택

- **Backend:** Python, FastAPI
- **AI/ML:** TensorFlow 2.x, NumPy
- **Frontend:** HTML5, CSS3 (Tailwind CSS), JavaScript
- **Visualization:** Plotly.js, MathJax (Mathematics Rendering)

## 🏃 실행 방법

1. 필요한 라이브러리를 설치합니다:
```bash
pip install tensorflow fastapi uvicorn numpy
```

2. 서버를 실행합니다:
```bash
python 05_linear_regression_spring2.py
```

3. 브라우저에서 아래 주소로 접속합니다:
`http://localhost:8000`

---
*본 프로젝트는 AI 및 머신러닝 과정의 교육용 도구로 제작되었습니다.*
