# 🗂️ Data Clustering AI Laboratory (K-Means)

이 프로젝트는 비지도 학습(Unsupervised Learning)의 대표적인 알고리즘인 **K-Means 클러스터링**을 시각화하고 체험할 수 있는 인터랙티브 웹 대시보드입니다. 정답이 없는 데이터 속에서 AI가 어떻게 스스로 패턴을 찾아 그룹화하는지 보여줍니다.

## 🚀 주요 기능

### 1. 동적 데이터 생성 (Dynamic Data Generation)
- 사용자가 데이터 샘플 수($N$), 군집 수($K$), 데이터의 분산($\sigma$)을 직접 조절하여 다양한 시나리오를 시뮬레이션할 수 있습니다.

### 2. 알고리즘 시각화 (K-Means Visualization)
- **2D Cluster Plot:** 데이터 포인트들이 각각의 군집으로 분류되는 과정과 최종 중심점(Centroid, 별 모양)의 위치를 실시간으로 확인할 수 있습니다.
- **Inertia Convergence:** 반복(Iteration) 횟수에 따라 군집 내 오차 제곱합(Inertia)이 어떻게 줄어들며 수렴하는지 그래프로 제공합니다.

### 3. 신규 데이터 분류 (Inference Sandbox)
- 학습이 완료된 후, 새로운 데이터(구매 금액, 방문 횟수 등)를 슬라이더로 입력하면 AI가 실시간으로 가장 적합한 그룹을 예측하고 중심점까지의 거리를 계산합니다.

### 4. 수학적 이론 및 EM 단계 설명
- 유클리디안 거리 수식과 목적 함수($J$)를 MathJax를 통해 선명하게 제공합니다.
- 알고리즘의 핵심인 **Expectation-Maximization (EM)** 과정을 단계별(초기화 -> 할당 -> 업데이트)로 상세히 설명합니다.

### 5. 현실 세계 활용 사례 정리
- 고객 세분화, 이미지 압축, 이상치 탐지 등 클러스터링 기술이 실제 산업 현장에서 어떻게 활용되는지 가이드를 포함합니다.

## 🛠 기술 스택

- **Backend:** Python, FastAPI
- **Algorithm:** K-Means (NumPy custom implementation)
- **Frontend:** HTML5, CSS3 (Tailwind CSS), JavaScript
- **Visualization:** Plotly.js, MathJax (Mathematics Rendering)

## 🏃 실행 방법

1. 필요한 라이브러리를 설치합니다:
```bash
pip install fastapi uvicorn numpy pydantic
```

2. 서버를 실행합니다:
```bash
python 06_unsupervised_clustering_web.py
```

3. 브라우저에서 아래 주소로 접속합니다:
`http://localhost:8000`

---
*본 프로젝트는 AI 입문자를 위한 비지도 학습 교육용 도구로 제작되었습니다.*
