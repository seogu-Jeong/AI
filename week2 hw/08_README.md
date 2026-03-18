# 📉 Gradient Descent AI Laboratory

이 프로젝트는 인공지능 최적화의 핵심 알고리즘인 **경사 하강법(Gradient Descent)**의 작동 원리를 시각화하는 인터랙티브 웹 대시보드입니다. 산 위에서 공을 굴려 가장 낮은 골짜기를 찾는 과정을 체험할 수 있습니다.

## 🚀 주요 기능

### 1. 하이퍼파라미터 제어 (Optimization Settings)
- **학습률 (Learning Rate, $\eta$):** 한 번의 업데이트에서 이동할 보폭을 결정합니다. 너무 크면 발산하고, 너무 작으면 느리게 수렴하는 과정을 직접 확인할 수 있습니다.
- **시작 위치 (Start x):** 다양한 지점에서 시작하여 전역 최솟값(Global Minimum)을 찾아가는 과정을 테스트합니다.
- **업데이트 횟수 (Steps):** 최적화 과정을 몇 단계까지 진행할지 설정합니다.

### 2. 동적 시각화 (Gradient Path)
- **Interactive Plot:** 손실 함수 $f(x)=x^2$ 곡선 위를 따라 점들이 최솟값으로 이동하는 경로를 Plotly 차트로 실시간 렌더링합니다.
- **Color Mapping:** 단계(Step)가 진행됨에 따라 점의 색상이 변하여 시간 흐름에 따른 최적화 과정을 명확히 보여줍니다.

### 3. 실시간 최적화 로그 (Optimization Log)
- 매 단계마다 업데이트되는 파라미터 $x$의 값과 손실(Loss) 변화를 테이블 형태로 제공합니다.

### 4. 수학적 이론 및 공식
- 경사 하강법의 업데이트 수식을 MathJax로 깔끔하게 제공하여, 기울기(Gradient)의 반대 방향으로 이동하는 수학적 근거를 설명합니다.

## 🛠 기술 스택

- **Backend:** Python, FastAPI
- **Algorithm:** Gradient Descent (NumPy custom implementation)
- **Frontend:** HTML5, CSS3 (Tailwind CSS), JavaScript
- **Visualization:** Plotly.js, MathJax

## 🏃 실행 방법

1. 필요한 라이브러리를 설치합니다:
```bash
pip install fastapi uvicorn numpy pydantic
```

2. 서버를 실행합니다:
```bash
python 08_gradient_descent_web.py
```

3. 브라우저에서 아래 주소로 접속합니다:
`http://localhost:8000`

---
*최적의 학습률을 찾는 것이 인공지능 학습의 절반입니다.*
