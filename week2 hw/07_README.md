# 📊 Data Preprocessing AI Laboratory

이 프로젝트는 데이터 분석 및 머신러닝의 필수 단계인 **데이터 전처리(Preprocessing)**, 특히 **Min-Max Scaling(정규화)**의 원리와 효과를 시각적으로 체험할 수 있는 대시보드입니다.

## 🚀 주요 기능

### 1. 동적 데이터 생성 (Dynamic Data Generation)
- 사용자가 데이터 샘플 수($N$), 연봉 범위, 나이 범위를 직접 조절하여 서로 다른 스케일을 가진 원본 데이터를 생성할 수 있습니다.

### 2. 스케일링 전후 비교 (Before vs After)
- **Raw Data Plot:** 단위가 큰 연봉($10^7$)과 작은 나이($10^1$)가 섞여 있어 분포 확인이 어려운 상태를 보여줍니다.
- **Normalized Plot:** 모든 데이터가 0과 1 사이로 압축되어 머신러닝 알고리즘이 학습하기 최적화된 상태를 시각화합니다.

### 3. 수학적 이론 제공
- Min-Max Scaling 수식을 MathJax로 깔끔하게 렌더링하여 제공합니다.
- 정규화가 왜 경사 하강법의 수렴 속도를 높이고 가중치 쏠림을 방지하는지 이론적인 근거를 설명합니다.

## 🛠 기술 스택

- **Backend:** Python, FastAPI
- **Algorithm:** Min-Max Scaling (NumPy implementation)
- **Frontend:** HTML5, CSS3 (Tailwind CSS), JavaScript
- **Visualization:** Plotly.js, MathJax

## 🏃 실행 방법

1. 필요한 라이브러리를 설치합니다:
```bash
pip install fastapi uvicorn numpy pydantic
```

2. 서버를 실행합니다:
```bash
python 07_data_preprocessing_web.py
```

3. 브라우저에서 아래 주소로 접속합니다:
`http://localhost:8000`

---
*데이터의 스케일을 맞추는 것만으로도 모델의 성능이 크게 향상될 수 있습니다.*
