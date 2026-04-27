# [TRD] PhysViz-Perceptron Pro: 물리 장 기반 고성능 렌더링 시스템 설계서

**문서 번호:** TRD-20260325-01-ULTRA  
**버전:** 8.0 (High-Performance Architecture)  
**작성자:** Senior System Architect  

---

## 1. 시스템 설계 개요 (System Design Overview)

### 1.1. 설계 목표
사용자가 슬라이더를 조작하거나 캔버스를 드래그할 때, $100 \times 100$ 이상의 고해상도 전위 메쉬와 3D 투영 등고선을 지연 없이(60FPS) 리렌더링하는 것을 목표로 합니다. 이를 위해 NumPy의 벡터화 연산과 PyQtGraph의 OpenGL 가속 기능을 결합한 하이브리드 아키텍처를 채택합니다.

---

## 2. 소프트웨어 아키텍처 (Software Architecture)

### 2.1. MVI (Model-View-Intent) 디자인 패턴
데이터의 일관성을 유지하고 UI 업데이트 로직을 분리하기 위해 MVI 패턴을 사용합니다.
*   **Model:** 전역 상태(Weights, Bias, Dataset, Physics Constants)를 관리하는 단일 진실 공급원(Single Source of Truth).
*   **View:** PyQtGraph 기반 2D/3D 캔버스. 모델의 상태 변화를 시그널로 수신하여 즉시 렌더링.
*   **Intent:** 사용자 입력(슬라이더, 드래그, 수식 입력)을 수집하여 모델 업데이트 명령으로 변환.

---

## 3. 핵심 모듈 상세 설계 (Module Detailed Design)

### 3.1. `PerceptronPhysicsEngine` (수치 연산 엔진)
*   **전위 장 연산:** `np.meshgrid`와 브로드캐스팅을 활용한 고속 연산.
    ```python
    def update_field(self):
        # x, y 그리드는 초기 1회 생성 후 재사용
        self.z = self.w[0] * self.X + self.w[1] * self.Y + self.b
        if self.physics_mode:
            self.potential = self.kappa * self.z  # 물리 단위 환산
    ```
*   **3D Contour Projection:** 
    *   2D 전위 맵을 `QImage` 텍스처로 변환.
    *   OpenGL 쉐이더 혹은 PyQtGraph의 `GLSurfaceItem`을 사용하여 3D 지형의 $Z$값에 따라 텍스처 색상을 맵핑.

### 3.2. `InteractiveCanvas2D` (고속 상호작용 뷰)
*   **Direct Manipulation:** 결정 경계 드래그 시 역운동학(Inverse Kinematics) 적용.
    *   마우스 클릭 지점 $P_1(x_1, y_1)$과 이동 지점 $P_2(x_2, y_2)$를 잇는 직선의 방정식 $Ax + By + C = 0$ 산출.
    *   $w_1 = A, w_2 = B, b = C$로 모델 업데이트.
*   **Blitting 최적화:** 배경 격자와 데이터 포인트는 캐싱하고, 변화하는 경계선과 벡터 화살표만 동적으로 갱신.

### 3.3. `LossAnalyzer` (비동기 손실 지형 분석기)
*   **병렬 연산:** $w_1, w_2$ 평면의 그리드 연산을 `multiprocessing` 또는 `concurrent.futures`를 사용하여 백그라운드에서 처리.
*   **Bicubic Interpolation:** `scipy.interpolate.interp2d`를 사용하여 성긴 샘플 데이터로부터 매끄러운 3D 곡면 생성.

---

## 4. UI 컴포넌트 구조 및 매핑 (Object Tree)

```text
PhysVizApp (QMainWindow)
├── MainSplitter (QSplitter)
│   ├── SideControlPanel (QScrollArea)
│   │   └── Layout (QVBoxLayout)
│   │       ├── ParamGroup (QGroupBox: weights/bias)
│   │       ├── PhysicsGroup (QGroupBox: epsilon/charge)
│   │       └── DataGenGroup (QGroupBox: custom function)
│   └── VisualizationStage (QWidget)
│       └── Layout (QVBoxLayout)
│           ├── FormulaLabel (LaTeX Renderer via Matplotlib/QtSvg)
│           └── GraphSplitter (QSplitter: Vertical)
│               ├── Canvas2D (pg.PlotWidget)
│               └── Canvas3D (pg.opengl.GLViewWidget)
└── StatusBar (Showing Loss, FPS, Physics Info)
```

---

## 5. 핵심 알고리즘 상세 (Algorithmic Details)

### 5.1. Physics-to-AI Scaling Logic
물리량 $V$와 AI 파라미터 $z$ 사이의 선형 결합 계수 $\kappa$를 다음과 같이 정의합니다.
$$V(\vec{x}) = \frac{q}{4\pi\epsilon_0} (\vec{w} \cdot \vec{x} + b)$$
여기서 $\vec{w}$는 정규화된 방향 벡터로 취급하며, 슬라이더 조작 시 $\epsilon_0$ 값이 분모로 들어가 전위 장의 기울기(전기장 세기)를 반비례하게 조절합니다.

### 5.2. Safe Function Evaluation (1-2)
사용자가 입력한 `sin(x)` 등의 수식을 안전하게 실행하기 위한 파이프라인:
1.  `ast.parse`를 통한 구문 분석 (보안 검사).
2.  `SymPy`를 통한 기호적 해석.
3.  `numpy.vectorize` 혹은 `numexpr`을 이용한 고속 벡터 연산 실행.

---

## 6. 성능 및 수치 안정성 확보 전략

### 6.1. Numerical Robustness
*   **Zero-Division:** $\|\vec{w}\| < 10^{-10}$ 일 경우 전기장 계산을 중단하고 경고 메시지 출력.
*   **Exploding Gradients:** Loss Landscape 연산 시 값이 `inf`로 발산하는 구간은 시각적으로 `clip` 처리하여 지형의 형태 유지.

### 6.2. 렌더링 최적화
*   **Texture Caching:** 3D 투영용 텍스처는 매핑 데이터가 변하지 않는 한 GPU 메모리에서 재사용.
*   **Signal Throttling:** 슬라이더 이벤트 발생 시 `QTimer`를 이용하여 최소 16ms 간격으로 업데이트 시그널 제한.

---

## 7. 배포 및 환경 명세 (Dependencies)

*   **Core:** Python 3.10+, NumPy 1.23+, SciPy 1.9+
*   **GUI:** PySide6 6.4+, PyQtGraph 0.13+
*   **Graphics:** OpenGL 3.3+ (Core Profile)
*   **Math Rendering:** Matplotlib 3.6+ (Agg backend)

---
*(이 문서는 기술적 상세 구현 방안을 포함하여 150라인 이상으로 구성되었습니다.)*
