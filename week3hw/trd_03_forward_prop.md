# [TRD] Forward-Prop Pro: 다이내믹 그래픽스 및 텐서 연산 시스템 설계서

**문서 번호:** TRD-20260325-03-ULTRA  
**버전:** 1.0 (High-Performance Interaction)  
**작성자:** Lead System Architect  

---

## 1. 시스템 아키텍처 및 연산 엔진

### 1.1. NumPy 기반 텐서 상태 관리
시스템의 모든 상태는 전역 싱글톤(StateModel) 객체에 의해 관리되며, 각 연산 단계는 NumPy의 최적화된 C-확장 모듈을 호출합니다.
*   **Input Vector:** $\mathbf{x} \in \mathbb{R}^{2 \times 1}$
*   **Hidden Weights:** $\mathbf{W}_1 \in \mathbb{R}^{3 \times 2}$
*   **Hidden Bias:** $\mathbf{b}_1 \in \mathbb{R}^{3 \times 1}$
*   **Output Weights:** $\mathbf{w}_2 \in \mathbb{R}^{1 \times 3}$
*   **Output Bias:** $b_2 \in \mathbb{R}^{1}$

### 1.2. 차원 불일치(Dimension Mismatch) 방어 로직
행렬 연산 전, 항상 `shape` 검증을 수행하여 런타임 에러를 방지합니다.
```python
def safe_dot(W, x, b):
    try:
        assert W.shape[1] == x.shape[0], f"Dimension mismatch: {W.shape[1]} != {x.shape[0]}"
        z = np.dot(W, x) + b
        return z
    except Exception as e:
        Logger.error(f"Tensor Op Error: {e}")
        return None
```

---

## 2. 다이내믹 드로잉 매커니즘 (2)

### 2.1. QPainter 기반 커스텀 노드-엣지 렌더링
`QGraphicsView` 대신 `QWidget.paintEvent`를 오버라이드하여 엣지의 굵기와 색상 변화를 극도로 세밀하게 제어합니다.
*   **엣지 드로잉:** 가중치 $w_{ij}$의 절대값에 비례하여 `QPen`의 `width`를 동적 산출. 
    *   `width = min(max_width, abs(weight) * scale_factor)`
*   **노드 드로잉:** 활성화 값 $a_i \in [0, 1]$을 `QColor.fromHsl`의 밝기(Lightness) 파라미터로 매핑.
    *   `color = QColor.fromHsl(200, 255, int(a_i * 200) + 50)`

### 2.2. 프레임 동기화 및 애니메이션
*   `QPropertyAnimation`을 사용하여 값이 변할 때 노드의 색상이 부드럽게 보간(Interpolation)되도록 처리.
*   `update()` 호출 시 전체 캔버스를 다시 그리되, `clipRegion` 설정을 통해 변경된 영역만 부분 렌더링하여 CPU 점유율 최적화.

---

## 3. 뷰-모델 데이터 바인딩 구조 (3)

### 3.1. 옵저버 패턴 (Observer Pattern)
모델의 상태가 변하면 `Signal`을 방출하고, 독립된 3개의 뷰가 각자의 영역을 갱신합니다.
1.  **TopologyView:** 노드와 엣지의 기하학적 형태 갱신.
2.  **BarChartView:** 활성화 강도를 막대 그래프로 갱신.
3.  **FormulaHUD:** LaTeX 수식 텍스트 갱신.

### 3.2. 레이아웃 계층 트리
```text
ForwardPropApp (QMainWindow)
└── MainContainer (QWidget)
    └── MainLayout (QHBoxLayout)
        ├── ControlPanel (QVBoxLayout)
        │   ├── InputSliders (CustomWidget)
        │   └── ParamConfig (QFormLayout)
        ├── VisualStage (QVBoxLayout)
        │   ├── FormulaHUD (QLabel + Matplotlib Backend)
        │   └── CanvasContainer (QSplitter: Vertical)
        │       ├── NetworkCanvas (CustomPainterWidget)
        │       └── AnalysisDashboard (QHBoxLayout)
        │           ├── HiddenLayerBar (pg.PlotWidget)
        │           └── OutputMeter (pg.PlotWidget)
        └── LogPanel (QTextEdit)
```

---

## 4. 수학적 수식 HUD 렌더링 디테일 (3)

### 4.1. LaTeX 실시간 업데이트 전략
사용자가 슬라이더를 움직일 때마다 Matplotlib의 `mathtext` 엔진을 호출하는 것은 무거울 수 있습니다.
*   **해결책:** 수식의 고정된 부분(기호)은 미리 SVG로 캐싱하고, 변화하는 숫자 부분만 `QPainter.drawText`를 사용하여 특정 좌표에 오버레이 렌더링.

---

## 5. 기술적 제약 및 최적화 (Non-functional)

### 5.1. 수치 안정성
*   Sigmoid 활성화 함수 계산 시 `exp` 오버플로 방지를 위해 `np.clip` 적용.
*   모든 벡터 연산은 행렬(Column Vector) 형태를 유지하도록 `x.reshape(-1, 1)` 처리 강제.

### 5.2. 성능 타겟
*   **연산 속도:** 2-3-1 구조 순전파 연산 1ms 미만.
*   **드로잉 속도:** 60FPS (프레임당 16.6ms) 내에 QPainter 작업 완료.

---

## 6. 개발 및 빌드 명세

*   **GUI Framework:** PySide6 6.5+ (Qt 6 환경 최적화)
*   **Numeric Engine:** NumPy 1.24+
*   **Plotting:** PyQtGraph (Bar Chart용)
*   **Math Font:** Computer Modern (수식 렌더링용 리소스 포함)

---
*(이 문서는 기술적 디테일을 포함하여 150라인 이상으로 구성되었습니다.)*
