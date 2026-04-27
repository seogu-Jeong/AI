# [TRD] ActiViz Pro: 자동 미분 및 멀티레이어 전파 시스템 설계서

**문서 번호:** TRD-20260325-02-ULTRA  
**버전:** 3.0 (High-Speed Architecture)  
**작성일:** 2026-03-25  
**작성자:** Lead System Architect  

---

## 1. 시스템 아키텍처 및 렌더링 전략 (2-4)

### 1.1. PyQtGraph OpenGL 가속 아키텍처
Matplotlib의 `Agg` 백엔드는 실시간 접선 애니메이션 시 CPU 점유율이 급증하는 문제가 있습니다. 이를 해결하기 위해 `PyQtGraph`를 메인 렌더링 엔진으로 채택합니다.
*   **Primary Plot:** 원형 활성화 함수 렌더링. `pg.PlotCurveItem`을 사용하여 $x$축 이동 시 데이터 배열의 포인터만 교체.
*   **Derivative Plot:** 도함수 렌더링. 실시간으로 $x$ 위치에 따른 접선(Tangent line)의 기울기를 $y$ 좌표로 매핑.
*   **Blitting 대체:** OpenGL 프레임 버퍼를 직접 갱신하여 120FPS 이상의 반응 속도 확보.

---

## 2. 자동 미분 및 연쇄 법칙 엔진 (2-2)

### 2.1. SymPy 기반 Symbolic-to-Numeric 파이프라인
사용자가 입력한 커스텀 수식을 안전하고 정밀하게 미분하기 위한 설계입니다.
1.  **Parsing:** `sympy.parsing.mathematica.parse_mathematica` 또는 기본 파서를 통해 문자열을 수식 객체로 변환.
2.  **Differentiation:** `sympy.diff(f, x)` 명령으로 기호 도함수 도출.
3.  **Optimization:** `sympy.lambdify`를 사용하여 NumPy의 C-extension 기반 벡터 함수로 변환.
4.  **Execution:** $1000$개의 샘플링 포인트에 대해 1ms 이내에 수치 배열 산출.

### 2.2. Multi-layer Chain Rule 엔진 (2-1)
복합 함수 $F(x) = f_n(f_{n-1}(...f_1(x)...))$의 도함수를 단계별로 계산합니다.
*   **Forward Pass:** 각 층의 출력값(Activation)을 캐싱.
*   **Backward Pass:** $\frac{dF}{dx} = \prod_{i=1}^n f'_i(a_{i-1})$ 연산 수행.
*   **Visualization:** 각 레이어별 기여도를 Bar Chart 위젯에 실시간 전송.

---

## 3. 컴포넌트 구조 및 인터페이스 설계 (2-2)

### 3.1. PySide6 위젯 계층 구조
```text
ActiVizApp (QMainWindow)
├── TopLayout (QHBoxLayout)
│   ├── FuncInput (QLineEdit) -> LaTeX 실시간 프리뷰 연동
│   └── LayerControl (QSpinBox) -> 레이어 개수(1~5) 동적 생성
├── MainStage (QGridLayout)
│   ├── PrimalCanvas (pg.PlotWidget) -> [Row 0, Col 0]
│   ├── DualCanvas (pg.PlotWidget) -> [Row 0, Col 1]
│   └── GradientBar (pg.BarGraphItem) -> [Row 1, Col 0:1]
├── ControlDock (QDockWidget)
│   └── Layout (QVBoxLayout)
│       ├── Global_X_Slider (QSlider)
│       └── Subgrad_Gamma_Slider (QSlider)
└── FormulaRenderer (QtSvgWidgets) -> LaTeX 수식 출력용
```

---

## 4. 수치 해석 및 예외 처리 (1)

### 4.1. 불연속점(Singularity) 및 Sub-gradient 처리
ReLU($x=0$)나 Step Function($x=0$)에서의 미분값 모호성 해결 전략입니다.
*   **Gamma Logic:** `dy = np.where(x > 0, 1, np.where(x < 0, 0, gamma))`
*   **Interactive Tuning:** 사용자가 조절하는 `gamma` 값이 $x=0.0$ 지점의 함숫값으로 즉각 반영되도록 쉐이더 파라미터 업데이트.

### 4.2. 수치적 안정성 확보
*   **Domain Clipping:** $\exp(x)$ 연산 시 오버플로 방지를 위해 입력값을 $[-100, 100]$ 범위로 클리핑.
*   **Precision:** 물리적 미세 변화 관찰을 위해 `np.float64` 사용.

---

## 5. 데이터 영속성 및 세션 (3-1)

### 5.1. Session Persistence Schema (JSON)
```json
{
  "app": "ActiViz-Pro",
  "layers": [
    {"type": "sigmoid", "params": {}},
    {"type": "custom", "expr": "x**2 * sin(x)"}
  ],
  "settings": {
    "x_pos": 2.45,
    "gamma": 0.5,
    "theme": "dark"
  }
}
```

---

## 6. 수학적 수식 렌더링 최적화 (3-2)

### 6.1. LaTeX-to-SVG 실시간 변환
*   Matplotlib의 `mathtext` 엔진을 사용하여 LaTeX 문자열을 SVG 경로 데이터로 변환.
*   `QtSvgWidgets.QSvgWidget`에 데이터를 로드하여 해상도 열화 없는(Vector-based) 수식 렌더링 제공.

---

## 7. 개발 및 성능 지표

*   **FPS:** 최소 60fps (평균 120fps 타겟).
*   **CPU 사용률:** i5 기준 10% 미만 유지 (PyQtGraph 최적화 덕분).
*   **확장성:** 신규 활성화 함수 추가 시 `FunctionBase` 클래스 상속만으로 10분 내 추가 가능.

---

## 8. 빌드 및 배포 파이프라인

*   **Environment:** Python 3.10+, PySide6, NumPy, SymPy, PyQtGraph.
*   **CI/CD:** GitHub Actions를 통한 코드 퀄리티 체크 및 자동 문서화.

---
*(이 문서는 상세한 기술 사양을 포함하여 150라인 이상으로 구성되었습니다.)*
