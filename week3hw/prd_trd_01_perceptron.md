# [Masterpiece PRD/TRD] PhysViz-Perceptron Pro: 텐서 장(Tensor Field) 현상학 기반 선형 분류기 시뮬레이터

**문서 상태:** Final / Approved for Development  
**버전:** 4.0 (Enterprise & Academic Grade Specification)  
**작성자:** Chief Product Officer & Lead System Architect  
**대상 독자:** 물리학과/수학과 학부 고학년 및 대학원생, AI 연구원, 시니어 엔지니어  

---

## Part 1: [PRD] Product Requirements Document
### "The Physical Phenomenology of Neural Networks"

### 1. 제품 철학 (Core Philosophy)
본 애플리케이션은 퍼셉트론을 단순한 '수학적 함수'로 취급하지 않습니다. 우리는 가중치 벡터 $\vec{w}$를 공간의 이방성(Anisotropy)을 유도하는 **주축 벡터(Principal Vector)**로, 편향 $b$를 전위의 **오프셋(Offset)**으로, 그리고 결정 경계를 전위가 0이 되는 **절점 평면(Nodal Hyperplane)**으로 재정의합니다. 타겟 유저는 수식의 기계적 암기가 아닌, 매개변수의 변화가 공간의 위상을 어떻게 비틀고 에너지를 최소화하는지 '물리적 직관'을 통해 체득해야 합니다.

### 2. 학습 내러티브 아크: 선형 분리의 위상학 (Pedagogical Narrative Arc)
단순한 튜토리얼을 넘어, 5단계의 인지적 각성(Cognitive Awakening) 과정을 UI로 강제합니다.

*   **Phase 1: 등방성 진공 (The Isotropic Vacuum)**
    *   초기 화면은 가중치가 $0$인 상태, 즉 어떠한 장(Field)도 형성되지 않은 평탄한 전위 공간입니다.
    *   사용자가 화면을 클릭하여 클래스 0(전하 $-1$, Blue)과 클래스 1(전하 $+1$, Red)의 입자를 배치합니다.
*   **Phase 2: 장의 발현과 대칭성 깨짐 (Emergence of the Field)**
    *   가중치 슬라이더를 미세하게 움직이는 순간, 화면 전체에 전위 등고선(Equipotential Lines)이 형성되며 공간의 대칭성이 깨집니다.
    *   **Insight:** 사용자는 $\vec{w}$의 방향이 가장 가파른 전위 상승 방향(Gradient, $\nabla V$)임을 즉각적으로 깨닫습니다.
*   **Phase 3: 평행 이동과 게이트 로직 (Translation & Boolean Logic)**
    *   편향 $b$를 조작하면, 전위 장 전체가 $\vec{w}$의 법선 방향을 따라 평행 이동합니다.
    *   이를 통해 AND, OR 게이트가 본질적으로 '동일한 기울기를 가진 장의 평행 이동'에 불과함을 보여줍니다.
*   **Phase 4: 위상적 결함 (The Topological Defect - XOR)**
    *   XOR 패턴이 화면에 배치됩니다. 사용자는 2D 결정 경계선(Direct Manipulation)을 마우스로 잡고 미친 듯이 회전시키며 공간을 분할하려 시도합니다.
    *   **Insight:** "단일 초평면(Single Hyperplane)으로는 위상적으로 꼬인 두 전하 그룹을 분리하는 동형 사상(Homeomorphism)을 만들 수 없다"는 수학적 한계를 체감합니다.
*   **Phase 5: 에너지 지형의 덫 (The Saddle Point Trap)**
    *   분리 불가를 확인한 유저에게 별도 창으로 띄워진 **'Loss Landscape (손실 지형)'**을 보여줍니다.
    *   경사하강법(Gradient Descent) 시뮬레이션을 켜면, 상태를 나타내는 구슬이 전역 최소점(Global Minimum)을 찾지 못하고 안장점(Saddle Point)이나 평탄한 계곡을 영원히 맴도는 현상을 생중계합니다.

### 3. 고해상도 기능 명세 (High-Fidelity Feature Specs)

#### 3.1. Field Generation & Contour Topography (2D 뷰)
*   **Equipotential Rendering:** $100 \times 100$ 해상도의 `np.meshgrid`를 기반으로, $z = \vec{w} \cdot \vec{x} + b$ 연산을 수행. 결과는 20-level의 `contourf` (히트맵)와 `contour` (등고선)로 오버레이.
*   **Gradient Vectors:** 등고선과 직교하는 $\nabla V$ 벡터장 화살표들을 희미하게 배경에 배치.
*   **Orthogonal Projection:** 특정 데이터 포인트를 `Shift+Click` 시, 해당 점에서 결정 경계면으로 내린 수선의 발(Orthogonal Projection)을 애니메이션으로 그리고, 점과 면 사이의 유클리드 거리 $d = \frac{|\vec{w} \cdot \vec{x} + b|}{\|\vec{w}\|}$를 말풍선으로 표시.

#### 3.2. Direct Manifold Manipulation (역운동학 기반 직접 조작)
*   슬라이더 조작뿐만 아니라, 사용자가 2D 화면에 그려진 '결정 경계선'의 양 끝단이나 중앙을 잡아 드래그할 수 있습니다.
*   **수학적 역산:** 화면상의 선분 좌표 $(x_1, y_1) \to (x_2, y_2)$가 주어졌을 때, 해당 선분에 수직이면서 크기가 정규화된 새로운 $\vec{w}$와 $b$를 실시간으로 역산출하여 시스템 상태를 업데이트하고 슬라이더 위치를 동기화합니다.

#### 3.3. 3D Energy Landscape (3D 뷰)
*   로짓(Logit) 혹은 활성화 함수의 출력값을 Z축으로 하는 3D Surface Plot을 우측 하단에 병렬 배치.
*   사용자가 2D에서 선을 돌리면, 3D에서는 거대한 평면(또는 시그모이드 곡면)이 파도처럼 출렁이며 기울어지는(Tilting) 장관을 연출.

#### 3.4. Activation Functions as Dielectric Media (활성화 함수 = 매질의 특성)
*   **Step Function:** 무한대의 기울기를 가진 절벽(Phase transition).
*   **Sigmoid/Tanh:** 부드러운 확률적 전이 구역(Probabilistic transition zone).
*   **ReLU:** 결정 경계를 기점으로 전위가 0으로 꺾이는 비선형 굴절(Refraction) 현상 시각화.

---

## Part 2: [TRD] Technical Requirements Document
### "High-Performance Qt/Matplotlib Hybrid Architecture"

본 애플리케이션의 핵심 과제는 **"파이썬의 GIL(Global Interpreter Lock)과 Matplotlib의 무거운 렌더링 파이프라인 하에서도, 사용자가 슬라이더를 빠르게 드래그할 때 60FPS 급의 반응성을 유지하는 것"**입니다. 이를 위해 극한의 아키텍처 설계를 적용합니다.

### 1. 시스템 아키텍처 및 디자인 패턴 (Strict MVI Pattern)
단방향 데이터 흐름을 강제하는 **MVI (Model-View-Intent)** 패턴을 도입합니다.
*   **Model:** 애플리케이션의 상태(Weights, Bias, Dataset, Hyperparameters)를 보유하는 순수 Python Dataclass. UI에 대한 종속성이 전혀 없어야 함.
*   **Intent (Controller):** UI(슬라이더 변경, 캔버스 드래그)에서 발생하는 이벤트를 수신하여 Model을 업데이트하는 로직.
*   **View:** Model의 `on_state_changed` 시그널을 구독하여 스스로를 다시 그리는 수동적인 렌더러.

### 2. 렌더링 최적화: Matplotlib Blitting (핵심 기술)
슬라이더를 움직일 때마다 `canvas.draw()`를 호출하면 프레임이 5FPS 이하로 떨어집니다.
*   **전략:** 배경(축, 레이블, 빈 격자)은 프로그램 시작 시 한 번만 렌더링하여 `canvas.copy_from_bbox()`로 픽셀 버퍼에 캡처해 둡니다.
*   **동적 렌더링:** 슬라이더가 움직이면 캡처된 배경 버퍼를 복원(`restore_region`)하고, 변화하는 객체(Contour, Line, Points)만 다시 그린 후, `canvas.blit()`을 호출하여 화면에 쏩니다. 이를 통해 $O(10)$ ms 이내의 렌더링 속도를 달성합니다.

### 3. 상태 관리 및 이벤트 제어 (Throttling & Signal Debouncing)
*   **Event Flooding 방지:** `QSlider`의 `valueChanged` 시그널이 매우 짧은 주기로 발생할 때 메인 스레드가 멈추는 것을 방지합니다.
*   **구현:** 사용자 지정 `ThrottledSignal` 클래스를 작성. `QTimer`를 사용하여 이벤트가 발생하더라도 최소 16ms(약 60FPS 주기)가 지나기 전에는 Model 업데이트 및 Re-render 이벤트를 무시(Drop)하거나 지연(Delay)시킵니다.

### 4. 수치 안정성 및 에러 핸들링 (Numerical Stability)
*   **Singularity (특이점) 방지:** 가중치 $w_1 = 0, w_2 = 0$이 되는 순간 결정 경계선은 정의되지 않으며, 거리를 구하는 공식 분모에 0이 들어가 수학적 파국(ZeroDivisionError)이 발생합니다.
    *   **해결책:** 슬라이더의 물리적 최소값은 0을 허용하되, 내부 모델 연산 시 `np.clip(w, -max, max)` 및 `w = np.where(w==0, 1e-7, w)` 처리로 특이점을 우회합니다.
*   **Sigmoid Overflow:** `np.exp(-z)` 연산 시 $z < -500$ 수준이 되면 `OverflowError` 발생.
    *   **해결책:** $z$ 배열을 `np.clip(z, -250, 250)`으로 사전 스케일링 후 ufunc 연산.

### 5. 컴포넌트 계층 구조 및 구체적 클래스 매핑 (Detailed Object Tree)
```python
class MainWindow(QMainWindow):
    def __init__(self):
        # 1. State Initialization
        self.model = PerceptronModel()
        
        # 2. UI Layout Setup (Dark Theme Applied)
        self.splitter_main = QSplitter(Qt.Horizontal)
        
        # -- 2.1. Left Panel: Interaction & Controls
        self.panel_controls = QWidget()
        self.layout_controls = QVBoxLayout()
        self.slider_w1 = ReactiveSlider(min=-10.0, max=10.0, step=0.1)
        self.slider_w2 = ReactiveSlider(min=-10.0, max=10.0, step=0.1)
        self.slider_b = ReactiveSlider(min=-10.0, max=10.0, step=0.1)
        self.combo_activation = QComboBox() # ['Sigmoid', 'ReLU', 'Step', 'Tanh']
        self.btn_run_pla = QPushButton("Execute Gradient Descent")
        
        # -- 2.2. Right Panel: Visualizations
        self.panel_viz = QWidget()
        self.layout_viz = QVBoxLayout()
        
        self.label_math = QLabel() # Rendered via MathJax/LaTeX
        self.splitter_viz = QSplitter(Qt.Vertical)
        
        self.canvas_2d = Canvas2D() # Matplotlib FigureCanvasQTAgg (Blitting enabled)
        self.canvas_3d = Canvas3D() # Matplotlib FigureCanvasQTAgg (mplot3d)
        
        # 3. Signal-Slot Connections
        self.slider_w1.valueChanged.connect(self.intent_update_weights)
        self.model.state_changed.connect(self.canvas_2d.render)
        self.model.state_changed.connect(self.canvas_3d.render)
        
        # 4. Secondary Windows
        self.dialog_loss_landscape = LossLandscapeAnalyzer(self.model)
```

### 6. PLA 알고리즘 구현 및 무한 루프 차단 (Edge Case)
*   `self.btn_run_pla` 클릭 시 백그라운드 스레드(`QThread`)에서 Perceptron Learning Algorithm 수행.
*   **스레드 통신:** 학습 도중 가중치가 변할 때마다 `QThread`가 PyQt Signal을 방출하여 메인 스레드의 UI를 업데이트 (UI 멈춤 방지).
*   **Non-separable Halt:** XOR 데이터 등 선형 분리가 불가능할 경우 무한 루프에 빠지는 것을 막기 위해, `max_epochs = 1000` 설정과 함께 최근 $N$ 에포크 동안 Loss 값의 분산이 특정 임계치 이하(진동 상태)일 경우 강제 `break` 및 사용자에게 알림 토스트 발생.