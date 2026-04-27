# Software Design Description
## PhysicsAI Simulator — System Architecture
**Document ID:** SDD-PHYSAI-001  
**Version:** 2.0  
**Date:** 2026-04-06  
**Standard:** IEEE Std 1016-2009  
**Parent Document:** SRS-PHYSAI-001  
**Status:** Approved  

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-04-06 | JSW | Initial draft |
| 2.0 | 2026-04-06 | JSW | IEEE 1016 restructure, viewpoint model added |

---

## 1. Introduction

### 1.1 Purpose

This Software Design Description (SDD) defines the architectural design of PhysicsAI Simulator. It describes the system from four viewpoints (Logical, Process, Physical, Development), documents design decisions with explicit rationale, and provides the component decomposition required for implementation.

### 1.2 Scope

This document covers the top-level system architecture, inter-component dependencies, module interfaces, concurrency model, and deployment strategy. Detailed component designs are specified in SDD-PHYSAI-002 (UI), SDD-PHYSAI-003 (Physics/ML), and SDD-PHYSAI-004 (Visualization).

### 1.3 Context

PhysicsAI Simulator is a single-process desktop application. It executes on a user's local machine without network dependencies. The primary design challenge is integrating three computationally intensive subsystems — TensorFlow model training, NumPy-based physics simulation, and Matplotlib rendering — within a responsive PySide6 event loop.

### 1.4 Design Viewpoints

This document employs the **4+1 View Model** (Kruchten, 1995):

| View | Description | Primary Audience |
|------|-------------|-----------------|
| Logical | Decomposition into classes and packages | Developers |
| Process | Concurrency, threads, synchronization | Developers, QA |
| Physical | Deployment environment and platform | DevOps, QA |
| Development | Source code organization, build | Developers |
| Scenarios | Key use case walkthroughs | All |

### 1.5 References

Inherits all references from SRS-PHYSAI-001. Additionally:  
[15] IEEE Std 1016-2009, *IEEE Standard for Information Technology — Systems Design — Software Design Descriptions*, IEEE, 2009.  
[16] P. Kruchten, "The 4+1 View Model of Architecture," *IEEE Software*, vol. 12, no. 6, pp. 42–50, 1995.  
[17] E. Gamma et al., *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley, 1994.  

---

## 2. System Architecture Overview

### 2.1 Architectural Style

PhysicsAI Simulator employs a **layered architecture** with event-driven inter-layer communication via Qt Signal/Slot:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Presentation (PySide6)                            │
│  MainWindow, ModuleTabs, ParamPanel, PlotCanvas, AnimCanvas │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Application Logic                                  │
│  BaseModule subclasses, ModuleStateManager                  │
├───────────────────────────┬─────────────────────────────────┤
│  Layer 2a: ML Engine      │  Layer 2b: Physics Engine       │
│  ModelFactory             │  RK4Integrator                  │
│  DataGenerators           │  ProjectilePhysics              │
│  TrainingWorker           │  PendulumPhysics                │
├───────────────────────────┴─────────────────────────────────┤
│  Layer 1: Infrastructure                                     │
│  TensorFlow/Keras, NumPy, Matplotlib, QThread               │
└─────────────────────────────────────────────────────────────┘
```

**Cross-layer communication rule:** Layers communicate only with adjacent layers. Layer 3 does not directly manipulate Layer 4 widgets; it emits signals. Layer 2 does not access Layer 4. This enforces testability (Layers 1–2 have no Qt dependency and can be unit-tested without a QApplication).

### 2.2 Architectural Rationale

**Decision: Layered architecture over direct MVC**  
Alternatives considered:

| Alternative | Reason Rejected |
|-------------|-----------------|
| Pure MVC | Qt's Signal/Slot mechanism already provides observer pattern; a separate Model layer would duplicate state management |
| MVVM | Overhead unjustified for a single-user desktop app with 5 modules |
| Event-driven only (no layers) | Physics and ML code would become entangled with PySide6 widget code; untestable |

**Chosen:** Layered with Signal/Slot at the Layer 3↔4 boundary. Layers 1–2 are pure Python (NumPy/TF) with no Qt imports, enabling isolated unit testing.

### 2.3 Design Patterns Applied

| Pattern | Location | Purpose |
|---------|----------|---------|
| Template Method [17] | `BaseModule.run()` | Defines training skeleton; subclasses override `_build_model()`, `_generate_data()` |
| Factory Method [17] | `ModelFactory` | Centralizes Keras model construction; decouples module logic from architecture details |
| Observer (via Signal/Slot) | `TrainingWorker` → Module | Decouples training thread from UI update logic |
| Strategy | `PhysicsEngine` subclasses | Swappable physics implementations (vacuum vs. air resistance) |
| State | `ModuleStateManager` | Explicit state machine per module (IDLE/TRAINING/TRAINED/ERROR) |

---

## 3. Logical View — Module Decomposition

### 3.1 Package Structure

```
physicsai/
├── main.py
├── app/
│   ├── main_window.py           # MainWindow, tab registration
│   ├── modules/
│   │   ├── base_module.py       # BaseModule abstract class
│   │   ├── mod01_function.py    # FunctionApproximationModule
│   │   ├── mod02_projectile.py  # ProjectileRegressionModule
│   │   ├── mod03_overfitting.py # OverfittingDemoModule
│   │   ├── mod04_pendulum.py    # PendulumModule
│   │   └── mod05_air_resist.py  # AirResistanceModule
│   ├── widgets/
│   │   ├── slider_spinbox.py    # SliderSpinBox composite widget
│   │   ├── param_group.py       # ParamGroup (QGroupBox wrapper)
│   │   ├── progress_panel.py    # ProgressPanel
│   │   ├── matplotlib_widget.py # MatplotlibWidget
│   │   └── pendulum_canvas.py   # PendulumCanvas (QPainter)
│   ├── physics/
│   │   ├── rk4.py              # rk4_step(), rk4_integrate()
│   │   ├── projectile.py       # ProjectilePhysics
│   │   └── pendulum.py         # PendulumPhysics
│   ├── ml/
│   │   ├── models.py           # ModelFactory
│   │   ├── data_generators.py  # DataGenerators
│   │   └── training_worker.py  # TrainingWorker, QtProgressCallback
│   └── utils/
│       ├── theme.py            # ThemeManager
│       ├── export.py           # ExportManager
│       └── platform_utils.py   # Font detection, OS-specific utilities
├── tests/
│   ├── unit/
│   │   ├── test_rk4.py
│   │   ├── test_physics.py
│   │   └── test_models.py
│   └── integration/
│       ├── test_training_worker.py
│       └── test_module_lifecycle.py
└── requirements.txt
```

### 3.2 Key Class Responsibilities

**`BaseModule(QWidget)` — Abstract**

| Method | Responsibility |
|--------|---------------|
| `_setup_ui()` | Build widget layout; called once in `__init__` |
| `_connect_signals()` | Wire internal signals to slots; called once in `__init__` |
| `_build_model() → keras.Model` | Construct and compile the Keras model |
| `_generate_data() → Tuple[ndarray, ndarray]` | Produce training data |
| `_get_training_config() → TrainingConfig` | Return epoch/batch/callbacks configuration |
| `run()` | Template method: calls _build_model, _generate_data, starts TrainingWorker |
| `stop()` | Requests graceful worker termination |
| `reset()` | Resets state, clears plots, restores default parameters |
| `_on_progress(epoch, loss, val_loss)` | Slot: updates ProgressBar and Loss label |
| `_on_training_finished(model, history)` | Slot: stores model, triggers plot update |
| `_on_training_error(msg)` | Slot: displays error, transitions to ERROR state |

**`ModuleStateManager`**

```
States: IDLE → TRAINING → TRAINED
                    ↓
                STOPPED → IDLE
                    ↓
                  ERROR → IDLE
```

Transitions are guarded: `TRAINING → TRAINED` requires the model parameter count to be non-zero.

**`TrainingConfig` (dataclass)**

```python
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    use_reduce_lr: bool = True
    use_early_stopping: bool = False
    early_stopping_patience: int = 500
    log_interval: int = 10  # epochs between signal emissions
```

---

## 4. Process View — Concurrency Model

### 4.1 Thread Architecture

```
Main Thread (Qt Event Loop)
│
├── QTimer: pendulum animation (~30 Hz)
├── Signal/Slot handling: UI updates
└── Matplotlib draw_idle() calls

Worker Threads (QThread subclasses)
├── TrainingWorker (one per module, active during training)
│   └── Emits: progress_updated, training_finished, training_error
└── LoadingWorker (one per module, during data generation)
    └── Emits: data_ready, generation_progress
```

### 4.2 Thread Safety Rules

**Rule T-01:** `canvas.draw_idle()` and all widget state mutations shall execute exclusively on the main thread, invoked via Signal/Slot connections with `Qt.AutoConnection` (default).

**Rule T-02:** `TrainingWorker` shall not hold references to any `QWidget` instance. All data transfer is via Signal payloads (primitive types, numpy arrays, dict objects).

**Rule T-03:** NumPy arrays passed via signals shall be passed by value (`.copy()`) if the worker continues to use them after emission, to prevent data races.

**Rule T-04:** `_stop_flag` in `TrainingWorker` shall be a standard Python `bool`, not a threading primitive. The Keras `on_epoch_end` callback runs synchronously within the worker thread, so no lock is required.

**Rule T-05:** MOD-03 spawns three `TrainingWorker` instances simultaneously. These workers share no mutable state; they train separate model instances on separate data copies.

### 4.3 Signal/Slot Interface Contract

| Signal | Emitter | Payload Types | Receiver Slot |
|--------|---------|---------------|---------------|
| `progress_updated` | TrainingWorker | `(int, float, float)` = epoch, loss, val_loss | `BaseModule._on_progress` |
| `training_finished` | TrainingWorker | `(keras.Model, keras.callbacks.History)` | `BaseModule._on_training_finished` |
| `training_error` | TrainingWorker | `(str,)` = error message | `BaseModule._on_training_error` |
| `data_ready` | LoadingWorker | `(np.ndarray, np.ndarray)` = X, y | Module-specific slot |
| `generation_progress` | LoadingWorker | `(int,)` = samples completed | `ProgressPanel.update_generation` |

### 4.4 Application Shutdown Sequence

```
1. User closes MainWindow
2. MainWindow.closeEvent() called
3. For each active TrainingWorker:
   a. Call worker.request_stop()
   b. Call worker.wait(timeout=2000 ms)
   c. If still running: call worker.terminate() (last resort)
4. For each active animation QTimer: timer.stop()
5. QApplication.quit()
```

---

## 5. Physical View — Deployment

### 5.1 Target Environment

```
┌─────────────────────────────────────────┐
│  User Workstation                        │
│  ┌─────────────────────────────────┐    │
│  │  CPython 3.10+                  │    │
│  │  ┌────────────┐ ┌────────────┐  │    │
│  │  │  PySide6   │ │ TensorFlow │  │    │
│  │  │  (Qt6 Qt)  │ │ (CPU/GPU)  │  │    │
│  │  └────────────┘ └────────────┘  │    │
│  │  ┌────────────┐ ┌────────────┐  │    │
│  │  │  NumPy     │ │ Matplotlib │  │    │
│  │  └────────────┘ └────────────┘  │    │
│  └─────────────────────────────────┘    │
│                                          │
│  ┌──────────┐  ┌──────────┐             │
│  │  CPU     │  │  GPU     │ (optional)  │
│  └──────────┘  └──────────┘             │
└─────────────────────────────────────────┘
```

### 5.2 Installation Method

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Launch
python main.py
```

**requirements.txt:**
```
PySide6>=6.6.0,<7.0.0
tensorflow>=2.15.0,<3.0.0
numpy>=1.26.0,<2.0.0
matplotlib>=3.8.0,<4.0.0
```

### 5.3 Entry Point — main.py

```python
# Entry point: must configure matplotlib backend before Qt is imported
import matplotlib
matplotlib.use('QtAgg')

import sys
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow
from app.utils.theme import ThemeManager
from app.utils.platform_utils import configure_korean_font

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PhysicsAI Simulator")
    app.setApplicationVersion("1.0.0")

    # Platform-level setup
    configure_korean_font()
    ThemeManager.apply_light(app)

    # TensorFlow GPU setup (non-blocking)
    _configure_tensorflow()

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

def _configure_tensorflow():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] TensorFlow GPU config failed: {e}. CPU mode active.")

if __name__ == '__main__':
    main()
```

---

## 6. Development View — Build and Test

### 6.1 Module Registration

New simulation modules are added without modifying existing module files:

```python
# app/main_window.py

from app.modules.mod01_function import FunctionApproximationModule
from app.modules.mod02_projectile import ProjectileRegressionModule
from app.modules.mod03_overfitting import OverfittingDemoModule
from app.modules.mod04_pendulum import PendulumModule
from app.modules.mod05_air_resist import AirResistanceModule

MODULES = [
    ("1D Function Approximation", FunctionApproximationModule),
    ("Projectile Motion",         ProjectileRegressionModule),
    ("Overfitting Demo",          OverfittingDemoModule),
    ("Pendulum Simulation",       PendulumModule),
    ("Air Resistance",            AirResistanceModule),
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tabs = QTabWidget()
        for label, ModuleClass in MODULES:
            self.tabs.addTab(ModuleClass(), label)
        self.setCentralWidget(self.tabs)
```

### 6.2 Testing Strategy

| Test Level | Scope | Tools | Coverage Target |
|------------|-------|-------|-----------------|
| Unit | Physics functions (rk4, projectile, pendulum), ModelFactory | pytest | 90%+ of physics/ml packages |
| Integration | TrainingWorker signal emission, data generators | pytest + pytest-qt | All Signal/Slot paths |
| System | Full module lifecycle (Run → Stop → Reset) | pytest-qt | 5 modules × 3 lifecycle paths |
| Performance | RK4 speed, draw_idle latency | cProfile + QElapsedTimer | NFR-01 through NFR-05 |

**Key test: physics correctness**
```python
# tests/unit/test_physics.py

def test_rk4_energy_conservation():
    physics = PendulumPhysics(L=1.0)
    traj = physics.simulate(theta0_deg=45.0, n_periods=3.0, dt=0.01)
    # Check energy: E = 0.5*ω² + (g/L)*(1 - cos(θ))
    E_initial = compute_energy(traj[0])
    E_final = compute_energy(traj[-1])
    assert abs(E_final - E_initial) / E_initial < 0.001  # < 0.1% drift

def test_rk4_vacuum_projectile():
    physics = ProjectilePhysics(k=0.0)
    traj = physics.simulate(v0=30.0, angle_deg=45.0)
    R_sim = traj[-1, 0]
    R_theory = physics.vacuum_range(30.0, 45.0)
    assert abs(R_sim - R_theory) < 0.01  # < 1 cm error
```

---

## 7. Scenarios — Key Use Case Walkthroughs

### 7.1 Scenario: Run MOD-01 Training

```
1. User launches app → MainWindow.show() → MOD-01 tab selected by default
2. User selects F-02 from function ComboBox
   → FunctionApproximationModule._on_function_changed()
   → Clears plot, regenerates preview data
3. User clicks [Run]
   → Module.run() called
   → _build_model() → ModelFactory.function_approximator([128,128], 'tanh', 0.01)
   → _generate_data() → DataGenerators.function_approximation('cos(x)+0.5sin(2x)')
   → TrainingWorker(model, X, y, config).start()
   → Module state: IDLE → TRAINING
   → Run button disabled; Stop button enabled
4. Every 10 epochs: TrainingWorker emits progress_updated(epoch, loss, val_loss)
   → Main thread slot: ProgressBar.setValue(), LossLabel.setText()
   → Loss curve updated via canvas.draw_idle()
5. Training completes: TrainingWorker emits training_finished(model, history)
   → Module._on_training_finished()
   → Computes y_pred on x_test
   → Updates all three subplots
   → Module state: TRAINING → TRAINED
   → Stop disabled; Run and Reset enabled
```

### 7.2 Scenario: Change k in MOD-05 After Training

```
1. Module is in TRAINED state, k=0.05
2. User drags k slider to 0.10
   → _on_k_changed(0.10) called
   → Warning label shown: "Drag coefficient changed. Retraining required."
   → RK4 re-simulation runs immediately with new k
   → Trajectory plot updated (air resistance curve changes)
   → AI prediction marker remains (stale, from k=0.05 model)
   → Run button highlighted amber (dirty state)
3. User clicks [Run]
   → Full data regeneration with k=0.10, retrain, redraw
```

---

## 8. Non-Functional Design Decisions

| NFR ID | Design Decision |
|--------|----------------|
| NFR-01 (launch < 5s) | TensorFlow import deferred to first Run click (lazy import in TrainingWorker); not imported at application start |
| NFR-02 (UI < 100ms during training) | All training in QThread; plot updates capped at 10-epoch intervals using draw_idle |
| NFR-03 (draw_idle < 33ms) | Figure sizes fixed at instantiation; tight_layout called only at figure creation, not on each update |
| NFR-06 (TF errors non-fatal) | try/except around TF import in _configure_tensorflow(); app continues in CPU-only mode |
| NFR-08 (clean shutdown) | closeEvent() iterates active workers, calls request_stop() + wait(2000) |
