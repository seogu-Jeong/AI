# Software Requirements Specification
## PhysicsAI Simulator — Product Overview
**Document ID:** SRS-PHYSAI-001  
**Version:** 2.0  
**Date:** 2026-04-06  
**Standard:** IEEE Std 830-1998  
**Status:** Approved  

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-04-06 | JSW | Initial draft |
| 2.0 | 2026-04-06 | JSW | IEEE 830 restructure, requirements formalized |

---

## Table of Contents

1. Introduction  
2. Overall Description  
3. External Interface Requirements  
4. System Features  
5. Non-Functional Requirements  
Appendix A: Glossary  
Appendix B: Requirements Traceability Matrix  
Appendix C: Related Work  
Appendix D: Open Issues  

---

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) defines the functional and non-functional requirements for **PhysicsAI Simulator**, a desktop application built with **PySide6** that integrates classical physics simulations with neural network-based regression models. The document governs the development of five interactive modules derived from the `week4` curriculum (scripts `01perfect1d.py` through `05_projectile_motion.py`).

This document is intended to serve as:
- A binding specification for development and verification activities
- A portfolio artifact demonstrating AI/ML system design competency
- An academic submission demonstrating requirements engineering methodology

### 1.2 Scope

**Product Name:** PhysicsAI Simulator  
**Version Covered:** 1.0.0  

The system shall provide an interactive PySide6 desktop application enabling users to:

1. Perform neural network function approximation on selected 1D mathematical functions
2. Regress projectile motion trajectories (with and without air resistance) using TensorFlow/Keras models
3. Demonstrate overfitting and underfitting phenomena through comparative model training
4. Predict and simulate pendulum oscillation periods using neural networks and RK4 numerical integration

The system does **not** cover: web deployment, cloud-based training, multi-user collaboration, or model persistence between sessions (deferred to v2.0).

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|------------|
| RK4 | Fourth-order Runge-Kutta numerical integration method with global truncation error O(h⁴) |
| UAT | Universal Approximation Theorem: a feedforward network with one hidden layer can approximate any continuous function on a compact subset of ℝⁿ (Cybenko, 1989) |
| MSE | Mean Squared Error: (1/n)Σ(yᵢ - ŷᵢ)² |
| MAE | Mean Absolute Error: (1/n)Σ\|yᵢ - ŷᵢ\| |
| MAPE | Mean Absolute Percentage Error: (1/n)Σ\|yᵢ - ŷᵢ\|/\|yᵢ\| × 100% |
| PySide6 | Qt for Python 6.x, the official Python binding for Qt6 framework |
| QThread | PySide6 thread class enabling concurrent execution without blocking the event loop |
| Signal/Slot | Qt's type-safe inter-object communication mechanism |
| FigureCanvasQTAgg | Matplotlib's Qt-compatible Agg rendering canvas |
| GPU | Graphics Processing Unit; used for TensorFlow CUDA acceleration when available |
| FR | Functional Requirement |
| NFR | Non-Functional Requirement |
| MOD-0x | Module identifier (01–05 corresponding to week4 scripts) |

### 1.4 References

[1] IEEE Std 830-1998, *IEEE Recommended Practice for Software Requirements Specifications*, IEEE, 1998.  
[2] G. Cybenko, "Approximation by superpositions of a sigmoidal function," *Mathematics of Control, Signals and Systems*, vol. 2, no. 4, pp. 303–314, 1989.  
[3] K. Hornik, M. Stinchcombe, and H. White, "Multilayer feedforward networks are universal approximators," *Neural Networks*, vol. 2, no. 5, pp. 359–366, 1989.  
[4] J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*, 3rd ed. Wiley, 2016.  
[5] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.  
[6] Qt Company, *Qt for Python (PySide6) Documentation*, https://doc.qt.io/qtforpython-6/, 2024.  
[7] TensorFlow Team, *TensorFlow 2.x Documentation*, https://www.tensorflow.org/api_docs, 2024.  
[8] J. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007.  

### 1.5 Document Overview

Section 2 establishes the product context and constraints. Section 3 specifies external interface requirements. Section 4 enumerates system features with traceable functional requirements. Section 5 defines non-functional requirements with measurable acceptance criteria. Appendices provide traceability, glossary, and related work.

---

## 2. Overall Description

### 2.1 Product Perspective

PhysicsAI Simulator operates as a standalone desktop application. It interfaces with the host operating system's GPU/CPU resources through TensorFlow and renders physics visualizations via Matplotlib embedded in PySide6 widgets.

**System Context Diagram:**

```
┌─────────────────────────────────────────────────────────────┐
│                    PhysicsAI Simulator                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              PySide6 Presentation Layer              │    │
│  │  MainWindow → ModuleTabs → ParamPanel + PlotCanvas   │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                          │ Signals / Slots                   │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │               Application Logic Layer                │    │
│  │   ModuleController → TrainingWorker (QThread)        │    │
│  └────────────┬────────────────────────┬───────────────┘    │
│               │                        │                     │
│  ┌────────────▼──────┐    ┌────────────▼───────────────┐    │
│  │   ML Engine       │    │   Physics Engine            │    │
│  │   (TensorFlow)    │    │   (NumPy + RK4)             │    │
│  └───────────────────┘    └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         │                                    │
┌────────▼────────┐                 ┌─────────▼────────┐
│  Host OS        │                 │  Filesystem       │
│  GPU/CPU        │                 │  (PNG/JSON export)│
└─────────────────┘                 └──────────────────┘
```

### 2.2 Product Functions Summary

| Module | ID | Input | Output | ML Task |
|--------|----|-------|--------|---------|
| 1D Function Approximation | MOD-01 | Function selection, architecture | Approximation plot, loss curve | Regression |
| Projectile Motion Regression | MOD-02 | v₀, θ, training config | Trajectory prediction | Regression |
| Overfitting Demonstration | MOD-03 | Noise level, sample count | 3-model comparison | Classification of fit quality |
| Pendulum Period Prediction | MOD-04 | L, θ₀, training config | Period prediction, RK4 animation | Regression |
| Air Resistance Projectile | MOD-05 | v₀, θ, drag coefficient k | Range prediction, trajectory comparison | Regression |

### 2.3 User Classes and Characteristics

**UC-01: AI/ML Student (Primary)**
- Background: Undergraduate or graduate-level familiarity with Python and basic neural networks
- Goal: Observe how neural networks approximate physical laws; understand training dynamics
- Technical proficiency: Can interpret loss curves and MSE metrics
- Frequency of use: Multiple sessions per week during coursework

**UC-02: Physics Educator (Secondary)**
- Background: Domain expertise in classical mechanics; limited ML knowledge
- Goal: Demonstrate physical phenomena interactively without writing code
- Technical proficiency: Comfortable with GUIs; not required to understand ML internals
- Frequency of use: Per-lecture sessions

**UC-03: ML Researcher (Tertiary)**
- Background: Deep expertise in neural networks and numerical methods
- Goal: Use as rapid prototyping tool for physics-informed neural network experiments
- Technical proficiency: Expert; will inspect model architecture and training metrics closely
- Frequency of use: Ad hoc, exploratory sessions

### 2.4 Operating Environment

| Constraint | Specification |
|------------|---------------|
| Operating System | macOS 13.0+, Windows 10 (22H2)+, Ubuntu 22.04 LTS |
| Python Runtime | CPython 3.10 – 3.12 |
| PySide6 | 6.6.0 or later |
| TensorFlow | 2.15.0 or later (CPU mandatory; GPU optional via CUDA 11.8+) |
| NumPy | 1.26.0 or later |
| Matplotlib | 3.8.0 or later |
| Display | Minimum 1280 × 800; HiDPI supported |
| RAM | Minimum 4 GB; 8 GB recommended for concurrent model training (MOD-03) |

### 2.5 Design and Implementation Constraints

**CON-01:** All neural network models shall be constructed using the TensorFlow 2.x Keras Sequential API to maintain educational clarity and reproducibility.  
**CON-02:** All GUI components shall be implemented using PySide6 exclusively. No hybrid tkinter, wx, or web-based rendering is permitted.  
**CON-03:** Physics simulation (RK4 integration) shall be implemented in pure NumPy without external ODE solvers (e.g., SciPy) to ensure algorithmic transparency.  
**CON-04:** Model training shall execute in a `QThread` to prevent blocking the Qt event loop.  
**CON-05:** All Matplotlib rendering shall occur on the main thread via `canvas.draw_idle()`.  
**CON-06:** The application shall not require an internet connection at runtime.  

### 2.6 User Documentation

The system shall ship with:
- Inline parameter tooltips (accessible via hover)
- Module-level help dialogs describing the underlying physics and ML concepts
- A `README.md` covering installation and launch instructions

### 2.7 Assumptions and Dependencies

**ASM-01:** The host machine has Python 3.10+ and pip installed.  
**ASM-02:** TensorFlow GPU acceleration is treated as a performance enhancement, not a requirement. The application shall function correctly on CPU-only systems.  
**ASM-03:** Training data is generated synthetically at runtime from physical models; no external datasets are required.  
**DEP-01:** `matplotlib` version 3.8+ is required for `FigureCanvasQTAgg` compatibility with PySide6 6.6+.  
**DEP-02:** The `QtAgg` matplotlib backend must be set before `QApplication` is instantiated.  

---

## 3. External Interface Requirements

### 3.1 User Interfaces

**UI-01:** The application shall present a `QMainWindow` with a minimum window size of 1200 × 800 pixels, containing a `QTabWidget` with five module tabs.  
**UI-02:** Each module tab shall follow a two-panel layout: a fixed-width (280 px) parameter panel on the left and a resizable plot area on the right.  
**UI-03:** All interactive parameters shall be controlled via `QSlider` paired with `QDoubleSpinBox`, ensuring both mouse-drag and keyboard entry are supported.  
**UI-04:** The application shall provide a persistent status bar displaying: (a) current module state, (b) GPU/CPU mode, and (c) active session duration.  
**UI-05:** A `QProgressBar` shall indicate training progress (0–100%) within each module's parameter panel during active training.  

### 3.2 Software Interfaces

**SI-01 (TensorFlow):** The ML engine shall interface with TensorFlow 2.x via the `tensorflow.keras` high-level API. Direct use of `tf.GradientTape` is prohibited in production code to maintain API stability.  
**SI-02 (Matplotlib):** Visualization shall use `matplotlib.backends.backend_qtagg.FigureCanvasQTAgg` as the rendering surface. The backend shall be configured as `'QtAgg'` before Qt application initialization.  
**SI-03 (NumPy):** All numerical computation outside of TensorFlow (physics simulation, data generation) shall use NumPy array operations. Python loops over large arrays are prohibited for performance-critical paths.  
**SI-04 (Filesystem):** Exported PNG files shall default to `~/Downloads/<module>_<timestamp>.png`. Exported JSON results shall follow the schema defined in TRD-04.  

### 3.3 Hardware Interfaces

**HI-01:** The application shall detect available GPU devices via `tf.config.list_physical_devices('GPU')` at startup and display the result in the status bar.  
**HI-02:** On systems with a detected GPU, memory growth shall be enabled via `tf.config.experimental.set_memory_growth(device, True)` to prevent full VRAM allocation.  
**HI-03:** The application shall function correctly without GPU hardware, with training executing on the CPU fallback.  

---

## 4. System Features

### 4.1 Module Lifecycle Management

**Description:** The application shall manage the lifecycle (Idle → Training → Trained → Reset) of each simulation module independently.  
**Priority:** High  

**Functional Requirements:**

| ID | Requirement |
|----|-------------|
| FR-01 | The system shall maintain an independent state machine for each module with states: `IDLE`, `TRAINING`, `TRAINED`, `ERROR`. |
| FR-02 | While a module is in `TRAINING` state, its Run button shall be disabled and its Stop button shall be enabled. |
| FR-03 | A module in `TRAINING` state shall transition to `IDLE` upon receiving a stop request, preserving any partially trained model if ≥10% of epochs completed. |
| FR-04 | A module in `ERROR` state shall display the error message inline and offer a Reset action. |
| FR-05 | Module state shall be preserved when switching between tabs; re-selecting a `TRAINED` module shall restore the last plot. |

### 4.2 Parameter Control

**Description:** Users shall control simulation parameters through synchronized slider and spinbox widgets.  
**Priority:** High  

| ID | Requirement |
|----|-------------|
| FR-06 | All `SliderSpinBox` widgets shall maintain two-way synchronization: slider drag updates spinbox value and vice versa, with a maximum propagation latency of 16 ms (one frame at 60 Hz). |
| FR-07 | Parameter changes shall trigger a visual "dirty" indicator on the Run button (e.g., color change to amber) when the current model was trained with different parameters. |
| FR-08 | Each parameter widget shall display its unit label (e.g., m/s, °, m) adjacent to the value field. |
| FR-09 | The Reset action shall restore all parameters to their documented default values as specified in PRD-02 through PRD-04. |

### 4.3 Training Execution

**Description:** Neural network training shall execute asynchronously without blocking the UI.  
**Priority:** High  

| ID | Requirement |
|----|-------------|
| FR-10 | Model training shall execute in a dedicated `QThread` instance per module. |
| FR-11 | Training progress shall be communicated to the main thread exclusively via Qt signals; direct widget manipulation from worker threads is prohibited. |
| FR-12 | The `QProgressBar` shall update every 10 epochs, showing `epoch / total_epochs × 100%`. |
| FR-13 | The current loss value shall be displayed as a `QLabel` and updated every 10 epochs with format `Loss: {loss:.6f}`. |
| FR-14 | Upon training completion, the elapsed time shall be displayed in the format `Completed in {ms}ms`. |
| FR-15 | A graceful stop mechanism shall set a `_stop_flag` on the worker, which the Keras callback checks at `on_epoch_end`, preventing abrupt thread termination. |

### 4.4 Visualization

**Description:** All simulation results shall be rendered as Matplotlib figures embedded in PySide6 via `FigureCanvasQTAgg`.  
**Priority:** High  

| ID | Requirement |
|----|-------------|
| FR-16 | Each module shall display at minimum: (a) a primary result plot, (b) a training loss curve. |
| FR-17 | All plots shall include axis labels with units, a legend, and a title containing the key performance metric (e.g., MSE). |
| FR-18 | The `NavigationToolbar2QT` toolbar shall be displayed above each plot canvas, enabling pan, zoom, and save operations. |
| FR-19 | Loss curves shall use a logarithmic y-axis scale when the loss range spans more than two orders of magnitude. |
| FR-20 | Plot updates during training shall use `canvas.draw_idle()` exclusively to prevent thread conflicts. |

### 4.5 Result Export

**Description:** Users shall be able to export plots and experimental parameters.  
**Priority:** Medium  

| ID | Requirement |
|----|-------------|
| FR-21 | The system shall provide a keyboard shortcut `Ctrl+E` and a toolbar button to export the currently visible plot as a PNG file at 150 DPI minimum. |
| FR-22 | The system shall export experimental parameters and final metrics (loss, MAE, MAPE where applicable) as a JSON file conforming to the schema in TRD-04 §8. |
| FR-23 | File save dialogs shall default to `~/Downloads/` with a pre-filled filename of the form `<module_id>_<YYYYMMDD_HHMMSS>.{png,json}`. |

---

## 5. Non-Functional Requirements

### 5.1 Performance Requirements

| ID | Requirement | Measurement Method |
|----|-------------|-------------------|
| NFR-01 | The application shall launch and display the main window within 5 seconds on a reference system (Intel Core i5, 8 GB RAM, SSD). | Wall-clock time from process start to `QMainWindow.show()` |
| NFR-02 | UI interactions (button clicks, slider drags) shall produce visible feedback within 100 ms during active training. | Qt event loop latency measurement |
| NFR-03 | `draw_idle()` calls shall not block the main thread for more than 33 ms (one frame at 30 Hz). | Profiler measurement on reference system |
| NFR-04 | MOD-03 (three concurrent training threads) shall complete without causing Qt event loop stalls exceeding 200 ms. | QElapsedTimer in event filter |
| NFR-05 | RK4 integration of pendulum trajectory (3 periods, dt=0.01) shall complete within 50 ms on the reference system. | `time.perf_counter()` benchmark |

### 5.2 Reliability and Error Handling

| ID | Requirement |
|----|-------------|
| NFR-06 | TensorFlow initialization failures (e.g., incompatible CUDA version) shall be caught at startup, logged to stderr, and presented to the user as a dismissable warning dialog; the application shall continue in CPU-only mode. |
| NFR-07 | Exceptions raised within `QThread.run()` shall be caught, formatted, and emitted via a `training_error` signal; they shall never propagate to crash the main thread. |
| NFR-08 | If a training worker is still running when the application is closed, the worker shall be requested to stop and joined with a 2-second timeout before `QApplication.quit()` is called. |

### 5.3 Usability

| ID | Requirement |
|----|-------------|
| NFR-09 | All interactive controls shall have tooltip text explaining the parameter's physical meaning and valid range. |
| NFR-10 | Font size shall be no smaller than 11pt across all widgets to meet basic readability standards. |
| NFR-11 | The application shall support both light and dark themes, toggled via `Ctrl+D`, with Matplotlib plot styles synchronized to the active theme. |
| NFR-12 | Keyboard shortcuts `1` through `5` shall navigate directly to the corresponding module tab. |

### 5.4 Software Quality Attributes

| Attribute | Requirement |
|-----------|-------------|
| Maintainability | Each module shall be implemented as a self-contained class inheriting from `BaseModule`; no module shall import from another module's implementation file. |
| Testability | Physics engine functions (`rk4_step`, `true_period`, `vacuum_range`) shall be pure functions with no side effects, enabling unit testing without a Qt context. |
| Portability | No platform-specific APIs shall be used except where required for font detection (Korean font fallback). All platform-specific code shall be isolated in `utils/platform.py`. |
| Scalability | Adding a new simulation module shall require only: (a) creating a new `ModuleBase` subclass, and (b) registering it in `main_window.py`. No other files shall require modification. |

### 5.5 Business Rules

**BR-01:** The application is for educational and portfolio use. No telemetry, analytics, or network calls shall be made at runtime.  
**BR-02:** All third-party dependencies are open-source with licenses compatible with MIT (PySide6: LGPL, TensorFlow: Apache 2.0, NumPy: BSD, Matplotlib: PSF-based).  

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| Drag coefficient (k) | Lumped aerodynamic drag term normalized by mass: k = C_d × A × ρ / (2m), units kg⁻¹ |
| Elliptic integral correction | Series approximation for large-angle pendulum period: T = T_small × (1 + θ₀²/16 + 11θ₀⁴/3072 + ...) |
| Phase space | 2D representation of pendulum state as (θ, ω); a closed orbit indicates energy conservation |
| Tanh activation | f(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ); preferred for periodic function approximation due to bounded, smooth, symmetric output |
| Universal Approximation Theorem | A network with one hidden layer and a non-polynomial activation function can approximate any continuous function on a compact domain to arbitrary precision [2][3] |
| Warm start | Resuming training from existing weights rather than random initialization; reduces convergence time for incremental parameter changes |

---

## Appendix B: Requirements Traceability Matrix

| FR ID | Description (Summary) | Module(s) | TRD Section |
|-------|----------------------|-----------|-------------|
| FR-01–FR-05 | Module lifecycle state machine | All | TRD-01 §6 |
| FR-06–FR-09 | Parameter control widgets | All | TRD-02 §3 |
| FR-10–FR-15 | Async training execution | All | TRD-03 §4 |
| FR-16–FR-20 | Visualization rendering | All | TRD-04 §2–§4 |
| FR-21–FR-23 | Export functionality | All | TRD-04 §5 |
| NFR-01–NFR-05 | Performance | All | TRD-01 §8 |
| NFR-06–NFR-08 | Error handling | All | TRD-03 §5 |

---

## Appendix C: Related Work

**Physics-Informed Neural Networks (PINNs):** Raissi et al. (2019) demonstrated that embedding physical laws as loss function terms significantly improves generalization in physics regression tasks. PhysicsAI Simulator adopts a data-driven (rather than physics-informed) approach intentionally, to contrast pure ML generalization against known analytical solutions — an educationally valuable comparison.

**Neural Network Function Approximation:** The seminal results of Cybenko [2] and Hornik et al. [3] establish theoretical justification for using feedforward networks to approximate the mathematical functions in MOD-01. Network depth and width selections in this project are informed by empirical results in Goodfellow et al. [5], Chapter 6.

**Numerical Integration in Physics Education:** The RK4 method used in MOD-04 and MOD-05 offers O(h⁴) global truncation error per Butcher [4], making it the standard choice for first-year physics simulation curricula. Its explicit form makes it pedagogically transparent compared to adaptive methods (e.g., Dormand-Prince).

---

## Appendix D: Open Issues

| ID | Issue | Priority | Resolution Target |
|----|-------|----------|-------------------|
| OI-01 | Korean font fallback on Linux requires `fc-list` call; behavior untested on Ubuntu 22.04 | Low | v1.1.0 |
| OI-02 | TF 2.16+ deprecates `keras.optimizers.Adam` in favor of `keras.optimizers.legacy.Adam`; migration path TBD | Medium | Pre-release |
| OI-03 | MOD-03 concurrent training with 3 QThreads may exceed RAM on systems with < 4 GB; graceful degradation undefined | High | v1.0.0 |
