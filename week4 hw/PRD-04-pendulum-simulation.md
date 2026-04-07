# Software Requirements Specification
## PhysicsAI Simulator — Module 04: Pendulum Simulation
**Document ID:** SRS-PHYSAI-004  
**Version:** 2.0  
**Date:** 2026-04-06  
**Standard:** IEEE Std 830-1998  
**Parent Document:** SRS-PHYSAI-001  
**Status:** Approved  

---

## 1. Introduction

### 1.1 Purpose

This SRS specifies requirements for **MOD-04: Pendulum Period Prediction**, a module that demonstrates neural network regression on a nonlinear physical system — the simple pendulum. It combines period prediction via a trained neural network with real-time RK4 motion simulation and an animated PySide6 visualization.

### 1.2 Scope

The module encompasses: period prediction accuracy (L, θ₀) → T, RK4-driven motion simulation, QPainter-based pendulum animation, phase space visualization, and educational comparison between the small-angle approximation and the exact (elliptic integral) solution.

### 1.3 Definitions

| Term | Definition |
|------|------------|
| Period T | Time for one complete oscillation; T > 0, units: seconds |
| Small-angle approximation | T_small = 2π√(L/g); valid for θ₀ ≲ 15°; error < 0.5% |
| Elliptic integral correction | T_exact = T_small × (1 + θ₀²/16 + 11θ₀⁴/3072 + ...); accurate to < 0.01% for θ₀ < 80° |
| Isochronism | Property that period is nearly independent of amplitude for small angles |
| Phase space | 2D state space (θ, ω) where ω = dθ/dt; closed orbits indicate energy conservation |
| Angular velocity ω | Rate of change of angle; ω = dθ/dt, units: rad/s |
| Pivot | Fixed suspension point of the pendulum |

### 1.4 Physical Laws

**Equation of motion (exact, nonlinear):**
```
d²θ/dt² = -(g/L)·sin(θ)
```

**State-space form for RK4:**
```
d[θ]/dt = [ω          ]
d[ω]/dt   [-(g/L)sin(θ)]
```

**Period formulas:**
```
T_small = 2π√(L/g)                                           [valid: θ₀ < 15°]
T_exact  = T_small × (1 + θ₀²/16 + 11θ₀⁴/3072 + ...)       [valid: θ₀ < 80°]
```

The nonlinearity (`sin(θ) ≠ θ` for large angles) makes T a function of both L and θ₀ — the machine learning target.

### 1.5 References

Inherits SRS-PHYSAI-001 references. Additionally:  
[13] G. Baker and J. Blackburn, *The Pendulum: A Case Study in Physics*. Oxford University Press, 2005.  
[14] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Chapter 17: Elliptic Integrals. Dover, 1964.  

---

## 2. Overall Description

### 2.1 Module Purpose and Educational Value

MOD-04 addresses a key pedagogical gap: most introductory physics courses teach only the small-angle approximation T ≈ 2π√(L/g), obscuring the amplitude dependence at large angles. This module:

1. Visually demonstrates the breakdown of the small-angle approximation
2. Shows that a neural network can learn the nonlinear (L, θ₀) → T relationship without being given the elliptic integral formula
3. Provides an animated simulation that makes the physical phenomenon concrete

### 2.2 Neural Network Learning Target

The regression target — period T as a function of (L, θ₀) — is nontrivial because:
- T depends on L through a square-root (nonlinear scaling)
- T depends on θ₀ through an infinite series (polynomial nonlinearity)
- The two dependencies are multiplicative (interaction term)

This makes it an ideal demonstration of neural network regression on a physics problem with a known analytical solution for validation.

---

## 3. External Interface Requirements

### 3.1 User Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  [Left Panel: 280px]    │  [Right Panel: Expandable]            │
│                         │                                        │
│  [GroupBox: Pendulum]   │  ┌────────────────┬───────────────┐  │
│   Length L: [═══] 1.0 m │  │ PendulumCanvas │  θ(t) Plot   │  │
│   Init Angle: [═══] 30° │  │ (QPainter)     │  (Matplotlib) │  │
│                         │  │                │               │  │
│  [GroupBox: Training]   │  └────────────────┴───────────────┘  │
│   Samples: [═══] 2000   │  ┌──────────────────────────────────┐ │
│   Epochs:  [═══] 100    │  │     Phase Space Plot (θ vs ω)    │ │
│   Noise σ: [═══] 0.010  │  └──────────────────────────────────┘ │
│                         │                                        │
│  [▶ Run] [■ Stop] [↺]  │  ┌──────────────────────────────────┐ │
│  [▶ Animate]  [⏸ Pause] │  │    Period Prediction Panel       │ │
│  Speed: [0.25x──────4x] │  │  T_small: X.XXX s  (err: X.X%)  │ │
│                         │  │  T_exact: X.XXX s               │ │
│  [▓▓▓▓▓░░░░] 50%        │  │  T_pred:  X.XXX s  (err: X.X%)  │ │
│  Loss: 0.000423         │  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 PendulumCanvas Requirements

The `PendulumCanvas` shall be a `QWidget` subclass using `QPainter` for all rendering. It shall not use Matplotlib.

| Element | Specification |
|---------|---------------|
| Background | White (light theme) or dark gray (dark theme) |
| Pivot point | Filled circle, 6px radius, black/white |
| Rod | Line from pivot to bob, width 2px, dark gray |
| Bob | Filled circle, 16px radius, `#2196F3` (Material Blue) |
| Angle arc | Dashed arc from vertical to current angle, radius 40px, gray |
| Angle label | Text "θ = XX.X°" adjacent to arc, 10pt |
| Trail | Last 30 bob positions connected by lines; alpha fades from 0 to 255 |
| Scale | Rod length in pixels = min(width, height) × 0.35 × (L_pixels/L_max) |

---

## 4. System Features

### 4.1 Period Prediction Model

**Description:** Neural network trained to predict oscillation period from pendulum parameters.  
**Priority:** High  

**Architecture specification:**
```
Input(2)  →  Dense(64, relu)  →  Dropout(0.1)
          →  Dense(32, relu)  →  Dropout(0.1)
          →  Dense(16, relu)  →  Dropout(0.1)
          →  Dense(1, linear)
```

**Architectural rationale:**
- Input dimension 2: (L, θ₀)
- Output dimension 1: scalar period T > 0
- relu hidden layers: the L → T relationship (√L) is monotonically increasing; relu captures this well
- Linear output: T is unbounded positive; no output activation constraint
- MAPE metric: percentage error is more meaningful than absolute error given T varies from ~1s to ~5s
- Adam lr=0.001: standard; ReduceLROnPlateau applied

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD04-01 | The model shall accept input shape (None, 2): (L in meters, θ₀ in degrees) and produce output shape (None, 1): period T in seconds. | High |
| FR-MOD04-02 | The model shall be compiled with loss='mse', metrics=['mae', 'mape']. | High |
| FR-MOD04-03 | Training data: L ∈ U[0.5, 3.0] m, θ₀ ∈ U[5°, 80°], n=2000; periods computed via elliptic integral approximation with multiplicative noise N(1, 0.01). | High |
| FR-MOD04-04 | A `ReduceLROnPlateau` callback shall be attached: factor=0.5, patience=20, min_lr=1e-5. | Medium |

### 4.2 Period Prediction Panel

**Description:** Real-time display comparing three period estimates.  
**Priority:** High  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD04-05 | The period panel shall display three values simultaneously: T_small (small-angle formula), T_exact (elliptic integral), and T_pred (neural network). | High |
| FR-MOD04-06 | For T_small, the relative error vs. T_exact shall be shown: `err = |T_small - T_exact| / T_exact × 100%`. | High |
| FR-MOD04-07 | For T_pred, the relative error vs. T_exact shall be shown with identical formatting. | High |
| FR-MOD04-08 | Changing L or θ₀ via sliders shall update T_small and T_exact immediately (no retraining required). T_pred shall also update if a model is `TRAINED`. | High |
| FR-MOD04-09 | When θ₀ > 30°, the T_small error shall be highlighted in amber to visually flag the approximation breakdown. | Medium |
| FR-MOD04-10 | A horizontal bar chart shall compare the three period values within the period panel. | Low |

### 4.3 RK4 Motion Simulation

**Description:** Numerical simulation of pendulum angular position over time.  
**Priority:** High  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD04-11 | The RK4 simulator shall use state vector [θ (rad), ω (rad/s)] and integrate the equations of motion with configurable dt ∈ {0.001, 0.005, 0.010} s (default: 0.01 s). | High |
| FR-MOD04-12 | The simulation shall cover exactly 3 complete periods (t_max = 3 × T_exact). | High |
| FR-MOD04-13 | Simulated θ values shall be stored in degrees for display; internal computation shall use radians. | High |
| FR-MOD04-14 | Changing L or θ₀ shall trigger immediate RK4 re-simulation; the resulting trajectory shall be reloaded into the animation controller within 200 ms. | High |

### 4.4 θ(t) Time-Domain Plot

**Description:** Matplotlib plot of angular position as a function of time.  
**Priority:** High  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD04-15 | The θ(t) plot shall display the full 3-period trajectory as a blue line (lw=2.0). | High |
| FR-MOD04-16 | Red vertical dashed lines shall be drawn at t = T_exact, 2T_exact, 3T_exact to mark period boundaries. | High |
| FR-MOD04-17 | A horizontal black dashed line at θ=0 shall mark the equilibrium. | Medium |
| FR-MOD04-18 | The plot title shall display: `θ₀={θ}°  T_exact={T:.3f}s  T_pred={T_pred:.3f}s`. | High |

### 4.5 Phase Space Visualization

**Description:** (θ, ω) phase portrait of the pendulum trajectory.  
**Priority:** High  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD04-19 | The phase space plot shall display θ (degrees) on the x-axis and ω (deg/s) on the y-axis. | High |
| FR-MOD04-20 | The full trajectory shall be rendered as a green line (lw=1.5, alpha=0.7). | High |
| FR-MOD04-21 | The initial state (θ₀, 0) shall be marked with a red filled circle (ms=10, label='Start'). | High |
| FR-MOD04-22 | The phase portrait shall form a closed orbit for a conservative pendulum; any significant gap between start and end points shall trigger an assertion warning in debug mode. | Low |
| FR-MOD04-23 | For the two-pendulum isochronism demo (Section 4.7), a second trajectory shall be overlaid in orange. | Medium |

### 4.6 Pendulum Animation

**Description:** Real-time QPainter-based animation of pendulum motion driven by the RK4 trajectory.  
**Priority:** High  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD04-24 | The animation shall advance one frame per `QTimer` timeout. The default timer interval shall be 33 ms (≈30 FPS). | High |
| FR-MOD04-25 | The animation shall loop continuously: after reaching the last trajectory frame, it shall restart from frame 0. | High |
| FR-MOD04-26 | A Play/Pause button shall toggle the QTimer. Keyboard shortcut `Space` shall perform the same toggle when the MOD-04 tab is active. | High |
| FR-MOD04-27 | A speed control (QSlider or QComboBox) shall adjust the QTimer interval to achieve playback speeds of 0.25×, 0.5×, 1×, 2×, 4×. | Medium |
| FR-MOD04-28 | The animation shall begin automatically when new RK4 data is loaded (after clicking Run or changing L/θ₀). | Medium |
| FR-MOD04-29 | The current angular position shall be displayed as a numeric label: "θ = {value:.1f}°" updated every frame. | Medium |

### 4.7 Isochronism Demonstration

**Description:** Two-pendulum simultaneous simulation showing period independence from amplitude.  
**Priority:** Medium  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD04-30 | A toggle button "Isochronism Demo" shall enable a second pendulum with θ₀ = 5° (fixed) and the same L as the primary pendulum. | Medium |
| FR-MOD04-31 | Both pendulums shall be animated simultaneously on the PendulumCanvas: primary in blue, secondary in orange. | Medium |
| FR-MOD04-32 | An information label shall display: `ΔT = |T_primary - T_secondary| = {ΔT:.4f} s ({ΔT/T_primary×100:.2f}% difference)`. | Medium |
| FR-MOD04-33 | The demo shall be automatically disabled when θ₀ is changed to emphasize that the comparison is most instructive at small vs. large angles. | Low |

### 4.8 Period Error Analysis View

**Description:** Static plot showing prediction accuracy across the full parameter space.  
**Priority:** Medium  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD04-34 | An "Analysis" tab shall display MAPE as a function of θ₀ (5°–80°) for fixed L=1.0 m, using the trained model. | Medium |
| FR-MOD04-35 | The same "Analysis" tab shall display predicted period vs. T_exact as a scatter plot across the full (L, θ₀) test set. | Medium |
| FR-MOD04-36 | A reference line shall be shown at MAPE = 1% to indicate the target accuracy threshold. | Low |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement |
|----|-------------|
| NFR-MOD04-01 | RK4 simulation covering 3 periods with dt=0.01 shall complete within 50 ms for any (L, θ₀) in the parameter range. |
| NFR-MOD04-02 | Animation frame rendering (QPainter paintEvent) shall complete within 16 ms (target 60 FPS canvas; actual animation capped at 30 FPS). |
| NFR-MOD04-03 | Slider → RK4 re-simulation → animation reload shall complete within 200 ms. |
| NFR-MOD04-04 | MOD-04 training (2000 samples, 100 epochs) shall complete within 15 seconds on the reference CPU system. |

### 5.2 Physics Accuracy

| ID | Requirement |
|----|-------------|
| NFR-MOD04-05 | The trained neural network shall achieve MAPE ≤ 1.0% across the full test set (L ∈ [0.5, 3.0] m, θ₀ ∈ [5°, 80°]). |
| NFR-MOD04-06 | The RK4 simulation energy error (|E_final - E_initial| / E_initial) shall be < 0.1% for a 3-period simulation with dt=0.01 s. |
| NFR-MOD04-07 | The small-angle approximation error shall display correctly: at θ₀=5° the error shall be < 0.1%; at θ₀=60° the error shall be > 5%. |

### 5.3 Usability

| ID | Requirement |
|----|-------------|
| NFR-MOD04-08 | The PendulumCanvas minimum size shall be 300×400 pixels to ensure the animation is legible. |
| NFR-MOD04-09 | The period panel values shall update within 100 ms of any slider change. |

---

## Appendix A: Acceptance Criteria

**AC-MOD04-01:**  
*Given* L=1.0 m and θ₀=10° (small angle),  
*When* T_small, T_exact, and T_pred are displayed,  
*Then* T_small error < 0.5%, T_pred error < 1.0%, and the θ₀ value shows no amber highlight (below 30° threshold).

**AC-MOD04-02:**  
*Given* L=1.0 m and θ₀=60° (large angle),  
*When* T_small error is displayed,  
*Then* the error value is > 5% and highlighted in amber, while T_pred error is < 1.0%.

**AC-MOD04-03:**  
*Given* the Isochronism Demo is enabled with θ₀_primary=45° and θ₀_secondary=5°,  
*When* animation is running,  
*Then* both pendulums are visible on the canvas in their respective colors, and ΔT/T_primary is displayed as approximately 1.9%.

**AC-MOD04-04:**  
*Given* the animation is playing,  
*When* the user presses Space,  
*Then* animation pauses within one frame period (33 ms), and pressing Space again resumes.

---

## Appendix B: Requirements Traceability

| Requirement ID | TRD Reference | Test Method |
|---------------|---------------|-------------|
| FR-MOD04-01 to FR-MOD04-04 | TRD-03 §2.1 | Unit test: model I/O shapes, MAPE metric |
| FR-MOD04-05 to FR-MOD04-10 | TRD-02 §6 | Integration test: slider change → panel update latency |
| FR-MOD04-11 to FR-MOD04-14 | TRD-03 §1.3 | Unit test: energy conservation, period accuracy |
| FR-MOD04-15 to FR-MOD04-18 | TRD-04 §2.4 | Visual inspection test |
| FR-MOD04-19 to FR-MOD04-23 | TRD-04 §2.4 | Visual inspection: closed orbit |
| FR-MOD04-24 to FR-MOD04-29 | TRD-02 §6, TRD-04 §4 | Integration test: FPS measurement, Space key |
| FR-MOD04-30 to FR-MOD04-33 | TRD-02 §6 | UI test: toggle, dual canvas rendering |
| FR-MOD04-34 to FR-MOD04-36 | TRD-04 §2.4 | System test: MAPE plot vs. threshold line |
