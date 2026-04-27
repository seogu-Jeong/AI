# Software Requirements Specification
## PhysicsAI Simulator — Module 02 & 05: Projectile Motion
**Document ID:** SRS-PHYSAI-003  
**Version:** 2.0  
**Date:** 2026-04-06  
**Standard:** IEEE Std 830-1998  
**Parent Document:** SRS-PHYSAI-001  
**Status:** Approved  

---

## 1. Introduction

### 1.1 Purpose

This SRS defines requirements for two projectile motion modules:

- **MOD-02:** Projectile Motion Regression — trains a neural network to map (v₀, θ, t) → (x, y) under vacuum conditions, then predicts full trajectories
- **MOD-05:** Air Resistance Projectile — uses RK4 integration to simulate drag-affected trajectories; a neural network predicts total range (x_final) from (v₀, θ)

The two modules together demonstrate the progression from ideal physics (analytically solvable) to real-world physics (requiring numerical methods), and the neural network's ability to learn both regimes.

### 1.2 Scope

This document covers user interaction, parameter controls, physics simulation, neural network training, visualization, and acceptance criteria for MOD-02 and MOD-05.

### 1.3 Definitions

| Term | Definition |
|------|------------|
| Drag force | F_d = -k·\|v\|·v; quadratic drag normalized by mass; units: m/s² |
| Range (R) | Horizontal distance from launch point to landing point (y=0) |
| Vacuum trajectory | Parabolic trajectory under gravity alone: x(t)=v₀cos(θ)·t, y(t)=v₀sin(θ)·t−½gt² |
| Optimal angle | Launch angle maximizing range; 45° in vacuum; typically ~40° under quadratic drag |
| Flight time | T_f = 2v₀sin(θ)/g (vacuum only; no closed form with drag) |
| Trajectory point | A (x, y) coordinate at a specific time t along the projectile's path |

### 1.4 Physical Laws

**Vacuum equations of motion:**
```
x(t) = v₀·cos(θ)·t
y(t) = v₀·sin(θ)·t − ½·g·t²
Range(vacuum) = v₀²·sin(2θ)/g
```

**Air resistance equations of motion (ODE system):**
```
dx/dt = vₓ
dy/dt = v_y
dvₓ/dt = −k·|v|·vₓ        where |v| = √(vₓ² + v_y²)
dv_y/dt = −g − k·|v|·v_y
```
No closed-form solution exists; RK4 numerical integration is required.

### 1.5 References

Inherits SRS-PHYSAI-001 references. Additionally:  
[11] H. C. Corben and P. Stehle, *Classical Mechanics*, 2nd ed. Dover, 1994.  
[12] W. H. Press et al., *Numerical Recipes: The Art of Scientific Computing*, 3rd ed. Cambridge University Press, 2007. (Chapter 17: Ordinary Differential Equations)  

---

## 2. Overall Description

### 2.1 Module Relationship

MOD-02 and MOD-05 form a pedagogical progression:

| Aspect | MOD-02 | MOD-05 |
|--------|--------|--------|
| Physics model | Ideal (vacuum) | Realistic (air resistance) |
| ML task | Trajectory regression (x,y) at time t | Range regression (scalar output) |
| Analytical solution | Exists | Does not exist |
| Reference data source | Analytical formula | RK4 simulation |
| Optimal angle | Always 45° | ~40° (depends on k) |

### 2.2 User Classes

Targets UC-01 (ML Student) and UC-02 (Physics Educator). MOD-05 is particularly relevant for UC-03 (ML Researcher) studying physics-informed vs. data-driven approaches.

---

## 3. External Interface Requirements

### 3.1 User Interface — MOD-02

**Left Panel:**
```
┌─────────────────────────────────┐
│ [GroupBox: Launch Parameters]   │
│  Init Velocity v₀: [═══] 30 m/s│
│  Launch Angle  θ:  [═══] 45 °  │
│                                 │
│ [GroupBox: Training Config]     │
│  Train Samples: [═══] 2000      │
│  Noise σ:       [═══] 0.50 m   │
│  Epochs:        [═══] 100       │
│                                 │
│ [▶ Run]  [■ Stop]  [↺ Reset]   │
│ [▓▓▓▓▓▓░░░░░░░] 72%            │
│ Loss:  0.382100                 │
│                                 │
│ [☆ Add Condition]               │
│ Active conditions: 1/5          │
└─────────────────────────────────┘
```

**Right Panel — GridSpec layout:**
```
┌──────────────────────────────────────────────┐
│          Trajectory Comparison (spans top)   │
│  True trajectory  ─── blue                  │
│  NN Prediction    --- red                    │
│  (Condition 2..5 overlaid)                   │
├──────────────────┬───────────────────────────┤
│  x(t) vs time    │  y(t) vs time             │
└──────────────────┴───────────────────────────┘
```

### 3.2 User Interface — MOD-05

**Left Panel:**
```
┌─────────────────────────────────┐
│ [GroupBox: Launch Parameters]   │
│  Init Velocity v₀: [═══] 50 m/s│
│  Launch Angle  θ:  [═══] 45 °  │
│  Drag Coeff.  k:   [═══] 0.05  │
│                                 │
│ [GroupBox: Training Config]     │
│  Train Samples: [═══] 2000      │
│  Epochs:        [═══] 100       │
│                                 │
│ [▶ Run]  [■ Stop]  [↺ Reset]   │
│ [▓▓▓▓▓▓▓░░░░░] 85%             │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ Vacuum Range:    254.7 m    │ │
│ │ Physics Range:   187.3 m    │ │
│ │ AI Predicted:    188.1 m    │ │
│ │ Range reduction: 26.5%      │ │
│ │ Optimal angle:   39.2°      │ │
│ └─────────────────────────────┘ │
└─────────────────────────────────┘
```

**Right Panel — 2×2 GridSpec:**
- (0,0): Trajectory comparison (vacuum vs air resistance vs AI landing point)
- (0,1): Range vs launch angle curve
- (1,0): AI prediction scatter plot (True vs Predicted)
- (1,1): Training loss curve

---

## 4. System Features

### 4.1 MOD-02: Trajectory Dataset Generation

**Description:** Synthetic dataset creation from projectile motion equations.  
**Priority:** High  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD02-01 | Training data shall be generated with: v₀ ∈ U[10, 50] m/s, θ ∈ U[20°, 70°], t ∈ U[0, 0.9·T_f]; n=2000 samples. | High |
| FR-MOD02-02 | Gaussian noise shall be added to x and y with σ configurable in [0.0, 2.0] m (default σ=0.5 m). | High |
| FR-MOD02-03 | Samples where y < 0 (below ground) shall be filtered out; the effective sample count after filtering shall be displayed. | High |
| FR-MOD02-04 | Test data shall be generated with n=500 samples and noise σ=0.0 (clean data) for unbiased evaluation. | High |
| FR-MOD02-05 | Data generation shall execute in a `LoadingWorker` QThread and emit a `data_ready` signal upon completion. | Medium |

### 4.2 MOD-02: Neural Network Architecture

**Description:** Fixed neural network for (v₀, θ, t) → (x, y) regression.  
**Priority:** High  

**Architecture specification:**
```
Input(3)  →  Dense(128, relu)  →  Dropout(0.1)
          →  Dense(64, relu)   →  Dropout(0.1)
          →  Dense(32, relu)   →  Dropout(0.1)
          →  Dense(2, linear)
```

**Architectural rationale:**
- Input dimension 3: (v₀, θ, t) — three independent physical variables
- Output dimension 2: (x, y) — joint position prediction enforces physical correlation
- relu activation: appropriate for non-periodic, unbounded outputs (x, y grow with v₀ and t)
- Dropout(0.1): light regularization; prevents overfitting on noisy data without excessive capacity reduction
- Adam optimizer, lr=0.001: effective default for regression [10]

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD02-06 | The model shall accept input shape (None, 3) and produce output shape (None, 2). | High |
| FR-MOD02-07 | The model shall be compiled with `loss='mse'` and `metrics=['mae']`. | High |
| FR-MOD02-08 | Validation split shall be 0.2 (20% of training data). | High |

### 4.3 MOD-02: Trajectory Prediction and Visualization

**Description:** After training, the model predicts full trajectories for user-specified conditions.  
**Priority:** High  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD02-09 | The trajectory prediction function shall sample 50 time points from t=0 to t=T_f for the given (v₀, θ). | High |
| FR-MOD02-10 | After training completion, changing v₀ or θ via the sliders shall trigger trajectory re-prediction and plot update within 500 ms, without retraining. | High |
| FR-MOD02-11 | The trajectory plot shall enforce `xlim(left=0)` and `ylim(bottom=0)` to show only physically valid regions. | High |
| FR-MOD02-12 | The system shall compute and annotate: max height (True vs Pred) and max range (True vs Pred) on the trajectory plot using `ax.annotate`. | Medium |
| FR-MOD02-13 | The error analysis sub-panel shall display MSE as a function of launch angle (20°–70°) and as a function of v₀ (10–50 m/s), computed post-training over a grid of 50 test conditions each. | Medium |

### 4.4 MOD-02: Multi-Condition Overlay

**Description:** Users can overlay up to five trajectory comparisons on a single plot.  
**Priority:** Medium  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD02-14 | The "Add Condition" button shall capture the current (v₀, θ) parameter values and add their true trajectory to the main plot as a new line with auto-assigned color from the `tab10` palette. | Medium |
| FR-MOD02-15 | Each added condition shall display a label in the legend: `Cond {n}: v₀={v}m/s, θ={a}°`. | Medium |
| FR-MOD02-16 | A maximum of 5 conditions shall be allowed simultaneously; the button shall be disabled at the limit. | Medium |
| FR-MOD02-17 | A "Clear Conditions" button shall remove all overlay conditions and reset the plot to the current single condition. | Medium |

### 4.5 MOD-05: RK4 Physics Simulation

**Description:** Numerical simulation of projectile motion with quadratic air resistance.  
**Priority:** High  

**RK4 Integration Specification:**

State vector: **s** = [x, y, vₓ, v_y] ∈ ℝ⁴  
Derivatives: f(**s**, t) = [vₓ, v_y, -k|**v**|vₓ, -g - k|**v**|v_y]  
Step size: dt = 0.01 s (configurable)  
Termination: y < 0 (landing condition)  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD05-01 | The RK4 integrator shall use the state representation and derivative function defined above. | High |
| FR-MOD05-02 | Simulation shall terminate when y < 0; the final range shall be taken as the x-coordinate at the last positive-y step. | High |
| FR-MOD05-03 | The drag coefficient k shall be configurable in [0.0, 0.20] kg⁻¹ with default k=0.05. | High |
| FR-MOD05-04 | When k=0.0, the simulated trajectory shall match the vacuum analytical formula with error < 0.01 m (used as a correctness verification). | High |
| FR-MOD05-05 | Changing k shall trigger immediate RK4 re-simulation (not retraining) and update the trajectory plot within 500 ms. | High |
| FR-MOD05-06 | The system shall compute and display the drag-induced range reduction percentage: `(1 - R_air/R_vacuum) × 100%`. | Medium |

### 4.6 MOD-05: Training Dataset from RK4 Simulation

**Description:** Training data for the AI model is generated by running the RK4 simulator over a grid of (v₀, θ) inputs.  
**Priority:** High  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD05-07 | Training data shall be generated with: v₀ ∈ U[10, 100] m/s, θ ∈ U[10°, 80°]; n=2000 samples, using the drag coefficient k active at the time of data generation. | High |
| FR-MOD05-08 | Data generation (2000 RK4 simulations) shall execute in a `LoadingWorker` QThread; a progress bar shall display generation progress in increments of 100 simulations. | High |
| FR-MOD05-09 | If k is changed after training, the system shall display a warning: "Drag coefficient changed. Retraining required for accurate AI predictions." | High |

### 4.7 MOD-05: AI Range Prediction and Visualization

**Description:** The trained model predicts landing range; results are compared against physics simulation.  
**Priority:** High  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD05-10 | After training, the AI predicted range shall be displayed as a green star marker (★) on the trajectory plot at (R_pred, 0). | High |
| FR-MOD05-11 | The "Range vs Angle" subplot shall display three curves: vacuum theory (blue dashed), RK4 physics (red solid), AI predictions (green dots) for angles 10°–80° at fixed v₀=60 m/s. | High |
| FR-MOD05-12 | The optimal launch angle (maximizing range for air resistance case) shall be identified by scanning the RK4 curve and annotated with a vertical red dashed line. | High |
| FR-MOD05-13 | The "AI Performance" scatter plot shall display test set True Range vs Predicted Range with a y=x reference line. The R² coefficient shall be computed and displayed in the plot title. | High |
| FR-MOD05-14 | The live information panel shall update in real time as v₀, θ, or k are changed, displaying: Vacuum Range, Air Resistance Range, AI Predicted Range, Range Reduction (%), and Optimal Angle. | High |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement |
|----|-------------|
| NFR-MOD02-01 | MOD-02 training (2000 samples, 100 epochs) shall complete within 20 seconds on the reference CPU system. |
| NFR-MOD02-02 | Post-training trajectory re-prediction (v₀/θ slider change → plot update) shall complete within 500 ms. |
| NFR-MOD05-01 | Single RK4 simulation (dt=0.01) shall complete within 5 ms on the reference system. |
| NFR-MOD05-02 | Full training dataset generation (2000 RK4 simulations) shall complete within 30 seconds on the reference system. |
| NFR-MOD05-03 | MOD-05 training (2000 samples, 100 epochs) shall complete within 20 seconds. |

### 5.2 Physics Accuracy

| ID | Requirement |
|----|-------------|
| NFR-MOD02-03 | The neural network trajectory prediction shall achieve MSE ≤ 1.0 m² on the clean test set (σ=0.0) after 100 epochs. |
| NFR-MOD05-04 | The RK4 simulator with k=0.0 shall agree with the analytical vacuum range formula to within 0.01 m for all (v₀, θ) in the training range. |
| NFR-MOD05-05 | The AI range prediction shall achieve R² ≥ 0.98 on the test set. |

---

## Appendix A: Acceptance Criteria

**AC-MOD02-01:**  
*Given* default parameters (v₀=30 m/s, θ=45°, σ=0.5 m),  
*When* training completes (100 epochs),  
*Then* the trajectory plot shows True and Predicted curves where the max-height error is < 5% and the max-range error is < 5%.

**AC-MOD05-01:**  
*Given* k is set from 0.0 to 0.05,  
*When* k changes,  
*Then* the trajectory plot updates within 500 ms showing a visibly shorter range and asymmetric descent; the range reduction percentage updates in the information panel.

**AC-MOD05-02:**  
*Given* a trained model with k=0.05,  
*When* the Range vs Angle plot is rendered,  
*Then* the optimal angle annotation appears at ~38–42°, confirming that drag shifts the optimum below 45°.

---

## Appendix B: Requirements Traceability

| Requirement ID | TRD Reference | Test Method |
|---------------|---------------|-------------|
| FR-MOD02-01 to FR-MOD02-05 | TRD-03 §3.2 | Unit test: data generator output shape and range |
| FR-MOD02-06 to FR-MOD02-08 | TRD-03 §2.1 | Unit test: model.input_shape, model.output_shape |
| FR-MOD02-09 to FR-MOD02-13 | TRD-04 §2.2 | Integration test: slider → plot update latency |
| FR-MOD02-14 to FR-MOD02-17 | TRD-02 §3.4 | UI test: condition count enforcement |
| FR-MOD05-01 to FR-MOD05-06 | TRD-03 §1.2 | Unit test: rk4_step output, k=0 correctness |
| FR-MOD05-07 to FR-MOD05-09 | TRD-03 §3.4 | Integration test: data gen progress, warning dialog |
| FR-MOD05-10 to FR-MOD05-14 | TRD-04 §2.5 | System test: visual inspection + R² measurement |
