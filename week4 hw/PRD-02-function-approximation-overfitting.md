# Software Requirements Specification
## PhysicsAI Simulator — Module 01 & 03: Function Approximation and Overfitting
**Document ID:** SRS-PHYSAI-002  
**Version:** 2.0  
**Date:** 2026-04-06  
**Standard:** IEEE Std 830-1998  
**Parent Document:** SRS-PHYSAI-001  
**Status:** Approved  

---

## 1. Introduction

### 1.1 Purpose

This SRS specifies functional and non-functional requirements for two pedagogically coupled modules:

- **MOD-01:** 1D Function Approximation — demonstrates the Universal Approximation Theorem by training neural networks to regress mathematical functions
- **MOD-03:** Overfitting vs. Underfitting Demonstration — illustrates the model complexity–generalization trade-off through three simultaneously trained models

Both modules are implemented as PySide6 `QWidget` subclasses within the PhysicsAI Simulator application.

### 1.2 Scope

This document covers all user-visible functionality, parameter controls, training execution, visualization, and acceptance criteria for MOD-01 and MOD-03. It does not cover the shared infrastructure (state management, QThread architecture) documented in SRS-PHYSAI-001 and TRD-01.

### 1.3 Definitions

| Term | Definition |
|------|------------|
| Good fit | A model whose validation loss converges to a value within 20% of training loss |
| Overfitting | Condition where validation loss exceeds 2× training loss after epoch 50 |
| Underfitting | Condition where training loss fails to decrease below 0.1 within 200 epochs on the target function |
| Approximation error | \|f(x) - f̂(x)\| where f is the ground truth function and f̂ is the network prediction |
| Activation regime | The characteristic of tanh to saturate at ±1, providing bounded outputs suitable for periodic targets |

### 1.4 References

Inherits all references from SRS-PHYSAI-001. Additionally:  
[9] N. Srivastava et al., "Dropout: A simple way to prevent neural networks from overfitting," *Journal of Machine Learning Research*, vol. 15, pp. 1929–1958, 2014.  
[10] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *ICLR 2015*, arXiv:1412.6980, 2015.  

---

## 2. Overall Description

### 2.1 Module Relationship

MOD-01 and MOD-03 are thematically linked:
- MOD-01 explores **capacity adequacy** — can a given architecture approximate the target function?
- MOD-03 explores **capacity calibration** — what happens when capacity is systematically too low or too high?

Together they form a complete introduction to neural network expressivity and generalization.

### 2.2 Target Functions

**MOD-01 Target Functions:**

| ID | Function | Domain | Complexity Rationale |
|----|----------|--------|----------------------|
| F-01 | f(x) = sin(x) | [-2π, 2π] | Baseline periodic; well-approximated by small networks [3] |
| F-02 | f(x) = cos(x) + 0.5·sin(2x) | [-2π, 2π] | Superposition of two frequencies; tests multi-frequency learning |
| F-03 | f(x) = x·sin(x) | [-2π, 2π] | Non-stationary envelope; amplitude grows with x |
| F-04 | f(x) = sin(x)+0.5sin(2x)+0.3cos(3x)+0.2sin(5x)+0.1x·cos(x) | [-3π, 3π] | Extreme complexity; 5-component superposition |

**MOD-03 Target Function:** f(x) = sin(2x) + 0.5x, domain [-2, 2], with additive Gaussian noise σ ∈ [0.0, 1.0].

### 2.3 User Classes

Inherits UC-01 (AI/ML Student) and UC-02 (Physics Educator) from SRS-PHYSAI-001. MOD-03 is primarily targeted at UC-01 who needs to understand generalization theory.

---

## 3. External Interface Requirements

### 3.1 User Interface — MOD-01

The MOD-01 tab shall present a two-panel layout:

**Left Panel — Parameter Controls (280 px width):**
```
┌─────────────────────────────────┐
│ [GroupBox: Target Function]     │
│  Function:  [ComboBox ▼]        │
│                                 │
│ [GroupBox: Network Architecture]│
│  Architecture: [ComboBox ▼]     │
│  Activation:   [ComboBox ▼]     │
│  Learning Rate: [════|══] 0.010 │
│                                 │
│ [GroupBox: Training]            │
│  Epochs:    [════|══] 3000      │
│  Batch Size: [════|══]  32      │
│                                 │
│ [▶ Run]  [■ Stop]  [↺ Reset]   │
│                                 │
│ [▓▓▓▓▓▓░░░░░░░] 60%            │
│ Loss:     0.000234              │
│ Epoch:    1800 / 3000           │
│ Completed in 4823ms             │
└─────────────────────────────────┘
```

**Right Panel — Plot Area:**
Three-column subplot layout (15:5 aspect ratio):
1. **Function Approximation:** True (blue solid) vs. Predicted (red dashed) + training scatter
2. **Training Loss:** log-scale y-axis, epoch vs. MSE
3. **Absolute Error:** |y_pred - y_true| with fill_between shading

### 3.2 User Interface — MOD-03

**Left Panel:** Parameter controls as specified in Section 4.2.

**Right Panel — Tab widget with three views:**
- **Tab "Predictions":** All three model predictions overlaid on a single plot
- **Tab "Loss Curves":** 1×3 subplot with train/val loss per model; overfitting epoch annotated
- **Tab "Error Analysis":** Absolute error per model + summary performance table

---

## 4. System Features

### 4.1 MOD-01: Network Architecture Selection

**Description:** Users select from four predefined neural network architectures; the system builds, trains, and visualizes the selected model.  
**Priority:** High  
**Stimulus:** User selects architecture from ComboBox and clicks Run.  
**Response:** System builds model, generates data, trains asynchronously, updates plots in real time.  

**Architecture Options:**

| Option Label | Hidden Layers | Parameters (approx.) | Use Case |
|-------------|---------------|----------------------|----------|
| Small [32] | [32] | ~130 | Demonstrates underfitting on complex functions |
| Medium [64, 64] | [64, 64] | ~4,800 | Adequate for F-01 and F-02 |
| Large [128, 128] | [128, 128] | ~18,000 | Recommended for F-01 through F-03 |
| XLarge [128, 128, 64] | [128, 128, 64] | ~26,000 | Required for F-04; invoked automatically |

**Functional Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD01-01 | The system shall provide exactly four architecture presets as listed in the table above, selectable via ComboBox. | High |
| FR-MOD01-02 | When function F-04 is selected, the system shall automatically set the architecture to XLarge and display a tooltip: "Extreme function requires XLarge architecture." | High |
| FR-MOD01-03 | The total parameter count for the selected architecture shall be displayed below the architecture selector in the format "Parameters: ~{n:,}". | Medium |
| FR-MOD01-04 | If the user changes architecture while a model is `TRAINED`, the system shall display an amber "dirty" indicator; a new Run shall rebuild and retrain the model from scratch. | High |

### 4.2 MOD-01: Activation Function Selection

**Description:** Users select the activation function applied to all hidden layers.  
**Priority:** Medium  

**Options:** `tanh` (default), `relu`, `sigmoid`  

**Activation Function Rationale (to be displayed as tooltip):**

| Activation | Characteristics | Recommended For |
|------------|----------------|-----------------|
| tanh | Bounded ±1, symmetric, smooth gradients; suited for periodic targets | F-01 through F-04 (default) |
| relu | Unbounded positive output, sparse gradients; risk of dead neurons with periodic targets | Experimental comparison |
| sigmoid | Bounded (0,1), asymmetric; slower convergence; not recommended for zero-mean functions | Demonstration of poor choice |

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD01-05 | The activation selector shall default to `tanh` upon module initialization or reset. | High |
| FR-MOD01-06 | Changing the activation function shall trigger a dirty indicator if a model is currently `TRAINED`. | Medium |
| FR-MOD01-07 | A tooltip on the activation selector shall explain why `tanh` is preferred for periodic functions, citing bounded output range. | Low |

### 4.3 MOD-01: Training Execution and Live Updates

**Description:** Real-time visualization of training dynamics.  
**Priority:** High  

**Stimulus/Response Sequence:**

```
1. User clicks [Run]
2. System: disables Run, enables Stop; sets state = TRAINING
3. System: generates training data (x_train: 200 pts, x_test: 400 pts)
4. System: builds Keras model per selected architecture and activation
5. QThread starts; Keras callback emits Signal every 10 epochs
6. Main thread slot: updates ProgressBar, Loss label, loss curve plot
7. On training_finished signal:
   a. Compute y_pred on x_test
   b. Update approximation plot and error plot
   c. Display MSE, MAE, Max Error in plot titles
   d. Set state = TRAINED; enable Run and Reset
8. User optionally changes parameters → dirty indicator shown
```

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD01-08 | Training data shall consist of 200 uniformly spaced points over the function domain; test data shall consist of 400 points. | High |
| FR-MOD01-09 | The loss curve shall be updated every 10 epochs using `canvas.draw_idle()` on the main thread. | High |
| FR-MOD01-10 | Upon training completion, the approximation plot shall display: True function (blue, lw=2.5, alpha=0.7), predicted function (red dashed, lw=2.0), and training data scatter (black, s=15, alpha=0.3, stride 10). | High |
| FR-MOD01-11 | The MSE, MAE, and Max Absolute Error shall be computed on the test set and displayed in the plot title as: `{func_name}\nMSE: {mse:.6f}  MAE: {mae:.6f}  MaxErr: {max_err:.6f}`. | High |
| FR-MOD01-12 | A `ReduceLROnPlateau` callback shall be attached with: `factor=0.9, patience=100, min_lr=1e-5`. | Medium |

### 4.4 MOD-01: Extreme Complexity Test (F-04)

**Description:** A dedicated test for the most complex target function.  
**Priority:** Medium  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD01-13 | When F-04 is selected, the system shall use 500 training points over [-3π, 3π]. | High |
| FR-MOD01-14 | F-04 training shall attach both `ReduceLROnPlateau` (factor=0.8, patience=100) and `EarlyStopping` (patience=500, restore_best_weights=True). | High |
| FR-MOD01-15 | The error distribution histogram shall be displayed as a fourth subplot when F-04 is active, showing frequency vs. absolute error with a vertical line at MAE. | Medium |

### 4.5 MOD-03: Three-Model Comparative Training

**Description:** Three models of increasing complexity are trained simultaneously on identical data.  
**Priority:** High  

**Model Definitions:**

| Model | Architecture | Regularization | Expected Behavior |
|-------|-------------|----------------|-------------------|
| Underfit | Input → Dense(4, relu) → Dense(1) | None | High train and val loss; fails to capture target pattern |
| Good Fit | Input → Dense(32, relu) → Dropout(0.2) → Dense(16, relu) → Dropout(0.2) → Dense(1) | Dropout p=0.2 [9] | Converging train/val loss; best test MSE |
| Overfit | Input → Dense(256) → Dense(128) → Dense(64) → Dense(32) → Dense(1) | None | Low train loss, diverging val loss |

**Functional Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD03-01 | All three models shall be trained on identical `(x_train, y_train)` and evaluated on identical `(x_val, y_val)`. Data splits: 100 train, 50 val, 200 test (noise-free). | High |
| FR-MOD03-02 | Each model shall run in an independent `QThread`; all three threads shall be started within 100 ms of the Run action. | High |
| FR-MOD03-03 | The parameter panel shall display a per-model completion indicator: ⏳ (training), ✓ (completed), ✗ (error). | High |
| FR-MOD03-04 | The "Predictions" tab shall use fixed colors: Underfit = `#2196F3` (blue), Good = `#4CAF50` (green), Overfit = `#F44336` (red). The True function shall be rendered in black (lw=2.5). | High |
| FR-MOD03-05 | The system shall detect the overfitting onset epoch as the first epoch where `val_loss > 2 × train_loss` (after epoch 20 to exclude warm-up). A vertical dashed orange line shall be drawn at this epoch on the Overfit model's loss subplot. | High |
| FR-MOD03-06 | The "Error Analysis" tab shall include a performance table with columns: Model \| Final Train Loss \| Final Val Loss \| Test MSE \| Test MAE, with the Good Fit row highlighted in `#C8E6C9`. | Medium |
| FR-MOD03-07 | Noise level (σ) and sample count (n_train) shall be controllable at runtime; changing either shall regenerate data and prompt the user to re-run. | Medium |
| FR-MOD03-08 | When all three models are trained, a summary annotation shall appear on the "Predictions" plot stating the test MSE for each model in the legend: e.g., `Good Fit (MSE=0.0031)`. | Medium |

### 4.6 MOD-03: Overfitting Detection Advisory

**Description:** The system shall actively advise users when overfitting is detected.  
**Priority:** Medium  

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MOD03-09 | When overfitting onset is detected (per FR-MOD03-05), a `QLabel` badge reading "⚠ Overfitting Detected at Epoch {n}" shall appear in the Overfit model's status row, styled in amber (`#FF9800`). | Medium |
| FR-MOD03-10 | An information tooltip on the badge shall explain: "Validation loss exceeded 2× training loss. The model is memorizing training noise rather than learning the underlying pattern." | Low |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement |
|----|-------------|
| NFR-MOD01-01 | MOD-01 training on F-01 with Large architecture and 3000 epochs shall complete within 30 seconds on the reference CPU system (Intel Core i5, no GPU). |
| NFR-MOD01-02 | MOD-01 training on F-04 with XLarge architecture and 8000 epochs shall complete within 120 seconds on the reference CPU system. |
| NFR-MOD03-01 | All three MOD-03 models shall reach training completion (200 epochs) within 60 seconds on the reference system, running concurrently. |

### 5.2 Accuracy Requirements

| ID | Requirement | Measurement |
|----|-------------|-------------|
| NFR-MOD01-03 | For F-01 with Large architecture and 3000 epochs, the final test MSE shall be ≤ 1×10⁻⁴. | Post-training evaluation on x_test |
| NFR-MOD01-04 | For F-02 with Large architecture, final test MSE shall be ≤ 5×10⁻⁴. | Post-training evaluation on x_test |
| NFR-MOD03-02 | The Good Fit model test MSE shall be lower than both Underfit and Overfit models' test MSE in ≥ 90% of independent runs (evaluated with fixed random seeds). | Statistical evaluation over 10 runs |

### 5.3 Usability

| ID | Requirement |
|----|-------------|
| NFR-MOD01-05 | Switching between function presets (F-01 through F-04) shall reset the plot area within 200 ms, without residual rendering artifacts from the previous function. |
| NFR-MOD03-03 | The three-model comparison plot shall be legible at 1280×800 resolution; line widths and legend font sizes shall be chosen accordingly (lw≥2.0, fontsize≥10). |

---

## Appendix A: Acceptance Criteria (Given/When/Then)

**AC-MOD01-01:**  
*Given* a user has selected F-01 (sin(x)) and Large architecture,  
*When* the user clicks Run and training completes,  
*Then* the approximation plot shows True and Predicted curves that are visually indistinguishable to the naked eye, and the displayed MSE is ≤ 1×10⁻⁴.

**AC-MOD01-02:**  
*Given* F-04 is selected,  
*When* the architecture selector is opened,  
*Then* the XLarge option is pre-selected and the tooltip reads "Extreme function requires XLarge architecture."

**AC-MOD03-01:**  
*Given* default parameters (n_train=100, σ=0.3),  
*When* all three models finish training,  
*Then* the Good Fit model displays the lowest test MSE among the three, and the Overfit model's loss subplot has an orange vertical line at the detected overfitting onset epoch.

**AC-MOD03-02:**  
*Given* the user increases σ to 0.8 after training is complete,  
*When* the parameter is changed,  
*Then* a prompt appears: "Parameters changed. Click Run to retrain with new noise level."

---

## Appendix B: Requirements Traceability

| Requirement ID | TRD Reference | Test Method |
|---------------|---------------|-------------|
| FR-MOD01-01 to FR-MOD01-07 | TRD-02 §3.1, TRD-03 §2 | UI inspection + unit test |
| FR-MOD01-08 to FR-MOD01-12 | TRD-03 §2, TRD-04 §2.1 | Integration test with mock TF model |
| FR-MOD01-13 to FR-MOD01-15 | TRD-03 §2, TRD-04 §2.1 | System test with real training |
| FR-MOD03-01 to FR-MOD03-06 | TRD-03 §3, TRD-04 §2.3 | System test with fixed seed |
| FR-MOD03-07 to FR-MOD03-10 | TRD-02 §4, TRD-03 §3 | UI integration test |
