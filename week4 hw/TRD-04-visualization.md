# Software Design Description
## PhysicsAI Simulator — Visualization Design
**Document ID:** SDD-PHYSAI-004  
**Version:** 2.0  
**Date:** 2026-04-06  
**Standard:** IEEE Std 1016-2009  
**Parent Document:** SDD-PHYSAI-001  
**Status:** Approved  

---

## 1. Introduction

### 1.1 Purpose

This SDD specifies the visualization architecture for PhysicsAI Simulator, covering the Matplotlib-PySide6 integration strategy, module-specific plot layouts, animation timing, thread-safe rendering patterns, and data export pipeline.

### 1.2 Core Constraint

All Matplotlib draw operations shall occur on the Qt main thread. This is a non-negotiable constraint imposed by Matplotlib's non-thread-safe renderer. The design in this document ensures this constraint is satisfied by construction.

### 1.3 Matplotlib Backend Selection

```
matplotlib.use('QtAgg')  ← Set before any Qt or Matplotlib import
```

**Why QtAgg over other backends:**

| Backend | Reason for Rejection |
|---------|---------------------|
| TkAgg | Requires Tkinter; conflicts with PySide6 event loop |
| WxAgg | Requires wxPython; additional dependency |
| WebAgg | Browser-based; not suitable for desktop embedding |
| Agg (non-interactive) | No embedding support |
| **QtAgg** | Native Qt6 canvas; `FigureCanvasQTAgg` inherits from `QWidget` |

---

## 2. Module-Specific Plot Layouts

### 2.1 MOD-01 — 1D Function Approximation

**Layout:** 1 row × 3 columns, `figsize=(15, 5)`

```python
def _setup_plots(self):
    self.mpl = MatplotlibWidget(figsize=(15, 5))
    self.axes = self.mpl.fresh_axes(nrows=1, ncols=3)
    ax_approx, ax_loss, ax_error = self.axes

    # --- Column 0: Function Approximation ---
    ax_approx.set_xlabel('x')
    ax_approx.set_ylabel('y')
    ax_approx.set_title('Function Approximation', fontweight='bold')
    ax_approx.grid(True, alpha=0.3)
    # Pre-create line artists for incremental update
    self._line_true,  = ax_approx.plot([], [], 'b-',  lw=2.5, label='True',  alpha=0.7)
    self._line_pred,  = ax_approx.plot([], [], 'r--', lw=2.0, label='Predicted')
    self._scat_train  = ax_approx.scatter([], [], c='k', s=15, alpha=0.3, label='Train data')
    ax_approx.legend(fontsize=9)

    # --- Column 1: Training Loss (log scale) ---
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss (MSE)')
    ax_loss.set_title('Training Loss', fontweight='bold')
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.3, which='both')
    self._line_loss, = ax_loss.plot([], [], 'g-', lw=1.5)

    # --- Column 2: Absolute Error ---
    ax_error.set_xlabel('x')
    ax_error.set_ylabel('|Error|')
    ax_error.set_title('Absolute Error', fontweight='bold')
    ax_error.grid(True, alpha=0.3)
    # fill_between must be fully recreated each update; store reference for clearing
    self._error_fill = None

    self.mpl.draw()
```

**Incremental update during training (main thread, called every 10 epochs):**

```python
def _update_loss_plot(self, epoch: int, loss: float):
    """Updates only the loss curve; avoids full figure redraw."""
    self._epochs_buf.append(epoch)
    self._losses_buf.append(loss)
    self._line_loss.set_data(self._epochs_buf, self._losses_buf)
    ax = self.axes[1]
    ax.relim()
    ax.autoscale_view()
    self.mpl.draw_idle()
```

**Full render after training completion:**

```python
def _render_complete(self, x_test, y_test, y_pred, loss_history):
    ax_approx, ax_loss, ax_error = self.axes
    mse     = np.mean((y_pred - y_test) ** 2)
    mae     = np.mean(np.abs(y_pred - y_test))
    max_err = np.max(np.abs(y_pred - y_test))

    # Update approximation
    self._line_true.set_data(x_test.flatten(), y_test.flatten())
    self._line_pred.set_data(x_test.flatten(), y_pred.flatten())
    ax_approx.set_title(
        f'{self._func_name}\nMSE: {mse:.6f}  MAE: {mae:.6f}  MaxErr: {max_err:.6f}',
        fontweight='bold', fontsize=10
    )
    ax_approx.relim(); ax_approx.autoscale_view()

    # Update loss
    epochs = list(range(1, len(loss_history) + 1))
    self._line_loss.set_data(epochs, loss_history)
    ax_loss.relim(); ax_loss.autoscale_view()

    # Rebuild error fill (fill_between cannot be updated in-place)
    if self._error_fill:
        self._error_fill.remove()
    error = np.abs(y_pred - y_test).flatten()
    self._error_fill = ax_error.fill_between(
        x_test.flatten(), 0, error, color='r', alpha=0.3
    )
    ax_error.plot(x_test.flatten(), error, 'r-', lw=1.5)
    ax_error.set_title(f'Error (Max: {max_err:.6f})', fontweight='bold')
    ax_error.relim(); ax_error.autoscale_view()

    self.mpl.draw()  # Full synchronous draw after training
```

### 2.2 MOD-02 — Projectile Motion

**Layout:** GridSpec 2×2 with merged top row

```python
def _setup_plots(self):
    self.mpl = MatplotlibWidget(figsize=(14, 9))
    gs = self.mpl.fresh_gridspec(2, 2, height_ratios=[2, 1], hspace=0.35)
    self._ax_traj = self.mpl.figure.add_subplot(gs[0, :])   # merged top
    self._ax_xt   = self.mpl.figure.add_subplot(gs[1, 0])
    self._ax_yt   = self.mpl.figure.add_subplot(gs[1, 1])

    # Trajectory plot
    self._ax_traj.set_xlabel('x (m)'); self._ax_traj.set_ylabel('y (m)')
    self._ax_traj.set_xlim(left=0);    self._ax_traj.set_ylim(bottom=0)
    self._line_traj_true, = self._ax_traj.plot([], [], 'b-',  lw=2.5, label='True')
    self._line_traj_pred, = self._ax_traj.plot([], [], 'r--', lw=2.0, label='NN Prediction')
    self._ax_traj.legend()
    self._ax_traj.grid(True, alpha=0.3)

    # x(t)
    self._ax_xt.set_xlabel('t (s)'); self._ax_xt.set_ylabel('x (m)')
    self._line_xt_true, = self._ax_xt.plot([], [], 'b-',  lw=2.0, label='True')
    self._line_xt_pred, = self._ax_xt.plot([], [], 'r--', lw=2.0, label='Pred')
    self._ax_xt.legend(fontsize=9); self._ax_xt.grid(True, alpha=0.3)

    # y(t)
    self._ax_yt.set_xlabel('t (s)'); self._ax_yt.set_ylabel('y (m)')
    self._line_yt_true, = self._ax_yt.plot([], [], 'b-',  lw=2.0, label='True')
    self._line_yt_pred, = self._ax_yt.plot([], [], 'r--', lw=2.0, label='Pred')
    self._ax_yt.legend(fontsize=9); self._ax_yt.grid(True, alpha=0.3)

    self.mpl.draw()
```

**Real-time trajectory update (post-training, on slider change):**

```python
def _update_trajectory(self, v0: float, theta: float):
    """Called when v0/theta slider changes; no retraining."""
    G = 9.81
    theta_rad = np.deg2rad(theta)
    t_flight  = 2 * v0 * np.sin(theta_rad) / G
    t_pts     = np.linspace(0, t_flight, 50)

    # True trajectory (analytical)
    x_true = v0 * np.cos(theta_rad) * t_pts
    y_true = v0 * np.sin(theta_rad) * t_pts - 0.5 * G * t_pts**2

    # NN prediction
    X_input = np.column_stack([
        np.full(50, v0), np.full(50, theta), t_pts
    ])
    pred = self._model.predict(X_input, verbose=0)
    x_pred, y_pred = pred[:, 0], pred[:, 1]

    # Update all three plots
    for line, x, y in [
        (self._line_traj_true, x_true, y_true),
        (self._line_traj_pred, x_pred, y_pred),
        (self._line_xt_true,   t_pts,  x_true),
        (self._line_xt_pred,   t_pts,  x_pred),
        (self._line_yt_true,   t_pts,  y_true),
        (self._line_yt_pred,   t_pts,  y_pred),
    ]:
        line.set_data(x, y)

    for ax in (self._ax_traj, self._ax_xt, self._ax_yt):
        ax.relim(); ax.autoscale_view()

    self.mpl.draw_idle()
```

### 2.3 MOD-03 — Overfitting Demo (Tab Widget + 3 Views)

The MOD-03 plot area uses a `QTabWidget` with three `MatplotlibWidget` instances:

```python
self.plot_tabs = QTabWidget()
self.mpl_pred   = MatplotlibWidget(figsize=(12, 5))  # Tab 0: Predictions
self.mpl_curves = MatplotlibWidget(figsize=(15, 5))  # Tab 1: Loss curves (1×3)
self.mpl_errors = MatplotlibWidget(figsize=(14, 8))  # Tab 2: Error analysis
self.plot_tabs.addTab(self.mpl_pred,   "Predictions")
self.plot_tabs.addTab(self.mpl_curves, "Loss Curves")
self.plot_tabs.addTab(self.mpl_errors, "Error Analysis")
```

**Overfitting onset detection:**

```python
def _detect_overfit_epoch(train_losses: list, val_losses: list,
                           warmup: int = 20) -> Optional[int]:
    """
    Returns the first epoch where val_loss > 2 × train_loss, after warmup.
    Returns None if overfitting not detected.
    """
    for i in range(warmup, len(train_losses)):
        if val_losses[i] > 2.0 * train_losses[i]:
            return i
    return None
```

**Predictions tab color spec:**

```python
COLORS = {
    'underfit': '#2196F3',   # Material Blue
    'good':     '#4CAF50',   # Material Green
    'overfit':  '#F44336',   # Material Red
    'true':     '#212121',   # Near-black
}
```

### 2.4 MOD-04 — Pendulum (Mixed: QPainter + Matplotlib)

MOD-04 uses a `QSplitter` dividing the canvas into:
- **Left (40%):** `PendulumCanvas` (QPainter — SDD-PHYSAI-002 §5)
- **Right (60%):** `QSplitter` vertical:
  - Top: θ(t) Matplotlib plot
  - Bottom: Phase space Matplotlib plot

```python
def _setup_layout(self):
    h_splitter = QSplitter(Qt.Horizontal)

    # Left: animation canvas
    self.pendulum_canvas = PendulumCanvas()
    h_splitter.addWidget(self.pendulum_canvas)

    # Right: two matplotlib plots stacked
    v_splitter = QSplitter(Qt.Vertical)
    self.mpl_theta = MatplotlibWidget(figsize=(8, 4))
    self.mpl_phase = MatplotlibWidget(figsize=(8, 4))
    v_splitter.addWidget(self.mpl_theta)
    v_splitter.addWidget(self.mpl_phase)
    h_splitter.addWidget(v_splitter)

    h_splitter.setSizes([400, 600])
```

**θ(t) plot with period markers:**

```python
def _render_theta_plot(self, t_arr: np.ndarray, theta_deg: np.ndarray,
                        T_exact: float, T_pred: float):
    ax = self.mpl_theta.fresh_axes()
    ax.plot(t_arr, theta_deg, 'b-', lw=2.0)
    ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)

    # Period boundary lines
    for k in range(1, 4):
        ax.axvline(k * T_exact, color='r', ls='--', lw=1.5, alpha=0.7,
                   label='Period' if k == 1 else None)

    ax.set_xlabel('Time (s)'); ax.set_ylabel('θ (°)')
    ax.set_title(f'θ₀={self._theta0_deg:.0f}°  '
                 f'T_exact={T_exact:.3f}s  T_pred={T_pred:.3f}s',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    self.mpl_theta.draw()
```

**Phase space plot:**

```python
def _render_phase_plot(self, theta_deg: np.ndarray, omega_deg_s: np.ndarray):
    ax = self.mpl_phase.fresh_axes()
    ax.plot(theta_deg, omega_deg_s, 'g-', lw=1.5, alpha=0.7)
    ax.scatter([theta_deg[0]], [omega_deg_s[0]], c='r', s=80, zorder=5, label='Start')
    ax.set_xlabel('θ (°)'); ax.set_ylabel('ω (°/s)')
    ax.set_title('Phase Space (θ vs ω)', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    self.mpl_phase.draw()
```

### 2.5 MOD-05 — Air Resistance (2×2 GridSpec)

```python
def _setup_plots(self):
    self.mpl = MatplotlibWidget(figsize=(14, 10))
    gs = self.mpl.fresh_gridspec(2, 2, hspace=0.4, wspace=0.35)
    self._ax_traj  = self.mpl.figure.add_subplot(gs[0, 0])
    self._ax_range = self.mpl.figure.add_subplot(gs[0, 1])
    self._ax_perf  = self.mpl.figure.add_subplot(gs[1, 0])
    self._ax_loss  = self.mpl.figure.add_subplot(gs[1, 1])
    self._annotate_axes()
    self.mpl.draw()

def _render_trajectory_panel(self, v0, theta, k, model):
    ax = self._ax_traj
    ax.clear()

    physics_air = ProjectilePhysics(k=k)
    physics_vac = ProjectilePhysics(k=0.0)

    traj_air = physics_air.simulate(v0, theta)
    traj_vac = physics_vac.simulate(v0, theta)
    R_pred = float(model.predict([[v0, theta]], verbose=0)[0, 0])
    R_air  = traj_air[-1, 0]
    R_vac  = traj_vac[-1, 0]

    ax.plot(traj_vac[:, 0], traj_vac[:, 1], 'b--', lw=2, label='Vacuum (Theory)')
    ax.plot(traj_air[:, 0], traj_air[:, 1], 'r-',  lw=2, label='Air Resistance (RK4)')
    ax.scatter([R_pred], [0], c='green', s=150, marker='*', zorder=5,
               label=f'AI Pred ({R_pred:.1f} m)')

    reduction = (1 - R_air / R_vac) * 100
    ax.text(0.05, 0.95, f"Range reduction: {reduction:.1f}%",
            transform=ax.transAxes, va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Trajectory (v₀={v0}m/s, θ={theta}°, k={k})', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0);   ax.set_ylim(bottom=0)
    self.mpl.draw_idle()
```

---

## 3. Thread-Safe Rendering Protocol

### 3.1 Data Flow Diagram

```
Worker Thread                         Main Thread (Qt Event Loop)
────────────────────────────────────────────────────────────────
Training running...
  QtProgressCallback.on_epoch_end()
    ↓ emit progress_updated(epoch, loss, val_loss)
                                    ↓ slot: _on_progress(epoch, loss, val_loss)
                                        ProgressPanel.update(epoch, loss, val_loss)
                                        _loss_buf.append(loss)
                                        _line_loss.set_data(epochs, losses)
                                        ax.relim(); ax.autoscale_view()
                                        canvas.draw_idle()   ← Main thread only ✓

Training complete:
  TrainingWorker.run() finishes
    ↓ emit training_finished(model, history)
                                    ↓ slot: _on_training_finished(model, history)
                                        Compute predictions (model.predict)
                                        Update all plot artists
                                        canvas.draw()        ← Main thread only ✓
```

### 3.2 draw_idle() vs draw() Policy

| Situation | Method | Rationale |
|-----------|--------|-----------|
| During training (every 10 epochs) | `draw_idle()` | Non-blocking; renderer may skip frames under load; prevents UI freeze |
| After training completes | `draw()` | Synchronous; ensures final result is fully rendered before user interaction |
| Parameter slider change (post-training) | `draw_idle()` | May be called rapidly; frame skipping acceptable |
| Export PNG | `draw()` then `savefig()` | Must be fully rendered before file write |
| Animation frame (QPainter, not Matplotlib) | `widget.update()` | QPainter uses Qt's paint event; not subject to Matplotlib threading |

### 3.3 Artist Pre-creation Pattern

For live-updating plots, all `Line2D` and `PathCollection` artists are **created once in `_setup_plots()`** and updated via `set_data()` in the update methods. This avoids the overhead of `ax.clear()` + full re-plot on every update:

```python
# Preferred pattern (fast):
self._line_loss, = ax.plot([], [], 'g-', lw=1.5)
# ... later in update callback:
self._line_loss.set_data(epochs, losses)
ax.relim(); ax.autoscale_view()
canvas.draw_idle()

# Anti-pattern (slow, causes flicker):
ax.clear()
ax.plot(epochs, losses, 'g-', lw=1.5)
canvas.draw_idle()
```

`fill_between()` is an exception: it cannot be updated in-place and must be removed and recreated. The reference is stored for efficient removal:
```python
if self._error_fill:
    self._error_fill.remove()
self._error_fill = ax.fill_between(x, 0, y, color='r', alpha=0.3)
```

---

## 4. Pendulum Animation Controller

```python
class PendulumAnimationController:
    """
    Drives the PendulumCanvas animation using a QTimer.
    Owns the timer; does not own the canvas (injected dependency).
    """
    TARGET_FPS = 30
    BASE_INTERVAL_MS = 1000 // TARGET_FPS  # 33 ms

    def __init__(self, canvas: PendulumCanvas):
        self._canvas   = canvas
        self._timer    = QTimer()
        self._timer.timeout.connect(self._advance_frame)
        self._trajectory: Optional[np.ndarray] = None  # shape (N, 2): [theta_rad, omega]
        self._frame_idx = 0
        self._speed     = 1.0

    def load(self, trajectory: np.ndarray, L: float):
        """Load new trajectory and reset to frame 0."""
        self._trajectory = trajectory
        self._frame_idx  = 0
        self._canvas.set_pendulum_length(L)

    def play(self):
        interval = max(1, int(self.BASE_INTERVAL_MS / self._speed))
        self._timer.start(interval)

    def pause(self):
        self._timer.stop()

    def toggle(self):
        self.pause() if self._timer.isActive() else self.play()

    def set_speed(self, multiplier: float):
        """multiplier: 0.25, 0.5, 1.0, 2.0, 4.0"""
        self._speed = multiplier
        if self._timer.isActive():
            self._timer.setInterval(max(1, int(self.BASE_INTERVAL_MS / multiplier)))

    def _advance_frame(self):
        if self._trajectory is None:
            return
        N = len(self._trajectory)
        if self._frame_idx >= N:
            self._frame_idx = 0  # Loop
        theta_rad = float(self._trajectory[self._frame_idx, 0])
        self._canvas.set_state(theta_rad)
        self._frame_idx += 1
```

**Frame rate analysis:**
- Timer interval: 33 ms (30 FPS target)
- `PendulumCanvas.paintEvent()` target: < 16 ms (NFR-MOD04-02)
- At 4× speed: interval = 8 ms; if paintEvent takes 16 ms, frames will be dropped gracefully by Qt's event coalescing

---

## 5. Export Pipeline

### 5.1 PNG Export

```python
# app/utils/export.py

class ExportManager:
    @staticmethod
    def export_png(mpl_widget: MatplotlibWidget,
                   module_id: str,
                   parent_widget=None) -> Optional[str]:
        """
        Saves the current figure to a user-selected PNG file.

        Returns: file path if saved, None if cancelled.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{module_id}_{timestamp}.png"
        default_dir  = str(Path.home() / "Downloads")

        path, _ = QFileDialog.getSaveFileName(
            parent=parent_widget,
            caption="Export Plot as PNG",
            dir=str(Path(default_dir) / default_name),
            filter="PNG Image (*.png);;All Files (*)"
        )
        if path:
            mpl_widget.figure.savefig(path, dpi=150, bbox_inches='tight')
        return path or None
```

### 5.2 JSON Export Schema

```python
@staticmethod
def export_json(module: 'BaseModule', parent_widget=None) -> Optional[str]:
    """
    Exports experiment parameters and results as a JSON file.
    Schema defined below.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "schema_version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "module": {
            "id": module.MODULE_ID,
            "name": module.MODULE_NAME,
        },
        "parameters": module.get_param_values(),    # dict: {param_name: value}
        "training": {
            "epochs_run": module.epochs_run,
            "final_loss": module.final_loss,
            "final_val_loss": module.final_val_loss,
            "elapsed_ms": module.training_elapsed_ms,
        },
        "metrics": module.get_metrics(),             # dict: {metric_name: value}
    }
    # ... file dialog and json.dump
```

**Example output (MOD-04):**
```json
{
  "schema_version": "1.0",
  "exported_at": "2026-04-06T11:23:45.123456",
  "module": { "id": "MOD-04", "name": "Pendulum Simulation" },
  "parameters": {
    "L_m": 1.0,
    "theta0_deg": 45.0,
    "n_samples": 2000,
    "epochs": 100,
    "noise_frac": 0.01
  },
  "training": {
    "epochs_run": 100,
    "final_loss": 0.000241,
    "final_val_loss": 0.000318,
    "elapsed_ms": 8432
  },
  "metrics": {
    "test_mse": 0.000228,
    "test_mae": 0.009134,
    "test_mape_pct": 0.412
  }
}
```

---

## 6. Matplotlib Style Synchronization with Theme

When the theme is toggled (SDD-PHYSAI-002 §7), all existing Matplotlib figures must update their background colors:

```python
class ThemeManager:
    @classmethod
    def _sync_matplotlib_figures(cls, dark: bool):
        """Updates all open Matplotlib figures to match the current theme."""
        import matplotlib.pyplot as plt
        style = 'dark_background' if dark else 'default'
        plt.style.use(style)

        # Apply to all existing figure canvases in the application
        for widget in QApplication.instance().allWidgets():
            if isinstance(widget, MatplotlibWidget):
                bg = '#1E1E1E' if dark else '#FFFFFF'
                widget.figure.set_facecolor(bg)
                for ax in widget.figure.get_axes():
                    ax.set_facecolor(bg)
                    ax.tick_params(colors='white' if dark else 'black')
                    ax.xaxis.label.set_color('white' if dark else 'black')
                    ax.yaxis.label.set_color('white' if dark else 'black')
                    ax.title.set_color('white' if dark else 'black')
                widget.draw_idle()
```

---

## 7. Accessibility and Legibility Standards

| Standard | Specification |
|----------|---------------|
| Minimum font size | 11pt for all text in plots and widgets |
| Line width | ≥ 2.0 for primary data lines; ≥ 1.5 for secondary |
| Color contrast | All line colors distinguishable in deuteranopia simulation (verified with `matplotlib.colors` HSV analysis) |
| Legend | Always present on plots with multiple data series; `fontsize=9` minimum |
| Axis labels | Always include units in parentheses: e.g., `x (m)`, `θ (°)`, `Loss (MSE)` |
| Grid | `alpha=0.3` on all plots to reduce visual noise |
| HiDPI | `canvas.setSizePolicy(Expanding, Expanding)` + `resizeEvent` draw_idle call |

---

## 8. Traceability to Requirements

| Requirement ID | Design Section | Notes |
|---------------|----------------|-------|
| FR-16 (primary + loss plot) | §2.1, §2.2, §2.4, §2.5 | Each module provides ≥2 plots |
| FR-17 (axis labels, title with metric) | §2.1 (_render_complete), §2.4 | Title format specified |
| FR-18 (NavigationToolbar) | SDD-PHYSAI-002 §4.1 | Toolbar included in MatplotlibWidget constructor |
| FR-19 (log scale y-axis) | §2.1 (ax_loss.set_yscale) | Applied when loss range > 2 OoM |
| FR-20 (draw_idle only during training) | §3.1, §3.2 | Protocol enforced by design |
| FR-21 (PNG export Ctrl+E) | §5.1 | Keyboard shortcut in SDD-PHYSAI-002 §8 |
| FR-22 (JSON export) | §5.2 | Schema defined above |
| FR-23 (default filename format) | §5.1, §5.2 | `{module_id}_{YYYYMMDD_HHMMSS}` |
| NFR-03 (draw_idle < 33ms) | §3.2, §3.3 | Pre-created artists minimize per-update work |
| NFR-MOD04-02 (paintEvent < 16ms) | §4 | QPainter operations bounded by canvas size |
