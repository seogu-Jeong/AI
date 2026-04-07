# Software Design Description
## PhysicsAI Simulator — UI Component Design
**Document ID:** SDD-PHYSAI-002  
**Version:** 2.0  
**Date:** 2026-04-06  
**Standard:** IEEE Std 1016-2009  
**Parent Document:** SDD-PHYSAI-001  
**Status:** Approved  

---

## 1. Introduction

### 1.1 Purpose

This SDD specifies the design of all PySide6 UI components: composite widgets, layout strategy, state-driven styling, keyboard navigation, and theme management. It provides implementation-ready class specifications with interfaces and invariants.

### 1.2 Design Principles

- **Composability:** Every reusable element is a `QWidget` subclass with a clean signal interface; no component reaches into another's internals.
- **State-driven appearance:** Widget styling is driven by the module state machine, not by ad hoc `setEnabled()` calls scattered across event handlers.
- **Separation of display and logic:** UI components emit signals when values change; they do not contain physics or ML logic.
- **Qt convention:** All widgets use `QSizePolicy` correctly; no hardcoded pixel sizes except where fixed width is a deliberate layout constraint.

---

## 2. Main Window Design

### 2.1 MainWindow Layout

```python
class MainWindow(QMainWindow):
    """
    Root window. Owns the tab widget and global menu/toolbar.
    Does not hold references to module state beyond what QTabWidget provides.
    """
    def __init__(self):
        self.setMinimumSize(1200, 800)
        self.setWindowTitle("PhysicsAI Simulator")

        # Central widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)  # macOS-native look
        self.setCentralWidget(self.tab_widget)

        self._setup_menu_bar()
        self._setup_toolbar()
        self._setup_status_bar()
        self._register_shortcuts()
```

### 2.2 Menu Bar

| Menu | Action | Shortcut | Behavior |
|------|--------|----------|----------|
| File | Export PNG | Ctrl+E | Calls ExportManager.export_png on current tab's plot |
| File | Export JSON | Ctrl+Shift+E | Exports current module's params + metrics |
| File | Quit | Ctrl+Q | Initiates clean shutdown (SDD-PHYSAI-001 §4.4) |
| View | Toggle Theme | Ctrl+D | Switches between light and dark theme |
| View | Reset Layout | — | Restores splitter positions to defaults |
| Help | Module Help | F1 | Opens module-specific help dialog |
| Help | About | — | Displays version, license, dependency list |

### 2.3 Toolbar

```python
self.toolbar = self.addToolBar("Main")
self.toolbar.setMovable(False)
self.toolbar.setIconSize(QSize(20, 20))

# Actions bound to the *currently active* tab's module
self.act_run    = QAction(QIcon(":/icons/run.svg"),    "Run    Ctrl+R", self)
self.act_stop   = QAction(QIcon(":/icons/stop.svg"),   "Stop   Ctrl+S", self)
self.act_reset  = QAction(QIcon(":/icons/reset.svg"),  "Reset", self)
self.act_export = QAction(QIcon(":/icons/export.svg"), "Export Ctrl+E", self)
```

Toolbar actions delegate to the active module:
```python
def _active_module(self) -> BaseModule:
    return self.tab_widget.currentWidget()
```

### 2.4 Status Bar

```
[Current module state: IDLE]    [CPU Mode | GPU: None]    [Session: 00:12:34]
 ← left (expanding) →               ← center →              ← right (fixed) →
```

The status bar updates via a `QTimer` (1-second interval) for session time, and via module signal for state changes.

### 2.5 Tab Keyboard Navigation

```python
for i in range(1, 6):
    shortcut = QShortcut(QKeySequence(str(i)), self)
    shortcut.activated.connect(lambda idx=i-1: self.tab_widget.setCurrentIndex(idx))
```

---

## 3. Composite Widgets

### 3.1 SliderSpinBox

**Purpose:** A composite widget pairing `QSlider` (integer internally) with `QDoubleSpinBox`, kept in two-way sync.

```python
class SliderSpinBox(QWidget):
    """
    Composite widget: QSlider + QDoubleSpinBox synchronized.

    Internally, the slider operates on integer steps for smooth dragging.
    Values are converted: slider_int = round((value - min) / step).

    Signals:
        value_changed(float): emitted when value changes from either control
    """
    value_changed = Signal(float)

    def __init__(self,
                 label: str,
                 min_val: float,
                 max_val: float,
                 default: float,
                 step: float = 1.0,
                 unit: str = "",
                 decimals: int = 2,
                 log_scale: bool = False,
                 parent=None):
        ...

    # Public interface
    @property
    def value(self) -> float: ...

    @value.setter
    def value(self, v: float): ...

    def reset(self): ...

    def set_enabled(self, enabled: bool): ...
```

**Layout:**
```
QVBoxLayout:
  QLabel("label [unit]")       ← bold, 11pt
  QHBoxLayout:
    QSlider(Qt.Horizontal)     ← stretches
    QDoubleSpinBox             ← fixed width 80px
```

**Sync logic:**
```python
def _on_slider_moved(self, int_val: int):
    float_val = self._int_to_float(int_val)
    self._spinbox.blockSignals(True)
    self._spinbox.setValue(float_val)
    self._spinbox.blockSignals(False)
    self.value_changed.emit(float_val)

def _on_spinbox_changed(self, float_val: float):
    int_val = self._float_to_int(float_val)
    self._slider.blockSignals(True)
    self._slider.setValue(int_val)
    self._slider.blockSignals(False)
    self.value_changed.emit(float_val)
```

`blockSignals()` prevents feedback loops; propagation latency is bounded by a single `setValue()` call, meeting FR-06 (< 16 ms).

### 3.2 ParamGroup

**Purpose:** Groups related `SliderSpinBox` and other controls into a titled `QGroupBox`.

```python
class ParamGroup(QGroupBox):
    """
    Container for related parameter widgets.
    Provides factory methods to add controls and a unified get/reset interface.
    """
    any_value_changed = Signal(str, object)  # (param_name, new_value)

    def add_slider(self, name: str, **kwargs) -> SliderSpinBox:
        widget = SliderSpinBox(**kwargs)
        widget.value_changed.connect(lambda v, n=name: self.any_value_changed.emit(n, v))
        self._widgets[name] = widget
        self._layout.addWidget(widget)
        return widget

    def add_combo(self, name: str, label: str, options: List[str]) -> QComboBox:
        ...

    def add_checkbox(self, name: str, label: str, default: bool = False) -> QCheckBox:
        ...

    def values(self) -> dict:
        """Returns {name: current_value} for all registered controls."""
        return {name: w.value for name, w in self._widgets.items()}

    def reset(self):
        """Restores all controls to their default values."""
        for w in self._widgets.values():
            w.reset()
```

### 3.3 ControlPanel

**Purpose:** Run/Stop/Reset buttons with state-aware enable/disable logic.

```python
class ControlPanel(QWidget):
    run_requested   = Signal()
    stop_requested  = Signal()
    reset_requested = Signal()

    def __init__(self, parent=None):
        self.run_btn   = QPushButton("▶  Run")
        self.stop_btn  = QPushButton("■  Stop")
        self.reset_btn = QPushButton("↺  Reset")

        self.run_btn.setObjectName("run_btn")
        self.stop_btn.setObjectName("stop_btn")
        # Styled via QSS (Section 7)

    def apply_state(self, state: ModuleState):
        """Single entry point for state-driven button updates."""
        self.run_btn.setEnabled(state in (ModuleState.IDLE, ModuleState.TRAINED))
        self.stop_btn.setEnabled(state == ModuleState.TRAINING)
        self.reset_btn.setEnabled(state in (ModuleState.TRAINED, ModuleState.ERROR))

        # Dirty indicator: amber Run button when params changed post-training
        if state == ModuleState.DIRTY:
            self.run_btn.setStyleSheet("background-color: #FF9800;")
        else:
            self.run_btn.setStyleSheet("")  # Revert to QSS default
```

**State → Button matrix:**

| Module State | Run | Stop | Reset |
|-------------|-----|------|-------|
| IDLE | ✓ enabled | ✗ | ✗ |
| TRAINING | ✗ | ✓ enabled | ✗ |
| TRAINED | ✓ (green) | ✗ | ✓ enabled |
| DIRTY | ✓ (amber) | ✗ | ✓ enabled |
| ERROR | ✗ | ✗ | ✓ enabled |

### 3.4 ProgressPanel

**Purpose:** Displays training progress with epoch counter, loss metrics, and elapsed time.

```python
class ProgressPanel(QWidget):
    def __init__(self):
        self.progress_bar    = QProgressBar()
        self.epoch_label     = QLabel("Epoch: — / —")
        self.loss_label      = QLabel("Loss: —")
        self.val_loss_label  = QLabel("Val Loss: —")
        self.time_label      = QLabel("")
        self._start_time: Optional[float] = None

    def start(self, total_epochs: int):
        """Called when training begins."""
        self.progress_bar.setRange(0, total_epochs)
        self.progress_bar.setValue(0)
        self._start_time = time.monotonic()
        self.time_label.clear()
        for label in (self.loss_label, self.val_loss_label):
            label.setStyleSheet("")  # Clear completion highlight

    def update(self, epoch: int, total: int, loss: float, val_loss: float):
        self.progress_bar.setValue(epoch)
        self.epoch_label.setText(f"Epoch: {epoch:,} / {total:,}")
        self.loss_label.setText(f"Loss: {loss:.6f}")
        if val_loss is not None:
            self.val_loss_label.setText(f"Val Loss: {val_loss:.6f}")

    def complete(self):
        elapsed_ms = int((time.monotonic() - self._start_time) * 1000)
        self.time_label.setText(f"Completed in {elapsed_ms:,} ms")
        self.loss_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.progress_bar.setValue(self.progress_bar.maximum())

    def error(self, message: str):
        self.time_label.setText(f"Error: {message[:60]}")
        self.time_label.setStyleSheet("color: #F44336;")

    def reset(self):
        self.progress_bar.setValue(0)
        for label in (self.epoch_label, self.loss_label, self.val_loss_label, self.time_label):
            label.clear()
            label.setStyleSheet("")
```

---

## 4. MatplotlibWidget

### 4.1 Class Design

```python
class MatplotlibWidget(QWidget):
    """
    Embeds a Matplotlib Figure in a PySide6 widget via FigureCanvasQTAgg.

    Thread safety:
        draw_idle() is the only allowed draw method during training.
        draw() may be called only from the main thread after training completes.

    Usage:
        widget = MatplotlibWidget(figsize=(12, 6), dpi=100)
        axes = widget.fresh_axes(nrows=1, ncols=3)
        axes[0].plot(x, y)
        widget.draw()
    """
    def __init__(self, figsize=(12, 6), dpi=100, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def fresh_axes(self, nrows=1, ncols=1, **subplot_kw):
        """Clears the figure and returns new axes array."""
        self.figure.clear()
        if nrows == 1 and ncols == 1:
            return self.figure.add_subplot(111, **subplot_kw)
        return self.figure.subplots(nrows, ncols, **subplot_kw)

    def fresh_gridspec(self, nrows, ncols, **gs_kw):
        """Returns a GridSpec for irregular subplot layouts."""
        self.figure.clear()
        return self.figure.add_gridspec(nrows, ncols, **gs_kw)

    def draw(self):
        """Full synchronous redraw. Main thread only."""
        self.figure.tight_layout()
        self.canvas.draw()

    def draw_idle(self):
        """Non-blocking redraw hint. Safe to call frequently."""
        self.canvas.draw_idle()

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def export_png(self, path: str, dpi: int = 150):
        self.figure.savefig(path, dpi=dpi, bbox_inches='tight')

    def resizeEvent(self, event):
        """Maintain aspect ratio on window resize."""
        super().resizeEvent(event)
        self.canvas.draw_idle()
```

### 4.2 Usage Pattern in Modules

```python
# During training (from Signal slot in main thread):
def _on_progress(self, epoch, loss, val_loss):
    self.progress_panel.update(epoch, self.config.epochs, loss, val_loss)
    # Update only the loss line; avoid full redraw
    self._loss_line.set_xdata(self._epochs_so_far)
    self._loss_line.set_ydata(self._losses_so_far)
    self._ax_loss.relim()
    self._ax_loss.autoscale_view()
    self.plot_widget.draw_idle()  # ← Non-blocking; may skip frames

# After training (full redraw):
def _on_training_finished(self, model, history):
    axes = self.plot_widget.fresh_axes(nrows=1, ncols=3)
    self._render_full_result(axes, model, history)
    self.plot_widget.draw()  # ← Full draw; safe here
```

---

## 5. PendulumCanvas

### 5.1 Class Design

```python
class PendulumCanvas(QWidget):
    """
    QPainter-based widget for pendulum animation.
    Receives angular position updates via set_state() called by PendulumAnimationController.
    Does not own the animation timer.
    """
    TRAIL_LENGTH = 30  # number of historical positions to render
    BOB_RADIUS   = 16  # pixels
    PIVOT_RADIUS  = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._theta_rad: float = 0.0
        self._L: float = 1.0          # meters, for visual scaling
        self._trail: deque = deque(maxlen=self.TRAIL_LENGTH)
        self._dark_mode: bool = False

    def set_state(self, theta_rad: float):
        """Called each animation frame with the current angle in radians."""
        self._theta_rad = theta_rad
        px, py = self._bob_pixel_position()
        self._trail.append((px, py))
        self.update()  # Schedules repaint

    def set_pendulum_length(self, L_meters: float):
        self._L = L_meters
        self._trail.clear()
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg = QColor("#1E1E1E") if self._dark_mode else QColor("#FFFFFF")
        painter.fillRect(self.rect(), bg)

        pivot = self._pivot_pixel()
        bob   = self._bob_pixel_position()

        self._draw_trail(painter)
        self._draw_angle_arc(painter, pivot)
        self._draw_rod(painter, pivot, bob)
        self._draw_bob(painter, bob)
        self._draw_pivot(painter, pivot)
        self._draw_angle_label(painter, pivot)

    def _pivot_pixel(self) -> QPoint:
        return QPoint(self.width() // 2, 60)

    def _bob_pixel_position(self) -> QPoint:
        pivot = self._pivot_pixel()
        scale = self._rod_pixel_length()
        px = pivot.x() + scale * math.sin(self._theta_rad)
        py = pivot.y() + scale * math.cos(self._theta_rad)
        return QPoint(int(px), int(py))

    def _rod_pixel_length(self) -> float:
        """Scales rod length to available canvas space."""
        max_rod = min(self.width() // 2, self.height() - 120)
        L_max = 3.0  # maximum pendulum length in meters
        return max_rod * (self._L / L_max)

    def _draw_trail(self, painter: QPainter):
        trail = list(self._trail)
        for i in range(1, len(trail)):
            alpha = int(30 + 225 * i / len(trail))
            color = QColor(33, 150, 243, alpha)
            painter.setPen(QPen(color, 2))
            painter.drawLine(trail[i-1][0], trail[i-1][1],
                             trail[i][0],   trail[i][1])

    def _draw_rod(self, painter, pivot: QPoint, bob: QPoint):
        rod_color = QColor("#424242") if self._dark_mode else QColor("#212121")
        painter.setPen(QPen(rod_color, 2))
        painter.drawLine(pivot, bob)

    def _draw_bob(self, painter, center: QPoint):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#2196F3")))
        painter.drawEllipse(center, self.BOB_RADIUS, self.BOB_RADIUS)
        # Highlight
        painter.setBrush(QBrush(QColor(255, 255, 255, 80)))
        painter.drawEllipse(center - QPoint(4, 4), 5, 5)

    def _draw_pivot(self, painter, center: QPoint):
        pivot_color = QColor("#FFFFFF") if self._dark_mode else QColor("#212121")
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(pivot_color))
        painter.drawEllipse(center, self.PIVOT_RADIUS, self.PIVOT_RADIUS)

    def _draw_angle_arc(self, painter, pivot: QPoint):
        if abs(self._theta_rad) < 0.01:
            return
        arc_r = 40
        painter.setPen(QPen(QColor("#9E9E9E"), 1, Qt.DashLine))
        painter.drawArc(pivot.x()-arc_r, pivot.y(), arc_r*2, arc_r*2,
                        90 * 16, int(-math.degrees(self._theta_rad) * 16))

    def _draw_angle_label(self, painter, pivot: QPoint):
        deg = math.degrees(self._theta_rad)
        painter.setPen(QColor("#757575"))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(pivot.x() + 50, pivot.y() + 20, f"θ = {deg:.1f}°")
```

---

## 6. Period Prediction Panel (MOD-04)

```python
class PeriodPredictionPanel(QWidget):
    """Displays T_small, T_exact, T_pred with error indicators."""

    OVERFIT_THRESHOLD_DEG = 30.0

    def update(self, L: float, theta0_deg: float, T_pred: Optional[float] = None):
        g = 9.81
        T_small = 2 * math.pi * math.sqrt(L / g)
        T_exact = PendulumPhysics.true_period(L, theta0_deg)

        err_small = abs(T_small - T_exact) / T_exact * 100

        self.t_small_label.setText(f"T_small  = {T_small:.4f} s  (err: {err_small:.2f}%)")

        # Amber highlight when small-angle approximation breaks down
        if theta0_deg > self.OVERFIT_THRESHOLD_DEG:
            self.t_small_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        else:
            self.t_small_label.setStyleSheet("")

        self.t_exact_label.setText(f"T_exact  = {T_exact:.4f} s")

        if T_pred is not None:
            err_pred = abs(T_pred - T_exact) / T_exact * 100
            self.t_pred_label.setText(f"T_pred   = {T_pred:.4f} s  (err: {err_pred:.2f}%)")
        else:
            self.t_pred_label.setText("T_pred   = — (not trained)")
```

---

## 7. Theme System

### 7.1 QSS Stylesheets

```python
# app/utils/theme.py

COMMON_QSS = """
QWidget { font-size: 11pt; }
QGroupBox {
    font-weight: bold;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 12px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QProgressBar {
    border-radius: 4px;
    text-align: center;
    height: 18px;
}
QProgressBar::chunk { border-radius: 3px; }
QPushButton {
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
    min-height: 32px;
}
QPushButton#run_btn  { background: #4CAF50; color: white; }
QPushButton#stop_btn { background: #F44336; color: white; }
QPushButton:disabled { background: #9E9E9E; color: #E0E0E0; }
QTabBar::tab { padding: 8px 20px; }
QTabBar::tab:selected { font-weight: bold; }
"""

LIGHT_QSS = COMMON_QSS + """
QMainWindow, QWidget { background: #FAFAFA; color: #212121; }
QGroupBox { border: 1px solid #E0E0E0; }
QTabWidget::pane { border: 1px solid #BDBDBD; }
QProgressBar { background: #E0E0E0; }
QProgressBar::chunk { background: #2196F3; }
"""

DARK_QSS = COMMON_QSS + """
QMainWindow, QWidget { background: #1E1E1E; color: #EEEEEE; }
QGroupBox { border: 1px solid #424242; }
QTabWidget::pane { border: 1px solid #616161; }
QProgressBar { background: #424242; }
QProgressBar::chunk { background: #42A5F5; }
QSlider::groove:horizontal { background: #424242; }
QSlider::handle:horizontal { background: #90CAF9; }
"""

class ThemeManager:
    _current: str = "light"

    @classmethod
    def apply_light(cls, app: QApplication):
        app.setStyleSheet(LIGHT_QSS)
        cls._current = "light"
        plt.style.use('default')

    @classmethod
    def apply_dark(cls, app: QApplication):
        app.setStyleSheet(DARK_QSS)
        cls._current = "dark"
        plt.style.use('dark_background')

    @classmethod
    def toggle(cls, app: QApplication):
        if cls._current == "light":
            cls.apply_dark(app)
        else:
            cls.apply_light(app)
```

### 7.2 Korean Font Detection

```python
# app/utils/platform_utils.py

def configure_korean_font():
    """Detects and sets the best available Korean-compatible font for Matplotlib."""
    PRIORITY = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Gulim', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in PRIORITY:
        if font in available:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False
```

---

## 8. Keyboard Shortcuts Registry

```python
# app/main_window.py — _register_shortcuts()

shortcuts = {
    "Ctrl+R": lambda: self._active_module().run(),
    "Ctrl+.": lambda: self._active_module().stop(),   # Ctrl+. as Stop (cross-platform)
    "Ctrl+E": lambda: ExportManager.export_png(self._active_plot_widget()),
    "Ctrl+Shift+E": lambda: ExportManager.export_json(self._active_module()),
    "Ctrl+D": lambda: ThemeManager.toggle(QApplication.instance()),
    "F1":     lambda: self._show_module_help(),
    **{str(i): (lambda idx=i-1: self.tab_widget.setCurrentIndex(idx))
       for i in range(1, 6)},
}
for key, handler in shortcuts.items():
    QShortcut(QKeySequence(key), self).activated.connect(handler)
```

Note: `Ctrl+S` is reserved for PySide6 internal use on some platforms; `Ctrl+.` is used for Stop to avoid conflicts.
