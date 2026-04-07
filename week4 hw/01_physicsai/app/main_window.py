"""
MainWindow — root window, tab registration, menu/toolbar, status bar, shortcuts.
Module registration follows TRD-01 §6.1.
"""
import time
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu, QToolBar,
    QStatusBar, QLabel, QApplication, QMessageBox,
)
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QIcon, QFont

from app.modules.mod01_function    import FunctionApproximationModule
from app.modules.mod02_projectile  import ProjectileRegressionModule
from app.modules.mod03_overfitting import OverfittingDemoModule
from app.modules.mod04_pendulum    import PendulumModule
from app.modules.mod05_air_resist  import AirResistanceModule
from app.utils.export              import ExportManager
from app.utils.theme               import ThemeManager

# ── Module registry (TRD-01 §6.1) ────────────────────────────────────────────
MODULES = [
    ("1D Function Approx",  FunctionApproximationModule),
    ("Projectile Motion",   ProjectileRegressionModule),
    ("Overfitting Demo",    OverfittingDemoModule),
    ("Pendulum Simulation", PendulumModule),
    ("Air Resistance",      AirResistanceModule),
]

# Emoji icons for tabs
TAB_ICONS = ["🔢", "🚀", "📊", "🕰️", "💨"]


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setMinimumSize(1280, 800)
        self.setWindowTitle("PhysicsAI Simulator  v1.0")
        self._session_start = time.monotonic()
        self._gpu_label_text = self._detect_gpu()

        self._setup_tab_widget()
        self._setup_menu_bar()
        self._setup_toolbar()
        self._setup_status_bar()
        self._register_shortcuts()
        self._start_session_timer()

    # ── Tab widget ────────────────────────────────────────────────────────────
    def _setup_tab_widget(self):
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self._modules = []
        for i, (label, ModCls) in enumerate(MODULES):
            mod = ModCls()
            tab_label = f"{TAB_ICONS[i]}  {label}"
            self.tabs.addTab(mod, tab_label)
            self._modules.append(mod)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

    # ── Menu bar ──────────────────────────────────────────────────────────────
    def _setup_menu_bar(self):
        mb = self.menuBar()

        # File
        m_file = mb.addMenu("&File")
        act_png  = QAction("Export PNG",  self, shortcut="Ctrl+E")
        act_json = QAction("Export JSON", self, shortcut="Ctrl+Shift+E")
        act_quit = QAction("Quit",        self, shortcut="Ctrl+Q")
        act_png.triggered.connect(self._export_png)
        act_json.triggered.connect(self._export_json)
        act_quit.triggered.connect(QApplication.quit)
        m_file.addAction(act_png)
        m_file.addAction(act_json)
        m_file.addSeparator()
        m_file.addAction(act_quit)

        # View
        m_view = mb.addMenu("&View")
        act_theme = QAction("Toggle Theme  (Dark/Light)", self, shortcut="Ctrl+D")
        act_theme.triggered.connect(
            lambda: ThemeManager.toggle(QApplication.instance()))
        m_view.addAction(act_theme)

        # Help
        m_help = mb.addMenu("&Help")
        act_help  = QAction("Module Help", self, shortcut="F1")
        act_about = QAction("About",       self)
        act_help.triggered.connect(self._show_help)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_help)
        m_help.addAction(act_about)

    # ── Toolbar ───────────────────────────────────────────────────────────────
    def _setup_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))

        def btn(text, shortcut, handler, tooltip=""):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(handler)
            if tooltip:
                a.setToolTip(tooltip)
            return a

        tb.addAction(btn("▶ Run",   "Ctrl+R", lambda: self._active().run(),   "Run training (Ctrl+R)"))
        tb.addAction(btn("■ Stop",  "Ctrl+.", lambda: self._active().stop(),  "Stop training (Ctrl+.)"))
        tb.addAction(btn("↺ Reset", "",       lambda: self._active().reset(), "Reset module"))
        tb.addSeparator()
        tb.addAction(btn("📷 Export PNG",  "Ctrl+E",       self._export_png,  "Export plot (Ctrl+E)"))
        tb.addAction(btn("📄 Export JSON", "Ctrl+Shift+E", self._export_json, "Export results (Ctrl+Shift+E)"))
        tb.addSeparator()
        tb.addAction(btn("🌙 Theme", "Ctrl+D",
                         lambda: ThemeManager.toggle(QApplication.instance()),
                         "Toggle dark/light theme (Ctrl+D)"))

    # ── Status bar ────────────────────────────────────────────────────────────
    def _setup_status_bar(self):
        sb = self.statusBar()
        sb.setFixedHeight(28)

        self._sb_module = QLabel("Ready")
        self._sb_gpu    = QLabel(self._gpu_label_text)
        self._sb_time   = QLabel("00:00:00")

        sb.addWidget(self._sb_module, 1)
        sb.addPermanentWidget(self._sb_gpu)
        sb.addPermanentWidget(QLabel("  |  "))
        sb.addPermanentWidget(self._sb_time)

    def _start_session_timer(self):
        t = QTimer(self)
        t.timeout.connect(self._tick_session)
        t.start(1000)

    def _tick_session(self):
        elapsed = int(time.monotonic() - self._session_start)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        self._sb_time.setText(f"{h:02d}:{m:02d}:{s:02d}")

        mod = self._active()
        if hasattr(mod, '_state'):
            self._sb_module.setText(
                f"{MODULES[self.tabs.currentIndex()][0]}  —  {mod._state.name}")

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    def _register_shortcuts(self):
        for i in range(1, 6):
            sc = QShortcut(QKeySequence(str(i)), self)
            sc.activated.connect(lambda idx=i-1: self.tabs.setCurrentIndex(idx))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _active(self):
        return self.tabs.currentWidget()

    def _on_tab_changed(self, idx: int):
        mod = self._modules[idx]
        self._sb_module.setText(f"{MODULES[idx][0]}  —  {mod._state.name}")

    def _export_png(self):
        mod = self._active()
        for attr in ('mpl', 'mpl_pred', 'mpl_theta'):
            if hasattr(mod, attr):
                ExportManager.export_png(getattr(mod, attr), mod.MODULE_ID, self)
                return

    def _export_json(self):
        ExportManager.export_json(self._active(), self)

    @staticmethod
    def _detect_gpu() -> str:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                return f"GPU: {gpus[0].name.split('/')[-1]}"
            return "CPU Mode"
        except Exception:
            return "CPU Mode"

    def _show_help(self):
        mod  = self._active()
        name = mod.MODULE_NAME
        desc = mod.MODULE_DESC
        QMessageBox.information(self, f"Help — {name}",
            f"<h3>{name}</h3><p>{desc}</p>"
            f"<p><b>Keyboard Shortcuts:</b><br>"
            f"<tt>Ctrl+R</tt>  Run training<br>"
            f"<tt>Ctrl+.</tt>  Stop training<br>"
            f"<tt>Ctrl+E</tt>  Export PNG<br>"
            f"<tt>Ctrl+D</tt>  Toggle theme<br>"
            f"<tt>1–5</tt>     Switch tabs</tt></p>")

    def _show_about(self):
        QMessageBox.about(self, "About PhysicsAI Simulator",
            "<h2>PhysicsAI Simulator</h2>"
            "<p>Version 1.0.0 — IEEE Std 830 / 1016 compliant</p>"
            "<p>Integrates classical physics simulations with neural network regression.</p>"
            "<hr>"
            "<p><b>Stack:</b> PySide6 · TensorFlow/Keras · NumPy · Matplotlib</p>"
            "<p><b>Author:</b> JSW · 2026-04-06</p>")

    # ── Clean shutdown (TRD-01 §4.4) ─────────────────────────────────────────
    def closeEvent(self, event):
        for mod in self._modules:
            mod.cleanup()
        event.accept()
