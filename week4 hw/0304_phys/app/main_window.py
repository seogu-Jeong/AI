"""MainWindow — 3-tab QTabWidget, menu, toolbar, status bar, shortcuts."""
import time
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtWidgets import (QMainWindow, QTabWidget, QLabel, QApplication,
                                QMessageBox)
from PySide6.QtGui import QAction, QKeySequence, QShortcut

from app.modules.mod02_projectile  import ProjectileMotionModule
from app.modules.mod04_pendulum    import PendulumModule
from app.modules.mod05_airresist   import AirResistanceModule
from app.utils.export  import ExportManager
from app.utils.theme   import ThemeManager

MODULES = [
    ("🚀  Projectile",  ProjectileMotionModule),
    ("🔭  Pendulum",    PendulumModule),
    ("💨  Air Resist",  AirResistanceModule),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(1280, 820)
        self.setWindowTitle("PhysicsAI 0304통합  v1.0")
        self._session_start = time.monotonic()
        self._setup_tabs()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._register_shortcuts()
        t = QTimer(self); t.timeout.connect(self._tick); t.start(1000)

    def _setup_tabs(self):
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self._mods = []
        for label, Cls in MODULES:
            m = Cls(); self.tabs.addTab(m, label); self._mods.append(m)
        self.tabs.currentChanged.connect(self._on_tab)
        self.setCentralWidget(self.tabs)

    def _setup_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        for text, key, handler in [
            ("Export PNG",  "Ctrl+E",       self._export_png),
            ("Export JSON", "Ctrl+Shift+E", self._export_json),
        ]:
            a = QAction(text, self, shortcut=key); a.triggered.connect(handler)
            m_file.addAction(a)
        m_file.addSeparator()
        a = QAction("Quit", self, shortcut="Ctrl+Q")
        a.triggered.connect(QApplication.quit); m_file.addAction(a)
        m_view = mb.addMenu("&View")
        a = QAction("Toggle Theme  Ctrl+D", self)
        a.triggered.connect(lambda: ThemeManager.toggle(QApplication.instance()))
        m_view.addAction(a)
        m_help = mb.addMenu("&Help")
        a = QAction("About", self); a.triggered.connect(self._about)
        m_help.addAction(a)

    def _setup_toolbar(self):
        tb = self.addToolBar("Main"); tb.setMovable(False); tb.setIconSize(QSize(18,18))
        for text, key, fn, tip in [
            ("▶ Run",   "Ctrl+R", lambda: self._active().run(),   "Run training"),
            ("■ Stop",  "Ctrl+.", lambda: self._active().stop(),  "Stop"),
            ("↺ Reset", "",       lambda: self._active().reset(), "Reset"),
        ]:
            a = QAction(text, self, toolTip=tip)
            if key: a.setShortcut(key)
            a.triggered.connect(fn); tb.addAction(a)
        tb.addSeparator()
        a = QAction("📷 PNG", self, shortcut="Ctrl+E")
        a.triggered.connect(self._export_png); tb.addAction(a)
        a = QAction("🌙 Theme", self, shortcut="Ctrl+D")
        a.triggered.connect(lambda: ThemeManager.toggle(QApplication.instance()))
        tb.addAction(a)

    def _setup_statusbar(self):
        sb = self.statusBar(); sb.setFixedHeight(28)
        self._sb_mod  = QLabel("Ready")
        self._sb_gpu  = QLabel(self._detect_gpu())
        self._sb_time = QLabel("00:00:00")
        sb.addWidget(self._sb_mod, 1)
        sb.addPermanentWidget(self._sb_gpu)
        sb.addPermanentWidget(QLabel("  |  "))
        sb.addPermanentWidget(self._sb_time)

    def _tick(self):
        e = int(time.monotonic() - self._session_start)
        h, r = divmod(e, 3600); m, s = divmod(r, 60)
        self._sb_time.setText(f"{h:02d}:{m:02d}:{s:02d}")
        mod = self._active()
        if hasattr(mod, '_state'):
            idx = self.tabs.currentIndex()
            self._sb_mod.setText(f"{MODULES[idx][0].split('  ')[1]}  —  {mod._state.name}")

    def _register_shortcuts(self):
        for i in range(1, 4):
            sc = QShortcut(QKeySequence(str(i)), self)
            sc.activated.connect(lambda idx=i-1: self.tabs.setCurrentIndex(idx))

    def _active(self): return self.tabs.currentWidget()
    def _on_tab(self, idx): self._sb_mod.setText(MODULES[idx][0].split("  ")[1])

    def _export_png(self):
        mod = self._active()
        for attr in ('mpl', 'mpl_theta', 'mpl_analysis'):
            if hasattr(mod, attr):
                ExportManager.export_png(getattr(mod, attr), mod.MODULE_ID, self); return

    def _export_json(self): ExportManager.export_json(self._active(), self)

    @staticmethod
    def _detect_gpu():
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return f"GPU: {gpus[0].name.split('/')[-1]}" if gpus else "CPU Mode"
        except Exception: return "CPU Mode"

    def _about(self):
        QMessageBox.about(self, "About PhysicsAI 0304통합",
            "<h2>PhysicsAI 0304통합 v1.0</h2>"
            "<p>MOD-02 · MOD-04 · MOD-05 — Unified Physics AI Simulator</p>"
            "<p>SRS-PHYSAI-003 + SRS-PHYSAI-004</p>"
            "<hr><p><b>Stack:</b> PySide6 · TensorFlow/Keras · NumPy · Matplotlib · QPainter</p>")

    def closeEvent(self, event):
        for m in self._mods: m.cleanup()
        event.accept()
