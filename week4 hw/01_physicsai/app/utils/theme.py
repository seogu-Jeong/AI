"""
ThemeManager — QSS stylesheets and Matplotlib style synchronization.
Toggled via Ctrl+D (TRD-01 §2.3, SRS §5.3 NFR-11).
"""
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication

# ─── Shared base styles ──────────────────────────────────────────────────────
_COMMON = """
QWidget {
    font-size: 11pt;
}
QGroupBox {
    font-weight: 600;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QProgressBar {
    border-radius: 5px;
    text-align: center;
    height: 20px;
    font-weight: bold;
    font-size: 9pt;
}
QProgressBar::chunk { border-radius: 4px; }
QPushButton {
    border-radius: 5px;
    padding: 7px 18px;
    font-weight: 700;
    min-height: 34px;
    font-size: 11pt;
}
QPushButton#run_btn  { background: #43A047; color: #fff; border: none; }
QPushButton#run_btn:hover  { background: #388E3C; }
QPushButton#run_btn[dirty="true"] { background: #FB8C00; }
QPushButton#stop_btn { background: #E53935; color: #fff; border: none; }
QPushButton#stop_btn:hover { background: #C62828; }
QPushButton#reset_btn { background: #546E7A; color: #fff; border: none; }
QPushButton#reset_btn:hover { background: #37474F; }
QPushButton:disabled { background: #9E9E9E; color: #E0E0E0; border: none; }
QTabBar::tab { padding: 9px 22px; font-size: 10pt; }
QTabBar::tab:selected { font-weight: 700; }
QSlider::groove:horizontal {
    border-radius: 3px;
    height: 6px;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QComboBox {
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 28px;
}
QSpinBox, QDoubleSpinBox {
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 26px;
}
QToolBar {
    spacing: 4px;
    padding: 4px;
}
QStatusBar {
    font-size: 10pt;
}
"""

_LIGHT = _COMMON + """
QMainWindow, QWidget { background: #F5F5F5; color: #212121; }
QGroupBox { border: 1px solid #DEDEDE; background: #FFFFFF; }
QTabWidget::pane { border: 1px solid #BDBDBD; background: #FFFFFF; }
QProgressBar { background: #E0E0E0; color: #212121; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #1976D2, stop:1 #42A5F5); }
QSlider::groove:horizontal { background: #BDBDBD; }
QSlider::sub-page:horizontal { background: #1976D2; border-radius: 3px; }
QSlider::handle:horizontal { background: #1976D2; border: 2px solid #fff; }
QComboBox { border: 1px solid #BDBDBD; background: #FFFFFF; }
QComboBox::drop-down { border: none; }
QSpinBox, QDoubleSpinBox { border: 1px solid #BDBDBD; background: #FFFFFF; }
QToolBar { background: #ECEFF1; border-bottom: 1px solid #CFD8DC; }
QStatusBar { background: #ECEFF1; border-top: 1px solid #CFD8DC; }
QMenuBar { background: #ECEFF1; }
QMenu { background: #FFFFFF; border: 1px solid #BDBDBD; }
QSplitter::handle { background: #E0E0E0; }
QScrollBar:vertical { background: #F5F5F5; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #BDBDBD; border-radius: 5px; min-height: 30px; }
"""

_DARK = _COMMON + """
QMainWindow, QWidget { background: #1E1E2E; color: #CDD6F4; }
QGroupBox { border: 1px solid #313244; background: #181825; }
QTabWidget::pane { border: 1px solid #313244; background: #181825; }
QProgressBar { background: #313244; color: #CDD6F4; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #89B4FA, stop:1 #74C7EC); }
QSlider::groove:horizontal { background: #313244; }
QSlider::sub-page:horizontal { background: #89B4FA; border-radius: 3px; }
QSlider::handle:horizontal { background: #89B4FA; border: 2px solid #1E1E2E; }
QComboBox { border: 1px solid #45475A; background: #313244; color: #CDD6F4; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #313244; color: #CDD6F4; border: 1px solid #45475A; }
QSpinBox, QDoubleSpinBox { border: 1px solid #45475A; background: #313244; color: #CDD6F4; }
QPushButton#run_btn  { background: #A6E3A1; color: #1E1E2E; }
QPushButton#run_btn:hover { background: #94E2D5; }
QPushButton#run_btn[dirty="true"] { background: #FAB387; color: #1E1E2E; }
QPushButton#stop_btn { background: #F38BA8; color: #1E1E2E; }
QPushButton#stop_btn:hover { background: #EBA0AC; }
QPushButton#reset_btn { background: #585B70; color: #CDD6F4; }
QPushButton#reset_btn:hover { background: #6C7086; }
QToolBar { background: #181825; border-bottom: 1px solid #313244; }
QStatusBar { background: #181825; border-top: 1px solid #313244; }
QMenuBar { background: #181825; color: #CDD6F4; }
QMenu { background: #313244; border: 1px solid #45475A; color: #CDD6F4; }
QSplitter::handle { background: #313244; }
QScrollBar:vertical { background: #1E1E2E; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #45475A; border-radius: 5px; min-height: 30px; }
QTabBar::tab { background: #181825; color: #A6ADC8; }
QTabBar::tab:selected { background: #313244; color: #CDD6F4; }
"""


class ThemeManager:
    _current: str = "light"

    @classmethod
    def apply_light(cls, app: QApplication):
        app.setStyleSheet(_LIGHT)
        cls._current = "light"
        plt.style.use('default')

    @classmethod
    def apply_dark(cls, app: QApplication):
        app.setStyleSheet(_DARK)
        cls._current = "dark"
        plt.style.use('dark_background')

    @classmethod
    def toggle(cls, app: QApplication):
        if cls._current == "light":
            cls.apply_dark(app)
        else:
            cls.apply_light(app)
        cls._sync_all_figures()

    @classmethod
    def is_dark(cls) -> bool:
        return cls._current == "dark"

    @classmethod
    def _sync_all_figures(cls):
        from app.widgets.matplotlib_widget import MatplotlibWidget
        dark = cls._current == "dark"
        bg   = '#181825' if dark else '#FFFFFF'
        fg   = '#CDD6F4' if dark else '#212121'
        for widget in QApplication.instance().allWidgets():
            if isinstance(widget, MatplotlibWidget):
                widget.figure.set_facecolor(bg)
                for ax in widget.figure.get_axes():
                    ax.set_facecolor(bg)
                    ax.tick_params(colors=fg)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#45475A' if dark else '#BDBDBD')
                    ax.xaxis.label.set_color(fg)
                    ax.yaxis.label.set_color(fg)
                    ax.title.set_color(fg)
                    if ax.legend_ is not None:
                        ax.legend_.get_frame().set_facecolor(bg)
                        for text in ax.legend_.get_texts():
                            text.set_color(fg)
                widget.draw_idle()
