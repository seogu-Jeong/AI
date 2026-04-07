from __future__ import annotations
import math
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QColor, QPalette


# ── Color tokens ─────────────────────────────────────────────────────────────
BG       = QColor("#0A0E1A")
SURFACE1 = QColor("#111827")
SURFACE2 = QColor("#1F2937")
CYAN     = QColor("#00D4FF")
MAGENTA  = QColor("#FF006E")
VIOLET   = QColor("#7C3AED")
EMERALD  = QColor("#10B981")
AMBER    = QColor("#F59E0B")
WHITE_60 = QColor(255, 255, 255, 153)
WHITE_40 = QColor(255, 255, 255, 102)


# ── Helper functions ──────────────────────────────────────────────────────────
def lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    """Linear interpolate between two QColors (t in 0..1)."""
    t = max(0.0, min(1.0, t))
    return QColor(
        int(c1.red()   + t * (c2.red()   - c1.red())),
        int(c1.green() + t * (c2.green() - c1.green())),
        int(c1.blue()  + t * (c2.blue()  - c1.blue())),
        int(c1.alpha() + t * (c2.alpha() - c1.alpha())),
    )


def ease_in_out(t: float) -> float:
    """Smooth cubic ease-in-out (t in 0..1)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def pulse(phase: float, lo: float = 0.6, hi: float = 1.0) -> float:
    """Returns a value oscillating between lo and hi based on phase (0..1)."""
    return lo + (hi - lo) * (0.5 + 0.5 * math.sin(phase * 2 * math.pi))


# ── QSS stylesheet ────────────────────────────────────────────────────────────
_QSS = """
QWidget {
    background-color: #0A0E1A;
    color: rgba(255, 255, 255, 0.85);
    font-family: 'SF Pro Display', 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}
QScrollArea, QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    border: none;
}
QScrollBar::handle:vertical {
    background: rgba(255,255,255,0.15);
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QPushButton {
    background: rgba(0, 212, 255, 0.12);
    color: #00D4FF;
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background: rgba(0, 212, 255, 0.22);
    border-color: #00D4FF;
}
QPushButton:pressed {
    background: rgba(0, 212, 255, 0.08);
}
QSlider::groove:horizontal {
    height: 4px;
    background: rgba(255,255,255,0.12);
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #00D4FF;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: rgba(0, 212, 255, 0.5);
    border-radius: 2px;
}
QTabWidget::pane {
    border: none;
    background: transparent;
}
QLabel {
    background: transparent;
}
"""


def apply_theme(widget: QWidget) -> None:
    """Apply dark Glassmorphism stylesheet to widget."""
    widget.setStyleSheet(_QSS)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, BG)
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255, 220))
    widget.setPalette(palette)
