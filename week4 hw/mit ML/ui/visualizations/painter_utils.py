from __future__ import annotations
import math
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QPainter, QColor, QPen, QBrush,
                            QRadialGradient, QLinearGradient, QPolygonF)

from ui.theme import SURFACE1, SURFACE2


def draw_glow(p: QPainter, center: QPointF, radius: float, color: QColor) -> None:
    """Three-layer radial glow effect centered at `center`."""
    layers = [
        (radius * 3.0, 15),
        (radius * 1.8, 35),
        (radius * 0.9, 80),
    ]
    for r, alpha in layers:
        g = QRadialGradient(center, r)
        c_inner = QColor(color.red(), color.green(), color.blue(), alpha)
        c_outer = QColor(color.red(), color.green(), color.blue(), 0)
        g.setColorAt(0.0, c_inner)
        g.setColorAt(1.0, c_outer)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawEllipse(QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r))


def draw_arrow(p: QPainter, cx: float, cy: float, direction: int,
               size: float, color: QColor, opacity: float = 1.0) -> None:
    """Draw a directional arrow at (cx, cy).

    direction: 0=Up, 1=Down, 2=Left, 3=Right
    """
    alpha = int(opacity * 255)
    fill = QColor(color.red(), color.green(), color.blue(), alpha)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(fill)

    half = size * 0.5
    stem_w = size * 0.25

    if direction == 0:   # Up
        pts = [QPointF(cx, cy - size),
               QPointF(cx - half, cy - size * 0.3),
               QPointF(cx - stem_w, cy - size * 0.3),
               QPointF(cx - stem_w, cy + size * 0.2),
               QPointF(cx + stem_w, cy + size * 0.2),
               QPointF(cx + stem_w, cy - size * 0.3),
               QPointF(cx + half, cy - size * 0.3)]
    elif direction == 1:  # Down
        pts = [QPointF(cx, cy + size),
               QPointF(cx - half, cy + size * 0.3),
               QPointF(cx - stem_w, cy + size * 0.3),
               QPointF(cx - stem_w, cy - size * 0.2),
               QPointF(cx + stem_w, cy - size * 0.2),
               QPointF(cx + stem_w, cy + size * 0.3),
               QPointF(cx + half, cy + size * 0.3)]
    elif direction == 2:  # Left
        pts = [QPointF(cx - size, cy),
               QPointF(cx - size * 0.3, cy - half),
               QPointF(cx - size * 0.3, cy - stem_w),
               QPointF(cx + size * 0.2, cy - stem_w),
               QPointF(cx + size * 0.2, cy + stem_w),
               QPointF(cx - size * 0.3, cy + stem_w),
               QPointF(cx - size * 0.3, cy + half)]
    else:                 # Right
        pts = [QPointF(cx + size, cy),
               QPointF(cx + size * 0.3, cy - half),
               QPointF(cx + size * 0.3, cy - stem_w),
               QPointF(cx - size * 0.2, cy - stem_w),
               QPointF(cx - size * 0.2, cy + stem_w),
               QPointF(cx + size * 0.3, cy + stem_w),
               QPointF(cx + size * 0.3, cy + half)]

    p.drawPolygon(QPolygonF(pts))


def draw_glass_rect(p: QPainter, rect: QRectF, radius: float = 12.0) -> None:
    """Draw a glassmorphism panel: 8% white fill + border glow."""
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255, 20))
    p.drawRoundedRect(rect, radius, radius)

    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(255, 255, 255, 30), 1))
    p.drawRoundedRect(rect, radius, radius)


def heatmap_color(value: float, min_val: float, max_val: float) -> QColor:
    """Map a scalar to a cyan→magenta color gradient."""
    if max_val == min_val:
        t = 0.5
    else:
        t = (value - min_val) / (max_val - min_val)
    t = max(0.0, min(1.0, t))
    # Cyan (0,212,255) → Magenta (255,0,110)
    r = int(0 + t * 255)
    g = int(212 - t * 212)
    b = int(255 - t * 145)
    return QColor(r, g, b)
