from __future__ import annotations
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QLabel)
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient

from ui.theme import CYAN, SURFACE1, WHITE_60, WHITE_40, pulse


# Navigation tree: (group_label, group_page_id_or_None, [(child_label, child_page_id)])
NAV_ITEMS = [
    ("Concepts", None, [
        ("RL Basics",    "concepts/rl-basics"),
        ("Value-Based",  "concepts/value-based"),
        ("Policy-Based", "concepts/policy-based"),
        ("Applications", "concepts/applications"),
    ]),
    ("Playground", None, [
        ("GridWorld", "playground"),
        ("CartPole",  "playground"),
        ("Maze",      "playground"),
    ]),
    ("Arena", "arena", []),
]


class SidebarItem(QWidget):
    clicked = Signal(str)

    def __init__(self, label: str, page_id: str, indent: int = 0, parent=None):
        super().__init__(parent)
        self.label = label
        self.page_id = page_id
        self.indent = indent
        self.setFixedHeight(36)
        self._active = False
        self._hover = False
        self._phase = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_active(self, active: bool):
        self._active = active; self.update()

    def set_phase(self, phase: float):
        self._phase = phase; self.update()

    def enterEvent(self, e): self._hover = True; self.update()
    def leaveEvent(self, e): self._hover = False; self.update()
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.clicked.emit(self.page_id)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        if self._active:
            g = QLinearGradient(0, 0, W, 0)
            g.setColorAt(0, QColor(0, 212, 255, 30))
            g.setColorAt(1, QColor(0, 212, 255, 5))
            p.fillRect(self.rect(), g)
            br = pulse(self._phase, 0.7, 1.0)
            p.fillRect(0, 4, 3, H - 8, QColor(0, int(212 * br), int(255 * br)))
        elif self._hover:
            p.fillRect(self.rect(), QColor(255, 255, 255, 8))

        color = CYAN if self._active else (WHITE_60 if self._hover else WHITE_40)
        f = QFont(); f.setPointSize(10)
        if self._active: f.setBold(True)
        p.setFont(f); p.setPen(color)
        p.drawText(QRectF(12 + self.indent * 16, 0, W - 24, H),
                   Qt.AlignmentFlag.AlignVCenter, self.label)
        p.end()


class Sidebar(QWidget):
    page_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self._items: list[SidebarItem] = []
        self._active_id = "concepts/rl-basics"
        self._phase = 0.0
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        logo = QLabel("RL Dashboard")
        logo.setFixedHeight(60)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("color: #00D4FF; font-size: 13pt; font-weight: 900; "
                           "border-bottom: 1px solid rgba(255,255,255,0.08);")
        lay.addWidget(logo)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 8, 0, 8); inner_lay.setSpacing(0)

        for group_label, group_id, children in NAV_ITEMS:
            grp = QLabel(f"  {group_label}")
            grp.setFixedHeight(32)
            grp.setStyleSheet("color: rgba(255,255,255,0.85); font-weight: 700; "
                              "font-size: 9pt; border-bottom: 1px solid rgba(255,255,255,0.05);"
                              " background: transparent;")
            inner_lay.addWidget(grp)

            if group_id:
                item = SidebarItem(group_label, group_id, indent=0)
                item.clicked.connect(self._on_click)
                self._items.append(item)
                inner_lay.addWidget(item)
            else:
                for child_label, child_id in children:
                    item = SidebarItem(child_label, child_id, indent=1)
                    item.clicked.connect(self._on_click)
                    self._items.append(item)
                    inner_lay.addWidget(item)

        inner_lay.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)
        self._refresh_active()

    def _on_click(self, page_id: str):
        self._active_id = page_id; self._refresh_active()
        self.page_requested.emit(page_id)

    def _refresh_active(self):
        for item in self._items:
            item.set_active(item.page_id == self._active_id)

    def set_phase(self, phase: float):
        self._phase = phase
        for item in self._items:
            if item._active: item.set_phase(phase)

    def set_active(self, page_id: str):
        self._active_id = page_id; self._refresh_active()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), SURFACE1)
        p.setPen(QPen(QColor(255, 255, 255, 15), 1))
        p.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        p.end()
