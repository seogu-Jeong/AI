"""
SliderSpinBox — composite widget: QSlider + QDoubleSpinBox synchronized.
Two-way sync via blockSignals() to prevent feedback loops.
"""
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDoubleSpinBox, QSizePolicy,
)
from PySide6.QtGui import QFont


class SliderSpinBox(QWidget):
    """
    Composite: QSlider (integer) + QDoubleSpinBox (float) kept in sync.
    Internally maps float ↔ integer via step resolution.
    """
    value_changed = Signal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, step: float = 1.0, unit: str = "",
                 decimals: int = 2, tooltip: str = "", parent=None):
        super().__init__(parent)
        self._min     = min_val
        self._max     = max_val
        self._step    = step
        self._default = default
        self._blocking = False

        # ── label ────────────────────────────────────────────────────
        lbl_text = f"{label}"
        if unit:
            lbl_text += f"  [{unit}]"
        self._label = QLabel(lbl_text)
        font = QFont()
        font.setWeight(QFont.Weight.DemiBold)
        self._label.setFont(font)

        # ── slider ───────────────────────────────────────────────────
        n_steps = max(1, round((max_val - min_val) / step))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, n_steps)
        self._slider.setValue(self._float_to_int(default))
        self._slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if tooltip:
            self._slider.setToolTip(tooltip)

        # ── spinbox ──────────────────────────────────────────────────
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(decimals)
        self._spin.setRange(min_val, max_val)
        self._spin.setSingleStep(step)
        self._spin.setValue(default)
        self._spin.setFixedWidth(88)
        if tooltip:
            self._spin.setToolTip(tooltip)

        # ── layout ───────────────────────────────────────────────────
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._slider)
        row.addWidget(self._spin)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 2, 0, 6)
        main.setSpacing(3)
        main.addWidget(self._label)
        main.addLayout(row)

        # ── signals ──────────────────────────────────────────────────
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    # ── internal helpers ─────────────────────────────────────────────
    def _float_to_int(self, v: float) -> int:
        return round((v - self._min) / self._step)

    def _int_to_float(self, i: int) -> float:
        return self._min + i * self._step

    def _on_slider(self, i: int):
        if self._blocking:
            return
        self._blocking = True
        fv = self._int_to_float(i)
        self._spin.setValue(fv)
        self._blocking = False
        self.value_changed.emit(fv)

    def _on_spin(self, fv: float):
        if self._blocking:
            return
        self._blocking = True
        self._slider.setValue(self._float_to_int(fv))
        self._blocking = False
        self.value_changed.emit(fv)

    # ── public API ───────────────────────────────────────────────────
    @property
    def value(self) -> float:
        return self._spin.value()

    @value.setter
    def value(self, v: float):
        self._spin.setValue(v)

    def reset(self):
        self.value = self._default

    def set_enabled(self, enabled: bool):
        self._slider.setEnabled(enabled)
        self._spin.setEnabled(enabled)
