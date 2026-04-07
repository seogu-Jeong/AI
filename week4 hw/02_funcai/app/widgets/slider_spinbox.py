"""
SliderSpinBox — QSlider + QDoubleSpinBox, two-way sync via blockSignals().
"""
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QSlider, QDoubleSpinBox, QSizePolicy)
from PySide6.QtGui import QFont


class SliderSpinBox(QWidget):
    value_changed = Signal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, step: float = 1.0, unit: str = "",
                 decimals: int = 2, tooltip: str = "", parent=None):
        super().__init__(parent)
        self._min      = min_val
        self._max      = max_val
        self._step     = step
        self._default  = default
        self._blocking = False

        lbl_text = f"{label}" + (f"  [{unit}]" if unit else "")
        lbl = QLabel(lbl_text)
        f = QFont(); f.setWeight(QFont.Weight.DemiBold)
        lbl.setFont(f)

        n_steps = max(1, round((max_val - min_val) / step))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, n_steps)
        self._slider.setValue(self._float_to_int(default))
        self._slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if tooltip: self._slider.setToolTip(tooltip)

        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(decimals)
        self._spin.setRange(min_val, max_val)
        self._spin.setSingleStep(step)
        self._spin.setValue(default)
        self._spin.setFixedWidth(88)
        if tooltip: self._spin.setToolTip(tooltip)

        row = QHBoxLayout(); row.setContentsMargins(0,0,0,0)
        row.addWidget(self._slider); row.addWidget(self._spin)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 2, 0, 6); main.setSpacing(3)
        main.addWidget(lbl); main.addLayout(row)

        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def _float_to_int(self, v): return round((v - self._min) / self._step)
    def _int_to_float(self, i): return self._min + i * self._step

    def _on_slider(self, i):
        if self._blocking: return
        self._blocking = True
        fv = self._int_to_float(i)
        self._spin.setValue(fv)
        self._blocking = False
        self.value_changed.emit(fv)

    def _on_spin(self, fv):
        if self._blocking: return
        self._blocking = True
        self._slider.setValue(self._float_to_int(fv))
        self._blocking = False
        self.value_changed.emit(fv)

    @property
    def value(self): return self._spin.value()
    @value.setter
    def value(self, v): self._spin.setValue(v)
    def reset(self): self.value = self._default
    def set_enabled(self, e): self._slider.setEnabled(e); self._spin.setEnabled(e)
