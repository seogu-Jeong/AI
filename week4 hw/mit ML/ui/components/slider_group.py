from __future__ import annotations
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QSlider)
from PySide6.QtCore import Qt, Signal


class SliderGroup(QWidget):
    """A vertical stack of labeled sliders."""

    value_changed = Signal(int, int)   # (slider_index, new_value)

    def __init__(self, specs: list[tuple[str, int, int, int]], parent=None):
        """specs: [(label, min, max, default), ...]"""
        super().__init__(parent)
        self._sliders: list[QSlider] = []
        self._value_labels: list[QLabel] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        for i, (label, lo, hi, default) in enumerate(specs):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(90)
            lbl.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 9pt;")

            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(default)
            sl.setFixedHeight(20)

            val_lbl = QLabel(str(default))
            val_lbl.setFixedWidth(36)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet("color: #00D4FF; font-size: 9pt; font-weight: 600;")

            idx = i

            def _on_change(v, _i=idx, _vl=val_lbl):
                _vl.setText(str(v))
                self.value_changed.emit(_i, v)

            sl.valueChanged.connect(_on_change)
            self._sliders.append(sl)
            self._value_labels.append(val_lbl)

            row.addWidget(lbl)
            row.addWidget(sl)
            row.addWidget(val_lbl)
            lay.addLayout(row)

    def values(self) -> list[int]:
        return [sl.value() for sl in self._sliders]

    def set_value(self, index: int, value: int) -> None:
        self._sliders[index].setValue(value)
