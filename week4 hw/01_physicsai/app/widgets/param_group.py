"""
ParamGroup — titled QGroupBox wrapping parameter widgets.
Provides factory methods and a unified get/reset interface.
"""
from typing import Dict, Any, List
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QCheckBox, QWidget,
)
from .slider_spinbox import SliderSpinBox


class ParamGroup(QGroupBox):
    any_value_changed = Signal(str, object)  # (param_name, new_value)

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self._layout  = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 16, 12, 12)
        self._layout.setSpacing(2)
        self._widgets: Dict[str, Any] = {}

    def add_slider(self, name: str, **kwargs) -> SliderSpinBox:
        w = SliderSpinBox(**kwargs)
        w.value_changed.connect(lambda v, n=name: self.any_value_changed.emit(n, v))
        self._widgets[name] = w
        self._layout.addWidget(w)
        return w

    def add_combo(self, name: str, label: str, options: List[str],
                  default_idx: int = 0, tooltip: str = "") -> QComboBox:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 2, 0, 6)
        lbl = QLabel(label)
        cb  = QComboBox()
        cb.addItems(options)
        cb.setCurrentIndex(default_idx)
        if tooltip:
            cb.setToolTip(tooltip)
        cb.currentTextChanged.connect(lambda t, n=name: self.any_value_changed.emit(n, t))
        row.addWidget(lbl)
        row.addWidget(cb, 1)
        self._widgets[name] = cb
        self._layout.addWidget(container)
        return cb

    def add_checkbox(self, name: str, label: str,
                     default: bool = False, tooltip: str = "") -> QCheckBox:
        ck = QCheckBox(label)
        ck.setChecked(default)
        if tooltip:
            ck.setToolTip(tooltip)
        ck.toggled.connect(lambda v, n=name: self.any_value_changed.emit(n, v))
        self._widgets[name] = ck
        self._layout.addWidget(ck)
        return ck

    def add_spacer(self):
        self._layout.addSpacing(6)

    def values(self) -> Dict[str, Any]:
        result = {}
        for name, w in self._widgets.items():
            if isinstance(w, SliderSpinBox):
                result[name] = w.value
            elif isinstance(w, QComboBox):
                result[name] = w.currentText()
            elif isinstance(w, QCheckBox):
                result[name] = w.isChecked()
        return result

    def reset(self):
        for w in self._widgets.values():
            if isinstance(w, SliderSpinBox):
                w.reset()

    def set_all_enabled(self, enabled: bool):
        for w in self._widgets.values():
            if isinstance(w, SliderSpinBox):
                w.set_enabled(enabled)
            else:
                w.setEnabled(enabled)
