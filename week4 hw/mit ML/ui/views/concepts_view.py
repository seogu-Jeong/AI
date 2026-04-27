from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget, QStackedWidget, QVBoxLayout

from ui.views.concepts.rl_basics_view import RLBasicsView
from ui.views.concepts.value_based_view import ValueBasedView
from ui.views.concepts.policy_based_view import PolicyBasedView
from ui.views.concepts.applications_view import ApplicationsView


class ConceptsView(QWidget):
    """Routes sub-pages: rl-basics, value-based, policy-based, applications."""

    PAGE_MAP = {
        "concepts/rl-basics":    0,
        "concepts/value-based":  1,
        "concepts/policy-based": 2,
        "concepts/applications": 3,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        self._rl_basics = RLBasicsView()
        self._value     = ValueBasedView()
        self._policy    = PolicyBasedView()
        self._apps      = ApplicationsView()

        self._stack.addWidget(self._rl_basics)
        self._stack.addWidget(self._value)
        self._stack.addWidget(self._policy)
        self._stack.addWidget(self._apps)
        lay.addWidget(self._stack)

    def show_page(self, page_id: str):
        self._stack.setCurrentIndex(self.PAGE_MAP.get(page_id, 0))

    def set_phase(self, phase: float):
        self._rl_basics.set_phase(phase)
        self._value.set_phase(phase)
        self._policy.set_phase(phase)

    def update_q(self, q: np.ndarray):
        self._value.update_q(q)

    def update_policy_probs(self, probs: np.ndarray):
        self._policy.update_probs(probs)

    def add_reinforce_reward(self, r: float):
        self._policy.add_reward(r)
