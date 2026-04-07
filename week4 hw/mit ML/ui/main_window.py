from __future__ import annotations
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget
from PySide6.QtCore import Qt

from ui.theme import apply_theme, BG
from ui.components.sidebar import Sidebar
from ui.views.concepts_view import ConceptsView
from ui.views.playground_view import PlaygroundView
from ui.views.arena_view import ArenaView
from controllers.concept_controller import ConceptController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL AI Dashboard")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)
        apply_theme(self)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        self._sidebar = Sidebar()
        self._sidebar.page_requested.connect(self._navigate)
        root.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        self._concepts_view    = ConceptsView()
        self._playground_view  = PlaygroundView()
        self._arena_view       = ArenaView()

        self._stack.addWidget(self._concepts_view)    # 0
        self._stack.addWidget(self._playground_view)  # 1
        self._stack.addWidget(self._arena_view)       # 2

        self._concept_ctrl = ConceptController(self)
        self._concept_ctrl.phase_updated.connect(self._on_phase)
        self._concept_ctrl.start(speed=0.008)

        self._navigate("concepts/rl-basics")

    def _navigate(self, page_id: str):
        self._sidebar.set_active(page_id)
        if page_id.startswith("concepts/"):
            self._stack.setCurrentIndex(0)
            self._concepts_view.show_page(page_id)
        elif page_id == "playground":
            self._stack.setCurrentIndex(1)
        elif page_id == "arena":
            self._stack.setCurrentIndex(2)

    def _on_phase(self, phase: float):
        self._sidebar.set_phase(phase)
        self._concepts_view.set_phase(phase)
        self._playground_view.set_phase(phase)
