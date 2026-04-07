"""MatplotlibWidget — FigureCanvasQTAgg embedded in QWidget."""
import warnings
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT


class MatplotlibWidget(QWidget):
    def __init__(self, figsize=(12, 5), dpi=100, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self.toolbar); lay.addWidget(self.canvas)

    def fresh_axes(self, nrows=1, ncols=1, **kw):
        self.figure.clear()
        if nrows == 1 and ncols == 1: return self.figure.add_subplot(111, **kw)
        return self.figure.subplots(nrows, ncols, **kw)

    def fresh_gridspec(self, nrows, ncols, **kw):
        self.figure.clear()
        return self.figure.add_gridspec(nrows, ncols, **kw)

    def draw(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try: self.figure.tight_layout(pad=1.5)
            except Exception: pass
        self.canvas.draw()

    def draw_idle(self): self.canvas.draw_idle()
    def clear(self): self.figure.clear(); self.canvas.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event); self.canvas.draw_idle()
