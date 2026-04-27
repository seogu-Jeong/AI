"""
MatplotlibWidget — FigureCanvasQTAgg embedded in a QWidget.
Thread-safe: draw_idle() is the only draw method safe during training.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)


class MatplotlibWidget(QWidget):
    """
    Embeds a Matplotlib Figure via FigureCanvasQTAgg.

    Usage:
        w = MatplotlibWidget(figsize=(12, 5))
        ax = w.fresh_axes()
        ax.plot(x, y)
        w.draw()
    """

    def __init__(self, figsize=(12, 5), dpi=100, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi, tight_layout=False)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def fresh_axes(self, nrows=1, ncols=1, **kw):
        self.figure.clear()
        if nrows == 1 and ncols == 1:
            return self.figure.add_subplot(111, **kw)
        return self.figure.subplots(nrows, ncols, **kw)

    def fresh_gridspec(self, nrows, ncols, **kw):
        self.figure.clear()
        return self.figure.add_gridspec(nrows, ncols, **kw)

    def draw(self):
        """Full synchronous redraw — main thread only."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.figure.tight_layout(pad=1.5)
            except Exception:
                pass
        self.canvas.draw()

    def draw_idle(self):
        """Non-blocking redraw hint — safe during training."""
        self.canvas.draw_idle()

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def export_png(self, path: str, dpi: int = 150):
        self.figure.savefig(path, dpi=dpi, bbox_inches='tight')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.canvas.draw_idle()
