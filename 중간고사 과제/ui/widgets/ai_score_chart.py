from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from PySide6.QtWidgets import QSizePolicy
from ui.styles.theme import COLORS

class AIScoreChart(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(facecolor='#c0c0c0', tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        self._show_message("Select Ticker to Load History")

    def _show_message(self, msg):
        self.ax.clear()
        self.ax.text(0.5, 0.5, msg, ha='center', va='center', color='#808080', fontsize=10)
        self.ax.set_axis_off()
        self.draw()

    def plot(self, history, ticker):
        if not history or len(history) < 2:
            self._show_message("No history data" if history else "No history data available")
            return

        self.ax.clear()
        self.ax.set_axis_on()
        self.ax.set_facecolor('#ffffff')
        
        dates = [item['date'] for item in history]
        ensemble = [item['ensemble'] for item in history]
        lstm = [item['lstm'] for item in history]
        cnn = [item['cnn'] for item in history]
        transformer = [item['transformer'] for item in history]
        mlp = [item['mlp'] for item in history]

        # Use colors from theme
        self.ax.plot(dates, lstm, color=COLORS['bull'], alpha=0.5, linewidth=1.0, label='LSTM')
        self.ax.plot(dates, cnn, color=COLORS['warning'], alpha=0.5, linewidth=1.0, label='CNN')
        self.ax.plot(dates, transformer, color='#9C27B0', alpha=0.5, linewidth=1.0, label='Transformer')
        self.ax.plot(dates, mlp, color=COLORS['bear'], alpha=0.5, linewidth=1.0, label='MLP')
        
        ensemble_color = COLORS.get('primary', COLORS['accent'])
        self.ax.plot(dates, ensemble, color=ensemble_color, linewidth=2.5, label='Ensemble', marker='o', markersize=4)

        # Horizontal reference line at 60
        self.ax.axhline(y=60, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='매수 고려 기준선')

        self.ax.set_ylim(0, 105)
        self.ax.set_title(f'AI 점수 추이 ({ticker})', fontsize=11, fontweight='bold', color=COLORS['accent'])
        
        # Legend
        self.ax.legend(loc='lower left', fontsize=8, frameon=True, facecolor='#ffffff', framealpha=0.8)
        
        # Formatting
        self.ax.tick_params(axis='both', which='major', labelsize=8)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        
        # X-axis date formatting
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        # Use step=3 for tick labels as requested (though it might depend on the number of points)
        # We can use a MonthLocator
        self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        self.figure.autofmt_xdate(rotation=30)

        for spine in self.ax.spines.values():
            spine.set_edgecolor('#808080')

        self.draw()
