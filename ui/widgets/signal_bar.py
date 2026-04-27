from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel
from PySide6.QtCore import Qt
from ui.styles.theme import COLORS

class SignalBar(QWidget):
    def __init__(self, buy: float = 0, hold: float = 0, sell: float = 0):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(12)
        
        self.buy_bar = self._create_bar("BUY", COLORS['bull'])
        self.hold_bar = self._create_bar("HOLD", COLORS['neutral'])
        self.sell_bar = self._create_bar("SELL", COLORS['bear'])
        
        self.layout.addLayout(self.buy_bar['layout'])
        self.layout.addLayout(self.hold_bar['layout'])
        self.layout.addLayout(self.sell_bar['layout'])
        
        self.update_signals(buy, hold, sell)

    def _create_bar(self, name: str, color: str):
        layout = QHBoxLayout()
        layout.setSpacing(12)
        
        name_label = QLabel(name)
        name_label.setFixedWidth(50)
        name_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        name_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 800; font-size: 10px; letter-spacing: 0.5px;")
        
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setTextVisible(False)
        progress.setFixedHeight(10)
        progress.setStyleSheet(f"""
            QProgressBar {{ background: #1A2235; border-radius: 5px; }}
            QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}
        """)
        
        pct_label = QLabel("0%")
        pct_label.setFixedWidth(45)
        pct_label.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 12px;")
        
        layout.addWidget(name_label)
        layout.addWidget(progress, 1)
        layout.addWidget(pct_label)
        
        return {'layout': layout, 'progress': progress, 'pct': pct_label}

    def update_signals(self, buy: float, hold: float, sell: float):
        total = buy + hold + sell
        if total > 0:
            buy_pct = (buy / total) * 100
            hold_pct = (hold / total) * 100
            sell_pct = (sell / total) * 100
        else:
            buy_pct = hold_pct = sell_pct = 0
            
        self.buy_bar['progress'].setValue(int(buy_pct))
        self.buy_bar['pct'].setText(f"{buy_pct:.1f}%")
        
        self.hold_bar['progress'].setValue(int(hold_pct))
        self.hold_bar['pct'].setText(f"{hold_pct:.1f}%")
        
        self.sell_bar['progress'].setValue(int(sell_pct))
        self.sell_bar['pct'].setText(f"{sell_pct:.1f}%")
