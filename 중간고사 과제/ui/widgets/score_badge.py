from PySide6.QtWidgets import QWidget, QHBoxLayout, QProgressBar, QLabel
from PySide6.QtCore import Qt
from ui.styles.theme import get_score_color, COLORS

class ScoreBadge(QWidget):
    def __init__(self, score: float = 0, show_label: bool = True):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedHeight(12)
        self.progress.setTextVisible(False)
        
        self.label = QLabel()
        self.label.setVisible(show_label)
        self.label.setFixedWidth(70)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.label.setStyleSheet(f"font-weight: 800; font-size: 11px; letter-spacing: 0.5px;")
        
        self.layout.addWidget(self.progress, 1)
        self.layout.addWidget(self.label)
        
        self.update_score(score)

    def set_score(self, score: float):
        self.update_score(score)

    def update_score(self, score: float):
        score_int = int(max(0, min(100, score)))
        self.progress.setValue(score_int)
        
        color = get_score_color(score)
        self.progress.setStyleSheet(f"""
            QProgressBar {{ background: #1A2235; border-radius: 6px; }}
            QProgressBar::chunk {{ background: {color}; border-radius: 6px; }}
        """)
        
        if score >= 80:
            text = 'STRONG BUY'
            self.label.setStyleSheet(f"color: {COLORS['bull']}; font-weight: 800; font-size: 11px;")
        elif score >= 60:
            text = 'BUY'
            self.label.setStyleSheet(f"color: {COLORS['bull']}; font-weight: 800; font-size: 11px;")
        elif score >= 40:
            text = 'NEUTRAL'
            self.label.setStyleSheet(f"color: {COLORS['neutral']}; font-weight: 800; font-size: 11px;")
        elif score >= 20:
            text = 'SELL'
            self.label.setStyleSheet(f"color: {COLORS['bear']}; font-weight: 800; font-size: 11px;")
        else:
            text = 'STRONG SELL'
            self.label.setStyleSheet(f"color: {COLORS['bear']}; font-weight: 800; font-size: 11px;")
            
        self.label.setText(text)
