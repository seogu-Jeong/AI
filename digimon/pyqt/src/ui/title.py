from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt

class TitleScreen(QWidget):
    def __init__(self, on_start):
        super().__init__()
        self.on_start = on_start
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.setStyleSheet("background-color: #000000; color: #8bac0f;")
        
        title_label = QLabel("DIGIMON RPG\nPyQt6 Edition", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Verdana", 36, QFont.Weight.Bold))
        
        start_btn = QPushButton("PRESS START", self)
        start_btn.setFont(QFont("Verdana", 18))
        start_btn.setFixedSize(200, 60)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #306230;
                color: #ffffff;
                border: 2px solid #8bac0f;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8bac0f;
                color: #000000;
            }
        """)
        start_btn.clicked.connect(self.on_start)
        
        layout.addStretch()
        layout.addWidget(title_label)
        layout.addSpacing(50)
        layout.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
