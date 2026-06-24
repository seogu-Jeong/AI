from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsOpacityEffect
from PyQt6.QtGui import QFont, QColor, QPixmap, QPainter, QRadialGradient
from PyQt6.QtCore import Qt, QPropertyAnimation, QPoint, QEasingCurve, QTimer, QSequentialAnimationGroup, QParallelAnimationGroup, QSize

class EvolutionOverlay(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(parent.size())
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)
        self.hide()

    def start_animation(self, old_name, new_name, on_finished):
        self.show()
        self.raise_()
        
        # Overlay Layout
        layout = QVBoxLayout(self)
        self.bg_color = QColor(255, 255, 255, 0)
        
        # Text Labels
        self.evo_label = QLabel("WHAT? AGUMON IS EVOLVING!", self)
        self.evo_label.setFont(QFont("Impact", 28, QFont.Weight.Bold))
        self.evo_label.setStyleSheet("color: white;")
        self.evo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.sprite_label = QLabel(self)
        self.sprite_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addStretch()
        layout.addWidget(self.evo_label)
        layout.addWidget(self.sprite_label)
        layout.addStretch()

        # Animation Sequence
        # 1. Fade in white background
        fade_in = QPropertyAnimation(self.opacity_effect, b"opacity")
        fade_in.setDuration(1000)
        fade_in.setStartValue(0)
        fade_in.setEndValue(1)
        
        # 2. Flash effect (pulsing)
        # 3. Text change
        
        self.seq = QSequentialAnimationGroup()
        self.seq.addAnimation(fade_in)
        
        # Dummy delay for effect
        self.seq.addPause(2000)
        
        self.seq.finished.connect(lambda: self.finish_evo(new_name, on_finished))
        self.seq.start()

    def finish_evo(self, new_name, on_finished):
        self.evo_label.setText(f"CONGRATULATIONS!\nHE EVOLVED INTO {new_name.upper()}!")
        QTimer.singleShot(2000, lambda: self.close_and_callback(on_finished))

    def close_and_callback(self, callback):
        self.hide()
        callback()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 200)) # Darken screen
