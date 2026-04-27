from PySide6.QtWidgets import QPushButton, QMessageBox
from PySide6.QtCore import Qt
from ui.styles.tooltips import TOOLTIPS

class HelpButton(QPushButton):
    def __init__(self, tooltip_key, parent=None):
        super().__init__("?", parent)
        self.tooltip_key = tooltip_key
        self.tooltip_text = TOOLTIPS.get(tooltip_key, "")
        
        self.setFixedSize(18, 18)
        if self.tooltip_text:
            # Set the first line as the hover tooltip
            first_line = self.tooltip_text.split('\n')[0]
            self.setToolTip(first_line)
            
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #1976D2;
                font-weight: bold;
                border: 1px solid #1976D2;
                border-radius: 9px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(25, 118, 210, 0.1);
            }
        """)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._show_help)

    def _show_help(self):
        if self.tooltip_text:
            QMessageBox.information(self, self.tooltip_key, self.tooltip_text)
