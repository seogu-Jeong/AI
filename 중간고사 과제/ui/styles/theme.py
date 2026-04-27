COLORS = {
    'bg_base':       '#c0c0c0',
    'bg_surface':    '#c0c0c0',
    'bg_card':       '#c0c0c0',
    'bg_elevated':   '#ffffff',
    'bg_hover':      '#000080',
    'sidebar_bg':    '#c0c0c0',
    'sidebar_item':  '#c0c0c0',
    'sidebar_sel':   '#000080',
    'accent':        '#000080',
    'accent_light':  '#0000cd',
    'accent_glow':   '#000080',
    'accent2':       '#008080',
    'bull':          '#008000',
    'bull_bg':       '#ccffcc',
    'bear':          '#cc0000',
    'bear_bg':       '#ffcccc',
    'neutral':       '#808080',
    'warning':       '#808000',
    'text_primary':  '#000000',
    'text_secondary':'#444444',
    'text_muted':    '#808080',
    'border':        '#808080',
    'border_bright': '#ffffff',
    'score_80':      '#008000',
    'score_60':      '#808000',
    'score_40':      '#808080',
    'score_20':      '#cc4400',
    'score_0':       '#cc0000',
}

def get_score_color(score: float) -> str:
    if score >= 80: return COLORS['score_80']
    if score >= 60: return COLORS['score_60']
    if score >= 40: return COLORS['score_40']
    if score >= 20: return COLORS['score_20']
    return COLORS['score_0']

def get_score_bg(score: float) -> str:
    if score >= 60: return COLORS['bull_bg']
    if score >= 40: return COLORS['bg_card']
    return COLORS['bear_bg']

def get_stylesheet() -> str:
    return f"""
/* === GLOBAL === */
QMainWindow, QWidget {{
    background-color: {COLORS['bg_base']};
    color: {COLORS['text_primary']};
    font-family: 'Apple SD Gothic Neo', Arial, sans-serif;
    font-size: 13px;
}}

/* === SIDEBAR === */
QListWidget#sidebar {{
    background-color: {COLORS['sidebar_bg']};
    border: none;
    border-right: 2px solid {COLORS['border']};
    outline: none;
    padding: 4px 0;
}}
QListWidget#sidebar::item {{
    height: 44px;
    color: {COLORS['text_primary']};
    padding: 0 16px;
    font-size: 13px;
    font-weight: 400;
}}
QListWidget#sidebar::item:hover {{
    background-color: {COLORS['bg_hover']};
    color: #ffffff;
}}
QListWidget#sidebar::item:selected {{
    background-color: {COLORS['sidebar_sel']};
    color: #ffffff;
    font-weight: 700;
}}

/* === TABLE === */
QTableWidget {{
    background-color: {COLORS['bg_elevated']};
    alternate-background-color: #f0f0f0;
    gridline-color: #d0d0d0;
    border-top: 2px solid {COLORS['border']};
    border-left: 2px solid {COLORS['border']};
    border-right: 2px solid {COLORS['border_bright']};
    border-bottom: 2px solid {COLORS['border_bright']};
    font-size: 13px;
    selection-background-color: {COLORS['accent']};
    selection-color: #ffffff;
    font-family: 'Menlo', 'Courier New', 'Apple SD Gothic Neo', monospace;
}}
QTableWidget::item {{
    padding: 6px 10px;
    border-bottom: 1px solid #d0d0d0;
    color: {COLORS['text_primary']};
}}
QTableWidget::item:selected {{
    background-color: {COLORS['accent']};
    color: #ffffff;
}}
QTableWidget::item:hover {{
    background-color: #dde8ff;
}}
QHeaderView::section {{
    background-color: {COLORS['bg_base']};
    color: {COLORS['text_primary']};
    font-size: 11px;
    font-weight: 700;
    padding: 6px 10px;
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
}}

/* === BUTTONS === */
QPushButton {{
    background-color: {COLORS['bg_base']};
    color: {COLORS['text_primary']};
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
    padding: 4px 16px;
    font-size: 12px;
    font-weight: 400;
    min-height: 22px;
}}
QPushButton:hover {{
    background-color: #d4d0c8;
}}
QPushButton:pressed {{
    border-top: 2px solid {COLORS['border']};
    border-left: 2px solid {COLORS['border']};
    border-right: 2px solid {COLORS['border_bright']};
    border-bottom: 2px solid {COLORS['border_bright']};
    padding: 5px 15px 3px 17px;
}}
QPushButton#secondary {{
    background-color: {COLORS['bg_base']};
    color: {COLORS['text_primary']};
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
}}
QPushButton#secondary:hover {{
    background-color: #d4d0c8;
}}
QPushButton#ghost {{
    background-color: transparent;
    color: {COLORS['text_secondary']};
    border: none;
    padding: 4px 10px;
}}
QPushButton#ghost:hover {{
    color: {COLORS['text_primary']};
    background-color: #d4d0c8;
}}

/* === INPUTS === */
QLineEdit, QComboBox {{
    background-color: {COLORS['bg_elevated']};
    color: {COLORS['text_primary']};
    border-top: 2px solid {COLORS['border']};
    border-left: 2px solid {COLORS['border']};
    border-right: 2px solid {COLORS['border_bright']};
    border-bottom: 2px solid {COLORS['border_bright']};
    padding: 3px 8px;
    font-size: 13px;
    font-family: 'Apple SD Gothic Neo', 'Menlo', monospace;
}}
QLineEdit:focus, QComboBox:focus {{
    border-color: {COLORS['accent']};
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 4px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_elevated']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent']};
    selection-color: #ffffff;
}}

/* === PROGRESS BARS === */
QProgressBar {{
    background-color: #d4d0c8;
    border-top: 1px solid {COLORS['border']};
    border-left: 1px solid {COLORS['border']};
    border-right: 1px solid {COLORS['border_bright']};
    border-bottom: 1px solid {COLORS['border_bright']};
    height: 16px;
    text-align: center;
    color: {COLORS['text_primary']};
    font-size: 10px;
}}
QProgressBar::chunk {{
    background-color: {COLORS['accent']};
}}

/* === SCROLLBAR === */
QScrollBar:vertical {{
    background: {COLORS['bg_base']};
    width: 16px;
    border: 1px solid {COLORS['border']};
}}
QScrollBar::handle:vertical {{
    background-color: {COLORS['bg_base']};
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: #d4d0c8;
}}
QScrollBar::sub-line:vertical {{
    height: 16px;
    background-color: {COLORS['bg_base']};
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
    subcontrol-position: top;
    subcontrol-origin: margin;
}}
QScrollBar::add-line:vertical {{
    height: 16px;
    background-color: {COLORS['bg_base']};
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
    subcontrol-position: bottom;
    subcontrol-origin: margin;
}}
QScrollBar:horizontal {{
    background: {COLORS['bg_base']};
    height: 16px;
    border: 1px solid {COLORS['border']};
}}
QScrollBar::handle:horizontal {{
    background-color: {COLORS['bg_base']};
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
}}

/* === FRAMES / CARDS === */
QFrame#card {{
    background-color: {COLORS['bg_card']};
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
    border-radius: 0px;
}}
QFrame#card_highlight {{
    background-color: {COLORS['bg_card']};
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
    border-radius: 0px;
}}
QFrame#inset {{
    background-color: {COLORS['bg_elevated']};
    border-top: 2px solid {COLORS['border']};
    border-left: 2px solid {COLORS['border']};
    border-right: 2px solid {COLORS['border_bright']};
    border-bottom: 2px solid {COLORS['border_bright']};
    border-radius: 0px;
}}

/* === LABELS === */
QLabel#disclaimer {{
    background-color: {COLORS['bg_base']};
    color: {COLORS['text_muted']};
    font-size: 11px;
    padding: 4px 16px;
    border-top: 1px solid {COLORS['border']};
}}
QLabel#page_title {{
    font-size: 18px;
    font-weight: 700;
    color: {COLORS['text_primary']};
}}
QLabel#section_title {{
    font-size: 11px;
    font-weight: 700;
    color: #ffffff;
    background-color: {COLORS['accent']};
    padding: 2px 6px;
}}

/* === RADIO BUTTONS === */
QRadioButton {{
    color: {COLORS['text_primary']};
    spacing: 6px;
}}
QRadioButton::indicator {{
    width: 13px;
    height: 13px;
    border: 2px solid {COLORS['border']};
    border-radius: 7px;
    background: {COLORS['bg_elevated']};
}}
QRadioButton::indicator:checked {{
    background: {COLORS['text_primary']};
    border: 2px solid {COLORS['border']};
}}

/* === TOOLTIP === */
QToolTip {{
    background-color: #ffffe1;
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['text_primary']};
    padding: 4px 8px;
    font-size: 11px;
}}

/* === DIALOG === */
QDialog {{
    background-color: {COLORS['bg_surface']};
}}

/* === SPLITTER === */
QSplitter::handle {{
    background-color: {COLORS['border']};
}}
"""
