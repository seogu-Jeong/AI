from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from ui.styles.theme import COLORS

class NewsCard(QFrame):
    def __init__(self, headline: str, source: str, score: float, label: str,
                 published: str = '', ticker: str = '', summary: str = ''):
        super().__init__()
        self.setObjectName('card')
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(12)

        score_col = QVBoxLayout()
        score_col.setSpacing(3)
        score_col.setAlignment(Qt.AlignTop)

        if score > 0.2:
            ind_color = COLORS['bull']
            bg_color = COLORS['bull_bg']
        elif score < -0.2:
            ind_color = COLORS['bear']
            bg_color = COLORS['bear_bg']
        else:
            ind_color = COLORS['neutral']
            bg_color = '#e0e0e0'

        if ticker:
            t_badge = QLabel(ticker)
            t_badge.setFixedWidth(54)
            t_badge.setAlignment(Qt.AlignCenter)
            t_badge.setStyleSheet('background-color: #000080; color: #ffffff; font-weight: 800; font-size: 10px; padding: 2px 0px;')
            score_col.addWidget(t_badge)

        score_text = ('+' if score > 0 else '') + f'{score:.2f}'
        indicator = QLabel(score_text)
        indicator.setFixedSize(54, 32)
        indicator.setAlignment(Qt.AlignCenter)
        indicator.setStyleSheet(f'background-color: {bg_color}; color: {ind_color}; font-weight: 800; font-size: 12px; border-top: 2px solid #ffffff; border-left: 2px solid #ffffff; border-right: 2px solid #808080; border-bottom: 2px solid #808080;')
        score_col.addWidget(indicator)

        lbl_tag = QLabel(label.upper())
        lbl_tag.setAlignment(Qt.AlignCenter)
        lbl_tag.setFixedWidth(54)
        lbl_tag.setStyleSheet(f'color: {ind_color}; font-size: 8px; font-weight: 800;')
        score_col.addWidget(lbl_tag)

        content_col = QVBoxLayout()
        content_col.setSpacing(4)

        hl = QLabel(headline)
        hl.setWordWrap(True)
        hl.setStyleSheet(f'font-size: 13px; font-weight: 700; color: {COLORS["text_primary"]};')
        content_col.addWidget(hl)

        if summary:
            sl = QLabel(summary)
            sl.setWordWrap(True)
            sl.setStyleSheet(f'font-size: 11px; color: {COLORS["text_secondary"]};')
            content_col.addWidget(sl)

        meta = QHBoxLayout()
        src_lbl = QLabel('[ ' + source + ' ]')
        src_lbl.setStyleSheet(f'color: {COLORS["accent"]}; font-size: 11px; font-weight: 600;')
        time_lbl = QLabel(published)
        time_lbl.setStyleSheet(f'color: {COLORS["text_muted"]}; font-size: 10px;')
        meta.addWidget(src_lbl)
        meta.addStretch()
        meta.addWidget(time_lbl)
        content_col.addLayout(meta)

        main_layout.addLayout(score_col)
        main_layout.addLayout(content_col, 1)
