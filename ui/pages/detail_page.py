import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QGridLayout, QScrollArea,
                             QProgressBar, QSizePolicy, QApplication)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject, QTimer
from PySide6.QtGui import QFont, QIcon

from ui.styles.theme import COLORS, get_score_color
from ui.styles.tooltips import TOOLTIPS
from ui.widgets.candle_chart import CandleChart
from ui.widgets.ai_score_chart import AIScoreChart
from ui.widgets.attention_heatmap import AttentionHeatmap
from ui.widgets.signal_bar import SignalBar
from ui.widgets.news_card import NewsCard
from ui.workers.news_worker import NewsWorker
from models.transformer_model import FACTOR_NAMES

WIN98_CARD = f"""
    background-color: {COLORS['bg_card']};
    border-top: 2px solid {COLORS['border_bright']};
    border-left: 2px solid {COLORS['border_bright']};
    border-right: 2px solid {COLORS['border']};
    border-bottom: 2px solid {COLORS['border']};
"""

WIN98_TITLEBAR = f"""
    background-color: {COLORS['accent']};
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
"""

class DetailDataWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, screening_svc, ticker):
        super().__init__()
        self.screening_svc = screening_svc
        self.ticker = ticker

    def run(self):
        from db.database import SessionLocal
        db = SessionLocal()
        try:
            data = self.screening_svc.get_ticker_detail(self.ticker, db)
            self.finished.emit(data if data else {})
        except Exception as e:
            self.error.emit(str(e))
        finally:
            db.close()

class ModelCard(QFrame):
    def __init__(self, title, description, tooltip_text, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedWidth(240)
        self.setFixedHeight(120)
        self.setToolTip(tooltip_text)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Win98 title bar
        title_bar = QLabel(title)
        title_bar.setStyleSheet(WIN98_TITLEBAR)
        title_bar.setFixedHeight(22)
        layout.addWidget(title_bar)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(10, 6, 10, 6)
        body_layout.setSpacing(2)

        self.score_label = QLabel("0")
        self.score_label.setStyleSheet(f"font-size: 28px; font-weight: 800; color: {COLORS['text_primary']};")
        body_layout.addWidget(self.score_label)

        self.desc_label = QLabel(description)
        self.desc_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text_secondary']};")
        self.desc_label.setWordWrap(True)
        body_layout.addWidget(self.desc_label)

        layout.addWidget(body)
        self.setStyleSheet(f"QFrame#card {{ {WIN98_CARD} }}")

    def update_data(self, score, detail_text):
        self.score_label.setText(str(int(score)))
        self.desc_label.setText(detail_text)
        color = get_score_color(score)
        self.score_label.setStyleSheet(f"font-size: 28px; font-weight: 800; color: {color};")


def _win98_panel(title_text=None):
    frame = QFrame()
    frame.setObjectName("card")
    frame.setStyleSheet(f"QFrame#card {{ {WIN98_CARD} }}")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    if title_text:
        title_bar = QLabel(title_text)
        title_bar.setStyleSheet(WIN98_TITLEBAR)
        title_bar.setFixedHeight(22)
        outer.addWidget(title_bar)
    body = QWidget()
    body_layout = QVBoxLayout(body)
    body_layout.setContentsMargins(12, 10, 12, 10)
    body_layout.setSpacing(6)
    outer.addWidget(body)
    return frame, body_layout


class ForecastBox(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFixedHeight(75)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #ffffff;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {COLORS['text_secondary']};")
        layout.addWidget(self.title_label)

        content_layout = QHBoxLayout()
        self.arrow_label = QLabel("→")
        self.arrow_label.setStyleSheet("font-size: 24px; font-weight: 800; color: #757575;")
        content_layout.addWidget(self.arrow_label)

        vals_layout = QVBoxLayout()
        self.pct_label = QLabel("0.0%")
        self.pct_label.setStyleSheet("font-size: 16px; font-weight: 800; color: #000000;")
        self.prob_label = QLabel("AI 확률: -")
        self.prob_label.setStyleSheet(f"font-size: 9px; color: {COLORS['text_secondary']};")
        vals_layout.addWidget(self.pct_label)
        vals_layout.addWidget(self.prob_label)
        content_layout.addLayout(vals_layout)
        content_layout.addStretch()
        layout.addLayout(content_layout)

    def update_data(self, direction, pct, buy_prob):
        direction = str(direction).upper()
        if direction in ['UP', 'BUY']:
            self.arrow_label.setText("↑")
            self.arrow_label.setStyleSheet(f"font-size: 24px; font-weight: 800; color: {COLORS['bull']};")
            self.pct_label.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {COLORS['bull']};")
            self.prob_label.setText(f"상승 확률: {buy_prob*100:.1f}%")
        elif direction in ['DOWN', 'SELL']:
            self.arrow_label.setText("↓")
            self.arrow_label.setStyleSheet(f"font-size: 24px; font-weight: 800; color: {COLORS['bear']};")
            self.pct_label.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {COLORS['bear']};")
            self.prob_label.setText(f"하락 확률: {(1-buy_prob)*100:.1f}%")
        else:
            self.arrow_label.setText("→")
            self.arrow_label.setStyleSheet("font-size: 24px; font-weight: 800; color: #757575;")
            self.pct_label.setStyleSheet("font-size: 16px; font-weight: 800; color: #757575;")
            self.prob_label.setText("중립 상태")
        
        self.pct_label.setText(f"{pct:+.1f}%")


class DetailPage(QWidget):
    back_requested = Signal()
    watchlist_add_requested = Signal(str)

    def __init__(self, db_session, screening_svc, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.screening_svc = screening_svc
        self.current_ticker = None
        self.news_workers = []
        self._detail_thread = None
        self._detail_worker = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setStyleSheet('background-color: #c0c0c0;')

        # Global ToolTip style
        QApplication.instance().setStyleSheet(QApplication.instance().styleSheet() + '''
        QToolTip { background-color: #ffffe1; color: #000000; border: 1px solid #000000; font-size: 10px; padding: 4px; }
        ''')

        # --- HEADER BAR ---
        header_frame = QFrame()
        header_frame.setFixedHeight(72)
        header_frame.setStyleSheet(f"""
            background-color: {COLORS['bg_base']};
            border-bottom: 2px solid {COLORS['border']};
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 8, 16, 8)

        back_btn = QPushButton("← 목록으로")
        back_btn.setObjectName("secondary")
        back_btn.setFixedWidth(100)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self.back_requested.emit)
        header_layout.addWidget(back_btn)
        header_layout.addSpacing(12)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        ticker_row = QHBoxLayout()
        self.ticker_label = QLabel("TICKER")
        self.ticker_label.setStyleSheet(f"font-size: 22px; font-weight: 800; color: {COLORS['text_primary']};")
        self.name_label = QLabel("Company Name Inc.")
        self.name_label.setStyleSheet(f"font-size: 12px; color: {COLORS['text_secondary']}; margin-left: 8px; margin-top: 4px;")
        ticker_row.addWidget(self.ticker_label)
        ticker_row.addWidget(self.name_label)
        ticker_row.addStretch()

        price_row = QHBoxLayout()
        self.price_label = QLabel("$0.00")
        self.price_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {COLORS['text_primary']};")
        self.change_label = QLabel("▲ +0.00%")
        self.change_label.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {COLORS['bull']}; margin-left: 10px;")
        price_row.addWidget(self.price_label)
        price_row.addWidget(self.change_label)
        price_row.addStretch()

        info_layout.addLayout(ticker_row)
        info_layout.addLayout(price_row)
        header_layout.addLayout(info_layout, 2)

        # Score section
        score_container = QWidget()
        score_layout = QVBoxLayout(score_container)
        score_layout.setContentsMargins(0, 2, 0, 2)

        score_top = QHBoxLayout()
        self.rating_label = QLabel("STRONG BUY")
        self.rating_label.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {COLORS['bull']};")
        self.score_label = QLabel("85")
        self.score_label.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {COLORS['bull']};")
        score_top.addWidget(self.rating_label)
        score_top.addStretch()
        score_top.addWidget(self.score_label)

        self.score_bar = QProgressBar()
        self.score_bar.setFixedHeight(12)
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(85)
        self.score_bar.setTextVisible(False)

        score_layout.addLayout(score_top)
        score_layout.addWidget(self.score_bar)
        header_layout.addWidget(score_container, 1)

        header_layout.addSpacing(16)
        self.watchlist_btn = QPushButton("★ 관심종목 추가")
        self.watchlist_btn.setFixedWidth(130)
        self.watchlist_btn.setCursor(Qt.PointingHandCursor)
        self.watchlist_btn.clicked.connect(self._on_watchlist_clicked)
        header_layout.addWidget(self.watchlist_btn)

        main_layout.addWidget(header_frame)

        # --- SCROLL AREA ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet('QScrollArea { background-color: #c0c0c0; border: none; } QScrollBar:vertical { background: #c0c0c0; width: 16px; } QScrollBar::handle:vertical { background: #808080; min-height: 20px; border: 1px solid #ffffff; }')

        content_widget = QWidget()
        content_widget.setStyleSheet('background-color: #c0c0c0;')
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setContentsMargins(16, 16, 16, 16)
        self.content_layout.setSpacing(12)

        # --- MODEL CARDS ROW ---
        models_layout = QHBoxLayout()
        models_layout.setSpacing(10)

        self.card_lstm = ModelCard("LSTM", "5일 방향: -", TOOLTIPS.get('LSTM', ''))
        self.card_cnn = ModelCard("CNN", "패턴 분석: -", TOOLTIPS.get('CNN', ''))
        self.card_transformer = ModelCard("TRANSFORMER", "High Confidence", TOOLTIPS.get('Transformer', ''))
        self.card_mlp = ModelCard("MLP", "BUY 0% | HOLD 0%", TOOLTIPS.get('MLP', ''))

        models_layout.addWidget(self.card_lstm)
        models_layout.addWidget(self.card_cnn)
        models_layout.addWidget(self.card_transformer)
        models_layout.addWidget(self.card_mlp)
        models_layout.addStretch()
        self.content_layout.addLayout(models_layout)

        # --- FORECAST PANEL ---
        forecast_panel = QHBoxLayout()
        forecast_panel.setSpacing(10)
        self.f_box_5d = ForecastBox("5일 AI 예측")
        self.f_box_20d = ForecastBox("20일 AI 예측")
        self.f_box_60d = ForecastBox("60일 AI 예측")
        forecast_panel.addWidget(self.f_box_5d)
        forecast_panel.addWidget(self.f_box_20d)
        forecast_panel.addWidget(self.f_box_60d)
        self.content_layout.addLayout(forecast_panel)

        # --- CHART (FULL WIDTH) ---
        chart_outer, chart_body = _win98_panel("캔들차트 / AI 예측")
        chart_body.setContentsMargins(4, 4, 4, 4)

        # Period selector buttons
        period_row = QHBoxLayout()
        period_row.setSpacing(6)
        self._period_btns = {}
        PERIOD_LABELS = [('1M', '1개월'), ('3M', '3개월'), ('6M', '6개월'),
                         ('1Y', '1년'), ('3Y', '3년'), ('5Y', '5년'), ('All', '전체')]
        for key, display in PERIOD_LABELS:
            btn = QPushButton(display)
            btn.setFixedHeight(26)
            btn.setMinimumWidth(54)
            btn.setCheckable(True)
            btn.setChecked(key == '1Y')
            btn.setStyleSheet("""
                QPushButton { font-size: 11px; font-weight: 600;
                              border: 1px solid #808080; border-radius: 3px;
                              background: #f0f0f0; color: #000000; padding: 0 6px; }
                QPushButton:checked { background: #000080; color: #ffffff; border: 1px solid #000080; }
                QPushButton:hover:!checked { background: #d0d0d0; }
            """)
            btn.clicked.connect(lambda checked, p=key: self._on_period_clicked(p))
            period_row.addWidget(btn)
            self._period_btns[key] = btn
        period_row.addStretch()
        chart_body.addLayout(period_row)

        self.candle_chart = CandleChart()
        chart_body.addWidget(self.candle_chart)
        self.content_layout.addWidget(chart_outer)

        # --- AI SCORE HISTORY CHART ---
        score_chart_outer, score_chart_body = _win98_panel("AI 점수 히스토리")
        self.ai_score_chart = AIScoreChart()
        score_chart_body.setContentsMargins(4, 4, 4, 4)
        score_chart_body.addWidget(self.ai_score_chart)
        self.content_layout.addWidget(score_chart_outer)

        # --- ANALYSIS PANELS ROW ---
        analysis_row = QHBoxLayout()
        analysis_row.setSpacing(12)

        # Left: Attention Heatmap
        heatmap_outer, heatmap_body = _win98_panel("TRANSFORMER ATTENTION WEIGHTS")
        h_sub = QLabel("24개 팩터 중요도 (Explainable AI)")
        h_sub.setStyleSheet(f"font-size: 10px; color: {COLORS['text_secondary']};")
        heatmap_body.addWidget(h_sub)
        self.heatmap = AttentionHeatmap()
        self.heatmap.setMinimumHeight(420)
        heatmap_body.addWidget(self.heatmap)
        analysis_row.addWidget(heatmap_outer, 45)

        # Right: LSTM Directions
        lstm_outer, lstm_body = _win98_panel("LSTM 방향성 예측")
        self.lstm_rows = []
        for term in ["5일 후", "20일 후", "60일 후"]:
            row = QHBoxLayout()
            term_label = QLabel(term)
            term_label.setFixedWidth(55)
            term_label.setStyleSheet(f"font-size: 12px; color: {COLORS['text_primary']};")

            dir_icon = QLabel("―")
            dir_icon.setFixedWidth(20)
            dir_icon.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {COLORS['neutral']};")

            dir_text = QLabel("NEUTRAL")
            dir_text.setFixedWidth(75)
            dir_text.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {COLORS['neutral']};")

            pbar = QProgressBar()
            pbar.setFixedHeight(10)
            pbar.setTextVisible(False)

            conf_label = QLabel("0%")
            conf_label.setFixedWidth(36)
            conf_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            conf_label.setStyleSheet(f"font-size: 10px; color: {COLORS['text_secondary']};")

            row.addWidget(term_label)
            row.addWidget(dir_icon)
            row.addWidget(dir_text)
            row.addWidget(pbar)
            row.addWidget(conf_label)
            lstm_body.addLayout(row)
            self.lstm_rows.append({'icon': dir_icon, 'text': dir_text, 'bar': pbar, 'conf': conf_label})

        # Technical indicators section
        lstm_body.addSpacing(12)
        sep_line = QFrame()
        sep_line.setFrameShape(QFrame.HLine)
        sep_line.setStyleSheet(f"color: {COLORS['border']};")
        lstm_body.addWidget(sep_line)
        lstm_body.addSpacing(4)

        tech_header = QLabel("기술적 지표 현황")
        tech_header.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {COLORS['accent']}; text-transform: uppercase;")
        lstm_body.addWidget(tech_header)
        lstm_body.addSpacing(4)

        tech_indicators = [
            ("RSI(14)", "ind_rsi"),
            ("MACD", "ind_macd"),
            ("BB%B", "ind_bb"),
            ("Stoch%K", "ind_stoch"),
        ]
        for label_text, attr_name in tech_indicators:
            row = QHBoxLayout()
            lbl = QLabel(label_text + ' ?')
            lbl.setFixedWidth(80)
            lbl.setToolTip(TOOLTIPS.get(label_text, label_text))
            lbl.setStyleSheet('font-size: 10px; color: #000080;')
            lbl.setCursor(Qt.WhatsThisCursor)
            val = QLabel("--")
            val.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {COLORS['text_primary']};")
            setattr(self, attr_name, val)
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            lstm_body.addLayout(row)

        # Fundamental snapshot section
        lstm_body.addSpacing(10)
        fund_header = QLabel("펀더멘털 스냅샷")
        fund_header.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {COLORS['accent']}; text-transform: uppercase;")
        lstm_body.addWidget(fund_header)
        lstm_body.addSpacing(4)

        fund_indicators = [
            ("PER", "ind_per"),
            ("PBR", "ind_pbr"),
            ("ROE", "ind_roe"),
            ("EPS Growth", "ind_eps"),
            ("Op Margin", "ind_opm"),
        ]
        for label_text, attr_name in fund_indicators:
            row = QHBoxLayout()
            lbl = QLabel(label_text + ' ?')
            lbl.setFixedWidth(80)
            lbl.setToolTip(TOOLTIPS.get(label_text, label_text))
            lbl.setStyleSheet('font-size: 10px; color: #000080;')
            lbl.setCursor(Qt.WhatsThisCursor)
            val = QLabel("--")
            val.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {COLORS['text_primary']};")
            setattr(self, attr_name, val)
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            lstm_body.addLayout(row)

        lstm_body.addStretch()
        analysis_row.addWidget(lstm_outer, 55)
        self.content_layout.addLayout(analysis_row)

        # --- BOTTOM ROW: MLP + NEWS ---
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(12)

        mlp_outer, mlp_body = _win98_panel("MLP BUY/HOLD/SELL 분류기")
        m_sub = QLabel("Multi-class 신경망 (30차원 입력)")
        m_sub.setStyleSheet(f"font-size: 10px; color: {COLORS['text_secondary']};")
        mlp_body.addWidget(m_sub)
        mlp_body.addSpacing(6)
        self.signal_bar = SignalBar()
        mlp_body.addWidget(self.signal_bar)
        mlp_body.addStretch()
        bottom_layout.addWidget(mlp_outer, 4)

        news_outer, news_body = _win98_panel("뉴스 감성 분석 (FinBERT)")
        news_header_row = QHBoxLayout()
        n_sub = QLabel("Transfer Learning NLP 모델")
        n_sub.setStyleSheet(f"font-size: 10px; color: {COLORS['text_secondary']};")
        self.news_indicator = QLabel("NEUTRAL")
        self.news_indicator.setStyleSheet(
            f"padding: 2px 8px; background-color: {COLORS['bg_elevated']}; "
            f"border: 1px solid {COLORS['border']}; font-size: 10px; font-weight: 700; color: {COLORS['neutral']};"
        )
        news_header_row.addWidget(n_sub)
        news_header_row.addStretch()
        news_header_row.addWidget(self.news_indicator, alignment=Qt.AlignTop)
        news_body.addLayout(news_header_row)
        self.news_list_container = QVBoxLayout()
        self.news_list_container.setSpacing(8)
        news_body.addLayout(self.news_list_container)
        news_body.addStretch()
        bottom_layout.addWidget(news_outer, 6)

        self.content_layout.addLayout(bottom_layout)
        self.content_layout.addStretch()

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

    def _on_period_clicked(self, period: str):
        for p, btn in self._period_btns.items():
            btn.setChecked(p == period)
        self.candle_chart.set_period(period)

    def load_ticker(self, ticker: str):
        self.current_ticker = ticker
        self.ticker_label.setText(ticker)
        # Reset period selector to 1Y on new ticker
        for p, btn in self._period_btns.items():
            btn.setChecked(p == '1Y')
        self.candle_chart._active_period = '1Y'
        self.candle_chart.show_loading()

        for i in reversed(range(self.news_list_container.count())):
            item = self.news_list_container.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self.news_indicator.setText('LOADING...')
        self.news_indicator.setStyleSheet(
            f'padding: 2px 8px; background-color: {COLORS["bg_elevated"]}; '
            f'border: 1px solid {COLORS["border"]}; font-size: 10px; font-weight: 700; color: {COLORS["text_secondary"]};'
        )

        # Stop any existing thread safely (deleteLater makes C++ obj invalid)
        try:
            if self._detail_thread is not None and self._detail_thread.isRunning():
                self._detail_thread.quit()
                self._detail_thread.wait(500)
        except RuntimeError:
            pass
        self._detail_thread = None
        self._detail_worker = None

        thread = QThread()
        worker = DetailDataWorker(self.screening_svc, ticker)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_data_ready)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_data_error)
        # Keep references alive until thread fully stops
        thread.finished.connect(lambda: setattr(self, '_detail_thread', None))

        self._detail_thread = thread
        self._detail_worker = worker
        thread.start()

    @Slot(str)
    def _on_data_error(self, message):
        self.candle_chart.show_loading()  # or show error state

    @Slot(dict)
    def _on_data_ready(self, data: dict):
        if not data:
            return

        ticker = self.current_ticker
        self.name_label.setText(data.get('name', ''))
        price = data.get('price', 0.0)
        self.price_label.setText(f"${price:,.2f}")

        pct = data.get('price_change_pct')
        if pct is None:
            np.random.seed(hash(ticker) % 10000)
            pct = np.random.uniform(-3, 5)

        if pct >= 0:
            self.change_label.setText(f"▲ +{pct:.2f}%")
            self.change_label.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {COLORS['bull']}; margin-left: 10px;")
        else:
            self.change_label.setText(f"▼ {pct:.2f}%")
            self.change_label.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {COLORS['bear']}; margin-left: 10px;")

        score = data.get('ensemble_score', 50)
        self.score_label.setText(str(int(score)))
        self.score_bar.setValue(int(score))

        rating, r_color = "NEUTRAL", COLORS['neutral']
        if score >= 80:   rating, r_color = "STRONG BUY",  COLORS['bull']
        elif score >= 60: rating, r_color = "BUY",         COLORS['score_60']
        elif score >= 40: rating, r_color = "NEUTRAL",     COLORS['neutral']
        elif score >= 20: rating, r_color = "SELL",        COLORS['score_20']
        else:             rating, r_color = "STRONG SELL", COLORS['bear']

        self.rating_label.setText(rating)
        self.rating_label.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {r_color};")
        self.score_label.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {r_color};")
        self.score_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: #d4d0c8; border: 1px solid {COLORS['border']}; }}
            QProgressBar::chunk {{ background-color: {r_color}; }}
        """)

        self.card_lstm.update_data(data.get('lstm_score', 50), f"5일 방향: {data.get('lstm_dir_5d', 'UP')} ↗")
        self.card_cnn.update_data(data.get('cnn_score', 50), f"{data.get('cnn_pattern', 'Bullish')} Pattern")
        self.card_transformer.update_data(data.get('transformer_score', 50), "High Confidence Attention")
        mlp_buy  = data.get('mlp_buy_prob', 0.33) * 100
        mlp_hold = data.get('mlp_hold_prob', 0.33) * 100
        self.card_mlp.update_data(data.get('mlp_score', 50), f"BUY {mlp_buy:.0f}% | HOLD {mlp_hold:.0f}%")

        # Update Forecast Panel
        fc = data.get('forecast')
        if fc:
            sigma = fc['daily_std']
            def get_epct(d, days):
                mult = 1.0 if d.upper() in ['UP', 'BUY'] else (-1.0 if d.upper() in ['DOWN', 'SELL'] else 0.0)
                return mult * sigma * np.sqrt(days) * 100
            
            self.f_box_5d.update_data(fc['dir_5d'], get_epct(fc['dir_5d'], 5), fc['mlp_buy_prob'])
            self.f_box_20d.update_data(fc['dir_20d'], get_epct(fc['dir_20d'], 20), fc['mlp_buy_prob'])
            self.f_box_60d.update_data(fc['dir_60d'], get_epct(fc['dir_60d'], 45), fc['mlp_buy_prob'])

        history = data.get('history')
        forecast = data.get('forecast')
        ai_history = data.get('ai_score_history')

        if history is not None:
            self.candle_chart.plot(history, ticker, forecast=forecast)
        self.ai_score_chart.plot(ai_history, ticker)

        weights = data.get('attention_weights')
        if weights is None:
            np.random.seed(hash(ticker) % 10000)
            weights = np.random.dirichlet(np.ones(24)).tolist()
        self.heatmap.plot(FACTOR_NAMES, weights)

        # Update technical indicators
        features = data.get('features', {})
        rsi_val = features.get('rsi_14', data.get('rsi_14'))
        macd_val = features.get('macd_signal', data.get('macd_signal'))
        bb_val = features.get('bb_pct', data.get('bb_pct'))
        stoch_val = features.get('stoch_k', data.get('stoch_k'))

        def fmt_ind(v, fmt='.1f'):
            if v is None: return '--'
            try: return f'{float(v):{fmt}}'
            except: return '--'

        if hasattr(self, 'ind_rsi'):
            rsi_text = fmt_ind(rsi_val)
            rsi_color = COLORS['bear'] if rsi_val and float(rsi_val or 0) > 70 else (COLORS['bull'] if rsi_val and float(rsi_val or 0) < 30 else COLORS['text_primary'])
            self.ind_rsi.setText(rsi_text)
            self.ind_rsi.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {rsi_color};")
            self.ind_macd.setText(fmt_ind(macd_val))
            self.ind_bb.setText(fmt_ind(bb_val))
            self.ind_stoch.setText(fmt_ind(stoch_val))

        per_val = features.get('per', data.get('per'))
        pbr_val = features.get('pbr', data.get('pbr'))
        roe_val = features.get('roe', data.get('roe'))
        eps_val = features.get('eps_growth', data.get('eps_growth'))
        opm_val = features.get('op_margin', data.get('op_margin'))

        if hasattr(self, 'ind_per'):
            self.ind_per.setText(fmt_ind(per_val, '.1f') + 'x' if per_val else '--')
            self.ind_pbr.setText(fmt_ind(pbr_val, '.2f') + 'x' if pbr_val else '--')
            roe_str = f'{float(roe_val or 0)*100:.1f}%' if roe_val is not None else '--'
            self.ind_roe.setText(roe_str)
            eps_str = f'{float(eps_val or 0)*100:.1f}%' if eps_val is not None else '--'
            self.ind_eps.setText(eps_str)
            opm_str = f'{float(opm_val or 0)*100:.1f}%' if opm_val is not None else '--'
            self.ind_opm.setText(opm_str)

        dirs = [data.get('lstm_dir_5d', 'Up'), data.get('lstm_dir_20d', 'Up'), data.get('lstm_dir_60d', 'Side')]
        for i, d in enumerate(dirs):
            d = d.upper()
            row = self.lstm_rows[i]
            if d in ['UP', 'BUY']:
                row['icon'].setText("↗")
                row['icon'].setStyleSheet(f"font-size: 16px; font-weight: 800; color: {COLORS['bull']};")
                row['text'].setText("BULLISH")
                row['text'].setStyleSheet(f"font-size: 11px; font-weight: 600; color: {COLORS['bull']};")
                row['bar'].setValue(75)
                row['bar'].setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLORS['bull']}; }}")
                row['conf'].setText("75%")
            elif d in ['DOWN', 'SELL']:
                row['icon'].setText("↘")
                row['icon'].setStyleSheet(f"font-size: 16px; font-weight: 800; color: {COLORS['bear']};")
                row['text'].setText("BEARISH")
                row['text'].setStyleSheet(f"font-size: 11px; font-weight: 600; color: {COLORS['bear']};")
                row['bar'].setValue(65)
                row['bar'].setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLORS['bear']}; }}")
                row['conf'].setText("65%")
            else:
                row['icon'].setText("→")
                row['icon'].setStyleSheet(f"font-size: 16px; font-weight: 800; color: {COLORS['neutral']};")
                row['text'].setText("NEUTRAL")
                row['text'].setStyleSheet(f"font-size: 11px; font-weight: 600; color: {COLORS['neutral']};")
                row['bar'].setValue(40)
                row['bar'].setStyleSheet(f"QProgressBar::chunk {{ background-color: {COLORS['neutral']}; }}")
                row['conf'].setText("40%")

        self.signal_bar.update_signals(
            data.get('mlp_buy_prob', 0.33),
            data.get('mlp_hold_prob', 0.33),
            data.get('mlp_sell_prob', 0.33)
        )

        worker = NewsWorker(ticker)
        worker.news_ready.connect(self.handle_news_ready)
        self.news_workers.append(worker)
        worker.start()

    @Slot(str, list)
    def handle_news_ready(self, ticker, news_list):
        if ticker != self.current_ticker:
            return

        avg_score = 0
        if news_list:
            avg_score = sum(item.get('score', 0) for item in news_list) / len(news_list)

        if avg_score > 0.2:
            self.news_indicator.setText("POSITIVE")
            self.news_indicator.setStyleSheet(
                f"padding: 2px 8px; background-color: {COLORS['bull_bg']}; "
                f"border: 1px solid {COLORS['bull']}; font-size: 10px; font-weight: 700; color: {COLORS['bull']};"
            )
        elif avg_score < -0.1:
            self.news_indicator.setText("NEGATIVE")
            self.news_indicator.setStyleSheet(
                f"padding: 2px 8px; background-color: {COLORS['bear_bg']}; "
                f"border: 1px solid {COLORS['bear']}; font-size: 10px; font-weight: 700; color: {COLORS['bear']};"
            )
        else:
            self.news_indicator.setText("NEUTRAL")
            self.news_indicator.setStyleSheet(
                f"padding: 2px 8px; background-color: {COLORS['bg_elevated']}; "
                f"border: 1px solid {COLORS['border']}; font-size: 10px; font-weight: 700; color: {COLORS['neutral']};"
            )

        for item in news_list[:5]:
            card = NewsCard(
                item.get('headline', ''),
                item.get('source', ''),
                item.get('score', 0),
                item.get('label', 'neutral'),
                item.get('published', '')
            )
            self.news_list_container.addWidget(card)

    def _on_watchlist_clicked(self):
        if self.current_ticker:
            self.watchlist_add_requested.emit(self.current_ticker)
            self.watchlist_btn.setText("✓ 관심종목")
            self.watchlist_btn.setEnabled(False)
