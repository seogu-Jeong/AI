import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QComboBox, QGridLayout, 
                             QProgressBar, QScrollArea, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ui.styles.theme import COLORS, get_score_color
from db.models import AIScore, RawOHLCV
from sqlalchemy import func

# Set matplotlib style for dark theme or consistency
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['figure.facecolor'] = COLORS['bg_base']
plt.rcParams['axes.edgecolor'] = COLORS['border']
plt.rcParams['axes.labelcolor'] = COLORS['text_primary']
plt.rcParams['xtick.color'] = COLORS['text_primary']
plt.rcParams['ytick.color'] = COLORS['text_primary']
plt.rcParams['legend.facecolor'] = '#ffffff'
plt.rcParams['grid.color'] = '#e0e0e0'

class CompareWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, tickers):
        super().__init__()
        self.tickers = [t for t in tickers if t]

    def run(self):
        from db.database import SessionLocal
        db = SessionLocal()
        try:
            results = {}
            three_months_ago = (datetime.now() - timedelta(days=90)).date()
            
            for ticker in self.tickers:
                # 1. Fetch Latest AI Scores
                latest_score = db.query(AIScore).filter(
                    AIScore.ticker == ticker
                ).order_by(AIScore.date.desc()).first()
                
                # 2. Fetch Price History for last 3 months
                prices = db.query(RawOHLCV.date, RawOHLCV.close).filter(
                    RawOHLCV.ticker == ticker,
                    RawOHLCV.date >= three_months_ago
                ).order_by(RawOHLCV.date.asc()).all()
                
                if not prices:
                    continue
                
                df_prices = pd.DataFrame(prices, columns=['date', 'close'])
                df_prices['date'] = pd.to_datetime(df_prices['date'])
                
                # Compute Returns
                # 1 month (approx 21 trading days)
                # 3 months (approx 63 trading days)
                curr_price = df_prices['close'].iloc[-1]
                
                ret_1m = 0.0
                if len(df_prices) >= 21:
                    prev_1m = df_prices['close'].iloc[-21]
                    ret_1m = (curr_price / prev_1m - 1) * 100
                elif len(df_prices) > 1:
                    prev_1m = df_prices['close'].iloc[0]
                    ret_1m = (curr_price / prev_1m - 1) * 100

                ret_3m = 0.0
                if len(df_prices) >= 63:
                    prev_3m = df_prices['close'].iloc[-63]
                    ret_3m = (curr_price / prev_3m - 1) * 100
                elif len(df_prices) > 1:
                    prev_3m = df_prices['close'].iloc[0]
                    ret_3m = (curr_price / prev_3m - 1) * 100

                # Normalized Prices
                base_price = df_prices['close'].iloc[0]
                df_prices['norm_close'] = (df_prices['close'] / base_price) * 100
                
                results[ticker] = {
                    'latest_score': latest_score,
                    'prices': df_prices,
                    'ret_1m': ret_1m,
                    'ret_3m': ret_3m
                }
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            db.close()

class ComparePage(QWidget):
    def __init__(self, db_session, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.init_ui()
        self.load_tickers()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 24, 30, 24)
        main_layout.setSpacing(20)

        # Title
        title_vbox = QVBoxLayout()
        title = QLabel("종목 비교")
        title.setObjectName("page_title")
        subtitle = QLabel("최대 3개 종목의 AI 점수 · 수익률 · 신호를 나란히 비교")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        title_vbox.addWidget(title)
        title_vbox.addWidget(subtitle)
        main_layout.addLayout(title_vbox)

        # Row 1: Ticker Selectors
        selector_frame = QFrame()
        selector_frame.setObjectName("card")
        selector_layout = QHBoxLayout(selector_frame)
        selector_layout.setContentsMargins(20, 15, 20, 15)
        selector_layout.setSpacing(20)

        self.combos = []
        for i in range(3):
            vbox = QVBoxLayout()
            label = QLabel(f"종목 {i+1}")
            label.setStyleSheet("font-weight: bold; font-size: 11px;")
            combo = QComboBox()
            combo.setEditable(True)
            combo.setFixedWidth(150)
            vbox.addWidget(label)
            vbox.addWidget(combo)
            selector_layout.addLayout(vbox)
            self.combos.append(combo)

        selector_layout.addStretch()
        
        self.compare_btn = QPushButton("비교 실행")
        self.compare_btn.setFixedHeight(40)
        self.compare_btn.setFixedWidth(120)
        self.compare_btn.setCursor(Qt.PointingHandCursor)
        self.compare_btn.clicked.connect(self.run_comparison)
        selector_layout.addWidget(self.compare_btn)

        main_layout.addWidget(selector_frame)

        # Scroll Area for results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(20)
        
        # Row 2: Stat Cards
        self.cards_grid = QGridLayout()
        self.cards_grid.setSpacing(20)
        self.content_layout.addLayout(self.cards_grid)

        # Row 3: Price Chart
        self.price_fig = Figure(figsize=(10, 4), dpi=100)
        self.price_canvas = FigureCanvas(self.price_fig)
        self.price_canvas.setFixedHeight(300)
        chart_frame = QFrame()
        chart_frame.setObjectName("card")
        chart_vbox = QVBoxLayout(chart_frame)
        chart_vbox.addWidget(self.price_canvas)
        self.content_layout.addWidget(chart_frame)

        # Row 4: Score Comparison Chart
        self.score_fig = Figure(figsize=(10, 4), dpi=100)
        self.score_canvas = FigureCanvas(self.score_fig)
        self.score_canvas.setFixedHeight(300)
        score_frame = QFrame()
        score_frame.setObjectName("card")
        score_vbox = QVBoxLayout(score_frame)
        score_vbox.addWidget(self.score_canvas)
        self.content_layout.addWidget(score_frame)

        scroll.setWidget(self.content_widget)
        main_layout.addWidget(scroll)

    def load_tickers(self):
        from db.database import SessionLocal
        db = SessionLocal()
        try:
            tickers = db.query(AIScore.ticker).distinct().order_by(AIScore.ticker).all()
            tickers = [t[0] for t in tickers]
            
            for combo in self.combos:
                current = combo.currentText()
                combo.clear()
                combo.addItems(tickers)
                if current in tickers:
                    combo.setCurrentText(current)
            
            # If nothing selected, set defaults
            if not self.combos[0].currentText() and len(tickers) >= 3:
                for i in range(3):
                    self.combos[i].setCurrentIndex(i)
        except Exception as e:
            print(f"Error loading tickers: {e}")
        finally:
            db.close()

    def refresh(self):
        self.load_tickers()

    def run_comparison(self):
        tickers = [c.currentText() for c in self.combos if c.currentText()]
        if not tickers:
            return
            
        self.compare_btn.setEnabled(False)
        self.compare_btn.setText("계산 중...")
        
        self.worker = CompareWorker(tickers)
        self.worker.finished.connect(self.on_comparison_finished)
        self.worker.error.connect(self.on_comparison_error)
        self.worker.start()

    def on_comparison_error(self, err_msg):
        print(f"Comparison Error: {err_msg}")
        self.compare_btn.setEnabled(True)
        self.compare_btn.setText("비교 실행")

    def on_comparison_finished(self, results):
        self.compare_btn.setEnabled(True)
        self.compare_btn.setText("비교 실행")
        
        # Clear existing cards
        for i in reversed(range(self.cards_grid.count())): 
            self.cards_grid.itemAt(i).widget().setParent(None)
            
        tickers = list(results.keys())
        
        # 1. Update Stat Cards
        for i, ticker in enumerate(tickers):
            data = results[ticker]
            score_obj = data['latest_score']
            
            card = QFrame()
            card.setObjectName("card")
            card.setStyleSheet(f"background-color: {COLORS['bg_elevated']};")
            vbox = QVBoxLayout(card)
            vbox.setSpacing(10)
            
            # Ticker Name
            t_label = QLabel(ticker)
            t_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #000080;")
            t_label.setAlignment(Qt.AlignCenter)
            vbox.addWidget(t_label)
            
            if score_obj:
                score = score_obj.ensemble_score
                # AI Score
                s_box = QHBoxLayout()
                s_label = QLabel(f"AI 점수: {score:.1f}")
                s_label.setStyleSheet(f"font-weight: bold; color: {get_score_color(score)};")
                s_box.addWidget(s_label)
                s_box.addStretch()
                vbox.addLayout(s_box)
                
                pbar = QProgressBar()
                pbar.setRange(0, 100)
                pbar.setValue(int(score))
                pbar.setTextVisible(False)
                pbar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {get_score_color(score)}; }}")
                vbox.addWidget(pbar)
                
                # MLP Signal
                sig = score_obj.mlp_signal or "Hold"
                sig_color = COLORS['bull'] if sig == "Buy" else (COLORS['bear'] if sig == "Sell" else COLORS['neutral'])
                sig_label = QLabel(f"MLP 신호: {sig}")
                sig_label.setStyleSheet(f"font-weight: bold; color: {sig_color}; padding: 4px; border: 1px solid {sig_color};")
                sig_label.setAlignment(Qt.AlignCenter)
                vbox.addWidget(sig_label)
            else:
                vbox.addWidget(QLabel("AI 점수 정보 없음"))

            # Returns
            r1m = data['ret_1m']
            r1m_c = COLORS['bull'] if r1m > 0 else (COLORS['bear'] if r1m < 0 else COLORS['text_primary'])
            vbox.addWidget(QLabel(f"1개월 수익률: <span style='color:{r1m_c}; font-weight:bold;'>{r1m:+.2f}%</span>"))
            
            r3m = data['ret_3m']
            r3m_c = COLORS['bull'] if r3m > 0 else (COLORS['bear'] if r3m < 0 else COLORS['text_primary'])
            vbox.addWidget(QLabel(f"3개월 수익률: <span style='color:{r3m_c}; font-weight:bold;'>{r3m:+.2f}%</span>"))

            if score_obj:
                # LSTM Direction
                vbox.addWidget(QLabel(f"LSTM 방향: {score_obj.lstm_dir_5d or '-'}"))
                # CNN Pattern
                vbox.addWidget(QLabel(f"CNN 패턴: {score_obj.cnn_pattern or '-'}"))

            self.cards_grid.addWidget(card, 0, i)

        # 2. Update Price Chart
        self.price_fig.clear()
        ax = self.price_fig.add_subplot(111)
        colors = [COLORS['accent'], COLORS['accent2'], COLORS['warning']]
        
        for i, ticker in enumerate(tickers):
            df = results[ticker]['prices']
            ax.plot(df['date'], df['norm_close'], label=ticker, color=colors[i % len(colors)], linewidth=2)
            
        ax.axhline(100, color='black', linestyle='--', alpha=0.3)
        ax.set_title("최근 3개월 가격 추이 (정규화)", fontsize=12, fontweight='bold', pad=15)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylabel("정규화 가격 (시작=100)")
        self.price_fig.tight_layout()
        self.price_canvas.draw()

        # 3. Update AI Sub-scores Chart
        self.score_fig.clear()
        ax2 = self.score_fig.add_subplot(111)
        
        models = ['LSTM', 'CNN', 'Transformer', 'MLP']
        x = np.arange(len(models))
        width = 0.25
        
        for i, ticker in enumerate(tickers):
            score_obj = results[ticker]['latest_score']
            if score_obj:
                vals = [score_obj.lstm_score, score_obj.cnn_score, score_obj.transformer_score, score_obj.mlp_score]
                ax2.bar(x + (i - (len(tickers)-1)/2) * width, vals, width, label=ticker, color=colors[i % len(colors)])
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.set_ylim(0, 100)
        ax2.set_title("AI 모델별 점수 비교", fontsize=12, fontweight='bold', pad=15)
        ax2.legend(loc='upper right', frameon=True)
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
        self.score_fig.tight_layout()
        self.score_canvas.draw()
