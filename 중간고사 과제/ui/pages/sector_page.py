import os
import pandas as pd
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, 
    QSpacerItem, QSizePolicy, QAbstractItemView, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sqlalchemy import func, desc

from ui.styles.theme import COLORS
from ui.widgets.help_button import HelpButton
from ui.styles.tooltips import TOOLTIPS
from db.models import AIScore, RawOHLCV
from data.collector import DataCollector

class SectorWorker(QObject):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        from db.database import SessionLocal
        db = SessionLocal()
        try:
            # 1. Get sector mapping
            ticker_sector = DataCollector.get_sp500_tickers_with_sectors()
            if not ticker_sector:
                self.error.emit("S&P500 섹터 정보를 가져오는데 실패했습니다.")
                return

            tickers = list(ticker_sector.keys())

            # 2. Get latest AI scores
            # Subquery to get latest date per ticker
            latest_date_sub = db.query(
                AIScore.ticker,
                func.max(AIScore.date).label('max_date')
            ).group_by(AIScore.ticker).subquery()

            latest_scores = db.query(
                AIScore.ticker,
                AIScore.ensemble_score,
                AIScore.mlp_signal
            ).join(
                latest_date_sub,
                (AIScore.ticker == latest_date_sub.c.ticker) & (AIScore.date == latest_date_sub.c.max_date)
            ).all()

            score_map = {s.ticker: (s.ensemble_score, s.mlp_signal) for s in latest_scores}

            # 3. Get 1-month return
            # Latest prices
            latest_price_sub = db.query(
                RawOHLCV.ticker,
                func.max(RawOHLCV.date).label('max_date')
            ).group_by(RawOHLCV.ticker).subquery()

            latest_prices = db.query(
                RawOHLCV.ticker,
                RawOHLCV.close
            ).join(
                latest_price_sub,
                (RawOHLCV.ticker == latest_price_sub.c.ticker) & (RawOHLCV.date == latest_price_sub.c.max_date)
            ).all()

            # Prices ~21 trading days ago
            # This is a bit tricky in SQLite, let's find the date that is roughly 30 days ago
            target_date = datetime.now().date() - timedelta(days=32)
            
            past_price_sub = db.query(
                RawOHLCV.ticker,
                func.max(RawOHLCV.date).label('past_date')
            ).filter(RawOHLCV.date <= target_date).group_by(RawOHLCV.ticker).subquery()

            past_prices = db.query(
                RawOHLCV.ticker,
                RawOHLCV.close
            ).join(
                past_price_sub,
                (RawOHLCV.ticker == past_price_sub.c.ticker) & (RawOHLCV.date == past_price_sub.c.past_date)
            ).all()

            curr_price_map = {p.ticker: p.close for p in latest_prices}
            past_price_map = {p.ticker: p.close for p in past_prices}

            # 4. Aggregate by sector
            sector_data = {} # sector -> [scores, returns, buy_count, ticker_count, top_ticker, top_score]

            for ticker, sector in ticker_sector.items():
                if sector not in sector_data:
                    sector_data[sector] = {'scores': [], 'returns': [], 'buy_count': 0, 'count': 0, 'top_ticker': None, 'top_score': -1.0}
                
                sector_data[sector]['count'] += 1
                
                if ticker in score_map:
                    score, signal = score_map[ticker]
                    sector_data[sector]['scores'].append(score)
                    if signal == 'Buy':
                        sector_data[sector]['buy_count'] += 1
                    
                    if score > sector_data[sector]['top_score']:
                        sector_data[sector]['top_score'] = score
                        sector_data[sector]['top_ticker'] = ticker
                
                if ticker in curr_price_map and ticker in past_price_map:
                    ret = (curr_price_map[ticker] / past_price_map[ticker]) - 1
                    sector_data[sector]['returns'].append(ret)

            # 5. Format results
            results = []
            for sector, data in sector_data.items():
                avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
                avg_return = sum(data['returns']) / len(data['returns']) if data['returns'] else 0
                buy_pct = (data['buy_count'] / data['count']) * 100 if data['count'] > 0 else 0
                
                results.append({
                    'sector': sector,
                    'count': data['count'],
                    'avg_score': avg_score,
                    'avg_return': avg_return,
                    'buy_pct': buy_pct,
                    'top_ticker': data['top_ticker']
                })

            # Sort by avg_score desc
            results.sort(key=lambda x: x['avg_score'], reverse=True)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            db.close()

class SectorPage(QWidget):
    def __init__(self, db_session, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.init_ui()

    def init_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet('QScrollArea { border: none; background: transparent; }')
        outer_layout.addWidget(scroll_area)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(20)
        scroll_area.setWidget(scroll_content)

        # 1. HEADER
        header_layout = QVBoxLayout()
        title_lbl = QLabel('섹터 로테이션 분석')
        title_lbl.setObjectName('page_title')
        
        subtitle_lbl = QLabel('11개 GICS 섹터별 AI 점수 · 수익률 · 매수 비율 히트맵')
        subtitle_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        
        header_layout.addWidget(title_lbl)
        header_layout.addWidget(subtitle_lbl)
        
        # Run Button Row
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton('분석 실행')
        self.run_btn.setMinimumWidth(120)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)
        
        self.status_lbl = QLabel('')
        self.status_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        btn_row.addWidget(self.status_lbl)
        btn_row.addStretch()
        
        header_layout.addLayout(btn_row)
        layout.addLayout(header_layout)

        # 2. CHARTS SECTION
        self.charts_card = QFrame()
        self.charts_card.setObjectName('card')
        charts_layout = QVBoxLayout(self.charts_card)
        
        self.figure = Figure(figsize=(10, 8), dpi=90, facecolor='#c0c0c0')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(600)
        charts_layout.addWidget(self.canvas)
        
        layout.addWidget(self.charts_card)
        self.charts_card.hide()

        # 3. TABLE SECTION
        self.table_card = QFrame()
        self.table_card.setObjectName('card')
        table_layout = QVBoxLayout(self.table_card)
        table_layout.setContentsMargins(15, 15, 15, 15)
        
        table_title = QLabel('섹터별 상세 지표')
        table_title.setObjectName('section_title')
        table_layout.addWidget(table_title)
        
        # Legend Row
        legend_layout = QHBoxLayout()
        legend_layout.setContentsMargins(5, 5, 5, 5)
        legend_layout.setSpacing(15)
        
        def add_legend_item(text, tooltip_key):
            item_lay = QHBoxLayout()
            item_lay.setSpacing(4)
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; font-weight: bold;")
            item_lay.addWidget(lbl)
            item_lay.addWidget(HelpButton(tooltip_key))
            legend_layout.addLayout(item_lay)

        add_legend_item("평균 AI 점수", "평균 AI 점수")
        add_legend_item("Buy%", "Buy%")
        legend_layout.addStretch()
        table_layout.addLayout(legend_layout)
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(['Sector', '# Tickers', 'Avg AI Score', 'Avg 1M Return', 'Buy%', 'Top Ticker'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(400)
        
        table_layout.addWidget(self.table)
        layout.addWidget(self.table_card)
        self.table_card.hide()

        layout.addStretch()

    def _on_run(self):
        self.run_btn.setEnabled(False)
        self.status_lbl.setText('분석 중...')
        
        self.thread = QThread()
        self.worker = SectorWorker()
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_results)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self._on_error)
        
        self.thread.start()

    def _on_error(self, message):
        self.status_lbl.setText(f"Error: {message}")
        self.run_btn.setEnabled(True)
        if hasattr(self, 'thread'):
            self.thread.quit()

    def _on_results(self, results):
        self.run_btn.setEnabled(True)
        self.status_lbl.setText('완료')
        
        if not results:
            self.status_lbl.setText('데이터가 없습니다.')
            return

        # 1. Update Charts
        self.figure.clear()
        
        # Color mapping for AI scores
        def get_score_color_mpl(score):
            if score >= 60: return '#008000' # Bull
            if score >= 40: return '#808080' # Neutral
            return '#cc0000' # Bear

        # Color mapping for returns
        def get_ret_color_mpl(ret):
            if ret > 0: return '#008000'
            if ret < 0: return '#cc0000'
            return '#808080'

        sectors = [r['sector'] for r in results]
        scores = [r['avg_score'] for r in results]
        returns = [r['avg_return'] * 100 for r in results]
        
        # Plot Score Chart
        ax1 = self.figure.add_subplot(211)
        ax1.set_facecolor('#ffffff')
        score_colors = [get_score_color_mpl(s) for s in scores]
        bars1 = ax1.barh(sectors, scores, color=score_colors)
        ax1.set_title('섹터별 평균 AI 점수 (Ensemble Score)', fontsize=10, fontweight='bold', pad=10)
        ax1.set_xlim(0, 100)
        ax1.invert_yaxis()
        ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # Plot Return Chart
        ax2 = self.figure.add_subplot(212)
        ax2.set_facecolor('#ffffff')
        ret_colors = [get_ret_color_mpl(r) for r in returns]
        bars2 = ax2.barh(sectors, returns, color=ret_colors)
        ax2.set_title('섹터별 평균 1개월 수익률 (%)', fontsize=10, fontweight='bold', pad=10)
        ax2.invert_yaxis()
        ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # Add labels on bars
        for i, bar in enumerate(bars1):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{scores[i]:.1f}", va='center', fontsize=8)
        for i, bar in enumerate(bars2):
            val = returns[i]
            x_pos = val + (0.5 if val >= 0 else -0.5)
            ha = 'left' if val >= 0 else 'right'
            ax2.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:+.1f}%", va='center', ha=ha, fontsize=8)

        self.figure.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.08, hspace=0.35)
        self.canvas.draw()
        self.charts_card.show()

        # 2. Update Table
        self.table.setRowCount(0)
        for r in results:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            self.table.setItem(row, 0, QTableWidgetItem(r['sector']))
            self.table.setItem(row, 1, QTableWidgetItem(str(r['count'])))
            
            score_item = QTableWidgetItem(f"{r['avg_score']:.1f}")
            score_item.setForeground(QColor(get_score_color_mpl(r['avg_score'])))
            score_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, score_item)
            
            ret_item = QTableWidgetItem(f"{r['avg_return']*100:+.2f}%")
            ret_item.setForeground(QColor(get_ret_color_mpl(r['avg_return'])))
            ret_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 3, ret_item)
            
            buy_item = QTableWidgetItem(f"{r['buy_pct']:.1f}%")
            buy_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 4, buy_item)
            
            self.table.setItem(row, 5, QTableWidgetItem(r['top_ticker'] or '-'))

        self.table_card.show()
