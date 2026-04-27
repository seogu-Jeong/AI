import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, 
    QAbstractItemView, QSplitter, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor, QFont

from ui.styles.theme import COLORS
from db.models import AIScore
from sqlalchemy import func

class MacroWorker(QObject):
    finished = Signal(list, list) # macro_events, earnings_events
    error = Signal(str)
    progress = Signal(int, int)

    def __init__(self):
        super().__init__()

    def run(self):
        from db.database import SessionLocal
        db = SessionLocal()
        try:
            # 1. Macro Events
            macro_events = self._get_macro_events()
            
            # 2. Earnings Events
            earnings_events = self._get_earnings_events(db)
            
            self.finished.emit(macro_events, earnings_events)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            db.close()

    def _get_macro_events(self):
        events = []
        today = datetime.now()
        end_date = today + timedelta(days=90)
        
        # FOMC 2026
        fomc_dates = [
            "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17", 
            "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16"
        ]
        for d_str in fomc_dates:
            d = datetime.strptime(d_str, "%Y-%m-%d")
            if today <= d <= end_date:
                events.append({"date": d, "event": "FOMC 정례회의 (금리 결정)", "importance": "HIGH", "estimate": "-"})

        # CPI 2026 (Approx/Known)
        cpi_dates = ["2026-04-30", "2026-05-27", "2026-06-25", "2026-07-29"]
        for d_str in cpi_dates:
            d = datetime.strptime(d_str, "%Y-%m-%d")
            if today <= d <= end_date:
                events.append({"date": d, "event": "CPI (소비자물가지수) 발표", "importance": "HIGH", "estimate": "-"})

        # NFP (First Friday)
        for i in range(4):
            # Start from first day of month
            first_day = (today.replace(day=1) + timedelta(days=32 * i)).replace(day=1)
            # Find first friday
            days_to_friday = (4 - first_day.weekday() + 7) % 7
            nfp_date = first_day + timedelta(days=days_to_friday)
            if today <= nfp_date <= end_date:
                events.append({"date": nfp_date, "event": "고용보고서 (NFP/실업률)", "importance": "HIGH", "estimate": "-"})

        # GDP 2026
        gdp_dates = ["2026-04-30", "2026-07-30", "2026-10-29"]
        for d_str in gdp_dates:
            d = datetime.strptime(d_str, "%Y-%m-%d")
            if today <= d <= end_date:
                events.append({"date": d, "event": "GDP 성장률 (속보치)", "importance": "MEDIUM", "estimate": "-"})

        # Sort by date
        events.sort(key=lambda x: x['date'])
        return events

    def _get_earnings_events(self, db):
        # Get 20 tickers from DB (latest scored)
        try:
            tickers_query = db.query(AIScore.ticker).distinct().limit(20).all()
            tickers = [t[0] for t in tickers_query]
        except:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JNJ", "V"]

        # Get latest AI scores
        latest_date_sub = db.query(
            AIScore.ticker,
            func.max(AIScore.date).label('max_date')
        ).group_by(AIScore.ticker).subquery()

        scores_query = db.query(
            AIScore.ticker,
            AIScore.ensemble_score
        ).join(
            latest_date_sub,
            (AIScore.ticker == latest_date_sub.c.ticker) & (AIScore.date == latest_date_sub.c.max_date)
        ).all()
        score_map = {s.ticker: s.ensemble_score for s in scores_query}

        earnings_list = []
        today = datetime.now()
        horizon = today + timedelta(days=30)

        total = len(tickers)
        for i, ticker in enumerate(tickers):
            self.progress.emit(i + 1, total)
            try:
                t = yf.Ticker(ticker)
                cal = t.get_calendar()
                if cal is not None and 'Earnings Date' in cal:
                    e_dates = cal['Earnings Date']
                    if isinstance(e_dates, list):
                        e_date = e_dates[0]
                    else:
                        e_date = e_dates
                    
                    # e_date might be timestamp or datetime
                    if hasattr(e_date, 'to_pydatetime'):
                        e_date = e_date.to_pydatetime()
                    
                    # Remove timezone if exists for comparison
                    if e_date.tzinfo is not None:
                        e_date = e_date.replace(tzinfo=None)

                    if today <= e_date <= horizon:
                        d_day = (e_date.date() - today.date()).days
                        eps_est = cal.get('EPS Estimate', '-')
                        
                        earnings_list.append({
                            'date': e_date,
                            'ticker': ticker,
                            'score': score_map.get(ticker, 0),
                            'eps_est': eps_est,
                            'd_day': d_day
                        })
            except Exception as e:
                print(f"Error fetching earnings for {ticker}: {e}")
                continue

        earnings_list.sort(key=lambda x: x['date'])
        return earnings_list

class MacroCalendarPage(QWidget):
    def __init__(self, db_session, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(20)

        # HEADER
        header = QVBoxLayout()
        title = QLabel('거시경제 캘린더')
        title.setObjectName('page_title')
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #000000;")
        
        subtitle = QLabel('향후 30일 주요 경제 이벤트 및 어닝 발표 일정')
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        
        header.addWidget(title)
        header.addWidget(subtitle)
        
        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton('갱신')
        self.refresh_btn.setMinimumWidth(100)
        self.refresh_btn.clicked.connect(self.refresh_data)
        btn_row.addWidget(self.refresh_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(200)
        btn_row.addWidget(self.progress_bar)
        
        self.status_lbl = QLabel('')
        self.status_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        btn_row.addWidget(self.status_lbl)
        btn_row.addStretch()
        
        header.addLayout(btn_row)
        layout.addLayout(header)

        # SPLITTER
        splitter = QSplitter(Qt.Vertical)
        
        # Section A: Macro Events
        macro_card = QFrame()
        macro_card.setObjectName('card')
        macro_card.setStyleSheet("background-color: #ffffff; border: 1px solid #808080;")
        macro_layout = QVBoxLayout(macro_card)
        
        macro_title = QLabel('주요 경제 이벤트 (90일 이내)')
        macro_title.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 5px;")
        macro_layout.addWidget(macro_title)
        
        self.macro_table = QTableWidget()
        self.macro_table.setColumnCount(4)
        self.macro_table.setHorizontalHeaderLabels(['날짜', '이벤트', '중요도', '예상치'])
        self.macro_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.macro_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.macro_table.verticalHeader().setVisible(False)
        self.macro_table.setAlternatingRowColors(True)
        macro_layout.addWidget(self.macro_table)
        
        splitter.addWidget(macro_card)
        
        # Section B: Earnings Events
        earnings_card = QFrame()
        earnings_card.setObjectName('card')
        earnings_card.setStyleSheet("background-color: #ffffff; border: 1px solid #808080;")
        earnings_layout = QVBoxLayout(earnings_card)
        
        earnings_title = QLabel('주요 종목 어닝 발표 일정 (30일 이내)')
        earnings_title.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 5px;")
        earnings_layout.addWidget(earnings_title)
        
        self.earnings_table = QTableWidget()
        self.earnings_table.setColumnCount(5)
        self.earnings_table.setHorizontalHeaderLabels(['날짜', '종목', 'AI 점수', '예상 EPS', 'D-Day'])
        self.earnings_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.earnings_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.earnings_table.verticalHeader().setVisible(False)
        self.earnings_table.setAlternatingRowColors(True)
        earnings_layout.addWidget(self.earnings_table)
        
        splitter.addWidget(earnings_card)
        
        layout.addWidget(splitter, 1)

        # Load initial data
        self.refresh_data()

    def refresh_data(self):
        # Guard: don't start if already running
        if hasattr(self, '_macro_thread') and self._macro_thread is not None:
            try:
                if self._macro_thread.isRunning():
                    return
            except RuntimeError:
                pass

        self.refresh_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_lbl.setText('데이터를 가져오는 중...')

        self._macro_thread = QThread()
        self._macro_worker = MacroWorker()
        self._macro_worker.moveToThread(self._macro_thread)

        self._macro_thread.started.connect(self._macro_worker.run)
        self._macro_worker.finished.connect(self._on_results)
        self._macro_worker.finished.connect(self._macro_thread.quit)
        self._macro_worker.error.connect(self._on_error)
        self._macro_worker.progress.connect(self._on_progress)
        self._macro_thread.finished.connect(self._on_thread_done)

        self._macro_thread.start()

    def _on_thread_done(self):
        self._macro_thread = None
        self._macro_worker = None

    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_error(self, message):
        self.refresh_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText(f'오류: {message}')

    def _on_results(self, macro_events, earnings_events):
        try:
            self.refresh_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_lbl.setText(f'마지막 갱신: {datetime.now().strftime("%H:%M:%S")}')

            self.macro_table.setRowCount(0)
            for ev in macro_events:
                try:
                    row = self.macro_table.rowCount()
                    self.macro_table.insertRow(row)
                    date_str = ev['date'].strftime('%Y-%m-%d (%a)')
                    self.macro_table.setItem(row, 0, QTableWidgetItem(date_str))
                    self.macro_table.setItem(row, 1, QTableWidgetItem(ev['event']))
                    imp_item = QTableWidgetItem(ev['importance'])
                    if ev['importance'] == 'HIGH':
                        imp_item.setBackground(QColor('#ffcccc'))
                        imp_item.setForeground(QColor('#cc0000'))
                    elif ev['importance'] == 'MEDIUM':
                        imp_item.setBackground(QColor('#fff4cc'))
                        imp_item.setForeground(QColor('#856404'))
                    else:
                        imp_item.setBackground(QColor('#f0f0f0'))
                    imp_item.setTextAlignment(Qt.AlignCenter)
                    self.macro_table.setItem(row, 2, imp_item)
                    self.macro_table.setItem(row, 3, QTableWidgetItem(str(ev.get('estimate', '-'))))
                except Exception as e:
                    print(f'macro row error: {e}')

            self.earnings_table.setRowCount(0)
            for ev in earnings_events:
                try:
                    row = self.earnings_table.rowCount()
                    self.earnings_table.insertRow(row)
                    date_str = ev['date'].strftime('%Y-%m-%d')
                    self.earnings_table.setItem(row, 0, QTableWidgetItem(date_str))
                    ticker_item = QTableWidgetItem(str(ev['ticker']))
                    ticker_item.setFont(QFont('', -1, QFont.Weight.Bold))
                    self.earnings_table.setItem(row, 1, ticker_item)
                    score = ev.get('score') or 0
                    score_item = QTableWidgetItem(f'{score:.1f}')
                    score_item.setTextAlignment(Qt.AlignCenter)
                    if score >= 60:
                        score_item.setForeground(QColor(COLORS['bull']))
                    elif score <= 40:
                        score_item.setForeground(QColor(COLORS['bear']))
                    self.earnings_table.setItem(row, 2, score_item)
                    self.earnings_table.setItem(row, 3, QTableWidgetItem(str(ev.get('eps_est', '-'))))
                    d_day = ev.get('d_day', 0)
                    d_day_str = f'D-{d_day}' if d_day > 0 else 'TODAY'
                    d_day_item = QTableWidgetItem(d_day_str)
                    d_day_item.setTextAlignment(Qt.AlignCenter)
                    if d_day <= 3:
                        d_day_item.setForeground(QColor('#cc0000'))
                    elif d_day <= 7:
                        d_day_item.setForeground(QColor('#ff8c00'))
                    else:
                        d_day_item.setForeground(QColor('#808080'))
                    self.earnings_table.setItem(row, 4, d_day_item)
                except Exception as e:
                    print(f'earnings row error: {e}')
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_lbl.setText(f'표시 오류: {e}')
