import random
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QLineEdit, QComboBox, QHeaderView,
    QFrame, QProgressBar, QAbstractItemView, QGraphicsDropShadowEffect,
    QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QColor, QFont, QIcon
from services.screening_service import ScreeningService
from db.database import SessionLocal
from ui.styles.theme import COLORS, get_score_color
from ui.widgets.help_button import HelpButton
from ui.styles.tooltips import TOOLTIPS
import yfinance as yf

class ScreeningWorker(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, n: int = 50, signal_filter: str = None, search: str = None):
        super().__init__()
        self.n = n
        self.signal_filter = signal_filter
        self.search = search
        self.service = ScreeningService()

    def run(self):
        db = SessionLocal()
        try:
            results = self.service.get_top_n(
                self.n, 
                db, 
                signal_filter=self.signal_filter if self.signal_filter != "전체" else None,
                search=self.search if self.search else None
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            db.close()

class EarningsWorker(QThread):
    earnings_ready = Signal(dict)
    
    def __init__(self, tickers: list):
        super().__init__()
        self.tickers = tickers

    def run(self):
        earnings_data = {}
        for t in self.tickers:
            if self.isInterruptionRequested():
                break
            try:
                info = yf.Ticker(t).get_calendar()
                if info and 'Earnings Date' in info:
                    ed = info['Earnings Date']
                    earnings_date = ed[0].date() if hasattr(ed[0], 'date') else None
                else:
                    earnings_date = None
            except Exception:
                earnings_date = None
            earnings_data[t] = earnings_date
        if not self.isInterruptionRequested():
            self.earnings_ready.emit(earnings_data)

class ScreeningPage(QWidget):
    ticker_selected = Signal(str)

    def __init__(self, db_session, screening_svc, export_svc, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.screening_svc = screening_svc
        self.export_svc = export_svc
        self.current_results = []
        self.worker = None
        self.earnings_worker = None
        self._stale_workers = []  # keeps old workers alive until they finish

        self.init_ui()

        # Debounce timer for search
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.refresh_data)

    def init_ui(self):
        self.setObjectName("screening_page")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 24, 30, 24)
        self.main_layout.setSpacing(20)

        # 1. Header & Stats Section
        top_layout = QVBoxLayout()
        
        header_hbox = QHBoxLayout()
        title_vbox = QVBoxLayout()
        title_label = QLabel("AI 시장 스크리닝")
        title_label.setObjectName("page_title")
        subtitle_label = QLabel("DeepLearning 앙상블 기반 실시간 종목 분석 및 투자 신호")
        subtitle_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        title_vbox.addWidget(title_label)
        title_vbox.addWidget(subtitle_label)
        header_hbox.addLayout(title_vbox)
        header_hbox.addStretch()
        
        self.refresh_btn = QPushButton("데이터 갱신")
        self.refresh_btn.setFixedHeight(36)
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh_data)
        header_hbox.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("CSV 내보내기")
        self.export_btn.setFixedHeight(36)
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.clicked.connect(self._on_export_csv)
        header_hbox.addWidget(self.export_btn)

        self.pdf_report_btn = QPushButton("PDF 리포트")
        self.pdf_report_btn.setFixedHeight(36)
        self.pdf_report_btn.setCursor(Qt.PointingHandCursor)
        self.pdf_report_btn.clicked.connect(self._on_export_pdf)
        header_hbox.addWidget(self.pdf_report_btn)

        top_layout.addLayout(header_hbox)
        top_layout.addSpacing(10)

        # Stats Cards Row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)
        
        self.stat_total = self._create_stat_card("분석된 종목수", "0", COLORS['accent'])
        self.stat_avg = self._create_stat_card("평균 AI 점수", "0.0", COLORS['warning'])
        self.stat_strong = self._create_stat_card("강한 매수 종목", "0", COLORS['bull'])
        
        stats_layout.addWidget(self.stat_total)
        stats_layout.addWidget(self.stat_avg)
        stats_layout.addWidget(self.stat_strong)
        top_layout.addLayout(stats_layout)
        
        self.main_layout.addLayout(top_layout)

        # 2. Filter Bar
        filter_bar = QFrame()
        filter_bar.setFixedHeight(60)
        filter_bar.setStyleSheet(f"background-color: {COLORS['bg_surface']}; border: 1px solid {COLORS['border']}; border-radius: 8px;")
        filter_layout = QHBoxLayout(filter_bar)
        filter_layout.setContentsMargins(15, 0, 15, 0)
        filter_layout.setSpacing(15)
        
        search_label = QLabel("SEARCH")
        search_label.setObjectName("section_title")
        filter_layout.addWidget(search_label)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("티커 또는 종목명 입력...")
        self.search_input.setFixedWidth(240)
        self.search_input.textChanged.connect(self.on_search_changed)
        filter_layout.addWidget(self.search_input)
        
        filter_layout.addSpacing(10)
        
        signal_label = QLabel("AI SIGNAL")
        signal_label.setObjectName("section_title")
        filter_layout.addWidget(signal_label)
        
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["전체", "Buy", "Hold", "Sell"])
        self.signal_combo.setFixedWidth(120)
        self.signal_combo.currentTextChanged.connect(self.refresh_data)
        filter_layout.addWidget(self.signal_combo)
        
        filter_layout.addStretch()
        
        self.status_label = QLabel("READY")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: 700; font-size: 11px;")
        filter_layout.addWidget(self.status_label)
        
        self.main_layout.addWidget(filter_bar)

        # 2.5 Legend Row
        legend_layout = QHBoxLayout()
        legend_layout.setContentsMargins(5, 0, 5, 0)
        legend_layout.setSpacing(15)
        
        def add_legend_item(text, tooltip_key):
            item_lay = QHBoxLayout()
            item_lay.setSpacing(4)
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; font-weight: bold;")
            item_lay.addWidget(lbl)
            item_lay.addWidget(HelpButton(tooltip_key))
            legend_layout.addLayout(item_lay)

        add_legend_item("AI 점수", "AI 점수")
        add_legend_item("신호", "신호")
        add_legend_item("방향성", "방향성")
        legend_layout.addStretch()
        
        self.main_layout.addLayout(legend_layout)

        # 3. Table Section
        table_container = QFrame()
        table_container.setObjectName("card")
        table_container.setStyleSheet(f"background-color: {COLORS['bg_surface']}; border-radius: 10px; border: 1px solid {COLORS['border']};")
        table_vbox = QVBoxLayout(table_container)
        table_vbox.setContentsMargins(1, 1, 1, 1)

        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "#", "종목", "AI 종합 점수", "신호", "5일 방향", "패턴", "가격", "변동률", "차트"
        ])
        
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.itemClicked.connect(self.on_item_clicked)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.Fixed)
        self.table.setColumnWidth(8, 80)
        header.setFixedHeight(44)
        
        table_vbox.addWidget(self.table)
        self.main_layout.addWidget(table_container, 1)

        # Loading Indicator
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.setFixedHeight(4)
        self.loading_bar.setVisible(False)
        self.main_layout.addWidget(self.loading_bar)

    def _update_stats_cards(self, results):
        if not results:
            self.stat_total.findChild(QLabel, 'stat_value').setText("0")
            self.stat_avg.findChild(QLabel, 'stat_value').setText("0.0")
            self.stat_strong.findChild(QLabel, 'stat_value').setText("0")
            return

        total_count = len(results)
        avg_score = sum(r['ensemble_score'] for r in results) / max(total_count, 1)
        strong_buys = sum(1 for r in results if r['ensemble_score'] >= 70)
        
        self.stat_total.findChild(QLabel, 'stat_value').setText(str(total_count))
        self.stat_avg.findChild(QLabel, 'stat_value').setText(f"{avg_score:.1f}")
        self.stat_strong.findChild(QLabel, 'stat_value').setText(str(strong_buys))

    def _create_stat_card(self, title, value, color):
        card = QFrame()
        card.setObjectName("card")
        card.setFixedHeight(80)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 12, 20, 12)
        
        title_lbl = QLabel(title)
        title_lbl.setObjectName("section_title")
        
        val_lbl = QLabel(value)
        val_lbl.setObjectName("stat_value")
        val_lbl.setStyleSheet(f"font-size: 24px; font-weight: 800; color: {color};")
        
        layout.addWidget(title_lbl)
        layout.addWidget(val_lbl)
        
        # Store label for updates
        card.val_lbl = val_lbl
        return card

    def _make_score_bar(self, score):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 6, 8, 6)
        
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(score))
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        
        color = get_score_color(score)
        bar.setStyleSheet(f"""
            QProgressBar {{ background: #1A2235; border-radius: 4px; }}
            QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}
        """)
        
        score_lbl = QLabel(f"{score:.0f}")
        score_lbl.setFixedWidth(30)
        score_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        score_lbl.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 14px;")
        
        layout.addWidget(bar, 1)
        layout.addWidget(score_lbl)
        return widget

    def _make_signal_badge(self, signal):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        
        label = QLabel(signal)
        label.setAlignment(Qt.AlignCenter)
        label.setFixedHeight(22)
        label.setContentsMargins(10, 0, 10, 0)
        
        if signal == 'Buy' or '매수' in signal:
            style = "background:#0A3D1F; color:#00E676; border:1px solid #00E67644; border-radius:11px; font-weight:700; font-size:11px;"
        elif signal == 'Sell' or '매도' in signal:
            style = "background:#3D0A0A; color:#FF1744; border:1px solid #FF174444; border-radius:11px; font-weight:700; font-size:11px;"
        else:
            style = "background:#1A2235; color:#78909C; border:1px solid #1E2D45; border-radius:11px; font-weight:700; font-size:11px;"
        label.setStyleSheet(style)
        
        layout.addWidget(label)
        return container

    def on_item_clicked(self, item):
        ticker = self.table.item(item.row(), 1).data(Qt.UserRole)
        self.ticker_selected.emit(ticker)

    def on_search_changed(self):
        self.search_timer.stop()
        self.search_timer.start(300)

    def refresh_data(self):
        # If a worker is already running, ignore this call to avoid double-start
        try:
            if self.worker is not None and self.worker.isRunning():
                return
        except RuntimeError:
            self.worker = None

        self.loading_bar.setVisible(True)
        self.status_label.setText("ANALYZING...")
        self.status_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: 700; font-size: 11px;")

        search = self.search_input.text().strip().upper()
        signal_filter = self.signal_combo.currentText()

        self.worker = ScreeningWorker(n=100, signal_filter=signal_filter, search=search)
        self.worker.finished.connect(self.on_data_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_data_loaded(self, results):
        self.loading_bar.setVisible(False)
        self.current_results = results
        
        if not results:
            self.table.setRowCount(0)
            self.status_label.setText("NO RESULTS")
            self._update_stats_cards([])
            return

        # Update Stats via refactored method
        self._update_stats_cards(results)

        # Optimization: Disable updates during bulk table population
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(0)
            self.table.setRowCount(len(results))
            for i, row in enumerate(results):
                # 1. Rank
                rank_item = QTableWidgetItem(f"{i+1:02d}")
                rank_item.setTextAlignment(Qt.AlignCenter)
                rank_item.setForeground(QColor(COLORS['text_secondary']))
                self.table.setItem(i, 0, rank_item)
                
                # 2. Ticker + Name
                ticker = row['ticker']
                ticker_item = QTableWidgetItem(ticker)
                ticker_item.setData(Qt.UserRole, ticker) # Store for click
                ticker_item.setFont(QFont("SF Pro Display", 11, QFont.Bold))
                ticker_item.setForeground(QColor(COLORS['accent']))
                self.table.setItem(i, 1, ticker_item)
                
                # 3. AI Score
                score_bar = self._make_score_bar(row['ensemble_score'])
                self.table.setCellWidget(i, 2, score_bar)
                
                # 4. Signal
                signal_badge = self._make_signal_badge(row['mlp_signal'])
                self.table.setCellWidget(i, 3, signal_badge)
                
                # 5. Direction
                dir_val = row['lstm_dir_5d']
                dir_item = QTableWidgetItem()
                dir_item.setTextAlignment(Qt.AlignCenter)
                if dir_val == 'Up':
                    dir_item.setText("↗ UP")
                    dir_item.setForeground(QColor(COLORS['bull']))
                elif dir_val == 'Down':
                    dir_item.setText("↘ DOWN")
                    dir_item.setForeground(QColor(COLORS['bear']))
                else:
                    dir_item.setText("→ FLAT")
                    dir_item.setForeground(QColor(COLORS['neutral']))
                dir_item.setFont(QFont("SF Pro Display", 10, QFont.Bold))
                self.table.setItem(i, 4, dir_item)
                
                # 6. Pattern
                pattern_item = QTableWidgetItem(row['cnn_pattern'])
                pattern_item.setTextAlignment(Qt.AlignCenter)
                pattern_item.setForeground(QColor(COLORS['text_primary']))
                self.table.setItem(i, 5, pattern_item)
                
                # 7. Price
                price_item = QTableWidgetItem(f"${row['close_price']:,.2f}")
                price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                price_item.setFont(QFont("Menlo", 11))
                price_item.setForeground(QColor(COLORS['text_primary']))
                self.table.setItem(i, 6, price_item)
                
                # 8. Change %
                change = row['price_change_pct']
                change_text = f"{'+' if change > 0 else ''}{change:.2f}%"
                change_item = QTableWidgetItem(change_text)
                change_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                change_item.setFont(QFont("Menlo", 11, QFont.Bold))
                if change > 0:
                    change_item.setForeground(QColor(COLORS['bull']))
                elif change < 0:
                    change_item.setForeground(QColor(COLORS['bear']))
                else:
                    change_item.setForeground(QColor(COLORS['neutral']))
                self.table.setItem(i, 7, change_item)

                # 9. Chart Preview Placeholder
                chart_btn = QPushButton("차트")
                chart_btn.setFixedWidth(60)
                chart_btn.setFixedHeight(24)
                chart_btn.setCursor(Qt.PointingHandCursor)
                chart_btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {COLORS['bg_surface']};
                        color: {COLORS['accent']};
                        border: 1px solid {COLORS['accent']};
                        border-radius: 4px;
                        font-size: 10px;
                        font-weight: 700;
                    }}
                    QPushButton:hover {{
                        background: {COLORS['accent']};
                        color: white;
                    }}
                """)
                # Connect to same handler as item click
                chart_btn.clicked.connect(lambda checked=False, t=ticker: self.ticker_selected.emit(t))
                
                btn_container = QWidget()
                btn_layout = QHBoxLayout(btn_container)
                btn_layout.setContentsMargins(0, 0, 0, 0)
                btn_layout.setAlignment(Qt.AlignCenter)
                btn_layout.addWidget(chart_btn)
                self.table.setCellWidget(i, 8, btn_container)
        finally:
            self.table.setUpdatesEnabled(True)

        self.status_label.setText("UPDATED")
        self.status_label.setStyleSheet(f"color: {COLORS['bull']}; font-weight: 700; font-size: 11px;")

        # Optimization 4: EarningsWorker launch disabled to save network resources
        # (Worker cleanup logic and launch removed as requested)

    def _on_earnings_ready(self, earnings_dict):
        today = datetime.now().date()
        for i in range(self.table.rowCount()):
            ticker_item = self.table.item(i, 1)
            if not ticker_item:
                continue
            ticker = ticker_item.data(Qt.UserRole)
            
            if ticker in earnings_dict:
                earnings_date = earnings_dict[ticker]
                if earnings_date:
                    days_diff = (earnings_date - today).days
                    if 0 <= days_diff <= 5:
                        # Find the row and add a QLabel badge widget in the ticker column using setCellWidget
                        container = QWidget()
                        layout = QHBoxLayout(container)
                        layout.setContentsMargins(10, 0, 10, 0)
                        layout.setSpacing(6)
                        layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                        
                        # 1. A QLabel with ticker text (bold, navy color)
                        # The prompt says both "navy color" and "Keep the warning color".
                        # Given the warning color is the "active" state for earnings, I will use it.
                        t_label = QLabel(ticker)
                        t_label.setStyleSheet(f"font-weight: bold; color: {COLORS['warning']};")
                        
                        # 2. A small QLabel badge: '📅 D-X' where X = days until earnings
                        # 3. Badge style: background #ff8000 (orange), white text, padding 1px 4px, font-size: 9px
                        badge = QLabel(f"📅 D-{days_diff}")
                        badge.setStyleSheet("""
                            background-color: #ff8000;
                            color: white;
                            padding: 1px 4px;
                            font-size: 9px;
                            border-radius: 2px;
                            font-weight: 700;
                        """)
                        
                        layout.addWidget(t_label)
                        layout.addWidget(badge)
                        
                        # Set tooltip on the container: 'Earnings: YYYY-MM-DD'
                        container.setToolTip(f"Earnings: {earnings_date.strftime('%Y-%m-%d')}")
                        
                        # Clear original text and set the cell widget
                        ticker_item.setText("")
                        self.table.setCellWidget(i, 1, container)
                        
                        # Keep the warning color on the ticker font
                        ticker_item.setForeground(QColor(COLORS['warning']))
                        ticker_item.setFont(QFont("SF Pro Display", 11, QFont.Bold))

    def _on_export_csv(self):
        if not self.current_results:
            return
            
        filepath = self.export_svc.export_screening_csv(self.current_results)
        QMessageBox.information(self, "내보내기 완료", f"저장 완료: {filepath}")

    def _on_export_pdf(self):
        try:
            filepath = self.export_svc.export_pdf_report(self.db_session)
            QMessageBox.information(self, "리포트 생성 완료", f"PDF 리포트가 생성되었습니다.\n경로: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "오류 발생", f"PDF 생성 중 오류가 발생했습니다: {str(e)}")

    def on_error(self, error_msg):
        self.loading_bar.setVisible(False)
        self.status_label.setText("ERROR")
        self.status_label.setStyleSheet(f"color: {COLORS['bear']}; font-weight: 700; font-size: 11px;")
