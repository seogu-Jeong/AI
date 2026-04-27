from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSplitter, QProgressBar, QDoubleSpinBox,
    QMessageBox, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor, QFont

from ui.styles.theme import COLORS
from ui.widgets.candle_chart import CandleChart


class CryptoWorker(QObject):
    finished = Signal(list)
    error = Signal(str)

    def run(self):
        try:
            from services.crypto_service import CryptoService
            results = CryptoService.get_all_scores()
            self.finished.emit(results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class CryptoPage(QWidget):
    def __init__(self, db_session, paper_svc, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.paper_svc = paper_svc
        self._coin_data = {}
        self._thread = None
        self._worker = None
        self._init_ui()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet('QScrollArea { border: none; background: #c0c0c0; }')
        outer.addWidget(scroll)

        content = QWidget()
        content.setStyleSheet('background-color: #c0c0c0;')
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(16)
        scroll.setWidget(content)

        # --- HEADER ---
        title = QLabel('코인 분석')
        title.setObjectName('page_title')
        subtitle = QLabel('주요 암호화폐 기술적 AI 점수 및 모의투자')
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # --- CONTROLS ---
        ctrl_row = QHBoxLayout()
        self.refresh_btn = QPushButton('데이터 갱신')
        self.refresh_btn.setMinimumWidth(110)
        self.refresh_btn.clicked.connect(self.refresh_data)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(160)
        self.status_lbl = QLabel('')
        self.status_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        ctrl_row.addWidget(self.refresh_btn)
        ctrl_row.addWidget(self.progress)
        ctrl_row.addWidget(self.status_lbl)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # --- SPLITTER: table left, detail right ---
        splitter = QSplitter(Qt.Horizontal)

        # LEFT: Coin table
        table_frame = QFrame()
        table_frame.setStyleSheet('background-color: #ffffff; border: 1px solid #808080;')
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(['코인', '현재가(USD)', '24H 변동', 'AI 점수', '신호', '지표'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #ffffff; font-size: 12px; }
            QTableWidget::item:selected { background-color: #000080; color: #ffffff; }
            QHeaderView::section { background-color: #c0c0c0; font-weight: bold; font-size: 11px; padding: 4px; border: 1px solid #808080; }
        """)
        self.table.selectionModel().selectionChanged.connect(self._on_coin_selected)
        table_layout.addWidget(self.table)
        splitter.addWidget(table_frame)

        # RIGHT: Detail panel
        detail_frame = QFrame()
        detail_frame.setStyleSheet('background-color: #f0f0f0; border: 1px solid #808080;')
        detail_layout = QVBoxLayout(detail_frame)
        detail_layout.setContentsMargins(12, 12, 12, 12)
        detail_layout.setSpacing(10)

        # Coin name header
        self.detail_name = QLabel('코인을 선택하세요')
        self.detail_name.setStyleSheet('font-size: 18px; font-weight: 800; color: #000080;')
        detail_layout.addWidget(self.detail_name)

        # Score row
        score_row = QHBoxLayout()
        self.detail_score_lbl = QLabel('AI 점수')
        self.detail_score_lbl.setStyleSheet('font-size: 11px; color: #555555;')
        self.detail_score_val = QLabel('--')
        self.detail_score_val.setStyleSheet('font-size: 28px; font-weight: 800; color: #000080;')
        self.detail_signal = QLabel('--')
        self.detail_signal.setStyleSheet('font-size: 14px; font-weight: 700; padding: 4px 12px; background: #c0c0c0; border: 1px solid #808080;')
        score_row.addWidget(self.detail_score_lbl)
        score_row.addWidget(self.detail_score_val)
        score_row.addSpacing(12)
        score_row.addWidget(self.detail_signal)
        score_row.addStretch()
        detail_layout.addLayout(score_row)

        # Indicators grid
        ind_frame = QFrame()
        ind_frame.setStyleSheet('background-color: #ffffff; border: 1px solid #c0c0c0; padding: 4px;')
        ind_grid = QHBoxLayout(ind_frame)
        ind_grid.setSpacing(16)
        self._ind_labels = {}
        for key, display in [('RSI', 'RSI(14)'), ('MACD', 'MACD'), ('BB%B', 'BB%B'), ('Stoch', 'Stoch%K'), ('1M', '1M 수익률')]:
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(display)
            lbl.setStyleSheet('font-size: 10px; color: #555555;')
            val = QLabel('--')
            val.setStyleSheet('font-size: 13px; font-weight: 700; color: #000000;')
            col.addWidget(lbl)
            col.addWidget(val)
            self._ind_labels[key] = val
            ind_grid.addLayout(col)
        detail_layout.addWidget(ind_frame)

        # AI 방향 예측 panel
        pred_lbl = QLabel('AI 방향 예측 (단타)')
        pred_lbl.setStyleSheet('font-size: 11px; font-weight: 700; color: #000080; margin-top: 6px;')
        detail_layout.addWidget(pred_lbl)

        pred_frame = QFrame()
        pred_frame.setStyleSheet('background-color: #ffffff; border: 1px solid #c0c0c0; padding: 4px;')
        pred_row = QHBoxLayout(pred_frame)
        pred_row.setSpacing(12)
        self._pred_labels = {}
        for horizon, key in [('5일', '5d'), ('20일', '20d'), ('60일', '60d')]:
            col = QVBoxLayout()
            col.setSpacing(2)
            h_lbl = QLabel(horizon)
            h_lbl.setStyleSheet('font-size: 10px; color: #555555;')
            h_lbl.setAlignment(Qt.AlignCenter)
            v_lbl = QLabel('--')
            v_lbl.setStyleSheet('font-size: 14px; font-weight: 800; color: #000000;')
            v_lbl.setAlignment(Qt.AlignCenter)
            col.addWidget(h_lbl)
            col.addWidget(v_lbl)
            self._pred_labels[key] = v_lbl
            pred_row.addLayout(col)
        detail_layout.addWidget(pred_frame)

        # Chart
        chart_lbl = QLabel('캔들 차트 (6개월)')
        chart_lbl.setStyleSheet('font-size: 11px; font-weight: 700; color: #000080; margin-top: 4px;')
        detail_layout.addWidget(chart_lbl)
        self.chart = CandleChart()
        self.chart.setMinimumHeight(300)
        self.chart.setMaximumHeight(360)
        detail_layout.addWidget(self.chart)

        # Paper trading panel
        trade_frame = QFrame()
        trade_frame.setStyleSheet('background-color: #ffffff; border: 1px solid #808080;')
        trade_layout = QVBoxLayout(trade_frame)
        trade_layout.setContentsMargins(12, 10, 12, 10)
        trade_layout.setSpacing(8)

        trade_title = QLabel('모의투자')
        trade_title.setStyleSheet('font-size: 12px; font-weight: 700; color: #000080;')
        trade_layout.addWidget(trade_title)

        qty_row = QHBoxLayout()
        qty_lbl = QLabel('수량:')
        qty_lbl.setStyleSheet('font-size: 11px;')
        self.qty_spin = QDoubleSpinBox()
        self.qty_spin.setRange(0.0001, 1000.0)
        self.qty_spin.setSingleStep(0.001)
        self.qty_spin.setDecimals(4)
        self.qty_spin.setValue(0.01)
        self.qty_spin.setFixedWidth(110)
        self.price_lbl = QLabel('현재가: --')
        self.price_lbl.setStyleSheet('font-size: 11px; color: #555555;')
        qty_row.addWidget(qty_lbl)
        qty_row.addWidget(self.qty_spin)
        qty_row.addSpacing(10)
        qty_row.addWidget(self.price_lbl)
        qty_row.addStretch()
        trade_layout.addLayout(qty_row)

        btn_row = QHBoxLayout()
        self.buy_btn = QPushButton('매수 (BUY)')
        self.buy_btn.setStyleSheet('background-color: #006400; color: #ffffff; font-weight: bold; padding: 6px 20px; border: 2px solid #004000;')
        self.buy_btn.setEnabled(False)
        self.buy_btn.clicked.connect(self._on_buy)
        self.sell_btn = QPushButton('매도 (SELL)')
        self.sell_btn.setStyleSheet('background-color: #8b0000; color: #ffffff; font-weight: bold; padding: 6px 20px; border: 2px solid #5a0000;')
        self.sell_btn.setEnabled(False)
        self.sell_btn.clicked.connect(self._on_sell)
        btn_row.addWidget(self.buy_btn)
        btn_row.addWidget(self.sell_btn)
        btn_row.addStretch()
        trade_layout.addLayout(btn_row)

        self.trade_status = QLabel('')
        self.trade_status.setStyleSheet('font-size: 11px; color: #555555;')
        trade_layout.addWidget(self.trade_status)

        detail_layout.addWidget(trade_frame)
        detail_layout.addStretch()
        splitter.addWidget(detail_frame)

        splitter.setSizes([480, 600])
        layout.addWidget(splitter, 1)

    def refresh_data(self):
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    return
            except RuntimeError:
                pass

        self.refresh_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_lbl.setText('코인 데이터 수집 중...')

        self._thread = QThread()
        self._worker = CryptoWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_data_ready)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._on_error)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    def _on_thread_done(self):
        self._thread = None
        self._worker = None

    def _on_error(self, msg):
        self.refresh_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_lbl.setText(f'오류: {msg}')

    def _on_data_ready(self, results: list):
        self.refresh_btn.setEnabled(True)
        self.progress.setVisible(False)
        from datetime import datetime
        self.status_lbl.setText(f"갱신: {datetime.now().strftime('%H:%M:%S')} | {len(results)}개 코인")

        self._coin_data = {d['ticker']: d for d in results}
        self.table.setRowCount(0)

        for d in results:
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Col 0: name + symbol
            name_item = QTableWidgetItem(f"{d['symbol']}  {d['name']}")
            name_item.setFont(QFont('', -1, QFont.Weight.Bold))
            name_item.setData(Qt.UserRole, d['ticker'])
            self.table.setItem(row, 0, name_item)

            # Col 1: price
            price = d['price']
            if price >= 1000:
                price_str = f"${price:,.0f}"
            elif price >= 1:
                price_str = f"${price:,.2f}"
            else:
                price_str = f"${price:.4f}"
            self.table.setItem(row, 1, QTableWidgetItem(price_str))

            # Col 2: 24H change
            chg = d['change_pct']
            chg_item = QTableWidgetItem(f"{chg:+.2f}%")
            chg_item.setForeground(QColor(COLORS['bull']) if chg >= 0 else QColor(COLORS['bear']))
            chg_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, chg_item)

            # Col 3: AI score
            score = d['score']
            score_item = QTableWidgetItem(f"{score:.1f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            if score >= 70:
                score_item.setForeground(QColor(COLORS['bull']))
            elif score >= 50:
                score_item.setForeground(QColor(COLORS['warning']))
            else:
                score_item.setForeground(QColor(COLORS['bear']))
            self.table.setItem(row, 3, score_item)

            # Col 4: signal
            sig = d['signal']
            sig_item = QTableWidgetItem(sig)
            sig_item.setTextAlignment(Qt.AlignCenter)
            if sig == 'BUY':
                sig_item.setForeground(QColor(COLORS['bull']))
                sig_item.setBackground(QColor('#e8f5e9'))
            elif sig == 'SELL':
                sig_item.setForeground(QColor(COLORS['bear']))
                sig_item.setBackground(QColor('#ffebee'))
            else:
                sig_item.setForeground(QColor(COLORS['neutral']))
            self.table.setItem(row, 4, sig_item)

            # Col 5: RSI indicator
            rsi_item = QTableWidgetItem(f"RSI {d['rsi']:.0f}")
            rsi_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 5, rsi_item)

    def _on_coin_selected(self):
        rows = self.table.selectedItems()
        if not rows:
            return
        row = self.table.currentRow()
        ticker_item = self.table.item(row, 0)
        if ticker_item is None:
            return
        ticker = ticker_item.data(Qt.UserRole)
        if ticker not in self._coin_data:
            return

        d = self._coin_data[ticker]
        self.detail_name.setText(f"[{d['symbol']}] {d['name']}")

        score = d['score']
        self.detail_score_val.setText(f"{score:.1f}")
        color = COLORS['bull'] if score >= 70 else (COLORS['warning'] if score >= 50 else COLORS['bear'])
        self.detail_score_val.setStyleSheet(f'font-size: 28px; font-weight: 800; color: {color};')

        sig = d['signal']
        sig_color = COLORS['bull'] if sig == 'BUY' else (COLORS['bear'] if sig == 'SELL' else COLORS['neutral'])
        self.detail_signal.setText(sig)
        self.detail_signal.setStyleSheet(f'font-size: 14px; font-weight: 700; padding: 4px 12px; color: {sig_color}; background: #ffffff; border: 2px solid {sig_color};')

        self._ind_labels['RSI'].setText(f"{d['rsi']:.1f}")
        self._ind_labels['MACD'].setText(f"{d['macd']:.2f}")
        self._ind_labels['BB%B'].setText(f"{d['bb_pct']:.2f}")
        self._ind_labels['Stoch'].setText(f"{d['stoch']:.1f}")
        self._ind_labels['1M'].setText(f"{d['mom_1m']:+.1f}%")

        # Update AI predictions
        DIR_TEXT = {'UP': '▲ 상승', 'FLAT': '→ 횡보', 'DOWN': '▼ 하락'}
        DIR_COLOR = {'UP': '#006400', 'FLAT': '#555555', 'DOWN': '#cc0000'}
        for key, field in [('5d', 'lstm_dir_5d'), ('20d', 'lstm_dir_20d'), ('60d', 'lstm_dir_60d')]:
            direction = d.get(field, 'FLAT')
            lbl = self._pred_labels[key]
            lbl.setText(DIR_TEXT.get(direction, direction))
            lbl.setStyleSheet(f'font-size: 14px; font-weight: 800; color: {DIR_COLOR.get(direction, "#555555")};')

        price = d['price']
        if price >= 1000:
            self.price_lbl.setText(f"현재가: ${price:,.0f}")
        elif price >= 1:
            self.price_lbl.setText(f"현재가: ${price:,.2f}")
        else:
            self.price_lbl.setText(f"현재가: ${price:.4f}")

        # Plot chart (set period to 6M = All for crypto)
        history = d.get('history')
        if history is not None and not history.empty:
            self.chart._active_period = 'All'
            self.chart.plot(history, ticker, forecast=d.get('forecast'))

        self.buy_btn.setEnabled(True)
        self.sell_btn.setEnabled(True)
        self.buy_btn.setText(f'매수 {d["symbol"]}')
        self.sell_btn.setText(f'매도 {d["symbol"]}')
        self.trade_status.setText('')

    def _on_buy(self):
        row = self.table.currentRow()
        if row < 0:
            return
        ticker_item = self.table.item(row, 0)
        if ticker_item is None:
            return
        ticker = ticker_item.data(Qt.UserRole)
        qty = self.qty_spin.value()
        try:
            result = self.paper_svc.buy(ticker, qty, self.db_session)
            if result.get('success'):
                self.trade_status.setText(f"✓ 매수 완료: {qty:.4f} @ ${result['price']:,.2f}")
                self.trade_status.setStyleSheet(f'font-size: 11px; color: {COLORS["bull"]};')
            else:
                self.trade_status.setText(f"✗ {result.get('error', '오류')}")
                self.trade_status.setStyleSheet(f'font-size: 11px; color: {COLORS["bear"]};')
        except Exception as e:
            self.trade_status.setText(f"✗ {e}")
            self.trade_status.setStyleSheet(f'font-size: 11px; color: {COLORS["bear"]};')

    def _on_sell(self):
        row = self.table.currentRow()
        if row < 0:
            return
        ticker_item = self.table.item(row, 0)
        if ticker_item is None:
            return
        ticker = ticker_item.data(Qt.UserRole)
        qty = self.qty_spin.value()
        try:
            result = self.paper_svc.sell(ticker, qty, self.db_session)
            if result.get('success'):
                self.trade_status.setText(f"✓ 매도 완료: {qty:.4f} @ ${result['price']:,.2f}")
                self.trade_status.setStyleSheet(f'font-size: 11px; color: {COLORS["bull"]};')
            else:
                self.trade_status.setText(f"✗ {result.get('error', '오류')}")
                self.trade_status.setStyleSheet(f'font-size: 11px; color: {COLORS["bear"]};')
        except Exception as e:
            self.trade_status.setText(f"✗ {e}")
            self.trade_status.setStyleSheet(f'font-size: 11px; color: {COLORS["bear"]};')

    def refresh(self):
        self.refresh_data()
