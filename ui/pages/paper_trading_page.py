from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTabWidget, QTableWidget, QTableWidgetItem,
    QLineEdit, QDoubleSpinBox, QMessageBox, QHeaderView,
    QAbstractItemView
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QFont, QColor, QDoubleValidator
from ui.styles.theme import COLORS, get_score_color
from services.paper_trading_service import PaperTradingService
import datetime

class PaperTradingPage(QWidget):
    def __init__(self, db_session, paper_svc, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.paper_svc = paper_svc  # PaperTradingService (static methods)
        
        self.init_ui()
        self.refresh_data()
        
        # Setup Auto Refresh Timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(30000)  # 30 seconds

    def init_ui(self):
        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 24, 30, 24)
        self.main_layout.setSpacing(24)

        # 1. HEADER ROW
        header_layout = QHBoxLayout()
        
        title_vbox = QVBoxLayout()
        title_hbox = QHBoxLayout()
        self.title_lbl = QLabel("모의 투자")
        self.title_lbl.setObjectName("page_title")
        
        # LIVE indicator
        self.live_indicator = QLabel("● LIVE")
        self.live_indicator.setStyleSheet(f"color: {COLORS['bull']}; font-weight: bold; font-size: 12px; margin-left: 10px; margin-top: 8px;")
        
        # Last update label
        self.last_update_label = QLabel("")
        self.last_update_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; margin-left: 10px; margin-top: 10px;")
        
        title_hbox.addWidget(self.title_lbl)
        title_hbox.addWidget(self.live_indicator)
        title_hbox.addWidget(self.last_update_label)
        title_hbox.addStretch()
        
        self.subtitle_lbl = QLabel("실제 거래 없이 AI 추천 종목으로 투자 연습")
        self.subtitle_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        
        title_vbox.addLayout(title_hbox)
        title_vbox.addWidget(self.subtitle_lbl)
        
        self.reset_btn = QPushButton("계좌 초기화")
        self.reset_btn.setObjectName("secondary")
        self.reset_btn.setFixedWidth(120)
        self.reset_btn.clicked.connect(self._on_reset)

        self.init_cash_input = QLineEdit()
        self.init_cash_input.setPlaceholderText("100,000")
        self.init_cash_input.setFixedWidth(110)
        self.init_cash_input.setValidator(QDoubleValidator(1000.0, 100000000.0, 0))
        init_lbl = QLabel("초기 자금:")
        init_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        init_usd = QLabel("USD")
        init_usd.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")

        right_controls = QHBoxLayout()
        right_controls.setSpacing(6)
        right_controls.addWidget(init_lbl)
        right_controls.addWidget(self.init_cash_input)
        right_controls.addWidget(init_usd)
        right_controls.addSpacing(12)
        right_controls.addWidget(self.reset_btn)

        header_layout.addLayout(title_vbox)
        header_layout.addStretch()
        header_layout.addLayout(right_controls)
        
        self.main_layout.addLayout(header_layout)

        # 2. SUMMARY CARDS ROW
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(16)
        
        self.card_initial = self._create_summary_card("초기 자금", "$100,000", COLORS['accent'])
        self.card_total = self._create_summary_card("현재 자산", "$0.00", COLORS['text_primary'])
        self.card_pnl = self._create_summary_card("총 수익", "$0.00 (0.0%)", COLORS['bull'])
        self.card_cash = self._create_summary_card("현금 잔액", "$0.00", COLORS['accent'])
        
        # Store labels for updating
        self.lbl_total_value = self.card_total.findChild(QLabel, "value_label")
        self.lbl_pnl = self.card_pnl.findChild(QLabel, "value_label")
        self.lbl_cash = self.card_cash.findChild(QLabel, "value_label")
        
        summary_layout.addWidget(self.card_initial)
        summary_layout.addWidget(self.card_total)
        summary_layout.addWidget(self.card_pnl)
        summary_layout.addWidget(self.card_cash)
        
        self.main_layout.addLayout(summary_layout)

        # 3. MAIN CONTENT ROW
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # LEFT SIDE - Tabs
        left_vbox = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # Tab 1: 보유 종목
        self.positions_table = QTableWidget()
        self._setup_table(self.positions_table, [
            "종목", "수량", "평균단가", "현재가", "평가금액", "손익", "손익률", "매도"
        ])
        
        # Tab 2: 거래 내역
        self.transactions_table = QTableWidget()
        self._setup_table(self.transactions_table, [
            "시간", "구분", "종목", "수량", "단가", "거래금액"
        ])
        
        self.tabs.addTab(self.positions_table, "보유 종목")
        self.tabs.addTab(self.transactions_table, "거래 내역")
        
        left_vbox.addWidget(self.tabs)
        content_layout.addLayout(left_vbox, stretch=5)

        # RIGHT SIDE - Trade Form & Quick Pick
        right_vbox = QVBoxLayout()
        right_vbox.setSpacing(20)

        # Trade Form Card
        self.trade_card = QFrame()
        self.trade_card.setObjectName("card")
        trade_layout = QVBoxLayout(self.trade_card)
        trade_layout.setContentsMargins(20, 20, 20, 20)
        trade_layout.setSpacing(15)

        trade_title = QLabel("AI 추천 주문")
        trade_title.setObjectName("section_title")
        trade_layout.addWidget(trade_title)

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("종목 티커 (예: AAPL)")
        self.ticker_input.textChanged.connect(self._on_ticker_changed)
        trade_layout.addWidget(self.ticker_input)

        qty_layout = QHBoxLayout()
        qty_label = QLabel("수량")
        self.qty_input = QDoubleSpinBox()
        self.qty_input.setRange(0.01, 9999.0)
        self.qty_input.setDecimals(2)
        self.qty_input.setValue(1.0)
        self.qty_input.valueChanged.connect(self._on_qty_changed)
        qty_layout.addWidget(qty_label)
        qty_layout.addWidget(self.qty_input)
        trade_layout.addLayout(qty_layout)

        self.price_label = QLabel("현재가: -")
        self.price_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        trade_layout.addWidget(self.price_label)

        self.cost_label = QLabel("예상 금액: -")
        self.cost_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        trade_layout.addWidget(self.cost_label)

        self.buy_btn = QPushButton("매수 (BUY)")
        self.buy_btn.setStyleSheet(f"background-color: {COLORS['bull']}; color: black; font-weight: bold;")
        self.buy_btn.clicked.connect(self._on_buy)
        trade_layout.addWidget(self.buy_btn)

        self.sell_btn = QPushButton("매도 (SELL)")
        self.sell_btn.setStyleSheet(f"background-color: {COLORS['bear']}; color: white; font-weight: bold;")
        self.sell_btn.clicked.connect(self._on_sell)
        trade_layout.addWidget(self.sell_btn)
        
        trade_layout.addStretch()
        right_vbox.addWidget(self.trade_card)

        # Quick Pick Card
        self.pick_card = QFrame()
        self.pick_card.setObjectName("card")
        pick_layout = QVBoxLayout(self.pick_card)
        pick_layout.setContentsMargins(20, 20, 20, 20)
        pick_layout.setSpacing(10)

        pick_title = QLabel("AI 추천 종목")
        pick_title.setObjectName("section_title")
        pick_layout.addWidget(pick_title)

        quick_stocks = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
        for stock in quick_stocks:
            btn = QPushButton(stock)
            btn.setObjectName("secondary")
            btn.clicked.connect(lambda checked, s=stock: self.ticker_input.setText(s))
            pick_layout.addWidget(btn)
            
        pick_layout.addStretch()
        right_vbox.addWidget(self.pick_card)

        content_layout.addLayout(right_vbox, stretch=3)
        self.main_layout.addLayout(content_layout)

    def _setup_table(self, table, headers):
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setHighlightSections(False)
        table.setStyleSheet(f"QTableWidget {{ background-color: {COLORS['bg_surface']}; }}")

    def _create_summary_card(self, title, value, color):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        
        t_lbl = QLabel(title)
        t_lbl.setObjectName("section_title")
        
        v_lbl = QLabel(value)
        v_lbl.setObjectName("value_label")
        v_lbl.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        
        layout.addWidget(t_lbl)
        layout.addWidget(v_lbl)
        return card

    def refresh_data(self):
        summary = self.paper_svc.get_portfolio_summary(self.db_session)
        positions = self.paper_svc.get_positions(self.db_session)
        transactions = self.paper_svc.get_transactions(self.db_session)
        
        # Update Summary
        self.lbl_total_value.setText(f"${summary['total_value']:,.2f}")
        
        pnl_color = COLORS['bull'] if summary['total_pnl'] >= 0 else COLORS['bear']
        self.lbl_pnl.setText(f"${summary['total_pnl']:,.2f} ({summary['pnl_pct']:.2f}%)")
        self.lbl_pnl.setStyleSheet(f"color: {pnl_color}; font-size: 18px; font-weight: bold;")
        
        self.lbl_cash.setText(f"${summary['cash']:,.2f}")
        
        # Update Tables
        self._update_positions_table(positions)
        self._update_transactions_table(transactions)
        
        # Update last refresh time
        self.last_update_label.setText(f"마지막 갱신: {datetime.datetime.now().strftime('%H:%M:%S')}")

    def _update_positions_table(self, positions):
        self.positions_table.setRowCount(0)
        for i, pos in enumerate(positions):
            self.positions_table.insertRow(i)
            self.positions_table.setRowHeight(i, 44)
            
            # Columns: ['종목', '수량', '평균단가', '현재가', '평가금액', '손익', '손익률', '매도']
            items = [
                pos['ticker'],
                f"{pos['quantity']:.2f}",
                f"${pos['avg_cost']:,.2f}",
                f"${pos['current_price']:,.2f}",
                f"${pos['current_value']:,.2f}",
                f"${pos['pnl']:,.2f}",
                f"{pos['pnl_pct']:.2f}%"
            ]
            
            for j, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if j == 5 or j == 6: # PnL columns
                    color = COLORS['bull'] if pos['pnl'] >= 0 else COLORS['bear']
                    item.setForeground(QColor(color))
                self.positions_table.setItem(i, j, item)
            
            # Sell Button
            sell_btn = QPushButton("매도")
            sell_btn.setObjectName("secondary")
            sell_btn.setFixedSize(60, 28)
            sell_btn.clicked.connect(lambda checked, t=pos['ticker'], q=pos['quantity']: self._quick_sell(t, q))
            
            btn_container = QWidget()
            btn_layout = QHBoxLayout(btn_container)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.setAlignment(Qt.AlignCenter)
            btn_layout.addWidget(sell_btn)
            self.positions_table.setCellWidget(i, 7, btn_container)

    def _update_transactions_table(self, txns):
        self.transactions_table.setRowCount(0)
        for i, tx in enumerate(txns):
            self.transactions_table.insertRow(i)
            self.transactions_table.setRowHeight(i, 44)
            
            # Columns: ['시간', '구분', '종목', '수량', '단가', '거래금액']
            timestamp = tx['timestamp'].strftime("%m-%d %H:%M") if isinstance(tx['timestamp'], datetime.datetime) else str(tx['timestamp'])
            
            items = [
                timestamp,
                tx['action'],
                tx['ticker'],
                f"{tx['quantity']:.2f}",
                f"${tx['price']:,.2f}",
                f"${tx['total']:,.2f}"
            ]
            
            for j, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if j == 1: # Action column
                    color = COLORS['bull'] if tx['action'] == 'BUY' else COLORS['bear']
                    item.setForeground(QColor(color))
                self.transactions_table.setItem(i, j, item)

    def _on_ticker_changed(self, text):
        text = text.upper().strip()
        if len(text) >= 2:
            price = self.paper_svc.get_current_price(text)
            if price > 0:
                self.price_label.setText(f"현재가: ${price:,.2f}")
                self._update_cost_label(price)
            else:
                self.price_label.setText("현재가: N/A")
                self.cost_label.setText("예상 금액: -")
        else:
            self.price_label.setText("현재가: -")
            self.cost_label.setText("예상 금액: -")

    def _on_qty_changed(self, value):
        ticker = self.ticker_input.text().upper().strip()
        if len(ticker) >= 2:
            price_text = self.price_label.text()
            if "$" in price_text:
                try:
                    price = float(price_text.split("$")[1].replace(",", ""))
                    self._update_cost_label(price)
                except Exception:
                    pass

    def _update_cost_label(self, price):
        qty = self.qty_input.value()
        total = qty * price
        self.cost_label.setText(f"예상 금액: ${total:,.2f}")

    def _on_buy(self):
        ticker = self.ticker_input.text().upper().strip()
        qty = self.qty_input.value()
        if not ticker:
            return
            
        result = self.paper_svc.buy(ticker, qty, self.db_session)
        if result['success']:
            QMessageBox.information(self, "매수 성공", f"{ticker} {qty}주 매수 완료\n총액: ${result['total']:,.2f}")
            self.refresh_data()
        else:
            QMessageBox.warning(self, "매수 실패", result.get('error', '알 수 없는 오류'))

    def _on_sell(self):
        ticker = self.ticker_input.text().upper().strip()
        qty = self.qty_input.value()
        if not ticker:
            return
            
        result = self.paper_svc.sell(ticker, qty, self.db_session)
        if result['success']:
            QMessageBox.information(self, "매도 성공", f"{ticker} {qty}주 매도 완료\n총액: ${result['total']:,.2f}")
            self.refresh_data()
        else:
            QMessageBox.warning(self, "매도 실패", result.get('error', '알 수 없는 오류'))

    def _quick_sell(self, ticker, quantity):
        self.ticker_input.setText(ticker)
        self.qty_input.setValue(quantity)
        self._on_sell()

    def _on_reset(self):
        cash_text = self.init_cash_input.text().replace(',', '').strip()
        new_cash = float(cash_text) if cash_text else 100000.0
        amount_str = f"${new_cash:,.0f}"
        reply = QMessageBox.question(
            self, "계좌 초기화",
            f"초기 자금 {amount_str} USD로 초기화합니다.\n모든 보유 종목과 거래 내역이 삭제됩니다.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.paper_svc.reset_account(self.db_session, new_initial_cash=new_cash)
            v = self.card_initial.findChild(QLabel, "value_label")
            if v:
                v.setText(amount_str)
            self.refresh_data()
            QMessageBox.information(self, "초기화 완료", f"계좌가 {amount_str} USD로 초기화되었습니다.")
