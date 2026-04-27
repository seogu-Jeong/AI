from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget, QLabel, QFrame, QPushButton,
    QSystemTrayIcon, QMenu, QProgressBar, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize
from PySide6.QtGui import QIcon, QFont, QColor, QAction, QLinearGradient, QBrush, QPalette

from ui.styles.theme import COLORS

class MainWindow(QMainWindow):
    def __init__(self, db_session):
        super().__init__()
        self.db_session = db_session
        self._loaded_pages = {}
        self._visited_pages = set()
        self._init_services()
        self._build_ui()
        self._setup_tray()
        
        # Start background data check timer (every 30 min)
        self.alert_timer = QTimer(self)
        self.alert_timer.timeout.connect(self._check_alerts)
        self.alert_timer.start(30 * 60 * 1000) 
        
    def _init_services(self):
        from services.screening_service import ScreeningService
        from services.portfolio_service import PortfolioService  
        from services.alert_service import AlertService
        from services.export_service import ExportService
        from services.backtest_service import BacktestService
        from services.paper_trading_service import PaperTradingService
        self.screening_svc = ScreeningService()
        self.portfolio_svc = PortfolioService()
        self.alert_svc = AlertService()
        self.export_svc = ExportService()
        self.backtest_svc = BacktestService()
        self.paper_svc = PaperTradingService()
        
    def _build_ui(self):
        self.setWindowTitle('SWS')
        self.setMinimumSize(1280, 800)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # TOP BAR
        top_bar = self._create_top_bar()
        
        # CONTENT AREA
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # SIDEBAR
        self.sidebar = self._create_sidebar()
        
        # STACKED WIDGET
        self.stack = QStackedWidget()
        self._create_pages()
        
        content_layout.addWidget(self.sidebar)
        content_layout.addWidget(self.stack, 1)

        # Connect sidebar AFTER stack is created
        self.sidebar.currentRowChanged.connect(self._on_nav_changed)
        self.sidebar.setCurrentRow(0)

        # DISCLAIMER BANNER
        disclaimer = QLabel('⚠️ AI 점수는 투자 조언이 아닙니다. 모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.')
        disclaimer.setObjectName('disclaimer')
        disclaimer.setAlignment(Qt.AlignCenter)
        disclaimer.setFixedHeight(30)
        
        main_layout.addWidget(top_bar)
        main_layout.addWidget(content, 1)
        main_layout.addWidget(disclaimer)
        
    def _create_top_bar(self) -> QWidget:
        top_bar = QFrame()
        top_bar.setFixedHeight(56)
        top_bar.setObjectName("top_bar")
        top_bar.setStyleSheet(f"""
            QFrame#top_bar {{
                background-color: #c0c0c0;
                border-top: 2px solid #ffffff;
                border-bottom: 2px solid #808080;
            }}
        """)
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(15)
        
        # Left: Logo & Badge
        logo_layout = QHBoxLayout()
        logo_label = QLabel("◈ SWS")
        logo_label.setStyleSheet("font-size: 18px; font-weight: 800; color: #000080; letter-spacing: -0.5px;")
        logo_layout.addWidget(logo_label)
        
        version_badge = QLabel("v2.0")
        version_badge.setStyleSheet("""
            background-color: #c0c0c0;
            color: #000000;
            font-size: 10px;
            font-weight: bold;
            padding: 2px 6px;
            border: 1px solid #808080;
        """)
        logo_layout.addWidget(version_badge)
        layout.addLayout(logo_layout)

        # Market Ticker Strip
        ticker_frame = QFrame()
        ticker_frame.setFixedWidth(500)
        ticker_layout = QHBoxLayout(ticker_frame)
        ticker_layout.setContentsMargins(20, 0, 20, 0)
        ticker_layout.setSpacing(20)

        market_data = [
            ("SPY", "588.24", "+0.82%", True),
            ("QQQ", "502.11", "+1.15%", True),
            ("DIA", "432.50", "-0.12%", False),
            ("VIX", "14.22", "-4.50%", True),  # VIX down is usually green/bullish for market
            ("GOLD", "2742.10", "+0.45%", True),
            ("BTC", "92,431", "+2.30%", True)
        ]

        for symbol, price, change, is_bull in market_data:
            item_label = QLabel()
            color = COLORS['bull'] if is_bull else COLORS['bear']
            arrow = "▲" if is_bull else "▼"
            item_label.setText(f"{symbol} <span style='color: #000000;'>{price}</span> <span style='color: {color};'>{arrow}{change}</span>")
            item_label.setStyleSheet(f"font-family: 'Menlo', 'Courier New', 'Apple SD Gothic Neo', monospace; font-size: 11px; color: #000000;")
            ticker_layout.addWidget(item_label)

        layout.addWidget(ticker_frame)
        
        layout.addStretch()
        
        # Center: Status Pill
        status_pill = QFrame()
        status_pill.setStyleSheet(f"""
            background-color: #ccffcc;
            border: 1px solid #008000;
            border-radius: 0px;
        """)
        status_pill.setFixedHeight(28)
        pill_layout = QHBoxLayout(status_pill)
        pill_layout.setContentsMargins(12, 0, 12, 0)
        
        status_dot = QLabel("●")
        status_dot.setStyleSheet(f"color: #008000; font-size: 10px;")
        pill_layout.addWidget(status_dot)
        
        status_text = QLabel("S&P500 | 503 종목 | AI 분석 완료")
        status_text.setStyleSheet(f"color: #008000; font-size: 11px; font-weight: 600;")
        pill_layout.addWidget(status_text)
        layout.addWidget(status_pill)
        
        layout.addStretch()
        
        # Right: Time, Refresh, Settings
        from datetime import datetime
        self.update_label = QLabel(datetime.now().strftime('%H:%M'))
        self.update_label.setStyleSheet(f"color: #000000; font-size: 12px; font-weight: 600;")
        layout.addWidget(self.update_label)
        
        sep = QFrame()
        sep.setFixedSize(1, 16)
        sep.setStyleSheet(f"background-color: #808080;")
        layout.addWidget(sep)
        
        refresh_btn = QPushButton("데이터 갱신")
        refresh_btn.setObjectName("secondary")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setFixedHeight(32)
        refresh_btn.clicked.connect(self._refresh_all_data)
        layout.addWidget(refresh_btn)
        
        settings_btn = QPushButton("⚙")
        settings_btn.setObjectName("ghost")
        settings_btn.setFixedSize(32, 32)
        settings_btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(settings_btn)
        
        return top_bar
        
    def _create_sidebar(self) -> QListWidget:
        sidebar = QListWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
        sidebar.setCursor(Qt.PointingHandCursor)
        # Style is now handled by get_stylesheet()
        
        items = [
            "📊  스크리닝",
            "🔍  종목 상세",
            "💼  포트폴리오",
            "📰  뉴스/감성",
            "⭐  관심 종목",
            "📈  백테스팅",
            "🌐  섹터 분석",
            "📅  경제 캘린더",
            "🎯  모의 투자",
            "⚖️  종목 비교",
            "₿  코인 분석",
            "⚙️  설정"
        ]
        
        for text in items:
            item = QListWidgetItem(text)
            sidebar.addItem(item)
            
        # Connect signal AFTER stack is ready (done in _build_ui)
        # sidebar.currentRowChanged.connect(self._on_nav_changed)
        # sidebar.setCurrentRow(0) -- deferred

        return sidebar
        
    def _create_pages(self):
        # Add 12 placeholder widgets first
        self._placeholders = []
        for _ in range(12):
            ph = QWidget()
            self.stack.addWidget(ph)
            self._placeholders.append(ph)
        
        self._page_factories = {
            0: lambda: self._make_screening_page(),
            1: lambda: self._make_detail_page(),
            2: lambda: self._make_portfolio_page(),
            3: lambda: self._make_news_page(),
            4: lambda: self._make_watchlist_page(),
            5: lambda: self._make_backtest_page(),
            6: lambda: self._make_sector_page(),
            7: lambda: self._make_macro_page(),
            8: lambda: self._make_paper_page(),
            9: lambda: self._make_compare_page(),
            10: lambda: self._make_crypto_page(),
            11: lambda: self._make_settings_page(),
        }
        
        # Pre-load first two pages only
        self._ensure_page(0)
        self._ensure_page(1)

    def _ensure_page(self, index):
        if index not in self._loaded_pages:
            page = self._page_factories[index]()
            self._loaded_pages[index] = page
            # Replace placeholder
            self.stack.removeWidget(self._placeholders[index])
            self._placeholders[index].deleteLater()
            self.stack.insertWidget(index, page)
            self._wire_page_signals(index, page)
        return self._loaded_pages[index]

    def _wire_page_signals(self, index, page):
        if index == 0: # ScreeningPage
            page.ticker_selected.connect(self._on_ticker_selected)
        elif index == 1: # DetailPage
            page.back_requested.connect(self._on_back_to_screening)
            page.watchlist_add_requested.connect(self._on_add_to_watchlist)
        elif index == 4: # WatchlistPage
            page.ticker_selected.connect(self._on_ticker_selected)

    def _make_screening_page(self):
        from ui.pages.screening_page import ScreeningPage
        self.screening_page = ScreeningPage(self.db_session, self.screening_svc, self.export_svc)
        return self.screening_page

    def _make_detail_page(self):
        from ui.pages.detail_page import DetailPage
        self.detail_page = DetailPage(self.db_session, self.screening_svc)
        return self.detail_page

    def _make_portfolio_page(self):
        from ui.pages.portfolio_page import PortfolioPage
        return PortfolioPage(self.db_session, self.portfolio_svc, self.export_svc)

    def _make_news_page(self):
        from ui.pages.news_page import NewsPage
        return NewsPage(self.db_session, self.alert_svc)

    def _make_watchlist_page(self):
        from ui.pages.watchlist_page import WatchlistPage
        return WatchlistPage(self.db_session, self.alert_svc, self.export_svc)

    def _make_backtest_page(self):
        from ui.pages.backtest_page import BacktestPage
        return BacktestPage(self.db_session, self.backtest_svc)

    def _make_sector_page(self):
        from ui.pages.sector_page import SectorPage
        return SectorPage(self.db_session)

    def _make_macro_page(self):
        from ui.pages.macro_calendar_page import MacroCalendarPage
        self.macro_page = MacroCalendarPage(self.db_session)
        return self.macro_page

    def _make_paper_page(self):
        from ui.pages.paper_trading_page import PaperTradingPage
        return PaperTradingPage(self.db_session, self.paper_svc)

    def _make_compare_page(self):
        from ui.pages.compare_page import ComparePage
        return ComparePage(self.db_session)

    def _make_crypto_page(self):
        from ui.pages.crypto_page import CryptoPage
        return CryptoPage(self.db_session, self.paper_svc)

    def _make_settings_page(self):
        from ui.pages.settings_page import SettingsPage
        return SettingsPage()

    def _on_nav_changed(self, index):
        page = self._ensure_page(index)
        self.stack.setCurrentIndex(index)
        
        # Optimization: Refresh only on first visit
        first_visit = index not in self._visited_pages
        self._visited_pages.add(index)
        
        if first_visit:
            if hasattr(page, 'refresh_data'):
                page.refresh_data()
            elif hasattr(page, 'refresh'):
                page.refresh()

    def _on_ticker_selected(self, ticker: str):
        # self.sidebar.item(1).setHidden(False)
        self.sidebar.setCurrentRow(1)
        detail_page = self._ensure_page(1)
        self.stack.setCurrentWidget(detail_page)
        detail_page.load_ticker(ticker)

    def _on_back_to_screening(self):
        self.sidebar.setCurrentRow(0)
        screening_page = self._ensure_page(0)
        self.stack.setCurrentWidget(screening_page)
        
    def _on_add_to_watchlist(self, ticker: str):
        self.alert_svc.add_ticker(ticker, self.db_session)
        self.tray_icon.showMessage(
            "관심 종목 추가",
            f"{ticker} 종목이 관심 목록에 추가되었습니다.",
            QSystemTrayIcon.Information,
            3000
        )
        
    def _setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        from PySide6.QtWidgets import QApplication as _App, QStyle
        self.tray_icon.setIcon(_App.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        
        tray_menu = QMenu()
        show_action = QAction("열기", self)
        show_action.triggered.connect(self.showNormal)
        quit_action = QAction("종료", self)
        quit_action.triggered.connect(self.close)
        
        tray_menu.addAction(show_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        
    def _check_alerts(self):
        # This calls the alert service to check for any new signals
        # and shows tray notifications
        alerts = self.alert_svc.check_alerts(self.db_session)
        for alert in alerts:
            message = f"{alert['ticker']}: AI 점수 {'↑' if alert['delta'] > 0 else '↓'} {alert['prev_score']:.0f} → {alert['curr_score']:.0f} ({alert['delta']:+.0f}점)"
            self.tray_icon.showMessage(
                "AI 시그널 알림",
                message,
                QSystemTrayIcon.Warning if alert['delta'] < 0 else QSystemTrayIcon.Information,
                5000
            )

    def _refresh_all_data(self):
        # Update timestamp
        from datetime import datetime
        self.update_label.setText(f"마지막 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Refresh current page
        page = self.stack.currentWidget()
        if hasattr(page, 'refresh_data'):
            page.refresh_data()
        elif hasattr(page, 'refresh'):
            page.refresh()

    def closeEvent(self, event):
        # Wait for background workers before closing to avoid SIGABRT
        candidate_pages = []
        for attr_name in ['screening_page', 'macro_page', 'detail_page']:
            p = getattr(self, attr_name, None)
            if p is not None:
                candidate_pages.append(p)
        for attr in ['worker', 'earnings_worker', '_detail_thread']:
            for page in candidate_pages:
                try:
                    w = getattr(page, attr, None)
                    if w is None:
                        continue
                    if hasattr(w, 'isRunning') and w.isRunning():
                        if hasattr(w, 'requestInterruption'):
                            w.requestInterruption()
                        w.wait(1500)
                except RuntimeError:
                    pass
        event.accept()
