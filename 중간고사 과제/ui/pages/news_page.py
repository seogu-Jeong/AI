from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QScrollArea, QFrame, QSizePolicy)
from PySide6.QtCore import Qt, QDate, Slot
from ui.styles.theme import COLORS
from ui.widgets.news_card import NewsCard
from ui.workers.news_worker import NewsWorker

DEFAULT_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'META', 'AMZN', 'GOOGL', 'TSLA', 'JPM']

class NewsPage(QWidget):
    def __init__(self, db_session, alert_service, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.alert_service = alert_service
        self.workers = []
        self.all_news = []
        self._news_queue = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(25)

        # Header Section
        header = QHBoxLayout()
        title_vbox = QVBoxLayout()
        title = QLabel("시장 감성 분석")
        title.setObjectName("page_title")
        
        self.date_label = QLabel(QDate.currentDate().toString("yyyy년 MM월 dd일 뉴스 피드"))
        self.date_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        
        title_vbox.addWidget(title)
        title_vbox.addWidget(self.date_label)
        
        refresh_btn = QPushButton("피드 동기화")
        refresh_btn.setFixedHeight(36)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh)
        
        header.addLayout(title_vbox)
        header.addStretch()
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        # Overall Market Sentiment Summary
        self.summary_card = QFrame()
        self.summary_card.setObjectName("card")
        summary_layout = QHBoxLayout(self.summary_card)
        
        summary_info = QVBoxLayout()
        summary_title = QLabel("포트폴리오 시장 감성 지수")
        summary_title.setObjectName("section_title")
        
        self.sentiment_label = QLabel("분석 중...")
        self.sentiment_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #ECEFF1;")
        
        summary_info.addWidget(summary_title)
        summary_info.addWidget(self.sentiment_label)
        
        summary_layout.addLayout(summary_info)
        summary_layout.addStretch()
        
        layout.addWidget(self.summary_card)

        # News List Area
        section_label = QLabel("LATEST NEWS FEED")
        section_label.setObjectName("section_title")
        layout.addWidget(section_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")
        
        self.news_container = QWidget()
        self.news_container.setStyleSheet("background-color: transparent;")
        self.news_layout = QVBoxLayout(self.news_container)
        self.news_layout.setSpacing(12)
        self.news_layout.setContentsMargins(0, 0, 10, 0)
        self.news_layout.addStretch()
        
        scroll.setWidget(self.news_container)
        layout.addWidget(scroll)

    def refresh(self):
        """Fetches news for all tickers in the watchlist or defaults."""
        # Clear existing news and stop active workers
        self.all_news = []
        for worker in self.workers:
            try:
                if worker.isRunning():
                    worker.requestInterruption()
                    worker.wait(2000)
            except RuntimeError:
                pass
        self.workers = []
        
        # Clear UI list
        self._clear_news_layout()

        # Get watchlist from alert service
        watchlist = self.alert_service.get_watchlist(self.db_session)
        
        tickers_to_fetch = []
        if not watchlist:
            tickers_to_fetch = DEFAULT_TICKERS
            self.sentiment_label.setText("기본 종목 분석 중...")
        else:
            tickers_to_fetch = [item['ticker'] for item in watchlist]
            self.sentiment_label.setText("관심 종목 분석 중...")

        self.sentiment_label.setStyleSheet(f"font-size: 32px; font-weight: 800; color: {COLORS['accent']};")

        # Build queue and start first batch (max 3)
        self._news_queue = list(tickers_to_fetch)
        for _ in range(min(3, len(self._news_queue))):
            ticker = self._news_queue.pop(0)
            self._start_news_worker(ticker)

    def _start_news_worker(self, ticker):
        worker = NewsWorker(ticker)
        worker.news_ready.connect(self._on_news_ready)
        worker.finished.connect(self._on_worker_done)
        worker.start()
        self.workers.append(worker)

    @Slot()
    def _on_worker_done(self):
        if self._news_queue:
            ticker = self._news_queue.pop(0)
            self._start_news_worker(ticker)

    def refresh_data(self):
        """Alias for compatibility with common page interface."""
        self.refresh()

    def _clear_news_layout(self):
        while self.news_layout.count() > 1:
            item = self.news_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_placeholder(self, text):
        placeholder = QLabel(text)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 15px; font-weight: 600; margin: 60px;")
        self.news_layout.insertWidget(0, placeholder)

    def _on_news_ready(self, ticker, news_list):
        self.all_news.extend(news_list)
        # Update overall sentiment based on all collected news so far
        self._update_market_sentiment()
        # Re-render the news cards sorted by impact
        self._refresh_news_cards()

    def _update_market_sentiment(self):
        if not self.all_news:
            return
            
        avg_score = sum(n['score'] for n in self.all_news) / len(self.all_news)
        
        if avg_score > 0.4:
            text = "EXTREME BULLISH"
            color = COLORS['bull']
        elif avg_score > 0.1:
            text = "SLIGHTLY BULLISH"
            color = COLORS['bull']
        elif avg_score > -0.1:
            text = "NEUTRAL"
            color = COLORS['neutral']
        elif avg_score > -0.4:
            text = "SLIGHTLY BEARISH"
            color = COLORS['bear']
        else:
            text = "EXTREME BEARISH"
            color = COLORS['bear']
            
        self.sentiment_label.setText(text)
        self.sentiment_label.setStyleSheet(f"font-size: 32px; font-weight: 800; color: {color};")

    def _refresh_news_cards(self):
        self._clear_news_layout()
        
        if not self.all_news:
            self._show_placeholder("최근 수집된 뉴스가 없습니다. RSS 피드를 확인하세요.")
            return

        # Sort by absolute score (most extreme/impactful first)
        sorted_news = sorted(self.all_news, key=lambda x: abs(x['score']), reverse=True)
        
        # Limit to top 25 news items for performance
        for news in sorted_news[:25]:
            card = NewsCard(
                headline=news['headline'],
                source=news.get('source', 'FINANCE NEWS'),
                score=news.get('score', 0),
                label=news.get('label', 'Neutral').capitalize(),
                published=news.get('published', 'JUST NOW'),
                ticker=news.get('ticker', ''),
                summary=news.get('summary', '')
            )
            self.news_layout.insertWidget(self.news_layout.count() - 1, card)
