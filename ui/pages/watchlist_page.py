import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QFrame, QDialog, QLineEdit, 
                             QSpinBox, QFormLayout, QMessageBox, QAbstractItemView)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from ui.styles.theme import COLORS, get_score_color
from db.models import Watchlist, AlertHistory

class AddTickerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("관심 종목 추가")
        self.setFixedWidth(300)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        form = QFormLayout()
        form.setSpacing(10)
        
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("예: AAPL, TSLA, NVDA")
        self.threshold_input = QSpinBox()
        self.threshold_input.setRange(1, 100)
        self.threshold_input.setValue(15)
        self.threshold_input.setSuffix(" 점")
        
        form.addRow("티커 (Ticker):", self.ticker_input)
        form.addRow("알림 임계값:", self.threshold_input)
        
        layout.addLayout(form)
        
        buttons = QHBoxLayout()
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setObjectName("secondary")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.add_btn = QPushButton("추가")
        self.add_btn.clicked.connect(self.accept)
        
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.add_btn)
        layout.addLayout(buttons)

    def get_data(self):
        return self.ticker_input.text().strip().upper(), self.threshold_input.value()

class WatchlistPage(QWidget):
    ticker_selected = Signal(str)

    def __init__(self, db_session, alert_service, export_svc=None, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.alert_service = alert_service
        self.export_svc = export_svc
        self.init_ui()
        self.refresh()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(20)

        # Header Section
        header = QHBoxLayout()
        title_vbox = QVBoxLayout()
        title = QLabel("관심 종목 관리")
        title.setObjectName("page_title")
        subtitle = QLabel("AI 점수 변동 모니터링 및 실시간 알림 설정")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        title_vbox.addWidget(title)
        title_vbox.addWidget(subtitle)
        
        add_btn = QPushButton("+ 종목 추가")
        add_btn.setFixedHeight(36)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self._on_add_clicked)
        
        header.addLayout(title_vbox)
        header.addStretch()
        header.addWidget(add_btn)
        layout.addLayout(header)

        # Table Section
        table_container = QFrame()
        table_container.setObjectName("card")
        table_container.setStyleSheet(f"background-color: {COLORS['bg_surface']}; border-radius: 10px; border: 1px solid {COLORS['border']};")
        table_vbox = QVBoxLayout(table_container)
        table_vbox.setContentsMargins(1, 1, 1, 1)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["종목", "AI점수", "변동", "마지막확인", "알림임계값", "삭제"])
        
        # Style table headers
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.Stretch)
        header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 종목 (ticker)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # AI점수 
        header_view.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 변동
        header_view.setSectionResizeMode(3, QHeaderView.Stretch)  # 마지막확인 (let this stretch)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 알림임계값
        header_view.setSectionResizeMode(5, QHeaderView.Fixed)  # 삭제 button
        header_view.setFixedHeight(44)
        
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(2, 70)
        self.table.setColumnWidth(4, 90)
        self.table.setColumnWidth(5, 70)
        
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideRight)
        
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        
        self.table.itemClicked.connect(self._on_row_clicked)
        table_vbox.addWidget(self.table)
        layout.addWidget(table_container)

        # Alert Banner
        self.alert_banner = QFrame()
        self.alert_banner.setObjectName("alertBanner")
        self.alert_banner.setStyleSheet(f"""
            QFrame#alertBanner {{
                background-color: {COLORS['bear_bg']};
                border: 1px solid {COLORS['bear']}44;
                border-radius: 8px;
                min-height: 50px;
            }}
            QLabel {{
                color: {COLORS['bear']};
                font-weight: bold;
                font-size: 14px;
            }}
        """)
        alert_layout = QHBoxLayout(self.alert_banner)
        alert_layout.setContentsMargins(15, 5, 15, 5)
        self.alert_label = QLabel("⚠️ TSLA: AI 점수 급락 (63→45, -18점)")
        alert_layout.addWidget(self.alert_label)
        
        layout.addWidget(self.alert_banner)
        # Initially hide if no alerts
        self.alert_banner.hide()

        # Export CSV Button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton('CSV 내보내기')
        self.export_btn.setObjectName('secondary')
        self.export_btn.setFixedWidth(160)
        self.export_btn.clicked.connect(self._on_export_csv)
        export_layout.addWidget(self.export_btn)
        layout.addLayout(export_layout)

    def refresh(self):
        # Fetch watchlist data
        items = self.alert_service.get_watchlist(self.db_session)
        
        self.table.setRowCount(len(items))
        for i, item in enumerate(items):
            # Ticker
            ticker = item['ticker']
            ticker_item = QTableWidgetItem(ticker)
            ticker_item.setTextAlignment(Qt.AlignCenter)
            ticker_item.setFont(QFont("SF Pro Display", 11, QFont.Bold))
            ticker_item.setForeground(QColor(COLORS['accent']))
            self.table.setItem(i, 0, ticker_item)
            
            # AI Score
            score = item.get('current_score')
            if score is not None:
                score_str = f"{score:.1f}"
                score_color = get_score_color(score)
            else:
                score_str = "N/A"
                score_color = COLORS['neutral']
                
            score_item = QTableWidgetItem(score_str)
            score_item.setTextAlignment(Qt.AlignCenter)
            score_item.setForeground(QColor(score_color))
            score_item.setFont(QFont("SF Pro Display", 11, QFont.Bold))
            self.table.setItem(i, 1, score_item)
            
            # Delta (Variation)
            delta = item.get('delta', 0.0)
            if score is not None:
                prefix = "▲" if delta > 0 else ("▼" if delta < 0 else "")
                delta_str = f"{prefix} {abs(delta):.1f}" if delta != 0 else "0.0"
                if delta > 0:
                    delta_color = COLORS['bull']
                elif delta < 0:
                    delta_color = COLORS['bear']
                else:
                    delta_color = COLORS['neutral']
            else:
                delta_str = "-"
                delta_color = COLORS['text_muted']
                
            delta_item = QTableWidgetItem(delta_str)
            delta_item.setTextAlignment(Qt.AlignCenter)
            delta_item.setFont(QFont("SF Pro Display", 10, QFont.Bold))
            delta_item.setForeground(QColor(delta_color))
            self.table.setItem(i, 2, delta_item)
            
            # Last Check
            date_str = item.get('added_at', 'N/A')
            if date_str and 'T' in date_str:
                date_str = date_str.split('T')[0]
            date_item = QTableWidgetItem(date_str)
            date_item.setTextAlignment(Qt.AlignCenter)
            date_item.setForeground(QColor(COLORS['text_secondary']))
            self.table.setItem(i, 3, date_item)
            
            # Alert Threshold
            threshold = item.get('alert_threshold', 15.0)
            thresh_item = QTableWidgetItem(f"±{threshold}")
            thresh_item.setTextAlignment(Qt.AlignCenter)
            thresh_item.setForeground(QColor(COLORS['text_primary']))
            self.table.setItem(i, 4, thresh_item)
            
            # Delete Button
            del_btn = QPushButton("삭제")
            del_btn.setObjectName("ghost")
            del_btn.setFixedWidth(60)
            del_btn.setStyleSheet(f"color: {COLORS['bear']}; font-weight: bold;")
            del_btn.clicked.connect(lambda _, t=ticker: self._on_delete_clicked(t))
            
            # Center the button in cell
            container = QWidget()
            btn_layout = QHBoxLayout(container)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.setAlignment(Qt.AlignCenter)
            btn_layout.addWidget(del_btn)
            self.table.setCellWidget(i, 5, container)

        # Check for unread alerts in history
        self._update_alert_banner()

    def _update_alert_banner(self):
        latest_alerts = self.db_session.query(AlertHistory).order_by(AlertHistory.fired_at.desc()).limit(1).all()
        if latest_alerts:
            a = latest_alerts[0]
            action = "급등" if a.delta > 0 else "급락"
            self.alert_label.setText(f"⚠️ {a.ticker}: AI 점수 {action} ({a.score_before:.1f}→{a.score_after:.1f}, {a.delta:+.1f}점)")
            self.alert_banner.show()
        else:
            self.alert_banner.hide()

    def _on_add_clicked(self):
        dialog = AddTickerDialog(self)
        if dialog.exec():
            ticker, threshold = dialog.get_data()
            if ticker:
                self.alert_service.add_ticker(ticker, self.db_session, threshold)
                self.refresh()

    def _on_delete_clicked(self, ticker):
        reply = QMessageBox.question(self, "삭제 확인", f"'{ticker}' 종목을 관심 종목에서 삭제하시겠습니까?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.alert_service.remove_ticker(ticker, self.db_session)
            self.refresh()

    def _on_row_clicked(self, item):
        ticker = self.table.item(item.row(), 0).text()
        self.ticker_selected.emit(ticker)

    def _on_export_csv(self):
        if not self.export_svc:
            return
        items = self.alert_service.get_watchlist(self.db_session)
        if not items:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, '알림', '내보낼 관심 종목이 없습니다.')
            return
        
        filepath = self.export_svc.export_watchlist_csv(items)
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, '저장 완료', f'저장 완료: {filepath}')
