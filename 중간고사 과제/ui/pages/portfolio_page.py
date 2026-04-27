from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget, 
    QTableWidgetItem, QLineEdit, QFrame, QHeaderView, QSizePolicy, QSpinBox, 
    QScrollArea, QGridLayout
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QDoubleValidator, QFont, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import json
from datetime import datetime

from ui.styles.theme import COLORS, get_score_color
from services.export_service import ExportService
from db.models import Portfolio
from PySide6.QtCore import Qt, QTimer, QThread, Signal

class PortfolioWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)
    def __init__(self, portfolio_svc, prefs, amount):
        super().__init__()
        self.portfolio_svc = portfolio_svc
        self.prefs = prefs
        self.amount = amount
    def run(self):
        from db.database import SessionLocal
        db = SessionLocal()
        try:
            result = self.portfolio_svc.build_portfolio_custom(self.prefs, self.amount, db)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            db.close()

class PortfolioPage(QWidget):
    def __init__(self, db_session, portfolio_service, export_svc, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.portfolio_service = portfolio_service
        self.export_svc = export_svc
        self.current_portfolio = None
        
        self._build_ui()

    def _build_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 24, 30, 24)
        self.main_layout.setSpacing(20)

        # Header
        header_vbox = QVBoxLayout()
        title_label = QLabel("포트폴리오 최적화")
        title_label.setObjectName("page_title")
        subtitle_label = QLabel("투자 성향과 AI 분석 점수를 결합한 맞춤형 자산 배분")
        subtitle_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        header_vbox.addWidget(title_label)
        header_vbox.addWidget(subtitle_label)
        self.main_layout.addLayout(header_vbox)

        # Input Section Card
        input_card = QFrame()
        input_card.setObjectName("card")
        input_card_layout = QHBoxLayout(input_card)
        input_card_layout.setContentsMargins(24, 24, 24, 24)
        input_card_layout.setSpacing(30)

        # LEFT SIDE (Inputs)
        left_side = QVBoxLayout()
        
        pref_title = QLabel("투자 선호도 설정")
        pref_title.setObjectName("section_title")
        left_side.addWidget(pref_title)

        inset_frame = QFrame()
        inset_frame.setObjectName("inset")
        inset_frame.setStyleSheet("""
            #inset {
                background-color: #d0d0d0;
                border: 1px solid #808080;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        grid = QGridLayout(inset_frame)
        grid.setSpacing(10)

        # Define Spinboxes with Tooltips and Descriptions
        self.spin_return = QSpinBox()
        self.spin_stability = QSpinBox()
        self.spin_dividend = QSpinBox()
        self.spin_growth = QSpinBox()
        self.spin_cash = QSpinBox()

        spins_config = [
            (self.spin_return, "수익 추구 (Return)", 3, '1=낮음 ~ 5=높음: AI 고득점 종목 비중', 'AI 고득점 종목 위주 구성'),
            (self.spin_stability, "안정성 (Stability)", 3, '1=낮음 ~ 5=높음: 저변동성 선호', '변동성이 적은 종목 선호'),
            (self.spin_dividend, "배당 수익 (Dividend)", 2, '1=낮음 ~ 5=높음: 고배당 종목 선호', '배당수익률이 높은 종목 선호'),
            (self.spin_growth, "성장성 (Growth)", 2, '1=낮음 ~ 5=높음: 고성장 종목 선호', '이익 성장세가 가파른 종목 선호'),
            (self.spin_cash, "현금 보유 (Cash)", 1, '1=낮음 ~ 5=높음: 현금 보유 비중', '불확실성 대비 현금 비중')
        ]

        for i, (spin, name, default, tip, desc) in enumerate(spins_config):
            lbl = QLabel(name)
            lbl.setFixedWidth(160)
            lbl.setStyleSheet("font-weight: bold; color: #333333;")
            
            spin.setRange(1, 5)
            spin.setValue(default)
            spin.setFixedWidth(60)
            spin.setToolTip(tip)
            
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet("color: #666666; font-size: 11px;")
            
            grid.addWidget(lbl, i, 0)
            grid.addWidget(spin, i, 1)
            grid.addWidget(desc_lbl, i, 2)

        left_side.addWidget(inset_frame)

        # Amount Input
        amount_layout = QHBoxLayout()
        amount_title = QLabel("운용 가능 자산")
        amount_title.setObjectName("section_title")
        amount_layout.addWidget(amount_title)
        
        self.amount_input = QLineEdit()
        self.amount_input.setPlaceholderText("10,000")
        self.amount_input.setValidator(QDoubleValidator(0.0, 1000000000.0, 2))
        self.amount_input.setFixedWidth(150)
        amount_layout.addWidget(self.amount_input)
        amount_layout.addWidget(QLabel("USD ($)"))
        amount_layout.addStretch()
        left_side.addLayout(amount_layout)

        # Build Button
        self.build_btn = QPushButton("▶ AI 포트폴리오 생성")
        self.build_btn.setMinimumHeight(36)
        self.build_btn.setCursor(Qt.PointingHandCursor)
        self.build_btn.clicked.connect(self._on_build_clicked)
        left_side.addWidget(self.build_btn)

        input_card_layout.addLayout(left_side, 3)

        # RIGHT SIDE (Chart)
        right_side = QVBoxLayout()
        chart_label = QLabel("포트폴리오 배분")
        chart_label.setObjectName("section_title")
        right_side.addWidget(chart_label)

        self.chart_canvas = FigureCanvas(Figure(figsize=(4, 4), dpi=90, facecolor='#c0c0c0'))
        self.chart_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chart_canvas.setMinimumSize(220, 220)
        
        # Initial Placeholder Text
        ax = self.chart_canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, "AI 포트폴리오 생성 후\n배분 차트가 표시됩니다", 
                ha='center', va='center', fontsize=10, color='#666666')
        ax.axis('off')
        
        right_side.addWidget(self.chart_canvas)
        input_card_layout.addLayout(right_side, 2)

        self.main_layout.addWidget(input_card)

        # Results Section (Initially Hidden)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(20)

        self.results_title = QLabel("추천 포트폴리오")
        self.results_title.setObjectName("page_title")
        self.results_layout.addWidget(self.results_title)

        # Summary Cards Row
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(15)
        self.card_total = self._create_summary_card("총 투자 규모", "$0", COLORS['accent'])
        self.card_return = self._create_summary_card("기대 수익률 (1년 추정)", "12.4%", COLORS['bull'])
        self.card_mdd = self._create_summary_card("위험도 (Max Drawdown)", "-15.2%", COLORS['bear'])
        summary_layout.addWidget(self.card_total)
        summary_layout.addWidget(self.card_return)
        summary_layout.addWidget(self.card_mdd)
        self.results_layout.addLayout(summary_layout)

        # Portfolio Table (Full Width)
        table_container = QFrame()
        table_container.setObjectName("card")
        table_vbox = QVBoxLayout(table_container)
        table_vbox.setContentsMargins(1, 1, 1, 1)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["종목", "비중", "금액", "AI점수", "매수 이유"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        table_vbox.addWidget(self.table)
        self.results_layout.addWidget(table_container)

        # Export Button Row
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton("PDF/CSV 레포트 저장")
        self.export_btn.setObjectName("secondary")
        self.export_btn.setFixedWidth(180)
        self.export_btn.clicked.connect(self._export_csv)
        export_layout.addWidget(self.export_btn)
        self.results_layout.addLayout(export_layout)

        self.results_container.setVisible(False)
        self.main_layout.addWidget(self.results_container, 1)

    def _create_summary_card(self, title, value, color):
        card = QFrame()
        card.setObjectName("card")
        card.setFixedHeight(80)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 12, 20, 12)
        
        t_label = QLabel(title)
        t_label.setObjectName("section_title")
        
        v_label = QLabel(value)
        v_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: 800;")
        v_label.setObjectName("value_label")
        
        layout.addWidget(t_label)
        layout.addWidget(v_label)
        return card

    def _on_build_clicked(self):
        prefs = {
            'return_w': self.spin_return.value() * 20,
            'stability_w': self.spin_stability.value() * 20,
            'dividend_w': self.spin_dividend.value() * 20,
            'growth_w': self.spin_growth.value() * 20,
            'cash_pct': self.spin_cash.value() / 5.0 * 0.4   # max 40% cash
        }
        
        amount_text = self.amount_input.text().replace(',', '')
        amount = float(amount_text) if amount_text else 10000.0

        self.build_btn.setEnabled(False)
        self.build_btn.setText("생성 중...")
        
        self.worker = PortfolioWorker(self.portfolio_service, prefs, amount)
        self.worker.finished.connect(self._on_portfolio_ready)
        self.worker.error.connect(self._on_portfolio_error)
        self.worker.start()

    def _on_portfolio_error(self, err):
        print(f"Portfolio Error: {err}")
        self.build_btn.setEnabled(True)
        self.build_btn.setText("▶ AI 포트폴리오 생성")

    def _on_portfolio_ready(self, portfolio):
        self.build_btn.setEnabled(True)
        self.build_btn.setText("▶ AI 포트폴리오 생성")
        
        self.current_portfolio = portfolio
        amount_text = self.amount_input.text().replace(',', '')
        amount = float(amount_text) if amount_text else 10000.0

        # Save to DB
        from db.database import SessionLocal
        db = SessionLocal()
        try:
            name = f'맞춤 포트폴리오 {datetime.now().strftime("%Y-%m-%d %H:%M")}'
            holdings_json = json.dumps(portfolio['holdings'], ensure_ascii=False)
            db_record = Portfolio(
                name=name,
                risk_level='custom',
                holdings=holdings_json
            )
            db.add(db_record)
            db.commit()
        except Exception as e:
            print(f"Error saving portfolio: {e}")
            db.rollback()
        finally:
            db.close()

        # Update UI
        self.results_title.setText("추천 포트폴리오")
        
        # Update summary cards
        self.card_total.findChild(QLabel, "value_label").setText(f"${amount:,.0f}")
        
        # Update expected return card
        exp_ret = portfolio.get('expected_return', 12.4)
        ref = portfolio.get('reference_period', '1년 (AI 앙상블 기반 추정)')
        self.card_return.findChild(QLabel, 'value_label').setText(f'{exp_ret}%')
        self.card_return.findChild(QLabel, 'section_title').setText(f'기대 수익률 ({ref})')
        
        # Update table
        holdings = portfolio['holdings']
        self.table.setRowCount(len(holdings))
        
        for i, h in enumerate(holdings):
            # Ticker
            ticker_item = QTableWidgetItem(h['ticker'])
            ticker_item.setTextAlignment(Qt.AlignCenter)
            ticker_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(i, 0, ticker_item)
            
            # Weight
            weight_item = QTableWidgetItem(f"{h['weight']:.1f}%")
            weight_item.setTextAlignment(Qt.AlignCenter)
            weight_item.setFont(QFont('Arial', 10, QFont.Bold))
            self.table.setItem(i, 1, weight_item)
            
            # Amount
            amount_item = QTableWidgetItem(f"${h['amount']:,.2f}")
            amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 2, amount_item)
            
            # AI Score
            score_item = QTableWidgetItem(str(h['score']))
            score_item.setTextAlignment(Qt.AlignCenter)
            score_item.setBackground(QColor(get_score_color(h['score'])))
            self.table.setItem(i, 3, score_item)
            
            # Reason
            reason_item = QTableWidgetItem(h['reason'])
            reason_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.table.setItem(i, 4, reason_item)

        self._update_pie_chart(holdings, portfolio.get('cash_pct', 0))
        
        self.results_container.setVisible(True)

    def _update_pie_chart(self, holdings, cash_pct):
        self.chart_canvas.figure.clear()
        ax = self.chart_canvas.figure.add_subplot(111)

        labels = [h['ticker'] for h in holdings]
        sizes = [h['weight'] for h in holdings]
        if cash_pct > 0:
            labels.append('Cash')
            sizes.append(cash_pct)

        # Sort descending for readability
        pairs = sorted(zip(sizes, labels), reverse=True)
        sizes, labels = zip(*pairs) if pairs else ([], [])
        sizes, labels = list(sizes), list(labels)

        palette = ['#000080','#008000','#cc0000','#008080','#808000',
                   '#4040c0','#40a040','#c04040','#4080a0','#808080']
        colors = [palette[i % len(palette)] for i in range(len(sizes))]

        # Horizontal bar chart — works at any aspect ratio
        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, sizes, color=colors, edgecolor='#808080',
                       linewidth=0.8, height=0.65)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9, color='#000000')
        ax.set_xlabel('비중 (%)', fontsize=8, color='#000000')
        ax.set_xlim(0, max(sizes) * 1.25 if sizes else 100)
        ax.invert_yaxis()  # largest at top

        for bar, val in zip(bars, sizes):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', ha='left',
                    fontsize=8, color='#000000', fontweight='bold')

        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelsize=8)
        ax.grid(axis='x', color='#d0d0d0', linestyle='--', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#808080')
        self.chart_canvas.figure.patch.set_facecolor('#c0c0c0')
        self.chart_canvas.figure.subplots_adjust(left=0.22, right=0.92, top=0.95, bottom=0.15)
        self.chart_canvas.draw()

    def _export_csv(self):
        if self.current_portfolio:
            filepath = self.export_svc.export_portfolio_csv(self.current_portfolio)
            # Optional: Show a message box or status update
            print(f"Portfolio exported to {filepath}")

