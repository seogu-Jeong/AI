import os
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QDateEdit, QSpinBox, QPushButton, QTableWidget, 
    QTableWidgetItem, QHeaderView, QSpacerItem, QSizePolicy,
    QAbstractItemView, QScrollArea
)
from PySide6.QtCore import Qt, QDate, QThread, Signal, QObject
from PySide6.QtGui import QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ui.styles.theme import COLORS
from ui.widgets.help_button import HelpButton
from ui.styles.tooltips import TOOLTIPS

class BacktestWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, backtest_svc, start_date, end_date, top_n, use_rl=False):
        super().__init__()
        self.backtest_svc = backtest_svc
        self.start_date = start_date
        self.end_date = end_date
        self.top_n = top_n
        self.use_rl = use_rl

    def run(self):
        from db.database import SessionLocal
        db = SessionLocal()
        try:
            if self.use_rl:
                from services.rl_portfolio_service import RLPortfolioService
                result = RLPortfolioService.run_rl_backtest(
                    db,
                    self.start_date,
                    self.end_date,
                    self.top_n
                )
            else:
                result = self.backtest_svc.run_backtest(
                    db,
                    self.start_date,
                    self.end_date,
                    self.top_n
                )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            db.close()

class BacktestPage(QWidget):
    def __init__(self, db_session, backtest_svc, parent=None):
        super().__init__(parent)
        self.db_session = db_session
        self.backtest_svc = backtest_svc

        self.init_ui()
        self._init_date_range_from_db()

    def init_ui(self):
        # Outer layout holds just the scroll area, no margins
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
        title_lbl = QLabel('전략 백테스팅')
        title_lbl.setObjectName('page_title')
        
        subtitle_lbl = QLabel('AI 상위 30 종목 포트폴리오 vs S&P500 수익률 비교')
        subtitle_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        
        header_layout.addWidget(title_lbl)
        header_layout.addWidget(subtitle_lbl)
        layout.addLayout(header_layout)

        # 2. CONTROLS CARD
        controls_card = QFrame()
        controls_card.setObjectName('card')
        controls_layout = QHBoxLayout(controls_card)
        controls_layout.setContentsMargins(20, 20, 20, 20)
        controls_layout.setSpacing(20)

        # Start Date
        start_lay = QVBoxLayout()
        start_lay.addWidget(QLabel('시작일'))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        # Default 2 years ago
        two_years_ago = date.today() - relativedelta(years=2)
        self.start_date_edit.setDate(QDate(two_years_ago.year, two_years_ago.month, two_years_ago.day))
        start_lay.addWidget(self.start_date_edit)
        controls_layout.addLayout(start_lay)

        # End Date
        end_lay = QVBoxLayout()
        end_lay.addWidget(QLabel('종료일'))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        end_lay.addWidget(self.end_date_edit)
        controls_layout.addLayout(end_lay)

        # Top N
        top_n_lay = QVBoxLayout()
        top_n_lay.addWidget(QLabel('상위 종목 수'))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 50)
        self.top_n_spin.setValue(30)
        top_n_lay.addWidget(self.top_n_spin)
        controls_layout.addLayout(top_n_lay)

        # RL Mode Toggle
        rl_lay = QVBoxLayout()
        rl_lay.addWidget(QLabel('강화학습 모드'))
        self.rl_btn = QPushButton('RL 비교 OFF')
        self.rl_btn.setCheckable(True)
        self.rl_btn.setChecked(False)
        self.rl_btn.setMinimumWidth(120)
        self.rl_btn.toggled.connect(self._on_rl_toggled)
        rl_lay.addWidget(self.rl_btn)
        controls_layout.addLayout(rl_lay)

        # Run Button
        self.run_btn = QPushButton('백테스트 실행')
        self.run_btn.setMinimumWidth(160)
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._on_run)
        controls_layout.addWidget(self.run_btn, 0, Qt.AlignBottom)

        # Status Label
        self.status_lbl = QLabel('')
        self.status_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        controls_layout.addWidget(self.status_lbl, 1, Qt.AlignRight | Qt.AlignBottom)

        layout.addWidget(controls_card)

        # 3. RESULTS SECTION (initially hidden)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(20)

        # a. STATS CARDS ROWS
        stats_container = QVBoxLayout()
        stats_container.setSpacing(15)
        
        stats_row1 = QHBoxLayout()
        stats_row1.setSpacing(15)
        self.card_return = self._make_stat_card('전략 수익률', '--', COLORS['bull'])
        self.card_bench = self._make_stat_card('S&P500', '--', COLORS['accent2'])
        self.card_mdd = self._make_stat_card('최대 낙폭', '--', COLORS['bear'])
        self.card_sharpe = self._make_stat_card('샤프 지수', '--', COLORS['warning'])
        stats_row1.addWidget(self.card_return)
        stats_row1.addWidget(self.card_bench)
        stats_row1.addWidget(self.card_mdd)
        stats_row1.addWidget(self.card_sharpe)
        
        stats_row2 = QHBoxLayout()
        stats_row2.setSpacing(15)
        self.card_var = self._make_stat_card('VaR 95%', '--', COLORS['bear'])
        self.card_sortino = self._make_stat_card('Sortino', '--', COLORS['warning'])
        self.card_winrate = self._make_stat_card('승률', '--', COLORS['bull'])
        stats_row2.addWidget(self.card_var)
        stats_row2.addWidget(self.card_sortino)
        stats_row2.addWidget(self.card_winrate)
        
        stats_container.addLayout(stats_row1)
        stats_container.addLayout(stats_row2)
        self.results_layout.addLayout(stats_container)

        # b. CHART CARD
        chart_card = QFrame()
        chart_card.setObjectName('card')
        chart_card_layout = QVBoxLayout(chart_card)
        
        self.figure = Figure(figsize=(10, 8), dpi=90, facecolor='#c0c0c0')
        self.chart_canvas = FigureCanvasQTAgg(self.figure)
        self.chart_canvas.setMinimumHeight(660)
        chart_card_layout.addWidget(self.chart_canvas)
        
        self.results_layout.addWidget(chart_card)

        # c. MONTHLY RETURNS TABLE
        table_card = QFrame()
        table_card.setObjectName('card')
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(15, 15, 15, 15)
        
        table_title = QLabel('월별 수익률 상세')
        table_title.setObjectName('section_title')
        table_layout.addWidget(table_title)
        
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(4)
        self.detail_table.setHorizontalHeaderLabels(['월', '전략 수익률', 'S&P500', '초과 수익률'])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detail_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.detail_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.detail_table.setShowGrid(False)
        self.detail_table.setAlternatingRowColors(True)
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setFixedHeight(340)
        
        table_layout.addWidget(self.detail_table)
        self.results_layout.addWidget(table_card)

        layout.addWidget(self.results_widget)
        self.results_widget.hide()

        # 4. SPACER
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def _make_stat_card(self, title, value, color):
        card = QFrame()
        card.setObjectName('card')
        card.setFixedHeight(80)
        
        lay = QVBoxLayout(card)
        lay.setContentsMargins(15, 10, 15, 10)
        
        # Title + Help Button Row
        title_lay = QHBoxLayout()
        title_lay.setSpacing(5)
        
        t_lbl = QLabel(title)
        t_lbl.setObjectName('section_title')
        title_lay.addWidget(t_lbl)
        
        help_btn = HelpButton(title)
        title_lay.addWidget(help_btn)
        title_lay.addStretch()
        
        lay.addLayout(title_lay)
        
        v_lbl = QLabel(value)
        v_lbl.setObjectName('val')
        v_lbl.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")
        
        lay.addWidget(v_lbl)
        return card

    def _init_date_range_from_db(self):
        from db.database import SessionLocal
        from db.models import AIScore
        db = SessionLocal()
        try:
            dates = db.query(AIScore.date).distinct().order_by(AIScore.date).all()
            dates = [r[0] for r in dates]
            if len(dates) < 2:
                return
            # Filter isolated outlier dates (gap > 400 days from previous)
            contiguous = [dates[0]]
            for i in range(1, len(dates)):
                gap = (dates[i] - dates[i-1]).days
                if gap <= 400:
                    contiguous.append(dates[i])
            if len(contiguous) < 2:
                return
            first_date = contiguous[0]
            self.start_date_edit.setDate(QDate(first_date.year, first_date.month, first_date.day))
            self.end_date_edit.setDate(QDate.currentDate())
        except Exception as e:
            print(f"Failed to init backtest date range from DB: {e}")
        finally:
            db.close()

    def _on_rl_toggled(self, checked):
        self.rl_btn.setText('RL 비교 ON' if checked else 'RL 비교 OFF')

    def _on_run(self):
        self.run_btn.setEnabled(False)
        self.status_lbl.setText('분석 중...')
        self.results_widget.hide()

        start = self.start_date_edit.date().toString('yyyy-MM-dd')
        end = self.end_date_edit.date().toString('yyyy-MM-dd')
        top_n = self.top_n_spin.value()

        self.thread = QThread()
        use_rl = self.rl_btn.isChecked()
        self.worker = BacktestWorker(self.backtest_svc, start, end, top_n, use_rl=use_rl)
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
        self.thread.quit()

    def _on_results(self, result: dict):
        self.run_btn.setEnabled(True)
        self.status_lbl.setText('')

        if result.get('error'):
            self.status_lbl.setText(f"Error: {result['error']}")
            return

        if result.get('demo'):
            self.status_lbl.setText('데모 모드 (AI 분석 데이터 없음 - SPY/QQQ 기반 시뮬레이션)')
            self.status_lbl.setStyleSheet("color: #FF8C00; font-weight: bold;")
        else:
            self.status_lbl.setText('')
            self.status_lbl.setStyleSheet('')

        # Update Stat Cards
        self.card_return.findChild(QLabel, 'val').setText(f"{result['total_return']:.1f}%")
        self.card_bench.findChild(QLabel, 'val').setText(f"{result['benchmark_return']:.1f}%")
        self.card_mdd.findChild(QLabel, 'val').setText(f"{result['max_drawdown']:.1f}%")
        self.card_sharpe.findChild(QLabel, 'val').setText(f"{result['sharpe']:.2f}")
        
        self.card_var.findChild(QLabel, 'val').setText(f"{result['var_95']:.1f}%")
        self.card_sortino.findChild(QLabel, 'val').setText(f"{result['sortino']:.2f}")
        self.card_winrate.findChild(QLabel, 'val').setText(f"{result['win_rate']:.0f}%")

        if result.get('rl_mode'):
            self.status_lbl.setText(
                f"RL 전략: {result.get('rl_total_return', 0):.1f}% | Sharpe {result.get('rl_sharpe', 0):.2f}  |  "
                f"등가중: {result.get('total_return', 0):.1f}%  |  S&P500: {result.get('benchmark_return', 0):.1f}%"
            )
            self.status_lbl.setStyleSheet('color: #FF8C00; font-weight: bold;')

        # Plot Charts
        self.figure.clear()
        
        # Subplot 1: Cumulative Return
        ax1 = self.figure.add_subplot(211)
        ax1.set_facecolor('#ffffff')
        self.figure.patch.set_facecolor('#c0c0c0')
        
        dates = result['dates']
        port = result['portfolio_curve']
        bench = result['benchmark_curve']
        rolling_sharpe = result.get('rolling_sharpe', [])
        
        x = list(range(len(dates)))
        rl = result.get('rl_curve', [])
        
        ax1.plot(x, port, color='#008000', label='AI 등가중', linewidth=2)
        ax1.plot(x, bench, color='#000080', label='S&P500', linewidth=1.5, alpha=0.8)
        if rl and len(rl) == len(x):
            ax1.plot(x, rl, color='#FF8C00', label='RL 최적화', linewidth=2, linestyle='--')
            train_size = result.get('train_size', 0)
            if 0 < train_size < len(x):
                ax1.axvline(x=train_size, color='#9400D3', linestyle=':', linewidth=1.5, alpha=0.8)
        ax1.axhline(y=1.0, color='#808080', linestyle=':', linewidth=1, alpha=0.7)
        
        all_vals = port + bench + (rl if rl else [])
        y_min = min(all_vals) * 0.92
        y_max = max(all_vals) * 1.05
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlim(-0.5, len(x) - 0.5)
        
        step = max(1, len(dates) // 8)
        tick_positions = x[::step]
        tick_labels = [dates[i] for i in tick_positions]
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([]) # Hide labels for top chart
        
        ax1.set_title('누적 수익률', color='#000000', pad=10, fontsize=11, fontweight='bold')
        ax1.set_ylabel('누적 배수', fontsize=8, color='#000000')
        ax1.tick_params(colors='#000000', labelsize=8)
        ax1.grid(True, color='#d0d0d0', linestyle='--', alpha=0.5)
        
        legend = ax1.legend(facecolor='#ffffff', edgecolor='#808080', fontsize=9)
        for text in legend.get_texts():
            text.set_color('#000000')
        
        # Subplot 2: Rolling Sharpe
        if len(rolling_sharpe) >= 3:
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            ax2.set_facecolor('#ffffff')
            
            # Use same x-axis
            ax2.plot(x, rolling_sharpe, color='#FF8C00', label='3개월 롤링 샤프', linewidth=1.5)
            ax2.axhline(y=1.0, color='#008000', linestyle='--', linewidth=1, alpha=0.8, label='좋음 (1.0)')
            ax2.axhline(y=0.0, color='#808080', linestyle='--', linewidth=1, alpha=0.5)
            
            ax2.set_title('3개월 롤링 샤프지수', color='#000000', pad=10, fontsize=11, fontweight='bold')
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
            ax2.set_ylabel('Sharpe Ratio', fontsize=8, color='#000000')
            ax2.tick_params(colors='#000000', labelsize=8)
            ax2.grid(True, color='#d0d0d0', linestyle='--', alpha=0.5)
            
            # Set height ratio (Main 2/3, Rolling 1/3)
            # This is handled by subplot grid, but we can adjust spacing
        else:
            # If not enough data, just show labels on ax1
            ax1.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)

        for spine in ax1.spines.values():
            spine.set_color('#808080')
        if 'ax2' in locals():
            for spine in ax2.spines.values():
                spine.set_color('#808080')
        
        self.figure.tight_layout()
        self.figure.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.15, hspace=0.35)
        self.chart_canvas.draw()

        # Populate Detail Table
        # monthly_returns is not explicitly in results but we can compute from curves 
        # Actually backtest_service computes monthly_returns but doesn't return them in result dict.
        # Wait, I should check backtest_service again.
        
        # Looking at backtest_service.py:
        # monthly_returns = [] ... simulation loop ... monthly_returns.append(port_ret)
        # It's not in the return dict. I should probably use portfolio_curve to derive them or modify service.
        # But the requirement says "Populate detail_table with monthly returns if available".
        
        # Let's derive them from the curves.
        self.detail_table.setRowCount(0)
        for i in range(1, len(dates)):
            p_ret = (port[i] / port[i-1]) - 1
            b_ret = (bench[i] / bench[i-1]) - 1
            diff = p_ret - b_ret
            
            row = self.detail_table.rowCount()
            self.detail_table.insertRow(row)
            
            self.detail_table.setItem(row, 0, QTableWidgetItem(dates[i]))
            self.detail_table.setItem(row, 1, QTableWidgetItem(f"{p_ret*100:+.2f}%"))
            self.detail_table.setItem(row, 2, QTableWidgetItem(f"{b_ret*100:+.2f}%"))
            
            diff_item = QTableWidgetItem(f"{diff*100:+.2f}%")
            if diff > 0:
                diff_item.setForeground(QColor(COLORS['bull']))
            elif diff < 0:
                diff_item.setForeground(QColor(COLORS['bear']))
            self.detail_table.setItem(row, 3, diff_item)

        self.results_widget.show()
