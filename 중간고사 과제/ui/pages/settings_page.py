import json
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QLineEdit, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QGroupBox, QScrollArea, QSpacerItem, QSizePolicy, QApplication, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from ui.styles.theme import COLORS, get_stylesheet
from app_paths import get_config_path

class DataWorkerThread(QThread):
    status = Signal(str)
    finished = Signal(str)
    def __init__(self, task: str):  # task = 'collect' or 'inference'
        super().__init__()
        self.task = task
    def run(self):
        import sys, os
        sys.path.insert(0, os.getcwd())
        try:
            if self.task == 'collect':
                self.status.emit('데이터 수집 중...')
                from scripts.seed_data import seed_data
                seed_data()
                self.finished.emit('데이터 수집 완료')
            elif self.task == 'inference':
                self.status.emit('AI 점수 계산 중...')
                from scripts.run_inference import run_inference
                run_inference()
                self.finished.emit('AI 점수 갱신 완료')
        except Exception as e:
            self.finished.emit(f'오류: {str(e)[:80]}')

class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(20)

        title = QLabel("설정")
        title.setObjectName("page_title")
        subtitle = QLabel("API 키 · 스케줄 · 알림 임계값 · 모델 파라미터")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")

        content = QWidget()
        content.setStyleSheet("background-color: transparent;")
        clayout = QVBoxLayout(content)
        clayout.setContentsMargins(0, 0, 0, 0)
        clayout.setSpacing(20)
        
        # --- THEME SECTION ---
        theme_group = self._make_group("테마 설정")
        theme_form = QFormLayout()
        theme_form.setSpacing(12)
        theme_form.setLabelAlignment(Qt.AlignRight)

        theme_label = QLabel("앱 테마")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["다크 모드 (Dark)", "라이트 모드 (Light)"])
        self.theme_combo.setCurrentIndex(0) # Default to Dark Mode

        theme_form.addRow(theme_label, self.theme_combo)
        theme_group.layout().addLayout(theme_form)
        clayout.addWidget(theme_group)

        # --- DATA SECTION ---
        data_group = self._make_group("데이터 소스 설정")
        data_form = QFormLayout()
        data_form.setSpacing(12)
        data_form.setLabelAlignment(Qt.AlignRight)

        self.fred_key = QLineEdit()
        self.fred_key.setPlaceholderText("FRED API 키 입력 (선택)")
        self.fred_key.setEchoMode(QLineEdit.Password)

        self.sp500_count = QSpinBox()
        self.sp500_count.setRange(10, 503)
        self.sp500_count.setValue(503)
        self.sp500_count.setSuffix("  종목")

        self.history_years = QSpinBox()
        self.history_years.setRange(1, 10)
        self.history_years.setValue(3)
        self.history_years.setSuffix("  년")

        data_form.addRow("FRED API KEY", self.fred_key)
        data_form.addRow("수집 종목 수", self.sp500_count)
        data_form.addRow("히스토리 기간", self.history_years)
        data_group.layout().addLayout(data_form)
        clayout.addWidget(data_group)

        # --- SCHEDULE SECTION ---
        sched_group = self._make_group("EOD 배치 스케줄")
        sched_form = QFormLayout()
        sched_form.setSpacing(12)
        sched_form.setLabelAlignment(Qt.AlignRight)

        self.eod_enabled = QCheckBox("매일 자동 실행")
        self.eod_enabled.setChecked(True)
        self.eod_enabled.setStyleSheet(f"color: {COLORS['text_primary']};")

        self.eod_hour = QSpinBox()
        self.eod_hour.setRange(0, 23)
        self.eod_hour.setValue(18)
        self.eod_hour.setSuffix(" 시")

        self.eod_workers = QSpinBox()
        self.eod_workers.setRange(1, 16)
        self.eod_workers.setValue(8)
        self.eod_workers.setSuffix(" 스레드")

        sched_form.addRow("자동 실행", self.eod_enabled)
        sched_form.addRow("실행 시각 (ET)", self.eod_hour)
        sched_form.addRow("병렬 워커 수", self.eod_workers)
        sched_group.layout().addLayout(sched_form)
        clayout.addWidget(sched_group)

        # --- ALERT SECTION ---
        alert_group = self._make_group("알림 설정")
        alert_form = QFormLayout()
        alert_form.setSpacing(12)
        alert_form.setLabelAlignment(Qt.AlignRight)

        self.alert_threshold = QDoubleSpinBox()
        self.alert_threshold.setRange(1.0, 50.0)
        self.alert_threshold.setValue(15.0)
        self.alert_threshold.setSuffix(" 점")
        self.alert_threshold.setDecimals(1)

        self.alert_enabled = QCheckBox("앱 내 알림 활성화")
        self.alert_enabled.setChecked(True)
        self.alert_enabled.setStyleSheet(f"color: {COLORS['text_primary']};")

        alert_form.addRow("점수 변동 임계값", self.alert_threshold)
        alert_form.addRow("알림 활성화", self.alert_enabled)
        alert_group.layout().addLayout(alert_form)
        clayout.addWidget(alert_group)

        # --- MODEL SECTION ---
        model_group = self._make_group("AI 모델 앙상블 가중치")
        model_form = QFormLayout()
        model_form.setSpacing(12)
        model_form.setLabelAlignment(Qt.AlignRight)

        self.w_lstm = QDoubleSpinBox()
        self.w_lstm.setRange(0.0, 1.0)
        self.w_lstm.setValue(0.30)
        self.w_lstm.setSingleStep(0.05)
        self.w_lstm.setDecimals(2)

        self.w_cnn = QDoubleSpinBox()
        self.w_cnn.setRange(0.0, 1.0)
        self.w_cnn.setValue(0.25)
        self.w_cnn.setSingleStep(0.05)
        self.w_cnn.setDecimals(2)

        self.w_transformer = QDoubleSpinBox()
        self.w_transformer.setRange(0.0, 1.0)
        self.w_transformer.setValue(0.25)
        self.w_transformer.setSingleStep(0.05)
        self.w_transformer.setDecimals(2)

        self.w_mlp = QDoubleSpinBox()
        self.w_mlp.setRange(0.0, 1.0)
        self.w_mlp.setValue(0.20)
        self.w_mlp.setSingleStep(0.05)
        self.w_mlp.setDecimals(2)

        weight_note = QLabel("※ 합계가 1.0이 되도록 조정하세요. 동적 가중치는 모델 정확도 기반으로 EOD 후 자동 업데이트됩니다.")
        weight_note.setWordWrap(True)
        weight_note.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        model_form.addRow("LSTM 가중치", self.w_lstm)
        model_form.addRow("CNN 가중치", self.w_cnn)
        model_form.addRow("Transformer 가중치", self.w_transformer)
        model_form.addRow("MLP 가중치", self.w_mlp)
        model_group.layout().addLayout(model_form)
        model_group.layout().addWidget(weight_note)
        clayout.addWidget(model_group)

        # --- DATA MANAGEMENT SECTION ---
        mgmt_group = self._make_group("데이터 관리")
        mgmt_layout = QVBoxLayout()
        
        mgmt_note = QLabel("데이터 수집 및 AI 점수 갱신은 처음 실행 시 또는 새 데이터 필요 시 사용하세요.")
        mgmt_note.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        mgmt_layout.addWidget(mgmt_note)
        
        self.collect_btn = QPushButton("데이터 수집 (S&P500 OHLCV)")
        self.collect_btn.clicked.connect(self._on_collect_data)
        mgmt_layout.addWidget(self.collect_btn)
        
        self.inference_btn = QPushButton("AI 점수 갱신 (Inference)")
        self.inference_btn.setObjectName("secondary")
        self.inference_btn.clicked.connect(self._on_run_inference)
        mgmt_layout.addWidget(self.inference_btn)
        
        self.data_status_label = QLabel("")
        self.data_status_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
        mgmt_layout.addWidget(self.data_status_label)
        
        mgmt_group.layout().addLayout(mgmt_layout)
        clayout.addWidget(mgmt_group)

        # --- ABOUT SECTION ---
        about_group = self._make_group("앱 정보")
        about_layout = QVBoxLayout()
        about_layout.setSpacing(6)

        info_items = [
            ("버전", "StockSense AI v2.0.0"),
            ("AI 모델", "LSTM + CNN + Transformer + MLP + FinBERT"),
            ("데이터 소스", "yfinance (S&P500) + FRED (USD/KRW) + RSS News"),
            ("DB", "SQLite (로컬 저장)"),
            ("프레임워크", "PySide6 + PyTorch 2.3"),
            ("면책 조항", "본 소프트웨어는 투자 조언을 제공하지 않습니다"),
        ]
        for key, val in info_items:
            row = QHBoxLayout()
            k_label = QLabel(key)
            k_label.setFixedWidth(120)
            k_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 600;")
            v_label = QLabel(val)
            v_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px; font-family: 'Menlo', monospace;")
            v_label.setWordWrap(True)
            row.addWidget(k_label)
            row.addWidget(v_label, 1)
            about_layout.addLayout(row)

        about_group.layout().addLayout(about_layout)
        clayout.addWidget(about_group)

        # Save Button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("설정 저장")
        save_btn.setFixedWidth(140)
        save_btn.setFixedHeight(40)
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)
        clayout.addLayout(btn_layout)
        clayout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _make_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 700;
                color: {COLORS['accent']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {COLORS['bg_card']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                background-color: {COLORS['bg_card']};
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            QLabel {{ color: {COLORS['text_primary']}; }}
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox, QCheckBox {{
                background-color: {COLORS['bg_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
                padding: 5px 8px;
                font-family: 'Menlo', monospace;
            }}
        """)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(18, 18, 18, 18)
        vbox.setSpacing(10)
        return group

    def _load_settings(self):
        try:
            config_path = str(get_config_path())
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.theme_combo.setCurrentIndex(config.get('theme', 0))
                    self.fred_key.setText(config.get('fred_key', ''))
                    self.sp500_count.setValue(config.get('sp500_count', 503))
                    self.history_years.setValue(config.get('history_years', 3))
                    self.eod_enabled.setChecked(config.get('eod_enabled', True))
                    self.eod_hour.setValue(config.get('eod_hour', 18))
                    self.eod_workers.setValue(config.get('eod_workers', 8))
                    self.alert_threshold.setValue(config.get('alert_threshold', 15.0))
                    self.alert_enabled.setChecked(config.get('alert_enabled', True))
                    self.w_lstm.setValue(config.get('w_lstm', 0.30))
                    self.w_cnn.setValue(config.get('w_cnn', 0.25))
                    self.w_transformer.setValue(config.get('w_transformer', 0.25))
                    self.w_mlp.setValue(config.get('w_mlp', 0.20))
        except:
            pass

    def _on_save(self):
        config = {
            'theme': self.theme_combo.currentIndex(),
            'fred_key': self.fred_key.text(),
            'sp500_count': self.sp500_count.value(),
            'history_years': self.history_years.value(),
            'eod_enabled': self.eod_enabled.isChecked(),
            'eod_hour': self.eod_hour.value(),
            'eod_workers': self.eod_workers.value(),
            'alert_threshold': self.alert_threshold.value(),
            'alert_enabled': self.alert_enabled.isChecked(),
            'w_lstm': self.w_lstm.value(),
            'w_cnn': self.w_cnn.value(),
            'w_transformer': self.w_transformer.value(),
            'w_mlp': self.w_mlp.value(),
        }
        config_path = str(get_config_path())
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        # Apply theme immediately
        if self.theme_combo.currentIndex() == 0: # dark
            QApplication.instance().setStyleSheet(get_stylesheet())
        else: # light
            QApplication.instance().setStyleSheet("""
                QWidget { background-color: #F5F5F5; color: #212121; font-family: 'AppleSDGothicNeo', 'Malgun Gothic', 'Roboto', sans-serif; font-size: 13px; }
                QLabel { color: #212121; }
                QFrame#card { background-color: white; border: 1px solid #E0E0E0; border-radius: 10px; }
                QGroupBox { 
                    color: #1976D2; 
                    border: 1px solid #BBDEFB; 
                    background-color: white;
                }
                QGroupBox::title {
                    background-color: white;
                }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #ECEFF1;
                    color: #212121;
                    border: 1px solid #CFD8DC;
                }
                QPushButton { background: #1976D2; color: white; border-radius: 6px; padding: 8px 16px; }
                QPushButton:hover { background: #1565C0; }
                QTableWidget { 
                    background: white; 
                    alternate-background-color: #F8F8F8;
                    color: #212121;
                    border: 1px solid #E0E0E0;
                }
                QTableWidget::item { padding: 8px 0px; }
                QHeaderView::section {
                    background-color: #E0E0E0;
                    color: #424242;
                    padding: 5px;
                    border: 1px solid #D6D6D6;
                    font-weight: bold;
                }
                QScrollArea { background-color: #F5F5F5; }
                #page_title { font-size: 24px; font-weight: 700; color: #212121; }
                #section_title { color: #546E7A; font-weight: 600; }
                #stat_value { color: #1976D2; }
            """)

        QMessageBox.information(self, '설정 저장 완료', '앱을 재시작하면 일부 변경사항이 적용됩니다.')

    def _on_collect_data(self):
        self.collect_btn.setEnabled(False)
        self.data_worker = DataWorkerThread('collect')
        self.data_worker.status.connect(self.data_status_label.setText)
        self.data_worker.finished.connect(lambda msg: (self.data_status_label.setText(msg), self.collect_btn.setEnabled(True)))
        self.data_worker.start()

    def _on_run_inference(self):
        self.inference_btn.setEnabled(False)
        self.data_worker = DataWorkerThread('inference')
        self.data_worker.status.connect(self.data_status_label.setText)
        self.data_worker.finished.connect(lambda msg: (self.data_status_label.setText(msg), self.inference_btn.setEnabled(True)))
        self.data_worker.start()
