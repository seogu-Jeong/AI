import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QGroupBox, QSplitter, QFrame, QGridLayout
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QLinearGradient, 
    QRadialGradient, QAntialiasingPainter, QPainterPath
)
import pyqtgraph as pg

# 전문가용 테마 상수
COLOR_BG = "#0A0A0A"
COLOR_ACCENT = "#00FF41"  # Matrix Green
COLOR_NEURON = "#00A3FF"  # Deep Blue
COLOR_TEXT = "#E0E0E0"
COLOR_EDGE_POS = "#00FF41"
COLOR_EDGE_NEG = "#FF3131"

class NeuralState(QObject):
    """MVI Architecture: Central Tensor State Model"""
    updated = Signal()

    def __init__(self):
        super().__init__()
        # 차원 정의: Input(2) -> Hidden(3) -> Output(1)
        self.x = np.array([[0.5], [-0.5]]) # Input Vector (2x1)
        
        # Hidden Layer (3x2 Weights, 3x1 Bias)
        self.W1 = np.random.randn(3, 2) * 0.5
        self.b1 = np.zeros((3, 1))
        
        # Output Layer (1x3 Weights, 1x1 Bias)
        self.W2 = np.random.randn(1, 3) * 0.5
        self.b2 = np.zeros((1, 1))
        
        self.forward()

    def forward(self):
        """순전파 텐서 연산: R^2 -> R^3 -> R^1"""
        # Layer 1: Linear Transform + Sigmoid
        self.z1 = np.dot(self.W1, self.x) + self.b1
        self.a1 = 1 / (1 + np.exp(-np.clip(self.z1, -10, 10)))
        
        # Layer 2: Linear Transform + Sigmoid
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = 1 / (1 + np.exp(-np.clip(self.z2, -10, 10)))
        self.updated.emit()

class TopologyCanvas(QWidget):
    """QPainter 기반 고성능 다이내믹 신경망 렌더러"""
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setMinimumSize(600, 400)
        self.state.updated.connect(self.update)
        
        # 노드 좌표 설정
        self.node_pos = {
            'in': [QPointF(100, 150), QPointF(100, 350)],
            'hid': [QPointF(350, 100), QPointF(350, 250), QPointF(350, 400)],
            'out': [QPointF(600, 250)]
        }

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLOR_BG))

        # 1. 엣지 드로잉 (가중치 강도 및 흐름 시각화)
        self._draw_edges(painter)
        
        # 2. 노드 드로잉 (활성화 값 기반 발광 효과)
        self._draw_nodes(painter)

    def _draw_edges(self, painter):
        # Input -> Hidden
        for i in range(2):
            for j in range(3):
                weight = self.state.W1[j, i]
                self._paint_edge(painter, self.node_pos['in'][i], self.node_pos['hid'][j], weight)
        
        # Hidden -> Output
        for i in range(3):
            weight = self.state.W2[0, i]
            self._paint_edge(painter, self.node_pos['hid'][i], self.node_pos['out'][0], weight)

    def _paint_edge(self, painter, p1, p2, weight):
        thickness = min(8, abs(weight) * 5 + 1)
        color = QColor(COLOR_EDGE_POS if weight > 0 else COLOR_EDGE_NEG)
        color.setAlpha(int(min(255, abs(weight) * 200 + 50)))
        
        path = QPainterPath()
        path.moveTo(p1)
        path.lineTo(p2)
        
        pen = QPen(color, thickness)
        if weight < 0: pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawPath(path)

    def _draw_nodes(self, painter):
        # Input Nodes
        for i, pos in enumerate(self.node_pos['in']):
            val = float(self.state.x[i])
            self._paint_node(painter, pos, val, f"x{i+1}")
            
        # Hidden Nodes
        for i, pos in enumerate(self.node_pos['hid']):
            val = float(self.state.a1[i])
            self._paint_node(painter, pos, val, f"h{i+1}")
            
        # Output Node
        pos = self.node_pos['out'][0]
        val = float(self.state.a2[0])
        self._paint_node(painter, pos, val, "y")

    def _paint_node(self, painter, pos, val, label):
        radius = 25
        # 발광(Glow) 효과
        glow = QRadialGradient(pos, radius * 1.5)
        color = QColor(COLOR_NEURON)
        glow.setColorAt(0, QColor(color.red(), color.green(), color.blue(), int(val * 150)))
        glow.setColorAt(1, Qt.transparent)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(pos, radius * 1.5, radius * 1.5)
        
        # 본체
        painter.setBrush(QBrush(QColor("#1A1A1A")))
        painter.setPen(QPen(QColor(COLOR_NEURON), 2))
        painter.drawEllipse(pos, radius, radius)
        
        # 라벨 및 값
        painter.setPen(QColor(COLOR_TEXT))
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(QRectF(pos.x()-20, pos.y()-10, 40, 20), Qt.AlignCenter, label)
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(QRectF(pos.x()-25, pos.y()+25, 50, 15), Qt.AlignCenter, f"{val:.2f}")

class MatrixHUD(QFrame):
    """실시간 행렬 연산 수식 전광판"""
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("background-color: #151515; border-radius: 10px; border: 1px solid #333;")
        self.layout = QVBoxLayout(self)
        
        self.label = QLabel()
        self.label.setFont(QFont("JetBrains Mono", 11))
        self.label.setStyleSheet("color: #00FFCC;")
        self.layout.addWidget(self.label)
        
        self.state.updated.connect(self.refresh)
        self.refresh()

    def refresh(self):
        # 행렬 연산 과정을 텍스트로 가시화
        w1_str = str(np.round(self.state.W1, 2)).replace('\n', '\n      ')
        x_str = str(np.round(self.state.x, 2)).replace('\n', '\n      ')
        b1_str = str(np.round(self.state.b1, 2)).replace('\n', '\n      ')
        z1_str = str(np.round(self.state.z1, 2)).replace('\n', '\n      ')
        
        hud_text = (
            f"LAYER 1: Linear Transformation\n"
            f"W₁·x + b₁ = z₁\n"
            f"{w1_str} · {x_str} + {b1_str} = {z1_str}"
        )
        self.label.setText(hud_text)

class ControlPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.layout = QVBoxLayout(self)
        self.setFixedWidth(300)
        
        # Input Control
        group_in = QGroupBox("Input Vector (X)")
        vbox_in = QVBoxLayout()
        self.sl_x1 = self._create_slider("x1", -2.0, 2.0, self.state.x[0,0])
        self.sl_x2 = self._create_slider("x2", -2.0, 2.0, self.state.x[1,0])
        self.sl_x1.valueChanged.connect(self._update_x)
        self.sl_x2.valueChanged.connect(self._update_x)
        vbox_in.addWidget(self.sl_x1)
        vbox_in.addWidget(self.sl_x2)
        group_in.setLayout(vbox_in)
        
        # Hidden Weight Control (Example: w11)
        group_w = QGroupBox("Param Perturbation")
        vbox_w = QVBoxLayout()
        self.sl_w11 = self._create_slider("W1[0,0]", -2.0, 2.0, self.state.W1[0,0])
        self.sl_w11.valueChanged.connect(self._update_w)
        vbox_w.addWidget(self.sl_w11)
        group_w.setLayout(vbox_w)
        
        self.layout.addWidget(group_in)
        self.layout.addWidget(group_w)
        self.layout.addStretch()

    def _create_slider(self, label, min_v, max_v, init_v):
        container = QWidget()
        l = QHBoxLayout(container)
        lbl = QLabel(label); lbl.setFixedWidth(60)
        s = QSlider(Qt.Horizontal); s.setRange(int(min_v*100), int(max_v*100)); s.setValue(int(init_v*100))
        val_lbl = QLabel(f"{init_v:.2f}"); val_lbl.setFixedWidth(40)
        l.addWidget(lbl); l.addWidget(s); l.addWidget(val_lbl)
        
        def on_change(v):
            real_v = v / 100.0
            val_lbl.setText(f"{real_v:.2f}")
            
        s.valueChanged.connect(on_change)
        s.valueChanged.connect(lambda: QTimer.singleShot(0, self.state.forward))
        return s

    def _update_x(self):
        self.state.x[0,0] = self.sl_x1.value() / 100.0
        self.state.x[1,0] = self.sl_x2.value() / 100.0

    def _update_w(self):
        self.state.W1[0,0] = self.sl_w11.value() / 100.0

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forward-Prop Pro: 2-3-1 Matrix Dynamics")
        self.resize(1200, 700)
        self.setStyleSheet(f"background-color: {COLOR_BG}; color: {COLOR_TEXT};")
        
        self.state = NeuralState()
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left: Controls
        self.controls = ControlPanel(self.state)
        
        # Center: Topology & HUD
        self.stage = QWidget()
        vbox_stage = QVBoxLayout(self.stage)
        self.hud = MatrixHUD(self.state)
        self.canvas = TopologyCanvas(self.state)
        vbox_stage.addWidget(self.hud, 1)
        vbox_stage.addWidget(self.canvas, 4)
        
        # Right: Bar Charts
        self.charts = pg.PlotWidget(title="Hidden Activations (a1)")
        self.charts.setBackground(COLOR_BG)
        self.bar_item = pg.BarGraphItem(x=[1, 2, 3], height=[0, 0, 0], width=0.6, brush=COLOR_ACCENT)
        self.charts.addItem(self.bar_item)
        self.charts.setYRange(0, 1)
        self.state.updated.connect(self._update_charts)
        
        main_layout.addWidget(self.controls)
        main_layout.addWidget(self.stage)
        main_layout.addWidget(self.charts)

    def _update_charts(self):
        self.bar_item.setOpts(height=self.state.a1.flatten())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
