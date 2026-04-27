import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QGroupBox, QSplitter, QComboBox, QFormLayout, QPushButton
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QColor

# 고성능 렌더링을 위한 전역 설정
pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
pg.setConfigOption('background', '#121212')
pg.setConfigOption('foreground', '#E0E0E0')

class StateModel(QObject):
    """MVI 아키텍처의 핵심: 단일 진실 공급원 (Single Source of Truth)"""
    state_changed = Signal()

    def __init__(self):
        super().__init__()
        self._w1 = 1.0
        self._w2 = 1.0
        self._b = 0.0
        self.epsilon_0 = 8.854e-12
        self.charge_q = 1.602e-19
        self.physics_mode = False
        
        # Grid 설정
        self.res = 100
        self.x = np.linspace(-5, 5, self.res)
        self.y = np.linspace(-5, 5, self.res)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # 샘플 데이터 (XOR 형태의 물리적 전하)
        self.data_x = np.array([-2, 2, -2, 2])
        self.data_y = np.array([-2, 2, 2, -2])
        self.data_c = np.array([0, 0, 1, 1]) # 0: Blue(-), 1: Red(+)

    @property
    def w1(self): return self._w1
    @w1.setter
    def w1(self, val): self._w1 = val; self.state_changed.emit()

    @property
    def w2(self): return self._w2
    @w2.setter
    def w2(self, val): self._w2 = val; self.state_changed.emit()

    @property
    def b(self): return self._b
    @b.setter
    def b(self, val): self._b = val; self.state_changed.emit()

    def get_potential_field(self):
        """벡터화된 전위 장(Potential Field) 고속 연산"""
        z = self._w1 * self.X + self._w2 * self.Y + self._b
        if self.physics_mode:
            kappa = self.charge_q / (4 * np.pi * self.epsilon_0)
            return kappa * z
        return z

class ReactiveSlider(QWidget):
    """정밀 소수점 제어 및 이벤트 스로틀링이 적용된 슬라이더"""
    valueChanged = Signal(float)

    def __init__(self, label, min_val, max_val, init_val, step=0.1):
        super().__init__()
        self.step = step
        self.multiplier = int(1 / step)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.name_label = QLabel(label)
        self.name_label.setFixedWidth(30)
        self.name_label.setFont(QFont("Consolas", 10, QFont.Bold))
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * self.multiplier), int(max_val * self.multiplier))
        self.slider.setValue(int(init_val * self.multiplier))
        
        self.val_label = QLabel(f"{init_val:.2f}")
        self.val_label.setFixedWidth(50)
        self.val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_label.setFont(QFont("Consolas", 10))
        
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_label)
        
        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, val):
        real_val = val / self.multiplier
        self.val_label.setText(f"{real_val:.2f}")
        self.valueChanged.emit(real_val)
        
    def set_value(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * self.multiplier))
        self.val_label.setText(f"{val:.2f}")
        self.slider.blockSignals(False)

class Visualizer2D(pg.PlotWidget):
    """등전위면과 전기장 벡터, 전하를 시각화하는 2D 평면"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setXRange(-5, 5)
        self.setYRange(-5, 5)
        self.setTitle("2D Potential Field & Nodal Line", color='#AAA')
        
        # 히트맵 (전위 장)
        self.img = pg.ImageItem()
        # coolwarm 유사 컬러맵 (Blue to Red)
        pos = np.array([0.0, 0.5, 1.0])
        color = np.array([[43, 131, 186, 200], [255, 255, 255, 50], [215, 25, 28, 200]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.img.setLookupTable(cmap.getLookupTable())
        self.addItem(self.img)
        
        # 맵 위치/스케일 조정
        rect = pg.QtCore.QRectF(-5, -5, 10, 10)
        self.img.setRect(rect)
        
        # 데이터 포인트 (전하)
        self.scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None))
        self.addItem(self.scatter)
        
        # 결정 경계선 (Nodal Line)
        self.boundary_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#FFFF00', width=2, style=Qt.DashLine))
        self.addItem(self.boundary_line)
        
        # 가중치 벡터 화살표 (전기장 방향)
        self.vector_arrow = pg.ArrowItem(pen='g', brush='g', tipAngle=30, baseAngle=20, tailLen=40, tailWidth=3)
        self.addItem(self.vector_arrow)

    def render(self):
        # 1. 히트맵 렌더링
        Z = self.model.get_potential_field()
        self.img.setImage(Z.T, autoLevels=False, levels=[-10, 10])
        
        # 2. 결정 경계 계산 (w1*x + w2*y + b = 0)
        w1, w2, b = self.model.w1, self.model.w2, self.model.b
        if w2 != 0:
            angle = np.degrees(np.arctan(-w1 / w2))
            pos = (0, -b / w2)
        elif w1 != 0:
            angle = 90
            pos = (-b / w1, 0)
        else:
            angle = 0
            pos = (0, 0)
            
        self.boundary_line.setPos(pos)
        self.boundary_line.setAngle(angle)
        
        # 3. 그래디언트 벡터 표시 (원점에서 법선 방향으로)
        norm = np.sqrt(w1**2 + w2**2) + 1e-8
        dir_x, dir_y = w1/norm, w2/norm
        self.vector_arrow.setPos(dir_x * 2, dir_y * 2)
        self.vector_arrow.setStyle(angle=180 - np.degrees(np.arctan2(dir_y, dir_x)))
        
        # 4. 전하 렌더링
        brushes = [pg.mkBrush('#FF4B4B') if c == 1 else pg.mkBrush('#4B4BFF') for c in self.model.data_c]
        self.scatter.setData(self.model.data_x, self.model.data_y, brush=brushes)

class Visualizer3D(gl.GLViewWidget):
    """공간의 곡률과 위상을 보여주는 3D OpenGL 서피스 뷰"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.setCameraPosition(distance=25, elevation=30, azimuth=45)
        
        # 축 그리드
        gx = gl.GLGridItem()
        gx.setSize(10, 10)
        self.addItem(gx)
        
        # 3D 곡면 
        self.surface = gl.GLSurfacePlotItem(computeNormals=False, smooth=False)
        self.surface.shader()
        self.addItem(self.surface)
        
    def render(self):
        Z = self.model.get_potential_field()
        
        # 로짓을 시그모이드(확률/활성화)로 변환하여 부드러운 단층(Phase transition) 시각화
        Z_sig = 10 * (1 / (1 + np.exp(-Z))) - 5 
        
        # 색상 매핑 (Z 높이에 따라)
        colors = np.empty((self.model.res, self.model.res, 4), dtype=np.float32)
        norm_z = (Z_sig + 5) / 10.0
        colors[..., 0] = norm_z          # Red
        colors[..., 1] = 0.2             # Green
        colors[..., 2] = 1 - norm_z      # Blue
        colors[..., 3] = 0.8             # Alpha
        
        self.surface.setData(x=self.model.x, y=self.model.y, z=Z_sig, colors=colors)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhysViz-Perceptron Pro: Tensor Field Simulator")
        self.resize(1400, 800)
        
        self.model = StateModel()
        self.init_ui()
        self.model.state_changed.connect(self.request_render)
        
        # 렌더링 스로틀링 (60FPS 제한)
        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.execute_render)
        self.render_pending = False
        
        self.execute_render() # 초기 렌더링

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 1. 컨트롤 패널 (좌측)
        control_panel = QWidget()
        control_panel.setFixedWidth(320)
        control_layout = QVBoxLayout(control_panel)
        
        # Tensor Field 제어부
        group_tensor = QGroupBox("Tensor Field Engineering (Weights & Bias)")
        group_tensor.setFont(QFont("Arial", 11, QFont.Bold))
        vbox_tensor = QVBoxLayout()
        
        self.sl_w1 = ReactiveSlider("w1", -5.0, 5.0, self.model.w1)
        self.sl_w2 = ReactiveSlider("w2", -5.0, 5.0, self.model.w2)
        self.sl_b = ReactiveSlider("b ", -10.0, 10.0, self.model.b)
        
        self.sl_w1.valueChanged.connect(lambda v: setattr(self.model, 'w1', v))
        self.sl_w2.valueChanged.connect(lambda v: setattr(self.model, 'w2', v))
        self.sl_b.valueChanged.connect(lambda v: setattr(self.model, 'b', v))
        
        vbox_tensor.addWidget(self.sl_w1)
        vbox_tensor.addWidget(self.sl_w2)
        vbox_tensor.addWidget(self.sl_b)
        group_tensor.setLayout(vbox_tensor)
        
        # 상태 정보 표시부
        group_info = QGroupBox("Mathematical State")
        form_info = QFormLayout()
        self.lbl_eq = QLabel()
        self.lbl_eq.setFont(QFont("Consolas", 12))
        self.lbl_eq.setStyleSheet("color: #00FFFF;")
        form_info.addRow("Equation:", self.lbl_eq)
        group_info.setLayout(form_info)
        
        control_layout.addWidget(group_tensor)
        control_layout.addWidget(group_info)
        control_layout.addStretch()
        
        # 2. 비주얼라이저 스테이지 (우측)
        splitter = QSplitter(Qt.Horizontal)
        self.view_2d = Visualizer2D(self.model)
        self.view_3d = Visualizer3D(self.model)
        
        splitter.addWidget(self.view_2d)
        splitter.addWidget(self.view_3d)
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(splitter)

    def request_render(self):
        """디바운싱/스로틀링을 통한 UI 프리징 방지"""
        if not self.render_pending:
            self.render_pending = True
            self.render_timer.start(16) # ~60FPS

    def execute_render(self):
        self.render_pending = False
        self.view_2d.render()
        self.view_3d.render()
        
        # 수식 업데이트
        eq_text = f"z = {self.model.w1:.2f}x₁ + {self.model.w2:.2f}x₂ + {self.model.b:.2f}"
        self.lbl_eq.setText(eq_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 다크 테마 애플리케이션 전역 설정
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.Window, QColor(18, 18, 18))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(25, 25, 25))
    palette.setColor(palette.ToolTipBase, Qt.white)
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, Qt.white)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
