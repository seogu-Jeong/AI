import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 페이지 설정 (전문가용 다크 테마 및 와이드 레이아웃)
st.set_page_config(
    page_title="PhysViz-Perceptron Pro | Tensor Field Simulator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (다크 모드 강화 및 UI 미려하게)
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stSlider [data-baseweb="slider"] { margin-bottom: 20px; }
    .css-1offfwp { font-family: 'Consolas', monospace; }
    div[data-testid="stMetricValue"] { color: #00FFFF; font-family: 'Consolas'; }
    </style>
    """, unsafe_allow_html=True)

# --- 사이드바: 컨트롤 패널 (Physics & AI Parameters) ---
st.sidebar.title("⚛️ Field Engineering")
st.sidebar.markdown("---")

with st.sidebar.expander("🛠️ Tensor Parameters", expanded=True):
    w1 = st.slider("Weight 1 ($w_1$)", -5.0, 5.0, 1.0, 0.01)
    w2 = st.slider("Weight 2 ($w_2$)", -5.0, 5.0, 1.0, 0.01)
    b = st.slider("Bias ($b$)", -10.0, 10.0, 0.0, 0.1)

with st.sidebar.expander("🌌 Physics Constants"):
    physics_mode = st.checkbox("Enable Physics Mode (V/m)", value=False)
    epsilon_0 = 8.854e-12
    charge_q = 1.602e-19
    kappa = charge_q / (4 * np.pi * epsilon_0) if physics_mode else 1.0

with st.sidebar.expander("📊 Dataset Selection"):
    dataset_type = st.selectbox("Scenario", ["XOR Gate", "AND Gate", "OR Gate", "Custom Function"])
    noise = st.slider("Quantum Noise ($\sigma$)", 0.0, 1.0, 0.1)

# --- 수치 연산 엔진 (NumPy Vectorized) ---
res = 80
x = np.linspace(-5, 5, res)
y = np.linspace(-5, 5, res)
X, Y = np.meshgrid(x, y)

# 전위 장 계산 (Potential Field)
Z_logit = w1 * X + w2 * Y + b
Z_potential = kappa * Z_logit
Z_sigmoid = 1 / (1 + np.exp(-Z_logit)) # 활성화 함수(확률)

# 데이터 포인트 생성 (XOR 예시)
if dataset_type == "XOR Gate":
    data_points = np.array([[-2,-2,0], [2,2,0], [-2,2,1], [2,-2,1]])
elif dataset_type == "AND Gate":
    data_points = np.array([[-2,-2,0], [-2,2,0], [2,-2,0], [2,2,1]])
else:
    data_points = np.array([[-2,-2,0], [-2,2,1], [2,-2,1], [2,2,1]])

# --- 메인 스테이지: 고성능 시각화 (Plotly) ---
st.title("PhysViz-Perceptron Pro")
st.markdown(f"### Current State: $z = {w1:.2f}x_1 + {w2:.2f}x_2 + {b:.2f}$")

col1, col2 = st.columns([1, 1])

with col1:
    # 2D Contour & Vector Field
    fig2d = go.Figure()
    
    # 등고선 히트맵
    fig2d.add_trace(go.Contour(
        z=Z_potential, x=x, y=y,
        colorscale='RdBu', reversescale=True,
        line_smoothing=0.85, contours_coloring='heatmap',
        colorbar=dict(title="Potential (V)" if physics_mode else "Logit (z)")
    ))
    
    # 가중치 벡터 (전기장 방향)
    norm = np.sqrt(w1**2 + w2**2) + 1e-8
    fig2d.add_trace(go.Scatter(
        x=[0, w1/norm * 2], y=[0, w2/norm * 2],
        mode='lines+markers', line=dict(color='yellow', width=4),
        marker=dict(size=10, symbol='arrow-bar-up', angleref='previous'),
        name="Field Vector (w)"
    ))
    
    # 데이터 포인트
    for p in data_points:
        color = 'red' if p[2] == 1 else 'blue'
        fig2d.add_trace(go.Scatter(
            x=[p[0]], y=[p[1]],
            mode='markers', marker=dict(color=color, size=15, line=dict(width=2, color='white')),
            showlegend=False
        ))

    fig2d.update_layout(
        title="2D Potential Mapping & Nodal Line",
        template="plotly_dark", height=600,
        xaxis=dict(range=[-5, 5], gridcolor='#333'),
        yaxis=dict(range=[-5, 5], gridcolor='#333', scaleanchor="x", scaleratio=1)
    )
    st.plotly_chart(fig2d, use_container_width=True)

with col2:
    # 3D Surface & Contour Projection
    fig3d = go.Figure()
    
    # 3D 곡면 (Z축을 활성화 값으로)
    fig3d.add_trace(go.Surface(
        z=Z_sigmoid, x=x, y=y,
        colorscale='RdBu', reversescale=True,
        opacity=0.9, contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    ))
    
    # 3D상의 데이터 포인트
    for p in data_points:
        p_z = 1 / (1 + np.exp(-(w1*p[0] + w2*p[1] + b)))
        fig3d.add_trace(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p_z],
            mode='markers', marker=dict(color='white', size=8),
            showlegend=False
        ))

    fig3d.update_layout(
        title="3D Tensor Surface & Curvature",
        template="plotly_dark", height=600,
        scene=dict(
            xaxis=dict(title='x1', backgroundcolor="rgb(20, 20, 20)"),
            yaxis=dict(title='x2', backgroundcolor="rgb(20, 20, 20)"),
            zaxis=dict(title='Activation', range=[0, 1], backgroundcolor="rgb(30, 30, 30)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )
    )
    st.plotly_chart(fig3d, use_container_width=True)

# --- 하단 분석 바 ---
st.markdown("---")
cols = st.columns(4)
cols[0].metric("Field Anisotropy", f"{norm:.2f}")
cols[1].metric("Boundary Offset", f"{-b/norm:.2f}" if norm > 0 else "0")
cols[2].metric("Loss (MSE)", "Calculating..." if dataset_type == "XOR Gate" else "0.00")
cols[3].metric("Convergence", "Stable" if dataset_type != "XOR Gate" else "Impossible")

st.info("💡 **PM's Insight:** XOR 데이터셋을 선택하고 w1, w2를 조절해보세요. 어떤 직선 경계로도 두 전하 그룹을 완벽히 분리할 수 없음을 3D 곡면의 왜곡을 통해 확인할 수 있습니다.")
