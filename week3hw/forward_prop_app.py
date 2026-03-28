import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="Forward-Prop Pro | Tensor Mapping Simulator",
    page_icon="🕸️",
    layout="wide"
)

# 커스텀 스타일 (매트릭스 테마)
st.markdown("""
    <style>
    .main { background-color: #080808; color: #00FF41; }
    .matrix-text { font-family: 'JetBrains Mono', monospace; color: #00FF41; background: #111; padding: 10px; border-radius: 5px; }
    div[data-testid="stMetricValue"] { color: #00FF41; }
    </style>
    """, unsafe_allow_html=True)

# --- 상태 및 연산 엔진 ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

# --- 사이드바: 텐서 제어판 ---
st.sidebar.title("🎛️ Tensor Controller")
st.sidebar.markdown("---")

with st.sidebar.expander("📥 Input Vector (x)", expanded=True):
    x1 = st.slider("x1", -2.0, 2.0, 0.5, 0.01)
    x2 = st.slider("x2", -2.0, 2.0, -0.5, 0.01)
    X = np.array([[x1], [x2]])

with st.sidebar.expander("🧠 Hidden Layer Weights (W1)"):
    # 전문가를 위한 가중치 미세 조정 (예시로 2개만 추출)
    w11 = st.slider("W1[0,0]", -2.0, 2.0, 0.85, 0.01)
    w12 = st.slider("W1[0,1]", -2.0, 2.0, -0.42, 0.01)
    
    # 나머지 가중치는 고정 또는 랜덤 생성 (시뮬레이션 용)
    W1 = np.array([
        [w11, w12],
        [0.5, 1.2],
        [-0.8, 0.3]
    ])
    b1 = np.array([[0.1], [-0.2], [0.0]])

with st.sidebar.expander("📤 Output Layer Weights (W2)"):
    w21 = st.slider("W2[0,0]", -2.0, 2.0, 1.1, 0.01)
    W2 = np.array([[w21, -0.5, 0.8]])
    b2 = np.array([[0.1]])

# --- 순전파 연산 ---
# Layer 1: R^2 -> R^3
Z1 = np.dot(W1, X) + b1
A1 = sigmoid(Z1)

# Layer 2: R^3 -> R^1
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)

# --- 메인 UI 스테이지 ---
st.title("Forward-Prop Pro: 2-3-1 Dimension Mapping")
st.markdown("### $\mathbb{R}^2$ (Input) $\longrightarrow$ $\mathbb{R}^3$ (Hidden) $\longrightarrow$ $\mathbb{R}^1$ (Output)")

# 1. 행렬 연산 HUD (LaTeX)
st.markdown("#### 📟 Matrix HUD: $\mathbf{a}^{(1)} = \sigma(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$")
st.latex(rf"""
\begin{bmatrix} a_1 \\ a_2 \\ a_3 \end{bmatrix} = \sigma \left(
\begin{bmatrix} {W1[0,0]:.2f} & {W1[0,1]:.2f} \\ {W1[1,0]:.2f} & {W1[1,1]:.2f} \\ {W1[2,0]:.2f} & {W1[2,1]:.2f} \end{bmatrix}
\begin{bmatrix} {X[0,0]:.2f} \\ {X[1,0]:.2f} \end{bmatrix} + 
\begin{bmatrix} {b1[0,0]:.2f} \\ {b1[1,0]:.2f} \\ {b1[2,0]:.2f} \end{bmatrix}
\right) = 
\begin{bmatrix} {A1[0,0]:.2f} \\ {A1[1,0]:.2f} \\ {A1[2,0]:.2f} \end{bmatrix}
""")

col1, col2 = st.columns([3, 2])

with col1:
    # 2. 네트워크 토폴로지 시각화 (Plotly)
    fig = go.Figure()
    
    # 노드 좌표
    nodes = {
        'in': [(0, 1), (0, -1)],
        'hid': [(1, 1.5), (1, 0), (1, -1.5)],
        'out': [(2, 0)]
    }
    
    # 엣지 드로잉 (가중치 기반)
    # In -> Hid
    for i, p1 in enumerate(nodes['in']):
        for j, p2 in enumerate(nodes['hid']):
            w = W1[j, i]
            fig.add_trace(go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                mode='lines',
                line=dict(color='#00FF41' if w > 0 else '#FF3131', width=abs(w)*5, dash='dash' if w < 0 else 'solid'),
                opacity=0.4, hoverinfo='none'
            ))
            
    # Hid -> Out
    for i, p1 in enumerate(nodes['hid']):
        w = W2[0, i]
        fig.add_trace(go.Scatter(
            x=[p1[0], p2[0] if 'p2' in locals() else 2], y=[p1[1], 0],
            mode='lines',
            line=dict(color='#00FF41' if w > 0 else '#FF3131', width=abs(w)*5),
            opacity=0.4, hoverinfo='none'
        ))

    # 노드 포인트 (활성화 값 기반 크기/색상)
    # Input
    fig.add_trace(go.Scatter(
        x=[p[0] for p in nodes['in']], y=[p[1] for p in nodes['in']],
        mode='markers+text', text=["x1", "x2"], textposition="middle left",
        marker=dict(size=30, color='#00FF41', line=dict(width=2, color='white')),
        name="Input"
    ))
    # Hidden
    fig.add_trace(go.Scatter(
        x=[p[0] for p in nodes['hid']], y=[p[1] for p in nodes['hid']],
        mode='markers+text', text=["h1", "h2", "h3"], textposition="top center",
        marker=dict(size=[float(a)*50+20 for a in A1], color='#00A3FF', line=dict(width=2, color='white')),
        name="Hidden"
    ))
    # Output
    fig.add_trace(go.Scatter(
        x=[2], y=[0],
        mode='markers+text', text=["y"], textposition="middle right",
        marker=dict(size=float(A2)*60+20, color='#FFFF00', line=dict(width=2, color='white')),
        name="Output"
    ))

    fig.update_layout(
        title="Network Topology & Active Flows",
        template="plotly_dark", showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # 3. 레이어별 활성화 막대 차트
    st.markdown("#### 📊 Layer Activations")
    
    # Hidden Layer Bar
    bar_hid = go.Figure(go.Bar(
        x=['h1', 'h2', 'h3'], y=A1.flatten(),
        marker_color='#00A3FF', text=[f"{float(a):.2f}" for a in A1], textposition='auto'
    ))
    bar_hid.update_layout(title="Hidden State Vector (a1)", template="plotly_dark", height=250, yaxis_range=[0, 1])
    st.plotly_chart(bar_hid, use_container_width=True)
    
    # Output Gauge
    st.markdown("#### 🎯 Final Output")
    st.metric("Prediction (y)", f"{float(A2):.4f}")
    st.progress(float(A2))

# --- 분석 통찰 ---
st.markdown("---")
cols = st.columns(3)
with cols[0]:
    st.info("**Dimensionality:** 2D $\\to$ 3D mapping expands features into higher space.")
with cols[1]:
    st.success("**Matrix Synergy:** Weights act as basis transformation matrices.")
with cols[2]:
    st.warning("**Non-linearity:** Sigmoid squashes the linear sum into [0, 1] range.")

st.markdown("💡 **Expert Tip:** 슬라이더를 조작할 때 행렬 수식 HUD의 숫자가 어떻게 실시간으로 변하며 최종 출력값(y)에 도달하는지 '차원의 흐름'을 관찰하세요.")
