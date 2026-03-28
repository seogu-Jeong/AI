import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp

# 페이지 설정
st.set_page_config(
    page_title="ActiViz Pro | Activation & Gradient Dynamics",
    page_icon="📉",
    layout="wide"
)

# 커스텀 스타일링
st.markdown("""
    <style>
    .main { background-color: #050505; color: #E0E0E0; }
    .stTextInput>div>div>input { color: #00FFCC; font-family: 'Consolas'; }
    .metric-card { background-color: #111; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 수치 해석 엔진 (SymPy & NumPy) ---
def get_activation_and_derivative(func_name, custom_expr=None, gamma=0.5):
    x = sp.Symbol('x')
    
    if func_name == "Sigmoid":
        f = 1 / (1 + sp.exp(-x))
    elif func_name == "Tanh":
        f = sp.tanh(x)
    elif func_name == "ReLU":
        f = sp.Piecewise((0, x < 0), (x, x >= 0))
    elif func_name == "Leaky ReLU":
        alpha = st.sidebar.slider("Alpha (α)", 0.0, 0.5, 0.1)
        f = sp.Piecewise((alpha * x, x < 0), (x, x >= 0))
    elif func_name == "Custom" and custom_expr:
        try:
            f = sp.sympify(custom_expr.replace('^', '**'))
        except:
            st.error("Invalid Mathematical Expression")
            f = x
    else:
        f = x

    # 기호 미분
    f_prime = sp.diff(f, x)
    
    # NumPy 함수로 변환 (벡터 연산 최적화)
    f_num = sp.lambdify(x, f, 'numpy')
    f_prime_num = sp.lambdify(x, f_prime, 'numpy')
    
    return f, f_prime, f_num, f_prime_num

# --- 사이드바 컨트롤 ---
st.sidebar.title("📉 Gradient Dynamics")
st.sidebar.markdown("---")

with st.sidebar.expander("🛠️ Function Configuration", expanded=True):
    func_choice = st.selectbox("Activation Function", ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "Custom"])
    custom_input = ""
    if func_choice == "Custom":
        custom_input = st.text_input("Enter f(x):", value="x * sin(x)")
    
    gamma = st.slider("Sub-gradient at x=0 (γ)", 0.0, 1.0, 0.5)

with st.sidebar.expander("🔗 Multi-layer Flow"):
    num_layers = st.number_input("Stack Depth (Layers)", 1, 5, 1)
    loss_gradient = st.slider("Initial Loss Gradient", 0.1, 2.0, 1.0)

# --- 연산 실행 ---
f_sym, fp_sym, f_func, fp_func = get_activation_and_derivative(func_choice, custom_input, gamma)

x_range = np.linspace(-10, 10, 1000)
y_vals = f_func(x_range)
# ReLU 등 불연속점 수치 보정
if func_choice in ["ReLU", "Leaky ReLU"]:
    y_vals = np.maximum(0, x_range) if func_choice == "ReLU" else y_vals
    
dy_vals = fp_func(x_range)
if isinstance(dy_vals, (int, float)): # 상수 도함수 처리
    dy_vals = np.full_like(x_range, dy_vals)

# --- 메인 UI ---
st.title("ActiViz Pro: Activation & Gradient Explorer")
st.latex(rf"f(x) = {sp.latex(f_sym)}")

# 현재 분석 지점 제어
current_x = st.slider("Target Analysis Point (x₀)", -10.0, 10.0, 0.0, 0.01)
current_y = float(f_func(current_x))
current_dy = float(fp_func(current_x))

# --- 고성능 차트 렌더링 (Linked Views) ---
fig = make_subplots(rows=1, cols=2, subplot_titles=("Primal Space: f(x)", "Dual Space: f'(x) / Gradient Flow"))

# 1. 원형 함수 그래프
fig.add_trace(go.Scatter(x=x_range, y=y_vals, name="Activation", line=dict(color='#00FFCC', width=3)), row=1, col=1)
# 접선(Tangent Line)
t_x = np.linspace(current_x - 2, current_x + 2, 10)
t_y = current_dy * (t_x - current_x) + current_y
fig.add_trace(go.Scatter(x=t_x, y=t_y, name="Tangent", line=dict(color='yellow', width=2, dash='dot')), row=1, col=1)
# 현재 포인트 점
fig.add_trace(go.Scatter(x=[current_x], y=[current_y], mode='markers', marker=dict(color='white', size=10)), row=1, col=1)

# 2. 도함수 그래프
fig.add_trace(go.Scatter(x=x_range, y=dy_vals, name="Derivative", line=dict(color='#FF007F', width=3)), row=1, col=2)
# 현재 기울기 위치
fig.add_trace(go.Scatter(x=[current_x], y=[current_dy], mode='markers', marker=dict(color='white', size=10)), row=1, col=2)

fig.update_layout(template="plotly_dark", height=500, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# --- 멀티레이어 전파 분석 (Chain Rule) ---
st.markdown("### ⛓️ Deep Stack Gradient Propagation")
if num_layers > 1:
    grads = [current_dy]
    # 단순화를 위해 동일 함수 중첩 시뮬레이션
    for i in range(1, num_layers):
        # 레이어를 통과할수록 입력값이 변함 (f(f(x))...)
        input_x = current_x
        for _ in range(i):
            input_x = float(f_func(input_x))
        grads.append(float(fp_func(input_x)))
    
    # 연쇄 법칙 결과
    total_grad = loss_gradient * np.prod(grads)
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        # 기울기 감쇄 막대 그래프
        grad_fig = go.Figure(go.Bar(
            x=[f"L{i+1}" for i in range(num_layers)],
            y=grads,
            marker_color=['#FF007F' if g < 0.1 else '#00FFCC' for g in grads]
        ))
        grad_fig.update_layout(title="Gradient Pass-through per Layer", template="plotly_dark", height=300)
        st.plotly_chart(grad_fig, use_container_width=True)
    
    with col_b:
        st.write("#### Resulting Gradient")
        st.metric("Effective ΔW", f"{total_grad:.6f}")
        if total_grad < 1e-4:
            st.error("⚠️ **Vanishing Gradient Detected!** 기울기가 너무 작아 학습이 거의 불가능합니다.")
        elif total_grad > 5.0:
            st.warning("💥 **Exploding Gradient!** 기울기가 폭주하여 시스템이 불안정할 수 있습니다.")
        else:
            st.success("✅ **Healthy Flow.** 기울기가 적절히 전달되고 있습니다.")

else:
    st.info("사이드바에서 Layer 개수를 늘려 '심층 기울기 소실' 현상을 시뮬레이션해 보세요.")

# --- 수치적 통찰 ---
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"**Locality Analysis**")
    st.write(f"Saturation: {'Yes' if abs(current_dy) < 0.05 else 'No'}")
with c2:
    st.markdown(f"**Sub-gradient @ 0**")
    st.write(f"Defined Value: {gamma if current_x == 0 else 'N/A'}")
with c3:
    st.markdown(f"**Mathematical Property**")
    is_mono = "Yes" if np.all(np.diff(y_vals) >= -1e-5) else "No"
    st.write(f"Monotonic: {is_mono}")
