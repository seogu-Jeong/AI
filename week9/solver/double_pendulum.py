import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.integrate import solve_ivp
import json

def solve(params: dict) -> dict:
    # 1. Constants and Parameters
    g = 9.81
    
    t1_deg = float(params.get('theta1_deg', 120.0))
    t2_deg = float(params.get('theta2_deg', -30.0))
    l1 = float(params.get('l1', 1.0))
    l2 = float(params.get('l2', 1.0))
    m1 = float(params.get('m1', 1.0))
    m2 = float(params.get('m2', 1.0))
    t_end = float(params.get('t_end', 20.0))
    compare_chaos = bool(params.get('compare_chaos', True))
    
    t1 = np.radians(t1_deg)
    t2 = np.radians(t2_deg)
    
    # 2. Equations of Motion
    def double_pendulum_derivs(t, state, l1, l2, m1, m2):
        th1, w1, th2, w2 = state
        
        delta = th1 - th2
        den = 2*m1 + m2 - m2*np.cos(2*delta)
        
        # d(th1)/dt = w1
        # d(w1)/dt
        dw1 = (-g*(2*m1 + m2)*np.sin(th1) - m2*g*np.sin(th1 - 2*th2) - 
               2*np.sin(delta)*m2*(w2**2*l2 + w1**2*l1*np.cos(delta))) / (l1*den)
        
        # d(th2)/dt = w2
        # d(w2)/dt
        dw2 = (2*np.sin(delta)*(w1**2*l1*(m1 + m2) + g*(m1 + m2)*np.cos(th1) + 
                               w2**2*l2*m2*np.cos(delta))) / (l2*den)
        
        return [w1, dw1, w2, dw2]

    # 3. Solver
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, int(t_end * 100))
    
    state0 = [t1, 0.0, t2, 0.0]
    sol1 = solve_ivp(double_pendulum_derivs, t_span, state0, 
                     args=(l1, l2, m1, m2), t_eval=t_eval, 
                     method='RK45', max_step=0.01)
    
    # 4. Chaos Comparison
    sol2 = None
    if compare_chaos:
        state0_perturbed = [t1 + np.radians(0.01), 0.0, t2, 0.0]
        sol2 = solve_ivp(double_pendulum_derivs, t_span, state0_perturbed, 
                         args=(l1, l2, m1, m2), t_eval=t_eval, 
                         method='RK45', max_step=0.01)

    # 5. Cartesian Conversion
    def get_cartesian(sol):
        th1 = sol.y[0]
        th2 = sol.y[2]
        x1 = l1 * np.sin(th1)
        y1 = -l1 * np.cos(th1)
        x2 = x1 + l2 * np.sin(th2)
        y2 = y1 - l2 * np.cos(th2)
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = get_cartesian(sol1)
    
    # 6. Plotting
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Pendulum Tip 2 Trajectory (Time-colored)", 
            "Angles vs Time", 
            "Phase Space (θ₁ vs ω₁)", 
            "Chaos Divergence: log(|Δθ₁|)" if compare_chaos else "Energy Conservation Check"
        )
    )
    
    # [1,1] Trajectory of tip 2
    fig.add_trace(go.Scatter(
        x=x2_1, y=y2_1,
        mode='lines',
        line=dict(width=1, color='rgba(0,0,0,0.2)'),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x2_1, y=y2_1,
        mode='markers',
        marker=dict(
            size=3,
            color=sol1.t,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time (s)", x=0.45)
        ),
        name="Tip 2 path"
    ), row=1, col=1)
    fig.update_xaxes(title_text="x (m)", row=1, col=1)
    fig.update_yaxes(title_text="y (m)", row=1, col=1, scaleanchor="x", scaleratio=1)
    
    # [1,2] Angles vs Time
    fig.add_trace(go.Scatter(x=sol1.t, y=np.degrees(sol1.y[0]), name="θ₁ (deg)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=sol1.t, y=np.degrees(sol1.y[2]), name="θ₂ (deg)"), row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Angle (deg)", row=1, col=2)
    
    # [2,1] Phase Space
    fig.add_trace(go.Scatter(x=sol1.y[0], y=sol1.y[1], name="θ₁ vs ω₁"), row=2, col=1)
    fig.update_xaxes(title_text="θ₁ (rad)", row=2, col=1)
    fig.update_yaxes(title_text="ω₁ (rad/s)", row=2, col=1)
    
    # [2,2] Chaos or Energy
    if compare_chaos and sol2 is not None:
        delta_theta = np.abs(sol1.y[0] - sol2.y[0])
        # Add small constant to avoid log(0)
        log_delta = np.log10(delta_theta + 1e-15)
        fig.add_trace(go.Scatter(x=sol1.t, y=log_delta, name="log₁₀|Δθ₁|", line=dict(color='red')), row=2, col=2)
        fig.update_yaxes(title_text="log₁₀|Δθ₁|", row=2, col=2)
    else:
        # Energy calculation
        th1, w1, th2, w2 = sol1.y
        T = 0.5 * m1 * (l1*w1)**2 + 0.5 * m2 * ((l1*w1)**2 + (l2*w2)**2 + 2*l1*l2*w1*w2*np.cos(th1-th2))
        V = -m1*g*l1*np.cos(th1) - m2*g*(l1*np.cos(th1) + l2*np.cos(th2))
        E = T + V
        fig.add_trace(go.Scatter(x=sol1.t, y=E, name="Total Energy"), row=2, col=2)
        fig.update_yaxes(title_text="Energy (J)", row=2, col=2)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_layout(height=800, title_text="Double Pendulum: Chaotic Dynamics")
    
    # 7. Results
    plotly_json = json.loads(pio.to_json(fig))
    
    # Numerical Summary
    max_theta1 = float(np.max(np.abs(np.degrees(sol1.y[0]))))
    max_theta2 = float(np.max(np.abs(np.degrees(sol1.y[2]))))
    
    lyapunov_estimate = 0.0
    if compare_chaos and sol2 is not None:
        # Simple estimate from the slope of log divergence
        # Use first 5 seconds where it's typically linear-ish before saturation
        mask = sol1.t < 10.0
        if np.any(mask):
            fit = np.polyfit(sol1.t[mask], log_delta[mask], 1)
            lyapunov_estimate = float(fit[0]) # slope in log10 units per second

    numerical_summary = {
        "max_angle1_deg": max_theta1,
        "max_angle2_deg": max_theta2,
        "lyapunov_exponent_estimate_log10": lyapunov_estimate,
        "total_steps": int(len(sol1.t))
    }
    
    steps = [
        "1. 이중 진자의 라그랑지안(Lagrangian)으로부터 비선형 연립 미분 방정식을 유도합니다.",
        "2. 초기 각도 θ₁, θ₂와 각속도를 설정하고, 미세한 변화(0.01°)를 준 또 다른 초기 조건을 생성합니다.",
        "3. Scipy의 solve_ivp(RK45)를 사용하여 시간에 따른 두 진자의 궤적을 고정밀도로 계산합니다.",
        "4. 극좌표계의 결과를 직교 좌표계(x, y)로 변환하여 2번 추의 카오스적인 운동 경로를 시각화합니다.",
        "5. 두 초기 조건 사이의 각도 차이(Δθ)가 시간에 따라 지수적으로 발산하는지 확인하여 결정론적 혼돈(Deterministic Chaos)을 분석합니다."
    ]
    
    return {
        "plotly_json": plotly_json,
        "numerical_summary": numerical_summary,
        "steps": steps
    }
