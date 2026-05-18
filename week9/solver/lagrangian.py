import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.integrate import solve_ivp
import json

def solve(params: dict) -> dict:
    # 1. Constants and Parameters
    g = 9.81
    
    L = float(params.get('length', 1.0))
    m = float(params.get('mass', 1.0))
    t0_deg = float(params.get('theta0_deg', 60.0))
    w0 = float(params.get('omega0', 0.0))
    t_end = float(params.get('t_end', 10.0))
    
    theta0 = np.radians(t0_deg)
    
    # 2. Formulations
    
    # Newtonian & Lagrangian (same ODE for simple pendulum)
    def newton_lagrange_derivs(t, state):
        theta, omega = state
        dtheta = omega
        domega = -(g / L) * np.sin(theta)
        return [dtheta, domega]
    
    # Hamiltonian
    # State: [theta, p_theta] where p_theta = m*L^2*omega
    def hamilton_derivs(t, state):
        theta, p = state
        dtheta = p / (m * L**2)
        dp = -m * g * L * np.sin(theta)
        return [dtheta, dp]
    
    # 3. Solver
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 500)
    
    # solve Newtonian
    sol_n = solve_ivp(newton_lagrange_derivs, t_span, [theta0, w0], t_eval=t_eval, method='RK45')
    
    # solve Hamiltonian
    p0 = m * L**2 * w0
    sol_h = solve_ivp(hamilton_derivs, t_span, [theta0, p0], t_eval=t_eval, method='RK45')
    
    # Convert Hamiltonian momentum back to omega for comparison
    omega_h = sol_h.y[1] / (m * L**2)
    
    # 4. Analytical (Small Angle Approximation)
    # theta(t) = theta0 * cos(sqrt(g/L)*t) + (w0/sqrt(g/L)) * sin(sqrt(g/L)*t)
    omega_n = np.sqrt(g / L)
    theta_small = theta0 * np.cos(omega_n * t_eval) + (w0 / omega_n) * np.sin(omega_n * t_eval)
    
    # 5. Energy Calculation
    def get_energy(theta, omega):
        ke = 0.5 * m * (L * omega)**2
        pe = m * g * L * (1 - np.cos(theta))
        return ke + pe
    
    energy_n = get_energy(sol_n.y[0], sol_n.y[1])
    energy_h = get_energy(sol_h.y[0], omega_h)
    
    # 6. Plotting
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Angle (θ) vs Time: Method Comparison", 
            "Phase Space (θ vs ω)", 
            "Total Mechanical Energy Conservation", 
            "Numerical Difference: |θ_Newton - θ_Hamilton|"
        )
    )
    
    # [1,1] Theta vs Time
    fig.add_trace(go.Scatter(x=t_eval, y=np.degrees(sol_n.y[0]), name="Newtonian/Lagrangian", line=dict(dash='solid')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_eval, y=np.degrees(sol_h.y[0]), name="Hamiltonian", line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_eval, y=np.degrees(theta_small), name="Small Angle Approx", line=dict(dash='dash', color='gray')), row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Angle (deg)", row=1, col=1)
    
    # [1,2] Phase Space
    fig.add_trace(go.Scatter(x=sol_n.y[0], y=sol_n.y[1], name="Newtonian (θ vs ω)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=sol_h.y[0], y=omega_h, name="Hamiltonian (θ vs ω)", line=dict(dash='dot')), row=1, col=2)
    fig.update_xaxes(title_text="θ (rad)", row=1, col=2)
    fig.update_yaxes(title_text="ω (rad/s)", row=1, col=2)
    
    # [2,1] Energy
    fig.add_trace(go.Scatter(x=t_eval, y=energy_n, name="Energy (Newton)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_eval, y=energy_h, name="Energy (Hamilton)", line=dict(dash='dot')), row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Energy (J)", row=2, col=1)
    
    # [2,2] Error Comparison
    error = np.abs(sol_n.y[0] - sol_h.y[0])
    fig.add_trace(go.Scatter(x=t_eval, y=error, name="Numerical Error", line=dict(color='red')), row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="|Δθ| (rad)", row=2, col=2)
    
    fig.update_layout(height=800, title_text=f"Pendulum Analysis: Multi-Formalism Comparison (θ₀={t0_deg}°, L={L}m)")
    
    # 7. Results
    plotly_json = json.loads(pio.to_json(fig))
    
    numerical_summary = {
        "final_energy_joules": float(energy_n[-1]),
        "energy_drift_percent": float(abs(energy_n[-1] - energy_n[0]) / energy_n[0] * 100),
        "max_numerical_diff_rad": float(np.max(error)),
        "small_angle_max_error_deg": float(np.max(np.abs(np.degrees(sol_n.y[0] - theta_small))))
    }
    
    steps = [
        "1. 뉴턴 역학(F=ma), 라그랑주 역학(L=T-V), 해밀턴 역학(H=T+V)의 세 가지 방식으로 단진자의 운동 방정식을 수립합니다.",
        "2. 라그랑주 역학의 오일러-라그랑주 방정식을 통해 뉴턴 역학과 동일한 2차 미분 방정식을 유도함을 확인합니다.",
        "3. 해밀턴 역학에서는 각도(θ)와 일반화 운동량(p)을 상태 변수로 하는 1차 연립 미분 방정식을 수립합니다.",
        "4. Scipy의 RK45 수치 해석기를 사용하여 세 가지 공식화 방법이 동일한 물리적 궤적을 생성함을 검증합니다.",
        "5. 시뮬레이션 결과와 소각 근사(Small-angle approximation) 해를 비교하여, 초기 각도가 커짐에 따라 비선형성이 어떻게 나타나는지 분석합니다."
    ]
    
    return {
        "plotly_json": plotly_json,
        "numerical_summary": numerical_summary,
        "steps": steps
    }
