import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json

def solve(params: dict) -> dict:
    # 1. Constants and Parameters
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    AU = 1.496e11     # m
    DAY = 86400       # s
    
    a_au = float(params.get('semi_major_axis', 1.0))
    e = float(params.get('eccentricity', 0.2))
    m_star = float(params.get('mass_star', 1.989e30))
    
    mu = G * m_star
    a = a_au * AU
    
    # 2. Initial Conditions (at Perihelion)
    r_p = a * (1 - e)
    v_p = np.sqrt(mu * (1 + e) / (a * (1 - e)))
    
    # State: [x, y, vx, vy]
    state0 = np.array([r_p, 0.0, 0.0, v_p])
    
    # 3. Simulation Time and Step
    T_theory = 2 * np.pi * np.sqrt(a**3 / mu)
    t_end = 1.05 * T_theory
    dt = T_theory / 2000 # 2000 steps per orbit
    
    t = np.arange(0, t_end, dt)
    n_steps = len(t)
    
    states = np.zeros((n_steps, 4))
    states[0] = state0
    
    # 4. RK4 Integration
    def derivatives(s):
        x, y, vx, vy = s
        r = np.sqrt(x**2 + y**2)
        ax = -mu * x / r**3
        ay = -mu * y / r**3
        return np.array([vx, vy, ax, ay])
    
    for i in range(n_steps - 1):
        k1 = derivatives(states[i])
        k2 = derivatives(states[i] + 0.5 * dt * k1)
        k3 = derivatives(states[i] + 0.5 * dt * k2)
        k4 = derivatives(states[i] + dt * k3)
        states[i+1] = states[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    x_orbit = states[:, 0]
    y_orbit = states[:, 1]
    
    # 5. Calculate Period from Simulation
    # Look for the first time it returns to y ~ 0 with vy > 0 after some time
    # Actually, it starts at y=0, vy > 0. It will cross y=0 again at half period (vy < 0).
    # Then cross again at full period (vy > 0).
    crossings = []
    for i in range(1, n_steps):
        if states[i-1, 1] < 0 and states[i, 1] >= 0:
            crossings.append(t[i])
            if len(crossings) >= 1: break
            
    T_sim = crossings[0] if crossings else T_theory
    period_days = T_sim / DAY
    
    # 6. Kepler Check
    kepler_ratio = (T_sim**2) / (a_au**3) # T^2 / a^3 in (seconds^2 / AU^3)
    # Theoretical ratio: (2*pi*sqrt(a^3/mu))^2 / (a/AU)^3 = 4*pi^2 * AU^3 / mu
    kepler_ratio_theory = 4 * np.pi**2 * AU**3 / mu
    deviation_pct = abs(kepler_ratio - kepler_ratio_theory) / kepler_ratio_theory * 100
    
    # 7. Generate Data for T^2 vs a^3 plot
    # Use 3 eccentricities as requested: 0.1, 0.4, 0.7
    a_values = np.linspace(0.5, 2.5, 5) # AU
    e_values = [0.1, 0.4, 0.7]
    
    kepler_plot_data = []
    for e_val in e_values:
        a_line = []
        t2_line = []
        for a_val in a_values:
            a_m = a_val * AU
            t_val = 2 * np.pi * np.sqrt(a_m**3 / mu)
            a_line.append(a_val**3)
            t2_line.append(t_val**2)
        kepler_plot_data.append((e_val, a_line, t2_line))

    # 8. Plotting
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Orbital Trajectory (x-y plane)", "Kepler's Third Law: T² vs a³")
    )
    
    # Trajectory
    fig.add_trace(go.Scatter(x=x_orbit/AU, y=y_orbit/AU, name=f"Orbit (e={e})", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[0], name="Star", mode='markers', marker=dict(size=12, color='orange')), row=1, col=1)
    fig.update_xaxes(title_text="x (AU)", row=1, col=1)
    fig.update_yaxes(title_text="y (AU)", row=1, col=1, scaleanchor="x", scaleratio=1)
    
    # Kepler Law
    for e_val, a3, t2 in kepler_plot_data:
        fig.add_trace(go.Scatter(x=a3, y=t2, name=f"e={e_val}", mode='lines+markers'), row=1, col=2)
    
    # Add the current orbit point
    fig.add_trace(go.Scatter(x=[a_au**3], y=[T_sim**2], name="Current Orbit", mode='markers', marker=dict(size=10, color='red', symbol='star')), row=1, col=2)
    
    fig.update_xaxes(title_text="a³ (AU³)", row=1, col=2)
    fig.update_yaxes(title_text="T² (s²)", row=1, col=2)
    
    fig.update_layout(height=500, title_text="Planetary Motion and Kepler's Laws")
    
    # 9. Results
    plotly_json = json.loads(pio.to_json(fig))
    
    numerical_summary = {
        "period_days": float(period_days),
        "kepler_ratio": float(kepler_ratio),
        "kepler_deviation_percent": float(deviation_pct)
    }
    
    steps = [
        "1. 행성의 초기 위치를 근일점(Perihelion)으로 설정하고, 해당 지점에서의 궤도 속도를 공식으로부터 유도합니다.",
        "2. 중력 가속도 a = -GM/r²를 바탕으로 하는 2체 문제(Two-body problem)의 미분 방정식을 수립합니다.",
        "3. RK4 수치 해석 기법을 사용하여 시간에 따른 행성의 위치(x, y)와 속도(vx, vy)를 계산합니다.",
        "4. 시뮬레이션 결과로부터 행성이 한 바퀴를 도는 데 걸리는 실제 주기(T)를 측정합니다.",
        "5. 케플러 제3법칙(조화의 법칙)인 T²/a³ = 상수 임을 확인하고, 이론값과의 오차율을 계산하여 수치 해석의 정확도를 검증합니다."
    ]
    
    return {
        "plotly_json": plotly_json,
        "numerical_summary": numerical_summary,
        "steps": steps
    }
