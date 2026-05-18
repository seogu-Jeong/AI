import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json

def solve(params: dict) -> dict:
    # 1. Parameter extraction
    omega = float(params.get('omega', 2.0))
    dt = float(params.get('dt', 0.05))
    t_end = float(params.get('t_end', 10.0))
    x0 = float(params.get('x0', 1.0))
    v0 = float(params.get('v0', 0.0))

    t = np.arange(0, t_end + dt, dt)
    n_steps = len(t)

    # 2. Numerical Integration
    # Euler Method
    x_euler = np.zeros(n_steps)
    v_euler = np.zeros(n_steps)
    x_euler[0], v_euler[0] = x0, v0

    # RK4 Method
    x_rk4 = np.zeros(n_steps)
    v_rk4 = np.zeros(n_steps)
    x_rk4[0], v_rk4[0] = x0, v0

    for i in range(n_steps - 1):
        # Euler
        x_euler[i+1] = x_euler[i] + dt * v_euler[i]
        v_euler[i+1] = v_euler[i] + dt * (-omega**2 * x_euler[i])

        # RK4
        def derivatives(x, v):
            return v, -omega**2 * x

        k1_x, k1_v = derivatives(x_rk4[i], v_rk4[i])
        k2_x, k2_v = derivatives(x_rk4[i] + 0.5 * dt * k1_x, v_rk4[i] + 0.5 * dt * k1_v)
        k3_x, k3_v = derivatives(x_rk4[i] + 0.5 * dt * k2_x, v_rk4[i] + 0.5 * dt * k2_v)
        k4_x, k4_v = derivatives(x_rk4[i] + dt * k3_x, v_rk4[i] + dt * k3_v)

        x_rk4[i+1] = x_rk4[i] + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v_rk4[i+1] = v_rk4[i] + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    # 3. Analytic Solution
    x_analytic = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    v_analytic = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)

    # 4. Energy Calculation
    def get_energy(x, v):
        return 0.5 * v**2 + 0.5 * omega**2 * x**2

    e_euler = get_energy(x_euler, v_euler)
    e_rk4 = get_energy(x_rk4, v_rk4)
    e_analytic = get_energy(x_analytic, v_analytic) # Constant E0
    e0 = e_analytic[0]

    err_euler = np.abs(e_euler - e0) / e0
    err_rk4 = np.abs(e_rk4 - e0) / e0

    # 5. Plotting
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Position vs Time", "Velocity vs Time", 
                        "Energy Error (Log Scale)", "Phase Space (x vs v)")
    )

    # Position
    fig.add_trace(go.Scatter(x=t, y=x_analytic, name="Analytic", line=dict(color='green', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=x_euler, name="Euler", line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=x_rk4, name="RK4", line=dict(color='blue')), row=1, col=1)

    # Velocity
    fig.add_trace(go.Scatter(x=t, y=v_analytic, name="Analytic", line=dict(color='green', dash='dot'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=v_euler, name="Euler", line=dict(color='red', dash='dash'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=v_rk4, name="RK4", line=dict(color='blue'), showlegend=False), row=1, col=2)

    # Energy Error
    fig.add_trace(go.Scatter(x=t, y=err_euler, name="Euler Error", line=dict(color='red', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=err_rk4, name="RK4 Error", line=dict(color='blue')), row=2, col=1)
    fig.update_yaxes(type="log", row=2, col=1)

    # Phase Space
    fig.add_trace(go.Scatter(x=x_analytic, y=v_analytic, name="Analytic", line=dict(color='green', dash='dot'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_euler, y=v_euler, name="Euler", line=dict(color='red', dash='dash'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_rk4, y=v_rk4, name="RK4", line=dict(color='blue'), showlegend=False), row=2, col=2)

    fig.update_layout(height=800, title_text=f"Harmonic Oscillator: Euler vs RK4 (dt={dt})")

    # 6. Results
    plotly_json = json.loads(pio.to_json(fig))
    
    numerical_summary = {
        "euler_energy_error_final": float(err_euler[-1]),
        "rk4_energy_error_final": float(err_rk4[-1]),
        "method_comparison": "RK4 shows significantly better energy conservation than Euler method." if err_rk4[-1] < err_euler[-1] else "Euler and RK4 performance is comparable at this step size."
    }

    steps = [
        "1. 조화 진동자의 운동 방정식을 F = -kx = ma 형태로 정의합니다.",
        "2. 오일러 방법(Euler method)은 현재의 기울기를 사용하여 다음 단계의 위치와 속도를 예측합니다.",
        "3. RK4(Runge-Kutta 4th order) 방법은 한 단계 내에서 네 번의 기울기를 샘플링하여 가중 평균을 사용함으로써 훨씬 더 높은 정확도를 제공합니다.",
        "4. 에너지 보존 법칙을 통해 수치 해석의 정확도를 검증합니다. 오일러 방법은 시간이 지남에 따라 에너지가 발산하는 경향이 있는 반면, RK4는 에너지를 매우 잘 보존합니다.",
        "5. 위상 공간(Phase Space) 그래프에서 오일러 방법은 나선형으로 바깥으로 벗어나는 것을 확인할 수 있지만, RK4는 닫힌 궤도를 유지합니다."
    ]

    return {
        "plotly_json": plotly_json,
        "numerical_summary": numerical_summary,
        "steps": steps
    }
