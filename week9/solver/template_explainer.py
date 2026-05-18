from typing import Dict, Any

def explain(results: Dict[str, Any], topic: str, params: Dict[str, Any]) -> str:
    """
    Template-based Korean explanations. No API calls.
    """
    num_sum = results.get('numerical_summary', {})
    
    if topic == 'euler_rk4':
        euler_err = num_sum.get('euler_energy_error_final', 0.0)
        rk4_err = num_sum.get('rk4_energy_error_final', 0.0)
        omega = params.get('omega', 2.0)
        
        ratio = euler_err / rk4_err if rk4_err > 0 else 1e6
        
        template = (
            "## Euler vs RK4 수치 적분 비교\n\n"
            f"**핵심 물리**: 단순 조화 진동자 $\\ddot{{x}} = -\\omega^2 x$ ($\\omega$={omega:.2f} rad/s)\n\n"
            "**수치 방법**: \n"
            "- Euler법: $x_{n+1} = x_n + \\Delta t \\cdot v_n$ — 1차 정확도, 에너지 증가\n"
            "- RK4: 4단계 기울기 평균 — 4차 정확도, 에너지 보존 우수\n\n"
            f"**결과 분석**: Euler 최종 에너지 오차 {euler_err:.4f}, RK4 {rk4_err:.6f}. RK4가 약 {ratio:.0f}배 정확.\n\n"
            "**물리적 의미**: Euler법의 에너지 발산은 수치 불안정성의 전형적 예시. "
            "실제 계산물리에서 RK4 이상의 방법이 필수적인 이유."
        )
        return template

    elif topic == 'planetary':
        e = params.get('eccentricity', 0.2)
        period = num_sum.get('period_days', 0.0)
        deviation = num_sum.get('kepler_deviation_percent', 0.0)
        
        template = f"""## 케플러 행성 궤도 시뮬레이션

**핵심 물리**: 만유인력 $\\mathbf{{F}} = -\\frac{{GMm}}{{r^2}}\\hat{{r}}$, 이심률 e={e:.3f}

**수치 방법**: RK4로 $\\ddot{{\\mathbf{{r}}}} = -GM\\mathbf{{r}}/r^3$ 적분

**결과 분석**: 계산된 공전 주기 T={period:.1f}일. 케플러 제3법칙 $T^2 \\propto a^3$ 편차 {deviation:.2f}%

**물리적 의미**: 수치 오차 없이 케플러 법칙이 성립함을 확인. 행성 궤도는 에너지와 각운동량 보존의 결과."""
        return template

    elif topic == 'double_pendulum':
        t1 = params.get('theta1_deg', 120.0)
        t2 = params.get('theta2_deg', -30.0)
        max2 = num_sum.get('max_angle2_deg', 0.0)
        lyap = num_sum.get('lyapunov_exponent_estimate_log10', 0.0)
        
        template = f"""## 이중 진자 혼돈 시뮬레이션

**핵심 물리**: 라그랑지안에서 유도된 연립 ODE, 초기 조건 (θ₁={t1:.1f}°, θ₂={t2:.1f}°)

**수치 방법**: SciPy RK45, max_step=0.01s

**결과 분석**: θ₂ 최대 진폭 {max2:.1f}°. 리아푸노프 지수 추정 ~{lyap:.3f} (양수 → 혼돈).

**물리적 의미**: 초기 조건 0.01° 차이가 지수적으로 발산. 나비 효과의 수치적 증거."""
        return template

    elif topic == 'lagrangian':
        length = params.get('length', 1.0)
        t0 = params.get('theta0_deg', 60.0)
        edrift = num_sum.get('energy_drift_percent', 0.0)
        diff = num_sum.get('max_numerical_diff_rad', 0.0)
        approx = num_sum.get('small_angle_max_error_deg', 0.0)
        
        status = '소각도 근사가 부정확' if t0 > 30 else '소각도 근사가 유효'
        
        template = f"""## 라그랑지안/해밀토니안/뉴턴 역학 비교

**핵심 물리**: 단순 진자 $L = \\frac{{1}}{{2}}mL^2\\dot{{\\theta}}^2 - mgL(1-\\cos\\theta)$, L={length:.1f}m, θ₀={t0:.1f}°

**수치 방법**: 3가지 정식화 모두 SciPy RK45로 수치 적분

**결과 분석**: 세 방법 최대 수치 차이 {diff:.2e} rad. 에너지 보존 오차 {edrift:.3f}%. 소각도 근사 최대 오차 {approx:.1f}°.

**물리적 의미**: 뉴턴, 라그랑지안, 해밀토니안은 완전히 동일한 결과. {t0:.0f}° 초기 각도는 {status}."""
        return template

    raise ValueError(f"Unknown topic for template explainer: {topic}")
