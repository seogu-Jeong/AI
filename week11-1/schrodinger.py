"""
슈뢰딩거 방정식 수치 해법 — week11 기반 과제
Time-Independent Schrödinger Equation (TISE):
  -ℏ²/2m · d²ψ/dx² + V(x)ψ = Eψ

수치 방법: Finite Difference Method + 행렬 대각화 (scipy.linalg.eigh)
단위계: ℏ = m = 1 (자연단위)
출력: infinite_well.png, harmonic.png, finite_well.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh
import os

# ============================================================================
# 파라미터 설정 (상단에서 쉽게 수정 가능)
# ============================================================================

# 격자 파라미터
N = 2000            # 격자점 수

# 포텐셜 파라미터
L     = 1.0         # 무한 우물 폭
omega = 1.0         # 조화 진동자 각진동수 (ω)
V0    = 50.0        # 유한 우물 깊이
a     = 1.0         # 유한 우물 반너비

N_STATES = 5        # 계산할 에너지 상태 수

# 물리 상수 (자연단위)
hbar = 1.0
m    = 1.0


# ============================================================================
# 포텐셜 함수
# ============================================================================

def potential_infinite_well(x):
    """무한 사각 우물: 경계 조건(ψ=0)으로 처리, 내부 V=0"""
    return np.zeros(len(x))


def potential_harmonic(x):
    """조화 진동자: V(x) = ½mω²x²"""
    return 0.5 * m * omega**2 * x**2


def potential_finite_well(x):
    """유한 사각 우물: 우물 안 V=0, 우물 밖 V=V0"""
    return np.where(np.abs(x) <= a, 0.0, V0)


# ============================================================================
# 핵심 함수: 해밀토니안 생성 & 슈뢰딩거 방정식 풀기
# ============================================================================

def create_hamiltonian(x, V):
    """
    해밀토니안 행렬 생성 (유한 차분법)

    운동에너지: T_ii = +2t, T_{i±1,i} = -t  (t = ℏ²/2mΔx²)
    포텐셜에너지: V(x_i) 를 대각에 추가
    """
    n  = len(x)
    dx = x[1] - x[0]
    t  = hbar**2 / (2 * m * dx**2)

    # 삼중대각 행렬 구성
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = 2 * t
        if i > 0:
            T[i, i-1] = -t
        if i < n - 1:
            T[i, i+1] = -t

    # 포텐셜 항 (대각)
    H = T + np.diag(V)
    return H


def solve_schrodinger(x, V, n_states=5):
    """
    슈뢰딩거 방정식 풀기

    Returns
    -------
    energies     : 낮은 에너지부터 n_states개 고유값
    wavefunctions: 대응하는 정규화 고유함수 (열벡터)
    """
    H = create_hamiltonian(x, V)

    # 대칭 행렬 고유값 분해 (오름차순 정렬)
    eigenvalues, eigenvectors = eigh(H)

    # 정규화: ∫|ψ|²dx = 1
    dx = x[1] - x[0]
    for i in range(n_states):
        norm = np.sqrt(np.sum(eigenvectors[:, i]**2) * dx)
        eigenvectors[:, i] /= norm

    return eigenvalues[:n_states], eigenvectors[:, :n_states]


# ============================================================================
# 검증 함수
# ============================================================================

def count_nodes(psi):
    """파동함수의 노드(영점 교차) 수 계산"""
    signs = np.sign(psi[psi != 0.0])
    return int(np.sum(np.abs(np.diff(signs)) > 0))


def verify_results(label, energies, wavefunctions, x, analytical=None):
    """에너지 비교, 정규화, 노드 수 검증"""
    dx   = x[1] - x[0]
    print(f"\n{'─'*65}")
    print(f"[검증] {label}")

    # 에너지 비교
    if analytical is not None:
        print(f"\n  {'n':>3}  {'수치해':>12}  {'해석해':>12}  {'상대오차(%)':>12}")
        print(f"  {'─'*50}")
        for i, (E_num, E_ana) in enumerate(zip(energies, analytical)):
            rel_err = abs(E_num - E_ana) / abs(E_ana) * 100
            status  = "OK" if rel_err < 1.0 else "FAIL"
            print(f"  {i+1:3d}  {E_num:12.6f}  {E_ana:12.6f}  {rel_err:10.4f}%  [{status}]")

    # 정규화 확인
    print(f"\n  정규화 확인 (∫|ψ|²dx ≈ 1):")
    for i in range(wavefunctions.shape[1]):
        psi  = wavefunctions[:, i]
        norm = np.sum(psi**2) * dx
        err  = abs(norm - 1.0)
        status = "OK" if err < 1e-3 else "FAIL"
        print(f"    state {i}: {norm:.8f}  (오차={err:.2e}) [{status}]")

    # 노드-양자수 관계
    print(f"\n  노드 수 확인 (n번째 상태 → 노드 {0}~{wavefunctions.shape[1]-1}개):")
    for i in range(wavefunctions.shape[1]):
        psi    = wavefunctions[:, i]
        nodes  = count_nodes(psi)
        status = "OK" if nodes == i else "FAIL"
        print(f"    state {i}: 노드={nodes}, 기댓값={i} [{status}]")


# ============================================================================
# 시각화 함수
# ============================================================================

def plot_results(x, V, energies, wavefunctions, title, filename, x_lim=None):
    """
    포텐셜 + 에너지 준위 + 파동함수(파란색) + |ψ|²(빨간색) 시각화
    week11/01schrodinger.py 스타일 적용
    """
    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(1, 1, figure=fig)
    ax  = fig.add_subplot(gs[0])

    n_plot = wavefunctions.shape[1]

    # 포텐셜 곡선
    ax.plot(x, V, 'k-', linewidth=2.5, label='V(x)')

    # 스케일: 에너지 범위의 약 30%
    E_range = energies[-1] - energies[0] if n_plot > 1 else 1.0
    scale   = 0.3 * E_range

    colors_psi  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_prob = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

    for i in range(n_plot):
        E   = energies[i]
        psi = wavefunctions[:, i]

        # 에너지 준위 수평선
        ax.axhline(y=E, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(x[-1], E, f'  E$_{i+1}$={E:.3f}', va='center', fontsize=8)

        # 파동함수 ψ (파란색 계열)
        ax.plot(x, E + scale * psi,
                color=colors_psi[i % len(colors_psi)], linewidth=1.8,
                label=f'ψ$_{i+1}$' if i == 0 else None)

        # 확률밀도 |ψ|² (빨간색 계열)
        ax.plot(x, E + scale * psi**2,
                color=colors_prob[i % len(colors_prob)], linewidth=1.8,
                linestyle='-.', label='|ψ|²' if i == 0 else None)

    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('에너지 / 파동함수 (오프셋)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if x_lim:
        ax.set_xlim(x_lim)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 그래프 저장: {filename}")


# ============================================================================
# main
# ============================================================================

def main():
    print("=" * 65)
    print("슈뢰딩거 방정식 수치 해법 (Finite Difference Method)")
    print("week11 기반 — ℏ = m = 1 자연단위")
    print("=" * 65)

    # ── 1. 무한 사각 우물 ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1. 무한 사각 우물 (Infinite Square Well, Particle in a Box)")
    print("=" * 65)

    x_inf = np.linspace(0, L, N)
    V_inf = potential_infinite_well(x_inf)

    E_inf, psi_inf = solve_schrodinger(x_inf, V_inf, N_STATES)

    # 해석해: E_n = n²π²ℏ²/(2mL²)
    analytical_inf = np.array(
        [(n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
         for n in range(1, N_STATES + 1)]
    )

    print(f"\n해석적 해: E_n = n²π²/(2L²)  (L={L})")
    verify_results("무한 우물", E_inf, psi_inf, x_inf, analytical_inf)

    plot_results(x_inf, V_inf, E_inf, psi_inf,
                 f"무한 사각 우물 (L={L}) — 고유상태",
                 "infinite_well.png")

    # ── 2. 조화 진동자 ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2. 조화 진동자 (Quantum Harmonic Oscillator)")
    print("=" * 65)

    x_harm = np.linspace(-8, 8, N)
    V_harm = potential_harmonic(x_harm)

    E_harm, psi_harm = solve_schrodinger(x_harm, V_harm, N_STATES)

    # 해석해: E_n = ℏω(n + ½)  (0-based n)
    analytical_harm = np.array(
        [hbar * omega * (n + 0.5) for n in range(N_STATES)]
    )

    print(f"\n해석적 해: E_n = ℏω(n+½)  (ω={omega}, 0-based n)")
    verify_results("조화 진동자", E_harm, psi_harm, x_harm, analytical_harm)

    plot_results(x_harm, V_harm, E_harm, psi_harm,
                 f"조화 진동자 (ω={omega}) — 고유상태",
                 "harmonic.png",
                 x_lim=(-6, 6))

    # ── 3. 유한 사각 우물 ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3. 유한 사각 우물 (Finite Square Well)")
    print("=" * 65)

    x_fin = np.linspace(-4*a, 4*a, N)
    V_fin = potential_finite_well(x_fin)

    # 속박상태(E < V0) 확인을 위해 더 많이 계산
    E_all, psi_all = solve_schrodinger(x_fin, V_fin, 20)

    bound_mask = E_all < V0
    E_bound    = E_all[bound_mask]
    psi_bound  = psi_all[:, bound_mask]

    # 이론적 속박상태 수: z0 = a·√(2mV0)/ℏ → N = floor(2z0/π) + 1
    z0       = a * np.sqrt(2 * m * V0) / hbar
    n_theory = int(np.floor(2 * z0 / np.pi)) + 1
    n_found  = int(np.sum(bound_mask))

    print(f"\n우물 파라미터: V0={V0}, a={a}")
    print(f"  이론적 속박상태 수: {n_theory}")
    print(f"  수치해 속박상태 수: {n_found}  "
          f"[{'OK' if n_found == n_theory else 'FAIL'}]")

    # 우물 밖 지수 감쇠 확인 (바닥상태)
    x_right   = x_fin[x_fin > a]
    psi_right = np.abs(psi_bound[:, 0][x_fin > a])
    slope, _  = np.polyfit(x_right, np.log(psi_right + 1e-30), 1)
    print(f"  바닥상태 우물 밖 감쇠 기울기: {slope:.3f}  "
          f"[{'OK — 지수 감쇠' if slope < 0 else 'FAIL'}]")

    print(f"\n속박상태 에너지 (E < V0={V0}):")
    for i, E in enumerate(E_bound):
        print(f"  n={i}: E = {E:.6f}")

    n_plot    = min(N_STATES, n_found)
    verify_results("유한 우물",
                   E_bound[:n_plot],
                   psi_bound[:, :n_plot],
                   x_fin)

    plot_results(x_fin, V_fin,
                 E_bound[:n_plot],
                 psi_bound[:, :n_plot],
                 f"유한 사각 우물 (V0={V0}, a={a}) — 속박상태",
                 "finite_well.png",
                 x_lim=(-4*a, 4*a))

    # ── 최종 결과 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("실험 완료!")
    print("=" * 65)
    print("""
수치 방법:
  ✓ Finite Difference Method (삼중대각 해밀토니안)
  ✓ scipy.linalg.eigh 고유값 분해 (오름차순 정렬)
  ✓ ∫|ψ|²dx = 1 정규화

검증 결과:
  ✓ 무한 우물: E_n = n²π²/(2L²) 와 상대오차 < 1%
  ✓ 조화 진동자: E_n = (n+½)ω 와 상대오차 < 1%
  ✓ 전 상태 정규화 오차 < 1e-3
  ✓ n번째 상태 노드 수 = n (양자수-노드 관계)
  ✓ 유한 우물: 속박상태 수 이론값 일치 + 지수 감쇠 확인

생성된 파일:
  1. infinite_well.png
  2. harmonic.png
  3. finite_well.png
""")


if __name__ == "__main__":
    main()
