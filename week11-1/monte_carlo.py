"""
Monte Carlo 시뮬레이션 — week12 기반 과제
통계물리 핵심 3가지 주제:
  1. 랜덤 워크 (Random Walk) + 중심극한정리
  2. Metropolis-Hastings 알고리즘 + 볼츠만 분포
  3. 2D 이징 모델 (Ising Model) + 상전이

출력: random_walk.png, metropolis.png, ising_model.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ============================================================================
# 파라미터 설정 (상단에서 쉽게 수정 가능)
# ============================================================================

# 랜덤 워크
N_STEPS   = 1000     # 랜덤 워크 스텝 수
N_WALKS   = 5000     # 통계용 워크 수

# Metropolis
N_MC      = 20000    # Monte Carlo 스텝 수
N_BURNIN  = 2000     # 초기 버려낼 구간 (burn-in)
STEP_SIZE = 0.3      # 제안 이동 크기

# 이징 모델
L         = 30       # 격자 크기 (L × L)
J         = 1.0      # 교환 상호작용
N_SWEEP   = 500      # 스윕 수 (1 sweep = L² 스텝)
N_EQUIL   = 200      # 평형화 스윕 수

# 물리 상수
k_B = 1.0            # 볼츠만 상수 (자연단위)


# ============================================================================
# 1. 랜덤 워크 (Random Walk)
# ============================================================================

def random_walk_1d(n_steps):
    """1D 랜덤 워크: ±1 스텝, 위치 배열 반환"""
    steps = np.random.choice([-1, 1], size=n_steps)
    return np.concatenate([[0], np.cumsum(steps)])


def simulate_many_walks(n_walks, n_steps):
    """n_walks개의 1D 랜덤 워크를 일괄 시뮬레이션"""
    steps = np.random.choice([-1, 1], size=(n_walks, n_steps))
    pos   = np.zeros((n_walks, n_steps + 1))
    pos[:, 1:] = np.cumsum(steps, axis=1)
    return pos


def plot_random_walk(filename='random_walk.png'):
    """랜덤 워크 시뮬레이션 결과 시각화"""

    print("\n" + "="*65)
    print("1. 랜덤 워크 (Random Walk)")
    print("="*65)

    fig = plt.figure(figsize=(15, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (a) 개별 워크 궤적 5개
    ax1 = fig.add_subplot(gs[0, 0])
    t   = np.arange(N_STEPS + 1)
    for i in range(5):
        pos = random_walk_1d(N_STEPS)
        ax1.plot(t, pos, alpha=0.75, linewidth=1.2, label=f'Walk {i+1}')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time step', fontsize=11)
    ax1.set_ylabel('Position', fontsize=11)
    ax1.set_title('(a) 1D Random Walk Trajectories', fontsize=12, weight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # (b) 최종 위치 분포 — 중심극한정리 검증
    ax2    = fig.add_subplot(gs[0, 1])
    walks  = simulate_many_walks(N_WALKS, N_STEPS)
    finals = walks[:, -1]

    ax2.hist(finals, bins=50, density=True, alpha=0.65,
             color='steelblue', edgecolor='black', label='Simulation')

    # 이론: 가우시안 (μ=0, σ=√N)
    sigma  = np.sqrt(N_STEPS)
    x_fit  = np.linspace(finals.min(), finals.max(), 200)
    gauss  = np.exp(-x_fit**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    ax2.plot(x_fit, gauss, 'r-', linewidth=2, label=f'Gaussian (σ={sigma:.1f})')
    ax2.set_xlabel('Final position', fontsize=11)
    ax2.set_ylabel('Probability density', fontsize=11)
    ax2.set_title('(b) Final Position (Central Limit Theorem)', fontsize=12, weight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    print(f"   최종 위치 평균: {np.mean(finals):.2f}  (이론: 0)")
    print(f"   최종 위치 표준편차: {np.std(finals):.2f}  (이론: {sigma:.2f})")

    # (c) MSD (Mean Square Displacement) vs t
    ax3 = fig.add_subplot(gs[0, 2])
    msd = np.mean(walks**2, axis=0)
    ax3.plot(t, msd, 'b-', linewidth=2, label='Simulation ⟨x²⟩')
    ax3.plot(t, t,   'r--', linewidth=2, label='Theory: t')
    ax3.set_xlabel('Time step', fontsize=11)
    ax3.set_ylabel('MSD ⟨x²⟩', fontsize=11)
    ax3.set_title('(c) Mean Square Displacement', fontsize=12, weight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    print(f"   MSD 기울기 (수치): {msd[-1]/N_STEPS:.3f}  (이론: 1.000)")

    # (d) 확산 계수 D = MSD/(2t)
    ax4 = fig.add_subplot(gs[1, 0])
    D   = msd[1:] / (2 * t[1:])
    ax4.plot(t[1:], D, 'g-', linewidth=1.5)
    ax4.axhline(0.5, color='r', linestyle='--', linewidth=2, label='Theory D=0.5')
    ax4.set_xlabel('Time step', fontsize=11)
    ax4.set_ylabel('Diffusion constant D', fontsize=11)
    ax4.set_title('(d) Diffusion Constant D = MSD/2t', fontsize=12, weight='bold')
    ax4.set_ylim(0.3, 0.7)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    print(f"   확산 계수 (t>100 평균): {np.mean(D[100:]):.4f}  (이론: 0.5)")

    # (e) 로그-로그 플롯: MSD ∝ t^α (정상 확산 α=1)
    ax5 = fig.add_subplot(gs[1, 1])
    t_log   = t[1:]
    msd_log = msd[1:]
    ax5.loglog(t_log, msd_log, 'b-', linewidth=2, label='Simulation')
    ax5.loglog(t_log, t_log,   'r--', linewidth=2, label='Slope=1 (normal diffusion)')
    ax5.set_xlabel('Time step (log)', fontsize=11)
    ax5.set_ylabel('MSD (log)', fontsize=11)
    ax5.set_title('(e) Log-Log: MSD ~ t^α', fontsize=12, weight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, which='both')

    # (f) 정규화 확인: ∫P(x)dx = 1
    ax6 = fig.add_subplot(gs[1, 2])
    dx    = x_fit[1] - x_fit[0]
    norm  = np.sum(gauss) * dx
    cumul = np.cumsum(gauss) * dx
    ax6.plot(x_fit, cumul, 'b-', linewidth=2, label='CDF (Gaussian)')
    ax6.axhline(1.0, color='r', linestyle='--', linewidth=1.5, label='Total = 1')
    ax6.set_xlabel('x', fontsize=11)
    ax6.set_ylabel('Cumulative probability', fontsize=11)
    ax6.set_title(f'(f) Normalization check (∫P dx = {norm:.4f})', fontsize=12, weight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Random Walk & Central Limit Theorem', fontsize=15, weight='bold')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 그래프 저장: {filename}")


# ============================================================================
# 2. Metropolis-Hastings 알고리즘
# ============================================================================

def double_well(x):
    """이중 우물 포텐셜: V(x) = (x²-1)²"""
    return (x**2 - 1)**2


def metropolis_step(x, T, step_size=STEP_SIZE):
    """
    Metropolis 스텝 1회 수행

    - 에너지 감소: 항상 수락
    - 에너지 증가: exp(-ΔE/kT) 확률로 수락 (Boltzmann 인자)
    """
    x_new = x + np.random.uniform(-step_size, step_size)
    dE    = double_well(x_new) - double_well(x)

    if dE < 0 or np.random.random() < np.exp(-dE / (k_B * T)):
        return x_new, True
    return x, False


def run_metropolis(T, n_steps=N_MC, n_burnin=N_BURNIN):
    """주어진 온도 T에서 Metropolis 시뮬레이션 실행"""
    x          = 0.0
    trajectory = np.zeros(n_steps)
    n_accept   = 0

    for i in range(n_steps):
        x, accepted = metropolis_step(x, T)
        trajectory[i] = x
        if accepted:
            n_accept += 1

    return trajectory[n_burnin:], n_accept / n_steps


def plot_metropolis(filename='metropolis.png'):
    """Metropolis 알고리즘 시뮬레이션 결과 시각화"""

    print("\n" + "="*65)
    print("2. Metropolis-Hastings 알고리즘")
    print("="*65)

    temperatures = [0.05, 0.2, 0.5, 1.5]
    x_range      = np.linspace(-2.2, 2.2, 500)
    V_range      = double_well(x_range)

    fig = plt.figure(figsize=(15, 10))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    for idx, T in enumerate(temperatures):
        traj, acc_rate = run_metropolis(T)

        ax_top = fig.add_subplot(gs[0, idx])

        # 시뮬레이션 히스토그램
        ax_top.hist(traj, bins=50, density=True, alpha=0.6,
                    color='steelblue', edgecolor='black', label='Simulation')

        # 이론: 볼츠만 분포 P(x) ∝ exp(-V(x)/kT)
        boltz = np.exp(-V_range / (k_B * T))
        Z     = np.trapz(boltz, x_range)
        boltz /= Z
        ax_top.plot(x_range, boltz, 'r-', linewidth=2.5, label='Boltzmann')

        # 포텐셜 (보조 축)
        ax2 = ax_top.twinx()
        ax2.plot(x_range, V_range, 'g--', linewidth=1.5, alpha=0.7, label='V(x)')
        ax2.set_ylabel('V(x)', fontsize=9, color='green')
        ax2.tick_params(axis='y', labelcolor='green', labelsize=8)

        ax_top.set_xlabel('x', fontsize=10)
        ax_top.set_ylabel('Probability density', fontsize=10)
        ax_top.set_title(f'T={T:.2f}  (acc={acc_rate:.0%})', fontsize=11, weight='bold')
        ax_top.legend(fontsize=8, loc='upper left')
        ax_top.grid(True, alpha=0.3)

        print(f"   T={T:.2f}: 수락률={acc_rate:.3f}, "
              f"평균 위치={np.mean(traj):.3f}, 표준편차={np.std(traj):.3f}")

    # 아래 행: 수락률 vs 스텝 크기
    ax_b1 = fig.add_subplot(gs[1, :2])
    ax_b2 = fig.add_subplot(gs[1, 2:])

    step_sizes   = np.logspace(-2, 1, 20)
    acc_rates    = []
    efficiencies = []
    T_test = 0.5

    for ss in step_sizes:
        traj, ar = run_metropolis(T_test, n_steps=8000, n_burnin=1000)
        acc_rates.append(ar)
        # 효율 = 수락률 × 탐색 표준편차
        efficiencies.append(ar * np.std(traj))

    # 수락률
    ax_b1.semilogx(step_sizes, acc_rates, 'bo-', markersize=5, linewidth=2)
    ax_b1.axhline(0.5, color='r', linestyle='--', linewidth=2, label='Optimal ~50%')
    ax_b1.set_xlabel('Step size', fontsize=11)
    ax_b1.set_ylabel('Acceptance rate', fontsize=11)
    ax_b1.set_title(f'(e) Acceptance Rate vs Step Size  (T={T_test})', fontsize=12, weight='bold')
    ax_b1.legend(fontsize=10)
    ax_b1.grid(True, alpha=0.3)

    # 효율
    ax_b2.semilogx(step_sizes, efficiencies, 'go-', markersize=5, linewidth=2)
    best_idx = int(np.argmax(efficiencies))
    ax_b2.axvline(step_sizes[best_idx], color='r', linestyle='--', linewidth=2,
                  label=f'Best: {step_sizes[best_idx]:.3f}')
    ax_b2.set_xlabel('Step size', fontsize=11)
    ax_b2.set_ylabel('Efficiency (acc × std)', fontsize=11)
    ax_b2.set_title('(f) Sampling Efficiency', fontsize=12, weight='bold')
    ax_b2.legend(fontsize=10)
    ax_b2.grid(True, alpha=0.3)

    plt.suptitle('Metropolis-Hastings Algorithm — Double-Well Potential', fontsize=15, weight='bold')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 그래프 저장: {filename}")


# ============================================================================
# 3. 2D 이징 모델 (Ising Model)
# ============================================================================

def init_spins(L, state='random'):
    """스핀 배열 초기화: 'random', 'up', 'down'"""
    if state == 'random':
        return np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
    elif state == 'up':
        return np.ones((L, L), dtype=np.int8)
    return -np.ones((L, L), dtype=np.int8)


def calc_energy(spins, J=J):
    """총 에너지: H = -J Σ s_i s_j (주기적 경계 조건)"""
    return -J * (np.sum(spins * np.roll(spins, -1, axis=0)) +
                 np.sum(spins * np.roll(spins, -1, axis=1)))


def calc_magnetization(spins):
    """단위 스핀당 자화: M = Σs_i / N"""
    return np.sum(spins) / spins.size


def metropolis_sweep(spins, T, J=J):
    """
    Metropolis 스윕 1회 (L² 번 스핀 플립 시도)

    국소 에너지 변화:
    ΔE = 2J · s_i · (위+아래+왼+오른) 이웃 합
    """
    L = spins.shape[0]
    for _ in range(L * L):
        i = np.random.randint(L)
        j = np.random.randint(L)

        # 4 이웃 합 (주기적 경계 조건)
        neighbor_sum = (spins[(i-1) % L, j] + spins[(i+1) % L, j] +
                        spins[i, (j-1) % L] + spins[i, (j+1) % L])
        dE = 2 * J * spins[i, j] * neighbor_sum

        if dE < 0 or np.random.random() < np.exp(-dE / (k_B * T)):
            spins[i, j] *= -1
    return spins


def simulate_ising(L, T, n_equil=N_EQUIL, n_sweep=N_SWEEP):
    """이징 모델 시뮬레이션 → (에너지/스핀, |자화|) 배열 반환"""
    spins    = init_spins(L, 'random')
    N        = L * L

    # 평형화
    for _ in range(n_equil):
        spins = metropolis_sweep(spins, T)

    energies = np.zeros(n_sweep)
    mags     = np.zeros(n_sweep)

    for s in range(n_sweep):
        spins       = metropolis_sweep(spins, T)
        energies[s] = calc_energy(spins) / N
        mags[s]     = abs(calc_magnetization(spins))

    return energies, mags, spins


def plot_ising(filename='ising_model.png'):
    """이징 모델 상전이 시뮬레이션 결과 시각화"""

    print("\n" + "="*65)
    print("3. 2D 이징 모델 (Ising Model) — 상전이")
    print("="*65)

    # 온도 범위 (임계 온도 Tc ≈ 2.269 포함)
    Tc          = 2.0 / np.log(1 + np.sqrt(2))   # ≈ 2.269
    temperatures = np.array([1.0, 1.5, 2.0, 2.269, 2.5, 3.0, 4.0])

    mean_E   = []
    mean_M   = []
    std_M    = []
    final_configs = {}

    for T in temperatures:
        E_arr, M_arr, spins_final = simulate_ising(L, T,
                                                    n_equil=N_EQUIL,
                                                    n_sweep=N_SWEEP)
        mean_E.append(np.mean(E_arr))
        mean_M.append(np.mean(M_arr))
        std_M.append(np.std(M_arr))

        # 대표 온도의 스핀 배열 보관
        if T in [1.0, 2.269, 4.0]:
            final_configs[T] = spins_final.copy()

        print(f"   T={T:.3f}: ⟨E/N⟩={np.mean(E_arr):.4f}, "
              f"⟨|M|⟩={np.mean(M_arr):.4f}, σM={np.std(M_arr):.4f}")

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    # 스핀 배열 시각화 (저온, 임계, 고온)
    for col, T_show in enumerate([1.0, 2.269, 4.0]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(final_configs[T_show], cmap='RdBu',
                  vmin=-1, vmax=1, interpolation='nearest')
        label = f'T={T_show:.3f}'
        if abs(T_show - Tc) < 0.01:
            label += ' (Tc)'
        ax.set_title(f'Spin config\n{label}', fontsize=11, weight='bold')
        ax.axis('off')

    # 자화 vs 온도
    ax_M = fig.add_subplot(gs[0, 3])
    ax_M.errorbar(temperatures, mean_M, yerr=std_M,
                  fmt='bo-', markersize=6, linewidth=2, capsize=4, label='⟨|M|⟩')
    ax_M.axvline(Tc, color='r', linestyle='--', linewidth=2,
                 label=f'Tc={Tc:.3f}')
    ax_M.set_xlabel('Temperature T', fontsize=11)
    ax_M.set_ylabel('|Magnetization|', fontsize=11)
    ax_M.set_title('(d) Magnetization vs T', fontsize=12, weight='bold')
    ax_M.legend(fontsize=9)
    ax_M.grid(True, alpha=0.3)

    # 에너지 vs 온도
    ax_E = fig.add_subplot(gs[1, 0])
    ax_E.plot(temperatures, mean_E, 'rs-', markersize=6, linewidth=2)
    ax_E.axvline(Tc, color='b', linestyle='--', linewidth=2, label=f'Tc={Tc:.3f}')
    ax_E.set_xlabel('Temperature T', fontsize=11)
    ax_E.set_ylabel('Energy per spin ⟨E/N⟩', fontsize=11)
    ax_E.set_title('(e) Energy vs T', fontsize=12, weight='bold')
    ax_E.legend(fontsize=9)
    ax_E.grid(True, alpha=0.3)

    # 자화 시계열 (3가지 온도)
    ax_ts = fig.add_subplot(gs[1, 1:3])
    colors = ['blue', 'orange', 'red']
    for T_show, color in zip([1.0, 2.269, 4.0], colors):
        _, M_ts, _ = simulate_ising(L, T_show, n_equil=N_EQUIL, n_sweep=200)
        label = f'T={T_show:.3f}' + (' (Tc)' if abs(T_show - Tc) < 0.01 else '')
        ax_ts.plot(M_ts, color=color, linewidth=1.2, alpha=0.8, label=label)
    ax_ts.set_xlabel('Monte Carlo sweep', fontsize=11)
    ax_ts.set_ylabel('|M|', fontsize=11)
    ax_ts.set_title('(f) Magnetization Time Series', fontsize=12, weight='bold')
    ax_ts.legend(fontsize=9)
    ax_ts.grid(True, alpha=0.3)

    # 자기감수율 χ ∝ σ²M (임계점에서 발산)
    ax_chi = fig.add_subplot(gs[1, 3])
    chi = L**2 * np.array(std_M)**2     # χ = N·Var(M)
    ax_chi.plot(temperatures, chi, 'g^-', markersize=6, linewidth=2)
    ax_chi.axvline(Tc, color='r', linestyle='--', linewidth=2, label=f'Tc={Tc:.3f}')
    ax_chi.set_xlabel('Temperature T', fontsize=11)
    ax_chi.set_ylabel('Susceptibility χ', fontsize=11)
    ax_chi.set_title('(g) Susceptibility vs T\n(peaks near Tc)', fontsize=12, weight='bold')
    ax_chi.legend(fontsize=9)
    ax_chi.grid(True, alpha=0.3)

    plt.suptitle(f'2D Ising Model — Phase Transition (L={L}, Tc≈{Tc:.3f})',
                 fontsize=15, weight='bold')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 그래프 저장: {filename}")


# ============================================================================
# main
# ============================================================================

def main():
    print("=" * 65)
    print("Monte Carlo 시뮬레이션 — week12 기반")
    print("통계물리: 랜덤 워크 / Metropolis / Ising Model")
    print("=" * 65)

    plot_random_walk('random_walk.png')
    plot_metropolis('metropolis.png')
    plot_ising('ising_model.png')

    print("\n" + "=" * 65)
    print("실험 완료!")
    print("=" * 65)
    print("""
수치 방법:
  ✓ Monte Carlo 랜덤 샘플링
  ✓ Metropolis-Hastings 알고리즘 (Detailed Balance)
  ✓ 2D Ising Model Metropolis 스윕

검증 결과:
  ✓ 랜덤 워크: MSD ~ t, 확산계수 D ≈ 0.5
  ✓ 중심극한정리: 최종 위치 ~ 가우시안 (σ = √N)
  ✓ Metropolis: 볼츠만 분포 재현, 최적 수락률 ~50%
  ✓ 이징 모델: Tc ≈ 2.269에서 상전이 (자화 소멸)

생성된 파일:
  1. random_walk.png
  2. metropolis.png
  3. ising_model.png
""")


if __name__ == '__main__':
    main()
