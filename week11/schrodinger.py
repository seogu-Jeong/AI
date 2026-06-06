"""
Solve 1-D time-independent Schrödinger equation via finite difference method.

  -ℏ²/(2m) ψ'' + V(x)ψ = Eψ,  ℏ = m = 1  (natural units)

Potentials:  infinite square well  ·  harmonic oscillator  ·  finite square well
Output:      infinite_well.png  harmonic.png  finite_well.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Parameters (edit freely) ──────────────────────────────────────────────
# Infinite square well
N_IW  = 1000    # interior grid points  (ψ = 0 at both ends)
L     = 1.0     # well width  (ℏ = m = 1)

# Harmonic oscillator
N_HO  = 2000    # interior grid points
X_HO  = 8.0    # domain half-width  [-X_HO, X_HO]
OMEGA = 1.0    # angular frequency  ω

# Finite square well  (symmetric, centred at x = 0)
N_FW  = 2000    # interior grid points
X_FW  = 6.0    # domain half-width
V0    = 50.0   # barrier height  (V = 0 inside |x| < A,  V = V0 outside)
A     = 0.5    # half-width of the well

N_STATES = 5   # eigenstates to compute / display
# ───────────────────────────────────────────────────────────────────────────


def build_hamiltonian(x, V):
    """Symmetric tridiagonal FDM Hamiltonian  (ℏ = m = 1)."""
    N, dx = len(x), x[1] - x[0]
    t     = 1.0 / (2.0 * dx ** 2)       # ℏ²/(2mΔx²)
    H     = np.diag(2 * t + V)
    off   = np.full(N - 1, -t)
    return H + np.diag(off, 1) + np.diag(off, -1)


def solve(x, V, n_states=5):
    """Diagonalise H; return n_states lowest eigenvalues and normalised ψ."""
    vals, vecs = np.linalg.eigh(build_hamiltonian(x, V))
    dx = x[1] - x[0]
    for i in range(n_states):
        vecs[:, i] /= np.sqrt(np.sum(vecs[:, i] ** 2) * dx)
    return vals[:n_states], vecs[:, :n_states]


def potential_infinite_well(x):
    return np.zeros_like(x)


def potential_harmonic(x, omega=OMEGA):
    return 0.5 * omega ** 2 * x ** 2


def potential_finite_well(x, V0=V0, a=A):
    V = np.full_like(x, V0)
    V[np.abs(x) < a] = 0.0
    return V


def count_nodes(psi, tol=1e-10):
    """Count interior zero-crossings (nodes) of psi."""
    s = np.sign(psi[np.abs(psi) > tol])
    return int(np.sum(np.diff(s) != 0))


def plot_results(x, V, energies, psi_mat, title, filename):
    """
    Potential curve (black) + horizontal energy levels (dashed grey) +
    ψ(x) offset to E_n in blue  +  |ψ(x)|² offset to E_n in red.
    """
    n = len(energies)
    min_gap = np.min(np.diff(energies)) if n > 1 else max(abs(energies[0]), 1.0)
    scale   = 0.35 * min_gap           # amplitude ≈ 35 % of smallest gap

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, V, 'k-', lw=2.5, label='V(x)', zorder=5)

    for i in range(n):
        E, psi = energies[i], psi_mat[:, i]
        ax.axhline(E, color='gray', lw=0.8, ls='--', alpha=0.7)
        ax.plot(x, E + psi * scale,        'b-', lw=1.5,
                label='ψ(x)'    if i == 0 else None)
        ax.plot(x, E + psi ** 2 * scale,   'r-', lw=1.5,
                label='|ψ(x)|²' if i == 0 else None)

    y_lo = min(np.min(V), energies[0]) - 0.08 * (energies[-1] - np.min(V) + 1)
    y_hi = energies[-1] + 2.0 * scale
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    SEP = '=' * 68
    HDR = f"  {'n':>3}  {'E_num':>12}  {'E_ana':>12}  {'rel_err%':>10}  {'nodes':>6}"

    # ── 1. Infinite square well ────────────────────────────────────────────
    print(SEP)
    print("1. Infinite Square Well  (L=1, ℏ=m=1)")
    print(SEP)
    x_iw = np.linspace(0, L, N_IW + 2)[1:-1]     # interior only → ψ=0 at ends
    V_iw = potential_infinite_well(x_iw)
    E_iw, psi_iw = solve(x_iw, V_iw, N_STATES)

    print(HDR)
    for n in range(N_STATES):
        E_a = (n + 1) ** 2 * np.pi ** 2 / (2 * L ** 2)
        rel = abs(E_iw[n] - E_a) / E_a * 100
        nd  = count_nodes(psi_iw[:, n])
        print(f"  {n:>3}  {E_iw[n]:>12.6f}  {E_a:>12.6f}  {rel:>10.6f}  {nd:>6}")
    plot_results(x_iw, V_iw, E_iw, psi_iw,
                 "Infinite Square Well (L=1)", "infinite_well.png")

    # ── 2. Harmonic oscillator ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("2. Harmonic Oscillator  (ω=1, ℏ=m=1)")
    print(SEP)
    x_ho = np.linspace(-X_HO, X_HO, N_HO + 2)[1:-1]
    V_ho = potential_harmonic(x_ho)
    E_ho, psi_ho = solve(x_ho, V_ho, N_STATES)

    print(HDR)
    for n in range(N_STATES):
        E_a = OMEGA * (n + 0.5)
        rel = abs(E_ho[n] - E_a) / E_a * 100
        nd  = count_nodes(psi_ho[:, n])
        print(f"  {n:>3}  {E_ho[n]:>12.6f}  {E_a:>12.6f}  {rel:>10.6f}  {nd:>6}")
    plot_results(x_ho, V_ho, E_ho, psi_ho,
                 "Harmonic Oscillator (ω=1)", "harmonic.png")

    # ── 3. Finite square well ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"3. Finite Square Well  (V0={V0}, a={A})")
    print(SEP)
    x_fw = np.linspace(-X_FW, X_FW, N_FW + 2)[1:-1]
    V_fw = potential_finite_well(x_fw)

    E_all, psi_all = solve(x_fw, V_fw, min(N_FW, 60))

    n_bound  = int(np.sum((E_all >= 0) & (E_all < V0)))
    z0       = A * np.sqrt(2 * V0)
    n_theory = int(np.floor(2 * z0 / np.pi)) + 1

    print(f"  Numerical bound states (0 ≤ E < V0={V0}): {n_bound}")
    print(f"  Theoretical  (z0 = {z0:.3f}):             {n_theory}")
    print(f"  Match: {'YES ✓' if n_bound == n_theory else 'NO ✗'}")

    n_plot = min(N_STATES, n_bound)
    plot_results(x_fw, V_fw, E_all[:n_plot], psi_all[:, :n_plot],
                 f"Finite Square Well (V0={V0}, a={A})", "finite_well.png")

    # ── Verification ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("Verification  (∫|ψ|²dx  and  node count)")
    print(SEP)
    all_ok = True
    for label, x_a, p_a in [
        ("Infinite well", x_iw, psi_iw),
        ("Harmonic     ", x_ho, psi_ho),
        ("Finite well  ", x_fw, psi_all[:, :n_plot]),
    ]:
        dx    = x_a[1] - x_a[0]
        norms = np.array([np.sum(p_a[:, i] ** 2) * dx for i in range(p_a.shape[1])])
        nodes = [count_nodes(p_a[:, i]) for i in range(p_a.shape[1])]
        n_err  = np.max(np.abs(norms - 1.0))
        ok    = n_err < 1e-3
        all_ok &= ok
        print(f"  {label}: norm_err={n_err:.2e}  nodes={nodes}  {'PASS ✓' if ok else 'FAIL ✗'}")

    print(f"\n  {'ALL PASS ✓' if all_ok else 'SOME CRITERIA FAILED ✗'}")
    print("\nOutput files: infinite_well.png  harmonic.png  finite_well.png")


if __name__ == "__main__":
    main()
