"""
1D Time-Independent Schrödinger Equation solver via Finite Difference Method
Units: ℏ = m = 1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Grid parameters ──────────────────────────────────────────────────────────
N = 2000          # number of interior grid points

# ── Potential parameters ──────────────────────────────────────────────────────
L = 1.0           # infinite well width
omega = 1.0       # harmonic oscillator frequency
V0 = 50.0         # finite well depth
a = 1.0           # finite well half-width

N_STATES = 5      # number of states to compute


# ── Potential functions ───────────────────────────────────────────────────────

def potential_infinite_well(x):
    V = np.zeros_like(x)
    # walls handled by boundary conditions (ψ=0 at ends)
    return V


def potential_harmonic(x):
    return 0.5 * omega**2 * x**2


def potential_finite_well(x):
    V = np.where(np.abs(x) <= a, 0.0, V0)
    return V


# ── Core numerical routines ───────────────────────────────────────────────────

def build_hamiltonian(x, V):
    """Tridiagonal Hamiltonian via second-order finite difference."""
    n = len(x)
    dx = x[1] - x[0]
    t = 1.0 / (2.0 * dx**2)          # ℏ²/(2m Δx²) with ℏ=m=1

    diag = 2.0 * t + V
    off = -t * np.ones(n - 1)
    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return H


def solve(x, V, n_states):
    """Return (energies, wavefunctions) for lowest n_states eigenstates."""
    H = build_hamiltonian(x, V)
    eigenvalues, eigenvectors = np.linalg.eigh(H)  # sorted ascending

    dx = x[1] - x[0]
    psis = []
    for i in range(n_states):
        psi = eigenvectors[:, i]
        norm = np.sqrt(np.trapz(psi**2, x))
        psi /= norm
        # sign convention: make first lobe positive
        idx = np.argmax(np.abs(psi))
        if psi[idx] < 0:
            psi = -psi
        psis.append(psi)

    return eigenvalues[:n_states], np.array(psis)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(x, V, energies, psis, title, filename, x_lim=None):
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    # potential
    ax.plot(x, V, 'k-', linewidth=2, label='V(x)')

    scale = 0.4 * (energies[-1] - energies[0]) if len(energies) > 1 else 1.0

    colors_psi = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_prob = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

    for i, (E, psi) in enumerate(zip(energies, psis)):
        prob = psi**2

        # energy level
        ax.axhline(E, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(x[-1], E, f' $E_{i+1}$={E:.3f}', va='center', fontsize=8)

        # ψ offset to energy level (blue)
        ax.plot(x, E + scale * psi, color=colors_psi[i], linewidth=1.5,
                label=f'$\\psi_{i+1}$' if i == 0 else None)

        # |ψ|² offset (red)
        ax.plot(x, E + scale * prob, color=colors_prob[i] if i < len(colors_prob) else 'r',
                linewidth=1.5, linestyle='-.',
                label='$|\\psi|^2$' if i == 0 else None)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Energy / wavefunction (offset)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    if x_lim:
        ax.set_xlim(x_lim)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ── Verification helpers ──────────────────────────────────────────────────────

def count_nodes(psi):
    """Count zero-crossings (sign changes) in psi."""
    signs = np.sign(psi[psi != 0.0])
    return int(np.sum(np.abs(np.diff(signs)) > 0))


def check_normalization(psis, x, tol=1e-3):
    ok = True
    for i, psi in enumerate(psis):
        norm2 = np.trapz(psi**2, x)
        err = abs(norm2 - 1.0)
        status = "OK" if err < tol else "FAIL"
        print(f"  state {i}: ∫|ψ|²dx = {norm2:.6f}  (err={err:.2e}) [{status}]")
        if err >= tol:
            ok = False
    return ok


def print_comparison(label, energies, analytical, n_states):
    print(f"\n{'─'*60}")
    print(f"{label}")
    print(f"{'n':>4} {'Numerical':>12} {'Analytical':>12} {'Rel.Err%':>10}")
    print(f"{'─'*60}")
    all_ok = True
    for i in range(n_states):
        rel_err = abs(energies[i] - analytical[i]) / abs(analytical[i]) * 100
        status = "OK" if rel_err < 1.0 else "FAIL"
        print(f"  {i+1:2d} {energies[i]:12.6f} {analytical[i]:12.6f} {rel_err:10.4f}%  [{status}]")
        if rel_err >= 1.0:
            all_ok = False
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results_ok = True

    # ── 1. Infinite Square Well ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("1. INFINITE SQUARE WELL")
    x_inf = np.linspace(0, L, N + 2)[1:-1]   # interior points only
    V_inf = potential_infinite_well(x_inf)
    E_inf, psi_inf = solve(x_inf, V_inf, N_STATES)

    analytical_inf = np.array([(n**2 * np.pi**2) / (2 * L**2)
                                for n in range(1, N_STATES + 1)])
    ok1 = print_comparison("Infinite Well: E_n = n²π²/(2L²)", E_inf, analytical_inf, N_STATES)
    results_ok &= ok1

    print("\nNormalization check:")
    ok_norm1 = check_normalization(psi_inf, x_inf)
    results_ok &= ok_norm1

    print("\nNode count check (should be n-1 for 1-based n):")
    node_ok1 = True
    for i, psi in enumerate(psi_inf):
        nodes = count_nodes(psi)
        expected = i   # 0-based index i → i nodes
        status = "OK" if nodes == expected else "FAIL"
        print(f"  state {i} (n={i+1}): nodes={nodes}, expected={expected} [{status}]")
        if nodes != expected:
            node_ok1 = False
    results_ok &= node_ok1

    plot_results(x_inf, V_inf, E_inf, psi_inf,
                 "Infinite Square Well — Eigenstates",
                 "infinite_well.png")

    # ── 2. Harmonic Oscillator ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("2. HARMONIC OSCILLATOR")
    x_harm = np.linspace(-8, 8, N + 2)[1:-1]
    V_harm = potential_harmonic(x_harm)
    E_harm, psi_harm = solve(x_harm, V_harm, N_STATES)

    analytical_harm = np.array([(n + 0.5) * omega for n in range(N_STATES)])
    ok2 = print_comparison("Harmonic: E_n = (n+½)ω  (ω=1, 0-based n)", E_harm, analytical_harm, N_STATES)
    results_ok &= ok2

    print("\nNormalization check:")
    ok_norm2 = check_normalization(psi_harm, x_harm)
    results_ok &= ok_norm2

    print("\nNode count check:")
    node_ok2 = True
    for i, psi in enumerate(psi_harm):
        nodes = count_nodes(psi)
        expected = i
        status = "OK" if nodes == expected else "FAIL"
        print(f"  state {i}: nodes={nodes}, expected={expected} [{status}]")
        if nodes != expected:
            node_ok2 = False
    results_ok &= node_ok2

    plot_results(x_harm, V_harm, E_harm, psi_harm,
                 "Harmonic Oscillator — Eigenstates",
                 "harmonic.png",
                 x_lim=(-6, 6))

    # ── 3. Finite Square Well ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("3. FINITE SQUARE WELL")
    x_fin = np.linspace(-4 * a, 4 * a, N + 2)[1:-1]
    V_fin = potential_finite_well(x_fin)
    E_fin_all, psi_fin_all = solve(x_fin, V_fin, 20)   # compute many, filter bound

    # bound states: E < V0
    bound_mask = E_fin_all < V0
    E_bound = E_fin_all[bound_mask]
    psi_bound = psi_fin_all[bound_mask]

    # theoretical bound states: z0 = a*sqrt(2V0), N = floor(2*z0/π) + 1
    z0 = a * np.sqrt(2 * V0)
    n_theory = int(np.floor(2 * z0 / np.pi)) + 1
    n_found = len(E_bound)

    print(f"  Theoretical bound states : {n_theory}")
    print(f"  Numerical bound states   : {n_found}")
    ok3 = (n_found == n_theory)
    print(f"  Match: {'OK' if ok3 else 'FAIL'}")
    results_ok &= ok3

    # check exponential decay outside well for ground state
    psi_gs = psi_bound[0]
    x_right = x_fin[x_fin > a]
    psi_right = np.abs(psi_gs[x_fin > a])
    if len(x_right) > 10:
        # fit log|ψ| vs x → should be linear (decay)
        log_psi = np.log(psi_right + 1e-30)
        slope, _ = np.polyfit(x_right, log_psi, 1)
        print(f"  Ground state decay slope outside well: {slope:.3f} (should be negative)")
        ok_decay = slope < 0
        print(f"  Exponential decay: {'OK' if ok_decay else 'FAIL'}")
        results_ok &= ok_decay

    # use only first N_STATES bound states for plot
    n_plot = min(N_STATES, n_found)
    print("\nNormalization check:")
    ok_norm3 = check_normalization(psi_bound[:n_plot], x_fin)
    results_ok &= ok_norm3

    print(f"\nBound state energies (E < V0={V0}):")
    for i, E in enumerate(E_bound[:n_plot]):
        print(f"  n={i}: E = {E:.4f}")

    plot_results(x_fin, V_fin, E_bound[:n_plot], psi_bound[:n_plot],
                 f"Finite Square Well (V0={V0}, a={a}) — Bound States",
                 "finite_well.png",
                 x_lim=(-4 * a, 4 * a))

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    if results_ok:
        print("ALL CRITERIA PASSED ✓")
    else:
        print("SOME CRITERIA FAILED — review output above")
    print("="*60)


if __name__ == "__main__":
    main()
