"""
1D Time-Independent Schrödinger Equation solver via Finite Difference Method
Units: ℏ = m = 1
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Grid parameters ───────────────────────────────────────────────────────────
N = 2000          # number of interior grid points

# ── Potential parameters ──────────────────────────────────────────────────────
L = 1.0           # infinite well width
omega = 1.0       # harmonic oscillator frequency
V0 = 50.0         # finite well depth
a = 1.0           # finite well half-width

N_STATES = 5      # number of states to compute


# ── Potential functions ───────────────────────────────────────────────────────

def potential_infinite_well(x):
    return np.zeros_like(x)


def potential_harmonic(x):
    return 0.5 * omega**2 * x**2


def potential_finite_well(x):
    return np.where(np.abs(x) <= a, 0.0, V0)


# ── Core numerical routines ───────────────────────────────────────────────────

def build_hamiltonian(x, V):
    """Tridiagonal Hamiltonian via second-order finite difference."""
    n = len(x)
    dx = x[1] - x[0]
    t = 1.0 / (2.0 * dx**2)

    diag = 2.0 * t + V
    off = -t * np.ones(n - 1)
    return np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)


def solve(x, V, n_states):
    """Return (energies, wavefunctions) for lowest n_states eigenstates."""
    eigenvalues, eigenvectors = np.linalg.eigh(build_hamiltonian(x, V))

    psis = []
    for i in range(n_states):
        psi = eigenvectors[:, i]
        psi /= np.sqrt(np.trapz(psi**2, x))
        if psi[np.argmax(np.abs(psi))] < 0:
            psi = -psi
        psis.append(psi)

    return eigenvalues[:n_states], np.array(psis)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(x, V, energies, psis, title, filename, x_lim=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, V, 'k-', linewidth=2, label='V(x)')

    scale = 0.4 * (energies[-1] - energies[0]) if len(energies) > 1 else 1.0
    colors_psi  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_prob = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

    for i, (E, psi) in enumerate(zip(energies, psis)):
        ax.axhline(E, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(x[-1], E, f' $E_{i+1}$={E:.3f}', va='center', fontsize=8)
        ax.plot(x, E + scale * psi,    color=colors_psi[i],  linewidth=1.5,
                label='$\\psi_n$' if i == 0 else None)
        ax.plot(x, E + scale * psi**2, color=colors_prob[i], linewidth=1.5,
                linestyle='-.', label='$|\\psi|^2$' if i == 0 else None)

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
    signs = np.sign(psi[psi != 0.0])
    return int(np.sum(np.abs(np.diff(signs)) > 0))


def check_normalization(psis, x, tol=1e-3):
    ok = True
    for i, psi in enumerate(psis):
        err = abs(np.trapz(psi**2, x) - 1.0)
        status = "OK" if err < tol else "FAIL"
        print(f"  state {i}: err={err:.2e} [{status}]")
        ok &= err < tol
    return ok


def print_comparison(label, energies, analytical):
    print(f"\n{'─'*60}\n{label}")
    print(f"{'n':>4} {'Numerical':>12} {'Analytical':>12} {'Rel.Err%':>10}")
    print('─'*60)
    all_ok = True
    for i, (E, A) in enumerate(zip(energies, analytical)):
        rel_err = abs(E - A) / abs(A) * 100
        status = "OK" if rel_err < 1.0 else "FAIL"
        print(f"  {i+1:2d} {E:12.6f} {A:12.6f} {rel_err:10.4f}%  [{status}]")
        all_ok &= rel_err < 1.0
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ok = True

    # 1. Infinite Square Well
    print("\n" + "="*60 + "\n1. INFINITE SQUARE WELL")
    x = np.linspace(0, L, N + 2)[1:-1]
    E, psis = solve(x, potential_infinite_well(x), N_STATES)
    analytical = np.array([(n**2 * np.pi**2) / (2 * L**2) for n in range(1, N_STATES + 1)])
    ok &= print_comparison("E_n = n²π²/(2L²)", E, analytical)
    print("\nNormalization:"); ok &= check_normalization(psis, x)
    print("\nNodes:")
    for i, psi in enumerate(psis):
        n = count_nodes(psi)
        s = "OK" if n == i else "FAIL"
        print(f"  state {i}: nodes={n}, expected={i} [{s}]")
        ok &= n == i
    plot_results(x, potential_infinite_well(x), E, psis,
                 "Infinite Square Well — Eigenstates", "infinite_well.png")

    # 2. Harmonic Oscillator
    print("\n" + "="*60 + "\n2. HARMONIC OSCILLATOR")
    x = np.linspace(-8, 8, N + 2)[1:-1]
    E, psis = solve(x, potential_harmonic(x), N_STATES)
    analytical = np.array([(n + 0.5) * omega for n in range(N_STATES)])
    ok &= print_comparison("E_n = (n+½)ω  (0-based)", E, analytical)
    print("\nNormalization:"); ok &= check_normalization(psis, x)
    print("\nNodes:")
    for i, psi in enumerate(psis):
        n = count_nodes(psi)
        s = "OK" if n == i else "FAIL"
        print(f"  state {i}: nodes={n}, expected={i} [{s}]")
        ok &= n == i
    plot_results(x, potential_harmonic(x), E, psis,
                 "Harmonic Oscillator — Eigenstates", "harmonic.png", x_lim=(-6, 6))

    # 3. Finite Square Well
    print("\n" + "="*60 + "\n3. FINITE SQUARE WELL")
    x = np.linspace(-4*a, 4*a, N + 2)[1:-1]
    V = potential_finite_well(x)
    E_all, psi_all = solve(x, V, 20)
    mask = E_all < V0
    E_b, psi_b = E_all[mask], psi_all[mask]

    n_theory = int(np.floor(2 * a * np.sqrt(2 * V0) / np.pi)) + 1
    n_found  = len(E_b)
    match = n_found == n_theory
    print(f"  Theory: {n_theory}, Found: {n_found} [{'OK' if match else 'FAIL'}]")
    ok &= match

    slope, _ = np.polyfit(x[x > a], np.log(np.abs(psi_b[0][x > a]) + 1e-30), 1)
    ok &= slope < 0
    print(f"  Decay slope: {slope:.3f} [{'OK' if slope < 0 else 'FAIL'}]")

    n_plot = min(N_STATES, n_found)
    print("\nNormalization:"); ok &= check_normalization(psi_b[:n_plot], x)
    for i, E in enumerate(E_b[:n_plot]):
        print(f"  n={i}: E = {E:.4f}")
    plot_results(x, V, E_b[:n_plot], psi_b[:n_plot],
                 f"Finite Square Well (V0={V0}, a={a})", "finite_well.png",
                 x_lim=(-4*a, 4*a))

    print("\n" + "="*60)
    print("ALL CRITERIA PASSED ✓" if ok else "SOME CRITERIA FAILED")
    print("="*60)


if __name__ == "__main__":
    main()
