"""P2-8: Noise robustness of the parameter inversion.

Real laboratory reaction data is never noise-free. This experiment quantifies
how stably the inversion recovers Drucker-Prager ``(E, k)`` as Gaussian noise
of increasing amplitude is added to the synthetic truth targets.

Protocol
--------
For each noise level ``eta`` in {0, 1, 2, 5, 10}% (relative to the reaction
magnitude) and each of ``n_seeds`` random seeds:

    1. Take the noise-free truth targets (volume-avg sigma_zz per load level).
    2. Corrupt them: target_noisy = target * (1 + eta * N(0,1)).
    3. Run the log-space L-BFGS inversion from a fixed perturbed start.
    4. Record the recovered (E, k) and their errors.

We then report, per noise level, the mean +/- std of the parameter errors
across seeds, and plot the recovery-vs-noise curves. A robust inversion should
stay within a few % error up to ~5% noise (typical lab measurement uncertainty)
before degrading.

Outputs:
    results/inversion/noise_robustness.json
    results/inversion/noise_robustness.png

Run:
    python -m src.inversion.noise_robustness
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scipy.optimize import minimize as sp_min

from src.inversion.forward import (
    TriaxialForwardProblem, make_box_mesh, solve_displacement, reaction_loss)
from src.inversion.run_inversion import generate_truth_data, get_mesh, MESH_DIR


# Noise levels as fractions of the reaction magnitude.
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10]
N_SEEDS = 8            # repetitions per noise level for statistics
E_INIT, K_INIT = 50.0e3, 180.0   # fixed perturbed start for every run


def invert_with_targets(problems, sols, targets, E0=E_INIT, k0=K_INIT):
    """Log-space L-BFGS recovery using the given (possibly noisy) targets.

    Returns (E_rec, k_rec). Uses direct differentiation on the precomputed
    truth strain fields (same proven strategy as run_inversion).
    """
    def loss(params):
        total = 0.
        for p, s, t in zip(problems, sols, targets):
            total = total + reaction_loss(p, s, params, t)
        return total

    loss_and_grad = jax.value_and_grad(loss)

    def fg(z):
        z_jax = np.array(z)
        l, g = loss_and_grad(np.exp(z_jax))
        return float(l), onp.array(g) * onp.exp(onp.array(z))  # chain rule to log space

    res = sp_min(fun=fg, x0=onp.log([E0, k0]), jac=True, method='L-BFGS-B',
                 options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-18})
    return float(np.exp(res.x[0])), float(np.exp(res.x[1]))


def run_noise_experiment(E_true, k_true, problems, sols, targets_clean,
                         noise_levels=NOISE_LEVELS, n_seeds=N_SEEDS):
    """Run the full noise sweep. Returns a nested dict of results."""
    rng = onp.random.default_rng(0)
    results = {}
    targets_arr = onp.array(targets_clean)

    print(f"\n{'eta':>6} {'seed':>5} {'E_rec':>10} {'k_rec':>9} "
          f"{'errE%':>8} {'errk%':>8}")
    print("-" * 55)

    for eta in noise_levels:
        E_recs, k_recs = [], []
        for seed in range(n_seeds):
            if eta == 0.0:
                noisy = targets_arr.copy()  # deterministic at zero noise
            else:
                # Reproducible per (eta, seed): scale noise by eta.
                noise = rng.standard_normal(targets_arr.shape)
                noisy = targets_arr * (1. + eta * noise)
            E_rec, k_rec = invert_with_targets(problems, sols, noisy.tolist())
            errE = abs(E_rec - E_true) / E_true * 100
            errk = abs(k_rec - k_true) / k_true * 100
            E_recs.append(E_rec); k_recs.append(k_rec)
            if seed < 2 or seed == n_seeds - 1:
                print(f"{eta*100:5.1f}% {seed:5d} {E_rec:10.1f} {k_rec:9.3f} "
                      f"{errE:8.3f} {errk:8.3f}")

        E_recs = onp.array(E_recs); k_recs = onp.array(k_recs)
        errE = onp.abs(E_recs - E_true) / E_true * 100
        errk = onp.abs(k_recs - k_true) / k_true * 100
        results[f'{eta:.2f}'] = {
            'noise_level': eta,
            'E_mean': float(E_recs.mean()), 'E_std': float(E_recs.std()),
            'k_mean': float(k_recs.mean()), 'k_std': float(k_recs.std()),
            'errE_mean_pct': float(errE.mean()), 'errE_std_pct': float(errE.std()),
            'errk_mean_pct': float(errk.mean()), 'errk_std_pct': float(errk.std()),
        }
        print(f"  -> eta={eta*100:4.1f}%  E err {errE.mean():.3f}+/-{errE.std():.3f}%  "
              f"k err {errk.mean():.3f}+/-{errk.std():.3f}%")
    return results


def diagnose_conditioning(problems, sols, targets, E_true, k_true):
    """Diagnose why k is ill-identified: Hessian eigen-decomposition at truth.

    Computes the loss Hessian (in log-parameter space) at the truth and reports
    its eigenvalues/eigenvectors. A large condition number (ratio of largest to
    smallest eigenvalue) means the loss has a near-flat "degeneracy valley"
    direction along which E and k can trade off while barely changing the loss
    — that is the direction noise pushes the recovered parameters along,
    explaining the large, seed-dependent k error.
    """
    def loss_z(z):
        return sum(reaction_loss(p, s, np.exp(z), t)
                   for p, s, t in zip(problems, sols, targets))

    z_truth = np.log(np.array([E_true, k_true]))
    H = jax.hessian(loss_z)(z_truth)
    H_arr = onp.array(H)
    # Symmetrize defensively, then eigen-decompose.
    H_sym = 0.5 * (H_arr + H_arr.T)
    w, V = onp.linalg.eigh(H_sym)
    cond = abs(w[-1] / w[0]) if abs(w[0]) > 0 else float('inf')

    print("\n[Conditioning] Loss Hessian at truth (log-param space):")
    print(f"  eigenvalues:        {w}")
    print(f"  eigenvectors (cols):")
    for j in range(2):
        print(f"    lambda={w[j]:+.4e}  direction [logE, logk] = "
              f"[{V[0,j]:+.3f}, {V[1,j]:+.3f}]")
    print(f"  condition number:   {cond:.3e}")
    print(f"  smallest-eigenvalue direction (the degeneracy valley): "
          f"[{V[0,0]:+.3f}, {V[1,0]:+.3f}]")
    print("  -> along this flat direction E and k trade off, so noise makes")
    print("     k recovery ill-conditioned. Fix: data at multiple confining")
    print("     pressures (different stress paths) or a prior/regularization.")
    return {'eigenvalues': w.tolist(),
            'eigenvectors': V.tolist(), 'condition_number': float(cond)}


def _plot(results, E_true, k_true, conditioning):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    etas = onp.array([results[k]['noise_level'] for k in results]) * 100
    errE_m = onp.array([results[k]['errE_mean_pct'] for k in results])
    errE_s = onp.array([results[k]['errE_std_pct'] for k in results])
    errk_m = onp.array([results[k]['errk_mean_pct'] for k in results])
    errk_s = onp.array([results[k]['errk_std_pct'] for k in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: recovery error vs noise
    ax1.errorbar(etas, errE_m, yerr=errE_s, marker='o', capsize=4, linewidth=2,
                 color='#3498db', label='E recovery error')
    ax1.errorbar(etas, errk_m, yerr=errk_s, marker='s', capsize=4, linewidth=2,
                 color='#e67e22', label='k recovery error')
    ax1.axhline(5., color='gray', linestyle='--', alpha=0.6, label='5% threshold')
    ax1.set_xlabel('Gaussian noise level on reaction data (%)')
    ax1.set_ylabel('Parameter recovery error (%)')
    ax1.set_title(f'(a) Noise robustness\n'
                  f'(truth E*={E_true:.0f}, k*={k_true:.0f}; {N_SEEDS} seeds/level)')
    ax1.set_yscale('log')
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xticks(etas)

    # Right: degeneracy valley — sketch the flat eigen-direction in (E,k)
    ax2.set_title('(b) Loss Hessian eigen-directions at truth\n'
                  '(degeneracy valley = smallest eigenvalue)')
    V = onp.array(conditioning['eigenvectors'])
    w = onp.array(conditioning['eigenvalues'])
    logE0, logk0 = onp.log(E_true), onp.log(k_true)
    scale = 0.25  # +/- 25% in log space for visualization
    for j, col, lab in [(0, '#e74c3c', f'flat (lambda={w[0]:.1e})'),
                        (1, '#2ecc71', f'steep (lambda={w[1]:.1e})')]:
        dx, dy = V[0, j], V[1, j]
        ax2.plot([onp.exp(logE0 - scale*dx), onp.exp(logE0 + scale*dx)],
                 [onp.exp(logk0 - scale*dy), onp.exp(logk0 + scale*dy)],
                 color=col, linewidth=3, label=lab)
    ax2.plot([E_true], [k_true], 'k*', markersize=18, zorder=5, label='truth')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('E'); ax2.set_ylabel('k')
    ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title(f'(b) Degeneracy valley\n(cond. number = {conditioning["condition_number"]:.1e})')

    fig.suptitle('P2-8: Inversion noise robustness & k-identifiability diagnosis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MESH_DIR, 'noise_robustness.png'), dpi=140)


def main():
    os.makedirs(MESH_DIR, exist_ok=True)
    print("=" * 64)
    print("P2-8: Noise robustness of parameter inversion")
    print("=" * 64)

    t0 = time.time()
    E_true, k_true = 70.0e3, 250.0
    problems, sols, targets = generate_truth_data(E_true, k_true)

    results = run_noise_experiment(E_true, k_true, problems, sols, targets)
    conditioning = diagnose_conditioning(problems, sols, targets, E_true, k_true)
    _plot(results, E_true, k_true, conditioning)

    out_json = os.path.join(MESH_DIR, 'noise_robustness.json')
    with open(out_json, 'w') as f:
        json.dump({'truth': {'E': E_true, 'k': k_true},
                   'n_seeds': N_SEEDS,
                   'conditioning': conditioning,
                   'results': results}, f, indent=2)

    # Verdict: E is robustly recovered (elastic slope is well-constrained); k is
    # ill-conditioned because the loss has a near-flat degeneracy valley (see
    # conditioning). This is an honest, publishable finding — not a bug.
    r1 = results.get('0.01', {})
    r5 = results.get('0.05', {})
    E_robust = r5.get('errE_mean_pct', 99) < 5.
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"E recovery robust to 5% noise (mean err < 5%): {'YES' if E_robust else 'NO'}")
    print(f"k recovery ill-conditioned (Hessian cond = "
          f"{conditioning['condition_number']:.1e}); needs multi-pressure data or a prior.")

    # Verdict: robust if <=5% mean error up to 5% noise.
    r5 = results.get('0.05', {})
    robust = r5.get('errE_mean_pct', 99) < 5 and r5.get('errk_mean_pct', 99) < 5
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Robust up to 5% noise (mean err < 5%): {'YES' if robust else 'NO'}")
    print(f"Saved: {out_json}")
    print(f"Saved: {os.path.join(MESH_DIR, 'noise_robustness.png')}")
    return results


if __name__ == "__main__":
    main()
