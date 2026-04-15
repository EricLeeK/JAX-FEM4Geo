"""
L1: FD Gradient Baseline — Full inversion comparison AD vs FD.

Unlike J4 (which only *estimates* FD time by sampling), this experiment
actually *runs* full FD-gradient inversions across multiple mesh sizes
and compares:
  - Total inversion time (wall-clock)
  - Final accuracy (L2 relative error of recovered E field)
  - Convergence behavior (loss vs iteration and time)
  - Gradient accuracy (FD vs AD gradient at initial point)

Mesh sizes: 5×5(25), 8×8(64), 10×10(100), 15×15(225)
  + AD-only reference: 20×20(400)

Setup: Two-region E field (left=50000, right=90000), traction=-50 MPa,
k=500 (elastic regime), no noise, no regularization, log-E space.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from common import (
    InversionHeterogeneousDP2D,
    create_2d_mesh_traction_bc,
    generate_synthetic_observation,
    make_heterogeneous_loss,
    two_region_E_field,
    log_to_E, E_to_log,
    plot_E_field,
    save_results,
    RESULTS_DIR,
)
from jax_fem.solver import ad_wrapper

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
K_FIXED = 500.0
TRACTION = -50.0
Lx, Ly = 10., 10.
E_LEFT, E_RIGHT = 50000., 90000.
E_INIT = 70000.

OUT_DIR = os.path.join(RESULTS_DIR, 'l1_fd_baseline')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# FD gradient computation
# ---------------------------------------------------------------------------

def fd_gradient_central(loss_fn, x, eps=0.01):
    """Central-difference gradient: O(2N) forward evaluations.

    Parameters
    ----------
    loss_fn : callable
        loss_fn(x) -> scalar
    x : 1D array, shape (N,)
    eps : float
        Perturbation step size in log-E space.

    Returns
    -------
    grad : ndarray, shape (N,)
    """
    N = len(x)
    grad = onp.zeros(N, dtype=onp.float64)
    for i in range(N):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = float(loss_fn(np.array(x_plus)))
        f_minus = float(loss_fn(np.array(x_minus)))
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


# ---------------------------------------------------------------------------
# Build problem for a given mesh size
# ---------------------------------------------------------------------------

def build_problem(Nx, Ny):
    """Build E_true, observation, loss_fn, and AD grad_fn for a given mesh size.

    Returns dict with all components needed for both AD and FD inversion.
    """
    nc = Nx * Ny
    E_true = two_region_E_field(Nx, Ny, E_LEFT, E_RIGHT)

    obs_data = generate_synthetic_observation(
        E_true, K_FIXED,
        traction=TRACTION, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        solver_options=SOLVER_OPTIONS,
    )
    u_obs = obs_data['u_obs']
    obs_indices = obs_data['obs_indices']

    mesh, bc_info, loc_fns, _ = create_2d_mesh_traction_bc(
        TRACTION, Lx, Ly, Nx, Ny)

    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        location_fns=loc_fns,
        E=E_INIT, k=K_FIXED,
        traction_value=TRACTION,
    )
    fwd_pred = ad_wrapper(problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    loss_fn = make_heterogeneous_loss(fwd_pred, u_obs, obs_indices)

    log_E_init = onp.array(E_to_log(np.full(nc, E_INIT)), dtype=onp.float64)

    return {
        'E_true': E_true,
        'loss_fn': loss_fn,
        'log_E_init': log_E_init,
        'nc': nc,
    }


# ---------------------------------------------------------------------------
# Run inversion with AD gradient
# ---------------------------------------------------------------------------

def run_ad_inversion(loss_fn, log_E_init, maxiter=200):
    """Run L-BFGS-B inversion using AD (jax.grad) gradient."""
    nc = len(log_E_init)
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    # JIT warmup
    _ = value_and_grad_fn(np.array(log_E_init))

    log_E_lo = float(onp.log(5000.))
    log_E_hi = float(onp.log(300000.))
    bounds = [(log_E_lo, log_E_hi)] * nc

    loss_history = []
    time_history = []
    t0 = time.time()

    def objective(x):
        log_E = np.array(x)
        loss, grad = value_and_grad_fn(log_E)
        loss_f = float(loss)
        grad_np = onp.array(grad, dtype=onp.float64)
        loss_history.append(loss_f)
        time_history.append(time.time() - t0)
        return loss_f, grad_np

    from scipy.optimize import minimize as scipy_minimize
    result = scipy_minimize(
        objective,
        x0=log_E_init.copy(),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12},
    )

    elapsed = time.time() - t0
    E_final = onp.exp(result.x)

    return {
        'E_final': E_final,
        'loss_history': loss_history,
        'time_history': time_history,
        'elapsed': elapsed,
        'nit': int(result.nit),
        'nfev': int(result.nfev),
        'converged': bool(result.success),
        'final_loss': float(result.fun),
    }


# ---------------------------------------------------------------------------
# Run inversion with FD gradient
# ---------------------------------------------------------------------------

def run_fd_inversion(loss_fn, log_E_init, maxiter=200, fd_eps=0.01):
    """Run L-BFGS-B inversion using central-difference FD gradient."""
    nc = len(log_E_init)

    # Warmup
    _ = float(loss_fn(np.array(log_E_init)))

    log_E_lo = float(onp.log(5000.))
    log_E_hi = float(onp.log(300000.))
    bounds = [(log_E_lo, log_E_hi)] * nc

    loss_history = []
    time_history = []
    n_fwd_evals = [0]
    t0 = time.time()

    def objective(x):
        x_np = onp.array(x, dtype=onp.float64)
        loss_val = float(loss_fn(np.array(x_np)))
        n_fwd_evals[0] += 1
        grad = fd_gradient_central(loss_fn, x_np, eps=fd_eps)
        n_fwd_evals[0] += 2 * nc  # central diff: 2N evals
        loss_history.append(loss_val)
        time_history.append(time.time() - t0)
        return loss_val, grad

    from scipy.optimize import minimize as scipy_minimize
    result = scipy_minimize(
        objective,
        x0=log_E_init.copy(),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12},
    )

    elapsed = time.time() - t0
    E_final = onp.exp(result.x)

    return {
        'E_final': E_final,
        'loss_history': loss_history,
        'time_history': time_history,
        'elapsed': elapsed,
        'nit': int(result.nit),
        'nfev': n_fwd_evals[0],
        'converged': bool(result.success),
        'final_loss': float(result.fun),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_l1_experiment():
    """Run L1: FD gradient baseline comparison across mesh sizes."""
    print("=" * 70)
    print("L1: FD GRADIENT BASELINE — AD vs FD FULL INVERSION")
    print("=" * 70)

    # Mesh configurations: (Nx, Ny, run_fd, maxiter_fd)
    # FD maxiter limited for larger meshes due to cost
    mesh_configs = [
        (5,  5,  True,  200),
        (8,  8,  True,  100),
        (10, 10, True,  50),
        (15, 15, True,  30),
        (20, 20, False, 0),   # AD only — FD too expensive
    ]

    all_results = []

    for Nx, Ny, run_fd, maxiter_fd in mesh_configs:
        nc = Nx * Ny
        print(f"\n{'═' * 60}")
        print(f"  Mesh: {Nx}×{Ny}  |  N = {nc} parameters")
        print(f"{'═' * 60}")

        # Build problem
        print("  Building problem...")
        setup = build_problem(Nx, Ny)
        E_true = setup['E_true']
        loss_fn = setup['loss_fn']
        log_E_init = setup['log_E_init']

        # --- AD inversion ---
        print(f"\n  [AD] Running inversion (maxiter=200)...")
        t_ad_start = time.time()
        ad_result = run_ad_inversion(loss_fn, log_E_init, maxiter=200)
        E_true_np = onp.array(E_true)
        ad_err = onp.abs(ad_result['E_final'] - E_true_np)
        ad_l2_rel = float(onp.sqrt(onp.mean((ad_err / E_true_np) ** 2)))
        print(f"  [AD] L2 rel error: {ad_l2_rel:.4f} ({ad_l2_rel:.2%})")
        print(f"  [AD] Time: {ad_result['elapsed']:.1f}s, "
              f"iters: {ad_result['nit']}, fwd evals: {ad_result['nfev']}")

        entry = {
            'Nx': Nx, 'Ny': Ny, 'nc': nc,
            'ad_l2_rel': ad_l2_rel,
            'ad_time': ad_result['elapsed'],
            'ad_nit': ad_result['nit'],
            'ad_nfev': ad_result['nfev'],
            'ad_converged': ad_result['converged'],
            'ad_final_loss': ad_result['final_loss'],
            'ad_loss_history': ad_result['loss_history'],
            'ad_time_history': ad_result['time_history'],
        }

        # --- FD inversion ---
        if run_fd:
            print(f"\n  [FD] Running inversion (maxiter={maxiter_fd})...")
            print(f"  [FD] Each gradient requires {2 * nc} forward solves "
                  f"(central diff, N={nc})")
            fd_result = run_fd_inversion(loss_fn, log_E_init,
                                         maxiter=maxiter_fd, fd_eps=0.01)
            fd_err = onp.abs(fd_result['E_final'] - E_true_np)
            fd_l2_rel = float(onp.sqrt(onp.mean((fd_err / E_true_np) ** 2)))
            print(f"  [FD] L2 rel error: {fd_l2_rel:.4f} ({fd_l2_rel:.2%})")
            print(f"  [FD] Time: {fd_result['elapsed']:.1f}s, "
                  f"iters: {fd_result['nit']}, fwd evals: {fd_result['nfev']}")

            speedup = fd_result['elapsed'] / max(ad_result['elapsed'], 0.01)
            print(f"  => AD speedup: {speedup:.1f}×")

            entry.update({
                'fd_l2_rel': fd_l2_rel,
                'fd_time': fd_result['elapsed'],
                'fd_nit': fd_result['nit'],
                'fd_nfev': fd_result['nfev'],
                'fd_converged': fd_result['converged'],
                'fd_final_loss': fd_result['final_loss'],
                'fd_loss_history': fd_result['loss_history'],
                'fd_time_history': fd_result['time_history'],
                'speedup': speedup,
            })

            # --- Gradient accuracy check at initial point ---
            print(f"\n  [Gradient check] Comparing AD vs FD gradient at init...")
            ad_grad = onp.array(jax.grad(loss_fn)(np.array(log_E_init)))
            fd_grad = fd_gradient_central(loss_fn, log_E_init, eps=0.01)
            grad_rel_err = float(onp.linalg.norm(ad_grad - fd_grad) /
                                 max(onp.linalg.norm(ad_grad), 1e-30))
            cos_sim = float(onp.dot(ad_grad, fd_grad) /
                            (onp.linalg.norm(ad_grad) * onp.linalg.norm(fd_grad) + 1e-30))
            print(f"  [Gradient check] Rel error: {grad_rel_err:.4e}, "
                  f"cosine similarity: {cos_sim:.6f}")
            entry['grad_rel_err'] = grad_rel_err
            entry['grad_cos_sim'] = cos_sim
        else:
            entry.update({
                'fd_l2_rel': None,
                'fd_time': None,
                'fd_nit': None,
                'fd_nfev': None,
                'speedup': None,
            })

        all_results.append(entry)

    # --- Summary table ---
    print(f"\n{'═' * 70}")
    print("L1 SUMMARY: AD vs FD Inversion")
    print(f"{'═' * 70}")
    header = (f"{'Mesh':>7s} | {'N':>5s} | {'AD time':>8s} | {'AD L2%':>7s} | "
              f"{'FD time':>8s} | {'FD L2%':>7s} | {'Speedup':>8s}")
    print(header)
    print("-" * 65)
    for r in all_results:
        mesh_str = f"{r['Nx']}×{r['Ny']}"
        ad_str = f"{r['ad_time']:.1f}s"
        ad_err = f"{r['ad_l2_rel']:.2%}"
        fd_str = f"{r['fd_time']:.1f}s" if r['fd_time'] is not None else "—"
        fd_err = f"{r['fd_l2_rel']:.2%}" if r['fd_l2_rel'] is not None else "—"
        sp_str = f"{r['speedup']:.1f}×" if r['speedup'] is not None else "—"
        print(f"{mesh_str:>7s} | {r['nc']:>5d} | {ad_str:>8s} | {ad_err:>7s} | "
              f"{fd_str:>8s} | {fd_err:>7s} | {sp_str:>8s}")

    # --- Save JSON (without large arrays) ---
    save_data = []
    for r in all_results:
        entry = {k: v for k, v in r.items()
                 if k not in ('ad_loss_history', 'ad_time_history',
                              'fd_loss_history', 'fd_time_history')}
        save_data.append(entry)

    json_path = os.path.join(OUT_DIR, 'l1_results.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # --- Plots ---
    print("\nGenerating plots...")
    _plot_results(all_results)

    return all_results


def _plot_results(all_results):
    """Generate L1 comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Plot 1: Time vs #params ──
    ax = axes[0, 0]
    Ns = [r['nc'] for r in all_results]
    ad_times = [r['ad_time'] for r in all_results]
    ax.loglog(Ns, ad_times, 'o-', color='#4C72B0', lw=2, ms=8, label='AD')

    fd_Ns = [r['nc'] for r in all_results if r['fd_time'] is not None]
    fd_times = [r['fd_time'] for r in all_results if r['fd_time'] is not None]
    if fd_times:
        ax.loglog(fd_Ns, fd_times, 's-', color='#C44E52', lw=2, ms=8, label='FD')
    ax.set_xlabel('Number of Parameters N')
    ax.set_ylabel('Total Inversion Time [s]')
    ax.set_title('Inversion Time vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Speedup vs #params ──
    ax = axes[0, 1]
    sp_Ns = [r['nc'] for r in all_results if r['speedup'] is not None]
    speedups = [r['speedup'] for r in all_results if r['speedup'] is not None]
    if speedups:
        ax.loglog(sp_Ns, speedups, 'o-', color='#55A868', lw=2, ms=8, label='Measured')
        ax.loglog(sp_Ns, sp_Ns, 'k--', alpha=0.4, label='O(N) theoretical')
        ax.set_xlabel('Number of Parameters N')
        ax.set_ylabel('AD Speedup (FD time / AD time)')
        ax.set_title('AD Speedup Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        for n, s in zip(sp_Ns, speedups):
            ax.annotate(f'{s:.0f}×', (n, s), textcoords='offset points',
                        xytext=(10, 5), fontsize=10)

    # ── Plot 3: L2 error vs #params ──
    ax = axes[1, 0]
    ad_errs = [r['ad_l2_rel'] * 100 for r in all_results]
    ax.semilogx(Ns, ad_errs, 'o-', color='#4C72B0', lw=2, ms=8, label='AD')
    fd_errs_N = [(r['nc'], r['fd_l2_rel'] * 100) for r in all_results
                 if r['fd_l2_rel'] is not None]
    if fd_errs_N:
        ax.semilogx([x[0] for x in fd_errs_N], [x[1] for x in fd_errs_N],
                     's-', color='#C44E52', lw=2, ms=8, label='FD')
    ax.axhline(5, color='gray', ls='--', lw=0.8, alpha=0.5, label='5% target')
    ax.set_xlabel('Number of Parameters N')
    ax.set_ylabel('L2 Relative Error [%]')
    ax.set_title('Inversion Accuracy vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Convergence curves (loss vs time) for selected sizes ──
    ax = axes[1, 1]
    colors_ad = ['#4C72B0', '#6C8EBF', '#8DAFD0']
    colors_fd = ['#C44E52', '#D4726C', '#E49A94']
    plot_idx = 0
    for r in all_results:
        if 'ad_loss_history' not in r:
            continue
        N = r['nc']
        if r['ad_time_history']:
            c_ad = colors_ad[min(plot_idx, len(colors_ad) - 1)]
            ax.semilogy(r['ad_time_history'], r['ad_loss_history'],
                        '-', color=c_ad, lw=1.5, alpha=0.8,
                        label=f'AD N={N}')
        if r.get('fd_time_history'):
            c_fd = colors_fd[min(plot_idx, len(colors_fd) - 1)]
            ax.semilogy(r['fd_time_history'], r['fd_loss_history'],
                        '--', color=c_fd, lw=1.5, alpha=0.8,
                        label=f'FD N={N}')
        plot_idx += 1
        if plot_idx >= 3:
            break
    ax.set_xlabel('Wall-Clock Time [s]')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence: Loss vs Time')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('L1: AD vs FD Full Inversion Comparison', fontsize=14, y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'AD与FD反演对比.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


if __name__ == "__main__":
    run_l1_experiment()
