#!/usr/bin/env python
"""
MC AD vs FD Speed Comparison.

Compares AD (adjoint) vs FD (central difference) gradient-based inversion
for the MC two-region c(x) problem. Single-step solve for simplicity.

For N=100 cells, each FD gradient evaluation requires 2N+1 = 201 forward
solves, while AD requires 1 forward + 1 adjoint solve.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from common import (
    InversionHeterogeneousMC2D,
    create_2d_mesh_and_bc,
    make_heterogeneous_loss,
    two_region_c_field,
    uniform_c_field,
    log_to_E as log_to_c,
    E_to_log as c_to_log,
    save_results,
    RESULTS_DIR,
)

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import solver, ad_wrapper
from scipy.optimize import minimize as scipy_minimize

Nx, Ny = 10, 10
Lx, Ly = 10., 10.
DISPLACEMENT = -0.03
PHI_DEG = 30.0
PSI_DEG = 15.0
E_FIXED = 70000.
C_LEFT, C_RIGHT = 30., 70.
C_INIT = 50.
NUM_CELLS = Nx * Ny
SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
MAXITER = 50  # enough to see convergence trend

OUT_DIR = os.path.join(RESULTS_DIR, 'mc_fd_comparison')
os.makedirs(OUT_DIR, exist_ok=True)


def fd_gradient_central(loss_fn, x, eps=0.01):
    N = len(x)
    grad = onp.zeros(N, dtype=onp.float64)
    for i in range(N):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (float(loss_fn(np.array(x_plus)))
                    - float(loss_fn(np.array(x_minus)))) / (2 * eps)
    return grad


def build_problem():
    c_true = two_region_c_field(Nx, Ny, C_LEFT, C_RIGHT)
    mesh, bc = create_2d_mesh_and_bc(DISPLACEMENT, Lx, Ly, Nx, Ny)
    obs_problem = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    obs_problem.set_params(np.array(c_true))
    u_obs = solver(obs_problem, solver_options=SOLVER_OPTIONS)[0]
    obs_indices = onp.arange(u_obs.shape[0])

    mesh2, bc2 = create_2d_mesh_and_bc(DISPLACEMENT, Lx, Ly, Nx, Ny)
    inv_problem = InversionHeterogeneousMC2D(
        mesh2, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc2,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    fwd_pred = ad_wrapper(inv_problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)
    loss_fn = make_heterogeneous_loss(fwd_pred, u_obs, obs_indices)
    log_c_init = onp.array(c_to_log(uniform_c_field(Nx, Ny, C_INIT)),
                           dtype=onp.float64)

    return c_true, loss_fn, log_c_init


def run_ad(loss_fn, x0):
    vg = jax.value_and_grad(loss_fn)
    # Warmup (exclude from timing)
    _ = vg(np.array(x0))

    bounds = [(float(onp.log(5.)), float(onp.log(500.)))] * NUM_CELLS
    history_loss, history_time = [], []
    last_good = [None, None]
    t0 = time.time()

    def objective(x):
        try:
            loss, grad = vg(np.array(x))
            lf = float(loss)
            gn = onp.array(grad, dtype=onp.float64)
            if not (onp.isfinite(lf) and onp.all(onp.isfinite(gn))):
                raise ValueError("NaN")
            last_good[0] = lf
            last_good[1] = gn
        except Exception:
            lf = last_good[0] * 10. if last_good[0] else 1e10
            gn = onp.zeros_like(x)
        history_loss.append(lf)
        history_time.append(time.time() - t0)
        return lf, gn

    result = scipy_minimize(
        objective, x0=x0.copy(), method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': MAXITER, 'ftol': 1e-20, 'gtol': 1e-12},
    )
    elapsed = time.time() - t0
    return {
        'c_final': onp.exp(result.x),
        'loss_history': history_loss,
        'time_history': history_time,
        'elapsed': elapsed,
        'nit': int(result.nit),
        'nfev': len(history_loss),
    }


def safe_loss(loss_fn, x):
    """Evaluate loss, return inf on failure."""
    try:
        val = float(loss_fn(np.array(x)))
        return val if onp.isfinite(val) else 1e20
    except Exception:
        return 1e20


def fd_gradient_safe(loss_fn, x, eps=0.01):
    N = len(x)
    grad = onp.zeros(N, dtype=onp.float64)
    for i in range(N):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        fp = safe_loss(loss_fn, x_plus)
        fm = safe_loss(loss_fn, x_minus)
        grad[i] = (fp - fm) / (2 * eps)
    return grad


def run_fd(loss_fn, x0, fd_eps=0.01):
    # Warmup
    _ = safe_loss(loss_fn, x0)

    bounds = [(float(onp.log(5.)), float(onp.log(500.)))] * NUM_CELLS
    history_loss, history_time = [], []
    n_fwd = [0]
    t0 = time.time()

    def objective(x):
        x_np = onp.array(x, dtype=onp.float64)
        lf = safe_loss(loss_fn, x_np)
        n_fwd[0] += 1
        grad = fd_gradient_safe(loss_fn, x_np, eps=fd_eps)
        n_fwd[0] += 2 * NUM_CELLS
        history_loss.append(lf)
        history_time.append(time.time() - t0)
        return lf, grad

    result = scipy_minimize(
        objective, x0=x0.copy(), method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': MAXITER, 'ftol': 1e-20, 'gtol': 1e-12},
    )
    elapsed = time.time() - t0
    return {
        'c_final': onp.exp(result.x),
        'loss_history': history_loss,
        'time_history': history_time,
        'elapsed': elapsed,
        'nit': int(result.nit),
        'nfev': n_fwd[0],
    }


def compute_errors(c_final, c_true):
    c_true_np = onp.array(c_true)
    rel = onp.abs(c_final - c_true_np) / c_true_np
    l2 = float(onp.sqrt(onp.mean(rel ** 2)))
    left_mask = c_true_np < 50.
    left_err = float(onp.mean(onp.abs(c_final[left_mask] - C_LEFT) / C_LEFT))
    right_err = float(onp.mean(onp.abs(c_final[~left_mask] - C_RIGHT) / C_RIGHT))
    return l2, left_err, right_err


def main():
    print("=" * 70)
    print("MC AD vs FD Speed Comparison")
    print(f"  Mesh: {Nx}×{Ny} ({NUM_CELLS} cells), maxiter={MAXITER}")
    print(f"  FD: 2×{NUM_CELLS}+1 = {2*NUM_CELLS+1} forward solves per gradient")
    print(f"  AD: 1 forward + 1 adjoint per gradient")
    print("=" * 70)

    c_true, loss_fn, x0 = build_problem()

    # --- AD inversion ---
    print("\n  Running AD inversion...")
    ad_res = run_ad(loss_fn, x0)
    l2_ad, left_ad, right_ad = compute_errors(ad_res['c_final'], c_true)
    print(f"  AD done: {ad_res['elapsed']:.1f}s, {ad_res['nfev']} evals, "
          f"L2={l2_ad:.2%}")

    # --- FD inversion ---
    print(f"\n  Running FD inversion...")
    fd_res = run_fd(loss_fn, x0)
    l2_fd, left_fd, right_fd = compute_errors(fd_res['c_final'], c_true)
    print(f"  FD done: {fd_res['elapsed']:.1f}s, {fd_res['nfev']} fwd evals, "
          f"L2={l2_fd:.2%}")

    # --- Summary ---
    speedup = fd_res['elapsed'] / max(ad_res['elapsed'], 0.01)
    print(f"\n{'=' * 70}")
    print(f"  Summary:")
    print(f"  {'':30s} {'AD':>12s} {'FD':>12s} {'Ratio':>8s}")
    print(f"  {'-' * 62}")
    print(f"  {'Wall time [s]':30s} {ad_res['elapsed']:12.1f} "
          f"{fd_res['elapsed']:12.1f} {speedup:7.1f}×")
    print(f"  {'Optimizer iterations':30s} {ad_res['nit']:12d} "
          f"{fd_res['nit']:12d}")
    print(f"  {'Forward evaluations':30s} {ad_res['nfev']:12d} "
          f"{fd_res['nfev']:12d}")
    print(f"  {'L2 relative error':30s} {l2_ad:11.2%} "
          f"{l2_fd:11.2%}")
    print(f"  {'Speedup (FD/AD)':30s} {'':12s} {'':12s} {speedup:7.1f}×")
    print(f"{'=' * 70}")

    # --- Gradient timing breakdown ---
    # Time a single gradient evaluation
    print("\n  Single gradient evaluation timing:")
    t_ad = time.time()
    _ = jax.value_and_grad(loss_fn)(np.array(x0))
    t_ad = time.time() - t_ad

    t_fd = time.time()
    _ = float(loss_fn(np.array(x0)))
    _ = fd_gradient_central(loss_fn, onp.array(x0, dtype=onp.float64))
    t_fd = time.time() - t_fd

    print(f"    AD (1 fwd + 1 adj):  {t_ad:.3f}s")
    print(f"    FD (201 fwd):        {t_fd:.3f}s")
    print(f"    Per-gradient speedup: {t_fd/max(t_ad, 0.001):.1f}×")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(ad_res['time_history'], ad_res['loss_history'],
                'b-o', ms=3, lw=1.5, label=f'AD ({ad_res["elapsed"]:.0f}s)')
    ax.semilogy(fd_res['time_history'], fd_res['loss_history'],
                'r-s', ms=3, lw=1.5, label=f'FD ({fd_res["elapsed"]:.0f}s)')
    ax.set_xlabel('Wall time [s]')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(ad_res['loss_history'], 'b-o', ms=3, lw=1.5, label='AD')
    ax.semilogy(fd_res['loss_history'], 'r-s', ms=3, lw=1.5, label='FD')
    ax.set_xlabel('Optimizer evaluation')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence vs Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'MC c(x) Inversion: AD vs FD  |  Speedup: {speedup:.0f}×',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mc_ad_vs_fd.png'), dpi=150)
    plt.close()

    save_results(OUT_DIR, {
        'ad_time': ad_res['elapsed'], 'fd_time': fd_res['elapsed'],
        'speedup': speedup,
        'ad_l2': l2_ad, 'fd_l2': l2_fd,
        'ad_nit': ad_res['nit'], 'fd_nit': fd_res['nit'],
        'ad_nfev': ad_res['nfev'], 'fd_nfev': fd_res['nfev'],
        'num_cells': NUM_CELLS, 'maxiter': MAXITER,
        'single_grad_ad_s': t_ad, 'single_grad_fd_s': t_fd,
    })
    print(f"\n  Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
