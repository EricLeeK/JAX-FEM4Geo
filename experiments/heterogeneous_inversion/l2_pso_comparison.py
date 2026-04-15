"""
L2: PSO (Particle Swarm Optimization) Comparison.

Verifies the hypothesis that gradient-free PSO cannot handle 25+ dimensional
parameter inversion effectively. Compares PSO with AD-based L-BFGS-B on the
two-region E-field benchmark across mesh sizes.

Mesh sizes: 5×5(25), 8×8(64), 10×10(100)
  (PSO becomes infeasible above ~100 dims with reasonable compute budget)

Setup: Two-region E field, traction=-50 MPa, k=500 (elastic regime),
no noise, no regularization, log-E space.

PSO config:
  - Particles: min(50, 2*N) for reasonable swarm size
  - Iterations: 200
  - Bounds: log(5000) to log(300000)
  - pyswarms GlobalBestPSO with default cognitive/social coefficients
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

OUT_DIR = os.path.join(RESULTS_DIR, 'l2_pso_comparison')
os.makedirs(OUT_DIR, exist_ok=True)


def build_problem(Nx, Ny):
    """Build loss function and ground truth for a given mesh size."""
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


def run_ad_inversion(loss_fn, log_E_init, maxiter=200):
    """L-BFGS-B with AD gradient (reference)."""
    nc = len(log_E_init)
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    _ = value_and_grad_fn(np.array(log_E_init))

    log_E_lo = float(onp.log(5000.))
    log_E_hi = float(onp.log(300000.))
    bounds = [(log_E_lo, log_E_hi)] * nc

    loss_history = []
    t0 = time.time()

    def objective(x):
        log_E = np.array(x)
        loss, grad = value_and_grad_fn(log_E)
        loss_f = float(loss)
        loss_history.append(loss_f)
        return loss_f, onp.array(grad, dtype=onp.float64)

    from scipy.optimize import minimize as scipy_minimize
    result = scipy_minimize(
        objective, x0=log_E_init.copy(),
        method='L-BFGS-B', jac=True, bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12},
    )
    elapsed = time.time() - t0
    E_final = onp.exp(result.x)

    return {
        'E_final': E_final,
        'loss_history': loss_history,
        'elapsed': elapsed,
        'nit': int(result.nit),
        'nfev': int(result.nfev),
        'final_loss': float(result.fun),
    }


def run_pso_inversion(loss_fn, nc, n_particles=None, pso_iters=200):
    """PSO inversion using pyswarms."""
    import pyswarms as ps

    if n_particles is None:
        n_particles = min(50, max(20, 2 * nc))

    log_E_lo = float(onp.log(5000.))
    log_E_hi = float(onp.log(300000.))

    bounds_lower = onp.full(nc, log_E_lo)
    bounds_upper = onp.full(nc, log_E_hi)

    loss_history = []
    n_fwd_evals = [0]

    def pso_objective(X):
        """Evaluate loss for all particles. X shape: (n_particles, nc)."""
        costs = onp.zeros(X.shape[0])
        for i in range(X.shape[0]):
            costs[i] = float(loss_fn(np.array(X[i])))
            n_fwd_evals[0] += 1
        return costs

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=nc,
        options=options,
        bounds=(bounds_lower, bounds_upper),
    )

    t0 = time.time()
    best_cost, best_pos = optimizer.optimize(
        pso_objective, iters=pso_iters, verbose=False,
    )
    elapsed = time.time() - t0

    # Extract per-iteration best cost from optimizer history
    cost_history = [float(c) for c in optimizer.cost_history]

    E_final = onp.exp(best_pos)

    return {
        'E_final': E_final,
        'loss_history': cost_history,
        'elapsed': elapsed,
        'nit': pso_iters,
        'nfev': n_fwd_evals[0],
        'final_loss': float(best_cost),
        'n_particles': n_particles,
    }


def run_l2_experiment():
    """Run L2: PSO vs AD comparison across mesh sizes."""
    print("=" * 70)
    print("L2: PSO vs AD INVERSION COMPARISON")
    print("=" * 70)

    # Mesh configs: (Nx, Ny, pso_iters, n_particles)
    mesh_configs = [
        (5,  5,  200, 30),
        (8,  8,  200, 40),
        (10, 10, 100, 50),
    ]

    all_results = []

    for Nx, Ny, pso_iters, n_particles in mesh_configs:
        nc = Nx * Ny
        print(f"\n{'═' * 60}")
        print(f"  Mesh: {Nx}×{Ny}  |  N = {nc} parameters")
        print(f"{'═' * 60}")

        setup = build_problem(Nx, Ny)
        E_true = setup['E_true']
        loss_fn = setup['loss_fn']
        log_E_init = setup['log_E_init']
        E_true_np = onp.array(E_true)

        # --- AD inversion ---
        print(f"\n  [AD] Running L-BFGS-B (maxiter=200)...")
        ad_res = run_ad_inversion(loss_fn, log_E_init, maxiter=200)
        ad_err = onp.abs(ad_res['E_final'] - E_true_np)
        ad_l2_rel = float(onp.sqrt(onp.mean((ad_err / E_true_np) ** 2)))
        print(f"  [AD] L2 rel: {ad_l2_rel:.4f} ({ad_l2_rel:.2%}), "
              f"time: {ad_res['elapsed']:.1f}s, iters: {ad_res['nit']}, "
              f"fwd evals: {ad_res['nfev']}")

        # --- PSO inversion ---
        print(f"\n  [PSO] Running ({n_particles} particles, {pso_iters} iters)...")
        print(f"  [PSO] Budget: ~{n_particles * pso_iters} forward evaluations")
        pso_res = run_pso_inversion(loss_fn, nc,
                                     n_particles=n_particles,
                                     pso_iters=pso_iters)
        pso_err = onp.abs(pso_res['E_final'] - E_true_np)
        pso_l2_rel = float(onp.sqrt(onp.mean((pso_err / E_true_np) ** 2)))
        print(f"  [PSO] L2 rel: {pso_l2_rel:.4f} ({pso_l2_rel:.2%}), "
              f"time: {pso_res['elapsed']:.1f}s, "
              f"fwd evals: {pso_res['nfev']}")

        speedup = pso_res['elapsed'] / max(ad_res['elapsed'], 0.01)
        print(f"\n  => AD speedup: {speedup:.1f}×")
        print(f"  => AD accuracy improvement: {pso_l2_rel / max(ad_l2_rel, 1e-10):.1f}× "
              f"better L2 error")

        entry = {
            'Nx': Nx, 'Ny': Ny, 'nc': nc,
            'ad_l2_rel': ad_l2_rel,
            'ad_time': ad_res['elapsed'],
            'ad_nit': ad_res['nit'],
            'ad_nfev': ad_res['nfev'],
            'ad_final_loss': ad_res['final_loss'],
            'ad_loss_history': ad_res['loss_history'],
            'pso_l2_rel': pso_l2_rel,
            'pso_time': pso_res['elapsed'],
            'pso_nit': pso_res['nit'],
            'pso_nfev': pso_res['nfev'],
            'pso_final_loss': pso_res['final_loss'],
            'pso_n_particles': pso_res['n_particles'],
            'pso_loss_history': pso_res['loss_history'],
            'speedup': speedup,
        }
        all_results.append(entry)

    # --- Summary ---
    print(f"\n{'═' * 70}")
    print("L2 SUMMARY: AD (L-BFGS-B) vs PSO")
    print(f"{'═' * 70}")
    header = (f"{'Mesh':>7s} | {'N':>5s} | {'AD time':>8s} | {'AD L2%':>7s} | "
              f"{'PSO time':>9s} | {'PSO L2%':>8s} | {'Speedup':>8s}")
    print(header)
    print("-" * 70)
    for r in all_results:
        mesh_str = f"{r['Nx']}×{r['Ny']}"
        print(f"{mesh_str:>7s} | {r['nc']:>5d} | {r['ad_time']:>7.1f}s | "
              f"{r['ad_l2_rel']:>6.2%} | {r['pso_time']:>8.1f}s | "
              f"{r['pso_l2_rel']:>7.2%} | {r['speedup']:>7.1f}×")

    # --- Save JSON ---
    save_data = []
    for r in all_results:
        entry = {k: v for k, v in r.items()
                 if k not in ('ad_loss_history', 'pso_loss_history')}
        save_data.append(entry)

    json_path = os.path.join(OUT_DIR, 'l2_results.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # --- Plots ---
    print("\nGenerating plots...")
    _plot_results(all_results)

    return all_results


def _plot_results(all_results):
    """Generate L2 comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Plot 1: Time comparison ──
    ax = axes[0, 0]
    Ns = [r['nc'] for r in all_results]
    ad_times = [r['ad_time'] for r in all_results]
    pso_times = [r['pso_time'] for r in all_results]
    x = onp.arange(len(Ns))
    w = 0.35
    ax.bar(x - w/2, ad_times, w, label='AD (L-BFGS-B)', color='#4C72B0', alpha=0.85)
    ax.bar(x + w/2, pso_times, w, label='PSO', color='#DD8452', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}' for n in Ns])
    ax.set_xlabel('Number of Parameters N')
    ax.set_ylabel('Time [s]')
    ax.set_title('Inversion Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: L2 error comparison ──
    ax = axes[0, 1]
    ad_errs = [r['ad_l2_rel'] * 100 for r in all_results]
    pso_errs = [r['pso_l2_rel'] * 100 for r in all_results]
    ax.bar(x - w/2, ad_errs, w, label='AD (L-BFGS-B)', color='#4C72B0', alpha=0.85)
    ax.bar(x + w/2, pso_errs, w, label='PSO', color='#DD8452', alpha=0.85)
    ax.axhline(5, color='gray', ls='--', lw=0.8, alpha=0.5, label='5% target')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}' for n in Ns])
    ax.set_xlabel('Number of Parameters N')
    ax.set_ylabel('L2 Relative Error [%]')
    ax.set_title('Inversion Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Forward evaluations comparison ──
    ax = axes[1, 0]
    ad_evals = [r['ad_nfev'] for r in all_results]
    pso_evals = [r['pso_nfev'] for r in all_results]
    ax.bar(x - w/2, ad_evals, w, label='AD (L-BFGS-B)', color='#4C72B0', alpha=0.85)
    ax.bar(x + w/2, pso_evals, w, label='PSO', color='#DD8452', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}' for n in Ns])
    ax.set_xlabel('Number of Parameters N')
    ax.set_ylabel('Forward Evaluations')
    ax.set_title('Computational Cost')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 4: PSO convergence curves ──
    ax = axes[1, 1]
    colors = ['#4C72B0', '#55A868', '#C44E52']
    for i, r in enumerate(all_results):
        c = colors[min(i, len(colors) - 1)]
        N = r['nc']
        if r['pso_loss_history']:
            ax.semilogy(r['pso_loss_history'], '-', color=c, lw=1.5,
                        label=f'PSO N={N}')
        if r['ad_loss_history']:
            ax.semilogy(r['ad_loss_history'], '--', color=c, lw=1.0,
                        alpha=0.6, label=f'AD N={N}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence: PSO vs AD')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('L2: PSO vs AD (L-BFGS-B) Inversion Comparison', fontsize=14, y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'PSO与AD反演对比.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


if __name__ == "__main__":
    run_l2_experiment()
