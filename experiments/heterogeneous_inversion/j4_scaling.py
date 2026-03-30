"""
J4: Dimension scaling experiment — AD vs FD gradient computation time.

Benchmarks how AD gradient time scales with the number of parameters (mesh elements)
compared to finite-difference gradient time. Demonstrates the O(1)-vs-forward-solve
advantage of adjoint-based AD.

Mesh sizes: 5×5 (25), 10×10 (100), 20×20 (400), 30×30 (900), 50×50 (2500, AD only)
True E field: two-region (left=50000, right=90000)
Traction: -50.0 MPa on top, k fixed at 50.0
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import json
import statistics

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
    log_to_E, E_to_log,
    RESULTS_DIR,
)
from jax_fem.solver import ad_wrapper


def build_two_region_E_field(Nx, Ny):
    """Left half E=50000, right half E=90000."""
    nc = Nx * Ny
    E_field = onp.full(nc, 90000.0)
    for ix in range(Nx):
        for iy in range(Ny):
            cell_idx = ix * Ny + iy
            if ix < Nx // 2:
                E_field[cell_idx] = 50000.0
    return np.array(E_field)


def _build_problem_and_loss(Nx, Ny, traction=-50.0, k_fixed=50.0):
    """Build mesh, problem, fwd_pred, loss_fn, E_true, log_E_init for a given mesh size."""
    Lx, Ly = 10., 10.
    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

    E_true = build_two_region_E_field(Nx, Ny)

    obs_data = generate_synthetic_observation(
        E_true, k_fixed,
        traction=traction, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        solver_options=solver_options,
    )
    u_obs = obs_data['u_obs']
    obs_indices = obs_data['obs_indices']

    mesh, bc_info, loc_fns, _ = create_2d_mesh_traction_bc(
        traction, Lx, Ly, Nx, Ny)

    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        location_fns=loc_fns,
        E=70000.0, k=k_fixed,
        traction_value=traction,
    )
    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    loss_fn = make_heterogeneous_loss(fwd_pred, u_obs, obs_indices)

    log_E_init = E_to_log(np.full(Nx * Ny, 70000.0))

    return loss_fn, log_E_init


def time_ad_gradient(Nx, Ny, traction=-50.0, k_fixed=50.0, n_repeats=3):
    """Time AD gradient computation for a given mesh size.

    Returns
    -------
    ad_time : float — median wall-clock time (seconds) for one gradient evaluation
    """
    print(f"  [AD] Building {Nx}×{Ny} problem...")
    loss_fn, log_E_init = _build_problem_and_loss(Nx, Ny, traction, k_fixed)
    grad_fn = jax.grad(loss_fn)

    # JIT warmup
    print(f"  [AD] JIT warmup...")
    _ = grad_fn(log_E_init)

    # Timed runs
    times = []
    for r in range(n_repeats):
        t0 = time.time()
        g = grad_fn(log_E_init)
        g.block_until_ready()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"    run {r+1}/{n_repeats}: {elapsed:.4f}s")

    ad_time = statistics.median(times)
    print(f"  [AD] median = {ad_time:.4f}s")
    return ad_time


def time_fd_gradient(Nx, Ny, n_params_sample=10, traction=-50.0,
                     k_fixed=50.0, eps=0.01):
    """Estimate FD gradient time by timing a subset of parameter perturbations.

    Uses central differences in log-E space:
        dL/d(log_E_i) ≈ (L(log_E+eps*e_i) - L(log_E-eps*e_i)) / (2*eps)

    Returns
    -------
    fd_time_per_param : float — time for one forward-difference pair (seconds)
    n_sampled : int — number of parameters actually sampled
    """
    print(f"  [FD] Building {Nx}×{Ny} problem...")
    loss_fn, log_E_init = _build_problem_and_loss(Nx, Ny, traction, k_fixed)
    N = len(log_E_init)

    # Clamp sample size
    n_sampled = min(n_params_sample, N)

    # Choose evenly spaced parameter indices
    indices = onp.linspace(0, N - 1, n_sampled, dtype=int)

    # Warmup: one forward eval
    print(f"  [FD] Warmup forward eval...")
    _ = loss_fn(log_E_init)

    # Time FD for sampled parameters
    print(f"  [FD] Timing {n_sampled} central-difference evaluations...")
    t0 = time.time()
    for i in indices:
        e_i = np.zeros(N).at[i].set(1.0)
        _ = loss_fn(log_E_init + eps * e_i)
        _ = loss_fn(log_E_init - eps * e_i)
    fd_elapsed = time.time() - t0

    fd_time_per_param = fd_elapsed / n_sampled
    print(f"  [FD] {n_sampled} params in {fd_elapsed:.4f}s "
          f"({fd_time_per_param:.4f}s per param)")
    return fd_time_per_param, n_sampled


def run_scaling_experiment():
    """Run the full scaling experiment across multiple mesh sizes."""
    print("=" * 70)
    print("J4: Dimension Scaling Experiment — AD vs FD Gradient Time")
    print("=" * 70)

    traction = -50.0
    k_fixed = 50.0

    # Mesh configurations: (Nx, Ny, run_fd)
    mesh_configs = [
        (5,  5,  True),
        (10, 10, True),
        (20, 20, True),
        (30, 30, True),
        (50, 50, False),  # AD only — FD too slow
    ]

    results = []

    for Nx, Ny, run_fd in mesh_configs:
        N = Nx * Ny
        print(f"\n{'─' * 60}")
        print(f"  Mesh: {Nx}×{Ny}  |  N = {N} parameters")
        print(f"{'─' * 60}")

        # AD timing
        ad_time = time_ad_gradient(Nx, Ny, traction, k_fixed)

        fd_time_est = None
        speedup = None

        if run_fd:
            n_sample = 10 if N <= 400 else 5
            fd_time_per_param, n_sampled = time_fd_gradient(
                Nx, Ny, n_params_sample=n_sample,
                traction=traction, k_fixed=k_fixed,
            )
            # Full FD gradient requires 2*N forward evals (central diff),
            # but we already timed pairs, so: estimated total = fd_time_per_param * N
            fd_time_est = fd_time_per_param * N
            speedup = fd_time_est / ad_time if ad_time > 0 else float('inf')
            print(f"  => FD full gradient estimate: {fd_time_est:.2f}s")
            print(f"  => Speedup (FD/AD): {speedup:.1f}×")

        entry = {
            'Nx': Nx, 'Ny': Ny, 'N': N,
            'ad_time_s': ad_time,
            'fd_time_est_s': fd_time_est,
            'speedup': speedup,
        }
        results.append(entry)

    # --- Print summary table ---
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    header = f"| {'Mesh':>7s} | {'N params':>8s} | {'AD grad (s)':>11s} | {'FD grad est (s)':>16s} | {'Speedup':>8s} |"
    sep = f"|{'-'*9}|{'-'*10}|{'-'*13}|{'-'*18}|{'-'*10}|"
    print(header)
    print(sep)
    for r in results:
        mesh_str = f"{r['Nx']}×{r['Ny']}"
        ad_str = f"{r['ad_time_s']:.4f}"
        fd_str = f"{r['fd_time_est_s']:.2f}" if r['fd_time_est_s'] is not None else "—"
        sp_str = f"{r['speedup']:.1f}×" if r['speedup'] is not None else "—"
        print(f"| {mesh_str:>7s} | {r['N']:>8d} | {ad_str:>11s} | {fd_str:>16s} | {sp_str:>8s} |")
    print()

    # --- Save results ---
    exp_dir = os.path.join(RESULTS_DIR, 'j4_scaling')
    os.makedirs(exp_dir, exist_ok=True)

    json_path = os.path.join(exp_dir, 'scaling_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    # --- Generate plot ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        Ns = [r['N'] for r in results]
        ad_times = [r['ad_time_s'] for r in results]
        ax.loglog(Ns, ad_times, 'o-', color='tab:blue', linewidth=2,
                  markersize=8, label='AD gradient')

        # FD: only entries with data
        fd_Ns = [r['N'] for r in results if r['fd_time_est_s'] is not None]
        fd_times = [r['fd_time_est_s'] for r in results if r['fd_time_est_s'] is not None]
        if fd_times:
            ax.loglog(fd_Ns, fd_times, 's--', color='tab:red', linewidth=2,
                      markersize=8, label='FD gradient (estimated)')

        ax.set_xlabel('Number of parameters N', fontsize=13)
        ax.set_ylabel('Time (s)', fontsize=13)
        ax.set_title('J4: AD vs FD Gradient Scaling', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(exp_dir, 'scaling_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")

    return results


if __name__ == "__main__":
    run_scaling_experiment()
