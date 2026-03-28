#!/usr/bin/env python
"""
F2: Gradient Timing — Wall-clock time for AD vs FD gradient computation.

Measures:
  - t_fwd: single forward solve
  - t_ad: jax.value_and_grad (forward + adjoint)
  - t_fd_2p: 2-param central FD (4 forward solves)
  - t_fd_4p: 4-param central FD (8 forward solves, simulated)

Across mesh sizes: 2×2×2, 4×4×4, 6×6×6
"""

import jax
import jax.numpy as np
import numpy as onp
import os, sys, json, time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from common import (
    InversionDruckerPrager, create_mesh_and_bc,
    volume_avg_sigma_zz, benchmark, RESULTS_DIR,
)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper, solver

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(RESULTS_DIR, 'f2_gradient_timing')
os.makedirs(OUT_DIR, exist_ok=True)

DISP = -0.028  # plastic
PARAMS = np.array([70000.0, 50.0])
MESH_CONFIGS = [(2, 2, 2), (4, 4, 4), (6, 6, 6)]
N_REPEAT = 5


def main():
    print("=" * 70)
    print("F2: GRADIENT TIMING — AD vs FD")
    print("=" * 70)

    results = []

    for Nx, Ny, Nz in MESH_CONFIGS:
        n_cells = Nx * Ny * Nz
        label = f"{Nx}×{Ny}×{Nz} ({n_cells} cells)"
        print(f"\n--- Mesh: {label} ---")

        mesh, bc = create_mesh_and_bc(DISP, Nx=Nx, Ny=Ny, Nz=Nz)
        problem = InversionDruckerPrager(mesh, vec=3, dim=3, dirichlet_bc_info=bc)
        problem.set_params(PARAMS)
        fwd_pred = ad_wrapper(problem, solver_options=SOLVER_OPTIONS,
                              adjoint_solver_options=SOLVER_OPTIONS)

        def loss_fn(params):
            E, k = params[0], params[1]
            sol = fwd_pred(params)[0]
            return volume_avg_sigma_zz(
                problem.fe, sol, problem.sigmas_old, problem.epsilons_old, E, k,
            )

        vg_fn = jax.value_and_grad(loss_fn)

        # --- Forward only ---
        def run_fwd():
            problem.set_params(PARAMS)
            return solver(problem, solver_options=SOLVER_OPTIONS)

        _, t_fwd, t_fwd_std = benchmark(run_fwd, n_warmup=2, n_repeat=N_REPEAT)
        print(f"  t_fwd     = {t_fwd:.4f} ± {t_fwd_std:.4f} s")

        # --- AD gradient ---
        def run_ad():
            return vg_fn(PARAMS)

        _, t_ad, t_ad_std = benchmark(run_ad, n_warmup=2, n_repeat=N_REPEAT)
        print(f"  t_ad      = {t_ad:.4f} ± {t_ad_std:.4f} s")

        # --- FD 2 params (4 solves) ---
        def run_fd_2p():
            eps_E, eps_k = 100.0, 1.0
            lpe = float(loss_fn(PARAMS.at[0].set(PARAMS[0] + eps_E)))
            lme = float(loss_fn(PARAMS.at[0].set(PARAMS[0] - eps_E)))
            lpk = float(loss_fn(PARAMS.at[1].set(PARAMS[1] + eps_k)))
            lmk = float(loss_fn(PARAMS.at[1].set(PARAMS[1] - eps_k)))
            return onp.array([(lpe - lme) / (2 * eps_E), (lpk - lmk) / (2 * eps_k)])

        _, t_fd2, t_fd2_std = benchmark(run_fd_2p, n_warmup=2, n_repeat=N_REPEAT)
        print(f"  t_fd_2p   = {t_fd2:.4f} ± {t_fd2_std:.4f} s  (4 solves)")

        # --- FD 4 params: extrapolate as 2 × t_fd_2p ---
        t_fd4 = 2 * t_fd2
        print(f"  t_fd_4p   = {t_fd4:.4f} s  (estimated, 8 solves)")

        # Ratios
        speedup_2p = t_fd2 / t_ad if t_ad > 0 else float('inf')
        speedup_4p = t_fd4 / t_ad if t_ad > 0 else float('inf')
        ad_overhead = t_ad / t_fwd if t_fwd > 0 else float('inf')
        print(f"  AD overhead vs fwd: {ad_overhead:.2f}×")
        print(f"  Speedup AD vs FD(2p): {speedup_2p:.2f}×")
        print(f"  Speedup AD vs FD(4p): {speedup_4p:.2f}×")

        results.append({
            'mesh': f"{Nx}x{Ny}x{Nz}", 'n_cells': n_cells,
            't_fwd': t_fwd, 't_fwd_std': t_fwd_std,
            't_ad': t_ad, 't_ad_std': t_ad_std,
            't_fd_2p': t_fd2, 't_fd_2p_std': t_fd2_std,
            't_fd_4p': t_fd4,
            'speedup_2p': speedup_2p, 'speedup_4p': speedup_4p,
            'ad_overhead': ad_overhead,
        })

    # --- Plot 1: Time vs mesh size ---
    print("\n[Plot] Generating timing plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cells = [r['n_cells'] for r in results]
    t_fwds = [r['t_fwd'] for r in results]
    t_ads = [r['t_ad'] for r in results]
    t_fd2s = [r['t_fd_2p'] for r in results]
    t_fd4s = [r['t_fd_4p'] for r in results]

    ax = axes[0]
    ax.plot(cells, t_fwds, 'k--o', label='Forward only', linewidth=1.5)
    ax.plot(cells, t_ads, 'b-s', label='AD (fwd + adjoint)', linewidth=2)
    ax.plot(cells, t_fd2s, 'r-^', label='FD 2-param', linewidth=1.5)
    ax.plot(cells, t_fd4s, 'r--v', label='FD 4-param (est.)', linewidth=1.5)
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('Wall-clock time [s]')
    ax.set_title('Gradient Computation Time vs Mesh Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Time vs number of parameters (at largest mesh)
    ax = axes[1]
    r = results[-1]  # largest mesh
    n_params = [1, 2, 3, 4]
    t_fd_by_n = [r['t_fwd'] * 2 * n for n in n_params]  # FD: 2N solves
    t_ad_by_n = [r['t_ad']] * 4  # AD: constant
    ax.plot(n_params, t_ad_by_n, 'b-s', label='AD (constant)', linewidth=2, markersize=8)
    ax.plot(n_params, t_fd_by_n, 'r-^', label='FD (2N solves)', linewidth=2, markersize=8)
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('Gradient time [s]')
    ax.set_title(f'Scaling with Parameters ({r["mesh"]} mesh)')
    ax.set_xticks(n_params)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('F2: AD vs FD Gradient Computation Cost', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'timing.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")

    # Save
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 70)
    print("F2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
