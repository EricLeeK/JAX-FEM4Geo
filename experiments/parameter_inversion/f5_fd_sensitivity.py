#!/usr/bin/env python
"""
F5: FD Step-Size Sensitivity.

Sweeps eps_E × eps_k grid, runs FD-based inversion for each combination.
Produces heatmaps showing how FD inversion quality depends on step-size
choice. AD (no hyperparameter) is the baseline.
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
    volume_avg_sigma_zz, fd_gradient, generate_observation,
    adam_optimize, RESULTS_DIR,
)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(RESULTS_DIR, 'f5_fd_sensitivity')
os.makedirs(OUT_DIR, exist_ok=True)

E_TRUE, K_TRUE = 70000.0, 50.0
DISP_PLASTIC = -0.028

# Step sizes to sweep
EPS_E_LIST = [1.0, 10.0, 100.0, 1000.0, 5000.0]
EPS_K_LIST = [0.01, 0.1, 1.0, 5.0, 10.0]

N_ITERS = 30  # Adam iterations per trial


def main():
    print("=" * 70)
    print("F5: FD STEP-SIZE SENSITIVITY")
    print("=" * 70)

    # Setup
    mesh_obs, bc_obs = create_mesh_and_bc(DISP_PLASTIC)
    prob_obs = InversionDruckerPrager(mesh_obs, vec=3, dim=3, dirichlet_bc_info=bc_obs)
    obs_plastic, _ = generate_observation(prob_obs, bc_obs, DISP_PLASTIC, E_TRUE, K_TRUE, SOLVER_OPTIONS)
    print(f"  Observation: σ_zz = {obs_plastic:.4f} MPa")

    # We fix E at true value and only invert k (1D) to keep the sweep fast.
    # This isolates the FD step-size effect on the plastic parameter.
    mesh, bc = create_mesh_and_bc(DISP_PLASTIC)
    problem = InversionDruckerPrager(mesh, vec=3, dim=3, dirichlet_bc_info=bc)
    fwd = ad_wrapper(problem, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)

    E_fix = E_TRUE

    def loss_fn(params):
        """Loss for (E, k) jointly — E fixed at true, k varies."""
        E, k = params[0], params[1]
        sol = fwd(params)[0]
        pred = volume_avg_sigma_zz(
            problem.fe, sol, problem.sigmas_old, problem.epsilons_old, E, k,
        )
        return (pred - obs_plastic) ** 2

    # Warm up
    _ = loss_fn(np.array([E_fix, 45.0]))

    # AD baseline
    print("\n  Running AD baseline...")
    vg_ad = jax.value_and_grad(loss_fn)
    _, hist_ad = adam_optimize(
        vg_ad, onp.array([E_fix, 45.0]), num_iters=N_ITERS, lr=5.0,
        bounds=[(E_fix - 1, E_fix + 1), (35., 80.)],
    )
    k_ad = hist_ad['params'][-1][1]
    err_ad = abs(k_ad - K_TRUE) / K_TRUE
    print(f"  AD: k = {k_ad:.4f}, err = {err_ad:.6%}")

    # FD sweep
    print(f"\n  Sweeping {len(EPS_E_LIST)} × {len(EPS_K_LIST)} = {len(EPS_E_LIST) * len(EPS_K_LIST)} combos...")
    err_grid = onp.full((len(EPS_K_LIST), len(EPS_E_LIST)), onp.nan)
    iters_grid = onp.full((len(EPS_K_LIST), len(EPS_E_LIST)), onp.nan)

    for j, eps_k in enumerate(EPS_K_LIST):
        for i, eps_E in enumerate(EPS_E_LIST):
            try:
                def vg_fd(params):
                    l = loss_fn(params)
                    g = fd_gradient(loss_fn, params, [eps_E, eps_k])
                    return l, np.array(g)

                _, hist = adam_optimize(
                    vg_fd, onp.array([E_fix, 45.0]), num_iters=N_ITERS, lr=5.0,
                    bounds=[(E_fix - 1, E_fix + 1), (35., 80.)],
                )
                k_final = hist['params'][-1][1]
                err = abs(k_final - K_TRUE) / K_TRUE
                err_grid[j, i] = err

                # Find when err < 1%
                for step, p in enumerate(hist['params']):
                    if abs(p[1] - K_TRUE) / K_TRUE < 0.01:
                        iters_grid[j, i] = step + 1
                        break
                else:
                    iters_grid[j, i] = N_ITERS  # didn't converge

            except Exception as e:
                print(f"    FAILED eps_E={eps_E}, eps_k={eps_k}: {e}")
                err_grid[j, i] = onp.nan
                iters_grid[j, i] = onp.nan

        print(f"  Row {j+1}/{len(EPS_K_LIST)} (eps_k={eps_k}) done")

    # --- Plotting ---
    print("\n[Plot] Generating heatmaps...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    eps_E_labels = [f"{e:.0e}" for e in EPS_E_LIST]
    eps_k_labels = [f"{e:.0e}" for e in EPS_K_LIST]

    # Error heatmap
    ax = axes[0]
    im = ax.imshow(onp.log10(onp.clip(err_grid, 1e-10, None)),
                   aspect='auto', cmap='RdYlGn_r', vmin=-6, vmax=0)
    ax.set_xticks(range(len(EPS_E_LIST)))
    ax.set_xticklabels(eps_E_labels, fontsize=8)
    ax.set_yticks(range(len(EPS_K_LIST)))
    ax.set_yticklabels(eps_k_labels, fontsize=8)
    ax.set_xlabel('eps_E')
    ax.set_ylabel('eps_k')
    ax.set_title('log₁₀(k relative error)')
    plt.colorbar(im, ax=ax)

    # Annotate cells
    for j in range(len(EPS_K_LIST)):
        for i in range(len(EPS_E_LIST)):
            v = err_grid[j, i]
            if not onp.isnan(v):
                ax.text(i, j, f"{v:.1e}", ha='center', va='center', fontsize=7,
                        color='white' if v > 0.01 else 'black')

    # AD baseline annotation
    ax.set_title(f'FD k Error  (AD baseline: {err_ad:.2e})')

    # Iterations heatmap
    ax = axes[1]
    im2 = ax.imshow(iters_grid, aspect='auto', cmap='YlOrRd', vmin=0, vmax=N_ITERS)
    ax.set_xticks(range(len(EPS_E_LIST)))
    ax.set_xticklabels(eps_E_labels, fontsize=8)
    ax.set_yticks(range(len(EPS_K_LIST)))
    ax.set_yticklabels(eps_k_labels, fontsize=8)
    ax.set_xlabel('eps_E')
    ax.set_ylabel('eps_k')
    ax.set_title('Iterations to 1% (max=not converged)')
    plt.colorbar(im2, ax=ax)

    for j in range(len(EPS_K_LIST)):
        for i in range(len(EPS_E_LIST)):
            v = iters_grid[j, i]
            if not onp.isnan(v):
                ax.text(i, j, f"{int(v)}", ha='center', va='center', fontsize=8)

    plt.suptitle('F5: FD Inversion Sensitivity to Step Size', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'heatmap.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")

    # Save
    results = {
        'eps_E_list': EPS_E_LIST, 'eps_k_list': EPS_K_LIST,
        'err_grid': err_grid.tolist(), 'iters_grid': iters_grid.tolist(),
        'ad_baseline': {'k_final': float(k_ad), 'err': float(err_ad)},
        'best_fd_err': float(onp.nanmin(err_grid)),
        'worst_fd_err': float(onp.nanmax(err_grid)),
    }
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 70)
    print("F5 SUMMARY")
    print("=" * 70)
    print(f"  AD baseline k error: {err_ad:.6%} (no tuning needed)")
    print(f"  FD best k error:     {onp.nanmin(err_grid):.6%}")
    print(f"  FD worst k error:    {onp.nanmax(err_grid):.6%}")
    print(f"  FD range: {onp.nanmin(err_grid):.2e} — {onp.nanmax(err_grid):.2e}")
    n_good = onp.sum(err_grid < 0.01)
    n_total = err_grid.size - onp.sum(onp.isnan(err_grid))
    print(f"  FD combos with <1% error: {n_good}/{n_total}")
    print("=" * 70)


if __name__ == "__main__":
    main()
