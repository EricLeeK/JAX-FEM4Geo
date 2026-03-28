#!/usr/bin/env python
"""
D2: Sensitivity Analysis — E-k Loss Landscape

Workflow:
  1. Set up two observations (elastic + plastic) at true params.
  2. Sweep E-k parameter space on a grid around true values.
  3. At each grid point, evaluate loss = (σ_pred - σ_obs)².
  4. Plot contour map + gradient field.
  5. Analyse: convexity, condition number, parameter coupling.

Key outputs:
  - Loss contour plot (E vs k)
  - Gradient field overlay
  - Hessian eigenvalues at the minimum → condition number
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
from matplotlib import cm

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from common import (
    InversionDruckerPrager, create_mesh_and_bc, update_bc,
    volume_avg_sigma_zz, generate_observation,
    RESULTS_DIR, stress_return_dp,
)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper, solver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
E_TRUE = 70000.0
K_TRUE = 50.0

DISP_ELASTIC = -0.02
DISP_PLASTIC = -0.03

# Grid: 15 x 15 points
E_RANGE = (40000., 100000.)
K_RANGE = (20., 80.)
N_E = 15
N_K = 15

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

OUT_DIR = os.path.join(RESULTS_DIR, 'd2_sensitivity')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("D2: SENSITIVITY ANALYSIS — E-k LOSS LANDSCAPE")
    print("=" * 70)

    # --- Step 1: Generate observations ---
    print("\n[1] Generating synthetic observations...")
    mesh_e, bc_e = create_mesh_and_bc(DISP_ELASTIC)
    prob_obs_e = InversionDruckerPrager(mesh_e, vec=3, dim=3, dirichlet_bc_info=bc_e)
    obs_elastic, _ = generate_observation(prob_obs_e, bc_e, DISP_ELASTIC, E_TRUE, K_TRUE, SOLVER_OPTIONS)

    mesh_p, bc_p = create_mesh_and_bc(DISP_PLASTIC)
    prob_obs_p = InversionDruckerPrager(mesh_p, vec=3, dim=3, dirichlet_bc_info=bc_p)
    obs_plastic, _ = generate_observation(prob_obs_p, bc_p, DISP_PLASTIC, E_TRUE, K_TRUE, SOLVER_OPTIONS)

    print(f"  σ_zz_obs elastic: {obs_elastic:.6f} MPa")
    print(f"  σ_zz_obs plastic: {obs_plastic:.6f} MPa")

    # --- Step 2: Create problems for evaluation ---
    print("\n[2] Setting up FEM problems for grid evaluation...")
    mesh_e2, bc_e2 = create_mesh_and_bc(DISP_ELASTIC)
    prob_e = InversionDruckerPrager(mesh_e2, vec=3, dim=3, dirichlet_bc_info=bc_e2)

    mesh_p2, bc_p2 = create_mesh_and_bc(DISP_PLASTIC)
    prob_p = InversionDruckerPrager(mesh_p2, vec=3, dim=3, dirichlet_bc_info=bc_p2)

    def eval_loss(E_val, k_val):
        """Evaluate loss at a single (E, k) point using direct solver."""
        params = np.array([E_val, k_val])

        # Elastic
        prob_e.set_params(params)
        sol_e = solver(prob_e, solver_options=SOLVER_OPTIONS)[0]
        pred_e = volume_avg_sigma_zz(
            prob_e.fe, sol_e, prob_e.sigmas_old, prob_e.epsilons_old, E_val, k_val,
        )

        # Plastic
        prob_p.set_params(params)
        sol_p = solver(prob_p, solver_options=SOLVER_OPTIONS)[0]
        pred_p = volume_avg_sigma_zz(
            prob_p.fe, sol_p, prob_p.sigmas_old, prob_p.epsilons_old, E_val, k_val,
        )

        loss = float((pred_e - obs_elastic) ** 2 + (pred_p - obs_plastic) ** 2)
        return loss, float(pred_e), float(pred_p)

    # --- Step 3: Sweep grid ---
    E_vals = onp.linspace(E_RANGE[0], E_RANGE[1], N_E)
    k_vals = onp.linspace(K_RANGE[0], K_RANGE[1], N_K)

    print(f"\n[3] Sweeping {N_E} x {N_K} = {N_E * N_K} grid points...")
    t0 = time.time()

    loss_grid = onp.zeros((N_K, N_E))
    pred_e_grid = onp.zeros((N_K, N_E))
    pred_p_grid = onp.zeros((N_K, N_E))

    for j, k_val in enumerate(k_vals):
        for i, E_val in enumerate(E_vals):
            try:
                loss, pe, pp = eval_loss(E_val, k_val)
                loss_grid[j, i] = loss
                pred_e_grid[j, i] = pe
                pred_p_grid[j, i] = pp
            except Exception as e:
                print(f"  FAILED at E={E_val:.0f}, k={k_val:.1f}: {e}")
                loss_grid[j, i] = onp.nan

        pct = (j + 1) / N_K * 100
        if (j + 1) % 3 == 0 or j == 0:
            print(f"  Row {j+1}/{N_K} (k={k_val:.1f}) done  [{pct:.0f}%]")

    t_sweep = time.time() - t0
    print(f"  Grid sweep completed in {t_sweep:.1f}s")

    # --- Step 4: Hessian at minimum (finite-difference approximation) ---
    print("\n[4] Computing Hessian at true params via AD...")

    fwd_e = ad_wrapper(prob_e, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)
    fwd_p = ad_wrapper(prob_p, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)

    def loss_fn(params):
        E, k = params[0], params[1]
        sol_e = fwd_e(params)[0]
        pred_e = volume_avg_sigma_zz(
            prob_e.fe, sol_e, prob_e.sigmas_old, prob_e.epsilons_old, E, k,
        )
        sol_p = fwd_p(params)[0]
        pred_p = volume_avg_sigma_zz(
            prob_p.fe, sol_p, prob_p.sigmas_old, prob_p.epsilons_old, E, k,
        )
        return (pred_e - obs_elastic) ** 2 + (pred_p - obs_plastic) ** 2

    grad_fn = jax.grad(loss_fn)

    # FD Hessian at true params
    params_true = np.array([E_TRUE, K_TRUE])
    eps_hess = [50.0, 0.5]  # step sizes for E and k
    H = onp.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            p_pp = onp.array([E_TRUE, K_TRUE])
            p_pm = onp.array([E_TRUE, K_TRUE])
            p_mp = onp.array([E_TRUE, K_TRUE])
            p_mm = onp.array([E_TRUE, K_TRUE])
            p_pp[i] += eps_hess[i]; p_pp[j] += eps_hess[j]
            p_pm[i] += eps_hess[i]; p_pm[j] -= eps_hess[j]
            p_mp[i] -= eps_hess[i]; p_mp[j] += eps_hess[j]
            p_mm[i] -= eps_hess[i]; p_mm[j] -= eps_hess[j]

            L_pp = float(loss_fn(np.array(p_pp)))
            L_pm = float(loss_fn(np.array(p_pm)))
            L_mp = float(loss_fn(np.array(p_mp)))
            L_mm = float(loss_fn(np.array(p_mm)))

            H[i, j] = (L_pp - L_pm - L_mp + L_mm) / (4 * eps_hess[i] * eps_hess[j])

    eigvals = onp.linalg.eigvalsh(H)
    cond = max(abs(eigvals)) / max(min(abs(eigvals)), 1e-30)

    print(f"  Hessian at true params:")
    print(f"    H[0,0] (∂²L/∂E²)  = {H[0,0]:.6e}")
    print(f"    H[1,1] (∂²L/∂k²)  = {H[1,1]:.6e}")
    print(f"    H[0,1] (∂²L/∂E∂k) = {H[0,1]:.6e}")
    print(f"  Eigenvalues: {eigvals}")
    print(f"  Condition number: {cond:.2f}")

    is_convex = onp.all(eigvals > 0)
    print(f"  Locally convex: {'YES' if is_convex else 'NO'}")

    # --- Step 5: Gradient field (subsampled) ---
    print("\n[5] Computing gradient field on coarse grid...")
    N_grad = 8
    E_grad = onp.linspace(E_RANGE[0], E_RANGE[1], N_grad)
    k_grad = onp.linspace(K_RANGE[0], K_RANGE[1], N_grad)
    grad_E_grid = onp.zeros((N_grad, N_grad))
    grad_k_grid = onp.zeros((N_grad, N_grad))

    for j, kv in enumerate(k_grad):
        for i, Ev in enumerate(E_grad):
            try:
                g = grad_fn(np.array([Ev, kv]))
                grad_E_grid[j, i] = float(g[0])
                grad_k_grid[j, i] = float(g[1])
            except Exception:
                grad_E_grid[j, i] = onp.nan
                grad_k_grid[j, i] = onp.nan

    # Normalize for visualization
    mag = onp.sqrt(grad_E_grid**2 + grad_k_grid**2)
    mag = onp.where(mag < 1e-30, 1., mag)
    grad_E_norm = grad_E_grid / mag
    grad_k_norm = grad_k_grid / mag

    # --- Step 6: Plotting ---
    print("\n[6] Generating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    E_mesh, k_mesh = onp.meshgrid(E_vals, k_vals)

    # 6a: Loss contour
    ax = axes[0]
    # Use log scale for better visualization
    loss_log = onp.log10(onp.clip(loss_grid, 1e-20, None))
    levels = onp.linspace(onp.nanmin(loss_log), onp.nanmax(loss_log), 25)
    cs = ax.contourf(E_mesh / 1000, k_mesh, loss_log, levels=levels, cmap='viridis')
    ax.contour(E_mesh / 1000, k_mesh, loss_log, levels=levels[::3], colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax, label='log₁₀(Loss)')
    ax.plot(E_TRUE / 1000, K_TRUE, 'r*', markersize=15, label='True')
    ax.set_xlabel('E [GPa]')
    ax.set_ylabel('k [MPa]')
    ax.set_title('Loss Landscape (log scale)')
    ax.legend()

    # 6b: Contour + gradient arrows
    ax = axes[1]
    cs2 = ax.contour(E_mesh / 1000, k_mesh, loss_log, levels=15, cmap='viridis')
    E_grad_mesh, k_grad_mesh = onp.meshgrid(E_grad, k_grad)
    ax.quiver(E_grad_mesh / 1000, k_grad_mesh,
              -grad_E_norm * 2, -grad_k_norm * 2,  # negative gradient = descent direction
              color='red', alpha=0.7, scale=30)
    ax.plot(E_TRUE / 1000, K_TRUE, 'r*', markersize=15, label='True')
    ax.set_xlabel('E [GPa]')
    ax.set_ylabel('k [MPa]')
    ax.set_title('Gradient Field (descent direction)')
    ax.legend()

    # 6c: Cross-sections
    ax = axes[2]
    # E cross-section at k=k_true
    k_idx = onp.argmin(onp.abs(k_vals - K_TRUE))
    ax.semilogy(E_vals / 1000, onp.clip(loss_grid[k_idx, :], 1e-20, None),
                'b-', linewidth=2, label=f'k = {k_vals[k_idx]:.0f} (fixed)')
    ax.axvline(E_TRUE / 1000, color='b', linestyle='--', alpha=0.5)

    # k cross-section at E=E_true
    E_idx = onp.argmin(onp.abs(E_vals - E_TRUE))
    ax_twin = ax.twiny()
    ax_twin.semilogy(k_vals, onp.clip(loss_grid[:, E_idx], 1e-20, None),
                     'r-', linewidth=2, label=f'E = {E_vals[E_idx]:.0f} (fixed)')
    ax_twin.axvline(K_TRUE, color='r', linestyle='--', alpha=0.5)
    ax_twin.set_xlabel('k [MPa]', color='r')
    ax_twin.tick_params(axis='x', labelcolor='r')

    ax.set_xlabel('E [GPa]', color='b')
    ax.set_ylabel('Loss')
    ax.set_title('Cross-Sections Through Minimum')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'loss_landscape.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved: {plot_path}")

    # --- Step 7: Save results ---
    results = {
        'E_true': E_TRUE, 'k_true': K_TRUE,
        'E_range': list(E_RANGE), 'k_range': list(K_RANGE),
        'grid_size': [N_E, N_K],
        'hessian': H.tolist(),
        'eigenvalues': eigvals.tolist(),
        'condition_number': float(cond),
        'locally_convex': bool(is_convex),
        'loss_min': float(onp.nanmin(loss_grid)),
        'loss_max': float(onp.nanmax(loss_grid)),
        'sweep_time_s': t_sweep,
    }
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {json_path}")

    # Save raw grid data
    onp.savez(os.path.join(OUT_DIR, 'grid_data.npz'),
              E_vals=E_vals, k_vals=k_vals,
              loss_grid=loss_grid,
              pred_e_grid=pred_e_grid, pred_p_grid=pred_p_grid)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Grid: {N_E} x {N_K} = {N_E * N_K} evaluations in {t_sweep:.1f}s")
    print(f"  Loss range: {onp.nanmin(loss_grid):.2e} — {onp.nanmax(loss_grid):.2e}")
    print(f"  Hessian eigenvalues: [{eigvals[0]:.4e}, {eigvals[1]:.4e}]")
    print(f"  Condition number: {cond:.2f}")
    print(f"  Locally convex at true params: {'YES' if is_convex else 'NO'}")
    if cond > 1000:
        print(f"  WARNING: High condition number suggests E-k coupling or poor scaling")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
