"""
J1: Layered rock mass inversion — horizontally layered E field.

Setup:
- 2D rectangular domain 10×10, 30×30 QUAD4 mesh (900 elements)
- True E field: 4 horizontal layers (E varies with y only)
    Layer 1 (y ∈ [0, 2.5]):   E = 40000
    Layer 2 (y ∈ [2.5, 5.0]): E = 70000
    Layer 3 (y ∈ [5.0, 7.5]): E = 90000
    Layer 4 (y ∈ [7.5, 10]):  E = 60000
- Fixed k=50
- Top traction BC (traction=-50.0), full-field displacement observation + 1% noise
- TV regularization (λ=0.01, E_ref=70000)
- L-BFGS-B optimizer in log-E space, maxiter=100

Expected: recover the layered structure with sharp interfaces.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
    plot_E_field,
    save_results,
    RESULTS_DIR,
)
from jax_fem.solver import ad_wrapper
from src.regularization import build_structured_neighbor_pairs, tv_regularizer


# Layer definitions: (y_lower, y_upper, E_value)
LAYERS = [
    (0.0,  2.5, 40000.0),
    (2.5,  5.0, 70000.0),
    (5.0,  7.5, 90000.0),
    (7.5, 10.0, 60000.0),
]


def build_layered_E_field(Nx, Ny, Ly=10.0):
    """Build a horizontally layered E field (E varies with y only).

    Cell ordering follows rectangle_mesh: cell_idx = ix * Ny + iy.
    Cell centroid y = (iy + 0.5) * Ly / Ny.

    Returns
    -------
    E_field : jax array, shape (Nx*Ny,)
    """
    nc = Nx * Ny
    E_field = onp.zeros(nc)
    for ix in range(Nx):
        for iy in range(Ny):
            cell_idx = ix * Ny + iy
            y_centroid = (iy + 0.5) * Ly / Ny
            for y_lo, y_hi, E_val in LAYERS:
                if y_lo <= y_centroid < y_hi:
                    E_field[cell_idx] = E_val
                    break
            else:
                # Top boundary: assign to last layer
                E_field[cell_idx] = LAYERS[-1][2]
    return np.array(E_field)


def run_layered_experiment(Nx=30, Ny=30, traction=-50.0,
                           noise_level=0.01, lam=0.01,
                           maxiter=100):
    """Run the layered rock mass inversion experiment."""
    print("=" * 70)
    print("J1: Layered Rock Mass Heterogeneous E Field Inversion")
    print("=" * 70)

    Lx, Ly = 10., 10.
    k_fixed = 50.0
    E_ref = 70000.0

    # --- Build true E field ---
    E_true = build_layered_E_field(Nx, Ny, Ly)
    nc = Nx * Ny
    print(f"\nProblem setup:")
    print(f"  Mesh: {Nx}×{Ny} = {nc} QUAD4 elements")
    print(f"  Domain: {Lx}×{Ly}")
    print(f"  Traction: {traction}")
    print(f"  k (fixed): {k_fixed}")
    print(f"  Noise level: {noise_level * 100:.0f}%")
    print(f"  Regularization: TV (λ={lam}, E_ref={E_ref})")
    print(f"  Parameters to invert: {nc}")
    for y_lo, y_hi, E_val in LAYERS:
        print(f"  Layer y∈[{y_lo}, {y_hi}]: E = {E_val:.0f}")

    # --- Generate synthetic observation with noise ---
    print("\nGenerating synthetic observation...")
    t0 = time.time()
    obs = generate_synthetic_observation(
        E_true, k_fixed,
        traction=traction, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        noise_level=noise_level,
    )
    u_obs = obs['u_obs']
    obs_indices = obs['obs_indices']
    u_full = obs['u_full']
    mesh = obs['mesh']
    bc_info = obs['dirichlet_bc_info']
    loc_fns = obs['location_fns']
    print(f"  Forward solve: {time.time() - t0:.2f}s")
    print(f"  Observation: {u_obs.shape[0]} nodes × {u_obs.shape[1]} components")
    print(f"  Max displacement: {float(np.max(np.abs(u_full))):.6e}")

    # --- Create problem for inversion ---
    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        location_fns=loc_fns,
        E=70000., k=k_fixed,
        traction_value=traction,
    )

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    # --- Build regularization ---
    neighbor_pairs = build_structured_neighbor_pairs(Nx, Ny)
    print(f"  Neighbor pairs: {len(neighbor_pairs)}")

    regularizer = lambda log_E: tv_regularizer(np.exp(log_E), neighbor_pairs, E_ref=E_ref)

    # --- Define loss in log-E space ---
    loss_fn = make_heterogeneous_loss(
        fwd_pred, u_obs, obs_indices,
        regularizer=regularizer, reg_weight=lam,
    )

    # --- Sanity checks ---
    log_E_true = E_to_log(E_true)
    loss_at_true = float(loss_fn(log_E_true))
    print(f"\n  Loss at true E: {loss_at_true:.6e}")

    log_E_init = E_to_log(np.full(nc, E_ref))
    loss_at_init = float(loss_fn(log_E_init))
    print(f"  Loss at init E: {loss_at_init:.6e}")

    # --- Gradient check ---
    print("\nGradient check...")
    t0 = time.time()
    grad = jax.grad(loss_fn)(log_E_init)
    print(f"  Grad computed in {time.time() - t0:.2f}s")
    print(f"  |grad|: {float(np.linalg.norm(grad)):.4e}")
    print(f"  grad range: [{float(np.min(grad)):.4e}, {float(np.max(grad)):.4e}]")

    # --- Run inversion via scipy L-BFGS-B in log-E space ---
    log_E_min = E_to_log(np.array(10000.0))
    log_E_max = E_to_log(np.array(200000.0))
    bounds = [(float(log_E_min), float(log_E_max))] * nc

    loss_history = []

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    def objective(x):
        log_E = np.array(x)
        loss_val, grad_val = value_and_grad_fn(log_E)
        loss_history.append(float(loss_val))
        return float(loss_val), onp.array(grad_val, dtype=onp.float64)

    print(f"\nRunning L-BFGS-B optimization (maxiter={maxiter})...")
    t0 = time.time()
    result = minimize(
        objective,
        x0=onp.array(log_E_init, dtype=onp.float64),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12},
    )
    opt_time = time.time() - t0

    E_final = log_to_E(np.array(result.x))

    # --- Metrics (computed inline) ---
    E_true_np = onp.array(E_true)
    E_final_np = onp.array(E_final)

    l2_rel = float(onp.linalg.norm(E_final_np - E_true_np) / onp.linalg.norm(E_true_np))
    mean_rel = float(onp.mean(onp.abs(E_final_np - E_true_np) / E_true_np))
    max_rel = float(onp.max(onp.abs(E_final_np - E_true_np) / E_true_np))

    print(f"\nInversion results:")
    print(f"  Final loss: {loss_history[-1]:.6e}")
    print(f"  L2 relative error: {l2_rel:.4e}")
    print(f"  Mean relative error: {mean_rel:.4e}")
    print(f"  Max relative error: {max_rel:.4e}")
    print(f"  Time: {opt_time:.1f}s")
    print(f"  Iterations: {result.nit}")

    # --- Layer-by-layer summary table ---
    print(f"\n  {'Layer':<8} {'y range':<14} {'True E':<10} {'Mean inv E':<12} "
          f"{'Std inv E':<12} {'Rel err':<10}")
    print(f"  {'-' * 66}")
    layer_stats = []
    for i, (y_lo, y_hi, E_val) in enumerate(LAYERS):
        mask = onp.zeros(nc, dtype=bool)
        for ix in range(Nx):
            for iy in range(Ny):
                cell_idx = ix * Ny + iy
                y_centroid = (iy + 0.5) * Ly / Ny
                if y_lo <= y_centroid < y_hi:
                    mask[cell_idx] = True
        # Include top boundary cells in last layer
        if i == len(LAYERS) - 1:
            for ix in range(Nx):
                for iy in range(Ny):
                    cell_idx = ix * Ny + iy
                    y_centroid = (iy + 0.5) * Ly / Ny
                    if y_centroid >= y_hi:
                        mask[cell_idx] = True

        mean_inv = E_final_np[mask].mean()
        std_inv = E_final_np[mask].std()
        rel_err = abs(mean_inv - E_val) / E_val
        print(f"  {i+1:<8} [{y_lo:.1f}, {y_hi:.1f}]{'':<5} {E_val:<10.0f} "
              f"{mean_inv:<12.1f} {std_inv:<12.1f} {rel_err:<10.4f}")
        layer_stats.append({
            'layer': i + 1,
            'y_range': [y_lo, y_hi],
            'E_true': E_val,
            'mean_inverted': float(mean_inv),
            'std_inverted': float(std_inv),
            'relative_error': float(rel_err),
        })

    # --- Save results ---
    exp_dir = os.path.join(RESULTS_DIR, 'j1_layered')
    os.makedirs(exp_dir, exist_ok=True)

    # E field comparison plot
    vmin = min(E_true_np.min(), E_final_np.min())
    vmax = max(E_true_np.max(), E_final_np.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_E_field(E_true, Nx, Ny, ax=axes[0], title='True E field',
                 vmin=vmin, vmax=vmax)
    plot_E_field(E_final, Nx, Ny, ax=axes[1], title='Inverted E field',
                 vmin=vmin, vmax=vmax)
    fig.suptitle(f'J1: Layered Inversion ({Nx}×{Ny}, TV λ={lam})')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'E_field.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Convergence plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(loss_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('J1: Convergence')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'L2_relative_error': l2_rel,
        'mean_relative_error': mean_rel,
        'max_relative_error': max_rel,
    }
    save_results(exp_dir, {
        'Nx': Nx, 'Ny': Ny, 'num_cells': nc,
        'traction': traction, 'k_fixed': k_fixed,
        'noise_level': noise_level,
        'reg_type': 'tv', 'lambda': lam, 'E_ref': E_ref,
        'maxiter': maxiter,
        'metrics': metrics,
        'layer_stats': layer_stats,
        'time_s': opt_time,
        'nit': result.nit,
        'final_loss': loss_history[-1],
        'loss_at_true': loss_at_true,
    })

    print(f"\n  Results saved to {exp_dir}")
    print("=" * 70)

    return result, metrics


if __name__ == "__main__":
    run_layered_experiment(Nx=30, Ny=30, traction=-50.0,
                           noise_level=0.01, lam=1e-5, maxiter=100)
