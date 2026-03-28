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
- Top compression BC (displacement=-0.1), full-field displacement observation + 1% noise
- TV regularization (λ=0.01, E_ref=70000)
- L-BFGS-B optimizer, maxiter=100

Expected: recover the layered structure with sharp interfaces.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from common import (
    InversionHeterogeneousDP2D,
    create_2d_mesh_and_bc,
    generate_synthetic_observation,
    displacement_loss,
    optimize_lbfgsb,
    plot_E_field_comparison,
    compute_inversion_metrics,
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


def run_layered_experiment(Nx=30, Ny=30, displacement=-0.1,
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
    print(f"  Displacement: {displacement}")
    print(f"  k (fixed): {k_fixed}")
    print(f"  Noise level: {noise_level * 100:.0f}%")
    print(f"  Regularization: TV (λ={lam}, E_ref={E_ref})")
    print(f"  Parameters to invert: {nc}")
    for y_lo, y_hi, E_val in LAYERS:
        print(f"  Layer y∈[{y_lo}, {y_hi}]: E = {E_val:.0f}")

    # --- Create mesh and problem ---
    mesh, bc_info = create_2d_mesh_and_bc(displacement, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info, E_init=E_ref, k=k_fixed,
    )

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    # --- Generate synthetic observation with noise ---
    print("\nGenerating synthetic observation...")
    t0 = time.time()
    key = jax.random.PRNGKey(42)
    obs_data, obs_indices, sol_true = generate_synthetic_observation(
        problem, fwd_pred, E_true,
        noise_level=noise_level, key=key,
    )
    print(f"  Forward solve: {time.time() - t0:.2f}s")
    print(f"  Observation: {obs_data.shape[0]} nodes × {obs_data.shape[1]} components")
    print(f"  Max displacement: {float(np.max(np.abs(sol_true))):.6e}")

    # --- Build regularization ---
    neighbor_pairs = build_structured_neighbor_pairs(Nx, Ny)
    print(f"  Neighbor pairs: {len(neighbor_pairs)}")

    regularizer = lambda E: tv_regularizer(E, neighbor_pairs, E_ref=E_ref)

    # --- Define loss ---
    def loss_fn(E_field):
        return displacement_loss(E_field, problem, fwd_pred, obs_data, obs_indices,
                                 regularizer=regularizer, reg_weight=lam)

    # --- Sanity checks ---
    loss_at_true = float(loss_fn(E_true))
    print(f"\n  Loss at true E: {loss_at_true:.6e}")

    E_init = np.full(nc, E_ref)
    loss_at_init = float(loss_fn(E_init))
    print(f"  Loss at init E: {loss_at_init:.6e}")

    # --- Gradient check ---
    print("\nGradient check...")
    t0 = time.time()
    grad = jax.grad(loss_fn)(E_init)
    print(f"  Grad computed in {time.time() - t0:.2f}s")
    print(f"  |grad|: {float(np.linalg.norm(grad)):.4e}")
    print(f"  grad range: [{float(np.min(grad)):.4e}, {float(np.max(grad)):.4e}]")

    # --- Run inversion ---
    print(f"\nRunning L-BFGS-B optimization (maxiter={maxiter})...")
    result = optimize_lbfgsb(loss_fn, E_init, E_min=10000.0, E_max=200000.0,
                              maxiter=maxiter)

    E_final = result['E_final']

    # --- Metrics ---
    metrics = compute_inversion_metrics(E_true, E_final)
    print(f"\nInversion results:")
    print(f"  Final loss: {result['loss_history'][-1]:.6e}")
    print(f"  L2 relative error: {metrics['L2_relative_error']:.4e}")
    print(f"  Mean relative error: {metrics['mean_relative_error']:.4e}")
    print(f"  Max relative error: {metrics['max_relative_error']:.4e}")
    print(f"  Time: {result['time_s']:.1f}s")
    print(f"  Iterations: {result['nit']}")

    # --- Layer-by-layer summary table ---
    E_true_np = onp.array(E_true)
    E_final_np = onp.array(E_final)

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

    plot_E_field_comparison(
        E_true, E_final, Nx, Ny, Lx, Ly,
        title=f'J1: Layered Inversion ({Nx}×{Ny}, TV λ={lam})',
        save_path=os.path.join(exp_dir, 'E_field.png'),
    )

    # Convergence plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(result['loss_history'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('J1: Convergence')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'convergence.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        pass

    # Save metrics to JSON
    save_data = {
        'Nx': Nx, 'Ny': Ny, 'num_cells': nc,
        'displacement': displacement, 'k_fixed': k_fixed,
        'noise_level': noise_level,
        'reg_type': 'tv', 'lambda': lam, 'E_ref': E_ref,
        'maxiter': maxiter,
        'metrics': metrics,
        'layer_stats': layer_stats,
        'time_s': result['time_s'],
        'nit': result['nit'],
        'final_loss': result['loss_history'][-1],
        'loss_at_true': loss_at_true,
    }
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to {exp_dir}")
    print("=" * 70)

    return result, metrics


if __name__ == "__main__":
    run_layered_experiment(Nx=30, Ny=30, displacement=-0.1,
                           noise_level=0.01, lam=0.01, maxiter=100)
