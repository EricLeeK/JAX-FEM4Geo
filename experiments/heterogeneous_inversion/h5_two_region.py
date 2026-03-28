"""
H5: First high-dimensional inversion experiment — two-region E field.

Setup:
- 2D rectangular domain 10×10, 20×20 QUAD4 mesh (400 elements)
- True E field: left half E=50000, right half E=90000
- Fixed k=50
- Top compression BC, full-field displacement observation
- No regularization
- L-BFGS-B optimizer

Expected: recover the two-region interface.
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
    optimize_adam,
    plot_E_field_comparison,
    compute_inversion_metrics,
    RESULTS_DIR,
)
from jax_fem.solver import ad_wrapper


def build_two_region_E_field(Nx, Ny):
    """Left half E=50000, right half E=90000. Cell ordering follows rectangle_mesh."""
    nc = Nx * Ny
    E_field = onp.full(nc, 90000.0)
    for ix in range(Nx):
        for iy in range(Ny):
            cell_idx = ix * Ny + iy
            if ix < Nx // 2:
                E_field[cell_idx] = 50000.0
    return np.array(E_field)


def run_two_region_experiment(Nx=20, Ny=20, displacement=-0.01,
                               optimizer='lbfgs', maxiter=100):
    """Run the two-region inversion experiment."""
    print("=" * 70)
    print("H5: Two-Region Heterogeneous E Field Inversion")
    print("=" * 70)

    Lx, Ly = 10., 10.
    k_fixed = 50.0

    # --- Build true E field ---
    E_true = build_two_region_E_field(Nx, Ny)
    nc = Nx * Ny
    print(f"\nProblem setup:")
    print(f"  Mesh: {Nx}×{Ny} = {nc} QUAD4 elements")
    print(f"  Domain: {Lx}×{Ly}")
    print(f"  Displacement: {displacement}")
    print(f"  k (fixed): {k_fixed}")
    print(f"  True E: left={float(E_true[0]):.0f}, right={float(E_true[-1]):.0f}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Parameters to invert: {nc}")

    # --- Create mesh and problem ---
    mesh, bc_info = create_2d_mesh_and_bc(displacement, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info, E_init=70000.0, k=k_fixed,
    )

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    # --- Generate synthetic observation ---
    print("\nGenerating synthetic observation...")
    t0 = time.time()
    obs_data, obs_indices, sol_true = generate_synthetic_observation(
        problem, fwd_pred, E_true,
    )
    print(f"  Forward solve: {time.time() - t0:.2f}s")
    print(f"  Observation: {obs_data.shape[0]} nodes × {obs_data.shape[1]} components")
    print(f"  Max displacement: {float(np.max(np.abs(sol_true))):.6e}")

    # --- Define loss ---
    def loss_fn(E_field):
        return displacement_loss(E_field, problem, fwd_pred, obs_data, obs_indices)

    # --- Quick sanity: loss at true params should be ~0 ---
    loss_at_true = float(loss_fn(E_true))
    print(f"\n  Loss at true E: {loss_at_true:.6e} (should be ~0)")

    # --- Initial guess: uniform E = 70000 ---
    E_init = np.full(nc, 70000.0)
    loss_at_init = float(loss_fn(E_init))
    print(f"  Loss at init E: {loss_at_init:.6e}")

    # --- Gradient check: verify gradient is non-zero ---
    print("\nGradient check...")
    t0 = time.time()
    grad = jax.grad(loss_fn)(E_init)
    print(f"  Grad computed in {time.time() - t0:.2f}s")
    print(f"  |grad|: {float(np.linalg.norm(grad)):.4e}")
    print(f"  grad range: [{float(np.min(grad)):.4e}, {float(np.max(grad)):.4e}]")

    # --- Run inversion ---
    print(f"\nRunning {optimizer} optimization (maxiter={maxiter})...")
    if optimizer == 'lbfgs':
        result = optimize_lbfgsb(loss_fn, E_init, E_min=10000.0, E_max=200000.0,
                                  maxiter=maxiter)
    else:
        result = optimize_adam(loss_fn, E_init, num_iters=maxiter, lr=500.0,
                               E_min=10000.0, E_max=200000.0)

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

    # --- E field statistics ---
    E_final_np = onp.array(E_final)
    left_mask = onp.array(E_true) < 70000
    right_mask = ~left_mask
    print(f"\n  Left region (true=50000):")
    print(f"    mean={E_final_np[left_mask].mean():.1f}, "
          f"std={E_final_np[left_mask].std():.1f}")
    print(f"  Right region (true=90000):")
    print(f"    mean={E_final_np[right_mask].mean():.1f}, "
          f"std={E_final_np[right_mask].std():.1f}")

    # --- Save results ---
    exp_dir = os.path.join(RESULTS_DIR, 'h5_two_region')
    os.makedirs(exp_dir, exist_ok=True)

    plot_E_field_comparison(
        E_true, E_final, Nx, Ny, Lx, Ly,
        title=f'H5: Two-Region Inversion ({Nx}×{Ny}, {optimizer})',
        save_path=os.path.join(exp_dir, f'E_field_{optimizer}.png'),
    )

    # Convergence plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(result['loss_history'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('H5: Convergence')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'convergence_{optimizer}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        pass

    # Save metrics to JSON
    save_data = {
        'Nx': Nx, 'Ny': Ny, 'num_cells': nc,
        'displacement': displacement, 'k_fixed': k_fixed,
        'optimizer': optimizer, 'maxiter': maxiter,
        'metrics': metrics,
        'time_s': result['time_s'],
        'nit': result['nit'],
        'final_loss': result['loss_history'][-1],
        'loss_at_true': loss_at_true,
        'left_region_mean': float(E_final_np[left_mask].mean()),
        'right_region_mean': float(E_final_np[right_mask].mean()),
    }
    with open(os.path.join(exp_dir, f'metrics_{optimizer}.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to {exp_dir}")
    print("=" * 70)

    return result, metrics


if __name__ == "__main__":
    run_two_region_experiment(Nx=20, Ny=20, displacement=-0.1,
                              optimizer='lbfgs', maxiter=100)
