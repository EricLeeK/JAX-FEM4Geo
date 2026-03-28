"""
J3: Gaussian random field inversion experiment.

Tests whether the framework can recover a spatially correlated random
heterogeneous E field from full-field displacement observations.

Setup:
- 2D rectangular domain 10×10, 30×30 QUAD4 mesh (900 elements)
- True E field: Gaussian random field (mean=70000, std=15000, corr_length=3.0)
  generated via Cholesky decomposition of squared-exponential covariance
  and clipped to [20000, 150000]
- Fixed k=50
- Top compression BC (displacement=-0.1), full-field + 1% noise observation
- Laplacian regularization (λ=0.1, E_ref=70000)
- L-BFGS-B optimizer, maxiter=150
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
from src.regularization import build_laplacian_matrix, laplacian_regularizer


def generate_gaussian_random_field(Nx, Ny, Lx, Ly, mean=70000, std=15000,
                                    corr_length=3.0, seed=42):
    """Generate a Gaussian random field via Cholesky decomposition.

    Uses a squared-exponential covariance kernel. The resulting field is
    clipped to [20000, 150000] for physical validity.

    Cell ordering: cell_idx = ix * Ny + iy.

    Returns
    -------
    E_field : jax array, shape (Nx*Ny,)
    """
    from scipy.spatial.distance import cdist

    rng = onp.random.RandomState(seed)
    nc = Nx * Ny

    # Cell centroids
    centroids = onp.zeros((nc, 2))
    for ix in range(Nx):
        for iy in range(Ny):
            centroids[ix * Ny + iy] = [(ix + 0.5) * Lx / Nx,
                                        (iy + 0.5) * Ly / Ny]

    # Squared exponential covariance
    D = cdist(centroids, centroids)
    C = std**2 * onp.exp(-0.5 * (D / corr_length)**2)
    C += 1e-6 * onp.eye(nc) * std**2  # numerical stability

    L = onp.linalg.cholesky(C)
    z = rng.randn(nc)
    E_field = mean + L @ z
    E_field = onp.clip(E_field, 20000, 150000)

    return np.array(E_field)


def run_random_field_experiment(Nx=30, Ny=30, displacement=-0.1,
                                 maxiter=150, reg_weight=0.1):
    """Run the Gaussian random field inversion experiment."""
    print("=" * 70)
    print("J3: Gaussian Random Field Inversion")
    print("=" * 70)

    Lx, Ly = 10., 10.
    k_fixed = 50.0
    nc = Nx * Ny
    noise_level = 0.01
    E_ref = 70000.0

    # --- Generate true E field ---
    E_true = generate_gaussian_random_field(Nx, Ny, Lx, Ly)
    E_true_np = onp.array(E_true)
    print(f"\nProblem setup:")
    print(f"  Mesh: {Nx}×{Ny} = {nc} QUAD4 elements")
    print(f"  Domain: {Lx}×{Ly}")
    print(f"  Displacement: {displacement}")
    print(f"  k (fixed): {k_fixed}")
    print(f"  Noise level: {noise_level * 100:.0f}%")
    print(f"  Regularization: Laplacian (λ={reg_weight}, E_ref={E_ref})")
    print(f"  Optimizer: L-BFGS-B (maxiter={maxiter})")
    print(f"  Parameters to invert: {nc}")
    print(f"\nTrue E field statistics:")
    print(f"  mean={E_true_np.mean():.1f}, std={E_true_np.std():.1f}")
    print(f"  min={E_true_np.min():.1f}, max={E_true_np.max():.1f}")

    # --- Create mesh and problem ---
    mesh, bc_info = create_2d_mesh_and_bc(displacement, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info, E_init=E_ref, k=k_fixed,
    )

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    # --- Generate synthetic observation with 1% noise ---
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
    L_mat = np.array(build_laplacian_matrix(Nx, Ny))
    regularizer = lambda E: laplacian_regularizer(E, L_mat, E_ref=E_ref)

    # --- Define loss ---
    def loss_fn(E_field):
        return displacement_loss(E_field, problem, fwd_pred, obs_data, obs_indices,
                                 regularizer=regularizer, reg_weight=reg_weight)

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
    E_final_np = onp.array(E_final)

    # --- Metrics ---
    metrics = compute_inversion_metrics(E_true, E_final)

    # Correlation coefficient
    corr_coeff = float(onp.corrcoef(E_true_np, E_final_np)[0, 1])

    print(f"\nInversion results:")
    print(f"  Final loss: {result['loss_history'][-1]:.6e}")
    print(f"  L2 relative error: {metrics['L2_relative_error']:.4e}")
    print(f"  Mean relative error: {metrics['mean_relative_error']:.4e}")
    print(f"  Max relative error: {metrics['max_relative_error']:.4e}")
    print(f"  Correlation coefficient: {corr_coeff:.6f}")
    print(f"  Time: {result['time_s']:.1f}s")
    print(f"  Iterations: {result['nit']}")

    # --- Statistics comparison ---
    print(f"\n  Field statistics comparison:")
    print(f"    True  — mean={E_true_np.mean():.1f}, std={E_true_np.std():.1f}, "
          f"min={E_true_np.min():.1f}, max={E_true_np.max():.1f}")
    print(f"    Inv   — mean={E_final_np.mean():.1f}, std={E_final_np.std():.1f}, "
          f"min={E_final_np.min():.1f}, max={E_final_np.max():.1f}")

    # --- Save results ---
    exp_dir = os.path.join(RESULTS_DIR, 'j3_random_field')
    os.makedirs(exp_dir, exist_ok=True)

    plot_E_field_comparison(
        E_true, E_final, Nx, Ny, Lx, Ly,
        title=f'J3: Random Field Inversion ({Nx}×{Ny}, corr_len=3.0)',
        save_path=os.path.join(exp_dir, 'E_field_comparison.png'),
    )

    # Convergence plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(result['loss_history'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('J3: Convergence')
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
        'reg_type': 'laplacian', 'reg_weight': reg_weight,
        'E_ref': E_ref,
        'optimizer': 'lbfgs', 'maxiter': maxiter,
        'metrics': metrics,
        'correlation_coefficient': corr_coeff,
        'time_s': result['time_s'],
        'nit': result['nit'],
        'final_loss': result['loss_history'][-1],
        'loss_at_true': loss_at_true,
        'true_field_stats': {
            'mean': float(E_true_np.mean()),
            'std': float(E_true_np.std()),
            'min': float(E_true_np.min()),
            'max': float(E_true_np.max()),
        },
        'inverted_field_stats': {
            'mean': float(E_final_np.mean()),
            'std': float(E_final_np.std()),
            'min': float(E_final_np.min()),
            'max': float(E_final_np.max()),
        },
    }
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to {exp_dir}")
    print("=" * 70)

    return result, metrics


if __name__ == "__main__":
    run_random_field_experiment(Nx=30, Ny=30, displacement=-0.1,
                                maxiter=150, reg_weight=0.1)
