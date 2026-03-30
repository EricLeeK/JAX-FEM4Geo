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
- Traction BC (traction=-50.0 MPa on top), full-field + 1% noise observation
- Laplacian regularization (λ=0.1, E_ref=70000)
- L-BFGS-B optimizer in log-E space, maxiter=150
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
    create_2d_mesh_traction_bc,
    generate_synthetic_observation,
    make_heterogeneous_loss,
    log_to_E, E_to_log,
    plot_E_field,
    save_results,
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


def run_random_field_experiment(Nx=30, Ny=30, traction=-50.0,
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

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

    # --- Generate true E field ---
    E_true = generate_gaussian_random_field(Nx, Ny, Lx, Ly)
    E_true_np = onp.array(E_true)
    print(f"\nProblem setup:")
    print(f"  Mesh: {Nx}×{Ny} = {nc} QUAD4 elements")
    print(f"  Domain: {Lx}×{Ly}")
    print(f"  Traction: {traction} MPa")
    print(f"  k (fixed): {k_fixed}")
    print(f"  Noise level: {noise_level * 100:.0f}%")
    print(f"  Regularization: Laplacian (λ={reg_weight}, E_ref={E_ref})")
    print(f"  Optimizer: L-BFGS-B in log-E space (maxiter={maxiter})")
    print(f"  Parameters to invert: {nc}")
    print(f"\nTrue E field statistics:")
    print(f"  mean={E_true_np.mean():.1f}, std={E_true_np.std():.1f}")
    print(f"  min={E_true_np.min():.1f}, max={E_true_np.max():.1f}")

    # --- Generate synthetic observation with 1% noise ---
    print("\nGenerating synthetic observation...")
    t0 = time.time()
    obs_data = generate_synthetic_observation(
        E_true, k_fixed,
        traction=traction, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        noise_level=noise_level,
        solver_options=solver_options,
    )
    u_obs = obs_data['u_obs']
    obs_indices = obs_data['obs_indices']
    u_full_true = obs_data['u_full']
    print(f"  Forward solve: {time.time() - t0:.2f}s")
    print(f"  Observation: {u_obs.shape[0]} nodes × {u_obs.shape[1]} components")
    print(f"  Max displacement: {float(np.max(np.abs(u_full_true))):.6e}")

    # --- Set up inversion problem ---
    mesh, bc_info, loc_fns, _ = create_2d_mesh_traction_bc(
        traction, Lx, Ly, Nx, Ny)

    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        location_fns=loc_fns,
        E=E_ref, k=k_fixed,
        traction_value=traction,
    )

    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    # --- Build regularization (Laplacian on log_E) ---
    L_mat = np.array(build_laplacian_matrix(Nx, Ny))
    regularizer = lambda log_E: laplacian_regularizer(np.exp(log_E), L_mat, E_ref=E_ref)

    # --- Build loss function (in log-E space) ---
    loss_fn = make_heterogeneous_loss(
        fwd_pred, u_obs, obs_indices,
        regularizer=regularizer, reg_weight=reg_weight,
    )

    value_and_grad_fn = jax.value_and_grad(loss_fn)

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

    # --- Run L-BFGS-B optimization in log-E space ---
    print(f"\nRunning L-BFGS-B optimization (maxiter={maxiter})...")
    from scipy.optimize import minimize as scipy_minimize

    history = {'loss': [], 'grad_norm': [], 'wallclock': []}
    t_opt_start = time.time()

    log_E_lo = float(onp.log(1000.))
    log_E_hi = float(onp.log(500000.))
    bounds = [(log_E_lo, log_E_hi)] * nc

    def objective(x):
        log_E = np.array(x)
        loss, grad = value_and_grad_fn(log_E)
        loss_f = float(loss)
        grad_np = onp.array(grad, dtype=onp.float64)
        grad_norm = float(onp.linalg.norm(grad_np))

        history['loss'].append(loss_f)
        history['grad_norm'].append(grad_norm)
        history['wallclock'].append(time.time() - t_opt_start)

        step = len(history['loss'])
        if step <= 5 or step % 10 == 0:
            E_cur = onp.exp(x)
            print(f"    Eval {step:4d}: loss={loss_f:.6e}  "
                  f"||grad||={grad_norm:.3e}  "
                  f"E range=[{E_cur.min():.0f}, {E_cur.max():.0f}]")

        return loss_f, grad_np

    result = scipy_minimize(
        objective,
        x0=onp.array(log_E_init, dtype=onp.float64),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'maxfun': 500, 'ftol': 1e-20, 'gtol': 1e-10},
    )

    t_opt = time.time() - t_opt_start
    log_E_final = result.x
    E_final = onp.exp(log_E_final)
    E_final_np = E_final

    # --- Metrics (computed inline) ---
    E_error = onp.abs(E_final_np - E_true_np)
    E_rel_error = E_error / E_true_np
    l2_rel_error = float(onp.sqrt(onp.mean(E_rel_error ** 2)))
    mean_rel_error = float(onp.mean(E_rel_error))
    max_rel_error = float(onp.max(E_rel_error))

    # Correlation coefficient
    corr_coeff = float(onp.corrcoef(E_true_np, E_final_np)[0, 1])

    print(f"\nInversion results:")
    print(f"  Final loss: {history['loss'][-1]:.6e}")
    print(f"  L2 relative error: {l2_rel_error:.4e}")
    print(f"  Mean relative error: {mean_rel_error:.4e}")
    print(f"  Max relative error: {max_rel_error:.4e}")
    print(f"  Correlation coefficient: {corr_coeff:.6f}")
    print(f"  Time: {t_opt:.1f}s")
    print(f"  Iterations: {result.nit}")

    # --- Statistics comparison ---
    print(f"\n  Field statistics comparison:")
    print(f"    True  — mean={E_true_np.mean():.1f}, std={E_true_np.std():.1f}, "
          f"min={E_true_np.min():.1f}, max={E_true_np.max():.1f}")
    print(f"    Inv   — mean={E_final_np.mean():.1f}, std={E_final_np.std():.1f}, "
          f"min={E_final_np.min():.1f}, max={E_final_np.max():.1f}")

    # --- Save results ---
    exp_dir = os.path.join(RESULTS_DIR, 'j3_random_field')
    os.makedirs(exp_dir, exist_ok=True)

    # E field comparison plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        vmin = min(E_true_np.min(), E_final_np.min()) * 0.95
        vmax = max(E_true_np.max(), E_final_np.max()) * 1.05

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plot_E_field(E_true, Nx, Ny, ax=axes[0], title='True E field',
                     vmin=vmin, vmax=vmax)
        plot_E_field(E_final, Nx, Ny, ax=axes[1], title='Recovered E field',
                     vmin=vmin, vmax=vmax)
        plot_E_field(E_error, Nx, Ny, ax=axes[2], title='|E_true - E_inv| error')
        plt.suptitle(f'J3: Random Field Inversion ({Nx}×{Ny}, corr_len=3.0, '
                     f'L2 rel err={l2_rel_error:.2%})', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'E_field_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        pass

    # Convergence plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(history['loss'])
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
    metrics = {
        'L2_relative_error': l2_rel_error,
        'mean_relative_error': mean_rel_error,
        'max_relative_error': max_rel_error,
    }

    save_data = {
        'Nx': Nx, 'Ny': Ny, 'num_cells': nc,
        'traction': traction, 'k_fixed': k_fixed,
        'noise_level': noise_level,
        'reg_type': 'laplacian', 'reg_weight': reg_weight,
        'E_ref': E_ref,
        'optimizer': 'L-BFGS-B (log-E)', 'maxiter': maxiter,
        'metrics': metrics,
        'correlation_coefficient': corr_coeff,
        'time_s': t_opt,
        'nit': int(result.nit),
        'final_loss': float(result.fun),
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
    save_results(exp_dir, save_data)

    print(f"\n  Results saved to {exp_dir}")
    print("=" * 70)

    return result, metrics


if __name__ == "__main__":
    run_random_field_experiment(Nx=30, Ny=30, traction=-50.0,
                                maxiter=150, reg_weight=0.1)
