"""
I3: Regularization weight selection experiment.

Compares TV and Laplacian regularization across a range of λ values
on the H5 two-region benchmark (20×20 QUAD4, E_left=50000, E_right=90000).

Also tests with noisy observations (1%, 3%, 5%) to demonstrate
regularization necessity.
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
    two_region_E_field,
    uniform_E_field,
    plot_E_field,
    save_results,
    RESULTS_DIR,
)
from jax_fem.solver import ad_wrapper
from src.regularization import (
    build_structured_neighbor_pairs,
    build_laplacian_matrix,
    tv_regularizer,
    laplacian_regularizer,
)
import scipy.optimize


def run_regularization_study(Nx=20, Ny=20, traction=-50.0,
                              noise_levels=None, lambda_values=None,
                              maxiter=100):
    """Run regularization comparison experiment."""
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.03, 0.05]
    if lambda_values is None:
        lambda_values = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    exp_dir = os.path.join(RESULTS_DIR, 'i3_regularization_study')
    os.makedirs(exp_dir, exist_ok=True)

    Lx, Ly = 10., 10.
    k_fixed = 50.0
    nc = Nx * Ny

    print("=" * 70)
    print("I3: Regularization Weight Selection Study")
    print("=" * 70)
    print(f"  Mesh: {Nx}×{Ny} = {nc} elements")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Lambda values: {lambda_values}")

    # --- Build E_true and regularization structures ---
    E_true = np.array(two_region_E_field(Nx, Ny, E_left=50000., E_right=90000.))
    neighbor_pairs = build_structured_neighbor_pairs(Nx, Ny)
    L_mat = np.array(build_laplacian_matrix(Nx, Ny))
    E_ref = 70000.0
    print(f"  Neighbor pairs: {len(neighbor_pairs)}")
    print(f"  E_ref: {E_ref}")

    log_E_init = E_to_log(np.full(nc, 70000.0))
    log_E_min = E_to_log(np.array(10000.0))
    log_E_max = E_to_log(np.array(200000.0))
    bounds = [(float(log_E_min), float(log_E_max))] * nc

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

    all_results = []

    for noise_level in noise_levels:
        print(f"\n{'─' * 60}")
        print(f"Noise level: {noise_level * 100:.0f}%")
        print(f"{'─' * 60}")

        # Generate observation with noise
        obs_data = generate_synthetic_observation(
            E_true, k_fixed,
            traction=traction, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
            noise_level=noise_level,
        )
        u_obs = obs_data['u_obs']
        obs_indices = obs_data['obs_indices']
        mesh = obs_data['mesh']
        bc_info = obs_data['dirichlet_bc_info']
        loc_fns = obs_data['location_fns']

        # Build inversion problem and AD wrapper
        problem = InversionHeterogeneousDP2D(
            mesh, vec=2, dim=2, ele_type='QUAD4',
            dirichlet_bc_info=bc_info,
            location_fns=loc_fns,
            E=70000., k=k_fixed,
            traction_value=traction,
        )
        fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                              adjoint_solver_options=solver_options)

        for reg_type in ['none', 'tv', 'laplacian']:
            for lam in lambda_values:
                # Skip lambda=0 for regularized, and lambda>0 for none
                if reg_type == 'none' and lam > 0:
                    continue
                if reg_type != 'none' and lam == 0:
                    continue

                label = f"noise={noise_level:.0%}_reg={reg_type}_lam={lam:.0e}"
                print(f"\n  [{label}]")

                # Build regularizer that operates on log_E but applies
                # TV/Laplacian to physical E
                if reg_type == 'tv':
                    regularizer = lambda log_E, _np=neighbor_pairs, _er=E_ref: \
                        tv_regularizer(np.exp(log_E), _np, E_ref=_er)
                elif reg_type == 'laplacian':
                    regularizer = lambda log_E, _lm=L_mat, _er=E_ref: \
                        laplacian_regularizer(np.exp(log_E), _lm, E_ref=_er)
                else:
                    regularizer = None

                loss_fn = make_heterogeneous_loss(
                    fwd_pred, u_obs, obs_indices,
                    regularizer=regularizer, reg_weight=lam,
                )

                try:
                    loss_and_grad = jax.value_and_grad(loss_fn)

                    def scipy_objective(x):
                        v, g = loss_and_grad(np.array(x))
                        return float(v), onp.array(g, dtype=onp.float64)

                    t0 = time.time()
                    result = scipy.optimize.minimize(
                        scipy_objective, onp.array(log_E_init),
                        method='L-BFGS-B', jac=True,
                        bounds=bounds,
                        options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12},
                    )
                    elapsed = time.time() - t0

                    E_final = onp.array(log_to_E(np.array(result.x)))

                    # Compute metrics inline
                    E_true_np = onp.array(E_true)
                    l2_err = float(onp.linalg.norm(E_final - E_true_np) /
                                   onp.linalg.norm(E_true_np))
                    mean_err = float(onp.mean(onp.abs(E_final - E_true_np) / E_true_np))

                    print(f"    L2 err: {l2_err:.4e}, "
                          f"mean err: {mean_err:.4e}, "
                          f"loss: {result.fun:.4e}, "
                          f"time: {elapsed:.1f}s")

                    entry = {
                        'noise_level': noise_level,
                        'reg_type': reg_type,
                        'lambda': lam,
                        'label': label,
                        'L2_relative_error': l2_err,
                        'mean_relative_error': mean_err,
                        'final_loss': float(result.fun),
                        'time_s': elapsed,
                        'nit': result.nit,
                    }
                    all_results.append(entry)

                    # Save E field plot for selected cases
                    if noise_level in [0.0, 0.03] and lam in [0.0, 1e-2, 1.0]:
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                        vmin = min(float(E_true.min()), float(E_final.min()))
                        vmax = max(float(E_true.max()), float(E_final.max()))
                        plot_E_field(E_true, Nx, Ny, ax=axes[0],
                                     title='True', vmin=vmin, vmax=vmax)
                        plot_E_field(E_final, Nx, Ny, ax=axes[1],
                                     title=f'Inverted ({label})',
                                     vmin=vmin, vmax=vmax)
                        plt.tight_layout()
                        plt.savefig(os.path.join(exp_dir, f'{label}.png'),
                                    dpi=150, bbox_inches='tight')
                        plt.close()

                except Exception as e:
                    print(f"    FAILED: {e}")
                    all_results.append({
                        'noise_level': noise_level,
                        'reg_type': reg_type,
                        'lambda': lam,
                        'label': label,
                        'error': str(e),
                    })

    # --- Save summary ---
    save_results(exp_dir, all_results)

    # --- Print summary table ---
    print(f"\n{'=' * 70}")
    print("Summary Table")
    print(f"{'=' * 70}")
    print(f"{'Noise':<8} {'Reg':<12} {'λ':<10} {'L2 err':<12} {'Mean err':<12} {'Time':<8}")
    print("-" * 62)
    for r in all_results:
        if 'error' in r:
            print(f"{r['noise_level']:<8.0%} {r['reg_type']:<12} "
                  f"{r['lambda']:<10.0e} FAILED")
        else:
            print(f"{r['noise_level']:<8.0%} {r['reg_type']:<12} "
                  f"{r['lambda']:<10.0e} {r['L2_relative_error']:<12.4e} "
                  f"{r['mean_relative_error']:<12.4e} {r['time_s']:<8.1f}")

    # --- L-curve plot ---
    try:
        import matplotlib.pyplot as plt

        for noise_level in noise_levels:
            if noise_level == 0.0:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, reg_type in zip(axes, ['tv', 'laplacian']):
                subset = [r for r in all_results
                          if r.get('noise_level') == noise_level
                          and r.get('reg_type') == reg_type
                          and 'error' not in r]
                if not subset:
                    continue
                lambdas = [r['lambda'] for r in subset]
                l2_errs = [r['L2_relative_error'] for r in subset]
                ax.semilogx(lambdas, l2_errs, 'o-')
                ax.set_xlabel('λ')
                ax.set_ylabel('L2 relative error')
                ax.set_title(f'{reg_type}, noise={noise_level:.0%}')
                ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir,
                        f'lambda_vs_error_noise{noise_level:.0%}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
    except ImportError:
        pass

    print(f"\nResults saved to {exp_dir}")
    return all_results


if __name__ == "__main__":
    run_regularization_study(
        Nx=20, Ny=20, traction=-50.0,
        noise_levels=[0.0, 0.01, 0.03],
        lambda_values=[0.0, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        maxiter=100,
    )
