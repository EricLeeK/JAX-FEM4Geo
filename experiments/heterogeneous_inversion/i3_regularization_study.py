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
    create_2d_mesh_and_bc,
    generate_synthetic_observation,
    displacement_loss,
    optimize_lbfgsb,
    plot_E_field_comparison,
    compute_inversion_metrics,
    RESULTS_DIR,
)
from jax_fem.solver import ad_wrapper
from src.regularization import (
    build_structured_neighbor_pairs,
    build_laplacian_matrix,
    tv_regularizer,
    laplacian_regularizer,
)


def build_two_region_E_field(Nx, Ny):
    nc = Nx * Ny
    E_field = onp.full(nc, 90000.0)
    for ix in range(Nx):
        for iy in range(Ny):
            if ix < Nx // 2:
                E_field[ix * Ny + iy] = 50000.0
    return np.array(E_field)


def run_regularization_study(Nx=20, Ny=20, displacement=-0.1,
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

    # --- Build mesh, problem, E_true ---
    E_true = build_two_region_E_field(Nx, Ny)
    mesh, bc_info = create_2d_mesh_and_bc(displacement, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info, E_init=70000.0, k=k_fixed,
    )
    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    # --- Build regularization structures ---
    neighbor_pairs = build_structured_neighbor_pairs(Nx, Ny)
    L_mat = np.array(build_laplacian_matrix(Nx, Ny))
    E_ref = 70000.0  # Reference E scale for normalization
    print(f"  Neighbor pairs: {len(neighbor_pairs)}")
    print(f"  E_ref: {E_ref}")

    E_init = np.full(nc, 70000.0)
    all_results = []

    for noise_level in noise_levels:
        print(f"\n{'─' * 60}")
        print(f"Noise level: {noise_level * 100:.0f}%")
        print(f"{'─' * 60}")

        # Generate observation with noise
        key = jax.random.PRNGKey(42) if noise_level > 0 else None
        obs_data, obs_indices, sol_true = generate_synthetic_observation(
            problem, fwd_pred, E_true,
            noise_level=noise_level, key=key,
        )

        for reg_type in ['none', 'tv', 'laplacian']:
            for lam in lambda_values:
                # Skip lambda=0 for regularized, and lambda>0 for none
                if reg_type == 'none' and lam > 0:
                    continue
                if reg_type != 'none' and lam == 0:
                    continue

                label = f"noise={noise_level:.0%}_reg={reg_type}_lam={lam:.0e}"
                print(f"\n  [{label}]")

                if reg_type == 'tv':
                    regularizer = lambda E: tv_regularizer(E, neighbor_pairs, E_ref=E_ref)
                elif reg_type == 'laplacian':
                    regularizer = lambda E: laplacian_regularizer(E, L_mat, E_ref=E_ref)
                else:
                    regularizer = None

                def loss_fn(E_field, _reg=regularizer, _lam=lam):
                    return displacement_loss(
                        E_field, problem, fwd_pred, obs_data, obs_indices,
                        regularizer=_reg, reg_weight=_lam,
                    )

                try:
                    result = optimize_lbfgsb(
                        loss_fn, E_init, E_min=10000.0, E_max=200000.0,
                        maxiter=maxiter, verbose=False,
                    )
                    metrics = compute_inversion_metrics(E_true, result['E_final'])

                    print(f"    L2 err: {metrics['L2_relative_error']:.4e}, "
                          f"mean err: {metrics['mean_relative_error']:.4e}, "
                          f"loss: {result['loss_history'][-1]:.4e}, "
                          f"time: {result['time_s']:.1f}s")

                    entry = {
                        'noise_level': noise_level,
                        'reg_type': reg_type,
                        'lambda': lam,
                        'label': label,
                        **metrics,
                        'final_loss': result['loss_history'][-1],
                        'time_s': result['time_s'],
                        'nit': result['nit'],
                    }
                    all_results.append(entry)

                    # Save E field plot for selected cases
                    if noise_level in [0.0, 0.03] and lam in [0.0, 1e-2, 1.0]:
                        plot_E_field_comparison(
                            E_true, result['E_final'], Nx, Ny, Lx, Ly,
                            title=f'{label}',
                            save_path=os.path.join(exp_dir, f'{label}.png'),
                        )

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
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

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
        Nx=20, Ny=20, displacement=-0.1,
        noise_levels=[0.0, 0.01, 0.03],
        lambda_values=[0.0, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        maxiter=100,
    )
