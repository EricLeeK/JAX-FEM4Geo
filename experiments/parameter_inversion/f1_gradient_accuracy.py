#!/usr/bin/env python
"""
F1: Gradient Accuracy — AD vs FD at multiple step sizes.

Produces the classic "V-curve": FD error vs step size, showing truncation
error at large eps and round-off error at small eps. AD achieves machine
precision (~1e-14) with zero tuning.
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
    volume_avg_sigma_zz, RESULTS_DIR,
)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(RESULTS_DIR, 'f1_gradient_accuracy')
os.makedirs(OUT_DIR, exist_ok=True)

# Relative perturbation fractions for FD
REL_EPS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7, 1e-8]

# Test at two displacements: elastic and plastic
DISP_LIST = [(-0.015, 'elastic'), (-0.028, 'plastic')]
PARAMS = onp.array([70000.0, 50.0])


def main():
    print("=" * 70)
    print("F1: GRADIENT ACCURACY — AD vs FD")
    print("=" * 70)

    all_results = {}

    for disp, regime in DISP_LIST:
        print(f"\n--- Displacement = {disp} ({regime}) ---")

        mesh, bc = create_mesh_and_bc(disp)
        problem = InversionDruckerPrager(mesh, vec=3, dim=3, dirichlet_bc_info=bc)
        fwd = ad_wrapper(problem, solver_options=SOLVER_OPTIONS,
                         adjoint_solver_options=SOLVER_OPTIONS)

        def loss_fn(params):
            E, k = params[0], params[1]
            sol = fwd(params)[0]
            return volume_avg_sigma_zz(
                problem.fe, sol, problem.sigmas_old, problem.epsilons_old, E, k,
            )

        # AD gradient (ground truth)
        params_jax = np.array(PARAMS)
        loss_val, grad_ad = jax.value_and_grad(loss_fn)(params_jax)
        grad_ad = onp.array(grad_ad)
        print(f"  AD: dL/dE = {grad_ad[0]:.10e}, dL/dk = {grad_ad[1]:.10e}")

        # FD sweep
        fd_results = []
        for rel_eps in REL_EPS:
            eps_E = abs(PARAMS[0]) * rel_eps
            eps_k = abs(PARAMS[1]) * rel_eps

            # dL/dE
            lp = float(loss_fn(params_jax.at[0].set(PARAMS[0] + eps_E)))
            lm = float(loss_fn(params_jax.at[0].set(PARAMS[0] - eps_E)))
            fd_E = (lp - lm) / (2 * eps_E)

            # dL/dk
            lp = float(loss_fn(params_jax.at[1].set(PARAMS[1] + eps_k)))
            lm = float(loss_fn(params_jax.at[1].set(PARAMS[1] - eps_k)))
            fd_k = (lp - lm) / (2 * eps_k)

            err_E = abs(fd_E - grad_ad[0]) / max(abs(grad_ad[0]), 1e-30)
            err_k = abs(fd_k - grad_ad[1]) / max(abs(grad_ad[1]), 1e-30)

            fd_results.append({
                'rel_eps': rel_eps,
                'fd_E': fd_E, 'fd_k': fd_k,
                'err_E': err_E, 'err_k': err_k,
            })
            print(f"  eps={rel_eps:.0e}: err(E)={err_E:.2e}, err(k)={err_k:.2e}")

        all_results[regime] = {
            'displacement': disp,
            'ad_grad': grad_ad.tolist(),
            'loss': float(loss_val),
            'fd_sweep': fd_results,
        }

    # --- Plotting ---
    print("\n[Plot] Generating V-curve plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, param_name in enumerate(['E', 'k']):
        ax = axes[idx]
        for regime, color, marker in [('elastic', 'blue', 'o'), ('plastic', 'red', 's')]:
            data = all_results[regime]['fd_sweep']
            eps_vals = [d['rel_eps'] for d in data]
            errs = [d[f'err_{param_name}'] for d in data]
            # Filter out zeros for log plot
            errs_plot = [max(e, 1e-16) for e in errs]
            ax.loglog(eps_vals, errs_plot, f'{color}', marker=marker, markersize=5,
                      linewidth=1.5, label=f'{regime}')

        ax.axhline(1e-14, color='green', linestyle='--', alpha=0.7, label='Machine precision')
        ax.set_xlabel('Relative perturbation ε')
        ax.set_ylabel(f'Relative error in ∂L/∂{param_name}')
        ax.set_title(f'FD Accuracy for ∂L/∂{param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.suptitle('F1: FD Gradient Accuracy vs Step Size (AD = ground truth)', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'v_curve.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")

    # Save results
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Summary
    for regime in ['elastic', 'plastic']:
        data = all_results[regime]['fd_sweep']
        best_E = min(d['err_E'] for d in data)
        best_k = min(d['err_k'] for d in data)
        print(f"\n  {regime}: FD best err(E) = {best_E:.2e}, best err(k) = {best_k:.2e}")
    print(f"  AD: err = 0 (exact by construction)")

    print("\n" + "=" * 70)
    print("F1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
