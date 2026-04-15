#!/usr/bin/env python
"""
P0: Precision Tuning — Improve heterogeneous inversion accuracy.

Tasks:
  P0-1: H5 two-region with regularization sweep (target: L2 < 5%)
  P0-2: J1 layered with λ sweep (target: L2 < 10%)

Runs all configurations and saves comparative results + plots.
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

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from common import (
    InversionHeterogeneousDP2D,
    create_2d_mesh_traction_bc,
    generate_synthetic_observation,
    make_heterogeneous_loss,
    two_region_E_field,
    uniform_E_field,
    layered_E_field,
    log_to_E, E_to_log,
    plot_E_field,
    save_results,
    RESULTS_DIR,
)

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_fem.solver import ad_wrapper
from src.regularization import (
    build_structured_neighbor_pairs,
    build_laplacian_matrix,
    tv_regularizer,
    laplacian_regularizer,
)
from scipy.optimize import minimize as scipy_minimize


SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}


def run_single_inversion(E_true, Nx, Ny, Lx, Ly, traction, k_fixed,
                          noise_level, reg_type, lam, E_ref, maxiter,
                          neighbor_pairs, L_mat):
    """Run one inversion configuration. Returns dict with results or error."""
    nc = Nx * Ny
    label = f"reg={reg_type}_lam={lam:.0e}"

    obs_data = generate_synthetic_observation(
        E_true, k_fixed,
        traction=traction, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        noise_level=noise_level,
    )
    u_obs = obs_data['u_obs']
    obs_indices = obs_data['obs_indices']

    mesh, bc_info, loc_fns, _ = create_2d_mesh_traction_bc(
        traction, Lx, Ly, Nx, Ny)

    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        location_fns=loc_fns,
        E=70000., k=k_fixed,
        traction_value=traction,
    )
    fwd_pred = ad_wrapper(problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    if reg_type == 'tv':
        regularizer = lambda log_E: tv_regularizer(
            np.exp(log_E), neighbor_pairs, E_ref=E_ref)
    elif reg_type == 'laplacian':
        regularizer = lambda log_E: laplacian_regularizer(
            np.exp(log_E), L_mat, E_ref=E_ref)
    elif reg_type == 'smoothness':
        # simple squared first-difference on structured grid
        def regularizer(log_E):
            g = log_E.reshape(Nx, Ny)
            dx = g[1:, :] - g[:-1, :]
            dy = g[:, 1:] - g[:, :-1]
            return np.mean(dx ** 2) + np.mean(dy ** 2)
    else:
        regularizer = None

    loss_fn = make_heterogeneous_loss(
        fwd_pred, u_obs, obs_indices,
        regularizer=regularizer, reg_weight=lam,
    )

    value_and_grad_fn = jax.value_and_grad(loss_fn)
    log_E_init = onp.array(E_to_log(np.full(nc, E_ref)), dtype=onp.float64)

    # warmup
    _ = value_and_grad_fn(np.array(log_E_init))

    log_E_lo = float(onp.log(5000.))
    log_E_hi = float(onp.log(300000.))
    bounds = [(log_E_lo, log_E_hi)] * nc

    loss_history = []

    def objective(x):
        log_E = np.array(x)
        loss, grad = value_and_grad_fn(log_E)
        loss_f = float(loss)
        grad_np = onp.array(grad, dtype=onp.float64)
        loss_history.append(loss_f)
        return loss_f, grad_np

    t0 = time.time()
    result = scipy_minimize(
        objective,
        x0=log_E_init,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'maxfun': maxiter * 5,
                 'ftol': 1e-20, 'gtol': 1e-12},
    )
    elapsed = time.time() - t0

    E_final = onp.exp(result.x)
    E_true_np = onp.array(E_true)

    err = onp.abs(E_final - E_true_np)
    rel_err = err / E_true_np
    l2_rel = float(onp.sqrt(onp.mean(rel_err ** 2)))
    mean_rel = float(onp.mean(rel_err))
    max_rel = float(onp.max(rel_err))
    l2_abs = float(onp.sqrt(onp.mean(err ** 2)))

    return {
        'reg_type': reg_type,
        'lambda': lam,
        'label': label,
        'l2_rel': l2_rel,
        'mean_rel': mean_rel,
        'max_rel': max_rel,
        'l2_abs': l2_abs,
        'final_loss': float(result.fun),
        'nit': int(result.nit),
        'nfev': int(result.nfev),
        'converged': bool(result.success),
        'message': str(result.message),
        'time_s': elapsed,
        'loss_history': loss_history,
        'E_final': E_final,
    }


# ═══════════════════════════════════════════════════════════════════════════
# P0-1: H5 Two-Region with Regularization
# ═══════════════════════════════════════════════════════════════════════════
def p0_1_h5_regularization():
    """H5 two-region inversion with regularization sweep."""
    print("\n" + "=" * 70)
    print("P0-1: H5 TWO-REGION WITH REGULARIZATION SWEEP")
    print("=" * 70)

    Nx, Ny = 20, 20
    Lx, Ly = 10., 10.
    traction = -50.0
    k_fixed = 500.0
    E_ref = 70000.0
    nc = Nx * Ny
    maxiter = 300

    E_true = onp.array(two_region_E_field(Nx, Ny, 50000., 90000.))
    neighbor_pairs = build_structured_neighbor_pairs(Nx, Ny)
    L_mat = np.array(build_laplacian_matrix(Nx, Ny))

    configs = [
        ('none',       0.0),
        ('smoothness', 1e-4),
        ('smoothness', 1e-3),
        ('smoothness', 1e-2),
        ('smoothness', 0.1),
        ('tv',         1e-4),
        ('tv',         1e-3),
        ('tv',         1e-2),
        ('tv',         0.1),
        ('laplacian',  1e-4),
        ('laplacian',  1e-3),
        ('laplacian',  1e-2),
        ('laplacian',  0.1),
    ]

    results = []
    for reg_type, lam in configs:
        print(f"\n  --- {reg_type} λ={lam:.0e} ---")
        try:
            r = run_single_inversion(
                E_true, Nx, Ny, Lx, Ly, traction, k_fixed,
                noise_level=0.0, reg_type=reg_type, lam=lam,
                E_ref=E_ref, maxiter=maxiter,
                neighbor_pairs=neighbor_pairs, L_mat=L_mat,
            )
            print(f"    L2 rel: {r['l2_rel']:.4f}  mean rel: {r['mean_rel']:.4f}  "
                  f"nit: {r['nit']}  time: {r['time_s']:.1f}s")
            results.append(r)
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                'reg_type': reg_type, 'lambda': lam,
                'label': f"reg={reg_type}_lam={lam:.0e}",
                'error': str(e),
            })

    # Find best
    valid = [r for r in results if 'l2_rel' in r]
    if valid:
        best = min(valid, key=lambda r: r['l2_rel'])
        print(f"\n  BEST: {best['label']}  L2 rel = {best['l2_rel']:.4f}")
    else:
        best = None
        print("\n  All configurations failed!")

    # --- Save & Plot ---
    out_dir = os.path.join(RESULTS_DIR, 'h5_two_region')
    os.makedirs(out_dir, exist_ok=True)

    # Summary JSON
    summary = []
    for r in results:
        entry = {k: v for k, v in r.items()
                 if k not in ('loss_history', 'E_final')}
        if 'E_final' in r:
            entry['E_range'] = [float(r['E_final'].min()),
                                float(r['E_final'].max())]
        summary.append(entry)

    with open(os.path.join(out_dir, 'p0_regularization_sweep.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # --- Comparison plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 0: E-field maps for best 3
    top3 = sorted(valid, key=lambda r: r['l2_rel'])[:3] if valid else []
    vmin = 50000 * 0.9
    vmax = 90000 * 1.1
    for col, r in enumerate(top3):
        ax = axes[0, col]
        plot_E_field(r['E_final'], Nx, Ny, ax=ax,
                     title=f"{r['reg_type']} λ={r['lambda']:.0e}\n"
                           f"L2 rel = {r['l2_rel']:.2%}",
                     vmin=vmin, vmax=vmax)

    # Fill empty slots with true field
    if len(top3) < 3:
        for col in range(len(top3), 3):
            plot_E_field(E_true, Nx, Ny, ax=axes[0, col],
                         title='True E field', vmin=vmin, vmax=vmax)

    # Row 1, Col 0: L2 error vs lambda for each reg type
    ax = axes[1, 0]
    for reg_type, color, marker in [('smoothness', '#4C72B0', 'o'),
                                     ('tv', '#55A868', 's'),
                                     ('laplacian', '#C44E52', '^')]:
        subset = [r for r in valid if r['reg_type'] == reg_type and r['lambda'] > 0]
        if subset:
            lams = [r['lambda'] for r in subset]
            errs = [r['l2_rel'] for r in subset]
            ax.semilogx(lams, errs, f'{marker}-', color=color, ms=7, lw=1.5,
                        label=reg_type.capitalize())
    # No-reg baseline
    no_reg = [r for r in valid if r['reg_type'] == 'none']
    if no_reg:
        ax.axhline(no_reg[0]['l2_rel'], color='gray', ls='--', lw=1,
                   label=f"No reg ({no_reg[0]['l2_rel']:.2%})")
    ax.axhline(0.05, color='red', ls=':', lw=0.8, alpha=0.5, label='5% target')
    ax.set_xlabel('Regularization Weight λ')
    ax.set_ylabel('L2 Relative Error')
    ax.set_title('H5: Error vs λ')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.25, max(r['l2_rel'] for r in valid) * 1.1) if valid else 0.3)

    # Row 1, Col 1: Line cut comparison (best vs no-reg)
    ax = axes[1, 1]
    iy_mid = Ny // 2
    x_centers = onp.linspace(Lx / (2 * Nx), Lx - Lx / (2 * Nx), Nx)
    E_true_line = E_true.reshape(Nx, Ny)[:, iy_mid]
    ax.plot(x_centers, E_true_line, 'k-', lw=2.5, label='True')
    if no_reg:
        E_noreg_line = no_reg[0]['E_final'].reshape(Nx, Ny)[:, iy_mid]
        ax.plot(x_centers, E_noreg_line, '--', color='gray', lw=1.2,
                label=f"No reg (L2={no_reg[0]['l2_rel']:.2%})")
    if best:
        E_best_line = best['E_final'].reshape(Nx, Ny)[:, iy_mid]
        ax.plot(x_centers, E_best_line, 'o-', color='#4C72B0', ms=4, lw=1.5,
                label=f"Best: {best['label']} ({best['l2_rel']:.2%})")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('E [MPa]')
    ax.set_title(f'Line Cut at y = {Ly/2:.0f} m')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1, Col 2: Convergence curves
    ax = axes[1, 2]
    for r in valid[:6]:
        if r['loss_history']:
            ax.semilogy(r['loss_history'], lw=1.2, alpha=0.8,
                        label=f"{r['reg_type']} λ={r['lambda']:.0e}")
    ax.set_xlabel('Function Evaluation')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('P0-1: H5 Two-Region Inversion — Regularization Sweep',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'P0_正则化调优对比.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {plot_path}")

    # Save best E field
    if best:
        onp.save(os.path.join(out_dir, 'E_recovered_best.npy'), best['E_final'])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# P0-2: J1 Layered with λ Sweep
# ═══════════════════════════════════════════════════════════════════════════
def p0_2_j1_tuning():
    """J1 layered inversion with regularization tuning."""
    print("\n" + "=" * 70)
    print("P0-2: J1 LAYERED INVERSION — REGULARIZATION TUNING")
    print("=" * 70)

    Nx, Ny = 30, 30
    Lx, Ly = 10., 10.
    traction = -50.0
    k_fixed = 500.0
    E_ref = 70000.0
    nc = Nx * Ny
    maxiter = 200
    noise_level = 0.01

    # 4-layer field
    E_true = onp.array(layered_E_field(Nx, Ny,
                        E_values=[40000., 70000., 90000., 60000.],
                        layer_boundaries=[0.25, 0.5, 0.75]))

    neighbor_pairs = build_structured_neighbor_pairs(Nx, Ny)
    L_mat = np.array(build_laplacian_matrix(Nx, Ny))

    configs = [
        ('none',       0.0),
        ('tv',         1e-4),
        ('tv',         1e-3),
        ('tv',         1e-2),
        ('tv',         0.05),
        ('tv',         0.1),
        ('laplacian',  1e-4),
        ('laplacian',  1e-3),
        ('laplacian',  1e-2),
        ('laplacian',  0.05),
        ('laplacian',  0.1),
        ('smoothness', 1e-3),
        ('smoothness', 1e-2),
        ('smoothness', 0.1),
    ]

    results = []
    for reg_type, lam in configs:
        print(f"\n  --- {reg_type} λ={lam:.0e} ---")
        try:
            r = run_single_inversion(
                E_true, Nx, Ny, Lx, Ly, traction, k_fixed,
                noise_level=noise_level, reg_type=reg_type, lam=lam,
                E_ref=E_ref, maxiter=maxiter,
                neighbor_pairs=neighbor_pairs, L_mat=L_mat,
            )
            print(f"    L2 rel: {r['l2_rel']:.4f}  mean rel: {r['mean_rel']:.4f}  "
                  f"nit: {r['nit']}  time: {r['time_s']:.1f}s")
            results.append(r)
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                'reg_type': reg_type, 'lambda': lam,
                'label': f"reg={reg_type}_lam={lam:.0e}",
                'error': str(e),
            })

    valid = [r for r in results if 'l2_rel' in r]
    if valid:
        best = min(valid, key=lambda r: r['l2_rel'])
        print(f"\n  BEST: {best['label']}  L2 rel = {best['l2_rel']:.4f}")
    else:
        best = None

    # --- Save & Plot ---
    out_dir = os.path.join(RESULTS_DIR, 'j1_layered')
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for r in results:
        entry = {k: v for k, v in r.items()
                 if k not in ('loss_history', 'E_final')}
        if 'E_final' in r:
            entry['E_range'] = [float(r['E_final'].min()),
                                float(r['E_final'].max())]
        summary.append(entry)

    with open(os.path.join(out_dir, 'p0_lambda_sweep.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Top row: E field maps
    vmin = 35000
    vmax = 95000
    plot_E_field(E_true, Nx, Ny, ax=axes[0, 0], title='True E field',
                 vmin=vmin, vmax=vmax)
    top2 = sorted(valid, key=lambda r: r['l2_rel'])[:2] if valid else []
    for col, r in enumerate(top2):
        plot_E_field(r['E_final'], Nx, Ny, ax=axes[0, col + 1],
                     title=f"{r['reg_type']} λ={r['lambda']:.0e}\n"
                           f"L2 rel = {r['l2_rel']:.2%}",
                     vmin=vmin, vmax=vmax)

    # Bottom left: L2 error vs lambda
    ax = axes[1, 0]
    for reg_type, color, marker in [('smoothness', '#4C72B0', 'o'),
                                     ('tv', '#55A868', 's'),
                                     ('laplacian', '#C44E52', '^')]:
        subset = [r for r in valid if r['reg_type'] == reg_type and r['lambda'] > 0]
        if subset:
            lams = [r['lambda'] for r in subset]
            errs = [r['l2_rel'] for r in subset]
            ax.semilogx(lams, errs, f'{marker}-', color=color, ms=7, lw=1.5,
                        label=reg_type.capitalize())
    no_reg = [r for r in valid if r['reg_type'] == 'none']
    if no_reg:
        ax.axhline(no_reg[0]['l2_rel'], color='gray', ls='--', lw=1,
                   label=f"No reg ({no_reg[0]['l2_rel']:.2%})")
    ax.axhline(0.10, color='red', ls=':', lw=0.8, alpha=0.5, label='10% target')
    ax.set_xlabel('Regularization Weight λ')
    ax.set_ylabel('L2 Relative Error')
    ax.set_title('J1: Error vs λ (1% noise)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom middle: layer-by-layer comparison
    ax = axes[1, 1]
    if best:
        layers_def = [(0., 2.5, 40000.), (2.5, 5., 70000.),
                      (5., 7.5, 90000.), (7.5, 10., 60000.)]
        layer_names = [f"L{i+1}" for i in range(4)]
        E_true_vals = [l[2] / 1000 for l in layers_def]
        E_inv_vals = []
        E_inv_stds = []
        for y_lo, y_hi, _ in layers_def:
            mask = onp.zeros(nc, dtype=bool)
            for ix in range(Nx):
                for iy in range(Ny):
                    y_c = (iy + 0.5) * Ly / Ny
                    if y_lo <= y_c < y_hi:
                        mask[ix * Ny + iy] = True
            E_inv_vals.append(best['E_final'][mask].mean() / 1000)
            E_inv_stds.append(best['E_final'][mask].std() / 1000)

        x = onp.arange(4)
        w = 0.35
        ax.bar(x - w/2, E_true_vals, w, label='True', color='#4C72B0')
        ax.bar(x + w/2, E_inv_vals, w, yerr=E_inv_stds,
               label='Best inverted', color='#DD8452', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names)
        ax.set_ylabel('E [GPa]')
        ax.set_title(f'Layer Recovery ({best["label"]})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom right: convergence
    ax = axes[1, 2]
    for r in valid[:6]:
        if r['loss_history']:
            ax.semilogy(r['loss_history'], lw=1.2, alpha=0.8,
                        label=f"{r['reg_type']} λ={r['lambda']:.0e}")
    ax.set_xlabel('Function Evaluation')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('P0-2: J1 Layered Inversion — Regularization Tuning (1% noise)',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'P0_正则化调优对比.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {plot_path}")

    if best:
        onp.save(os.path.join(out_dir, 'E_recovered_best.npy'), best['E_final'])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("P0: PRECISION TUNING FOR HETEROGENEOUS INVERSION")
    print("=" * 70)
    t0 = time.time()

    h5_results = p0_1_h5_regularization()
    j1_results = p0_2_j1_tuning()

    t_total = time.time() - t0

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("P0 SUMMARY")
    print("=" * 70)

    for name, results in [("H5 Two-Region", h5_results),
                           ("J1 Layered", j1_results)]:
        valid = [r for r in results if 'l2_rel' in r]
        if valid:
            best = min(valid, key=lambda r: r['l2_rel'])
            print(f"\n  {name}:")
            print(f"    Best config:  {best['label']}")
            print(f"    L2 rel error: {best['l2_rel']:.4f} ({best['l2_rel']:.2%})")
            print(f"    Mean rel err: {best['mean_rel']:.4f}")
            print(f"    Max rel err:  {best['max_rel']:.4f}")
            print(f"    Iterations:   {best['nit']}")
            target = 0.05 if 'H5' in name else 0.10
            status = "PASS" if best['l2_rel'] < target else "NEEDS WORK"
            print(f"    Status:       {status} (target < {target:.0%})")

    print(f"\n  Total time: {t_total:.0f}s ({t_total/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
