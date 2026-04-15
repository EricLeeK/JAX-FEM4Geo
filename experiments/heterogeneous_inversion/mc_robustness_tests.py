#!/usr/bin/env python
"""
MC Robustness Tests: noise, sparse observations, layered field.

Tests the MC c(x) inversion under progressively harder conditions:
  T1: Baseline (no noise, full obs) — already done in mc_two_region.py
  T2: 1% observation noise
  T3: 3% observation noise
  T4: Sparse observations (boundary nodes only)
  T5: 3-layer c field (c=20/60/40)
  T6: 1% noise + smoothness regularization

All use displacement control, 10×10 mesh, disp=-0.05.
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
    InversionHeterogeneousMC2D,
    create_2d_mesh_and_bc,
    make_heterogeneous_loss,
    smoothness_regularizer,
    two_region_c_field,
    uniform_c_field,
    log_to_E as log_to_c,
    E_to_log as c_to_log,
    plot_c_field,
    save_results,
    RESULTS_DIR,
)

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import rectangle_mesh, Mesh
from scipy.optimize import minimize as scipy_minimize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
Nx, Ny = 10, 10
Lx, Ly = 10., 10.
DISPLACEMENT = -0.05
PHI_DEG = 30.0
PSI_DEG = 15.0
E_FIXED = 70000.
C_LEFT, C_RIGHT = 30., 70.
C_INIT = 50.
NUM_CELLS = Nx * Ny
SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
MAXITER = 200

OUT_DIR = os.path.join(RESULTS_DIR, 'mc_robustness')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Field generators
# ---------------------------------------------------------------------------

def layered_c_field(Nx, Ny, c_values, boundaries):
    """Horizontally layered c field.

    boundaries: list of normalized y-fractions (e.g. [0.33, 0.67] for 3 layers).
    c_values: list of c per layer from bottom to top.
    """
    c = onp.zeros(Nx * Ny)
    for ix in range(Nx):
        for iy in range(Ny):
            y_frac = (iy + 0.5) / Ny
            layer = 0
            for b in boundaries:
                if y_frac > b:
                    layer += 1
            c[ix * Ny + iy] = c_values[min(layer, len(c_values) - 1)]
    return c


def get_boundary_node_indices(Nx, Ny, Lx, Ly):
    """Return node indices on the boundary of a rectangle mesh."""
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    pts = meshio_mesh.points[:, :2]
    on_boundary = (
        onp.isclose(pts[:, 0], 0., atol=1e-5) |
        onp.isclose(pts[:, 0], Lx, atol=1e-5) |
        onp.isclose(pts[:, 1], 0., atol=1e-5) |
        onp.isclose(pts[:, 1], Ly, atol=1e-5)
    )
    return onp.where(on_boundary)[0]


# ---------------------------------------------------------------------------
# Core inversion runner
# ---------------------------------------------------------------------------

def run_inversion(c_true, obs_indices, noise_level=0.0, regularizer=None,
                  reg_weight=0.0, label='test', maxiter=MAXITER):
    """Run a single MC c(x) inversion experiment.

    Returns dict with results.
    """
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")

    # Generate observation
    mesh_obs, bc_obs = create_2d_mesh_and_bc(DISPLACEMENT, Lx, Ly, Nx, Ny)
    prob_obs = InversionHeterogeneousMC2D(
        mesh_obs, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_obs,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    prob_obs.set_params(np.array(c_true))
    sol_list = solver(prob_obs, solver_options=SOLVER_OPTIONS)
    u_full = sol_list[0]

    u_obs = u_full[obs_indices]
    if noise_level > 0.0:
        max_disp = float(np.max(np.abs(u_full)))
        noise = noise_level * max_disp * jax.random.normal(
            jax.random.PRNGKey(42), u_obs.shape)
        u_obs = u_obs + noise

    print(f"  Obs nodes: {len(obs_indices)}/{u_full.shape[0]}, "
          f"noise: {noise_level:.0%}")

    # Set up inversion
    mesh_inv, bc_inv = create_2d_mesh_and_bc(DISPLACEMENT, Lx, Ly, Nx, Ny)
    inv_prob = InversionHeterogeneousMC2D(
        mesh_inv, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_inv,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    fwd_pred = ad_wrapper(inv_prob, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    loss_fn = make_heterogeneous_loss(fwd_pred, u_obs, obs_indices,
                                      regularizer=regularizer,
                                      reg_weight=reg_weight)
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    log_c_init = onp.array(c_to_log(uniform_c_field(Nx, Ny, C_INIT)),
                           dtype=onp.float64)

    # Warmup
    l0, g0 = value_and_grad_fn(np.array(log_c_init))
    grad_norm = float(np.linalg.norm(g0))
    print(f"  Init loss: {float(l0):.6e}, ||grad||: {grad_norm:.6e}")

    if grad_norm < 1e-20:
        print(f"  ⚠ Zero gradient — skipping")
        return {'label': label, 'status': 'SKIP', 'l2_rel': 1.0}

    # Optimize
    bounds = [(float(onp.log(5.)), float(onp.log(500.)))] * NUM_CELLS
    last_good = [float(l0), onp.array(g0, dtype=onp.float64)]
    history = []

    def objective(x):
        try:
            loss, grad = value_and_grad_fn(np.array(x))
            lf = float(loss)
            gn = onp.array(grad, dtype=onp.float64)
            if not (onp.isfinite(lf) and onp.all(onp.isfinite(gn))):
                raise ValueError("NaN")
            last_good[0] = lf
            last_good[1] = gn
        except Exception:
            lf = last_good[0] * 10.
            gn = onp.zeros_like(last_good[1])
        history.append(lf)
        return lf, gn

    t0 = time.time()
    result = scipy_minimize(
        objective, x0=log_c_init, method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'maxfun': 500,
                 'ftol': 1e-20, 'gtol': 1e-12},
    )
    dt = time.time() - t0

    c_final = onp.exp(result.x)
    c_true_np = onp.array(c_true)
    rel_err = onp.abs(c_final - c_true_np) / c_true_np
    l2_rel = float(onp.sqrt(onp.mean(rel_err ** 2)))
    mean_rel = float(onp.mean(rel_err))

    status = "PASS" if l2_rel < 0.10 else "IMPROVED" if l2_rel < 0.25 else "NEEDS WORK"

    print(f"  L2 rel error: {l2_rel:.4f} ({l2_rel:.2%})  [{status}]")
    print(f"  c range: [{c_final.min():.1f}, {c_final.max():.1f}]")
    print(f"  Iters: {result.nit}, Evals: {result.nfev}, Time: {dt:.1f}s")

    return {
        'label': label,
        'c_true': c_true_np,
        'c_final': c_final,
        'l2_rel': l2_rel,
        'mean_rel': mean_rel,
        'nit': result.nit,
        'nfev': result.nfev,
        'time_s': dt,
        'status': status,
        'history': history,
        'noise_level': noise_level,
        'n_obs': len(obs_indices),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MC Robustness Tests")
    print(f"  Mesh: {Nx}×{Ny}, disp={DISPLACEMENT}, phi={PHI_DEG}°")
    print("=" * 70)

    all_obs = onp.arange((Nx + 1) * (Ny + 1))
    boundary_obs = get_boundary_node_indices(Nx, Ny, Lx, Ly)
    c_two_region = two_region_c_field(Nx, Ny, C_LEFT, C_RIGHT)
    c_layered = layered_c_field(Nx, Ny, [20., 60., 40.], [0.33, 0.67])
    reg_fn = smoothness_regularizer(Nx, Ny)

    # 3-region vertical split (x-varying): c=25/60/40
    # Note: horizontal layers (y-varying) cause Newton divergence under
    # y-compression because layers stack in the loading direction.
    c_3region = onp.full(NUM_CELLS, 60.)
    for ix in range(Nx):
        for iy in range(Ny):
            x_frac = (ix + 0.5) / Nx
            if x_frac < 0.33:
                c_3region[ix * Ny + iy] = 25.
            elif x_frac > 0.67:
                c_3region[ix * Ny + iy] = 40.

    tests = [
        # (c_true, obs_indices, noise, reg, reg_w, label)
        (c_two_region, all_obs, 0.0, None, 0.0,
         'T1: Baseline (no noise, full obs)'),
        (c_two_region, all_obs, 0.01, None, 0.0,
         'T2: 1% noise, no reg'),
        (c_two_region, all_obs, 0.01, reg_fn, 1e-5,
         'T3: 1% noise + reg'),
        (c_two_region, all_obs, 0.03, reg_fn, 5e-6,
         'T4: 3% noise + reg'),
        (c_two_region, boundary_obs, 0.0, reg_fn, 5e-6,
         'T5: Sparse obs + reg'),
        (c_3region, all_obs, 0.0, None, 0.0,
         'T6: 3-region field'),
    ]

    results = []
    for c_true, obs_idx, noise, reg, reg_w, label in tests:
        res = run_inversion(c_true, obs_idx, noise, reg, reg_w, label)
        results.append(res)

    # ===================================================================
    # Summary table
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"{'Test':<40s} {'L2':>8s} {'Status':>10s} {'Obs':>5s} {'Time':>6s}")
    print(f"{'─' * 40} {'─' * 8} {'─' * 10} {'─' * 5} {'─' * 6}")
    for r in results:
        print(f"{r['label']:<40s} {r['l2_rel']:>7.2%} {r.get('status','?'):>10s} "
              f"{r.get('n_obs', '?'):>5} {r.get('time_s', 0):>5.0f}s")

    # ===================================================================
    # Plots
    # ===================================================================

    # Per-test inversion comparison
    for r in results:
        if 'c_final' not in r:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        c_all = onp.concatenate([r['c_true'], r['c_final']])
        vmin, vmax = c_all.min() * 0.8, c_all.max() * 1.2
        plot_c_field(r['c_true'], Nx, Ny, ax=axes[0], title='True c(x)',
                     vmin=vmin, vmax=vmax)
        plot_c_field(r['c_final'], Nx, Ny, ax=axes[1],
                     title=f"Inverted (L2={r['l2_rel']:.2%})",
                     vmin=vmin, vmax=vmax)
        # Line cut
        ax = axes[2]
        iy_mid = Ny // 2
        x_c = onp.linspace(Lx / (2 * Nx), Lx - Lx / (2 * Nx), Nx)
        ax.plot(x_c, r['c_true'].reshape(Nx, Ny)[:, iy_mid], 'k-', lw=2,
                label='True')
        ax.plot(x_c, r['c_final'].reshape(Nx, Ny)[:, iy_mid], 'ro-', ms=5,
                lw=1.5, label='Inverted')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('c [MPa]')
        ax.set_title('Line cut')
        ax.legend()
        ax.grid(True, alpha=0.3)

        safe_label = r['label'].split(':')[0].strip()
        fig.suptitle(f"{r['label']}  —  L2={r['l2_rel']:.2%}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{safe_label}_inversion.png'),
                    dpi=150)
        plt.close()

    # Combined summary bar chart
    labels = [r['label'].split(':')[0].strip() for r in results if 'l2_rel' in r]
    l2s = [r['l2_rel'] for r in results if 'l2_rel' in r]
    statuses = [r.get('status', '?') for r in results if 'l2_rel' in r]
    colors = ['#2ecc71' if s == 'PASS' else '#f39c12' if s == 'IMPROVED'
              else '#e74c3c' for s in statuses]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, [x * 100 for x in l2s], color=colors, edgecolor='k',
                  alpha=0.85)
    ax.axhline(10, ls='--', color='green', alpha=0.5, label='PASS threshold (10%)')
    ax.axhline(25, ls='--', color='orange', alpha=0.5, label='IMPROVED threshold (25%)')
    for bar, l2 in zip(bars, l2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{l2:.1%}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('L2 Relative Error [%]')
    ax.set_title('MC c(x) Inversion Robustness Tests')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'robustness_summary.png'), dpi=150)
    plt.close()

    # Save all results
    save_data = {}
    for r in results:
        key = r['label'].split(':')[0].strip()
        save_data[key] = {k: v for k, v in r.items()
                          if k not in ('c_true', 'c_final', 'history')}
        save_data[key]['l2_rel'] = r['l2_rel']
    save_results(OUT_DIR, save_data)

    print(f"\n  Results saved to {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
