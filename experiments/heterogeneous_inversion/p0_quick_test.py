#!/usr/bin/env python
"""
P0 Quick Test: Validate BC fix + elastic regime (high k) on H5 two-region.

Changes from original H5:
1. Bottom BC: u_y=0 only + corner u_x=0 (no full x-clamp)
2. k=500 (ensure elastic regime, far from yield)
3. Fewer iterations for quick turnaround
"""

import jax
import jax.numpy as np
import numpy as onp
import os, sys, time

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
    smoothness_regularizer,
    two_region_E_field,
    uniform_E_field,
    log_to_E, E_to_log,
    plot_E_field,
    RESULTS_DIR,
)

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper
from scipy.optimize import minimize as scipy_minimize

Nx, Ny = 20, 20
Lx, Ly = 10., 10.
TRACTION = -50.0
K_FIXED = 500.0  # HIGH k to stay elastic
E_LEFT, E_RIGHT = 50000., 90000.
E_INIT = 70000.
NUM_CELLS = Nx * Ny
SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

OUT_DIR = os.path.join(RESULTS_DIR, 'h5_two_region')


def main():
    print("=" * 70)
    print("P0 QUICK TEST: BC fix + elastic regime (k=500)")
    print("=" * 70)
    t0 = time.time()

    E_true = two_region_E_field(Nx, Ny, E_LEFT, E_RIGHT)

    # Generate observation with high k
    obs_data = generate_synthetic_observation(
        E_true, K_FIXED,
        traction=TRACTION, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        noise_level=0.0, solver_options=SOLVER_OPTIONS,
    )
    u_obs = obs_data['u_obs']
    obs_indices = obs_data['obs_indices']
    print(f"  ||u_obs|| = {float(np.linalg.norm(u_obs)):.6e}")

    # Set up inversion
    mesh, bc_info, loc_fns, _ = create_2d_mesh_traction_bc(TRACTION, Lx, Ly, Nx, Ny)
    inv_problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info, location_fns=loc_fns,
        E=E_INIT, nu=0.3, alpha=0.3, k=K_FIXED,
        traction_value=TRACTION,
    )
    fwd_pred = ad_wrapper(inv_problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    # No regularization first
    loss_fn = make_heterogeneous_loss(fwd_pred, u_obs, obs_indices)
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    log_E_init = onp.array(E_to_log(uniform_E_field(Nx, Ny, E_INIT)), dtype=onp.float64)

    # Warmup
    print("\n  Warming up...")
    l0, g0 = value_and_grad_fn(np.array(log_E_init))
    print(f"  Initial loss: {float(l0):.6e}, ||grad||: {float(np.linalg.norm(g0)):.6e}")

    # Optimize
    log_E_lo = float(onp.log(5000.))
    log_E_hi = float(onp.log(300000.))
    bounds = [(log_E_lo, log_E_hi)] * NUM_CELLS

    history = []

    def objective(x):
        log_E = np.array(x)
        loss, grad = value_and_grad_fn(log_E)
        lf = float(loss)
        gn = onp.array(grad, dtype=onp.float64)
        history.append(lf)
        step = len(history)
        if step <= 3 or step % 20 == 0:
            E_cur = onp.exp(x)
            print(f"    Eval {step}: loss={lf:.6e}  E=[{E_cur.min():.0f}, {E_cur.max():.0f}]")
        return lf, gn

    print("\n  Running L-BFGS-B (maxiter=200)...")
    result = scipy_minimize(
        objective, x0=log_E_init, method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-20, 'gtol': 1e-12},
    )

    E_final = onp.exp(result.x)
    E_true_np = onp.array(E_true)
    rel_err = onp.abs(E_final - E_true_np) / E_true_np
    l2_rel = float(onp.sqrt(onp.mean(rel_err ** 2)))
    mean_rel = float(onp.mean(rel_err))

    left_mask = E_true_np < 70000
    right_mask = ~left_mask
    left_err = float(onp.mean(onp.abs(E_final[left_mask] - E_LEFT) / E_LEFT))
    right_err = float(onp.mean(onp.abs(E_final[right_mask] - E_RIGHT) / E_RIGHT))

    t_total = time.time() - t0

    print(f"\n  Result:")
    print(f"    Converged: {result.success}  ({result.message})")
    print(f"    Iterations: {result.nit}, Evals: {result.nfev}")
    print(f"    E range: [{E_final.min():.0f}, {E_final.max():.0f}]")
    print(f"    L2 rel error: {l2_rel:.4f} ({l2_rel:.2%})")
    print(f"    Mean rel error: {mean_rel:.4f}")
    print(f"    Left region err: {left_err:.4%}")
    print(f"    Right region err: {right_err:.4%}")
    print(f"    Time: {t_total:.1f}s")

    status = "PASS" if l2_rel < 0.05 else "IMPROVED" if l2_rel < 0.10 else "NEEDS WORK"
    print(f"    Status: {status}")

    # Quick plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmin = E_LEFT * 0.9
    vmax = E_RIGHT * 1.1
    plot_E_field(E_true, Nx, Ny, ax=axes[0], title='True E', vmin=vmin, vmax=vmax)
    plot_E_field(E_final, Nx, Ny, ax=axes[1],
                 title=f'Inverted (L2={l2_rel:.2%})', vmin=vmin, vmax=vmax)

    ax = axes[2]
    iy_mid = Ny // 2
    x_c = onp.linspace(Lx/(2*Nx), Lx - Lx/(2*Nx), Nx)
    ax.plot(x_c, E_true_np.reshape(Nx, Ny)[:, iy_mid], 'k-', lw=2, label='True')
    ax.plot(x_c, E_final.reshape(Nx, Ny)[:, iy_mid], 'bo-', ms=4, lw=1.5, label='Inverted')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('E [MPa]')
    ax.set_title(f'Line cut y={Ly/2:.0f}m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'P0 Quick Test: k={K_FIXED}, BC=corner-only, L2={l2_rel:.2%}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'P0_快速测试_弹性域.png'), dpi=150)
    plt.close()
    print(f"\n  Plot saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
