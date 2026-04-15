#!/usr/bin/env python
"""
MC Two-Region Inversion: Recover heterogeneous c(x) field using Mohr-Coulomb.

Setup:
- 10×10 QUAD4 mesh (100 elements), 10×10 domain
- True c: left half = 30 MPa, right half = 70 MPa
- Fixed phi = 30°, E = 70000 MPa
- Displacement-controlled compression: u_y = -0.03 on top
- Observation: full-field displacement, no noise
- Optimizer: L-BFGS-B in log-c space
- Initial guess: uniform c = 50 MPa
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time

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
from src.models.mohr_coulomb_2d import MohrCoulombPlasticity2D

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

OUT_DIR = os.path.join(RESULTS_DIR, 'mc_two_region')
os.makedirs(OUT_DIR, exist_ok=True)


def generate_obs_displacement(c_field):
    """Forward solve with given c(x) field, return full displacement."""
    mesh, bc_info = create_2d_mesh_and_bc(DISPLACEMENT, Lx, Ly, Nx, Ny)
    problem = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    problem.set_params(np.array(c_field))
    sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
    return sol_list[0], mesh, bc_info


def main():
    print("=" * 70)
    print("MC Two-Region Inversion: c(x) field recovery")
    print(f"  Mesh: {Nx}×{Ny}, disp={DISPLACEMENT}, phi={PHI_DEG}°")
    print(f"  c_true: left={C_LEFT}, right={C_RIGHT}")
    print("=" * 70)
    t0 = time.time()

    c_true = two_region_c_field(Nx, Ny, C_LEFT, C_RIGHT)

    # Generate synthetic observation
    print("\n  Generating synthetic observation...")
    u_full, mesh, bc_info = generate_obs_displacement(c_true)
    u_obs = u_full
    obs_indices = onp.arange(u_full.shape[0])
    print(f"  ||u_obs|| = {float(np.linalg.norm(u_obs)):.6e}")
    print(f"  u_y range: [{float(np.min(u_obs[:, 1])):.6f}, "
          f"{float(np.max(u_obs[:, 1])):.6f}]")

    # Set up inversion problem
    mesh2, bc_info2 = create_2d_mesh_and_bc(DISPLACEMENT, Lx, Ly, Nx, Ny)
    inv_problem = InversionHeterogeneousMC2D(
        mesh2, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info2,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    fwd_pred = ad_wrapper(inv_problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    loss_fn = make_heterogeneous_loss(fwd_pred, u_obs, obs_indices)
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    log_c_init = onp.array(c_to_log(uniform_c_field(Nx, Ny, C_INIT)),
                           dtype=onp.float64)

    # Warmup
    print("\n  Warming up (first forward+backward)...")
    l0, g0 = value_and_grad_fn(np.array(log_c_init))
    print(f"  Initial loss: {float(l0):.6e}, ||grad||: {float(np.linalg.norm(g0)):.6e}")

    if float(np.linalg.norm(g0)) < 1e-20:
        print("\n  ⚠ Zero gradient — material is elastic, c has no effect.")
        print("  Try larger displacement or smaller c values.")
        return

    # Optimize
    log_c_lo = float(onp.log(5.))
    log_c_hi = float(onp.log(500.))
    bounds = [(log_c_lo, log_c_hi)] * NUM_CELLS

    history = []

    last_good = [float(l0), onp.array(g0, dtype=onp.float64)]

    def objective(x):
        log_c = np.array(x)
        try:
            loss, grad = value_and_grad_fn(log_c)
            lf = float(loss)
            gn = onp.array(grad, dtype=onp.float64)
            if not (onp.isfinite(lf) and onp.all(onp.isfinite(gn))):
                raise ValueError("NaN in loss or grad")
            last_good[0] = lf
            last_good[1] = gn
        except Exception:
            lf = last_good[0] * 10.
            gn = onp.zeros_like(last_good[1])
        history.append(lf)
        step = len(history)
        if step <= 3 or step % 20 == 0:
            c_cur = onp.exp(x)
            print(f"    Eval {step:4d}: loss={lf:.6e}  "
                  f"c=[{c_cur.min():.1f}, {c_cur.max():.1f}]")
        return lf, gn

    print(f"\n  Running L-BFGS-B (maxiter=200)...")
    result = scipy_minimize(
        objective, x0=log_c_init, method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-20, 'gtol': 1e-12},
    )

    c_final = onp.exp(result.x)
    c_true_np = onp.array(c_true)
    rel_err = onp.abs(c_final - c_true_np) / c_true_np
    l2_rel = float(onp.sqrt(onp.mean(rel_err ** 2)))
    mean_rel = float(onp.mean(rel_err))

    left_mask = c_true_np < 50.
    right_mask = ~left_mask
    left_err = float(onp.mean(onp.abs(c_final[left_mask] - C_LEFT) / C_LEFT))
    right_err = float(onp.mean(onp.abs(c_final[right_mask] - C_RIGHT) / C_RIGHT))

    t_total = time.time() - t0

    print(f"\n  Result:")
    print(f"    Converged: {result.success}  ({result.message})")
    print(f"    Iterations: {result.nit}, Evals: {result.nfev}")
    print(f"    c range: [{c_final.min():.2f}, {c_final.max():.2f}]")
    print(f"    L2 rel error: {l2_rel:.4f} ({l2_rel:.2%})")
    print(f"    Mean rel error: {mean_rel:.4f}")
    print(f"    Left region (c={C_LEFT}) err: {left_err:.4%}")
    print(f"    Right region (c={C_RIGHT}) err: {right_err:.4%}")
    print(f"    Time: {t_total:.1f}s")

    status = "PASS" if l2_rel < 0.10 else "IMPROVED" if l2_rel < 0.25 else "NEEDS WORK"
    print(f"    Status: {status}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmin = C_LEFT * 0.8
    vmax = C_RIGHT * 1.2
    plot_c_field(c_true, Nx, Ny, ax=axes[0], title='True c(x)', vmin=vmin, vmax=vmax)
    plot_c_field(c_final, Nx, Ny, ax=axes[1],
                 title=f'Inverted (L2={l2_rel:.2%})', vmin=vmin, vmax=vmax)

    ax = axes[2]
    iy_mid = Ny // 2
    x_c = onp.linspace(Lx / (2 * Nx), Lx - Lx / (2 * Nx), Nx)
    ax.plot(x_c, c_true_np.reshape(Nx, Ny)[:, iy_mid], 'k-', lw=2, label='True')
    ax.plot(x_c, c_final.reshape(Nx, Ny)[:, iy_mid], 'ro-', ms=5, lw=1.5,
            label='Inverted')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('c [MPa]')
    ax.set_title(f'Line cut y={Ly / 2:.0f}m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'MC c(x) Inversion: phi={PHI_DEG}°, disp={DISPLACEMENT}, '
                 f'L2={l2_rel:.2%}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mc_two_region_inversion.png'), dpi=150)
    plt.close()

    # Convergence history
    if history:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.semilogy(history, 'b-', lw=1.5)
        ax2.set_xlabel('Evaluation')
        ax2.set_ylabel('Loss')
        ax2.set_title('Convergence history')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'mc_convergence.png'), dpi=150)
        plt.close()

    save_results(OUT_DIR, {
        'c_true': c_true_np,
        'c_final': c_final,
        'l2_rel_error': l2_rel,
        'left_err': left_err,
        'right_err': right_err,
        'loss_history': history,
        'nit': result.nit,
        'nfev': result.nfev,
        'time_s': t_total,
        'status': status,
    })

    print(f"\n  Results saved to {OUT_DIR}")
    print("=" * 70)


from scipy.optimize import minimize as scipy_minimize

if __name__ == "__main__":
    main()
