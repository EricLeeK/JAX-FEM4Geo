#!/usr/bin/env python
"""
MC Two-Region Inversion with Incremental Loading.

Splits the total displacement into N_STEPS load steps. The first (N-1) steps
run as non-differentiable forward solves to build up internal variables
(stress/strain history). Only the final step is wrapped with ad_wrapper
for gradient computation. This improves return mapping accuracy while
keeping the adjoint method simple.
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
DISPLACEMENT = -0.03
N_STEPS = 5
PHI_DEG = 30.0
PSI_DEG = 15.0
E_FIXED = 70000.
C_LEFT, C_RIGHT = 30., 70.
C_INIT = 50.
NUM_CELLS = Nx * Ny
SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

OUT_DIR = os.path.join(RESULTS_DIR, 'mc_two_region_incremental')
os.makedirs(OUT_DIR, exist_ok=True)


def create_mesh_and_bc(disp):
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')

    def bottom(p): return np.isclose(p[1], 0., atol=1e-5)
    def top(p): return np.isclose(p[1], Ly, atol=1e-5)
    def corner(p): return np.logical_and(
        np.isclose(p[0], 0., atol=1e-5), np.isclose(p[1], 0., atol=1e-5))

    bc = [[bottom, top, corner], [1, 1, 0],
          [lambda p: 0., lambda p, _d=disp: _d, lambda p: 0.]]
    return mesh, bc


def incremental_solve(c_field, n_steps=N_STEPS):
    """Forward solve with incremental loading (non-differentiable)."""
    disps = onp.linspace(DISPLACEMENT / n_steps, DISPLACEMENT, n_steps)

    mesh, bc = create_mesh_and_bc(float(disps[0]))
    problem = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    problem.set_params(np.array(c_field))

    for i, d in enumerate(disps):
        bc[-1][1] = lambda p, _d=float(d): _d
        problem.fe.update_Dirichlet_boundary_conditions(bc)
        sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
        problem.update_stress_strain(sol_list[0])

    return sol_list[0], problem


def run_warmup_steps(problem, c_field, n_steps):
    """Run first (n_steps-1) load steps to build internal variables.

    Non-differentiable. Updates problem.internal_vars in-place.
    """
    if n_steps <= 1:
        return

    disps = onp.linspace(DISPLACEMENT / n_steps, DISPLACEMENT, n_steps)
    bc = problem.fe.dirichlet_bc_info_raw  # stash original

    for i in range(n_steps - 1):
        d = float(disps[i])
        bc_step = [bc[0], bc[1],
                   [lambda p: 0., lambda p, _d=d: _d, lambda p: 0.]]
        problem.fe.update_Dirichlet_boundary_conditions(bc_step)
        problem.set_params(c_field)
        sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
        problem.update_stress_strain(sol_list[0])

    # Restore final-step BCs
    d_final = float(disps[-1])
    bc_final = [bc[0], bc[1],
                [lambda p: 0., lambda p, _d=d_final: _d, lambda p: 0.]]
    problem.fe.update_Dirichlet_boundary_conditions(bc_final)


def main():
    print("=" * 70)
    print("MC Two-Region Inversion (Incremental Loading)")
    print(f"  Mesh: {Nx}×{Ny}, disp={DISPLACEMENT}, steps={N_STEPS}")
    print(f"  phi={PHI_DEG}°, c_true: left={C_LEFT}, right={C_RIGHT}")
    print("=" * 70)
    t0 = time.time()

    c_true = two_region_c_field(Nx, Ny, C_LEFT, C_RIGHT)

    # Generate observation with incremental loading
    print("\n  Generating synthetic observation (incremental)...")
    u_full, _ = incremental_solve(c_true, N_STEPS)
    u_obs = u_full
    obs_indices = onp.arange(u_full.shape[0])
    print(f"  ||u_obs|| = {float(np.linalg.norm(u_obs)):.6e}")

    # Set up inversion: final-step problem with ad_wrapper
    d_final = DISPLACEMENT
    mesh, bc = create_mesh_and_bc(d_final)
    inv_problem = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    # Stash BC info for warmup steps
    inv_problem.fe.dirichlet_bc_info_raw = bc

    fwd_pred = ad_wrapper(inv_problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    n_obs_dofs = u_obs.shape[0] * u_obs.shape[1]

    def loss_fn(log_c):
        c_field = log_to_c(log_c)
        # Reset internal vars and run warmup steps
        nc, nq = len(inv_problem.fe.cells), inv_problem.fe.num_quads
        inv_problem.sigmas_old = np.zeros((nc, nq, 3, 3))
        inv_problem.epsilons_old = np.zeros((nc, nq, 3, 3))
        inv_problem.internal_vars[0] = inv_problem.sigmas_old
        inv_problem.internal_vars[1] = inv_problem.epsilons_old

        run_warmup_steps(inv_problem, jax.lax.stop_gradient(c_field), N_STEPS)

        # Final step: differentiable
        sol_list = fwd_pred(c_field)
        u_pred = sol_list[0][obs_indices]
        return np.sum((u_pred - u_obs) ** 2) / n_obs_dofs

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    log_c_init = onp.array(c_to_log(uniform_c_field(Nx, Ny, C_INIT)),
                           dtype=onp.float64)

    print("\n  Warming up...")
    l0, g0 = value_and_grad_fn(np.array(log_c_init))
    print(f"  Initial loss: {float(l0):.6e}, ||grad||: {float(np.linalg.norm(g0)):.6e}")

    if float(np.linalg.norm(g0)) < 1e-20:
        print("  ⚠ Zero gradient")
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
                raise ValueError("NaN")
            last_good[0] = lf
            last_good[1] = gn
        except Exception:
            lf = last_good[0] * 10.
            gn = onp.zeros_like(last_good[1])
        history.append(lf)
        step = len(history)
        if step <= 3 or step % 10 == 0:
            c_cur = onp.exp(x)
            print(f"    Eval {step:4d}: loss={lf:.6e}  "
                  f"c=[{c_cur.min():.1f}, {c_cur.max():.1f}]")
        return lf, gn

    print(f"\n  Running L-BFGS-B (maxiter=150)...")
    result = scipy_minimize(
        objective, x0=log_c_init, method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': 150, 'maxfun': 400, 'ftol': 1e-20, 'gtol': 1e-12},
    )

    c_final = onp.exp(result.x)
    c_true_np = onp.array(c_true)
    rel_err = onp.abs(c_final - c_true_np) / c_true_np
    l2_rel = float(onp.sqrt(onp.mean(rel_err ** 2)))

    left_mask = c_true_np < 50.
    right_mask = ~left_mask
    left_err = float(onp.mean(onp.abs(c_final[left_mask] - C_LEFT) / C_LEFT))
    right_err = float(onp.mean(onp.abs(c_final[right_mask] - C_RIGHT) / C_RIGHT))
    t_total = time.time() - t0

    print(f"\n  Result:")
    print(f"    Converged: {result.success}")
    print(f"    c range: [{c_final.min():.2f}, {c_final.max():.2f}]")
    print(f"    L2 rel error: {l2_rel:.4f} ({l2_rel:.2%})")
    print(f"    Left (c={C_LEFT}) err: {left_err:.4%}")
    print(f"    Right (c={C_RIGHT}) err: {right_err:.4%}")
    print(f"    Time: {t_total:.1f}s")

    status = "PASS" if l2_rel < 0.10 else "IMPROVED" if l2_rel < 0.25 else "NEEDS WORK"
    print(f"    Status: {status}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmin, vmax = C_LEFT * 0.8, C_RIGHT * 1.2
    plot_c_field(c_true, Nx, Ny, ax=axes[0], title='True c(x)',
                 vmin=vmin, vmax=vmax)
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

    fig.suptitle(f'MC Incremental ({N_STEPS} steps): L2={l2_rel:.2%}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mc_incremental_inversion.png'), dpi=150)
    plt.close()

    if history:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.semilogy(history, 'b-', lw=1.5)
        ax2.set_xlabel('Evaluation')
        ax2.set_ylabel('Loss')
        ax2.set_title('Convergence')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'mc_incremental_convergence.png'),
                    dpi=150)
        plt.close()

    save_results(OUT_DIR, {
        'c_true': c_true_np, 'c_final': c_final,
        'l2_rel_error': l2_rel, 'left_err': left_err, 'right_err': right_err,
        'n_steps': N_STEPS, 'loss_history': history,
        'nit': result.nit, 'time_s': t_total, 'status': status,
    })
    print(f"  Results saved to {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
