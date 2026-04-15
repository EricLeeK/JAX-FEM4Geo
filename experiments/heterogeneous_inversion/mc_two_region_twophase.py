#!/usr/bin/env python
"""
MC Two-Region Inversion with Two-Phase Optimization.

Phase 1: Incremental loading (final-step-only loss with stop_gradient
         warmup steps). Accurately recovers the left/plastic region
         but overestimates the right/elastic region.

Phase 2: Single-step solve (no history) starting from Phase 1 result.
         Corrects the right region while preserving the left.

This exploits the complementary strengths of each approach without
needing to change JAX-FEM's ad_wrapper.
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

OUT_DIR = os.path.join(RESULTS_DIR, 'mc_two_region_twophase')
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


# ---------------------------------------------------------------------------
# Phase 1: Incremental loading (from mc_two_region_incremental.py)
# ---------------------------------------------------------------------------

def incremental_solve(c_field, n_steps=N_STEPS):
    disps = onp.linspace(DISPLACEMENT / n_steps, DISPLACEMENT, n_steps)
    mesh, bc = create_mesh_and_bc(float(disps[0]))
    problem = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    problem.set_params(np.array(c_field))

    for d in disps:
        bc[-1][1] = lambda p, _d=float(d): _d
        problem.fe.update_Dirichlet_boundary_conditions(bc)
        sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
        problem.update_stress_strain(sol_list[0])

    return sol_list[0], problem


def run_warmup_steps(problem, c_field, n_steps):
    if n_steps <= 1:
        return
    disps = onp.linspace(DISPLACEMENT / n_steps, DISPLACEMENT, n_steps)
    bc = problem.fe.dirichlet_bc_info_raw

    for i in range(n_steps - 1):
        d = float(disps[i])
        bc_step = [bc[0], bc[1],
                   [lambda p: 0., lambda p, _d=d: _d, lambda p: 0.]]
        problem.fe.update_Dirichlet_boundary_conditions(bc_step)
        problem.set_params(c_field)
        sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
        problem.update_stress_strain(sol_list[0])

    d_final = float(disps[-1])
    bc_final = [bc[0], bc[1],
                [lambda p: 0., lambda p, _d=d_final: _d, lambda p: 0.]]
    problem.fe.update_Dirichlet_boundary_conditions(bc_final)


def run_optimizer(objective_fn, x0, bounds, maxiter, maxfun, label):
    """Run L-BFGS-B and return result + history."""
    history = []
    last_good = [None, None]

    def objective(x):
        log_c = np.array(x)
        try:
            loss, grad = objective_fn(log_c)
            lf = float(loss)
            gn = onp.array(grad, dtype=onp.float64)
            if not (onp.isfinite(lf) and onp.all(onp.isfinite(gn))):
                raise ValueError("NaN")
            last_good[0] = lf
            last_good[1] = gn
        except Exception:
            lf = last_good[0] * 10. if last_good[0] is not None else 1e10
            gn = onp.zeros_like(x)
        history.append(lf)
        step = len(history)
        if step <= 3 or step % 10 == 0:
            c_cur = onp.exp(x)
            print(f"    [{label}] Eval {step:4d}: loss={lf:.6e}  "
                  f"c=[{c_cur.min():.1f}, {c_cur.max():.1f}]")
        return lf, gn

    # Warmup
    l0, g0 = objective_fn(np.array(x0))
    last_good[0] = float(l0)
    last_good[1] = onp.array(g0, dtype=onp.float64)
    print(f"  [{label}] Initial loss: {float(l0):.6e}, "
          f"||grad||: {float(np.linalg.norm(g0)):.6e}")

    if float(np.linalg.norm(g0)) < 1e-20:
        print(f"  [{label}] ⚠ Zero gradient")
        return None, history

    result = scipy_minimize(
        objective, x0=x0, method='L-BFGS-B', jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'maxfun': maxfun,
                 'ftol': 1e-20, 'gtol': 1e-12},
    )
    return result, history


def main():
    print("=" * 70)
    print("MC Two-Region Inversion (Two-Phase Optimization)")
    print(f"  Mesh: {Nx}×{Ny}, disp={DISPLACEMENT}, steps={N_STEPS}")
    print(f"  phi={PHI_DEG}°, c_true: left={C_LEFT}, right={C_RIGHT}")
    print("=" * 70)
    t0 = time.time()

    c_true = two_region_c_field(Nx, Ny, C_LEFT, C_RIGHT)
    log_c_lo = float(onp.log(5.))
    log_c_hi = float(onp.log(500.))
    bounds = [(log_c_lo, log_c_hi)] * NUM_CELLS

    # ===================================================================
    # Phase 1: Incremental loading — good for left/plastic region
    # ===================================================================
    print("\n" + "=" * 50)
    print("Phase 1: Incremental loading (final-step loss)")
    print("=" * 50)

    # Generate incremental observation
    print("  Generating synthetic observation (incremental)...")
    u_full, _ = incremental_solve(c_true, N_STEPS)
    u_obs = u_full
    obs_indices = onp.arange(u_full.shape[0])
    print(f"  ||u_obs|| = {float(np.linalg.norm(u_obs)):.6e}")

    # Set up incremental inversion problem
    d_final = DISPLACEMENT
    mesh1, bc1 = create_mesh_and_bc(d_final)
    inv_problem1 = InversionHeterogeneousMC2D(
        mesh1, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc1,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    inv_problem1.fe.dirichlet_bc_info_raw = bc1
    fwd_pred1 = ad_wrapper(inv_problem1, solver_options=SOLVER_OPTIONS,
                           adjoint_solver_options=SOLVER_OPTIONS)

    nc, nq = len(inv_problem1.fe.cells), inv_problem1.fe.num_quads
    n_obs_dofs = u_obs.shape[0] * u_obs.shape[1]

    def loss_fn_phase1(log_c):
        c_field = log_to_c(log_c)
        inv_problem1.sigmas_old = np.zeros((nc, nq, 3, 3))
        inv_problem1.epsilons_old = np.zeros((nc, nq, 3, 3))
        inv_problem1.internal_vars[0] = inv_problem1.sigmas_old
        inv_problem1.internal_vars[1] = inv_problem1.epsilons_old
        run_warmup_steps(inv_problem1, jax.lax.stop_gradient(c_field), N_STEPS)
        sol_list = fwd_pred1(c_field)
        u_pred = sol_list[0][obs_indices]
        return np.sum((u_pred - u_obs) ** 2) / n_obs_dofs

    vg_fn1 = jax.value_and_grad(loss_fn_phase1)

    log_c_init = onp.array(c_to_log(uniform_c_field(Nx, Ny, C_INIT)),
                           dtype=onp.float64)

    result1, history1 = run_optimizer(vg_fn1, log_c_init, bounds,
                                      maxiter=100, maxfun=300, label='P1')

    c_phase1 = onp.exp(result1.x)
    c_true_np = onp.array(c_true)
    rel1 = onp.abs(c_phase1 - c_true_np) / c_true_np
    l2_phase1 = float(onp.sqrt(onp.mean(rel1 ** 2)))
    left_mask = c_true_np < 50.
    right_mask = ~left_mask
    print(f"\n  Phase 1 result: L2={l2_phase1:.2%}")
    print(f"    c range: [{c_phase1.min():.1f}, {c_phase1.max():.1f}]")
    print(f"    Left err: {float(onp.mean(onp.abs(c_phase1[left_mask]-C_LEFT)/C_LEFT)):.2%}")
    print(f"    Right err: {float(onp.mean(onp.abs(c_phase1[right_mask]-C_RIGHT)/C_RIGHT)):.2%}")

    # ===================================================================
    # Phase 2: Single-step solve — corrects right/elastic region
    # ===================================================================
    print("\n" + "=" * 50)
    print("Phase 2: Single-step (correcting right region)")
    print("=" * 50)

    mesh2, bc2 = create_2d_mesh_and_bc(DISPLACEMENT, Lx, Ly, Nx, Ny)
    inv_problem2 = InversionHeterogeneousMC2D(
        mesh2, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc2,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    fwd_pred2 = ad_wrapper(inv_problem2, solver_options=SOLVER_OPTIONS,
                           adjoint_solver_options=SOLVER_OPTIONS)

    # Single-step observation (one-shot solve with true params)
    print("  Generating single-step observation...")
    u_ss, _, _ = _generate_single_step_obs(c_true)
    print(f"  ||u_obs_ss|| = {float(np.linalg.norm(u_ss)):.6e}")

    loss_fn_phase2 = make_heterogeneous_loss(fwd_pred2, u_ss, obs_indices)
    vg_fn2 = jax.value_and_grad(loss_fn_phase2)

    # Start from Phase 1 result
    log_c_phase1 = onp.array(result1.x, dtype=onp.float64)
    result2, history2 = run_optimizer(vg_fn2, log_c_phase1, bounds,
                                      maxiter=150, maxfun=400, label='P2')

    c_final = onp.exp(result2.x)
    rel_err = onp.abs(c_final - c_true_np) / c_true_np
    l2_rel = float(onp.sqrt(onp.mean(rel_err ** 2)))

    left_err = float(onp.mean(onp.abs(c_final[left_mask] - C_LEFT) / C_LEFT))
    right_err = float(onp.mean(onp.abs(c_final[right_mask] - C_RIGHT) / C_RIGHT))
    t_total = time.time() - t0

    print(f"\n  Final Result:")
    print(f"    c range: [{c_final.min():.2f}, {c_final.max():.2f}]")
    print(f"    L2 rel error: {l2_rel:.4f} ({l2_rel:.2%})")
    print(f"    Left (c={C_LEFT}) err: {left_err:.4%}")
    print(f"    Right (c={C_RIGHT}) err: {right_err:.4%}")
    print(f"    Time: {t_total:.1f}s")

    status = "PASS" if l2_rel < 0.10 else "IMPROVED" if l2_rel < 0.25 else "NEEDS WORK"
    print(f"    Status: {status}")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmin, vmax = C_LEFT * 0.8, C_RIGHT * 1.2
    plot_c_field(c_true, Nx, Ny, ax=axes[0], title='True c(x)',
                 vmin=vmin, vmax=vmax)
    plot_c_field(c_phase1, Nx, Ny, ax=axes[1],
                 title=f'Phase 1 (L2={l2_phase1:.2%})', vmin=vmin, vmax=vmax)
    plot_c_field(c_final, Nx, Ny, ax=axes[2],
                 title=f'Phase 2 (L2={l2_rel:.2%})', vmin=vmin, vmax=vmax)

    ax = axes[3]
    iy_mid = Ny // 2
    x_c = onp.linspace(Lx / (2 * Nx), Lx - Lx / (2 * Nx), Nx)
    ax.plot(x_c, c_true_np.reshape(Nx, Ny)[:, iy_mid], 'k-', lw=2, label='True')
    ax.plot(x_c, c_phase1.reshape(Nx, Ny)[:, iy_mid], 'b^--', ms=5, lw=1,
            label='Phase 1')
    ax.plot(x_c, c_final.reshape(Nx, Ny)[:, iy_mid], 'ro-', ms=5, lw=1.5,
            label='Phase 2')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('c [MPa]')
    ax.set_title(f'Line cut y={Ly / 2:.0f}m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'MC Two-Phase: L2={l2_rel:.2%} '
                 f'(P1: incremental→P2: single-step)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mc_twophase_inversion.png'), dpi=150)
    plt.close()

    # Convergence
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.semilogy(history1, 'b-', lw=1.5, label='Phase 1 (incremental)')
    offset = len(history1)
    ax2.semilogy(range(offset, offset + len(history2)), history2,
                 'r-', lw=1.5, label='Phase 2 (single-step)')
    ax2.axvline(offset, color='gray', ls='--', alpha=0.5)
    ax2.set_xlabel('Evaluation')
    ax2.set_ylabel('Loss')
    ax2.set_title('Two-Phase Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mc_twophase_convergence.png'), dpi=150)
    plt.close()

    save_results(OUT_DIR, {
        'c_true': c_true_np, 'c_phase1': c_phase1, 'c_final': c_final,
        'l2_phase1': l2_phase1, 'l2_rel_error': l2_rel,
        'left_err': left_err, 'right_err': right_err,
        'n_steps': N_STEPS, 'history_phase1': history1,
        'history_phase2': history2,
        'time_s': t_total, 'status': status,
    })
    print(f"  Results saved to {OUT_DIR}")
    print("=" * 70)


def _generate_single_step_obs(c_field):
    """Single-step forward solve to generate observation."""
    mesh, bc_info = create_2d_mesh_and_bc(DISPLACEMENT, Lx, Ly, Nx, Ny)
    problem = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    problem.set_params(np.array(c_field))
    sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
    return sol_list[0], mesh, bc_info


if __name__ == "__main__":
    main()
