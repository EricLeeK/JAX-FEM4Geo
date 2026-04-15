#!/usr/bin/env python
"""
MC Joint c(x)+φ(x) Inversion.

Simultaneously recovers heterogeneous cohesion c(x) and friction angle φ(x)
fields from displacement observations.

Setup:
- 10×10 QUAD4 mesh, displacement-controlled compression
- True fields: c left=30/right=70, φ left=25°/right=35°
- Fixed: E=70000, ν=0.3, ψ=φ/2 (non-associative)
- Parameterization: log(c) + log(tan(φ)) for positivity + (0,π/2) range
- Two-phase optimization (incremental → single-step)
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
    InversionJointMC2D,
    two_region_c_field,
    uniform_c_field,
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
E_FIXED = 70000.

# True parameter fields
C_LEFT, C_RIGHT = 30., 70.
PHI_LEFT_DEG, PHI_RIGHT_DEG = 25., 35.

# Initial guesses
C_INIT = 50.
PHI_INIT_DEG = 30.

# psi = phi / 2 (non-associative)
PSI_RATIO = 0.5

NUM_CELLS = Nx * Ny
SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

OUT_DIR = os.path.join(RESULTS_DIR, 'mc_joint_inversion')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Parameterization: log(c) + log(tan(phi)) → physical (c, phi)
# ---------------------------------------------------------------------------

def params_to_physical(x):
    """Convert optimization params to physical (c_field, phi_field).

    x[:nc] = log(c),  x[nc:] = log(tan(phi))
    """
    nc = NUM_CELLS
    c_field = np.exp(x[:nc])
    phi_field = np.arctan(np.exp(x[nc:]))
    return c_field, phi_field


def physical_to_params(c_field, phi_field):
    """Convert physical fields to optimization params."""
    log_c = onp.log(c_field)
    log_tan_phi = onp.log(onp.tan(phi_field))
    return onp.concatenate([log_c, log_tan_phi])


# ---------------------------------------------------------------------------
# Field generators
# ---------------------------------------------------------------------------

def two_region_phi_field(Nx, Ny, phi_left_rad, phi_right_rad):
    """Two-region friction angle field (radians)."""
    phi = onp.full(Nx * Ny, phi_right_rad)
    for ix in range(Nx // 2):
        for iy in range(Ny):
            phi[ix * Ny + iy] = phi_left_rad
    return phi


def uniform_phi_field(Nx, Ny, phi_rad):
    return onp.full(Nx * Ny, phi_rad)


# ---------------------------------------------------------------------------
# Mesh & BC
# ---------------------------------------------------------------------------

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
# Forward solve helpers
# ---------------------------------------------------------------------------

def make_stacked_params(c_field, phi_field):
    """Stack c and phi fields into ad_wrapper params format."""
    return np.concatenate([np.array(c_field), np.array(phi_field)])


def incremental_solve(c_field, phi_field, n_steps=N_STEPS):
    """Non-differentiable incremental forward solve."""
    disps = onp.linspace(DISPLACEMENT / n_steps, DISPLACEMENT, n_steps)
    mesh, bc = create_mesh_and_bc(float(disps[0]))
    problem = InversionJointMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_INIT_DEG,
        psi_deg=PHI_INIT_DEG * PSI_RATIO,
    )
    params = make_stacked_params(c_field, phi_field)
    problem.set_params(params)

    for d in disps:
        bc[-1][1] = lambda p, _d=float(d): _d
        problem.fe.update_Dirichlet_boundary_conditions(bc)
        sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
        problem.update_stress_strain(sol_list[0])

    return sol_list[0], problem


def run_warmup_steps(problem, stacked_params, n_steps):
    if n_steps <= 1:
        return
    disps = onp.linspace(DISPLACEMENT / n_steps, DISPLACEMENT, n_steps)
    bc = problem.fe.dirichlet_bc_info_raw

    for i in range(n_steps - 1):
        d = float(disps[i])
        bc_step = [bc[0], bc[1],
                   [lambda p: 0., lambda p, _d=d: _d, lambda p: 0.]]
        problem.fe.update_Dirichlet_boundary_conditions(bc_step)
        problem.set_params(stacked_params)
        sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
        problem.update_stress_strain(sol_list[0])

    d_final = float(disps[-1])
    bc_final = [bc[0], bc[1],
                [lambda p: 0., lambda p, _d=d_final: _d, lambda p: 0.]]
    problem.fe.update_Dirichlet_boundary_conditions(bc_final)


# ---------------------------------------------------------------------------
# Optimizer wrapper
# ---------------------------------------------------------------------------

def run_optimizer(objective_fn, x0, bounds, maxiter, maxfun, label):
    history = []
    last_good = [None, None]

    def objective(x):
        xj = np.array(x)
        try:
            loss, grad = objective_fn(xj)
            lf = float(loss)
            gn = onp.array(grad, dtype=onp.float64)
            if not (onp.isfinite(lf) and onp.all(onp.isfinite(gn))):
                raise ValueError("NaN")
            last_good[0] = lf
            last_good[1] = gn
        except Exception as e:
            lf = last_good[0] * 10. if last_good[0] is not None else 1e10
            gn = onp.zeros_like(x)
        history.append(lf)
        step = len(history)
        if step <= 3 or step % 10 == 0:
            c_cur, phi_cur = params_to_physical(np.array(x))
            c_np = onp.array(c_cur)
            phi_np = onp.degrees(onp.array(phi_cur))
            print(f"    [{label}] Eval {step:4d}: loss={lf:.6e}  "
                  f"c=[{c_np.min():.1f},{c_np.max():.1f}]  "
                  f"φ=[{phi_np.min():.1f}°,{phi_np.max():.1f}°]")
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
    print("MC Joint c(x)+φ(x) Inversion (Two-Phase)")
    print(f"  Mesh: {Nx}×{Ny}, disp={DISPLACEMENT}, steps={N_STEPS}")
    print(f"  c_true: [{C_LEFT},{C_RIGHT}], φ_true: [{PHI_LEFT_DEG}°,{PHI_RIGHT_DEG}°]")
    print("=" * 70)
    t0 = time.time()

    # True fields
    c_true = two_region_c_field(Nx, Ny, C_LEFT, C_RIGHT)
    phi_true = two_region_phi_field(Nx, Ny,
                                    onp.radians(PHI_LEFT_DEG),
                                    onp.radians(PHI_RIGHT_DEG))

    # Bounds in optimization space
    log_c_lo, log_c_hi = float(onp.log(5.)), float(onp.log(500.))
    log_tan_phi_lo = float(onp.log(onp.tan(onp.radians(5.))))
    log_tan_phi_hi = float(onp.log(onp.tan(onp.radians(60.))))
    bounds = ([(log_c_lo, log_c_hi)] * NUM_CELLS +
              [(log_tan_phi_lo, log_tan_phi_hi)] * NUM_CELLS)

    # Initial guess
    c_init = uniform_c_field(Nx, Ny, C_INIT)
    phi_init = uniform_phi_field(Nx, Ny, onp.radians(PHI_INIT_DEG))
    x_init = onp.array(physical_to_params(c_init, phi_init), dtype=onp.float64)

    obs_indices = onp.arange((Nx + 1) * (Ny + 1))  # all nodes

    # ===================================================================
    # Phase 1: Incremental loading
    # ===================================================================
    print("\n" + "=" * 50)
    print("Phase 1: Incremental loading")
    print("=" * 50)

    print("  Generating observation (incremental)...")
    u_obs, _ = incremental_solve(c_true, phi_true, N_STEPS)
    n_obs_dofs = u_obs.shape[0] * u_obs.shape[1]
    print(f"  ||u_obs|| = {float(np.linalg.norm(u_obs)):.6e}")

    mesh1, bc1 = create_mesh_and_bc(DISPLACEMENT)
    inv1 = InversionJointMC2D(
        mesh1, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc1,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_INIT_DEG,
        psi_deg=PHI_INIT_DEG * PSI_RATIO,
    )
    inv1.fe.dirichlet_bc_info_raw = bc1
    fwd1 = ad_wrapper(inv1, solver_options=SOLVER_OPTIONS,
                      adjoint_solver_options=SOLVER_OPTIONS)

    nc, nq = len(inv1.fe.cells), inv1.fe.num_quads

    def loss_phase1(x):
        c_f, phi_f = params_to_physical(x)
        stacked = np.concatenate([c_f, phi_f])

        inv1.sigmas_old = np.zeros((nc, nq, 3, 3))
        inv1.epsilons_old = np.zeros((nc, nq, 3, 3))
        inv1.internal_vars[0] = inv1.sigmas_old
        inv1.internal_vars[1] = inv1.epsilons_old
        run_warmup_steps(inv1, jax.lax.stop_gradient(stacked), N_STEPS)

        sol_list = fwd1(stacked)
        u_pred = sol_list[0][obs_indices]
        return np.sum((u_pred - u_obs) ** 2) / n_obs_dofs

    vg1 = jax.value_and_grad(loss_phase1)
    res1, hist1 = run_optimizer(vg1, x_init, bounds,
                                maxiter=100, maxfun=300, label='P1')

    c_p1, phi_p1 = params_to_physical(np.array(res1.x))
    c_p1_np = onp.array(c_p1)
    phi_p1_np = onp.array(phi_p1)

    print(f"\n  Phase 1 result:")
    print(f"    c: [{c_p1_np.min():.1f}, {c_p1_np.max():.1f}]")
    print(f"    φ: [{onp.degrees(phi_p1_np.min()):.1f}°, {onp.degrees(phi_p1_np.max()):.1f}°]")

    # ===================================================================
    # Phase 2: Single-step solve
    # ===================================================================
    print("\n" + "=" * 50)
    print("Phase 2: Single-step (correcting)")
    print("=" * 50)

    # Single-step observation
    print("  Generating single-step observation...")
    mesh_ss, bc_ss = create_mesh_and_bc(DISPLACEMENT)
    prob_ss = InversionJointMC2D(
        mesh_ss, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_ss,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_INIT_DEG,
        psi_deg=PHI_INIT_DEG * PSI_RATIO,
    )
    prob_ss.set_params(make_stacked_params(c_true, phi_true))
    u_ss = solver(prob_ss, solver_options=SOLVER_OPTIONS)[0]
    print(f"  ||u_obs_ss|| = {float(np.linalg.norm(u_ss)):.6e}")

    mesh2, bc2 = create_mesh_and_bc(DISPLACEMENT)
    inv2 = InversionJointMC2D(
        mesh2, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc2,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_INIT_DEG,
        psi_deg=PHI_INIT_DEG * PSI_RATIO,
    )
    fwd2 = ad_wrapper(inv2, solver_options=SOLVER_OPTIONS,
                      adjoint_solver_options=SOLVER_OPTIONS)

    def loss_phase2(x):
        c_f, phi_f = params_to_physical(x)
        stacked = np.concatenate([c_f, phi_f])
        sol_list = fwd2(stacked)
        u_pred = sol_list[0][obs_indices]
        return np.sum((u_pred - u_ss) ** 2) / n_obs_dofs

    vg2 = jax.value_and_grad(loss_phase2)
    x_phase1 = onp.array(res1.x, dtype=onp.float64)
    res2, hist2 = run_optimizer(vg2, x_phase1, bounds,
                                maxiter=150, maxfun=400, label='P2')

    # ===================================================================
    # Results
    # ===================================================================
    c_final, phi_final = params_to_physical(np.array(res2.x))
    c_final_np = onp.array(c_final)
    phi_final_np = onp.array(phi_final)
    c_true_np = onp.array(c_true)
    phi_true_np = onp.array(phi_true)

    # Errors
    c_rel = onp.abs(c_final_np - c_true_np) / c_true_np
    phi_rel = onp.abs(phi_final_np - phi_true_np) / phi_true_np
    l2_c = float(onp.sqrt(onp.mean(c_rel ** 2)))
    l2_phi = float(onp.sqrt(onp.mean(phi_rel ** 2)))

    left_mask = c_true_np < 50.
    right_mask = ~left_mask
    c_left_err = float(onp.mean(onp.abs(c_final_np[left_mask] - C_LEFT) / C_LEFT))
    c_right_err = float(onp.mean(onp.abs(c_final_np[right_mask] - C_RIGHT) / C_RIGHT))
    phi_left_err = float(onp.mean(onp.abs(onp.degrees(phi_final_np[left_mask]) - PHI_LEFT_DEG) / PHI_LEFT_DEG))
    phi_right_err = float(onp.mean(onp.abs(onp.degrees(phi_final_np[right_mask]) - PHI_RIGHT_DEG) / PHI_RIGHT_DEG))

    t_total = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  Final Result:")
    print(f"    c range: [{c_final_np.min():.2f}, {c_final_np.max():.2f}]")
    print(f"    φ range: [{onp.degrees(phi_final_np.min()):.2f}°, "
          f"{onp.degrees(phi_final_np.max()):.2f}°]")
    print(f"    L2 c error: {l2_c:.4f} ({l2_c:.2%})")
    print(f"    L2 φ error: {l2_phi:.4f} ({l2_phi:.2%})")
    print(f"    c left (true={C_LEFT}): {c_left_err:.2%}")
    print(f"    c right (true={C_RIGHT}): {c_right_err:.2%}")
    print(f"    φ left (true={PHI_LEFT_DEG}°): {phi_left_err:.2%}")
    print(f"    φ right (true={PHI_RIGHT_DEG}°): {phi_right_err:.2%}")
    print(f"    Time: {t_total:.1f}s")

    status_c = "PASS" if l2_c < 0.15 else "IMPROVED" if l2_c < 0.30 else "NEEDS WORK"
    status_phi = "PASS" if l2_phi < 0.15 else "IMPROVED" if l2_phi < 0.30 else "NEEDS WORK"
    print(f"    c status: {status_c}, φ status: {status_phi}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # c fields
    vmin_c, vmax_c = C_LEFT * 0.7, C_RIGHT * 1.3
    plot_c_field(c_true, Nx, Ny, ax=axes[0, 0], title='True c(x)',
                 vmin=vmin_c, vmax=vmax_c)
    plot_c_field(c_final_np, Nx, Ny, ax=axes[0, 1],
                 title=f'Inverted c (L2={l2_c:.2%})', vmin=vmin_c, vmax=vmax_c)

    ax = axes[0, 2]
    iy_mid = Ny // 2
    x_c = onp.linspace(Lx / (2 * Nx), Lx - Lx / (2 * Nx), Nx)
    ax.plot(x_c, c_true_np.reshape(Nx, Ny)[:, iy_mid], 'k-', lw=2, label='True')
    ax.plot(x_c, c_final_np.reshape(Nx, Ny)[:, iy_mid], 'ro-', ms=5, lw=1.5,
            label='Inverted')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('c [MPa]')
    ax.set_title('c line cut')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # phi fields
    phi_true_deg = onp.degrees(phi_true_np).reshape(Nx, Ny).T
    phi_final_deg = onp.degrees(phi_final_np).reshape(Nx, Ny).T
    vmin_p, vmax_p = PHI_LEFT_DEG * 0.7, PHI_RIGHT_DEG * 1.3

    im0 = axes[1, 0].imshow(phi_true_deg, origin='lower', aspect='equal',
                              vmin=vmin_p, vmax=vmax_p, cmap='RdYlBu_r')
    axes[1, 0].set_title('True φ(x)')
    plt.colorbar(im0, ax=axes[1, 0], label='φ [°]')

    im1 = axes[1, 1].imshow(phi_final_deg, origin='lower', aspect='equal',
                              vmin=vmin_p, vmax=vmax_p, cmap='RdYlBu_r')
    axes[1, 1].set_title(f'Inverted φ (L2={l2_phi:.2%})')
    plt.colorbar(im1, ax=axes[1, 1], label='φ [°]')

    ax = axes[1, 2]
    ax.plot(x_c, onp.degrees(phi_true_np.reshape(Nx, Ny)[:, iy_mid]),
            'k-', lw=2, label='True')
    ax.plot(x_c, onp.degrees(phi_final_np.reshape(Nx, Ny)[:, iy_mid]),
            'ro-', ms=5, lw=1.5, label='Inverted')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('φ [°]')
    ax.set_title('φ line cut')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'MC Joint Inversion: c L2={l2_c:.2%}, φ L2={l2_phi:.2%}',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mc_joint_inversion.png'), dpi=150)
    plt.close()

    # Convergence
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.semilogy(hist1, 'b-', lw=1.5, label='Phase 1')
    off = len(hist1)
    ax2.semilogy(range(off, off + len(hist2)), hist2, 'r-', lw=1.5,
                 label='Phase 2')
    ax2.axvline(off, color='gray', ls='--', alpha=0.5)
    ax2.set_xlabel('Evaluation')
    ax2.set_ylabel('Loss')
    ax2.set_title('Joint Inversion Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mc_joint_convergence.png'), dpi=150)
    plt.close()

    save_results(OUT_DIR, {
        'c_true': c_true_np, 'c_final': c_final_np,
        'phi_true_deg': onp.degrees(phi_true_np),
        'phi_final_deg': onp.degrees(phi_final_np),
        'l2_c': l2_c, 'l2_phi': l2_phi,
        'c_left_err': c_left_err, 'c_right_err': c_right_err,
        'phi_left_err': phi_left_err, 'phi_right_err': phi_right_err,
        'time_s': t_total, 'status_c': status_c, 'status_phi': status_phi,
    })
    print(f"  Results saved to {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
