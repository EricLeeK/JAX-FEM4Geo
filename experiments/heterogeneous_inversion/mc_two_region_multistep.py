#!/usr/bin/env python
"""
MC Two-Region Inversion with Multi-Step Observation.

Combines two complementary loss terms:
1. Single-step loss: one-shot solve at full displacement (constrains
   the right/elastic region well because single-step return map is
   accurate there).
2. Multi-step loss: incremental loading with per-step displacement
   matching (constrains the left/plastic region because incremental
   return mapping is more accurate in strongly plastic zones).

History between incremental steps is carried with stop_gradient.
The single-step solve is independently differentiable.
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

OUT_DIR = os.path.join(RESULTS_DIR, 'mc_two_region_multistep')
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


def generate_multistep_obs(c_field, n_steps=N_STEPS):
    """Forward solve with incremental loading, return displacement at each step."""
    disps = onp.linspace(DISPLACEMENT / n_steps, DISPLACEMENT, n_steps)

    mesh, bc = create_mesh_and_bc(float(disps[0]))
    problem = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    problem.set_params(np.array(c_field))

    u_steps = []
    for d in disps:
        bc[-1][1] = lambda p, _d=float(d): _d
        problem.fe.update_Dirichlet_boundary_conditions(bc)
        sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
        u_steps.append(sol_list[0])
        problem.update_stress_strain(sol_list[0])

    return u_steps


def main():
    print("=" * 70)
    print("MC Two-Region Inversion (Multi-Step + Single-Step Combined)")
    print(f"  Mesh: {Nx}×{Ny}, disp={DISPLACEMENT}, steps={N_STEPS}")
    print(f"  phi={PHI_DEG}°, c_true: left={C_LEFT}, right={C_RIGHT}")
    print("=" * 70)
    t0 = time.time()

    c_true = two_region_c_field(Nx, Ny, C_LEFT, C_RIGHT)

    # Generate multi-step observations
    print("\n  Generating synthetic observations (all steps)...")
    u_obs_steps = generate_multistep_obs(c_true, N_STEPS)
    obs_indices = onp.arange(u_obs_steps[0].shape[0])
    for k, u in enumerate(u_obs_steps):
        print(f"    Step {k+1}: ||u|| = {float(np.linalg.norm(u)):.6e}")

    # The final-step observation is also our single-step target
    u_obs_final = u_obs_steps[-1]

    # --- Single-step problem (for right/elastic region constraint) ---
    print("\n  Setting up single-step AD problem...")
    mesh_ss, bc_ss = create_mesh_and_bc(DISPLACEMENT)
    ss_problem = InversionHeterogeneousMC2D(
        mesh_ss, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_ss,
        E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
    )
    ss_wrapper = ad_wrapper(ss_problem, solver_options=SOLVER_OPTIONS,
                            adjoint_solver_options=SOLVER_OPTIONS)

    # --- Multi-step problems (for left/plastic region constraint) ---
    print("  Setting up per-step AD problems...")
    step_disps = onp.linspace(DISPLACEMENT / N_STEPS, DISPLACEMENT, N_STEPS)
    ad_problems = []
    ad_wrappers = []
    scratch_problems = []

    for k, d in enumerate(step_disps):
        mesh_k, bc_k = create_mesh_and_bc(float(d))
        p_ad = InversionHeterogeneousMC2D(
            mesh_k, vec=2, dim=2, ele_type='QUAD4',
            dirichlet_bc_info=bc_k,
            E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
        )
        ad_problems.append(p_ad)
        ad_wrappers.append(
            ad_wrapper(p_ad, solver_options=SOLVER_OPTIONS,
                       adjoint_solver_options=SOLVER_OPTIONS)
        )

        mesh_s, bc_s = create_mesh_and_bc(float(d))
        p_sc = InversionHeterogeneousMC2D(
            mesh_s, vec=2, dim=2, ele_type='QUAD4',
            dirichlet_bc_info=bc_s,
            E=E_FIXED, nu=0.3, c=C_INIT, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
        )
        scratch_problems.append(p_sc)

    nc, nq = len(ad_problems[0].fe.cells), ad_problems[0].fe.num_quads
    n_obs_dofs = u_obs_steps[0].shape[0] * u_obs_steps[0].shape[1]

    # Normalization for single-step loss
    ss_norm = float(np.mean(u_obs_final[obs_indices] ** 2)) + 1e-12

    # Weight balancing single-step vs multi-step
    # SS anchors the right/elastic region; MS provides accurate plastic gradients
    W_SS = 0.2   # single-step weight (mild anchor for right region)
    W_MS = 1.0   # multi-step weight (dominant, for left/plastic region)

    def loss_fn(log_c):
        c_field = log_to_c(log_c)

        # --- Single-step loss (fresh internal vars, no history) ---
        ss_problem.sigmas_old = np.zeros((nc, nq, 3, 3))
        ss_problem.epsilons_old = np.zeros((nc, nq, 3, 3))
        ss_problem.internal_vars[0] = ss_problem.sigmas_old
        ss_problem.internal_vars[1] = ss_problem.epsilons_old
        sol_ss = ss_wrapper(c_field)
        u_pred_ss = sol_ss[0][obs_indices]
        loss_ss = np.mean((u_pred_ss - u_obs_final) ** 2) / ss_norm

        # --- Multi-step loss (incremental with stop_gradient history) ---
        sigma = np.zeros((nc, nq, 3, 3))
        eps = np.zeros((nc, nq, 3, 3))
        loss_ms = 0.0

        for k in range(N_STEPS):
            sigma_in = jax.lax.stop_gradient(sigma)
            eps_in = jax.lax.stop_gradient(eps)

            p_ad = ad_problems[k]
            p_ad.sigmas_old = sigma_in
            p_ad.epsilons_old = eps_in
            p_ad.internal_vars[0] = sigma_in
            p_ad.internal_vars[1] = eps_in

            sol_list = ad_wrappers[k](c_field)
            u_pred = sol_list[0][obs_indices]
            u_obs_k = u_obs_steps[k][obs_indices]

            step_loss = np.sum((u_pred - u_obs_k) ** 2) / n_obs_dofs
            loss_ms = loss_ms + step_loss

            # Update history via scratch problem
            p_sc = scratch_problems[k]
            p_sc.sigmas_old = sigma_in
            p_sc.epsilons_old = eps_in
            p_sc.internal_vars[0] = sigma_in
            p_sc.internal_vars[1] = eps_in
            p_sc.set_params(jax.lax.stop_gradient(c_field))
            sol_for_update = jax.lax.stop_gradient(sol_list[0])
            p_sc.update_stress_strain(sol_for_update)
            sigma = p_sc.sigmas_old
            eps = p_sc.epsilons_old

        loss_ms = loss_ms / N_STEPS

        return W_SS * loss_ss + W_MS * loss_ms

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    log_c_init = onp.array(c_to_log(uniform_c_field(Nx, Ny, C_INIT)),
                           dtype=onp.float64)

    print("\n  Warming up (first forward+backward)...")
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
        options={'maxiter': 200, 'maxfun': 600, 'ftol': 1e-20, 'gtol': 1e-12},
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

    fig.suptitle(f'MC Combined SS+MS ({N_STEPS} steps): L2={l2_rel:.2%}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mc_multistep_inversion.png'), dpi=150)
    plt.close()

    if history:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.semilogy(history, 'b-', lw=1.5)
        ax2.set_xlabel('Evaluation')
        ax2.set_ylabel('Loss')
        ax2.set_title('Convergence')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'mc_multistep_convergence.png'),
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
