#!/usr/bin/env python
"""
E3: Triaxial Test Parameter Inversion

Validates the two-stage inversion approach under triaxial loading conditions:
  - Cylindrical specimen (H:D = 2:1) with confining pressure
  - Incremental loading to reach deep plasticity
  - Two-stage inversion: elastic obs → E, plastic obs → k

Compares with uniaxial results from D1 to assess generality.
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
    InversionTriaxialDP, stress_return_dp,
    RESULTS_DIR, cylinder_mesh_gmsh, Mesh, update_bc,
)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper, solver

BASE_SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(RESULTS_DIR, 'e3_triaxial_inversion')
os.makedirs(OUT_DIR, exist_ok=True)

E_TRUE, K_TRUE = 70000.0, 50.0
E_INIT, K_INIT = 55000.0, 45.0
CONFINING_PRESSURE = 10.0  # MPa

# Incremental loading parameters
N_LOAD_STEPS = 10  # steps to reach plastic displacement


def triaxial_volume_avg_sigma_zz(fe, sol, sigma_old, epsilon_old, E, k):
    """Volume-averaged sigma_zz using explicit E, k for AD."""
    u_grads = fe.sol_to_grad(sol)
    c, q, _, dim = u_grads.shape

    def one_quad(ug, so, eo):
        return stress_return_dp(ug, so, eo, E, k, dim)

    sig = jax.vmap(jax.vmap(one_quad))(u_grads, sigma_old, epsilon_old)
    JxW = fe.JxW
    sig_zz = sig[..., 2, 2]
    return np.sum(sig_zz * JxW) / np.sum(JxW)


def make_triaxial_bc(H, displacement):
    """Build Dirichlet BC info for triaxial test."""
    tol = 1e-5

    def bottom(p):
        return np.isclose(p[2], 0., atol=tol)

    def top(p):
        return np.isclose(p[2], H, atol=tol)

    def center_bottom(p):
        return np.logical_and(
            np.logical_and(np.abs(p[0]) < tol, np.abs(p[1]) < tol),
            np.isclose(p[2], 0., atol=tol),
        )

    return [
        [bottom, center_bottom, center_bottom, top],
        [2, 0, 1, 2],
        [lambda p: 0., lambda p: 0., lambda p: 0.,
         lambda p, _d=displacement: _d],
    ]


def update_triaxial_bc(problem, bc_info, disp):
    """Update top-face displacement for triaxial problem."""
    bc_info[-1][3] = lambda p, _d=disp: _d
    problem.fes[0].update_Dirichlet_boundary_conditions(bc_info)


def incremental_triaxial_solve(problem, bc_info, target_disp, n_steps,
                               solver_options):
    """Incrementally load triaxial specimen to target_disp."""
    ls_opts = {**solver_options, 'line_search_flag': True,
               'line_search_max_iters': 15}
    prev_sol = None
    for i in range(n_steps):
        disp_i = target_disp * (i + 1) / n_steps
        update_triaxial_bc(problem, bc_info, disp_i)
        step_opts = dict(ls_opts)
        if prev_sol is not None:
            step_opts['initial_guess'] = [prev_sol]
        sol = solver(problem, solver_options=step_opts)[0]
        problem.update_stress_strain(sol)
        prev_sol = sol
    return sol


def main():
    print("=" * 70)
    print("E3: TRIAXIAL TEST PARAMETER INVERSION (with incremental loading)")
    print("=" * 70)
    print(f"  True params:   E = {E_TRUE}, k = {K_TRUE}")
    print(f"  Initial guess: E = {E_INIT}, k = {K_INIT}")
    print(f"  Confining pressure: {CONFINING_PRESSURE} MPa")
    print(f"  Load steps for plastic: {N_LOAD_STEPS}")
    t0_total = time.time()

    R, H = 2.5, 10.0
    DISP_ELASTIC = -0.005   # safely elastic (single step OK)
    DISP_PLASTIC = -0.030   # deeper into plastic regime

    print(f"\n[1] Displacement thresholds: elastic={DISP_ELASTIC}, plastic={DISP_PLASTIC}")

    # --- Generate mesh ONCE ---
    print("\n[2] Generating cylindrical mesh (once)...")
    data_dir = os.path.join(RESULTS_DIR, '_triaxial_mesh_cache')
    os.makedirs(data_dir, exist_ok=True)
    meshio_mesh = cylinder_mesh_gmsh(
        data_dir=data_dir, R=R, H=H, circle_mesh=3, hight_mesh=4, rect_ratio=0.4)
    mesh_points = meshio_mesh.points
    mesh_cells = meshio_mesh.cells_dict['hexahedron']
    print(f"  Nodes: {mesh_points.shape[0]}, Elements: {mesh_cells.shape[0]}")

    tol = 1e-5

    def lateral_surface(p):
        r = np.sqrt(p[0]**2 + p[1]**2)
        z = p[2]
        on_surface = np.isclose(r, R, atol=tol * 10)
        not_top_bottom = np.logical_and(z > tol, z < H - tol)
        return np.logical_and(on_surface, not_top_bottom)

    location_fns = [lateral_surface]

    def make_problem(disp):
        mesh = Mesh(mesh_points, mesh_cells, ele_type='HEX8')
        bc = make_triaxial_bc(H, disp)
        return InversionTriaxialDP(
            mesh, vec=3, dim=3, dirichlet_bc_info=bc,
            location_fns=location_fns, confining_pressure=CONFINING_PRESSURE), bc

    # --- Generate observations ---
    print("\n[3] Generating observations...")

    # Elastic observation (single step)
    prob_oe, _ = make_problem(DISP_ELASTIC)
    prob_oe.set_params(np.array([E_TRUE, K_TRUE]))
    sol_oe = solver(prob_oe, solver_options=BASE_SOLVER_OPTIONS)[0]
    obs_elastic = float(triaxial_volume_avg_sigma_zz(
        prob_oe.fe, sol_oe, prob_oe.sigmas_old, prob_oe.epsilons_old,
        E_TRUE, K_TRUE))
    print(f"  σ_zz_obs (elastic, disp={DISP_ELASTIC}): {obs_elastic:.4f} MPa")

    # Plastic observation (incremental loading)
    prob_op, bc_op = make_problem(DISP_PLASTIC)
    prob_op.set_params(np.array([E_TRUE, K_TRUE]))
    sol_op = incremental_triaxial_solve(
        prob_op, bc_op, DISP_PLASTIC, N_LOAD_STEPS, BASE_SOLVER_OPTIONS)
    obs_plastic = float(triaxial_volume_avg_sigma_zz(
        prob_op.fe, sol_op, prob_op.sigmas_old, prob_op.epsilons_old,
        E_TRUE, K_TRUE))
    print(f"  σ_zz_obs (plastic, disp={DISP_PLASTIC}): {obs_plastic:.4f} MPa")

    # --- Stage 1: Invert E ---
    print("\n" + "=" * 70)
    print("STAGE 1: Invert E from elastic observation")
    print("=" * 70)

    prob_e, _ = make_problem(DISP_ELASTIC)
    fwd_e = ad_wrapper(prob_e, solver_options=BASE_SOLVER_OPTIONS,
                       adjoint_solver_options=BASE_SOLVER_OPTIONS)

    def loss_E(params_E):
        E = params_E[0]
        k_dummy = 200.0
        sol = fwd_e(np.array([E, k_dummy]))[0]
        return (triaxial_volume_avg_sigma_zz(
            prob_e.fe, sol, prob_e.sigmas_old, prob_e.epsilons_old, E, k_dummy
        ) - obs_elastic) ** 2

    print("  Warming up JIT...")
    _ = jax.value_and_grad(loss_E)(np.array([E_INIT]))

    from scipy.optimize import minimize_scalar
    history_E = []

    def track_E(E_val):
        l = float(loss_E(np.array([E_val])))
        history_E.append({'E': E_val, 'loss': l})
        return l

    res_E = minimize_scalar(track_E, bounds=(30000, 120000), method='bounded',
                            options={'xatol': 1.0, 'maxiter': 30})
    E_stage1 = res_E.x
    err_E_s1 = abs(E_stage1 - E_TRUE) / E_TRUE
    print(f"  E = {E_stage1:.2f} (err {err_E_s1:.6%}, {res_E.nfev} evals)")

    # --- Stage 2: Invert k using incremental loading ---
    print("\n" + "=" * 70)
    print(f"STAGE 2: Invert k (E fixed at {E_stage1:.0f}) with incremental loading")
    print("=" * 70)

    E_fixed = E_stage1

    def loss_k_eval(k_val):
        """Evaluate loss for a given k using incremental loading."""
        prob_k, bc_k = make_problem(DISP_PLASTIC)
        prob_k.set_params(np.array([E_fixed, k_val]))
        try:
            sol_k = incremental_triaxial_solve(
                prob_k, bc_k, DISP_PLASTIC, N_LOAD_STEPS, BASE_SOLVER_OPTIONS)
            pred = float(triaxial_volume_avg_sigma_zz(
                prob_k.fe, sol_k, prob_k.sigmas_old, prob_k.epsilons_old,
                E_fixed, k_val))
            return (pred - obs_plastic) ** 2
        except Exception:
            return 1e20

    print("  Warming up...")
    _ = loss_k_eval(K_INIT)

    history_k = []

    def track_k(k_val):
        l = loss_k_eval(k_val)
        if l < 1e20:
            history_k.append({'k': k_val, 'loss': l})
        return l

    res_k = minimize_scalar(track_k, bounds=(20, 100), method='bounded',
                            options={'xatol': 0.1, 'maxiter': 30})
    k_stage2 = res_k.x
    err_k_s2 = abs(k_stage2 - K_TRUE) / K_TRUE
    print(f"  k = {k_stage2:.4f} (err {err_k_s2:.6%}, {res_k.nfev} evals)")

    t_total = time.time() - t0_total

    # --- Plotting ---
    print("\n[4] Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.text(0.5, 0.5, f'Elastic obs:\nσ_zz = {obs_elastic:.2f} MPa\ndisp = {DISP_ELASTIC}\n\n'
            f'Plastic obs:\nσ_zz = {obs_plastic:.2f} MPa\ndisp = {DISP_PLASTIC}\n'
            f'({N_LOAD_STEPS} load steps)',
            transform=ax.transAxes, ha='center', va='center', fontsize=11)
    ax.set_title(f'Triaxial Setup (σ₃={CONFINING_PRESSURE} MPa)')
    ax.set_axis_off()

    ax = axes[1]
    if history_E:
        ax.semilogy([h['loss'] for h in history_E], 'b-o', markersize=4, label='Stage 1 (E)')
    if history_k:
        n1 = len(history_E)
        ax.semilogy(range(n1, n1 + len(history_k)),
                     [h['loss'] for h in history_k], 'r-o', markersize=4, label='Stage 2 (k)')
    ax.set_xlabel('Function evaluation')
    ax.set_ylabel('Loss')
    ax.set_title('Inversion Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if history_E:
        ax.plot([h['E'] for h in history_E],
                [K_INIT] * len(history_E), 'b.-', markersize=4, label='Stage 1')
    if history_k:
        ax.plot([E_stage1] * len(history_k),
                [h['k'] for h in history_k], 'r.-', markersize=4, label='Stage 2')
    ax.plot(E_INIT, K_INIT, 'gs', markersize=10, label='Start')
    ax.plot(E_TRUE, K_TRUE, 'r*', markersize=15, label='True')
    ax.plot(E_stage1, k_stage2, 'kD', markersize=10, label='Final')
    ax.set_xlabel('E [MPa]')
    ax.set_ylabel('k [MPa]')
    ax.set_title('Parameter Space')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'E3: Triaxial Inversion (σ₃={CONFINING_PRESSURE} MPa, {N_LOAD_STEPS} steps)',
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'triaxial_inversion.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")

    # --- Save results ---
    results = {
        'E_true': E_TRUE, 'k_true': K_TRUE,
        'confining_pressure': CONFINING_PRESSURE,
        'n_load_steps': N_LOAD_STEPS,
        'disp_elastic': DISP_ELASTIC,
        'disp_plastic': DISP_PLASTIC,
        'obs_elastic': obs_elastic,
        'obs_plastic': obs_plastic,
        'stage1': {'E': float(E_stage1), 'err_E': float(err_E_s1), 'nfev': res_E.nfev},
        'stage2': {'k': float(k_stage2), 'err_k': float(err_k_s2), 'nfev': res_k.nfev},
        'final': {
            'E': float(E_stage1), 'k': float(k_stage2),
            'err_E': float(err_E_s1), 'err_k': float(err_k_s2),
        },
        'total_time_s': t_total,
    }
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("E3 SUMMARY")
    print("=" * 70)
    print(f"  Confining pressure: {CONFINING_PRESSURE} MPa")
    print(f"  Load steps (plastic): {N_LOAD_STEPS}")
    print(f"  Stage 1: E = {E_stage1:.2f} (err {err_E_s1:.6%})")
    print(f"  Stage 2: k = {k_stage2:.4f} (err {err_k_s2:.6%})")
    pass_E = err_E_s1 < 0.01
    pass_k = err_k_s2 < 0.05  # 5% threshold for k (triaxial is harder)
    print(f"\n  E convergence (<1%): {'PASS' if pass_E else 'FAIL'}")
    print(f"  k convergence (<5%): {'PASS' if pass_k else 'FAIL'}")
    overall = pass_E and pass_k
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"  Total time: {t_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
