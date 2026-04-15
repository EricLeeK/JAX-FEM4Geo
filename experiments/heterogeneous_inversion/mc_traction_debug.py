#!/usr/bin/env python
"""
Debug MC Newton convergence under traction (Neumann) loading.

Strategy:
1. Find the traction threshold where single-step Newton diverges
2. Test incremental loading (N steps) to get past the threshold
3. Test line_search_flag to stabilize Newton
4. Verify the solved displacement depends on c (identifiability check)
"""

import jax
import jax.numpy as np
import numpy as onp
import os, sys, time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from common import InversionHeterogeneousMC2D
from jax_fem.solver import solver
from jax_fem.generate_mesh import rectangle_mesh, Mesh

Nx, Ny = 10, 10
Lx, Ly = 10., 10.
E, nu = 70000., 0.3
C_VAL, PHI_DEG, PSI_DEG = 50., 30., 15.
SOLVER_LU = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
SOLVER_LS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'},
             'line_search_flag': True}


def create_traction_problem(traction):
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')

    def bottom(p): return np.isclose(p[1], 0., atol=1e-5)
    def corner(p): return np.logical_and(
        np.isclose(p[0], 0., atol=1e-5), np.isclose(p[1], 0., atol=1e-5))

    bc = [[bottom, corner], [1, 0],
          [lambda p: 0., lambda p: 0.]]

    def top(p): return np.isclose(p[1], Ly, atol=1e-5)

    problem = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc, location_fns=[top],
        E=E, nu=nu, c=C_VAL, phi_deg=PHI_DEG, psi_deg=PSI_DEG,
        traction_value=traction,
    )
    return problem


def try_solve(traction, solver_opts, label=""):
    """Try single-step solve, return (success, max_disp, time)."""
    problem = create_traction_problem(traction)
    t0 = time.time()
    try:
        sol = solver(problem, solver_options=solver_opts)[0]
        dt = time.time() - t0
        max_d = float(np.max(np.abs(sol)))
        return True, max_d, dt
    except Exception as e:
        dt = time.time() - t0
        return False, 0., dt


def try_incremental(traction, n_steps, solver_opts):
    """Incremental loading: apply traction in n_steps."""
    tractions = onp.linspace(traction / n_steps, traction, n_steps)
    problem = create_traction_problem(float(tractions[0]))
    prev_sol = None

    for i, t_val in enumerate(tractions):
        # Rebuild traction for this step
        problem._setup_traction(float(t_val))
        # Re-init surface integrals
        if hasattr(problem, 'fes'):
            problem.fes[0].neumann_boundary_inds_list = problem.fes[0].get_boundary_conditions_inds(
                problem.location_fns) if hasattr(problem, 'location_fns') and problem.location_fns else []

        opts = dict(solver_opts)
        if prev_sol is not None:
            opts['initial_guess'] = [prev_sol]

        try:
            sol = solver(problem, solver_options=opts)[0]
            problem.update_stress_strain(sol)
            prev_sol = sol
        except Exception as e:
            return False, i, 0.

    max_d = float(np.max(np.abs(prev_sol)))
    return True, n_steps, max_d


def main():
    import logging
    logging.getLogger('jax_fem').setLevel(logging.WARNING)

    print("=" * 65)
    print("MC Traction Loading — Newton Convergence Debug")
    print(f"  c={C_VAL}, φ={PHI_DEG}°, E={E}, mesh={Nx}×{Ny}")
    print("=" * 65)

    # --- Test 1: Find divergence threshold (single-step, no line search) ---
    print("\n[Test 1] Single-step Newton (no line search)")
    print(f"  {'Traction':>10s}  {'Status':>8s}  {'|u|_max':>10s}  {'Time':>6s}")
    for t in [-10, -20, -30, -40, -50, -60, -80, -100, -120]:
        ok, d, dt = try_solve(t, SOLVER_LU)
        status = "✅ OK" if ok else "❌ FAIL"
        print(f"  {t:>10.0f}  {status:>8s}  {d:>10.4f}  {dt:5.2f}s")

    # --- Test 2: With line search ---
    print("\n[Test 2] Single-step Newton + line search")
    print(f"  {'Traction':>10s}  {'Status':>8s}  {'|u|_max':>10s}  {'Time':>6s}")
    for t in [-10, -20, -30, -40, -50, -60, -80, -100, -120]:
        ok, d, dt = try_solve(t, SOLVER_LS)
        status = "✅ OK" if ok else "❌ FAIL"
        print(f"  {t:>10.0f}  {status:>8s}  {d:>10.4f}  {dt:5.2f}s")

    # --- Test 3: Incremental loading at high traction ---
    print("\n[Test 3] Incremental loading (traction=-120)")
    for n in [2, 5, 10, 20]:
        ok, steps_done, d = try_incremental(-120., n, SOLVER_LU)
        status = f"✅ {steps_done}/{n}" if ok else f"❌ at step {steps_done}/{n}"
        print(f"  {n:3d} steps: {status:>16s}  |u|_max={d:.4f}")

    # --- Test 4: Incremental + line search ---
    print("\n[Test 4] Incremental + line search (traction=-120)")
    for n in [2, 5, 10]:
        ok, steps_done, d = try_incremental(-120., n, SOLVER_LS)
        status = f"✅ {steps_done}/{n}" if ok else f"❌ at step {steps_done}/{n}"
        print(f"  {n:3d} steps: {status:>16s}  |u|_max={d:.4f}")

    # --- Test 5: c sensitivity under traction (elastic range) ---
    print("\n[Test 5] c sensitivity check (single-step, traction=-30)")
    for c_val in [30., 50., 70., 100.]:
        problem = create_traction_problem(-30.)
        nc = len(problem.fe.cells)
        problem.set_params(np.full(nc, c_val))
        try:
            sol = solver(problem, solver_options=SOLVER_LU)[0]
            max_d = float(np.max(np.abs(sol)))
            print(f"  c={c_val:5.0f}: |u|_max={max_d:.6f}")
        except Exception:
            print(f"  c={c_val:5.0f}: FAIL")

    # --- Test 6: c sensitivity with incremental loading to various tractions ---
    # MC uniaxial yield: σ_y = 2c·cosφ/(1-sinφ)
    # c=30 → σ_y=104, c=50 → σ_y=173, c=70 → σ_y=242, c=100 → σ_y=346
    print("\n[Test 6] Yield stress estimates: σ_y = 2c·cosφ/(1-sinφ)")
    for c_val in [30., 50., 70., 100.]:
        phi_r = onp.radians(PHI_DEG)
        sy = 2 * c_val * onp.cos(phi_r) / (1 - onp.sin(phi_r))
        print(f"  c={c_val:5.0f}: σ_y = {sy:.0f} MPa")

    for target_t in [-120., -200., -300.]:
        n_steps = max(10, int(abs(target_t) / 10))
        print(f"\n[Test 6] c sensitivity (incr {n_steps} steps, traction={target_t})")
        for c_val in [30., 50., 70., 100.]:
            problem = create_traction_problem(float(target_t / n_steps))
            nc = len(problem.fe.cells)
            problem.set_params(np.full(nc, c_val))

            tractions = onp.linspace(target_t / n_steps, target_t, n_steps)
            ok = True
            for t_val in tractions:
                problem._setup_traction(float(t_val))
                try:
                    sol = solver(problem, solver_options=SOLVER_LU)[0]
                    problem.update_stress_strain(sol)
                except Exception:
                    ok = False
                    break

            if ok:
                max_d = float(np.max(np.abs(sol)))
                print(f"  c={c_val:5.0f}: |u|_max={max_d:.6f}")
            else:
                print(f"  c={c_val:5.0f}: FAIL")

    # --- Test 7: Fresh problem per traction level (no reuse) ---
    print("\n[Test 7] Fresh problem per c (single-step, traction=-50)")
    for c_val in [30., 50., 70., 100.]:
        prob = create_traction_problem(-50.)
        nc = len(prob.fe.cells)
        prob.set_params(np.full(nc, c_val))
        try:
            sol = solver(prob, solver_options=SOLVER_LU)[0]
            max_d = float(np.max(np.abs(sol)))
            min_d = float(np.min(sol))
            print(f"  c={c_val:5.0f}: |u|_max={max_d:.6f}, u_min={min_d:.6f}")
        except Exception:
            print(f"  c={c_val:5.0f}: FAIL")

    # Test with lower c values that should yield at -50 MPa
    # c=10 → σ_y=35 MPa, should yield at -50
    print("\n[Test 7b] Low c values (single-step, traction=-50)")
    for c_val in [5., 10., 15., 20., 30.]:
        phi_r = onp.radians(PHI_DEG)
        sy = 2 * c_val * onp.cos(phi_r) / (1 - onp.sin(phi_r))
        prob = create_traction_problem(-50.)
        nc = len(prob.fe.cells)
        prob.set_params(np.full(nc, c_val))
        try:
            sol = solver(prob, solver_options=SOLVER_LU)[0]
            max_d = float(np.max(np.abs(sol)))
            print(f"  c={c_val:5.0f} (σ_y={sy:.0f}): |u|_max={max_d:.6f}")
        except Exception:
            print(f"  c={c_val:5.0f} (σ_y={sy:.0f}): FAIL")

    # --- Test 8: Proper incremental with fresh problems ---
    print("\n[Test 8] Proper incremental (fresh problems, traction=-50)")
    n_steps = 50
    for c_val in [10., 20., 30., 50.]:
        tractions = onp.linspace(-50. / n_steps, -50., n_steps)
        sigma_old = np.zeros((Nx*Ny, 4, 3, 3))  # nc, nq, 3, 3
        eps_old = np.zeros((Nx*Ny, 4, 3, 3))
        ok = True
        sol = None

        for step, t_val in enumerate(tractions):
            prob = create_traction_problem(float(t_val))
            nc = len(prob.fe.cells)
            prob.set_params(np.full(nc, c_val))
            prob.sigmas_old = sigma_old
            prob.epsilons_old = eps_old
            prob.internal_vars[0] = sigma_old
            prob.internal_vars[1] = eps_old

            try:
                sol = solver(prob, solver_options=SOLVER_LU)[0]
                prob.update_stress_strain(sol)
                sigma_old = prob.sigmas_old
                eps_old = prob.epsilons_old
            except Exception:
                ok = False
                print(f"  c={c_val:5.0f}: FAIL at step {step+1}/{n_steps} (t={t_val:.0f})")
                break

        if ok:
            max_d = float(np.max(np.abs(sol)))
            print(f"  c={c_val:5.0f}: |u|_max={max_d:.6f} ✅")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
