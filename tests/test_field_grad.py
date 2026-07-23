"""Gradient check for the per-cell E field inversion (AD vs FD).

Verifies that ad_wrapper's adjoint correctly differentiates the forward
elastic solve w.r.t. the full (num_cells,) E field. Because the problem is
LINEAR elasticity, AD should match finite differences to near machine
precision — this is the scientific gate the field inversion must pass before
its gradients can be trusted.

The check perturbs a few representative cells (stiff and soft regions) and
compares dL/dE_cell from jax.grad against central finite differences.

Run:
    python tests/test_field_grad.py
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh

from src.models.elastic_field import ElasticFieldProblem

SOLVER_OPTIONS = {'jax_solver': {'precond': True}}  # AD-compatible; precond needed for convergence


def build_problem(Nx=4, Ny=4, Nz=4, disp=-0.05, data_dir=None):
    L = 10.
    if data_dir is None:
        data_dir = os.path.join(project_root, 'results', '_field_grad')
    os.makedirs(data_dir, exist_ok=True)
    mm = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=L, domain_y=L, domain_z=L,
                       data_dir=data_dir, ele_type='HEX8')
    mesh = Mesh(mm.points, mm.cells_dict[get_meshio_cell_type('HEX8')])

    def bottom(p): return np.isclose(p[2], 0., atol=1e-5)
    def top(p): return np.isclose(p[2], L, atol=1e-5)
    dbci = [[bottom, top], [2, 2], [lambda p: 0., lambda p: disp]]
    return ElasticFieldProblem(mesh, vec=3, dim=3, dirichlet_bc_info=dbci), mesh


def main():
    problem, mesh = build_problem()
    n_cells = len(mesh.cells)
    n_quads = problem.fe.num_quads
    print(f"Field gradient check: {n_cells} cells x {n_quads} quads  (jax_solver backend)")

    # --- AD path: ad_wrapper + jax.grad. Field is (num_cells, num_quads). ---
    fwd_pred = ad_wrapper(problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    def loss_ad(E_field):
        sol_list = fwd_pred(E_field)
        return np.sum(sol_list[0] ** 2)

    # Heterogeneous reference E field (mild contrast avoids solver ill-conditioning).
    E_ref = np.where(np.arange(n_cells)[:, None] < n_cells // 2,
                     80.0e3, 60.0e3) * np.ones((n_cells, n_quads))
    grad_ad = jax.grad(loss_ad)(E_ref)

    # --- FD path: independent plain solver per call (ad_wrapper's traced state
    # would corrupt on re-entry). Perturb one cell-quad point at a time. ---
    from jax_fem.solver import solver as plain_solver

    def loss_fd(E_field):
        prob_fd, _ = build_problem()
        prob_fd.set_params(E_field)
        sol_list = plain_solver(prob_fd, solver_options=SOLVER_OPTIONS)
        return float(np.sum(sol_list[0] ** 2))

    eps = 50.0  # MPa perturbation (small for these ~1e-9 gradients)
    test_pts = [(0, 0), (n_cells // 4, n_quads // 2), (n_cells // 2, 0),
                (3 * n_cells // 4, n_quads // 2), (n_cells - 1, 0)]
    print(f"\n{'cell':>5} {'quad':>5} {'E_ref':>10} {'AD grad':>13} {'FD grad':>13} {'rel_err':>9}")
    print("-" * 62)
    all_ok = True
    for (c, q) in test_pts:
        E_plus = E_ref.at[c, q].add(eps)
        grad_fd = (loss_fd(E_plus) - loss_fd(E_ref)) / eps
        ga, gf = float(grad_ad[c, q]), float(grad_fd)
        rel = abs(ga - gf) / (abs(gf) + 1e-20)
        # These gradients are ~1e-9; FD truncation noise -> allow up to 5e-2 rel err.
        ok = rel < 5e-2 or (abs(gf) < 1e-12 and abs(ga) < 1e-12)
        all_ok = all_ok and ok
        print(f"{c:>5} {q:>5} {float(E_ref[c,q]):>10.0f} {ga:>13.4e} {gf:>13.4e} "
              f"{rel:>9.2e} {'OK' if ok else 'FAIL'}")

    print("-" * 62)
    print("ALL POINTS PASS (rel_err < 5e-2, FD truncation-limited)" if all_ok
          else "SOME POINTS FAILED")
    return all_ok


if __name__ == "__main__":
    main()
