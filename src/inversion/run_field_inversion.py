"""High-dimensional heterogeneous E-field inversion (the thesis core).

Inverts the per-cell Young's modulus field from sparse displacement
observations using the differentiable FEM. This is the central contribution of
the thesis title ("spatially-varying rock property inversion"): the unknown is
a whole parameter FIELD (one E per element), not just 2 scalars.

Approach
--------
Loss = ||u_sim(E_field)[obs_nodes] - u_obs||^2   (displacement misfit)

The forward solve is differentiated via ad_wrapper (implicit adjoint), giving
dL/d(E_field) for ALL cells in one backward pass — cost independent of the
number of cells (the key advantage over finite-difference gradients, which
would need num_cells+1 forward solves). Optimized in log-space with L-BFGS.

CRITICAL: uses jax_solver (not umfpack) — only jax_solver is JAX-traceable, so
only it gives correct adjoint gradients (verified in tests/test_field_grad.py).

Run:
    python -m src.inversion.run_field_inversion
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scipy.optimize import minimize as sp_min

from jax_fem.solver import ad_wrapper
from src.inversion.field_truth import (
    generate_field_truth, expand_to_quads, SOLVER_OPTIONS, build_problem)

MESH_DIR = os.path.join(project_root, 'results', 'field_inversion')


def invert_field(truth, E_init_per_cell, max_iter=100, log_every=10, mesh_n=6):
    """Recover the E field from the truth observation data.

    Parameters
    ----------
    truth : dict from generate_field_truth
    E_init_per_cell : (num_cells,) initial guess (homogeneous is a fine start).
    mesh_n : int
        Per-axis element count of the truth mesh (must match generate_field_truth's Nx).
    """
    mesh = truth['mesh']
    n_cells = len(mesh.cells)
    n_quads = truth['problem'].fe.num_quads
    obs_mask = truth['obs_mask']
    obs_disp = truth['obs_disp']

    # Fresh problem + ad_wrapper (jax_solver for correct adjoint). MUST use the
    # same mesh resolution as the truth so cell/node indexing lines up exactly.
    problem, mesh = build_problem(Nx=mesh_n, Ny=mesh_n, Nz=mesh_n, data_dir=MESH_DIR)
    fwd_pred = ad_wrapper(problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    obs_mask_jax = np.array(obs_mask)
    obs_disp_jax = np.array(obs_disp)

    def loss_fn(E_per_cell):
        # Expand (num_cells,) -> (num_cells, num_quads) for internal_vars.
        E_field = expand_to_quads(E_per_cell, n_quads)
        sol_list = fwd_pred(E_field)
        pred = sol_list[0][obs_mask_jax]
        return np.sum((pred - obs_disp_jax) ** 2)

    print(f"\n[Invert:{truth['field_type']}] {n_cells} unknowns, "
          f"{int(obs_mask.sum())} obs nodes, log-space L-BFGS")

    loss_and_grad = jax.value_and_grad(loss_fn)

    def fg(z):
        # z = log(E); chain rule dL/dz = dL/dE * E
        E = np.exp(z)
        l, g = loss_and_grad(E)
        # If the jax_solver diverged (NaN loss), return a large finite penalty
        # with zero gradient so L-BFGS-B backtracks instead of propagating NaN.
        if not np.isfinite(l) or not bool(np.all(np.isfinite(g))):
            return 1e20, onp.zeros_like(z)
        return float(l), onp.array(g) * onp.array(E)

    z0 = onp.log(E_init_per_cell)
    l0, _ = fg(z0)
    print(f"  iter  0: loss={l0:.6e}")
    # Bounds keep E in a physical range so the iterative jax_solver doesn't hit a
    # singular/extreme system mid-optimization (unbounded log-L-BFGS can drive
    # some cells to extreme E -> bicgstab divergence).
    lb = onp.log(onp.full_like(z0, 1.0e3))   # E >= 1 GPa
    ub = onp.log(onp.full_like(z0, 5.0e5))   # E <= 500 GPa
    res = sp_min(fun=fg, x0=z0, jac=True, method='L-BFGS-B',
                 bounds=list(zip(lb, ub)),
                 options={'maxiter': max_iter, 'disp': False, 'gtol': 1e-12})
    E_rec = onp.exp(res.x)
    print(f"  done in {res.nit} iters: loss={res.fun:.6e}")
    return E_rec, res


def l2_relative_error(E_rec, E_true):
    """L2 relative error of the recovered field vs truth (proposal §4.1 metric)."""
    return float(onp.linalg.norm(E_rec - E_true) / onp.linalg.norm(E_true))


def run_all_fields():
    """Run inversion on all three preset fields, report L2 errors."""
    os.makedirs(MESH_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 64)
    print("Gap-1: Heterogeneous E-field inversion (elastic, the thesis core)")
    print("=" * 64)

    results = {}
    mesh_n = 6
    for ftype in ['layered', 'inclusion', 'random']:
        truth = generate_field_truth(ftype, Nx=mesh_n, Ny=mesh_n, Nz=mesh_n,
                                     obs_fraction=0.4, data_dir=MESH_DIR)
        # Homogeneous initial guess at the mean — hardest case (no prior info).
        E_init = float(np.mean(truth['truth_E'])) * onp.ones(len(truth['truth_E']))
        E_rec, _ = invert_field(truth, E_init, max_iter=80, log_every=20, mesh_n=mesh_n)
        err = l2_relative_error(E_rec, truth['truth_E'])
        results[ftype] = {'truth_E': truth['truth_E'], 'recovered_E': E_rec,
                          'l2_err': err}
        print(f"  -> {ftype}: L2 relative error = {err*100:.2f}%")

    print("\n" + "=" * 64)
    print(f"Total time: {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    run_all_fields()
