#!/usr/bin/env python
"""
AD vs FD single-gradient timing comparison for MC c(x) inversion.

Measures wall time for ONE gradient evaluation at increasing mesh sizes.
No optimization loop — just timing the gradient itself.
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

from common import InversionHeterogeneousMC2D, make_heterogeneous_loss
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import rectangle_mesh, Mesh


SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}


def build(Nx, Ny):
    Lx, Ly = 10., 10.
    disp = -0.03
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')

    def bottom(p): return np.isclose(p[1], 0., atol=1e-5)
    def top(p): return np.isclose(p[1], Ly, atol=1e-5)
    def corner(p): return np.logical_and(
        np.isclose(p[0], 0., atol=1e-5), np.isclose(p[1], 0., atol=1e-5))
    bc = [[bottom, top, corner], [1, 1, 0],
          [lambda p: 0., lambda p, _d=disp: _d, lambda p: 0.]]

    nc = Nx * Ny
    c_true = onp.where(onp.arange(nc) < nc // 2, 30., 70.)

    # Observation
    prob_obs = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=bc,
        E=70000., nu=0.3, c=50., phi_deg=30., psi_deg=15.)
    prob_obs.set_params(np.array(c_true))
    u_obs = solver(prob_obs, solver_options=SOLVER_OPTIONS)[0]
    obs_idx = onp.arange(u_obs.shape[0])

    # Inversion problem
    meshio2 = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh2 = Mesh(meshio2.points, meshio2.cells_dict['quad'], ele_type='QUAD4')
    bc2 = [[bottom, top, corner], [1, 1, 0],
           [lambda p: 0., lambda p, _d=disp: _d, lambda p: 0.]]
    prob_inv = InversionHeterogeneousMC2D(
        mesh2, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=bc2,
        E=70000., nu=0.3, c=50., phi_deg=30., psi_deg=15.)
    fwd = ad_wrapper(prob_inv, solver_options=SOLVER_OPTIONS,
                     adjoint_solver_options=SOLVER_OPTIONS)
    loss_fn = make_heterogeneous_loss(fwd, u_obs, obs_idx)
    x0 = np.log(np.full(nc, 50.))
    return loss_fn, x0, nc


def time_ad(loss_fn, x0):
    vg = jax.value_and_grad(loss_fn)
    vg(x0)  # warmup/compile
    t0 = time.time()
    vg(x0)
    return time.time() - t0


def time_fd(loss_fn, x0, nc):
    loss_fn(x0)  # warmup
    eps = 0.01
    t0 = time.time()
    for i in range(nc):
        xp = x0.at[i].set(x0[i] + eps)
        xm = x0.at[i].set(x0[i] - eps)
        _ = loss_fn(xp)
        _ = loss_fn(xm)
    return time.time() - t0


def main():
    print("=" * 65)
    print("MC c(x) Inversion: Single Gradient Timing — AD vs FD")
    print("=" * 65)
    print(f"  {'Mesh':>8s}  {'N_cells':>7s}  {'AD [s]':>8s}  {'FD [s]':>8s}  "
          f"{'Speedup':>8s}  {'FD fwd calls':>12s}")
    print(f"  {'-'*57}")

    for Nx, Ny in [(3, 3), (5, 5), (8, 8), (10, 10)]:
        nc = Nx * Ny
        loss_fn, x0, _ = build(Nx, Ny)

        t_ad = time_ad(loss_fn, x0)
        t_fd = time_fd(loss_fn, x0, nc)
        speedup = t_fd / max(t_ad, 1e-6)

        print(f"  {Nx}×{Ny:>2d}     {nc:>5d}    {t_ad:7.3f}   {t_fd:7.1f}   "
              f"{speedup:7.0f}×    {2*nc:>10d}")

    print("=" * 65)


if __name__ == "__main__":
    main()
