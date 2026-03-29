"""
Gradient verification for heterogeneous per-element E field inversion.

Verifies that AD gradients through ad_wrapper with per-element E field
match central finite differences. Uses a small mesh and elastic regime
to avoid yield-surface nonsmoothness.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import rectangle_mesh, Mesh
from src.models.drucker_prager_2d import DruckerPragerPlasticity2D


class HeterogeneousDP2DForTest(DruckerPragerPlasticity2D):
    """Minimal heterogeneous DP2D for gradient testing."""

    def set_params(self, E_field):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        E_quad = np.repeat(E_field[:, None, None], nq, axis=1)
        self.internal_vars[2] = E_quad


def _setup_problem(Nx=4, Ny=4, Lx=10., Ly=10., displacement=-0.005,
                   k_fixed=50.0):
    """Create mesh, problem, and AD-wrapped forward function."""
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'],
                ele_type='QUAD4')

    def bottom(p):
        return np.isclose(p[1], 0., atol=1e-5)

    def top(p):
        return np.isclose(p[1], Ly, atol=1e-5)

    def corner(p):
        return np.logical_and(np.isclose(p[0], 0., atol=1e-5),
                              np.isclose(p[1], 0., atol=1e-5))

    bc_info = [
        [bottom, top, corner],
        [1, 1, 0],
        [lambda p: 0., lambda p, _d=displacement: _d, lambda p: 0.],
    ]

    problem = HeterogeneousDP2DForTest(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        E=70000., nu=0.3, alpha=0.3, k=k_fixed,
    )
    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd = ad_wrapper(problem, solver_options=solver_options,
                     adjoint_solver_options=solver_options)
    return problem, fwd


def run_heterogeneous_grad_test():
    """Test AD vs FD gradients for per-element E field."""
    Nx, Ny = 4, 4
    num_cells = Nx * Ny

    problem, fwd = _setup_problem(Nx=Nx, Ny=Ny)

    def loss_fn(E_field):
        sol = fwd(E_field)[0]
        return np.sum(sol ** 2)

    print("\n" + "=" * 60)
    print("Heterogeneous E field — AD vs FD gradient test")
    print(f"  Mesh: {Nx}×{Ny} = {num_cells} elements")
    print("=" * 60)

    # Base E field (non-uniform to be more realistic)
    E_base = onp.linspace(50000., 90000., num_cells)
    E_base_jax = np.array(E_base)

    # --- AD gradient ---
    t0 = time.time()
    loss_val, grad_ad = jax.value_and_grad(loss_fn)(E_base_jax)
    t_ad = time.time() - t0
    print(f"\nAD gradient computed in {t_ad:.3f}s")
    print(f"  Loss: {float(loss_val):.6e}")
    print(f"  ||grad_ad||: {float(np.linalg.norm(grad_ad)):.6e}")

    # --- FD gradient for selected elements ---
    test_indices = [0, 5, 10, 15]
    eps = 50.0  # FD step size

    print(f"\nFinite differences (central, eps={eps})...")
    grad_fd = onp.zeros(num_cells)
    all_pass = True

    for idx in test_indices:
        E_plus = onp.array(E_base)
        E_minus = onp.array(E_base)
        E_plus[idx] += eps
        E_minus[idx] -= eps

        loss_plus = float(loss_fn(np.array(E_plus)))
        loss_minus = float(loss_fn(np.array(E_minus)))
        grad_fd[idx] = (loss_plus - loss_minus) / (2. * eps)

        ad_val = float(grad_ad[idx])
        fd_val = grad_fd[idx]
        abs_err = abs(ad_val - fd_val)
        denom = max(abs(ad_val), abs(fd_val), 1e-12)
        rel_err = abs_err / denom

        status = "✓" if (rel_err < 1e-2 or abs_err < 1e-6) else "✗"
        if status == "✗":
            all_pass = False

        print(f"  Element {idx:3d}: AD={ad_val:+.6e}  FD={fd_val:+.6e}  "
              f"rel_err={rel_err:.2e}  {status}")

    # --- Directional derivative test ---
    print("\nDirectional derivative test (random direction)...")
    key = jax.random.PRNGKey(0)
    direction = jax.random.normal(key, (num_cells,))
    direction = direction / np.linalg.norm(direction)

    ad_directional = float(np.dot(grad_ad, direction))

    eps_dir = 10.0
    E_plus_dir = E_base + eps_dir * onp.array(direction)
    E_minus_dir = E_base - eps_dir * onp.array(direction)
    loss_plus_dir = float(loss_fn(np.array(E_plus_dir)))
    loss_minus_dir = float(loss_fn(np.array(E_minus_dir)))
    fd_directional = (loss_plus_dir - loss_minus_dir) / (2. * eps_dir)

    dir_abs_err = abs(ad_directional - fd_directional)
    dir_denom = max(abs(ad_directional), abs(fd_directional), 1e-12)
    dir_rel_err = dir_abs_err / dir_denom

    print(f"  AD directional:  {ad_directional:+.6e}")
    print(f"  FD directional:  {fd_directional:+.6e}")
    print(f"  Relative error:  {dir_rel_err:.2e}")

    dir_pass = dir_rel_err < 1e-2 or dir_abs_err < 1e-6
    if not dir_pass:
        all_pass = False
    print(f"  {'✓' if dir_pass else '✗'} Directional derivative test")

    # Assertions
    assert all_pass, "Some gradient checks failed!"
    print("\n  ✓ All AD vs FD gradient checks passed")


def test_heterogeneous_gradients():
    """Pytest entry point."""
    run_heterogeneous_grad_test()


if __name__ == "__main__":
    test_heterogeneous_gradients()
