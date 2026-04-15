"""
Differentiability test for 2D Mohr-Coulomb: AD vs FD.

Verifies that automatic differentiation through the MC return mapping
produces gradients consistent with finite differences for:
  - ∂L/∂c  (cohesion sensitivity)
  - ∂L/∂φ  (friction angle sensitivity)

Loss = volume-averaged σ_yy (vertical stress).
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import rectangle_mesh, Mesh
from src.models.mohr_coulomb_2d import MohrCoulombPlasticity2D, _mc_return_map_3d


def create_mesh_and_bc(displacement, Lx=10., Ly=10., Nx=5, Ny=5):
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')

    def bottom(p):
        return np.isclose(p[1], 0., atol=1e-5)

    def corner(p):
        return np.logical_and(np.isclose(p[0], 0., atol=1e-5),
                              np.isclose(p[1], 0., atol=1e-5))

    def top(p):
        return np.isclose(p[1], Ly, atol=1e-5)

    bc = [
        [bottom, corner, top],
        [1, 0, 1],
        [lambda p: 0., lambda p: 0., lambda p, _d=displacement: _d],
    ]
    return mesh, bc


def volume_avg_sigma_yy(fe, sol, sigma_old, epsilon_old, c, phi, E, nu, psi,
                         transition_angle, a_apex_ratio):
    """Compute volume-averaged σ_yy using MC return map with explicit JAX args."""
    u_grads = fe.sol_to_grad(sol)

    def single_qp_stress(u_grad_2d, sig_old, eps_old):
        sigma_3d = _mc_return_map_3d(
            u_grad_2d, sig_old, eps_old,
            E, nu, c, phi, psi, transition_angle, a_apex_ratio,
        )
        return sigma_3d

    vmap_stress = jax.vmap(jax.vmap(single_qp_stress))
    sigmas = vmap_stress(u_grads, sigma_old, epsilon_old)

    sigma_yy_avg = np.sum(sigmas[:, :, 1, 1] * fe.JxW) / np.sum(fe.JxW)
    return sigma_yy_avg


def fd_gradient(loss_fn, params, eps_list):
    """Central-difference gradient."""
    grad = []
    for i in range(len(params)):
        p_plus = params.at[i].set(params[i] + eps_list[i])
        p_minus = params.at[i].set(params[i] - eps_list[i])
        grad.append((loss_fn(p_plus) - loss_fn(p_minus)) / (2 * eps_list[i]))
    return np.array(grad)


def test_mc_gradient(displacement=-0.015, regime='elastic',
                     c_true=50.0, phi_deg=30.0):
    """Test AD vs FD gradient for MC model."""
    print(f"\n{'=' * 60}")
    print(f"MC 2D gradient test — regime: {regime}, disp: {displacement}")
    print(f"  c={c_true}, phi={phi_deg}°")
    print(f"{'=' * 60}")

    mesh, bc = create_mesh_and_bc(displacement)
    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

    E_val = 70000.0
    nu_val = 0.3
    phi_true = np.radians(phi_deg)
    psi_deg = phi_deg / 2.0
    psi_val = np.radians(psi_deg)
    transition_angle = np.radians(25.0)
    a_apex_ratio = 0.01

    problem = MohrCoulombPlasticity2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc,
        E=E_val, nu=nu_val, c=c_true,
        phi_deg=phi_deg, psi_deg=psi_deg,
        transition_angle=25.0,
    )

    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    def loss_fn(params):
        c, phi = params[0], params[1]
        sol = fwd_pred(params)[0]
        return volume_avg_sigma_yy(
            problem.fe, sol, problem.sigmas_old, problem.epsilons_old,
            c, phi, E_val, nu_val, psi_val, transition_angle, a_apex_ratio,
        )

    params = np.array([c_true, phi_true])

    # AD gradient
    loss_val, ad_grad = jax.value_and_grad(loss_fn)(params)
    print(f"  Loss (σ_yy avg): {float(loss_val):.6e}")
    print(f"  AD grad: [∂L/∂c={float(ad_grad[0]):.6e}, ∂L/∂φ={float(ad_grad[1]):.6e}]")

    # FD gradient
    fd_eps = [0.5, 0.001]
    fd_grad = fd_gradient(loss_fn, params, fd_eps)
    print(f"  FD grad: [∂L/∂c={float(fd_grad[0]):.6e}, ∂L/∂φ={float(fd_grad[1]):.6e}]")

    # Compare
    all_pass = True
    for name, ad_g, fd_g in [("∂L/∂c", ad_grad[0], fd_grad[0]),
                               ("∂L/∂φ", ad_grad[1], fd_grad[1])]:
        abs_err = abs(float(ad_g - fd_g))
        ref = max(abs(float(ad_g)), abs(float(fd_g)), 1e-30)
        rel_err = abs_err / ref
        status = "✅" if rel_err < 0.05 else "⚠️" if rel_err < 0.2 else "❌"
        print(f"  {name}: abs_err={abs_err:.4e}, rel_err={rel_err:.4e} {status}")
        if rel_err >= 0.05 and abs_err > 1e-6:
            all_pass = False

    return all_pass


if __name__ == "__main__":
    # Elastic regime: small displacement, no yielding
    test_mc_gradient(displacement=-0.005, regime='elastic', c_true=250.0)

    # Plastic regime: displacement large enough to trigger yielding with c=50
    test_mc_gradient(displacement=-0.03, regime='plastic', c_true=50.0)

    # Plastic regime with lower phi
    test_mc_gradient(displacement=-0.02, regime='plastic', c_true=50.0, phi_deg=20.0)

    print("\nDone.")
