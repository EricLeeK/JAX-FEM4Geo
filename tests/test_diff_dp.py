"""
Differentiability test for Drucker-Prager: AD vs FD.

Loss = volume-averaged sigma_zz. Stress post-processing uses the same map as the
FE residual but takes (E, k) as explicit JAX arguments so gradients combine
correctly with ad_wrapper's implicit differentiation.
"""

import jax
import jax.numpy as np
import os
import sys
import time

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh


def _safe_divide(x, y):
    tiny = 1e-30
    y_safe = np.where(np.abs(y) < tiny, 1., y)
    return np.where(np.abs(y) < tiny, 0., x / y_safe)


def stress_return_dp(u_grad, sigma_old, epsilon_old, E, k, dim):
    """DP return map; E, k must be JAX scalars/arrays for AD through loss."""
    nu = 0.3
    alpha = 0.3
    a = 0.1 * k
    mu = E / (2. * (1. + nu))
    lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
    bulk_k = lmbda + 2. * mu / 3.

    epsilon_crt = 0.5 * (u_grad + u_grad.T)
    epsilon_inc = epsilon_crt - epsilon_old
    sigma_trial = lmbda * np.trace(epsilon_inc) * np.eye(dim) + 2. * mu * epsilon_inc + sigma_old

    I1 = np.trace(sigma_trial)
    s_dev = sigma_trial - (I1 / 3.) * np.eye(dim)
    J2 = 0.5 * np.sum(s_dev * s_dev)

    sqrt_J2_reg = np.sqrt(J2 + a * a)
    f_yield = sqrt_J2_reg + alpha * I1 - k

    f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
    n_dev = _safe_divide(s_dev, sqrt_J2_reg)
    denom = mu + 9. * bulk_k * alpha * alpha
    delta_lambda = _safe_divide(f_yield_plus, denom)
    sigma = sigma_trial - delta_lambda * (
        mu * n_dev + 3. * bulk_k * alpha * np.eye(dim)
    )

    sigma_apex = (k / (3. * alpha)) * np.eye(dim)
    at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
    sigma = np.where(at_apex, sigma_apex, sigma)
    return sigma


class DifferentiableDruckerPrager(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.epsilons_old = np.zeros((nc, nq, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        E_field = np.full((nc, nq, 1), 70000.0)
        k_field = np.full((nc, nq, 1), 50.0)
        self.internal_vars = [self.sigmas_old, self.epsilons_old, E_field, k_field]

    def set_params(self, params):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.internal_vars[2] = np.full((nc, nq, 1), params[0])
        self.internal_vars[3] = np.full((nc, nq, 1), params[1])

    def get_tensor_map(self):
        dim = self.dim

        def stress_return_map(u_grad, sigma_old, epsilon_old, E_arr, k_arr):
            return stress_return_dp(
                u_grad, sigma_old, epsilon_old, E_arr[0], k_arr[0], dim,
            )

        return stress_return_map


def volume_avg_sigma_zz(fe, sol, E, k):
    u_grads = fe.sol_to_grad(sol)
    c, q, _, dim = u_grads.shape
    sigma_old = np.zeros((c, q, dim, dim))
    epsilon_old = np.zeros_like(sigma_old)

    def one_quad(ug, so, eo):
        return stress_return_dp(ug, so, eo, E, k, dim)

    sig = jax.vmap(jax.vmap(one_quad))(u_grads, sigma_old, epsilon_old)
    JxW = fe.JxW
    sig_zz = sig[..., 2, 2]
    return np.sum(sig_zz * JxW) / np.sum(JxW)


def run_dp_grad_test(displacement):
    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 2, 2, 2

    data_dir = os.path.join(project_root, 'results', 'test_output_dp')
    os.makedirs(data_dir, exist_ok=True)

    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(
        Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz,
        data_dir=data_dir, ele_type=ele_type,
    )
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def bottom(p):
        return np.isclose(p[2], 0.)

    def top(p):
        return np.isclose(p[2], Lz)

    def corner(p):
        return np.logical_and(
            np.logical_and(np.isclose(p[0], 0.), np.isclose(p[1], 0.)),
            np.isclose(p[2], 0.),
        )

    dirichlet_bc_info = [
        [bottom, top, corner, corner],
        [2, 2, 0, 1],
        [lambda p: 0., lambda p: displacement, lambda p: 0., lambda p: 0.],
    ]

    problem = DifferentiableDruckerPrager(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options, adjoint_solver_options=solver_options)

    def loss_fn(params):
        sol_list = fwd_pred(params)
        E, k = params[0], params[1]
        return volume_avg_sigma_zz(problem.fe, sol_list[0], E, k)

    print("\n" + "=" * 60)
    print("Drucker-Prager AD vs FD (loss = volume avg sigma_zz)")
    print(f"Displacement: {displacement}")
    print("=" * 60)

    params_init = np.array([70000.0, 50.0])

    t0 = time.time()
    loss_val, grad_ad = jax.value_and_grad(loss_fn)(params_init)
    t1 = time.time()
    print(f"AD gradient computed in {t1 - t0:.4f}s")
    print(f"  Loss: {loss_val:.6e}")
    print(f"  dLoss/dE (AD): {grad_ad[0]:.8e}")
    print(f"  dLoss/dk (AD): {grad_ad[1]:.8e}")

    print("\nFinite differences (central)...")
    eps_E = 100.0
    eps_k = 1.0

    loss_plus_E = loss_fn(params_init + np.array([eps_E, 0.0]))
    loss_minus_E = loss_fn(params_init - np.array([eps_E, 0.0]))
    grad_fd_E = (loss_plus_E - loss_minus_E) / (2. * eps_E)

    loss_plus_k = loss_fn(params_init + np.array([0.0, eps_k]))
    loss_minus_k = loss_fn(params_init - np.array([0.0, eps_k]))
    grad_fd_k = (loss_plus_k - loss_minus_k) / (2. * eps_k)

    print(f"  dLoss/dE (FD): {grad_fd_E:.8e}")
    print(f"  dLoss/dk (FD): {grad_fd_k:.8e}")

    abs_err_E = np.abs(grad_ad[0] - grad_fd_E)
    abs_err_k = np.abs(grad_ad[1] - grad_fd_k)
    err_E = abs_err_E / np.maximum(np.maximum(np.abs(grad_fd_E), np.abs(grad_ad[0])), 1e-12)
    err_k = abs_err_k / np.maximum(np.maximum(np.abs(grad_fd_k), np.abs(grad_ad[1])), 1e-12)

    print("\nRelative errors:")
    print(f"  dLoss/dE: {err_E:.2e}")
    print(f"  dLoss/dk: {err_k:.2e}")

    assert err_E < 1e-2 or abs_err_E < 1e-3, (
        f"dLoss/dE mismatch for displacement={displacement}: rel={err_E}, abs={abs_err_E}"
    )
    assert err_k < 1e-2 or abs_err_k < 1e-3, (
        f"dLoss/dk mismatch for displacement={displacement}: rel={err_k}, abs={abs_err_k}"
    )


def test_dp_gradients():
    for disp in [-0.01, -0.02, -0.03]:
        run_dp_grad_test(disp)


if __name__ == "__main__":
    test_dp_gradients()
