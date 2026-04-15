"""
Differentiability test for 2D plane strain Drucker-Prager: AD vs FD.

Loss = volume-averaged sigma_yy.  Stress post-processing uses the same 3D
return map as the FE residual but takes (E, k) as explicit JAX arguments so
gradients combine correctly with ad_wrapper's implicit differentiation.
"""

import jax
import jax.numpy as np
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import rectangle_mesh, Mesh


# ---------------------------------------------------------------------------
# Standalone 2D plane-strain DP return map (for loss post-processing)
# ---------------------------------------------------------------------------

def _safe_divide(x, y):
    tiny = 1e-30
    y_safe = np.where(np.abs(y) < tiny, 1., y)
    return np.where(np.abs(y) < tiny, 0., x / y_safe)


def _softplus(x, sharpness=100.):
    """Smooth approximation of max(x, 0) for AD-friendly yield check."""
    sx = sharpness * x
    return np.where(sx > 20., x, np.log1p(np.exp(np.clip(sx, -30., 20.))) / sharpness)


def stress_return_dp_2d(u_grad_2d, sigma_old_3x3, epsilon_old_3x3, E, k):
    """Full 3D return map from 2×2 u_grad (plane strain). Returns 3×3 stress."""
    nu = 0.3
    alpha = 0.3
    a = 0.1 * k

    mu = E / (2. * (1. + nu))
    lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
    bulk_k = lmbda + 2. * mu / 3.

    u_grad = np.zeros((3, 3))
    u_grad = u_grad.at[:2, :2].set(u_grad_2d)

    epsilon_crt = 0.5 * (u_grad + u_grad.T)
    epsilon_inc = epsilon_crt - epsilon_old_3x3
    sigma_trial = (lmbda * np.trace(epsilon_inc) * np.eye(3)
                   + 2. * mu * epsilon_inc + sigma_old_3x3)

    I1 = np.trace(sigma_trial)
    s_dev = sigma_trial - (I1 / 3.) * np.eye(3)
    J2 = 0.5 * np.sum(s_dev * s_dev)

    sqrt_J2_reg = np.sqrt(J2 + a * a)
    f_yield = sqrt_J2_reg + alpha * I1 - k

    f_yield_plus = _softplus(f_yield)
    n_dev = _safe_divide(s_dev, sqrt_J2_reg)
    denom = mu + 9. * bulk_k * alpha * alpha
    delta_lambda = _safe_divide(f_yield_plus, denom)
    sigma_3d = sigma_trial - delta_lambda * (
        mu * n_dev + 3. * bulk_k * alpha * np.eye(3)
    )

    return sigma_3d


# ---------------------------------------------------------------------------
# Problem class used inside ad_wrapper
# ---------------------------------------------------------------------------

class DifferentiableDruckerPrager2D(Problem):
    """2D plane-strain DP problem for use inside ad_wrapper."""

    def custom_init(self):
        self.fe = self.fes[0]
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.sigmas_old = np.zeros((nc, nq, 3, 3))
        self.epsilons_old = np.zeros((nc, nq, 3, 3))
        E_field = np.full((nc, nq, 1), 70000.0)
        k_field = np.full((nc, nq, 1), 50.0)
        self.internal_vars = [self.sigmas_old, self.epsilons_old, E_field, k_field]

    def set_params(self, params):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.internal_vars[2] = np.full((nc, nq, 1), params[0])
        self.internal_vars[3] = np.full((nc, nq, 1), params[1])

    def get_tensor_map(self):
        nu = 0.3
        alpha = 0.3
        a_ratio = 0.1  # a = 0.1*k  (matching the standalone fn)

        def safe_divide(x, y):
            tiny = 1e-30
            y_safe = np.where(np.abs(y) < tiny, 1., y)
            return np.where(np.abs(y) < tiny, 0., x / y_safe)

        def stress_return_map(u_grad_2d, sigma_old_3x3, epsilon_old_3x3, E_arr, k_arr):
            E = E_arr[0]
            k = k_arr[0]
            a = a_ratio * k

            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            bulk_k = lmbda + 2. * mu / 3.

            u_grad = np.zeros((3, 3))
            u_grad = u_grad.at[:2, :2].set(u_grad_2d)

            epsilon_crt = 0.5 * (u_grad + u_grad.T)
            epsilon_inc = epsilon_crt - epsilon_old_3x3
            sigma_trial = (lmbda * np.trace(epsilon_inc) * np.eye(3)
                           + 2. * mu * epsilon_inc + sigma_old_3x3)

            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(3)
            J2 = 0.5 * np.sum(s_dev * s_dev)

            sqrt_J2_reg = np.sqrt(J2 + a * a)
            f_yield = sqrt_J2_reg + alpha * I1 - k

            f_yield_plus = _softplus(f_yield)
            n_dev = safe_divide(s_dev, sqrt_J2_reg)
            denom = mu + 9. * bulk_k * alpha * alpha
            delta_lambda = safe_divide(f_yield_plus, denom)
            sigma_3d = sigma_trial - delta_lambda * (
                mu * n_dev + 3. * bulk_k * alpha * np.eye(3)
            )

            return sigma_3d[:2, :2]

        return stress_return_map


# ---------------------------------------------------------------------------
# Loss: volume-averaged sigma_yy (full 3D stress for diagnostics)
# ---------------------------------------------------------------------------

def volume_avg_sigma_yy(fe, sol, E, k):
    u_grads = fe.sol_to_grad(sol)  # (nc, nq, 2, 2)
    c, q = u_grads.shape[:2]
    sigma_old = np.zeros((c, q, 3, 3))
    epsilon_old = np.zeros_like(sigma_old)

    def one_quad(ug, so, eo):
        return stress_return_dp_2d(ug, so, eo, E, k)

    sig_3d = jax.vmap(jax.vmap(one_quad))(u_grads, sigma_old, epsilon_old)  # (c,q,3,3)
    JxW = fe.JxW
    sig_yy = sig_3d[..., 1, 1]
    return np.sum(sig_yy * JxW) / np.sum(JxW)


def volume_avg_stress_3d(fe, sol, E, k):
    """Return volume-averaged full 3×3 stress (for plane-strain diagnostics)."""
    u_grads = fe.sol_to_grad(sol)
    c, q = u_grads.shape[:2]
    sigma_old = np.zeros((c, q, 3, 3))
    epsilon_old = np.zeros_like(sigma_old)

    def one_quad(ug, so, eo):
        return stress_return_dp_2d(ug, so, eo, E, k)

    sig_3d = jax.vmap(jax.vmap(one_quad))(u_grads, sigma_old, epsilon_old)
    JxW = fe.JxW
    return np.sum(sig_3d * JxW[:, :, None, None], axis=(0, 1)) / np.sum(JxW)


# ---------------------------------------------------------------------------
# Gradient test
# ---------------------------------------------------------------------------

def run_dp2d_grad_test(displacement):
    Lx, Ly = 10., 10.
    Nx, Ny = 2, 2

    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')

    def bottom(p):
        return np.isclose(p[1], 0.)

    def top(p):
        return np.isclose(p[1], Ly)

    def corner(p):
        return np.logical_and(np.isclose(p[0], 0.), np.isclose(p[1], 0.))

    dirichlet_bc_info = [
        [bottom, top, corner],
        [1, 1, 0],
        [lambda p: 0., lambda p: displacement, lambda p: 0.],
    ]

    problem = DifferentiableDruckerPrager2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=dirichlet_bc_info,
    )

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    def loss_fn(params):
        sol_list = fwd_pred(params)
        E, k = params[0], params[1]
        return volume_avg_sigma_yy(problem.fe, sol_list[0], E, k)

    print("\n" + "=" * 60)
    print("2D Plane Strain DP  — AD vs FD (loss = vol avg σ_yy)")
    print(f"Displacement: {displacement}")
    print("=" * 60)

    params_init = np.array([70000.0, 50.0])

    # --- Plane-strain sanity check (elastic regime, small disp) -----------
    if abs(displacement) < 0.02:
        sol_list = fwd_pred(params_init)
        sig_avg = volume_avg_stress_3d(problem.fe, sol_list[0],
                                       params_init[0], params_init[1])
        nu = 0.3
        sig_xx, sig_yy, sig_zz = sig_avg[0, 0], sig_avg[1, 1], sig_avg[2, 2]
        sig_zz_expected = nu * (sig_xx + sig_yy)
        print(f"\n  Plane-strain check (elastic):")
        print(f"    σ_xx = {sig_xx:.4e}")
        print(f"    σ_yy = {sig_yy:.4e}")
        print(f"    σ_zz = {sig_zz:.4e}")
        print(f"    ν(σ_xx+σ_yy) = {sig_zz_expected:.4e}")
        zz_err = np.abs(sig_zz - sig_zz_expected) / np.maximum(np.abs(sig_zz_expected), 1e-12)
        print(f"    Relative error σ_zz: {zz_err:.2e}")
        assert zz_err < 1e-3, f"σ_zz plane-strain check failed: {zz_err}"
        print("    ✓ σ_zz = ν(σ_xx + σ_yy) verified")

    # --- AD gradient -------------------------------------------------------
    t0 = time.time()
    loss_val, grad_ad = jax.value_and_grad(loss_fn)(params_init)
    t1 = time.time()
    print(f"\nAD gradient computed in {t1 - t0:.4f}s")
    print(f"  Loss: {loss_val:.6e}")
    print(f"  dLoss/dE (AD): {grad_ad[0]:.8e}")
    print(f"  dLoss/dk (AD): {grad_ad[1]:.8e}")

    # --- FD gradient -------------------------------------------------------
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
    print("\n  ✓ AD vs FD gradients match")


def test_dp2d_gradients():
    for disp in [-0.01, -0.02, -0.03]:
        run_dp2d_grad_test(disp)


if __name__ == "__main__":
    test_dp2d_gradients()
