"""
2D Plane Strain Drucker-Prager Plasticity Model

This module implements the Drucker-Prager yield criterion for 2D plane strain
using JAX-FEM. The return mapping is performed in full 3D (since σ_33 ≠ 0
even though ε_33 = 0), and only the 2×2 in-plane stress is returned to the
FEM kernel.

E and k are passed through internal_vars (indices 2 and 3) so that
ad_wrapper can differentiate through them correctly.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem


class DruckerPragerPlasticity2D(Problem):
    """
    2D plane strain Drucker-Prager plasticity with hyperbolic apex
    regularization.  Inherits from JAX-FEM Problem class.

    Internal history variables (sigma_old, epsilon_old) are stored as full
    3×3 tensors so that the out-of-plane stress σ_33 is tracked correctly.
    """

    def __init__(self, mesh, vec=2, dim=2, ele_type='QUAD4',
                 dirichlet_bc_info=None,
                 E=70.0e3, nu=0.3, alpha=0.3, k=250.0, a=None):
        super().__init__(mesh, vec=vec, dim=dim, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info)
        self.E = E
        self.nu = nu
        self.alpha = alpha
        self.k = k
        self.a = a if a is not None else 0.01 * k
        self._a_ratio = self.a / self.k

    def custom_init(self):
        self.fe = self.fes[0]
        nc, nq = len(self.fe.cells), self.fe.num_quads
        # Full 3×3 history tensors (plane strain: ε_33=0 but σ_33≠0)
        self.sigmas_old = np.zeros((nc, nq, 3, 3))
        self.epsilons_old = np.zeros((nc, nq, 3, 3))
        E_field = np.full((nc, nq, 1), self.E)
        k_field = np.full((nc, nq, 1), self.k)
        self.internal_vars = [self.sigmas_old, self.epsilons_old, E_field, k_field]

    def set_params(self, params):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.internal_vars[2] = np.full((nc, nq, 1), params[0])
        self.internal_vars[3] = np.full((nc, nq, 1), params[1])

    def get_tensor_map(self):
        nu = self.nu
        alpha = self.alpha
        a_ratio = self._a_ratio

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

            # Expand 2×2 u_grad to 3×3 (plane strain: row/col 3 = 0)
            u_grad = np.zeros((3, 3))
            u_grad = u_grad.at[:2, :2].set(u_grad_2d)

            # Full 3D return mapping
            epsilon_crt = 0.5 * (u_grad + u_grad.T)
            epsilon_inc = epsilon_crt - epsilon_old_3x3
            sigma_trial = (lmbda * np.trace(epsilon_inc) * np.eye(3)
                           + 2. * mu * epsilon_inc + sigma_old_3x3)

            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(3)
            J2 = 0.5 * np.sum(s_dev * s_dev)

            sqrt_J2_reg = np.sqrt(J2 + a * a)
            f_yield = sqrt_J2_reg + alpha * I1 - k

            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)

            n_dev = safe_divide(s_dev, sqrt_J2_reg)
            denom = mu + 9. * bulk_k * alpha * alpha
            delta_lambda = safe_divide(f_yield_plus, denom)
            sigma_3d = sigma_trial - delta_lambda * (
                mu * n_dev + 3. * bulk_k * alpha * np.eye(3)
            )

            sigma_apex = (k / (3. * alpha)) * np.eye(3)
            at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
            sigma_3d = np.where(at_apex, sigma_apex, sigma_3d)

            # Return only 2×2 in-plane stress for the FEM kernel
            return sigma_3d[:2, :2]

        return stress_return_map

    def get_maps(self):
        nu = self.nu
        alpha = self.alpha
        a_ratio = self._a_ratio

        def safe_divide(x, y):
            tiny = 1e-30
            y_safe = np.where(np.abs(y) < tiny, 1., y)
            return np.where(np.abs(y) < tiny, 0., x / y_safe)

        def strain_2d_to_3d(u_grad_2d):
            """Compute full 3×3 strain from 2×2 u_grad (plane strain)."""
            u_grad = np.zeros((3, 3))
            u_grad = u_grad.at[:2, :2].set(u_grad_2d)
            return 0.5 * (u_grad + u_grad.T)

        def stress_return_map_3d(u_grad_2d, sigma_old_3x3, epsilon_old_3x3, E_arr, k_arr):
            """Full 3D return map; returns 3×3 stress (including σ_33)."""
            E = E_arr[0]
            k = k_arr[0]
            a = a_ratio * k

            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            bulk_k = lmbda + 2. * mu / 3.

            epsilon_crt = strain_2d_to_3d(u_grad_2d)
            epsilon_inc = epsilon_crt - epsilon_old_3x3
            sigma_trial = (lmbda * np.trace(epsilon_inc) * np.eye(3)
                           + 2. * mu * epsilon_inc + sigma_old_3x3)

            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(3)
            J2 = 0.5 * np.sum(s_dev * s_dev)

            sqrt_J2_reg = np.sqrt(J2 + a * a)
            f_yield = sqrt_J2_reg + alpha * I1 - k

            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)

            n_dev = safe_divide(s_dev, sqrt_J2_reg)
            denom = mu + 9. * bulk_k * alpha * alpha
            delta_lambda = safe_divide(f_yield_plus, denom)
            sigma_3d = sigma_trial - delta_lambda * (
                mu * n_dev + 3. * bulk_k * alpha * np.eye(3)
            )

            sigma_apex = (k / (3. * alpha)) * np.eye(3)
            at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
            sigma_3d = np.where(at_apex, sigma_apex, sigma_3d)

            return sigma_3d

        return strain_2d_to_3d, stress_return_map_3d

    def stress_strain_fns(self):
        strain_2d_to_3d, stress_return_map_3d = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain_2d_to_3d))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map_3d))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol):
        u_grads = self.fe.sol_to_grad(sol)  # (nc, nq, 2, 2)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()
        self.sigmas_old = vmap_stress_rm(
            u_grads, self.sigmas_old, self.epsilons_old,
            self.internal_vars[2], self.internal_vars[3],
        )
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = [self.sigmas_old, self.epsilons_old,
                              self.internal_vars[2], self.internal_vars[3]]

    def compute_avg_stress(self):
        """Return volume-averaged 3×3 stress (includes σ_33)."""
        sigma = np.sum(
            self.sigmas_old.reshape(-1, 3, 3) *
            self.fe.JxW.reshape(-1)[:, None, None],
            axis=0,
        )
        vol = np.sum(self.fe.JxW)
        return sigma / vol
