"""
Drucker-Prager Plasticity Model Definition

This module implements the Drucker-Prager yield criterion using JAX-FEM.

E and k are passed through internal_vars (indices 2 and 3) so that
ad_wrapper can differentiate through them correctly.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys

# Ensure jax-fem-main is in path
# Assuming the project root is 2 levels up from src/models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem

class DruckerPragerPlasticity(Problem):
    """
    Drucker-Prager plasticity with hyperbolic apex regularization.
    Inherits from JAX-FEM Problem class.
    """
    def __init__(self, mesh, vec=3, dim=3, dirichlet_bc_info=None,
                 E=70.0e3, nu=0.3, alpha=0.3, k=250.0, a=None):
        """
        Args:
            mesh: The FEM mesh
            vec: Vector dimension of the variable
            dim: Spatial dimension
            dirichlet_bc_info: BC configuration
            E: Young's Modulus
            nu: Poisson's Ratio
            alpha: Friction coefficient
            k: Cohesion parameter
            a: Apex regularization parameter (default: 0.01*k)
        """
        # Set material params BEFORE super().__init__ because __post_init__
        # calls custom_init() which needs self.E, self.k, etc.
        self.E = E
        self.nu = nu
        self.alpha = alpha
        self.k = k
        self.a = a if a is not None else 0.01 * k
        self._a_ratio = self.a / self.k
        super().__init__(mesh, vec=vec, dim=dim, dirichlet_bc_info=dirichlet_bc_info)

    def custom_init(self):
        """Initialize internal variables for stress and strain history."""
        self.fe = self.fes[0]
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.epsilons_old = np.zeros((nc, nq, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        E_field = np.full((nc, nq, 1), self.E)
        k_field = np.full((nc, nq, 1), self.k)
        self.internal_vars = [self.sigmas_old, self.epsilons_old, E_field, k_field]

    def set_params(self, params):
        """Update E and k fields from a parameter array [E, k]."""
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.internal_vars[2] = np.full((nc, nq, 1), params[0])
        self.internal_vars[3] = np.full((nc, nq, 1), params[1])

    def get_tensor_map(self):
        """Return the stress computation function for the FEM solver."""
        nu = self.nu
        alpha = self.alpha
        dim = self.dim
        a_ratio = self._a_ratio

        def safe_divide(x, y):
            tiny = 1e-30
            y_safe = np.where(np.abs(y) < tiny, 1., y)
            return np.where(np.abs(y) < tiny, 0., x / y_safe)

        def stress_return_map(u_grad, sigma_old, epsilon_old, E_arr, k_arr):
            E = E_arr[0]
            k = k_arr[0]
            a = a_ratio * k

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

            n_dev = safe_divide(s_dev, sqrt_J2_reg)
            denom = mu + 9. * bulk_k * alpha * alpha
            delta_lambda = safe_divide(f_yield_plus, denom)
            sigma = sigma_trial - delta_lambda * (
                mu * n_dev + 3. * bulk_k * alpha * np.eye(dim)
            )

            sigma_apex = (k / (3. * alpha)) * np.eye(dim)
            at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
            sigma = np.where(at_apex, sigma_apex, sigma)

            return sigma

        return stress_return_map

    def get_maps(self):
        """Define strain and DP return mapping functions (backward compat)."""
        nu = self.nu
        alpha = self.alpha
        dim = self.dim
        a_ratio = self._a_ratio

        def safe_divide(x, y):
            tiny = 1e-30
            y_safe = np.where(np.abs(y) < tiny, 1., y)
            return np.where(np.abs(y) < tiny, 0., x / y_safe)

        def strain(u_grad):
            return 0.5 * (u_grad + u_grad.T)

        def stress_return_map(u_grad, sigma_old, epsilon_old, E_arr, k_arr):
            E = E_arr[0]
            k = k_arr[0]
            a = a_ratio * k

            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            bulk_k = lmbda + 2. * mu / 3.

            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = lmbda * np.trace(epsilon_inc) * np.eye(dim) + 2. * mu * epsilon_inc + sigma_old

            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(dim)
            J2 = 0.5 * np.sum(s_dev * s_dev)

            sqrt_J2_reg = np.sqrt(J2 + a * a)
            f_yield = sqrt_J2_reg + alpha * I1 - k

            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)

            n_dev = safe_divide(s_dev, sqrt_J2_reg)
            denom = mu + 9. * bulk_k * alpha * alpha
            delta_lambda = safe_divide(f_yield_plus, denom)
            sigma = sigma_trial - delta_lambda * (
                mu * n_dev + 3. * bulk_k * alpha * np.eye(dim)
            )

            sigma_apex = (k / (3. * alpha)) * np.eye(dim)
            at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
            sigma = np.where(at_apex, sigma_apex, sigma)

            return sigma

        return strain, stress_return_map

    def stress_strain_fns(self):
        strain, stress_return_map = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol):
        u_grads = self.fe.sol_to_grad(sol)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()
        self.sigmas_old = vmap_stress_rm(
            u_grads, self.sigmas_old, self.epsilons_old,
            self.internal_vars[2], self.internal_vars[3],
        )
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = [self.sigmas_old, self.epsilons_old,
                              self.internal_vars[2], self.internal_vars[3]]

    def compute_avg_stress(self):
        sigma = np.sum(
            self.sigmas_old.reshape(-1, self.fe.vec, self.dim) * 
            self.fe.JxW.reshape(-1)[:, None, None], 
            axis=0
        )
        vol = np.sum(self.fe.JxW)
        return sigma / vol
