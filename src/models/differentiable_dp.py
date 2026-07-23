"""Single-step differentiable Drucker-Prager problem for parameter inversion.

This is a JAX-FEM ``Problem`` whose forward solve is *single-step* (no stress
history carried across load increments). The single-step formulation keeps the
automatic-differentiation (AD) path clean and is what every existing AD-vs-FD
test in this project validates against. Path-dependent (incremental) inversion
is left to a later milestone.

Parameters are exposed to the AD framework through ``set_params`` so that
``jax_fem.solver.ad_wrapper`` can backpropagate a loss into them. The
invertible parameters are:

    params = [E, k]          # Young's modulus and DP cohesion

``nu``, ``alpha`` and the apex regularization ``a`` are kept fixed (constants
captured in the stress closure) to keep the inverse problem well-posed; they
can be switched on by extending ``set_params`` and the closure below.
"""

import jax
import jax.numpy as np

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem


class DifferentiableDruckerPrager(Problem):
    """Single-step, parameter-dependent Drucker-Prager plasticity.

    The stress return-mapping uses the hyperbolic apex regularization
    ``f = sqrt(J2 + a^2) + alpha*I1 - k`` so that the apex region stays smooth
    and differentiable. All branches use ``np.where`` to remain JAX-compatible.
    """

    def __init__(self, mesh, vec=3, dim=3, dirichlet_bc_info=None,
                 location_fns=None, nu=0.3, alpha=0.3):
        # Fixed material constants (not inverted). The invertible E and k are
        # injected per forward solve via set_params.
        self.nu = nu
        self.alpha = alpha
        # Location fns are required by Problem when surface integrals (Neumann
        # BCs) are used; we accept and forward them.
        super().__init__(mesh, vec=vec, dim=dim,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)

    def custom_init(self):
        """Initialise internal state for a single forward solve.

        Because the problem is single-step, history arrays start at zero every
        solve. They are still allocated (JAX-FEM expects ``internal_vars``) but
        never updated across solves.
        """
        self.fe = self.fes[0]
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads,
                                      self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]
        # Default parameter values (overwritten by set_params before each solve)
        self.E_val = 70.0e3
        self.k_val = 250.0

    def set_params(self, params):
        """Inject invertible parameters. ``params`` is a pytree leaf array."""
        self.E_val = params[0]
        self.k_val = params[1]

    def get_tensor_map(self):
        """Return the stress function consumed by the FEM solver.

        E and k are read from instance attributes (set by ``set_params``) so the
        same function object is reused across solves while its parameter
        dependency is captured by the adjoint via ``ad_wrapper``.
        """
        E = self.E_val
        k = self.k_val
        nu, alpha = self.nu, self.alpha
        a = 0.1 * k  # apex regularization, scaled with cohesion
        dim = self.dim

        def safe_divide(x, y):
            return np.where(y == 0., 0., x / y)

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))

            epsilon_crt = 0.5 * (u_grad + u_grad.T)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = lmbda * np.trace(epsilon_inc) * np.eye(dim) \
                + 2. * mu * epsilon_inc + sigma_old

            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(dim)
            J2 = 0.5 * np.sum(s_dev * s_dev)

            sqrt_J2_reg = np.sqrt(J2 + a * a)
            f_yield = sqrt_J2_reg + alpha * I1 - k

            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            n_dev = safe_divide(s_dev, 2. * sqrt_J2_reg)
            delta_lambda = f_yield_plus / (1. + 3. * alpha * alpha)

            sigma = sigma_trial - delta_lambda * (n_dev + alpha * np.eye(dim))

            # Apex return: if trial stress is beyond the cone tip, map to apex
            sigma_apex = (k / (3. * alpha)) * np.eye(dim)
            at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
            sigma = np.where(at_apex, sigma_apex, sigma)
            return sigma

        return stress_return_map

    def compute_avg_stress_from_sol(self, sol):
        """Volume-averaged stress evaluated from the current solution field.

        This is the differentiable observable used by the inversion loss.
        Unlike ``compute_avg_stress`` (which reads the history array
        ``sigmas_old``), this re-evaluates the constitutive law on the solved
        displacement gradients, so it carries a live dependence on the inverted
        parameters (E, k) through the stress closure.

        Under displacement control the displacement field itself is nearly
        independent of the material stiffness, so a loss built on ``u`` alone
        has a degenerate (near-zero) gradient. The *reaction* (work-conjugate
        to the prescribed displacement) is proportional to the integrated
        stress and is the correct, informative signal.
        """
        u_grads = self.fe.sol_to_grad(sol)
        stress_fn = self.get_tensor_map()
        vmap_stress = jax.vmap(jax.vmap(stress_fn))
        # Single-step: sigma_old / epsilon_old are the zero history arrays.
        sigmas = vmap_stress(u_grads, self.sigmas_old, self.epsilons_old)
        weighted = sigmas.reshape(-1, self.dim, self.dim) \
            * self.fe.JxW.reshape(-1)[:, None, None]
        return np.sum(weighted, axis=0) / np.sum(self.fe.JxW)
