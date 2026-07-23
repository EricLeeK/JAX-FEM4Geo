"""P2-7: Drucker-Prager plasticity with isotropic hardening (differentiable).

Extends the single-step differentiable DP model with an isotropic hardening
law: the cohesion ``k`` grows (hardening) or shrinks (softening) with the
accumulated equivalent plastic strain within a step:

    k_eff = k0 + H * eps_p_eq

where ``H`` is the hardening modulus (H > 0 hardening, H < 0 softening) and
``eps_p_eq`` is the equivalent plastic strain accumulated during the return
mapping of that step. The friction coefficient ``alpha`` is held constant
(isotropic hardening preserves the cone opening angle).

Differentiability & limitation
------------------------------
Hardening is handled *within* the single-step return mapping (eps_p_eq is
computed from delta_lambda of the same step), so it stays compatible with the
direct (fixed-displacement) differentiation strategy used by the inversion —
no cross-step state enters the AD path. The hardening modulus ``H`` is exposed
as an additional invertible parameter via ``set_params``.

IMPORTANT limitation: true isotropic hardening is a **path-dependent,
multi-step** phenomenon — the equivalent plastic strain accumulates across
load increments, and the yield surface evolves with the *history*. The
single-step formulation here only captures the hardening active *within one
return mapping*, where the elastic return-mapping modulus ``H_elastic``
(~bulk/shear stiffness, ~1e4 MPa) dominates a realistic hardening modulus
``H`` (~1e2 MPa). Consequently the single-step response is only weakly
sensitive to ``H``. Capturing full path-dependent hardening requires the
multi-step solver with an implicit adjoint through the load history, which the
P0/P1 work showed is unreliable at the elastic/plastic switch (the adjoint
freezes the active-set partition). That multi-step differentiable hardening is
left as future work. The single-step model here is still a correct, useful
surrogate for testing the inversion machinery with a 3-parameter model.
"""

import jax
import jax.numpy as np

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem


class DifferentiableDPHardening(Problem):
    """Single-step differentiable DP with isotropic hardening.

    Invertible parameters: ``params = [E, k0, H]`` (Young's modulus, initial
    cohesion, hardening modulus). Fixed: ``nu``, ``alpha``, apex reg. ``a``.
    """

    def __init__(self, mesh, vec=3, dim=3, dirichlet_bc_info=None,
                 location_fns=None, nu=0.3, alpha=0.3):
        self.nu = nu
        self.alpha = alpha
        super().__init__(mesh, vec=vec, dim=dim,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)

    def custom_init(self):
        self.fe = self.fes[0]
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads,
                                      self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]
        self.E_val = 70.0e3
        self.k0_val = 250.0
        self.H_val = 0.0     # default: ideal plasticity (H=0)

    def set_params(self, params):
        """params = [E, k0, H]."""
        self.E_val = params[0]
        self.k0_val = params[1]
        self.H_val = params[2]

    def get_tensor_map(self):
        E = self.E_val
        k0 = self.k0_val
        H = self.H_val
        nu, alpha = self.nu, self.alpha
        a = 0.1 * k0
        dim = self.dim

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            bulk = lmbda + 2. / 3. * mu
            eps_crt = 0.5 * (u_grad + u_grad.T)
            eps_inc = eps_crt - epsilon_old
            sigma_trial = lmbda * np.trace(eps_inc) * np.eye(dim) \
                + 2. * mu * eps_inc + sigma_old

            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(dim)
            J2 = 0.5 * np.sum(s_dev * s_dev)
            sqrt_J2_reg = np.sqrt(J2 + a * a)

            # Isotropic linear hardening via the consistency condition.
            # f = sqrt(J2+a^2) + alpha*I1 - (k0 + H*ep) = 0, with the equivalent
            # plastic strain ep ~ delta_lambda. The associated-flow consistency
            # gives delta_lambda = f_trial / (H_elastic + H), where the elastic
            # return-mapping modulus is
            #   H_elastic = mu*(n_dev:n_dev) + 9*alpha^2*bulk
            # (n_dev:n_dev ~ 1 at the deviatoric limit; using the regularized
            #  form below keeps it smooth and bounded). H is the user hardening
            # modulus in stress units (MPa).
            n_dev_sq = J2 / (J2 + a * a)            # bounded in [0,1), regularized
            H_elastic = mu * n_dev_sq + 9. * alpha * alpha * bulk
            denom = H_elastic + H
            f_yield_0 = sqrt_J2_reg + alpha * I1 - k0
            delta_lambda = np.where(f_yield_0 > 0., f_yield_0, 0.) / denom

            k_eff = k0 + H * delta_lambda
            # Stress correction uses the (dimensionless) associated flow
            # direction times delta_lambda, same proven form as the ideal model.
            n_dev = np.where(sqrt_J2_reg == 0., 0., s_dev / (2. * sqrt_J2_reg))
            sigma = sigma_trial - delta_lambda * (mu * n_dev + 3. * bulk * alpha * np.eye(dim))

            # Apex return (at the current k_eff).
            sigma_apex = (k_eff / (3. * alpha)) * np.eye(dim)
            at_apex = np.logical_and(f_yield_0 > 0., I1 > k_eff / alpha)
            sigma = np.where(at_apex, sigma_apex, sigma)
            return sigma

        return stress_return_map

    def compute_avg_stress_from_sol(self, sol):
        """Volume-averaged stress (differentiable observable for inversion)."""
        u_grads = self.fe.sol_to_grad(sol)
        stress_fn = self.get_tensor_map()
        vmap_stress = jax.vmap(jax.vmap(stress_fn))
        sigmas = vmap_stress(u_grads, self.sigmas_old, self.epsilons_old)
        weighted = sigmas.reshape(-1, self.dim, self.dim) \
            * self.fe.JxW.reshape(-1)[:, None, None]
        return np.sum(weighted, axis=0) / np.sum(self.fe.JxW)
