"""
2D Plane Strain Mohr-Coulomb Plasticity Model

Implements the Mohr-Coulomb yield criterion for 2D plane strain with:
  - Abbo & Sloan (1995) hyperbolic smoothing at the apex and π-plane corners
  - Non-associative flow rule (dilation angle ψ ≠ friction angle φ)
  - Full 3D return mapping (σ_33 ≠ 0 in plane strain)

The MC yield function in invariant form:
  f = I₁·sinφ/3 + √(J₂·K²(θ) + a²·sin²φ) - c·cosφ

where K(θ) is the MC shape function on the π-plane with smooth
corner rounding near θ = ±π/6 (Abbo & Sloan 1995).

Parameters passed through internal_vars (indices 2, 3):
  - internal_vars[2]: c (cohesion) field, shape (nc, nq, 1)
  - internal_vars[3]: phi (friction angle in radians) field, shape (nc, nq, 1)
  E and nu are fixed material constants set at __init__.
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


class MohrCoulombPlasticity2D(Problem):
    """
    2D plane strain Mohr-Coulomb plasticity with Abbo & Sloan smoothing.

    Parameters
    ----------
    E : float
        Young's modulus [MPa].
    nu : float
        Poisson's ratio.
    c : float
        Cohesion [MPa].
    phi_deg : float
        Friction angle [degrees].
    psi_deg : float
        Dilation angle [degrees]. Default = phi_deg (associative).
    transition_angle : float
        Smoothing transition angle [degrees] for π-plane corners.
        Default = 25° (Abbo & Sloan recommend 25-29°).
    a_apex : float or None
        Hyperbolic smoothing parameter for the apex [MPa].
        Default = 0.01 * c.
    """

    def __init__(self, mesh, vec=2, dim=2, ele_type='QUAD4',
                 dirichlet_bc_info=None,
                 E=70.0e3, nu=0.3, c=250.0, phi_deg=30.0, psi_deg=None,
                 transition_angle=25.0, a_apex=None):
        self.E = E
        self.nu = nu
        self.c = c
        self.phi = np.radians(phi_deg)
        self.psi = np.radians(psi_deg) if psi_deg is not None else self.phi
        self.phi_deg = phi_deg
        self.psi_deg = psi_deg if psi_deg is not None else phi_deg
        self.transition_angle = np.radians(transition_angle)
        self.a_apex = a_apex if a_apex is not None else 0.01 * c
        self._a_apex_ratio = self.a_apex / max(c, 1e-10)
        super().__init__(mesh, vec=vec, dim=dim, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info)

    def custom_init(self):
        self.fe = self.fes[0]
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.sigmas_old = np.zeros((nc, nq, 3, 3))
        self.epsilons_old = np.zeros((nc, nq, 3, 3))
        c_field = np.full((nc, nq, 1), self.c)
        phi_field = np.full((nc, nq, 1), self.phi)
        self.internal_vars = [self.sigmas_old, self.epsilons_old, c_field, phi_field]

    def set_params(self, params):
        """Set scalar (c, phi) parameters. phi in radians."""
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.internal_vars[2] = np.full((nc, nq, 1), params[0])
        self.internal_vars[3] = np.full((nc, nq, 1), params[1])

    def get_tensor_map(self):
        nu = self.nu
        E_val = self.E
        psi = self.psi
        transition_angle = self.transition_angle
        a_apex_ratio = self._a_apex_ratio

        def stress_return_map(u_grad_2d, sigma_old, epsilon_old, c_arr, phi_arr):
            sigma_3d = _mc_return_map_3d(
                u_grad_2d, sigma_old, epsilon_old,
                E_val, nu, c_arr[0], phi_arr[0], psi,
                transition_angle, a_apex_ratio,
            )
            return sigma_3d[:2, :2]

        return stress_return_map

    def get_maps(self):
        nu = self.nu
        E_val = self.E
        psi = self.psi
        transition_angle = self.transition_angle
        a_apex_ratio = self._a_apex_ratio

        def strain_2d_to_3d(u_grad_2d):
            u_grad = np.zeros((3, 3))
            u_grad = u_grad.at[:2, :2].set(u_grad_2d)
            return 0.5 * (u_grad + u_grad.T)

        def stress_return_map_3d(u_grad_2d, sigma_old, epsilon_old, c_arr, phi_arr):
            return _mc_return_map_3d(
                u_grad_2d, sigma_old, epsilon_old,
                E_val, nu, c_arr[0], phi_arr[0], psi,
                transition_angle, a_apex_ratio,
            )

        return strain_2d_to_3d, stress_return_map_3d

    def stress_strain_fns(self):
        strain_fn, stress_fn = self.get_maps()
        return jax.vmap(jax.vmap(strain_fn)), jax.vmap(jax.vmap(stress_fn))

    def update_stress_strain(self, sol):
        u_grads = self.fe.sol_to_grad(sol)
        vmap_strain, vmap_stress = self.stress_strain_fns()
        self.sigmas_old = vmap_stress(
            u_grads, self.sigmas_old, self.epsilons_old,
            self.internal_vars[2], self.internal_vars[3],
        )
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = [self.sigmas_old, self.epsilons_old,
                              self.internal_vars[2], self.internal_vars[3]]

    def compute_avg_stress(self):
        sigma = np.sum(
            self.sigmas_old.reshape(-1, 3, 3) *
            self.fe.JxW.reshape(-1)[:, None, None],
            axis=0,
        )
        return sigma / np.sum(self.fe.JxW)


# ═══════════════════════════════════════════════════════════════════════════
# Core return mapping (module-level for reuse in tests)
# ═══════════════════════════════════════════════════════════════════════════

def _safe_divide(x, y):
    tiny = 1e-30
    y_safe = np.where(np.abs(y) < tiny, 1., y)
    return np.where(np.abs(y) < tiny, 0., x / y_safe)


def _softplus(x, sharpness=100.):
    """Smooth approximation of max(x, 0).

    Uses log(1 + exp(sharpness * x)) / sharpness, which is C∞ and produces
    a smooth tangent stiffness through AD.  At sharpness=100 the error at
    x=0 is ln(2)/100 ≈ 0.007, negligible vs typical yield stresses O(10²).
    """
    sx = sharpness * x
    # Numerically stable: for large sx use x directly, for small sx use log1p
    return np.where(sx > 20., x, np.log1p(np.exp(np.clip(sx, -30., 20.))) / sharpness)


def _lode_angle(J2, J3):
    """Compute Lode angle θ ∈ [-π/6, π/6] from J2, J3.

    Uses the regularized formula:
        sin(3θ) = -(3√3/2) · J3 / J2^(3/2)
    Returns 0 when J2 ≈ 0 (hydrostatic / zero stress).
    """
    J2_safe = np.maximum(J2, 1e-10)
    arg = -1.5 * np.sqrt(3.) * J3 / (J2_safe ** 1.5)
    arg = np.clip(arg, -1.0 + 1e-12, 1.0 - 1e-12)
    theta = np.arcsin(arg) / 3.
    return np.where(J2 < 1e-10, 0., theta)


def _mc_K_and_dK(theta, ang, transition_angle):
    """Smoothed MC shape function K(θ, η) and dK/dθ on π-plane.

    Uses the Abbo & Sloan (1995) smooth corner approximation:
    - For |θ| ≤ θ_T: K = cos(θ) - sin(θ)·sin(η)/√3
    - For |θ| > θ_T: K = A - B·sin(3θ) with matched value and slope at θ_T

    Parameters: ang is φ for yield surface, ψ for plastic potential.

    Returns (K, dK/dθ).
    """
    s = np.where(theta >= 0., 1., -1.)
    sang = np.sin(ang) / np.sqrt(3.)

    th_T = transition_angle
    cth = np.cos(th_T)
    sth = np.sin(th_T)
    tth = np.tan(th_T)
    t3th = np.tan(3. * th_T)
    c3th = np.cos(3. * th_T)

    K_mc = np.cos(theta) - np.sin(theta) * sang
    dK_mc = -np.sin(theta) - np.cos(theta) * sang

    A = (cth / 3.) * (3. + tth * t3th + s * (t3th - 3. * tth) * sang)
    B = (s * sth + cth * sang) / (3. * c3th)

    K_sm = A - B * np.sin(3. * theta)
    dK_sm = -3. * B * np.cos(3. * theta)

    use_smooth = np.abs(theta) > th_T
    K = np.where(use_smooth, K_sm, K_mc)
    dK = np.where(use_smooth, dK_sm, dK_mc)
    return K, dK


def _mc_gradient(sin_ang, K, dK, sqrt_term, q, J2, tan3t, cos3t_safe, s_dev, s2):
    """Compute ∂f/∂σ (or ∂g/∂σ) for smoothed MC, regularized for J2→0.

    Uses combined forms to avoid 0/0:
      α/(2q) = K/(2·sqrt_term)  (finite as q→0)
    For the C3·(s²-2J₂/3·I) term: since (s²-2J₂/3·I) ~ O(J₂) and the
    coefficient has 1/(q·sqrt_term) ~ 1/(√J₂·...), the product is O(√J₂)→0.
    We combine C3·(s²-2J₂/3·I) = -√3·K·dK·(s²-2J₂/3·I)/(2·q·sqrt_term·cos3θ)
    and rewrite as a safe ratio using q in the denominator only once.
    """
    K_combo = K - tan3t * dK
    C2_over_2q = _safe_divide(K * K_combo, 2. * sqrt_term)

    # C3·(s²-2J₂/3·I): the tensor (s²-2J₂/3·I) has norm ~ O(J₂),
    # and C3 ~ 1/(q·sqrt_term), so the product ~ O(q). We compute
    # the combined term = -√3·K·dK/(2·sqrt_term·cos3θ) · (s²-2J₂/3·I)/q
    # where (s²-2J₂/3·I)/q is finite (→0 as q→0).
    t3_tensor = s2 - (2. * J2 / 3.) * np.eye(3)
    c3_coeff = -np.sqrt(3.) * _safe_divide(K * dK, 2. * sqrt_term * cos3t_safe)
    c3_term = c3_coeff * _safe_divide(t3_tensor, q)

    return (sin_ang / 3.) * np.eye(3) + C2_over_2q * s_dev + c3_term


def _mc_return_map_3d(u_grad_2d, sigma_old, epsilon_old,
                       E, nu, c, phi, psi,
                       transition_angle, a_apex_ratio):
    """Mohr-Coulomb return mapping in full 3D for plane strain.

    Uses smoothed yield surface (Abbo & Sloan 1995):
        f = I₁·sinφ/3 + √(J₂·K²(θ,φ) + a²·sin²φ) - c·cosφ

    Parameters
    ----------
    u_grad_2d : (2, 2) current displacement gradient
    sigma_old : (3, 3) previous stress
    epsilon_old : (3, 3) previous strain
    E, nu : elastic constants
    c : cohesion
    phi : friction angle (radians)
    psi : dilation angle (radians)
    transition_angle : smoothing angle (radians)
    a_apex_ratio : apex smoothing parameter ratio (a/c)
    """
    a_apex = a_apex_ratio * c
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_psi = np.sin(psi)

    mu = E / (2. * (1. + nu))
    lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
    bulk_k = lmbda + 2. * mu / 3.

    # Expand to 3D
    u_grad = np.zeros((3, 3))
    u_grad = u_grad.at[:2, :2].set(u_grad_2d)

    epsilon_crt = 0.5 * (u_grad + u_grad.T)
    epsilon_inc = epsilon_crt - epsilon_old
    sigma_trial = (lmbda * np.trace(epsilon_inc) * np.eye(3)
                   + 2. * mu * epsilon_inc + sigma_old)

    # Invariants
    I1 = np.trace(sigma_trial)
    s_dev = sigma_trial - (I1 / 3.) * np.eye(3)
    J2 = 0.5 * np.sum(s_dev * s_dev)
    J3 = np.linalg.det(s_dev)

    # Lode angle
    theta = _lode_angle(J2, J3)
    q = np.sqrt(np.maximum(J2, 1e-10))

    # Smoothed K and dK/dθ for yield surface (φ) and plastic potential (ψ)
    K_phi, dK_phi = _mc_K_and_dK(theta, phi, transition_angle)
    K_psi, dK_psi = _mc_K_and_dK(theta, psi, transition_angle)

    # Smoothed yield function with apex regularization
    sqrt_term = np.sqrt(J2 * K_phi * K_phi + a_apex * a_apex * sin_phi * sin_phi)
    f_yield = I1 * sin_phi / 3. + sqrt_term - c * cos_phi
    f_yield_plus = _softplus(f_yield)

    # MC gradient: ∂f/∂σ = (C1/3)·I + (C2/(2q))·s + C3·(s²- (2J2/3)·I)
    # where α(η) = q·K / sqrt(q²K² + a²sin²η)
    # To avoid 0/0 when J2→0 (where α→0, q→0), combine:
    #   α/(2q) = K/(2·sqrt_term)  and  α/(2·J2) = K/(2·q·sqrt_term)
    cos3t = np.cos(3. * theta)
    cos3t_safe = np.where(np.abs(cos3t) < 1e-12, 1e-12 * np.sign(cos3t + 1e-30), cos3t)
    tan3t = np.sin(3. * theta) / cos3t_safe

    s2 = s_dev @ s_dev

    n_grad = _mc_gradient(sin_phi, K_phi, dK_phi, sqrt_term,
                          q, J2, tan3t, cos3t_safe, s_dev, s2)

    # Plastic potential gradient (use ψ)
    sqrt_term_psi = np.sqrt(J2 * K_psi * K_psi + a_apex * a_apex * sin_psi * sin_psi)

    m_grad = _mc_gradient(sin_psi, K_psi, dK_psi, sqrt_term_psi,
                          q, J2, tan3t, cos3t_safe, s_dev, s2)

    # C:m  (elastic stiffness applied to flow direction)
    Ce_m = lmbda * np.trace(m_grad) * np.eye(3) + 2. * mu * m_grad

    # Δλ = f / (n : C : m)
    denom = np.sum(n_grad * Ce_m)
    denom = np.maximum(denom, 1e-20)
    delta_lambda = _safe_divide(f_yield_plus, denom)

    sigma_3d = sigma_trial - delta_lambda * Ce_m

    # NOTE: No hard apex return. The Abbo & Sloan hyperbolic smoothing
    # (sqrt(J2·K² + a²·sin²φ)) already rounds the yield surface apex,
    # so the standard return mapping handles the apex region smoothly.
    # A hard np.where to a constant sigma_apex would zero the Jacobian
    # (∂σ/∂u_grad = 0), causing Newton divergence under traction loading.

    return sigma_3d
