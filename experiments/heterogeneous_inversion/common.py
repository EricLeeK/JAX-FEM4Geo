"""
Shared utilities for heterogeneous parameter field inversion (Stage H).

Provides:
- InversionHeterogeneousDP2D: 2D plane-strain DP problem with per-element E field
- Synthetic observation generator
- High-dimensional loss function
- Optimizer wrappers (L-BFGS-B, Adam)
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import rectangle_mesh, Mesh

RESULTS_DIR = os.path.join(project_root, 'results', 'heterogeneous_inversion')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Standalone 2D plane-strain DP return map (for loss post-processing)
# ---------------------------------------------------------------------------

def _safe_divide(x, y):
    tiny = 1e-30
    y_safe = np.where(np.abs(y) < tiny, 1., y)
    return np.where(np.abs(y) < tiny, 0., x / y_safe)


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

    f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
    n_dev = _safe_divide(s_dev, sqrt_J2_reg)
    denom = mu + 9. * bulk_k * alpha * alpha
    delta_lambda = _safe_divide(f_yield_plus, denom)
    sigma_3d = sigma_trial - delta_lambda * (
        mu * n_dev + 3. * bulk_k * alpha * np.eye(3)
    )

    sigma_apex = (k / (3. * alpha)) * np.eye(3)
    at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
    sigma_3d = np.where(at_apex, sigma_apex, sigma_3d)
    return sigma_3d


# ---------------------------------------------------------------------------
# H1: FEM Problem class with per-element E field
# ---------------------------------------------------------------------------

class InversionHeterogeneousDP2D(Problem):
    """2D plane-strain DP problem with per-element E field for high-dim inversion.

    params: shape (num_cells,) — each element has an independent E value.
    k is fixed at construction time.
    """

    def __init__(self, mesh, vec=2, dim=2, ele_type='QUAD4',
                 dirichlet_bc_info=None, E_init=70000.0, k=50.0):
        self.E_init = E_init
        self.k_fixed = k
        super().__init__(mesh, vec=vec, dim=dim, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info)

    def custom_init(self):
        self.fe = self.fes[0]
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.sigmas_old = np.zeros((nc, nq, 3, 3))
        self.epsilons_old = np.zeros((nc, nq, 3, 3))
        E_field = np.full((nc, nq, 1), self.E_init)
        k_field = np.full((nc, nq, 1), self.k_fixed)
        self.internal_vars = [self.sigmas_old, self.epsilons_old, E_field, k_field]

    def set_params(self, params):
        """params: (num_cells,) per-element E values."""
        nc, nq = len(self.fe.cells), self.fe.num_quads
        # Expand per-cell E to per-quadrature-point: (nc,) -> (nc, nq, 1)
        E_field = np.repeat(params[:, None, None], nq, axis=1)
        self.internal_vars[2] = E_field
        # k stays fixed
        self.internal_vars[3] = np.full((nc, nq, 1), self.k_fixed)

    def get_tensor_map(self):
        nu = 0.3
        alpha = 0.3
        a_ratio = 0.1

        def safe_divide(x, y):
            tiny = 1e-30
            y_safe = np.where(np.abs(y) < tiny, 1., y)
            return np.where(np.abs(y) < tiny, 0., x / y_safe)

        def stress_return_map(u_grad_2d, sigma_old_3x3, epsilon_old_3x3,
                              E_arr, k_arr):
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

            return sigma_3d[:2, :2]

        return stress_return_map

    def stress_strain_fns(self):
        nu = 0.3
        alpha = 0.3
        a_ratio = 0.1

        def safe_divide(x, y):
            tiny = 1e-30
            y_safe = np.where(np.abs(y) < tiny, 1., y)
            return np.where(np.abs(y) < tiny, 0., x / y_safe)

        def strain_2d_to_3d(u_grad_2d):
            u_grad = np.zeros((3, 3))
            u_grad = u_grad.at[:2, :2].set(u_grad_2d)
            return 0.5 * (u_grad + u_grad.T)

        def stress_return_map_3d(u_grad_2d, sigma_old_3x3, epsilon_old_3x3,
                                 E_arr, k_arr):
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

        return jax.vmap(jax.vmap(strain_2d_to_3d)), \
               jax.vmap(jax.vmap(stress_return_map_3d))

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

    def reset_internal_vars(self):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.sigmas_old = np.zeros((nc, nq, 3, 3))
        self.epsilons_old = np.zeros((nc, nq, 3, 3))
        self.internal_vars[0] = self.sigmas_old
        self.internal_vars[1] = self.epsilons_old


# ---------------------------------------------------------------------------
# Mesh & BC helpers
# ---------------------------------------------------------------------------

def create_2d_mesh_and_bc(displacement, Lx=10., Ly=10., Nx=20, Ny=20):
    """Create 2D QUAD4 mesh + compression BCs.

    Returns (mesh, dirichlet_bc_info).
    """
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'],
                ele_type='QUAD4')

    def bottom(p):
        return np.isclose(p[1], 0.)

    def top(p):
        return np.isclose(p[1], Ly)

    def corner(p):
        return np.logical_and(np.isclose(p[0], 0.), np.isclose(p[1], 0.))

    dirichlet_bc_info = [
        [bottom, top, corner],
        [1, 1, 0],
        [lambda p: 0., lambda p, _d=displacement: _d, lambda p: 0.],
    ]
    return mesh, dirichlet_bc_info


def update_bc_2d(problem, dirichlet_bc_info, disp):
    """Update top-face displacement BC."""
    dirichlet_bc_info[-1][1] = lambda p, _d=disp: _d
    problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)


# ---------------------------------------------------------------------------
# H2: Synthetic observation data generator
# ---------------------------------------------------------------------------

def generate_synthetic_observation(problem, fwd_pred, E_true_field,
                                   obs_node_indices=None, noise_level=0.0,
                                   key=None):
    """Generate synthetic displacement observation from a known E field.

    Parameters
    ----------
    problem : InversionHeterogeneousDP2D
    fwd_pred : callable from ad_wrapper
    E_true_field : (num_cells,) true E field
    obs_node_indices : (num_obs,) node indices for sparse observation.
        If None, use all nodes (full-field).
    noise_level : float, relative Gaussian noise std
    key : jax.random.PRNGKey (required if noise_level > 0)

    Returns
    -------
    obs_data : (num_obs, vec) observed displacements
    obs_indices : (num_obs,) node indices used
    sol_true : full displacement solution
    """
    sol_list = fwd_pred(E_true_field)
    sol_true = sol_list[0]  # (num_nodes, vec)

    if obs_node_indices is None:
        obs_indices = np.arange(sol_true.shape[0])
    else:
        obs_indices = obs_node_indices

    obs_data = sol_true[obs_indices]

    if noise_level > 0.0 and key is not None:
        noise_scale = noise_level * np.max(np.abs(obs_data))
        noise = noise_scale * jax.random.normal(key, obs_data.shape)
        obs_data = obs_data + noise

    return obs_data, obs_indices, sol_true


# ---------------------------------------------------------------------------
# H3: High-dimensional loss function
# ---------------------------------------------------------------------------

def displacement_loss(E_field, problem, fwd_pred, obs_data, obs_indices,
                      regularizer=None, reg_weight=0.0):
    """Loss = ||u_FEM(E) - u_obs||^2 + lambda * R(E).

    Parameters
    ----------
    E_field : (num_cells,) per-element E values
    obs_data : (num_obs, vec) observed displacements
    obs_indices : (num_obs,) observation node indices
    regularizer : callable(E_field) -> scalar, or None
    reg_weight : float, regularization weight lambda

    Returns
    -------
    loss : scalar
    """
    sol_list = fwd_pred(E_field)
    u_pred = sol_list[0][obs_indices]
    data_misfit = np.sum((u_pred - obs_data) ** 2)

    reg_term = 0.0
    if regularizer is not None:
        reg_term = reg_weight * regularizer(E_field)

    return data_misfit + reg_term


# ---------------------------------------------------------------------------
# H4: Optimizer wrappers
# ---------------------------------------------------------------------------

def softplus_parameterization(theta, E_min=1000.0):
    """Map unconstrained theta -> E > E_min via softplus."""
    return E_min + jax.nn.softplus(theta)


def inv_softplus(E, E_min=1000.0):
    """Inverse of softplus parameterization: E -> theta."""
    x = E - E_min
    x = np.maximum(x, 1e-6)
    return np.log(np.exp(x) - 1.)


def optimize_lbfgsb(loss_fn, E_init, E_min=1000.0, E_max=500000.0,
                     maxiter=100, verbose=True):
    """L-BFGS-B optimizer with parameter bounds.

    Parameters
    ----------
    loss_fn : callable(E_field) -> scalar (must be JAX-differentiable)
    E_init : (num_cells,) initial E field
    E_min, E_max : float, bounds
    maxiter : int

    Returns
    -------
    result : dict with keys 'E_final', 'loss_history', 'time_s', 'nit'
    """
    import scipy.optimize

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    # Warm up JIT
    _ = value_and_grad_fn(E_init)

    history = {'loss': [], 'grad_norm': []}

    big_loss = 1e10

    def objective(E_flat):
        E_jax = np.array(E_flat)
        try:
            loss_val, grad = value_and_grad_fn(E_jax)
            loss_val = float(loss_val)
            grad_np = onp.array(grad, dtype=onp.float64)
            if onp.isnan(loss_val) or onp.any(onp.isnan(grad_np)):
                raise ValueError("NaN in loss or gradient")
        except Exception:
            loss_val = big_loss
            grad_np = onp.zeros_like(E_flat)
        history['loss'].append(loss_val)
        history['grad_norm'].append(float(onp.linalg.norm(grad_np)))
        if verbose and len(history['loss']) % 10 == 1:
            print(f"  iter {len(history['loss']):4d}: "
                  f"loss = {loss_val:.6e}, |grad| = {history['grad_norm'][-1]:.4e}")
        return loss_val, grad_np

    bounds = [(E_min, E_max)] * len(E_init)

    t0 = time.time()
    res = scipy.optimize.minimize(
        objective, onp.array(E_init, dtype=onp.float64),
        method='L-BFGS-B', jac=True, bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12},
    )
    elapsed = time.time() - t0

    if verbose:
        print(f"  L-BFGS-B finished: {res.nit} iters, {elapsed:.1f}s, "
              f"success={res.success}")

    return {
        'E_final': np.array(res.x),
        'loss_history': history['loss'],
        'grad_norm_history': history['grad_norm'],
        'time_s': elapsed,
        'nit': res.nit,
        'success': res.success,
    }


def optimize_adam(loss_fn, E_init, num_iters=200, lr=500.0,
                  E_min=1000.0, E_max=500000.0, verbose=True):
    """Adam optimizer with clamping.

    Parameters
    ----------
    loss_fn : callable(E_field) -> scalar
    E_init : (num_cells,) initial E field
    num_iters : int
    lr : float
    E_min, E_max : clamp bounds

    Returns
    -------
    result : dict
    """
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    # Warm up JIT
    _ = value_and_grad_fn(E_init)

    m = onp.zeros_like(E_init)
    v = onp.zeros_like(E_init)
    params = onp.array(E_init, dtype=onp.float64)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    history = {'loss': [], 'grad_norm': [], 'params_snapshots': []}

    t0 = time.time()
    for i in range(num_iters):
        loss_val, grad = value_and_grad_fn(np.array(params))
        grad = onp.array(grad)
        loss_val = float(loss_val)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))

        params = params - lr * m_hat / (onp.sqrt(v_hat) + eps)
        params = onp.clip(params, E_min, E_max)

        history['loss'].append(loss_val)
        history['grad_norm'].append(float(onp.linalg.norm(grad)))

        if verbose and (i % 20 == 0 or i == num_iters - 1):
            print(f"  Adam iter {i:4d}: loss = {loss_val:.6e}, "
                  f"|grad| = {history['grad_norm'][-1]:.4e}")

    elapsed = time.time() - t0

    return {
        'E_final': np.array(params),
        'loss_history': history['loss'],
        'grad_norm_history': history['grad_norm'],
        'time_s': elapsed,
        'nit': num_iters,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_E_field_comparison(E_true, E_inverted, Nx, Ny, Lx, Ly,
                            title='', save_path=None):
    """Plot true vs inverted E field side by side."""
    import matplotlib.pyplot as plt

    E_true_2d = onp.array(E_true).reshape(Nx, Ny)
    E_inv_2d = onp.array(E_inverted).reshape(Nx, Ny)

    vmin = min(E_true_2d.min(), E_inv_2d.min())
    vmax = max(E_true_2d.max(), E_inv_2d.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(E_true_2d.T, origin='lower', vmin=vmin, vmax=vmax,
                          extent=[0, Lx, 0, Ly], cmap='viridis')
    axes[0].set_title('True E field')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(E_inv_2d.T, origin='lower', vmin=vmin, vmax=vmax,
                          extent=[0, Lx, 0, Ly], cmap='viridis')
    axes[1].set_title('Inverted E field')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    err_2d = onp.abs(E_inv_2d - E_true_2d)
    im2 = axes[2].imshow(err_2d.T, origin='lower',
                          extent=[0, Lx, 0, Ly], cmap='hot')
    axes[2].set_title('Absolute error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")
    plt.close()


def compute_inversion_metrics(E_true, E_inverted):
    """Compute error metrics between true and inverted E fields."""
    E_true = onp.array(E_true)
    E_inverted = onp.array(E_inverted)

    abs_err = onp.abs(E_inverted - E_true)
    rel_err = abs_err / onp.maximum(E_true, 1e-12)

    l2_err = onp.sqrt(onp.sum((E_inverted - E_true) ** 2) / onp.sum(E_true ** 2))
    max_rel_err = onp.max(rel_err)
    mean_rel_err = onp.mean(rel_err)

    return {
        'L2_relative_error': float(l2_err),
        'max_relative_error': float(max_rel_err),
        'mean_relative_error': float(mean_rel_err),
        'max_absolute_error': float(onp.max(abs_err)),
    }
