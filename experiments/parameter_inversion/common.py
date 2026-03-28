"""
Shared utilities for parameter inversion experiments (Phase D).

Provides mesh creation, FEM problem class, loss functions, and
gradient wrappers used across D1/D2/D3 experiments.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import box_mesh_gmsh, cylinder_mesh_gmsh, get_meshio_cell_type, Mesh

RESULTS_DIR = os.path.join(project_root, 'results', 'parameter_inversion')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# DP return mapping (standalone, for loss post-processing)
# ---------------------------------------------------------------------------

def _safe_divide(x, y):
    tiny = 1e-30
    y_safe = np.where(np.abs(y) < tiny, 1., y)
    return np.where(np.abs(y) < tiny, 0., x / y_safe)


def stress_return_dp(u_grad, sigma_old, epsilon_old, E, k, dim):
    """Drucker-Prager return map with explicit E, k for AD."""
    nu = 0.3
    alpha = 0.3
    a = 0.1 * k
    mu = E / (2. * (1. + nu))
    lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
    bulk_k = lmbda + 2. * mu / 3.

    epsilon_crt = 0.5 * (u_grad + u_grad.T)
    epsilon_inc = epsilon_crt - epsilon_old
    sigma_trial = (lmbda * np.trace(epsilon_inc) * np.eye(dim)
                   + 2. * mu * epsilon_inc + sigma_old)

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


# ---------------------------------------------------------------------------
# FEM Problem class (internal_vars mode)
# ---------------------------------------------------------------------------

class InversionDruckerPrager(Problem):
    """DP problem for inversion experiments. E, k via internal_vars."""

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

    def stress_strain_fns(self):
        nu = 0.3
        dim = self.dim

        def strain_fn(u_grad):
            return 0.5 * (u_grad + u_grad.T)

        def stress_fn(u_grad, sigma_old, epsilon_old, E_arr, k_arr):
            return stress_return_dp(
                u_grad, sigma_old, epsilon_old, E_arr[0], k_arr[0], dim,
            )

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

    def reset_internal_vars(self):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.epsilons_old = np.zeros((nc, nq, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars[0] = self.sigmas_old
        self.internal_vars[1] = self.epsilons_old


# ---------------------------------------------------------------------------
# Mesh & BC helpers
# ---------------------------------------------------------------------------

def create_mesh_and_bc(displacement, Lx=10., Ly=10., Lz=10., Nx=2, Ny=2, Nz=2):
    """Create mesh, BC info, and return (mesh, dirichlet_bc_info)."""
    data_dir = os.path.join(RESULTS_DIR, '_mesh_cache')
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
        [lambda p: 0., lambda p, _d=displacement: _d, lambda p: 0., lambda p: 0.],
    ]
    return mesh, dirichlet_bc_info


def update_bc(problem, dirichlet_bc_info, disp):
    """Update top-face displacement BC."""
    dirichlet_bc_info[-1][1] = lambda p, _d=disp: _d
    problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def volume_avg_sigma_zz(fe, sol, sigma_old, epsilon_old, E, k):
    """Volume-averaged sigma_zz using explicit E, k for AD."""
    u_grads = fe.sol_to_grad(sol)
    c, q, _, dim = u_grads.shape

    def one_quad(ug, so, eo):
        return stress_return_dp(ug, so, eo, E, k, dim)

    sig = jax.vmap(jax.vmap(one_quad))(u_grads, sigma_old, epsilon_old)
    JxW = fe.JxW
    sig_zz = sig[..., 2, 2]
    return np.sum(sig_zz * JxW) / np.sum(JxW)


def generate_observation(problem, dirichlet_bc_info, displacement, E_true, k_true,
                         solver_options):
    """Generate synthetic observation data at given displacement."""
    params_true = np.array([E_true, k_true])
    update_bc(problem, dirichlet_bc_info, displacement)
    problem.set_params(params_true)
    sol_list = solver(problem, solver_options=solver_options)
    sol = sol_list[0]
    sigma_zz_obs = volume_avg_sigma_zz(
        problem.fe, sol, problem.sigmas_old, problem.epsilons_old, E_true, k_true,
    )
    return float(sigma_zz_obs), sol


# ---------------------------------------------------------------------------
# Simple Adam optimizer (no external deps)
# ---------------------------------------------------------------------------

def adam_optimize(value_and_grad_fn, params_init, num_iters=100, lr=1.0,
                  bounds=None, callback=None):
    """Minimize with Adam. Returns (params_final, history).

    Parameters
    ----------
    bounds : list of (lo, hi) per parameter, or None
    callback : callable(step, params, loss, grad) or None
    """
    m = onp.zeros_like(params_init)
    v = onp.zeros_like(params_init)
    params = onp.array(params_init, dtype=onp.float64)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    history = {'loss': [], 'params': [], 'grad_norm': []}

    for i in range(num_iters):
        loss_val, grad = value_and_grad_fn(np.array(params))
        grad = onp.array(grad)
        loss_val = float(loss_val)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))

        params = params - lr * m_hat / (onp.sqrt(v_hat) + eps)

        # Clip to bounds
        if bounds is not None:
            for j, (lo, hi) in enumerate(bounds):
                if lo is not None:
                    params[j] = max(params[j], lo)
                if hi is not None:
                    params[j] = min(params[j], hi)

        history['loss'].append(loss_val)
        history['params'].append(params.copy())
        history['grad_norm'].append(float(onp.linalg.norm(grad)))

        if callback:
            callback(i, params, loss_val, grad)

    return np.array(params), history


# ---------------------------------------------------------------------------
# Finite-difference gradient
# ---------------------------------------------------------------------------

def fd_gradient(loss_fn, params, eps_list):
    """Central-difference gradient. eps_list[i] is the step for params[i]."""
    params = np.array(params)
    grad = onp.zeros(len(params))
    for i in range(len(params)):
        p_plus = params.at[i].set(params[i] + eps_list[i])
        p_minus = params.at[i].set(params[i] - eps_list[i])
        grad[i] = (float(loss_fn(p_plus)) - float(loss_fn(p_minus))) / (2. * eps_list[i])
    return grad


def fd_value_and_grad(loss_fn, eps_list):
    """Return a function matching jax.value_and_grad signature but using FD."""
    def vg(params):
        loss_val = loss_fn(params)
        grad = fd_gradient(loss_fn, params, eps_list)
        return loss_val, np.array(grad)
    return vg


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------

def two_stage_inversion(sigma_zz_elastic_obs, sigma_zz_plastic_obs,
                        E_init=55000.0, K_init=45.0,
                        E_bounds=(30000, 120000), k_bounds=(35, 80),
                        solver_options=None, disp_elastic=-0.015,
                        disp_plastic=-0.028, verbose=True):
    """Two-stage σ_zz inversion: elastic obs → E, plastic obs → k.

    Returns dict with keys: E_final, k_final, err_E, err_k, n_evals, time_s,
    history_E, history_k.
    """
    from scipy.optimize import minimize_scalar
    if solver_options is None:
        solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

    t0 = time.time()
    n_evals = 0

    # --- Stage 1: E from elastic observation ---
    mesh_e, bc_e = create_mesh_and_bc(disp_elastic)
    prob_e = InversionDruckerPrager(mesh_e, vec=3, dim=3, dirichlet_bc_info=bc_e)
    fwd_e = ad_wrapper(prob_e, solver_options=solver_options,
                       adjoint_solver_options=solver_options)

    def loss_E(params_E):
        E = params_E[0]
        k_dummy = 200.0
        sol = fwd_e(np.array([E, k_dummy]))[0]
        pred = volume_avg_sigma_zz(
            prob_e.fe, sol, prob_e.sigmas_old, prob_e.epsilons_old, E, k_dummy,
        )
        return (pred - sigma_zz_elastic_obs) ** 2

    # Warm up JIT
    _ = jax.value_and_grad(loss_E)(np.array([E_init]))

    history_E = []

    def track_E(E_val):
        nonlocal n_evals
        n_evals += 1
        l = float(loss_E(np.array([E_val])))
        history_E.append({'E': E_val, 'loss': l})
        return l

    res_E = minimize_scalar(track_E, bounds=E_bounds, method='bounded',
                            options={'xatol': 1.0, 'maxiter': 30})
    E_final = res_E.x
    if verbose:
        print(f"  Stage 1: E = {E_final:.2f} ({res_E.nfev} evals)")

    # --- Stage 2: k from plastic observation ---
    mesh_p, bc_p = create_mesh_and_bc(disp_plastic)
    prob_p = InversionDruckerPrager(mesh_p, vec=3, dim=3, dirichlet_bc_info=bc_p)
    fwd_p = ad_wrapper(prob_p, solver_options=solver_options,
                       adjoint_solver_options=solver_options)

    def loss_k(params_k):
        k = params_k[0]
        sol = fwd_p(np.array([E_final, k]))[0]
        pred = volume_avg_sigma_zz(
            prob_p.fe, sol, prob_p.sigmas_old, prob_p.epsilons_old, E_final, k,
        )
        return (pred - sigma_zz_plastic_obs) ** 2

    _ = jax.value_and_grad(loss_k)(np.array([K_init]))

    history_k = []

    def track_k(k_val):
        nonlocal n_evals
        n_evals += 1
        l = float(loss_k(np.array([k_val])))
        history_k.append({'k': k_val, 'loss': l})
        return l

    res_k = minimize_scalar(track_k, bounds=k_bounds, method='bounded',
                            options={'xatol': 0.1, 'maxiter': 30})
    k_final = res_k.x
    total_time = time.time() - t0

    if verbose:
        print(f"  Stage 2: k = {k_final:.4f} ({res_k.nfev} evals)")
        print(f"  Total: {total_time:.2f}s, {n_evals} evals")

    return {
        'E_final': float(E_final),
        'k_final': float(k_final),
        'n_evals': n_evals,
        'time_s': total_time,
        'history_E': history_E,
        'history_k': history_k,
    }


def incremental_solve(problem, dirichlet_bc_info, target_disp, n_steps=10,
                      solver_options=None):
    """Incrementally load to target_disp in n_steps, updating internal vars.

    Returns the solution at the final step.
    """
    if solver_options is None:
        solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    ls_opts = {**solver_options, 'line_search_flag': True, 'line_search_max_iters': 15}

    prev_sol = None
    for i in range(n_steps):
        disp_i = target_disp * (i + 1) / n_steps
        update_bc(problem, dirichlet_bc_info, disp_i)
        step_opts = dict(ls_opts)
        if prev_sol is not None:
            step_opts['initial_guess'] = [prev_sol]
        sol = solver(problem, solver_options=step_opts)[0]
        problem.update_stress_strain(sol)
        prev_sol = sol

    return sol


def benchmark(fn, n_warmup=2, n_repeat=10):
    """Time a function. Returns (result, mean_time, std_time) in seconds."""
    # Warmup
    for _ in range(n_warmup):
        result = fn()
    # Timed runs
    times = []
    for _ in range(n_repeat):
        t0 = time.time()
        result = fn()
        times.append(time.time() - t0)
    times = onp.array(times)
    return result, float(onp.mean(times)), float(onp.std(times))


# ---------------------------------------------------------------------------
# Triaxial inversion infrastructure
# ---------------------------------------------------------------------------

class InversionTriaxialDP(Problem):
    """DP problem for triaxial inversion. E, k via internal_vars.
    Confining pressure via get_surface_maps() (not differentiated).
    """

    def __init__(self, mesh, vec=3, dim=3, ele_type='HEX8',
                 dirichlet_bc_info=None, location_fns=None,
                 confining_pressure=0.0):
        self.confining_pressure = confining_pressure
        super().__init__(mesh, vec=vec, dim=dim, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)

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

    def get_surface_maps(self):
        confining_p = self.confining_pressure

        def lateral_traction(u, point, *args):
            x, y = point[0], point[1]
            r = np.sqrt(x**2 + y**2)
            r_safe = np.where(r > 1e-10, r, 1.0)
            nx = -x / r_safe
            ny = -y / r_safe
            return confining_p * np.array([nx, ny, 0.])

        return [lateral_traction]

    def stress_strain_fns(self):
        dim = self.dim

        def strain_fn(u_grad):
            return 0.5 * (u_grad + u_grad.T)

        def stress_fn(u_grad, sigma_old, epsilon_old, E_arr, k_arr):
            return stress_return_dp(
                u_grad, sigma_old, epsilon_old, E_arr[0], k_arr[0], dim,
            )

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

    def reset_internal_vars(self):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.epsilons_old = np.zeros((nc, nq, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars[0] = self.sigmas_old
        self.internal_vars[1] = self.epsilons_old


def create_triaxial_mesh_and_bc(displacement, R=2.5, H=10.0, circle_mesh=3,
                                 height_mesh=4, confining_pressure=50.0):
    """Create cylindrical mesh + BCs for triaxial test.

    Returns (mesh, dirichlet_bc_info, location_fns, confining_pressure).
    """
    data_dir = os.path.join(RESULTS_DIR, '_triaxial_mesh_cache')
    os.makedirs(data_dir, exist_ok=True)

    meshio_mesh = cylinder_mesh_gmsh(
        data_dir=data_dir, R=R, H=H,
        circle_mesh=circle_mesh, hight_mesh=height_mesh,
        rect_ratio=0.4,
    )
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'],
                ele_type='HEX8')

    tol = 1e-5

    def bottom(p):
        return np.isclose(p[2], 0., atol=tol)

    def top(p):
        return np.isclose(p[2], H, atol=tol)

    def center_bottom(p):
        return np.logical_and(
            np.logical_and(np.abs(p[0]) < tol, np.abs(p[1]) < tol),
            np.isclose(p[2], 0., atol=tol),
        )

    dirichlet_bc_info = [
        [bottom, center_bottom, center_bottom, top],
        [2, 0, 1, 2],
        [lambda p: 0., lambda p: 0., lambda p: 0.,
         lambda p, _d=displacement: _d],
    ]

    def lateral_surface(p):
        r = np.sqrt(p[0]**2 + p[1]**2)
        z = p[2]
        on_surface = np.isclose(r, R, atol=tol * 10)
        not_top_bottom = np.logical_and(z > tol, z < H - tol)
        return np.logical_and(on_surface, not_top_bottom)

    location_fns = [lateral_surface]

    return mesh, dirichlet_bc_info, location_fns
