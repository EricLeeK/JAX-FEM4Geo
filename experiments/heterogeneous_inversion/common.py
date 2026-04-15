"""
Shared utilities for heterogeneous parameter field inversion (Stage H).

Provides:
- InversionHeterogeneousDP2D: 2D plane-strain DP with per-element E field
- Synthetic observation generation
- Loss functions with regularization
- Mesh/BC helpers for 2D rectangle domain
- Log-parameterization helpers
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import json

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import rectangle_mesh, Mesh
from src.models.drucker_prager_2d import DruckerPragerPlasticity2D
from src.models.mohr_coulomb_2d import MohrCoulombPlasticity2D, _mc_return_map_3d

RESULTS_DIR = os.path.join(project_root, 'results', 'heterogeneous_inversion')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Problem class: per-element E field
# ---------------------------------------------------------------------------

class InversionHeterogeneousDP2D(DruckerPragerPlasticity2D):
    """2D plane-strain DP with per-element E field for heterogeneous inversion.
    
    Overrides set_params to accept E_field of shape (num_cells,) instead of
    scalar parameters. k is fixed at the value set during __init__.
    
    Supports optional traction loading via get_surface_maps (pass
    location_fns and traction_value to __init__).
    """

    def __init__(self, mesh, vec=2, dim=2, ele_type='QUAD4',
                 dirichlet_bc_info=None, location_fns=None,
                 E=70.0e3, nu=0.3, alpha=0.3, k=250.0, a=None,
                 traction_value=None):
        self.traction_value = traction_value
        # Material params must be set before super().__init__
        self.E = E
        self.nu = nu
        self.alpha = alpha
        self.k = k
        self.a = a if a is not None else 0.01 * k
        self._a_ratio = self.a / self.k
        # Dynamically add get_surface_maps only when traction is used
        if traction_value is not None:
            self._setup_traction(traction_value)
        # Skip DruckerPragerPlasticity2D.__init__, call Problem directly
        from jax_fem.problem import Problem
        Problem.__init__(self, mesh, vec=vec, dim=dim, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)

    def _setup_traction(self, traction_value):
        """Dynamically add get_surface_maps for traction loading."""
        traction = traction_value

        def get_surface_maps(self_ignored=None):
            def traction_fn(u, point):
                return np.array([0., traction])
            return [traction_fn]

        self.get_surface_maps = get_surface_maps

    def set_params(self, E_field):
        """Set per-element Young's modulus field.
        
        Parameters
        ----------
        E_field : jax array, shape (num_cells,)
            Young's modulus for each element.
        """
        nc, nq = len(self.fe.cells), self.fe.num_quads
        E_quad = np.repeat(E_field[:, None, None], nq, axis=1)  # (nc, nq, 1)
        self.internal_vars[2] = E_quad


class InversionHeterogeneousMC2D(MohrCoulombPlasticity2D):
    """2D plane-strain MC with per-element c field for heterogeneous inversion.

    Overrides set_params to accept c_field of shape (num_cells,).
    phi is fixed at the value set during __init__.

    Supports optional traction loading via get_surface_maps.
    """

    def __init__(self, mesh, vec=2, dim=2, ele_type='QUAD4',
                 dirichlet_bc_info=None, location_fns=None,
                 E=70.0e3, nu=0.3, c=50.0, phi_deg=30.0, psi_deg=None,
                 transition_angle=25.0, a_apex=None,
                 traction_value=None):
        self.traction_value = traction_value
        if traction_value is not None:
            self._setup_traction(traction_value)
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
        from jax_fem.problem import Problem
        Problem.__init__(self, mesh, vec=vec, dim=dim, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)

    def _setup_traction(self, traction_value):
        traction = traction_value

        def get_surface_maps(self_ignored=None):
            def traction_fn(u, point):
                return np.array([0., traction])
            return [traction_fn]

        self.get_surface_maps = get_surface_maps

    def set_params(self, c_field):
        """Set per-element cohesion field.

        Parameters
        ----------
        c_field : jax array, shape (num_cells,)
            Cohesion for each element.
        """
        nc, nq = len(self.fe.cells), self.fe.num_quads
        c_quad = np.repeat(c_field[:, None, None], nq, axis=1)
        self.internal_vars[2] = c_quad


class InversionJointMC2D(MohrCoulombPlasticity2D):
    """2D MC with per-element c AND phi fields for joint inversion.

    set_params accepts a stacked array [c_field; phi_field] of shape
    (2*num_cells,). phi_field is in radians. psi is computed as
    phi * psi_ratio inside the return map.
    """

    def __init__(self, mesh, vec=2, dim=2, ele_type='QUAD4',
                 dirichlet_bc_info=None, location_fns=None,
                 E=70.0e3, nu=0.3, c=50.0, phi_deg=30.0, psi_deg=None,
                 transition_angle=25.0, a_apex=None,
                 traction_value=None):
        self.traction_value = traction_value
        if traction_value is not None:
            self._setup_traction(traction_value)
        self.E = E
        self.nu = nu
        self.c = c
        self.phi = np.radians(phi_deg)
        self.psi = np.radians(psi_deg) if psi_deg is not None else self.phi
        self.phi_deg = phi_deg
        self.psi_deg = psi_deg if psi_deg is not None else phi_deg
        self.psi_ratio = self.psi / np.maximum(self.phi, 1e-10)
        self.transition_angle = np.radians(transition_angle)
        self.a_apex = a_apex if a_apex is not None else 0.01 * c
        self._a_apex_ratio = self.a_apex / max(c, 1e-10)
        from jax_fem.problem import Problem
        Problem.__init__(self, mesh, vec=vec, dim=dim, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)

    def _setup_traction(self, traction_value):
        traction = traction_value

        def get_surface_maps(self_ignored=None):
            def traction_fn(u, point):
                return np.array([0., traction])
            return [traction_fn]

        self.get_surface_maps = get_surface_maps

    def set_params(self, params):
        """Set per-element c and phi fields.

        Parameters
        ----------
        params : jax array, shape (2*num_cells,)
            First num_cells entries: cohesion c per element.
            Last num_cells entries: friction angle phi (radians) per element.
        """
        nc, nq = len(self.fe.cells), self.fe.num_quads
        c_field = params[:nc]
        phi_field = params[nc:]
        self.internal_vars[2] = np.repeat(c_field[:, None, None], nq, axis=1)
        self.internal_vars[3] = np.repeat(phi_field[:, None, None], nq, axis=1)

    def get_tensor_map(self):
        nu = self.nu
        E_val = self.E
        psi_ratio = self.psi_ratio
        transition_angle = self.transition_angle
        a_apex_ratio = self._a_apex_ratio

        def stress_return_map(u_grad_2d, sigma_old, epsilon_old, c_arr, phi_arr):
            phi = phi_arr[0]
            psi = phi * psi_ratio
            sigma_3d = _mc_return_map_3d(
                u_grad_2d, sigma_old, epsilon_old,
                E_val, nu, c_arr[0], phi, psi,
                transition_angle, a_apex_ratio,
            )
            return sigma_3d[:2, :2]

        return stress_return_map

    def get_maps(self):
        nu = self.nu
        E_val = self.E
        psi_ratio = self.psi_ratio
        transition_angle = self.transition_angle
        a_apex_ratio = self._a_apex_ratio

        def strain_2d_to_3d(u_grad_2d):
            u_grad = np.zeros((3, 3))
            u_grad = u_grad.at[:2, :2].set(u_grad_2d)
            return 0.5 * (u_grad + u_grad.T)

        def stress_return_map_3d(u_grad_2d, sigma_old, epsilon_old, c_arr, phi_arr):
            phi = phi_arr[0]
            psi = phi * psi_ratio
            return _mc_return_map_3d(
                u_grad_2d, sigma_old, epsilon_old,
                E_val, nu, c_arr[0], phi, psi,
                transition_angle, a_apex_ratio,
            )

        return strain_2d_to_3d, stress_return_map_3d


# ---------------------------------------------------------------------------
# Log-parameterization helpers
# ---------------------------------------------------------------------------

def log_to_E(log_E):
    """Convert log-parameterized values to physical E (ensures E > 0)."""
    return np.exp(log_E)


def E_to_log(E):
    """Convert physical E to log-parameterized values."""
    return np.log(E)


# ---------------------------------------------------------------------------
# Mesh & BC helpers
# ---------------------------------------------------------------------------

def create_2d_mesh_and_bc(displacement, Lx=10., Ly=10., Nx=20, Ny=20):
    """Create 2D QUAD4 mesh with compression BCs.
    
    BCs:
    - Bottom (y=0): u_y = 0
    - Top (y=Ly): u_y = displacement
    - Corner (x=0, y=0): u_x = 0  (remove rigid body)
    
    Parameters
    ----------
    displacement : float
        Prescribed y-displacement at top face.
    Lx, Ly : float
        Domain dimensions.
    Nx, Ny : int
        Number of elements along each axis.
    
    Returns
    -------
    mesh : Mesh
    dirichlet_bc_info : list
    """
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')

    def bottom(p):
        return np.isclose(p[1], 0., atol=1e-5)

    def top(p):
        return np.isclose(p[1], Ly, atol=1e-5)

    def corner(p):
        return np.logical_and(np.isclose(p[0], 0., atol=1e-5),
                              np.isclose(p[1], 0., atol=1e-5))

    dirichlet_bc_info = [
        [bottom, top, corner],
        [1, 1, 0],
        [lambda p: 0., lambda p, _d=displacement: _d, lambda p: 0.],
    ]
    return mesh, dirichlet_bc_info


def create_2d_mesh_traction_bc(traction, Lx=10., Ly=10., Nx=20, Ny=20):
    """Create 2D QUAD4 mesh with traction loading on top face.
    
    With traction loading, the displacement field depends on material
    properties (softer regions deform more), enabling E-field inversion.
    
    BCs:
    - Bottom (y=0): u_y = 0
    - Corner (x=0, y=0): u_x = 0  (remove rigid body)
    - Top (y=Ly): applied traction (Neumann)
    
    Parameters
    ----------
    traction : float
        Applied y-traction on top face [MPa] (negative = compression).
    Lx, Ly : float
        Domain dimensions.
    Nx, Ny : int
        Number of elements along each axis.
    
    Returns
    -------
    mesh : Mesh
    dirichlet_bc_info : list
    location_fns : list
        Surface location functions for Neumann BC.
    traction_value : float
        The traction value (stored for reference).
    """
    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')

    def bottom(p):
        return np.isclose(p[1], 0., atol=1e-5)

    def corner(p):
        return np.logical_and(np.isclose(p[0], 0., atol=1e-5),
                              np.isclose(p[1], 0., atol=1e-5))

    dirichlet_bc_info = [
        [bottom, corner],
        [1, 0],
        [lambda p: 0., lambda p: 0.],
    ]

    def top_surface(p):
        return np.isclose(p[1], Ly, atol=1e-5)

    location_fns = [top_surface]

    return mesh, dirichlet_bc_info, location_fns, traction


def solve_mc_incremental(c_field, mesh, bc_info, loc_fns,
                         traction, n_steps,
                         E=70.0e3, nu=0.3, phi_deg=30.0, psi_deg=None,
                         transition_angle=25.0, a_apex=None,
                         solver_options=None):
    """Solve MC traction problem with incremental loading.

    Creates a fresh Problem at each load step (with scaled traction)
    and carries forward stress/strain history. Returns the final
    displacement field.

    Parameters
    ----------
    c_field : jax array, shape (num_cells,)
        Per-element cohesion.
    n_steps : int
        Number of load increments.

    Returns
    -------
    sol : jax array, shape (num_nodes, 2)
        Final displacement field.
    """
    if solver_options is None:
        solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

    nc = len(mesh.cells)
    nq = 4  # QUAD4
    sigma_old = np.zeros((nc, nq, 3, 3))
    eps_old = np.zeros((nc, nq, 3, 3))
    sol = None
    prev_sol_list = None

    for step in range(n_steps):
        frac = (step + 1) / n_steps
        t_step = traction * frac

        prob = InversionHeterogeneousMC2D(
            mesh, vec=2, dim=2, ele_type='QUAD4',
            dirichlet_bc_info=bc_info,
            location_fns=loc_fns,
            E=E, nu=nu, c=50., phi_deg=phi_deg, psi_deg=psi_deg,
            transition_angle=transition_angle, a_apex=a_apex,
            traction_value=t_step,
        )
        prob.set_params(c_field)
        prob.sigmas_old = sigma_old
        prob.epsilons_old = eps_old
        prob.internal_vars[0] = sigma_old
        prob.internal_vars[1] = eps_old

        opts = dict(solver_options)
        if prev_sol_list is not None:
            opts['initial_guess'] = prev_sol_list

        sol = solver(prob, solver_options=opts)[0]
        prev_sol_list = [sol]
        prob.update_stress_strain(sol)
        sigma_old = prob.sigmas_old
        eps_old = prob.epsilons_old

    return sol


def get_cell_centroids(mesh):
    """Compute centroids of all cells.
    
    Parameters
    ----------
    mesh : Mesh
    
    Returns
    -------
    centroids : onp array, shape (num_cells, dim)
    """
    return onp.mean(onp.take(mesh.points, mesh.cells, axis=0), axis=1)


# ---------------------------------------------------------------------------
# Synthetic observation generation
# ---------------------------------------------------------------------------

def generate_synthetic_observation(E_true_field, k_fixed,
                                   traction=-50.0,
                                   Lx=10., Ly=10., Nx=20, Ny=20,
                                   obs_node_indices=None, noise_level=0.0,
                                   solver_options=None,
                                   nu=0.3, alpha=0.3):
    """Generate synthetic displacement observation from a known E field.
    
    Uses traction loading (Neumann BC) so that the displacement field
    depends on material stiffness — softer regions deform more.
    
    Parameters
    ----------
    E_true_field : array, shape (num_cells,)
        True per-element Young's modulus field.
    k_fixed : float
        Fixed cohesion parameter.
    traction : float
        Applied y-traction on top face [MPa] (negative = compression).
    obs_node_indices : array or None
        Node indices for observation. None = all nodes (full-field).
    noise_level : float
        Gaussian noise std as fraction of max displacement magnitude.
    solver_options : dict or None
    
    Returns
    -------
    dict with keys:
        'u_obs': observed displacement, shape (n_obs, 2) or (n_nodes, 2)
        'obs_indices': node indices used
        'u_full': full displacement field (for plotting)
        'mesh': the mesh object
        'dirichlet_bc_info': BC info
        'location_fns': surface location functions
    """
    if solver_options is None:
        solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

    mesh, bc_info, loc_fns, trac = create_2d_mesh_traction_bc(
        traction, Lx, Ly, Nx, Ny)

    truth_problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        location_fns=loc_fns,
        E=70000., nu=nu, alpha=alpha, k=k_fixed,
        traction_value=traction,
    )
    truth_problem.set_params(np.array(E_true_field))
    sol_list = solver(truth_problem, solver_options=solver_options)
    u_full = sol_list[0]  # (num_nodes, 2)

    if obs_node_indices is None:
        obs_indices = onp.arange(u_full.shape[0])
    else:
        obs_indices = onp.array(obs_node_indices)

    u_obs = u_full[obs_indices]

    if noise_level > 0.0:
        max_disp = float(np.max(np.abs(u_full)))
        noise = noise_level * max_disp * jax.random.normal(
            jax.random.PRNGKey(42), u_obs.shape)
        u_obs = u_obs + noise

    return {
        'u_obs': u_obs,
        'obs_indices': obs_indices,
        'u_full': u_full,
        'mesh': mesh,
        'dirichlet_bc_info': bc_info,
        'location_fns': loc_fns,
    }


def generate_synthetic_observation_mc(c_true_field, phi_deg=30.0,
                                       traction=-50.0,
                                       Lx=10., Ly=10., Nx=20, Ny=20,
                                       obs_node_indices=None, noise_level=0.0,
                                       solver_options=None,
                                       E=70.0e3, nu=0.3, psi_deg=None):
    """Generate synthetic displacement observation from a known c(x) field (MC).

    Uses traction loading so that the displacement field depends on
    spatial cohesion — weaker regions deform more.

    Parameters
    ----------
    c_true_field : array, shape (num_cells,)
        True per-element cohesion field.
    phi_deg : float
        Fixed friction angle [degrees].
    traction : float
        Applied y-traction on top face [MPa] (negative = compression).
    """
    if solver_options is None:
        solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

    mesh, bc_info, loc_fns, trac = create_2d_mesh_traction_bc(
        traction, Lx, Ly, Nx, Ny)

    truth_problem = InversionHeterogeneousMC2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        location_fns=loc_fns,
        E=E, nu=nu, c=50., phi_deg=phi_deg, psi_deg=psi_deg,
        traction_value=traction,
    )
    truth_problem.set_params(np.array(c_true_field))
    sol_list = solver(truth_problem, solver_options=solver_options)
    u_full = sol_list[0]

    if obs_node_indices is None:
        obs_indices = onp.arange(u_full.shape[0])
    else:
        obs_indices = onp.array(obs_node_indices)

    u_obs = u_full[obs_indices]

    if noise_level > 0.0:
        max_disp = float(np.max(np.abs(u_full)))
        noise = noise_level * max_disp * jax.random.normal(
            jax.random.PRNGKey(42), u_obs.shape)
        u_obs = u_obs + noise

    return {
        'u_obs': u_obs,
        'obs_indices': obs_indices,
        'u_full': u_full,
        'mesh': mesh,
        'dirichlet_bc_info': bc_info,
        'location_fns': loc_fns,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def make_heterogeneous_loss(fwd_pred, u_obs, obs_indices,
                             regularizer=None, reg_weight=0.0):
    """Create a loss function for heterogeneous inversion.
    
    Parameters
    ----------
    fwd_pred : callable
        AD-wrapped forward prediction from ad_wrapper.
    u_obs : array, shape (n_obs, 2)
        Observed displacement.
    obs_indices : array
        Node indices of observations.
    regularizer : callable or None
        R(log_E) -> scalar. Applied to log_E (not physical E).
    reg_weight : float
        Regularization weight lambda.
    
    Returns
    -------
    loss_fn : callable
        loss_fn(log_E) -> scalar loss value.
    """
    n_obs_dofs = u_obs.shape[0] * u_obs.shape[1]

    def loss_fn(log_E):
        E_field = log_to_E(log_E)
        sol = fwd_pred(E_field)[0]
        u_pred = sol[obs_indices]
        data_misfit = np.sum((u_pred - u_obs) ** 2) / n_obs_dofs

        reg_term = 0.0
        if regularizer is not None and reg_weight > 0.0:
            reg_term = reg_weight * regularizer(log_E)

        return data_misfit + reg_term

    return loss_fn


# ---------------------------------------------------------------------------
# Regularizers (simple structured-grid versions)
# ---------------------------------------------------------------------------

def smoothness_regularizer(Nx, Ny):
    """Squared first-difference regularizer on structured grid.
    
    Works on log_E reshaped as (Nx, Ny) grid.
    Note: rectangle_mesh uses indexing='ij', so cells are ordered as
    cell_index = ix * Ny + iy, where ix is x-index and iy is y-index.
    
    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions (number of elements).
    
    Returns
    -------
    regularizer : callable
        regularizer(log_E) -> scalar
    """
    def regularizer(log_E):
        g = log_E.reshape(Nx, Ny)
        dx = g[1:, :] - g[:-1, :]
        dy = g[:, 1:] - g[:, :-1]
        return np.mean(dx ** 2) + np.mean(dy ** 2)

    return regularizer


# ---------------------------------------------------------------------------
# E field generators (for truth / initial guess)
# ---------------------------------------------------------------------------

def two_region_E_field(Nx, Ny, E_left=50000., E_right=90000.):
    """Create a two-region E field: left half = E_left, right half = E_right.
    
    Cell ordering: cell_index = ix * Ny + iy (from rectangle_mesh).
    Left = ix < Nx//2, Right = ix >= Nx//2.
    """
    E = onp.full(Nx * Ny, E_right)
    for ix in range(Nx // 2):
        for iy in range(Ny):
            E[ix * Ny + iy] = E_left
    return E


def uniform_E_field(Nx, Ny, E_val=70000.):
    """Create a uniform E field."""
    return onp.full(Nx * Ny, E_val)


def two_region_c_field(Nx, Ny, c_left=30., c_right=70.):
    """Create a two-region cohesion field: left half = c_left, right half = c_right."""
    c = onp.full(Nx * Ny, c_right)
    for ix in range(Nx // 2):
        for iy in range(Ny):
            c[ix * Ny + iy] = c_left
    return c


def uniform_c_field(Nx, Ny, c_val=50.):
    """Create a uniform cohesion field."""
    return onp.full(Nx * Ny, c_val)


def layered_E_field(Nx, Ny, E_values, layer_boundaries):
    """Create a horizontally layered E field.
    
    Parameters
    ----------
    E_values : list of float
        E value for each layer (from bottom to top).
    layer_boundaries : list of float
        Normalized y-boundaries between layers (e.g., [0.3, 0.7] for 3 layers).
    """
    E = onp.zeros(Nx * Ny)
    for ix in range(Nx):
        for iy in range(Ny):
            y_frac = (iy + 0.5) / Ny
            layer_idx = 0
            for b in layer_boundaries:
                if y_frac > b:
                    layer_idx += 1
            E[ix * Ny + iy] = E_values[min(layer_idx, len(E_values) - 1)]
    return E


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_E_field(E_field, Nx, Ny, ax=None, title='E field', vmin=None, vmax=None):
    """Plot an E field on a structured grid.
    
    Cell ordering: cell_index = ix * Ny + iy.
    Reshape to (Nx, Ny) and transpose for display (x=horizontal, y=vertical).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    
    grid = onp.array(E_field).reshape(Nx, Ny).T  # Transpose: x=col, y=row
    im = ax.imshow(grid, origin='lower', aspect='equal',
                   vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_xlabel('x element index')
    ax.set_ylabel('y element index')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='E [MPa]')
    return im


def plot_c_field(c_field, Nx, Ny, ax=None, title='c field', vmin=None, vmax=None):
    """Plot a cohesion field on a structured grid."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()

    grid = onp.array(c_field).reshape(Nx, Ny).T
    im = ax.imshow(grid, origin='lower', aspect='equal',
                   vmin=vmin, vmax=vmax, cmap='RdYlBu_r')
    ax.set_xlabel('x element index')
    ax.set_ylabel('y element index')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='c [MPa]')
    return im


def save_results(out_dir, results_dict):
    """Save results as JSON (handles numpy types)."""
    os.makedirs(out_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (onp.integer, onp.floating)):
            return float(obj)
        if isinstance(obj, onp.ndarray):
            return obj.tolist()
        if hasattr(obj, '__jax_array__') or str(type(obj)).startswith("<class 'jax"):
            return onp.array(obj).tolist()
        return obj
    
    path = os.path.join(out_dir, 'results.json')
    with open(path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=convert)
    return path
