"""Forward model for Drucker-Prager parameter inversion.

Defines the triaxial boundary-value problem and the differentiable reaction
observable used as the inversion target.

Strategy — *direct* (fixed-displacement) differentiation
-------------------------------------------------------
We solve the BVP once for the displacement field ``u``, then evaluate the
reaction (volume-averaged axial stress) by re-applying the constitutive law to
the solved strain field. The loss is differentiated w.r.t. ``(E, k)`` keeping
``u`` fixed (``u`` is treated as a constant JAX array, not differentiated).

Why not differentiate through the implicit FEM solve (the ``ad_wrapper``
adjoint)? Under displacement control the prescribed dofs are fixed and the free
dofs barely move with the material stiffness, so ``du/dtheta ~= 0`` and the
*total* gradient is dominated by the *partial* ``dR/dtheta|_u``. More
importantly, the return-mapping has an elastic/plastic switch
(``np.where(f > 0, ...)``) that is non-smooth: when ``k`` is perturbed,
quadrature points flip across the yield surface, and the frozen-partition
adjoint cannot capture this, producing a wrong ``dL/dk`` through the implicit
path. Holding ``u`` fixed removes the implicit term entirely, and the direct
path is exact (verified: AD == FD to machine precision in the gradient check).

The trade-off is that we ignore the small ``du/dtheta`` correction. For a
homogeneous specimen under displacement control this is negligible; for
heterogeneous / force-controlled problems the full adjoint would be needed
(future work).
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

from jax_fem.solver import solver
from jax_fem.generate_mesh import (
    box_mesh_gmsh, cylinder_mesh_gmsh, get_meshio_cell_type, Mesh)

from src.models.differentiable_dp import DifferentiableDruckerPrager


# PETSc-free direct solver (scipy). Works on macOS; switch to 'petsc_solver'
# with LU/MUMPS on Linux/GPU for large problems.
SOLVER_OPTIONS = {'umfpack_solver': {}}


class TriaxialForwardProblem(DifferentiableDruckerPrager):
    """Triaxial BVP: confining pressure (Neumann) + axial compression (Dirichlet).

    The confining pressure is essential to make the cohesion ``k``
    identifiable: in pure uniaxial-strain compression the stress path never
    reaches the DP yield surface, so ``k`` carries no gradient. Lateral
    confinement pushes the stress state onto the yield surface so the plastic
    plateau (governed by ``k``) shows up in the reaction curve.
    """

    def __init__(self, mesh, confining_p, axial_disp,
                 nu=0.3, alpha=0.3):
        self.confining_p = confining_p
        self.axial_disp = axial_disp
        Lz = float(np.max(mesh.points[:, 2]))

        def bottom(p): return np.isclose(p[2], 0., atol=1e-5)
        def top(p): return np.isclose(p[2], Lz, atol=1e-5)
        def left(p): return np.isclose(p[0], 0., atol=1e-5)
        def right(p): return np.isclose(p[0], float(np.max(mesh.points[:, 0])), atol=1e-5)
        def front(p): return np.isclose(p[1], 0., atol=1e-5)
        def back(p): return np.isclose(p[1], float(np.max(mesh.points[:, 1])), atol=1e-5)

        dirichlet_bc_info = [[bottom, top], [2, 2],
                             [lambda p: 0., lambda p: axial_disp]]
        location_fns = [left, right, front, back]
        super().__init__(mesh, vec=3, dim=3,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns, nu=nu, alpha=alpha)

    def get_surface_maps(self):
        """Confining pressure tractions on the four lateral faces."""
        p = self.confining_p
        return [
            lambda u, x, *a: np.array([p, 0., 0.]),    # left  (+x)
            lambda u, x, *a: np.array([-p, 0., 0.]),   # right (-x)
            lambda u, x, *a: np.array([0., p, 0.]),    # front (+y)
            lambda u, x, *a: np.array([0., -p, 0.]),   # back  (-y)
        ]


class CylindricalTriaxialForwardProblem(DifferentiableDruckerPrager):
    """Triaxial BVP on a CYLINDRICAL specimen (real triaxial geometry).

    Same loading concept as the box version (confining pressure + axial
    compression) but on a cylinder: the lateral confining pressure acts on the
    single curved side surface and points in the outward RADIAL direction
    (which varies with position around the circumference), and the top/bottom
    caps take the axial Dirichlet BC.

    Use ``make_cylinder_mesh`` to build the mesh. The cylinder axis is z.
    """

    def __init__(self, mesh, confining_p, axial_disp, R,
                 nu=0.3, alpha=0.3):
        self.confining_p = confining_p
        self.axial_disp = axial_disp
        self.R = R
        Lz = float(np.max(mesh.points[:, 2]))

        def bottom(p): return np.isclose(p[2], 0., atol=1e-5)
        def top(p): return np.isclose(p[2], Lz, atol=1e-5)
        # Lateral curved surface: points at radius ~ R (allow mesh tolerance).
        def lateral(p): return np.isclose(np.sqrt(p[0]**2 + p[1]**2), R, atol=1e-3)

        dirichlet_bc_info = [[bottom, top], [2, 2],
                             [lambda p: 0., lambda p: axial_disp]]
        location_fns = [lateral]
        super().__init__(mesh, vec=3, dim=3,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns, nu=nu, alpha=alpha)

    def get_surface_maps(self):
        """Confining pressure on the curved lateral surface, radial outward.

        The traction at a surface point (x, y, z) is p * (x, y, 0)/R, i.e.
        magnitude p in the outward radial direction. This is the physically
        correct confining pressure for a cylinder (unlike the box, where each
        face has a fixed normal).
        """
        p = self.confining_p
        R = self.R

        def lateral_traction(u, x, *a):
            r = np.sqrt(x[0]**2 + x[1]**2)
            # Outward unit radial normal; guard against r=0 (never on the
            # surface, but keeps AD safe).
            n = np.where(r > 0., x[0] / r, 0.), np.where(r > 0., x[1] / r, 0.), 0.
            return np.array([p * n[0], p * n[1], n[2]])
        return [lateral_traction]


def make_box_mesh(Nx=2, Ny=2, Nz=2, Lx=10., Ly=10., Lz=10., data_dir=None):
    """Build a HEX8 box mesh. Returns the jax-fem ``Mesh``."""
    if data_dir is None:
        data_dir = os.path.join(project_root, 'results', '_mesh_tmp')
    os.makedirs(data_dir, exist_ok=True)
    mm = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz,
                       data_dir=data_dir, ele_type='HEX8')
    return Mesh(mm.points, mm.cells_dict[get_meshio_cell_type('HEX8')])


def make_cylinder_mesh(R=5., H=10., circle_mesh=6, height_mesh=10,
                       data_dir=None):
    """Build a structured HEX cylinder mesh (the real triaxial specimen shape).

    A box/cube is a fine demo geometry, but real triaxial tests use cylindrical
    specimens. This wraps jax-fem's ``cylinder_mesh_gmsh`` (transfinite extruded
    mesh: a square core + 4 arc segments per layer, extruded along the axis).
    The cylinder axis is z, height H along z, radius R in the xy-plane.
    """
    if data_dir is None:
        data_dir = os.path.join(project_root, 'results', '_mesh_tmp')
    os.makedirs(data_dir, exist_ok=True)
    mm = cylinder_mesh_gmsh(data_dir=data_dir, R=R, H=H,
                            circle_mesh=circle_mesh, hight_mesh=height_mesh)
    return Mesh(mm.points, mm.cells_dict[get_meshio_cell_type('HEX8')])


def dp_stress(epsilon, E, k, nu, alpha, dim):
    """Pure-JAX single-step DP stress with explicit (traced) parameters.

    Keeping E and k as explicit function arguments (rather than reading them
    from a closure / instance attribute) is what keeps them in the autodiff
    graph for the direct-differentiation loss.
    """
    a = 0.1 * k
    mu = E / (2. * (1. + nu))
    lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
    sigma_trial = lmbda * np.trace(epsilon) * np.eye(dim) + 2. * mu * epsilon
    I1 = np.trace(sigma_trial)
    s_dev = sigma_trial - (I1 / 3.) * np.eye(dim)
    J2 = 0.5 * np.sum(s_dev * s_dev)
    sqrt_J2_reg = np.sqrt(J2 + a * a)
    f_yield = sqrt_J2_reg + alpha * I1 - k
    f_plus = np.where(f_yield > 0., f_yield, 0.)
    denom = 1. + 3. * alpha * alpha
    n_dev = np.where(sqrt_J2_reg == 0., 0., s_dev / (2. * sqrt_J2_reg))
    sigma = sigma_trial - (f_plus / denom) * (n_dev + alpha * np.eye(dim))
    sigma_apex = (k / (3. * alpha)) * np.eye(dim)
    at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
    return np.where(at_apex, sigma_apex, sigma)


def solve_displacement(problem, params):
    """Run one forward BVP solve for the given parameters.

    Returns the solved displacement dofs as a plain (detached) JAX array. The
    solve itself is *not* differentiated (direct strategy); only the subsequent
    stress evaluation is.
    """
    problem.set_params(params)
    sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
    return sol_list[0]


def reaction_loss(problem, sol, params, target_sigma_zz):
    """Differentiable reaction-force loss for one load level.

    ``L = (sigma_zz_avg(u(params); params) - target)^2`` where ``u`` is held
    fixed at the solved field ``sol`` (direct differentiation).

    Parameters
    ----------
    problem : TriaxialForwardProblem
        Provides ``fe`` (for strain recovery + quadrature weights) and the
        fixed constants nu, alpha.
    sol : array
        Solved displacement dofs (treated as a constant).
    params : array [E, k]
        Inverted parameters (traced by autodiff).
    target_sigma_zz : float
        Reference reaction stress for this load level.
    """
    E, k = params[0], params[1]
    nu, alpha, dim = problem.nu, problem.alpha, problem.dim
    fe = problem.fe

    u_grads = fe.sol_to_grad(sol)
    strains = 0.5 * (u_grads + jax.numpy.transpose(u_grads, (0, 1, 3, 2)))
    vmap_stress = jax.vmap(jax.vmap(
        lambda eps: dp_stress(eps, E, k, nu, alpha, dim)))
    sigmas = vmap_stress(strains)
    weighted = sigmas.reshape(-1, dim, dim) * fe.JxW.reshape(-1)[:, None, None]
    avg_sigma = np.sum(weighted, axis=0) / np.sum(fe.JxW)
    return (avg_sigma[2, 2] - target_sigma_zz) ** 2


def total_loss(problem_list, sol_list, params, targets):
    """Sum of reaction losses over multiple load levels.

    Using several load levels jointly identifies both E (from the elastic
    slope) and k (from the plastic plateau): a single level can only constrain
    one combination of the two.
    """
    total = 0.
    for prob, sol, tgt in zip(problem_list, sol_list, targets):
        total = total + reaction_loss(prob, sol, params, tgt)
    return total
