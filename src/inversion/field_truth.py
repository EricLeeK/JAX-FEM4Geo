"""Synthetic heterogeneous truth fields + sparse observation data generation.

Implements the three preset heterogeneous E fields required by the thesis
proposal §3.2(3): layered, soft-inclusion, and random distribution. For each,
runs a forward solve at the truth field and samples displacement at a SPARSE
set of observation nodes (mimicking limited field monitoring), producing the
"observed data" for inversion.

This is the "synthetic-data-then-invert" paradigm the proposal §4.1(1)
prescribes for objective, quantitative accuracy assessment.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jax_fem.solver import solver
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh

from src.models.elastic_field import ElasticFieldProblem

# Truth-data forward solve: NON-differentiated, so we use umfpack (robust direct
# solver) — it handles heterogeneous stiffness fields reliably, unlike the
# iterative jax_solver. The inversion's ad_wrapper path uses jax_solver instead
# (see run_field_inversion.SOLVER_OPTIONS) because only jax_solver is
# JAX-traceable for the adjoint.
TRUTH_SOLVER_OPTIONS = {'umfpack_solver': {}}

# jax_solver options for the differentiated path (passed through to callers).
SOLVER_OPTIONS = {'jax_solver': {'precond': True}}


# ---------------------------------------------------------------------------
# Mesh + problem factory

def build_problem(Nx=6, Ny=6, Nz=6, L=10., disp=-0.05, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(project_root, 'results', '_field_mesh')
    os.makedirs(data_dir, exist_ok=True)
    mm = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=L, domain_y=L, domain_z=L,
                       data_dir=data_dir, ele_type='HEX8')
    mesh = Mesh(mm.points, mm.cells_dict[get_meshio_cell_type('HEX8')])

    def bottom(p): return np.isclose(p[2], 0., atol=1e-5)
    def top(p): return np.isclose(p[2], L, atol=1e-5)
    dbci = [[bottom, top], [2, 2], [lambda p: 0., lambda p: disp]]
    return ElasticFieldProblem(mesh, vec=3, dim=3, dirichlet_bc_info=dbci), mesh


# ---------------------------------------------------------------------------
# Preset heterogeneous E fields

def cell_centroids(mesh):
    """Centroid of each cell (num_cells, 3) for field assignment by position."""
    pts = onp.array(mesh.points)
    cells = onp.array(mesh.cells)
    return pts[cells].mean(axis=1)


def layered_field(mesh, E_high=80.0e3, E_low=40.0e3):
    """Horizontally layered: stiff/soft alternating by z-height."""
    c = cell_centroids(mesh)
    z = c[:, 2]
    z_mid = (z.min() + z.max()) / 2.
    return np.where(z < z_mid, E_high, E_low)  # (num_cells,)


def inclusion_field(mesh, E_host=70.0e3, E_soft=25.0e3,
                    frac=0.25):
    """Host matrix with a central soft inclusion (weak layer).

    A spherical region at the centre is set to E_soft; the rest is E_host.
    ``frac`` controls the inclusion radius relative to the domain.
    """
    c = cell_centroids(mesh)
    center = c.mean(axis=0)
    r = onp.linalg.norm(c - center, axis=1)
    r_incl = frac * onp.linalg.norm(c - center, axis=1).max()
    return np.where(r < r_incl, E_soft, E_host)


def random_field(mesh, E_mean=70.0e3, E_std=15.0e3, seed=0):
    """Spatially correlated random field (Gaussian around E_mean).

    A smooth random perturbation: each cell gets E_mean + E_std * N(0,1) with
    a fixed seed for reproducibility. Clipped to a physically positive range.
    """
    rng = onp.random.default_rng(seed)
    n = len(mesh.cells)
    raw = rng.standard_normal(n) * E_std
    E = onp.clip(E_mean + raw, 20.0e3, 120.0e3)
    return np.array(E)


FIELD_FACTORIES = {
    'layered': layered_field,
    'inclusion': inclusion_field,
    'random': random_field,
}


def expand_to_quads(E_per_cell, num_quads):
    """Broadcast a (num_cells,) field to (num_cells, num_quads) for internal_vars."""
    return np.broadcast_to(E_per_cell[:, None], (len(E_per_cell), num_quads))


# ---------------------------------------------------------------------------
# Observation sampling

def observation_nodes(mesh, fraction=0.3, seed=1):
    """Pick a random sparse subset of NON-Dirichlet nodes as observation points.

    Excludes the bottom (z=0, fixed) and top (z=L, prescribed) faces so we
    observe interior/lateral response only — the realistic "sparse monitoring"
    scenario. Returns boolean mask over nodes.
    """
    pts = onp.array(mesh.points)
    L = pts[:, 2].max()
    free = ~((onp.isclose(pts[:, 2], 0., atol=1e-5)) |
             (onp.isclose(pts[:, 2], L, atol=1e-5)))
    free_inds = onp.where(free)[0]
    rng = onp.random.default_rng(seed)
    n_obs = max(1, int(fraction * len(free_inds)))
    chosen = rng.choice(free_inds, size=n_obs, replace=False)
    mask = onp.zeros(len(pts), dtype=bool)
    mask[chosen] = True
    return mask


# ---------------------------------------------------------------------------
# Full truth-data generation for one field type

def generate_field_truth(field_type, Nx=6, Ny=6, Nz=6, obs_fraction=0.3,
                         data_dir=None):
    """Build a truth field, solve forward, sample sparse observations.

    Returns dict with: truth_E (num_cells,), obs_mask (num_nodes bool),
    obs_disp ((n_obs,3)), mesh, problem.
    """
    problem, mesh = build_problem(Nx=Nx, Ny=Ny, Nz=Nz, data_dir=data_dir)
    n_quads = problem.fe.num_quads

    E_truth = FIELD_FACTORIES[field_type](mesh)  # (num_cells,)
    E_field = expand_to_quads(E_truth, n_quads)
    problem.set_params(E_field)

    sol_list = solver(problem, solver_options=TRUTH_SOLVER_OPTIONS)
    sol = sol_list[0]  # (num_nodes, 3)

    obs_mask = observation_nodes(mesh, fraction=obs_fraction)
    obs_disp = onp.array(sol)[obs_mask]  # (n_obs, 3)

    print(f"[Truth:{field_type}] {len(mesh.cells)} cells, "
          f"{int(obs_mask.sum())} obs nodes, "
          f"E range [{float(E_truth.min()):.0f}, {float(E_truth.max()):.0f}] MPa")
    return {'field_type': field_type, 'truth_E': onp.array(E_truth),
            'obs_mask': obs_mask, 'obs_disp': obs_disp,
            'mesh': mesh, 'problem': problem}
