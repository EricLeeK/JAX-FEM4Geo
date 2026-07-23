"""Per-cell Young's modulus field — differentiable linear elasticity.

This is the foundation for high-dimensional heterogeneous parameter-field
inversion (the thesis title's core: "spatially-varying rock properties").
Each finite element carries its own Young's modulus E_cell; the whole E field
is the unknown of the inverse problem.

Mechanism (verified against jax-fem source + the compute_gradients tutorial):
    set_params(params) stores the E field (shape (num_cells, num_quads)) in
    self.internal_vars. jax-fem's kernel machinery vmaps tensor_map over
    (cells, quads), so inside tensor_map E arrives as a scalar at one quadrature
    point. The ad_wrapper adjoint (jax.vjp) differentiates the forward solve
    w.r.t. the full field in one backward pass — cost independent of cell count.

IMPORTANT — solver choice for the adjoint:
    The implicit adjoint (ad_wrapper + jax.grad) requires a JAX-traceable
    linear solver. The 'umfpack_solver' (scipy spsolve) is NOT traceable, so
    gradients through it are WRONG (verified: AD vs FD mismatch by ~1e4). Use
    'jax_solver' (jax.scipy.sparse bicgstab) for any problem differentiated via
    ad_wrapper. The 'umfpack_solver' remains correct and faster for pure
    forward solves and for the direct (fixed-u) differentiation of P0/P1 (which
    never enters the implicit adjoint). This is a critical Mac-adaptation note:
    the SparseMat refactor is fine for forward, but the adjoint needs jax_solver.

We start with LINEAR elasticity: the problem is linear so AD is exact and
reliable, validating the field-inversion machinery before extending to
plasticity (where the adjoint has known active-set issues).
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


class ElasticFieldProblem(Problem):
    """Linear elasticity with a per-cell Young's modulus field.

    The inverted unknown is the E field, ``params`` of shape ``(num_cells,)``.
    Poisson's ratio ``nu`` is fixed (homogeneous), so only the stiffness
    magnitude varies spatially — the simplest heterogeneous field that still
    exercises the full per-cell AD path.
    """

    def __init__(self, mesh, vec=3, dim=3, dirichlet_bc_info=None,
                 location_fns=None, nu=0.3):
        self.nu = nu
        super().__init__(mesh, vec=vec, dim=dim,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)

    def custom_init(self):
        self.fe = self.fes[0]
        # internal_vars holds the E field of shape (num_cells, num_quads).
        # Set by set_params before each solve; default homogeneous for safety.
        n_cells = len(self.fe.cells)
        n_quads = self.fe.num_quads
        self.internal_vars = [70.0e3 * np.ones((n_cells, n_quads))]

    def set_params(self, E_field):
        """Inject the E field, shape ``(num_cells, num_quads)``.

        Following the compute_gradients tutorial exactly: the field is passed
        directly as internal_vars (NOT captured in the tensor_map closure, which
        the tutorial explicitly warns gives WRONG gradients). The inversion
        unknown is per-cell ``(num_cells,)``; expand to quads in the loss
        wrapper via np.broadcast_to (differentiable).
        """
        self.internal_vars = [E_field]

    def get_tensor_map(self):
        """Stress function. Inside the vmap, E is a scalar at one quad point."""
        nu = self.nu
        dim = self.dim

        def stress_fn(u_grad, E):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(dim) + 2. * mu * epsilon

        return stress_fn
