import jax
import jax.numpy as np
import os
import sys
import time

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh

# =============================================================================
# Differentiable Problem (Verification of Pipeline)
# =============================================================================

class DiffElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        # Parameter: Young's Modulus E
        self.E_field = 70.0e3 * np.ones((len(self.fe.cells), self.fe.num_quads))
        self.internal_vars = [self.E_field]
        
    def set_params(self, params):
        # params: scalar E
        self.E_field = params * np.ones((len(self.fe.cells), self.fe.num_quads))
        self.internal_vars = [self.E_field]

    def get_tensor_map(self):
        nu = 0.3
        def stress_fn(u_grad, E_local):
            epsilon = 0.5 * (u_grad + u_grad.T)
            mu = E_local / (2. * (1. + nu))
            lmbda = E_local * nu / ((1. + nu) * (1. - 2. * nu))
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2. * mu * epsilon
        return stress_fn

# =============================================================================
# Execution
# =============================================================================

def run_grad_test():
    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 2, 2, 2
    
    data_dir = os.path.join(project_root, 'results', 'test_output_elastic')
    os.makedirs(data_dir, exist_ok=True)
    
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz, data_dir=data_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def bottom(p): return np.isclose(p[2], 0.)
    def top(p): return np.isclose(p[2], Lz)
    def corner(p): return np.logical_and(np.isclose(p[0], 0.), np.isclose(p[1], 0.))
    
    location_fns = [bottom, top, corner, corner]
    vecs = [2, 2, 0, 1]
    value_fns = [lambda p: 0., lambda p: -0.1, lambda p: 0., lambda p: 0.]
    dirichlet_bc_info = [location_fns, vecs, value_fns]
    
    problem = DiffElasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    
    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options, adjoint_solver_options=solver_options)

    def loss_fn(E_val):
        sol_list = fwd_pred(E_val)
        return np.sum(sol_list[0]**2)

    print("\nStarting Pipeline Smoke Test (Elasticity)...")
    
    E_init = 70.0e3
    loss_val, grad_E = jax.value_and_grad(loss_fn)(E_init)
    
    print(f"\nResults:")
    print(f"  Loss Value: {loss_val:.6e}")
    print(f"  Gradient dLoss/dE: {grad_E:.6e}")
    
    if np.abs(grad_E) > 1e-15:
        print("\nSUCCESS: Pipeline is differentiable!")
    else:
        print("\nFAILURE: Gradient is zero.")

if __name__ == "__main__":
    run_grad_test()
