"""
Differentiability Test for Drucker-Prager Model
"""

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

class DifferentiableDruckerPrager(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        # Initial internal variables
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def set_params(self, params):
        """params is a list: [E, k]"""
        self.E_val = params[0]
        self.k_val = params[1]

    def get_tensor_map(self):
        def safe_sqrt(x):
            return np.where(x > 0., np.sqrt(x), 0.)

        def safe_divide(x, y):
            return np.where(y == 0., 0., x / y)

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            # Material constants from set_params
            E = self.E_val
            k = self.k_val
            nu = 0.3
            alpha = 0.3
            a = 0.1 * k # increased regularization

            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))

            epsilon_crt = 0.5 * (u_grad + u_grad.T)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = lmbda * np.trace(epsilon_inc) * np.eye(self.dim) + 2. * mu * epsilon_inc + sigma_old

            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(self.dim)
            J2 = 0.5 * np.sum(s_dev * s_dev)

            sqrt_J2_reg = np.sqrt(J2 + a * a)
            f_yield = sqrt_J2_reg + alpha * I1 - k
            
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            n_dev = safe_divide(s_dev, 2. * sqrt_J2_reg)
            delta_lambda = f_yield_plus / (1. + 3. * alpha * alpha)
            
            sigma = sigma_trial - delta_lambda * (n_dev + alpha * np.eye(self.dim))
            
            # Apex check
            sigma_apex = (k / (3. * alpha)) * np.eye(self.dim)
            at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
            sigma = np.where(at_apex, sigma_apex, sigma)
            return sigma

        return stress_return_map

def run_dp_grad_test():
    # Setup small mesh for speed
    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 2, 2, 2
    
    # Use temp output dir
    data_dir = os.path.join(project_root, 'results', 'test_output_dp')
    os.makedirs(data_dir, exist_ok=True)
    
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz, data_dir=data_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # BCs
    def bottom(p): return np.isclose(p[2], 0.)
    def top(p): return np.isclose(p[2], Lz)
    dirichlet_bc_info = [[bottom, top], [2, 2], [lambda p: 0., lambda p: -1e-5]] # Extremely small displacement
    
    problem = DifferentiableDruckerPrager(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    
    # ad_wrapper enables adjoint-based AD
    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options, adjoint_solver_options=solver_options)

    def loss_fn(params):
        """Loss = sum of squared displacements"""
        sol_list = fwd_pred(params)
        return np.sum(sol_list[0]**2)

    print("\n" + "="*60)
    print("Drucker-Prager Differentiability Verification (AD vs FD)")
    print("="*60)

    # Initial parameters [E, k]
    params_init = np.array([70000.0, 250.0])
    
    # 1. Compute Gradient via AD
    t0 = time.time()
    loss_val, grad_ad = jax.value_and_grad(loss_fn)(params_init)
    t1 = time.time()
    print(f"AD Gradient computed in {t1-t0:.4f}s")
    print(f"  dLoss/dE (AD): {grad_ad[0]:.8e}")
    print(f"  dLoss/dk (AD): {grad_ad[1]:.8e}")

    # 2. Compute Gradient via Finite Difference (FD) for verification
    print("\nComputing Finite Difference (FD) for verification...")
    eps = 1.0 # 1 MPa perturbation
    
    # dLoss/dE
    loss_plus_E = loss_fn(params_init + np.array([eps, 0.0]))
    grad_fd_E = (loss_plus_E - loss_val) / eps
    
    # dLoss/dk
    loss_plus_k = loss_fn(params_init + np.array([0.0, eps]))
    grad_fd_k = (loss_plus_k - loss_val) / eps
    
    print(f"  dLoss/dE (FD): {grad_fd_E:.8e}")
    print(f"  dLoss/dk (FD): {grad_fd_k:.8e}")

    # 3. Summary
    err_E = np.abs(grad_ad[0] - grad_fd_E) / (np.abs(grad_fd_E) + 1e-10)
    err_k = np.abs(grad_ad[1] - grad_fd_k) / (np.abs(grad_fd_k) + 1e-10)
    
    print("\nRelative Errors:")
    print(f"  Error in dLoss/dE: {err_E:.2e}")
    print(f"  Error in dLoss/dk: {err_k:.2e}")

    if err_E < 1e-3 and err_k < 1e-3:
        print("\nSUCCESS: Drucker-Prager AD gradient matches FD gradient!")
    else:
        print("\nWARNING: Significant discrepancy between AD and FD gradients.")

if __name__ == "__main__":
    run_dp_grad_test()
