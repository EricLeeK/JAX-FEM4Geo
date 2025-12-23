"""
Strategy 3: Smooth Approximation
"""
import jax
import jax.numpy as np
import os
import sys

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh

class SmoothDruckerPrager(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]
        self.smooth_k = 50.0 # Sigmoid steepness

    def set_params(self, params):
        self.E_val = params[0]
        self.k_val = params[1]

    def get_tensor_map(self):
        def safe_divide(x, y):
            # Also smooth safe divide? Usually simpler to add epsilon
            return x / (y + 1e-12)

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            E = self.E_val
            k = self.k_val
            nu = 0.3
            alpha = 0.3
            a = 0.1 * k

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
            
            # Smooth Approximation
            # 1. Softplus for Macaulay bracket: <x> ~ softplus(x) = log(1+exp(x))
            # Scaled softplus: softplus(beta*x)/beta -> max(0,x) as beta -> inf
            beta = self.smooth_k
            f_yield_plus = jax.nn.softplus(beta * f_yield) / beta
            
            # 2. Sigmoid for conditional switching (at apex)
            # Not strictly needed if we just ignore apex singularity or smooth it naturally, 
            # but let's implement the blending as requested.
            
            # Standard return map direction
            n_dev = safe_divide(s_dev, 2. * sqrt_J2_reg)
            
            # Plastic multiplier
            delta_lambda = f_yield_plus / (1. + 3. * alpha * alpha)
            
            sigma_regular = sigma_trial - delta_lambda * (n_dev + alpha * np.eye(self.dim))
            
            # Apex logic (smoothed)
            # condition: (f_yield > 0) AND (I1 > k/alpha)
            # sigmoid(k * x) -> 0 or 1
            
            cond1 = jax.nn.sigmoid(beta * f_yield)
            cond2 = jax.nn.sigmoid(beta * (I1 - k / alpha))
            at_apex_weight = cond1 * cond2
            
            sigma_apex = (k / (3. * alpha)) * np.eye(self.dim)
            
            # Blend
            sigma = at_apex_weight * sigma_apex + (1. - at_apex_weight) * sigma_regular
            
            return sigma

        return stress_return_map

def run_test(displacement):
    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 2, 2, 2
    data_dir = os.path.join(project_root, 'results', 'test_output_dp_s3')
    os.makedirs(data_dir, exist_ok=True)
    
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz, data_dir=data_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def bottom(p): return np.isclose(p[2], 0.)
    def top(p): return np.isclose(p[2], Lz)
    dirichlet_bc_info = [[bottom, top], [2, 2], [lambda p: 0., lambda p: displacement]]
    
    problem = SmoothDruckerPrager(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    
    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options, adjoint_solver_options=solver_options)

    def loss_fn(params):
        sol_list = fwd_pred(params)
        return np.sum(sol_list[0]**2)

    params_init = np.array([70000.0, 250.0])
    
    print(f"Testing Displacement: {displacement}")
    loss_val, grad_ad = jax.value_and_grad(loss_fn)(params_init)
    
    eps = 1.0
    loss_plus_E = loss_fn(params_init + np.array([eps, 0.0]))
    grad_fd_E = (loss_plus_E - loss_val) / eps
    
    err_E = np.abs(grad_ad[0] - grad_fd_E) / (np.abs(grad_fd_E) + 1e-10)
    print(f"  AD Grad E: {grad_ad[0]:.4e}, FD Grad E: {grad_fd_E:.4e}, Error: {err_E:.2e}")
    
    if err_E < 1e-3:
        print("  SUCCESS")
    else:
        print("  GRADIENT MISMATCH")

if __name__ == "__main__":
    displacements = [-0.001, -0.01, -0.1]
    print("Strategy 3: Smooth Approximation")
    print("="*40)
    for disp in displacements:
        try:
            run_test(disp)
        except Exception as e:
            print(f"  FAILURE at {disp}: {e}")
