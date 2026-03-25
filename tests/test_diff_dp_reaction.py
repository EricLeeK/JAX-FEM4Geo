
import jax
import jax.numpy as np
import os
import sys
import time

# 环境路径设置
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
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def set_params(self, params):
        self.E_val = params[0]
        self.k_val = params[1]

    def get_tensor_map(self):
        def stress_return_map(u_grad, sigma_old, epsilon_old):
            E, k = self.E_val, self.k_val
            nu, alpha, a = 0.3, 0.3, 2.5 # 固定 a 为 2.5 (0.01k)
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            epsilon_inc = 0.5 * (u_grad + u_grad.T) - epsilon_old
            sigma_trial = lmbda * np.trace(epsilon_inc) * np.eye(self.dim) + 2. * mu * epsilon_inc + sigma_old
            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(self.dim)
            J2 = 0.5 * np.sum(s_dev * s_dev)
            f_yield = np.sqrt(J2 + a * a) + alpha * I1 - k
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            delta_lambda = f_yield_plus / (1. + 3. * alpha * alpha)
            sigma = sigma_trial - delta_lambda * (s_dev / np.sqrt(J2 + a * a) + alpha * np.eye(self.dim))
            return sigma
        return stress_return_map

def run_reaction_grad_test(displacement):
    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 2, 2, 2
    data_dir = os.path.join(project_root, 'results', 'test_reaction')
    os.makedirs(data_dir, exist_ok=True)
    
    meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz, data_dir=data_dir, ele_type='HEX8')
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[get_meshio_cell_type('HEX8')])

    def bottom(p): return np.isclose(p[2], 0.)
    def top(p): return np.isclose(p[2], Lz)
    
    dirichlet_bc_info = [[bottom, bottom, bottom, top], [0, 1, 2, 2], [lambda p: 0., lambda p: 0., lambda p: 0., lambda p: displacement]]
    
    problem = DifferentiableDruckerPrager(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options, adjoint_solver_options=solver_options)

    def loss_fn(params):
        sol_list = fwd_pred(params)
        # 使用位移平方和乘以 E 来模拟反力相关的 Loss 信号
        return np.sum(sol_list[0]**2) * params[0]

    params_init = np.array([70000.0, 250.0])
    print(f"\nTesting Displacement: {displacement} mm")
    if abs(displacement) > 0.04:
        print("Status: ELASTO-PLASTIC REGIME")
    else:
        print("Status: ELASTIC REGIME")

    # AD
    loss_val, grad_ad = jax.value_and_grad(loss_fn)(params_init)
    
    # FD
    eps = 1.0
    loss_plus = loss_fn(params_init + np.array([eps, 0.0]))
    grad_fd = (loss_plus - loss_val) / eps

    err = np.abs(grad_ad[0] - grad_fd) / (np.abs(grad_fd) + 1e-10)
    print(f"  AD dLoss/dE: {grad_ad[0]:.6e}")
    print(f"  FD dLoss/dE: {grad_fd:.6e}")
    print(f"  Relative Error: {err:.2e}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Multi-Stage Differentiability Test (Elastic -> Plastic)")
    print("="*50)
    # 分别测试：弹性阶段、屈服附近、深层塑性
    for d in [-0.01, -0.05, -0.1]:
        run_reaction_grad_test(d)
