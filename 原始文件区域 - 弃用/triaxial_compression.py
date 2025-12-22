import jax
import jax.numpy as np
import os
import sys

# Add local jax_fem path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'jax-fem-main'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh


# =============================================================================
# Material Parameters
# =============================================================================
E_val = 70.0e3
nu_val = 0.3
alpha_val = 0.3
k_val = 250.0
a_val = 0.01 * k_val

confining_pressure = 20.0  # MPa

# =============================================================================
# Triaxial Problem Class
# =============================================================================

class TriaxialDP(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def get_tensor_map(self):
        def strain(u_grad):
            return 0.5 * (u_grad + u_grad.T)

        def elastic_stress(epsilon):
            mu = E_val / (2. * (1. + nu_val))
            lmbda = E_val * nu_val / ((1. + nu_val) * (1. - 2. * nu_val))
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2. * mu * epsilon

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = elastic_stress(epsilon_inc) + sigma_old

            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(self.dim)
            J2 = 0.5 * np.sum(s_dev * s_dev)

            sqrt_J2_reg = np.sqrt(J2 + a_val * a_val)
            f_yield = sqrt_J2_reg + alpha_val * I1 - k_val

            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            n_dev = np.where(sqrt_J2_reg > 1e-12, s_dev / (2. * sqrt_J2_reg), np.zeros_like(s_dev))
            delta_lambda = f_yield_plus / (1. + 3. * alpha_val * alpha_val)
            
            sigma = sigma_trial - delta_lambda * (n_dev + alpha_val * np.eye(self.dim))

            sigma_apex = (k_val / (3. * alpha_val)) * np.eye(self.dim)
            at_apex = np.logical_and(f_yield > 0., I1 > k_val / alpha_val)
            sigma = np.where(at_apex, sigma_apex, sigma)

            return sigma

        return stress_return_map

    def get_surface_maps(self):
        """
        Define traction on boundary surfaces.
        The indices correspond to location_fns provided during initialization.
        """
        # We will provide 4 location functions for Xmin, Xmax, Ymin, Ymax
        def x_min_traction(u, x):
            return np.array([confining_pressure, 0., 0.])
        
        def x_max_traction(u, x):
            return np.array([-confining_pressure, 0., 0.])
            
        def y_min_traction(u, x):
            return np.array([0., confining_pressure, 0.])
            
        def y_max_traction(u, x):
            return np.array([0., -confining_pressure, 0.])

        return [x_min_traction, x_max_traction, y_min_traction, y_max_traction]

    def update_stress_strain(self, sol):
        u_grads = self.fe.sol_to_grad(sol)
        strain_fn = lambda g: 0.5 * (g + g.T)
        vmap_strain = jax.vmap(jax.vmap(strain_fn))
        stress_fn = self.get_tensor_map()
        vmap_stress = jax.vmap(jax.vmap(stress_fn))
        
        self.sigmas_old = vmap_stress(u_grads, self.sigmas_old, self.epsilons_old)
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def compute_avg_stress(self):
        sigma = np.sum(self.sigmas_old.reshape(-1, 3, 3) * self.fe.JxW.reshape(-1)[:, None, None], axis=0)
        vol = np.sum(self.fe.JxW)
        return sigma / vol


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 5, 5, 5
    
    data_dir = 'data_triaxial'
    os.makedirs(data_dir, exist_ok=True)
    
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz, 
                                 data_dir=data_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # Location functions for Neumann surfaces
    def x_min(p): return np.isclose(p[0], 0., atol=1e-5)
    def x_max(p): return np.isclose(p[0], Lx, atol=1e-5)
    def y_min(p): return np.isclose(p[1], 0., atol=1e-5)
    def y_max(p): return np.isclose(p[1], Ly, atol=1e-5)
    
    # Dirichlet BCs (Z compression and stability)
    def z_bottom(p): return np.isclose(p[2], 0., atol=1e-5)
    def z_top(p): return np.isclose(p[2], Lz, atol=1e-5)
    
    # Fix center point in X and Y to prevent rigid body motion
    # Since we have traction on all X/Y sides, we need some constraint.
    # Alternatively, use symmetry or fix one corner.
    def corner(p): return np.logical_and(np.isclose(p[0], 0, atol=1e-5), np.isclose(p[1], 0, atol=1e-5))

    disps_z = np.linspace(0., -0.2, 21) # Up to 2% strain
    
    # BC Info
    location_fns_dirichlet = [z_bottom, z_top, corner, corner]
    vecs_dirichlet = [2, 2, 0, 1] # Z, Z, X, Y
    value_fns_dirichlet = [lambda p: 0., lambda p: disps_z[0], lambda p: 0., lambda p: 0.]
    dirichlet_bc_info = [location_fns_dirichlet, vecs_dirichlet, value_fns_dirichlet]

    # Initialize problem with location functions for Neumann BCs
    problem = TriaxialDP(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, 
                         location_fns=[x_min, x_max, y_min, y_max])
    
    avg_stresses = []
    
    print(f"\nTriaxial Compression Test (Confining Pressure = {confining_pressure} MPa)")
    
    for i, disp in enumerate(disps_z):
        print(f"Step {i}/{len(disps_z)-1}, Disp = {disp:.4f}")
        dirichlet_bc_info[-1][1] = lambda p: disp # Update top Z disp
        problem.fe.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        
        sol_list = solver(problem, solver_options={'petsc_solver': {}})
        problem.update_stress_strain(sol_list[0])
        
        avg_sigma = problem.compute_avg_stress()
        avg_stresses.append(avg_sigma)
        print(f"  Avg sigma_zz: {avg_sigma[2,2]:.2f} MPa, sigma_xx: {avg_sigma[0,0]:.2f} MPa")

    avg_stresses = np.array(avg_stresses)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(-disps_z/Lz*100, -avg_stresses[:, 2, 2], 'b-o', label=f'Confining={confining_pressure} MPa')
    plt.xlabel('Axial Strain [%]')
    plt.ylabel('Axial Stress (Compression +) [MPa]')
    plt.title('Triaxial Compression Test (Drucker-Prager)')
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, 'triaxial_stress_strain.png'))
    print(f"Result saved to {data_dir}/triaxial_stress_strain.png")
