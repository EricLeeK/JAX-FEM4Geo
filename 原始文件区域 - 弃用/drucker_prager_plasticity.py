"""
Drucker-Prager Plasticity with Hyperbolic Apex Regularization

This module implements the Drucker-Prager yield criterion using JAX-FEM.
For detailed theory, see: drucker_prager_theory.md
"""

import jax
import jax.numpy as np
import numpy as onp
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

# Elastic properties
E = 70.0e3      # Young's modulus [MPa]
nu = 0.3        # Poisson's ratio [-]

# Drucker-Prager parameters (example: concrete-like material)
alpha = 0.3     # Friction coefficient [-]
k = 250.0       # Cohesion parameter [MPa]

# Apex regularization parameter (see drucker_prager_theory.md Section 4)
a = 0.01 * k    # Small positive value to smooth the apex


# =============================================================================
# Drucker-Prager Plasticity Problem Class
# =============================================================================

class DruckerPragerPlasticity(Problem):
    """
    Drucker-Prager plasticity with hyperbolic apex regularization.
    Inherits from JAX-FEM Problem class.
    """

    def custom_init(self):
        """Initialize internal variables for stress and strain history."""
        self.fe = self.fes[0]
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, 
                                       self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def get_tensor_map(self):
        """Return the stress computation function for the FEM solver."""
        _, stress_return_map = self.get_maps()
        return stress_return_map

    def get_maps(self):
        """Define strain, elastic stress, and DP return mapping functions."""
        
        def safe_sqrt(x):
            """Differentiable sqrt that handles x=0."""
            return np.where(x > 0., np.sqrt(x), 0.)

        def safe_divide(x, y):
            """Safe division to avoid NaN when y=0."""
            return np.where(y == 0., 0., x / y)

        def strain(u_grad):
            """Compute small strain tensor from displacement gradient."""
            return 0.5 * (u_grad + u_grad.T)

        def elastic_stress(epsilon):
            """Compute elastic stress using Hooke's law."""
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2. * mu * epsilon

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            """
            Drucker-Prager return mapping with hyperbolic regularization.
            
            The regularized yield function: f = sqrt(J2 + a^2) + alpha*I1 - k
            This avoids the apex singularity (see drucker_prager_theory.md).
            """
            # Compute current strain and trial stress
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = elastic_stress(epsilon_inc) + sigma_old

            # Compute stress invariants
            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(self.dim)
            J2 = 0.5 * np.sum(s_dev * s_dev)

            # Regularized yield function (hyperbolic smoothing at apex)
            sqrt_J2_reg = np.sqrt(J2 + a * a)
            f_yield = sqrt_J2_reg + alpha * I1 - k

            # Plastic correction (only if f_yield > 0)
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            
            # Flow direction and consistency parameter
            # For associated flow with regularization
            n_dev = safe_divide(s_dev, 2. * sqrt_J2_reg)
            delta_lambda = f_yield_plus / (1. + 3. * alpha * alpha)
            
            # Return stress
            sigma = sigma_trial - delta_lambda * (n_dev + alpha * np.eye(self.dim))

            # Handle apex return (when I1 would exceed apex location)
            sigma_apex = (k / (3. * alpha)) * np.eye(self.dim)
            at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
            sigma = np.where(at_apex, sigma_apex, sigma)

            return sigma

        return strain, stress_return_map

    def stress_strain_fns(self):
        """Return vectorized strain and stress functions."""
        strain, stress_return_map = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol):
        """Update internal stress and strain variables after each load step."""
        u_grads = self.fe.sol_to_grad(sol)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()
        self.sigmas_old = vmap_stress_rm(u_grads, self.sigmas_old, self.epsilons_old)
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def compute_avg_stress(self):
        """Compute volume-averaged stress for post-processing."""
        sigma = np.sum(
            self.sigmas_old.reshape(-1, self.fe.vec, self.dim) * 
            self.fe.JxW.reshape(-1)[:, None, None], 
            axis=0
        )
        vol = np.sum(self.fe.JxW)
        return sigma / vol


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # Mesh Generation
    # -------------------------------------------------------------------------
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data_dp')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)

    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 5, 5, 5

    meshio_mesh = box_mesh_gmsh(
        Nx=Nx, Ny=Ny, Nz=Nz,
        domain_x=Lx, domain_y=Ly, domain_z=Lz,
        data_dir=data_dir, ele_type=ele_type,
    )
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # -------------------------------------------------------------------------
    # Boundary Conditions (Uniaxial Compression Test)
    # -------------------------------------------------------------------------
    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def dirichlet_val_bottom(point):
        return 0.

    def get_dirichlet_top(disp):
        def val_fn(point):
            return disp
        return val_fn

    # Loading sequence: compress then unload
    disps = np.hstack((np.linspace(0., -0.15, 16), np.linspace(-0.14, 0., 15)))

    location_fns = [bottom, top]
    value_fns = [dirichlet_val_bottom, get_dirichlet_top(disps[0])]
    vecs = [2, 2]  # z-component
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    # -------------------------------------------------------------------------
    # Problem Setup and Solution
    # -------------------------------------------------------------------------
    problem = DruckerPragerPlasticity(mesh, vec=3, dim=3, 
                                       dirichlet_bc_info=dirichlet_bc_info)
    
    avg_stresses = []
    
    print("\n" + "="*60)
    print("Drucker-Prager Plasticity with Hyperbolic Apex Regularization")
    print("="*60)
    print(f"Material: E={E} MPa, nu={nu}")
    print(f"DP params: alpha={alpha}, k={k} MPa, a={a} MPa")
    print("="*60 + "\n")

    overall_start = time.time()

    for i, disp in enumerate(disps):
        print(f"Step {i+1}/{len(disps)}, displacement = {disp:.4f} mm")
        
        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
        problem.fe.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        
        sol_list = solver(problem, solver_options={'petsc_solver': {}}) 
        problem.update_stress_strain(sol_list[0])
        
        avg_stress = problem.compute_avg_stress()
        print(f"  Avg stress (zz): {avg_stress[2, 2]:.6f} MPa")
        avg_stresses.append(avg_stress)
        
        vtk_path = os.path.join(data_dir, f'vtk/u_{i:03d}.vtu')
        save_sol(problem.fe, sol_list[0], vtk_path)

    avg_stresses = np.array(avg_stresses)
    
    # Save to CSV
    strains_zz = disps / Lz # approximation for engineering strain
    stresses_zz = avg_stresses[:, 2, 2]
    data = np.column_stack((strains_zz, stresses_zz))
    onp.savetxt("study-fold/fem_data.csv", data, delimiter=",", header="Strain_ZZ,Stress_ZZ", comments="")
    print("FEM Data saved to study-fold/fem_data.csv")

    overall_end = time.time()
    total_time = overall_end - overall_start

    # -------------------------------------------------------------------------
    # Post-processing: Stress-Strain Plot
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"Simulation Complete in {total_time:.2f} seconds.")
    print("Generating stress-strain plot...")
    print("="*60)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(disps / Lz * 100, avg_stresses[:, 2, 2], 
             color='blue', marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.xlabel('Strain (z-z) [%]', fontsize=16)
    plt.ylabel('Volume Averaged Stress (z-z) [MPa]', fontsize=16)
    plt.title('Drucker-Prager Plasticity: Uniaxial Compression Test', fontsize=18)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(data_dir, 'stress_strain_curve.png')
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to: {fig_path}")
    plt.show()
