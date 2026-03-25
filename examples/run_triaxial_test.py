"""
Triaxial Compression Test Simulation with Drucker-Prager Model

This script simulates a standard triaxial compression test:
1. Stage 1 (Consolidation): Apply confining pressure σ3 on all faces
2. Stage 2 (Shearing): Keep confining pressure, apply axial compression

Key outputs:
- Deviatoric stress q = σ1 - σ3
- Mean stress p = (σ1 + 2σ3) / 3
- q-p stress path diagram

Author: Li Shiyao
Date: 2026-01-08
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh


class TriaxialDruckerPrager(Problem):
    """
    Drucker-Prager model for triaxial compression test.
    
    Supports:
    - Neumann BC for confining pressure on lateral faces
    - Dirichlet BC for axial displacement control on top
    - History-dependent plasticity with stress/strain tracking
    """
    
    def __init__(self, mesh, vec=3, dim=3, ele_type='HEX8', 
                 dirichlet_bc_info=None, location_fns=None,
                 E=70.0e3, nu=0.3, alpha=0.3, k=250.0, a=None,
                 confining_pressure=0.0):
        """
        Args:
            confining_pressure: σ3 (positive = compression in geomechanics convention)
        """
        # Store material parameters before calling super().__init__
        self.E = E
        self.nu = nu
        self.alpha = alpha
        self.k = k
        self.a = a if a is not None else 0.01 * k
        self.confining_pressure = confining_pressure
        
        super().__init__(mesh, vec=vec, dim=dim, ele_type=ele_type,
                        dirichlet_bc_info=dirichlet_bc_info,
                        location_fns=location_fns)

    def custom_init(self):
        """Initialize internal variables for stress and strain history."""
        self.fe = self.fes[0]
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, 
                                       self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def set_params(self, params):
        """Update confining pressure (can be called to change during simulation)."""
        self.confining_pressure = params

    def get_tensor_map(self):
        """Return the stress computation function for the FEM solver."""
        E, nu, alpha, k, a = self.E, self.nu, self.alpha, self.k, self.a
        dim = self.dim

        def safe_divide(x, y):
            return np.where(y == 0., 0., x / y)

        def strain(u_grad):
            return 0.5 * (u_grad + u_grad.T)

        def elastic_stress(epsilon):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            return lmbda * np.trace(epsilon) * np.eye(dim) + 2. * mu * epsilon

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = elastic_stress(epsilon_inc) + sigma_old

            I1 = np.trace(sigma_trial)
            s_dev = sigma_trial - (I1 / 3.) * np.eye(dim)
            J2 = 0.5 * np.sum(s_dev * s_dev)

            sqrt_J2_reg = np.sqrt(J2 + a * a)
            f_yield = sqrt_J2_reg + alpha * I1 - k

            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            
            n_dev = safe_divide(s_dev, 2. * sqrt_J2_reg)
            delta_lambda = f_yield_plus / (1. + 3. * alpha * alpha)
            
            sigma = sigma_trial - delta_lambda * (n_dev + alpha * np.eye(dim))

            # Apex handling
            sigma_apex = (k / (3. * alpha)) * np.eye(dim)
            at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
            sigma = np.where(at_apex, sigma_apex, sigma)

            return sigma

        return stress_return_map

    def get_surface_maps(self):
        """
        Define Neumann boundary conditions for confining pressure.
        
        In geomechanics convention: positive pressure = compression
        In FEM (traction): we apply negative traction (pointing inward)
        
        Returns a list of surface map functions, one for each location_fn.
        """
        confining_p = self.confining_pressure
        
        def traction_x_faces(u, point, *args):
            """Traction on faces normal to x-axis (left/right faces)"""
            # Traction vector: pressure acts normal to surface (inward)
            # For x-face: normal is ±x, so traction = [∓p, 0, 0]
            # The sign is handled by the face normal in JAX-FEM
            # We just specify the magnitude of pressure pointing inward
            return np.array([-confining_p, 0., 0.])
        
        def traction_y_faces(u, point, *args):
            """Traction on faces normal to y-axis (front/back faces)"""
            return np.array([0., -confining_p, 0.])
        
        # Return list matching the order of location_fns
        # Order: [left, right, front, back]
        return [traction_x_faces, traction_x_faces, traction_y_faces, traction_y_faces]

    def stress_strain_fns(self):
        """Get vectorized stress-strain functions."""
        def strain(u_grad):
            return 0.5 * (u_grad + u_grad.T)
        
        stress_return_map = self.get_tensor_map()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol):
        """Update internal variables after a load step."""
        u_grads = self.fe.sol_to_grad(sol)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()
        self.sigmas_old = vmap_stress_rm(u_grads, self.sigmas_old, self.epsilons_old)
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def compute_avg_stress(self):
        """Compute volume-averaged stress tensor."""
        sigma = np.sum(
            self.sigmas_old.reshape(-1, self.fe.vec, self.dim) * 
            self.fe.JxW.reshape(-1)[:, None, None], 
            axis=0
        )
        vol = np.sum(self.fe.JxW)
        return sigma / vol
    
    def compute_stress_invariants(self):
        """
        Compute stress invariants for triaxial analysis.
        
        Returns:
            p: Mean stress (positive = compression)
            q: Deviatoric stress
            sigma_1: Major principal stress (axial)
            sigma_3: Minor principal stress (confining)
        """
        avg_sigma = self.compute_avg_stress()
        
        # For triaxial: σ1 = σ_zz (axial), σ2 = σ3 = σ_xx = σ_yy (confining)
        sigma_1 = avg_sigma[2, 2]  # axial stress (z-direction)
        sigma_3 = (avg_sigma[0, 0] + avg_sigma[1, 1]) / 2.0  # confining stress
        
        # Geomechanics convention: compression positive
        # FEM convention: tension positive
        # So we negate for geomechanics output
        p = -(sigma_1 + 2 * sigma_3) / 3.0  # mean stress (compression positive)
        q = -(sigma_1 - sigma_3)  # deviatoric stress
        
        return p, q, -sigma_1, -sigma_3


def run_triaxial_simulation():
    """
    Run a triaxial compression test simulation.
    
    Test setup:
    - Cubic specimen (simplified from cylinder for easier meshing)
    - Stage 1: Isotropic consolidation under confining pressure
    - Stage 2: Axial compression with constant confining pressure
    """
    
    print("\n" + "="*70)
    print("  TRIAXIAL COMPRESSION TEST - Drucker-Prager Model")
    print("="*70)
    
    # =========================================================================
    # Material Parameters
    # =========================================================================
    E = 70.0e3      # Young's modulus [MPa]
    nu = 0.3        # Poisson's ratio
    alpha = 0.3     # DP friction parameter
    k = 250.0       # DP cohesion parameter [MPa]
    a = 2.5         # Apex regularization (0.01 * k)
    
    # Confining pressure (σ3)
    confining_pressure = 50.0  # [MPa] - typical for rock testing
    
    print(f"\nMaterial Parameters:")
    print(f"  E = {E} MPa, ν = {nu}")
    print(f"  α = {alpha}, k = {k} MPa")
    print(f"  Confining Pressure σ3 = {confining_pressure} MPa")
    
    # =========================================================================
    # Mesh Setup
    # =========================================================================
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    
    # Output directory
    results_dir = os.path.join(project_root, 'results', 'triaxial_test')
    os.makedirs(results_dir, exist_ok=True)
    vtk_dir = os.path.join(results_dir, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    
    # Cubic specimen (10mm x 10mm x 10mm)
    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 3, 3, 3  # Coarse mesh for demo
    
    print(f"\nMesh: {Nx}x{Ny}x{Nz} HEX8 elements")
    print(f"Specimen size: {Lx} x {Ly} x {Lz} mm")
    
    meshio_mesh = box_mesh_gmsh(
        Nx=Nx, Ny=Ny, Nz=Nz,
        domain_x=Lx, domain_y=Ly, domain_z=Lz,
        data_dir=results_dir, ele_type=ele_type,
    )
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    
    # =========================================================================
    # Boundary Conditions
    # =========================================================================
    
    # Dirichlet BCs: Fix bottom, apply displacement on top
    def bottom(point): 
        return np.isclose(point[2], 0., atol=1e-5)
    
    def top(point): 
        return np.isclose(point[2], Lz, atol=1e-5)
    
    def zero_val(point): 
        return 0.
    
    def get_disp_top(disp):
        def val_fn(point):
            return disp
        return val_fn
    
    # Neumann BCs: Confining pressure on lateral faces
    def left(point): 
        return np.isclose(point[0], 0., atol=1e-5)
    
    def right(point): 
        return np.isclose(point[0], Lx, atol=1e-5)
    
    def front(point): 
        return np.isclose(point[1], 0., atol=1e-5)
    
    def back(point): 
        return np.isclose(point[1], Ly, atol=1e-5)
    
    # Dirichlet: bottom fixed (all DOFs), top z-displacement controlled
    # Also fix x,y at bottom to prevent rigid body motion
    dirichlet_bc_info = [
        [bottom, bottom, bottom, top],  # location functions
        [0, 1, 2, 2],                    # DOF indices (x, y, z, z)
        [zero_val, zero_val, zero_val, get_disp_top(0.)]  # values
    ]
    
    # Neumann: confining pressure on 4 lateral faces
    location_fns = [left, right, front, back]
    
    # =========================================================================
    # Problem Setup
    # =========================================================================
    problem = TriaxialDruckerPrager(
        mesh, vec=3, dim=3, ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        E=E, nu=nu, alpha=alpha, k=k, a=a,
        confining_pressure=confining_pressure
    )
    
    # =========================================================================
    # Stage 1: Isotropic Consolidation
    # =========================================================================
    print("\n" + "-"*50)
    print("Stage 1: Isotropic Consolidation")
    print("-"*50)
    
    # Apply confining pressure, no axial displacement yet
    problem.confining_pressure = confining_pressure
    
    sol_list = solver(problem, solver_options={'petsc_solver': {}})
    problem.update_stress_strain(sol_list[0])
    
    p0, q0, sig1_0, sig3_0 = problem.compute_stress_invariants()
    print(f"  After consolidation:")
    print(f"    p = {p0:.2f} MPa, q = {q0:.2f} MPa")
    print(f"    σ1 = {sig1_0:.2f} MPa, σ3 = {sig3_0:.2f} MPa")
    
    vtk_path = os.path.join(vtk_dir, 'u_000_consolidation.vtu')
    save_sol(problem.fe, sol_list[0], vtk_path)
    
    # =========================================================================
    # Stage 2: Axial Compression (Shearing)
    # =========================================================================
    print("\n" + "-"*50)
    print("Stage 2: Axial Compression (Shearing)")
    print("-"*50)
    
    # Axial displacement increments (compression = negative)
    max_axial_disp = -0.5  # mm (5% axial strain)
    n_steps = 20
    disps = np.linspace(0., max_axial_disp, n_steps + 1)[1:]  # skip 0
    
    # Storage for stress path
    p_history = [float(p0)]
    q_history = [float(q0)]
    strain_history = [0.0]
    stress_history = [float(sig1_0)]
    
    overall_start = time.time()
    
    for i, disp in enumerate(disps):
        step_start = time.time()
        
        # Update top displacement BC
        dirichlet_bc_info[-1][-1] = get_disp_top(disp)
        problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        
        # Solve
        sol_list = solver(problem, solver_options={'petsc_solver': {}})
        problem.update_stress_strain(sol_list[0])
        
        # Compute stress invariants
        p, q, sig1, sig3 = problem.compute_stress_invariants()
        
        # Record history
        axial_strain = -disp / Lz * 100  # percentage, compression positive
        p_history.append(float(p))
        q_history.append(float(q))
        strain_history.append(float(axial_strain))
        stress_history.append(float(sig1))
        
        step_time = time.time() - step_start
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Step {i+1:3d}/{n_steps}: ε_a = {axial_strain:.2f}%, "
                  f"q = {q:.1f} MPa, p = {p:.1f} MPa ({step_time:.2f}s)")
        
        # Save VTK
        vtk_path = os.path.join(vtk_dir, f'u_{i+1:03d}.vtu')
        save_sol(problem.fe, sol_list[0], vtk_path)
    
    total_time = time.time() - overall_start
    print(f"\nTotal simulation time: {total_time:.1f} seconds")
    
    # =========================================================================
    # Post-processing and Plotting
    # =========================================================================
    print("\n" + "-"*50)
    print("Post-processing")
    print("-"*50)
    
    # Convert to numpy for plotting
    p_arr = onp.array(p_history)
    q_arr = onp.array(q_history)
    strain_arr = onp.array(strain_history)
    stress_arr = onp.array(stress_history)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Stress-Strain Curve
    ax1 = axes[0]
    ax1.plot(strain_arr, q_arr, 'b-o', linewidth=2, markersize=4, label='q (deviatoric)')
    ax1.set_xlabel('Axial Strain ε_a [%]', fontsize=12)
    ax1.set_ylabel('Deviatoric Stress q [MPa]', fontsize=12)
    ax1.set_title('Triaxial Test: Stress-Strain Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: q-p Stress Path
    ax2 = axes[1]
    ax2.plot(p_arr, q_arr, 'r-o', linewidth=2, markersize=4, label='Stress Path')
    ax2.scatter(p_arr[0], q_arr[0], s=100, c='green', marker='s', zorder=5, label='Start')
    ax2.scatter(p_arr[-1], q_arr[-1], s=100, c='red', marker='*', zorder=5, label='End')
    
    # Plot DP yield surface in q-p space
    # For DP: f = sqrt(J2) + alpha*I1 - k = 0
    # In q-p space: q = M*p + d, where M = 6*alpha/(1-alpha) for triaxial compression
    # Simplified: q = 3*alpha * (3*p) - 3*k (approximation)
    p_line = onp.linspace(0, max(p_arr) * 1.2, 100)
    # For DP in triaxial: q = 6*sin(phi)/(3-sin(phi)) * p + 6*c*cos(phi)/(3-sin(phi))
    # Using alpha = sin(phi)/sqrt(3), k related to c
    M_dp = 6 * alpha / (1 - alpha)  # slope in q-p space (approximate)
    d_dp = 6 * k / (3 - alpha)       # intercept (approximate)
    q_yield = M_dp * p_line + d_dp
    ax2.plot(p_line, q_yield, 'k--', linewidth=1.5, alpha=0.7, label='DP Yield (approx)')
    
    ax2.set_xlabel('Mean Stress p [MPa]', fontsize=12)
    ax2.set_ylabel('Deviatoric Stress q [MPa]', fontsize=12)
    ax2.set_title(f'q-p Stress Path (σ₃ = {confining_pressure} MPa)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'triaxial_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")
    
    # Save data to CSV
    data = onp.column_stack((strain_arr, p_arr, q_arr, stress_arr))
    csv_path = os.path.join(results_dir, 'triaxial_data.csv')
    onp.savetxt(csv_path, data, delimiter=',', 
                header='Axial_Strain_pct,Mean_Stress_p_MPa,Deviatoric_Stress_q_MPa,Axial_Stress_MPa',
                comments='')
    print(f"  Data saved: {csv_path}")
    
    print("\n" + "="*70)
    print("  SIMULATION COMPLETE")
    print("="*70)
    
    return p_arr, q_arr, strain_arr


if __name__ == "__main__":
    run_triaxial_simulation()
