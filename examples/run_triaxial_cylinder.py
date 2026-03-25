"""
Triaxial Compression Test Simulation with Cylindrical Mesh

Standard triaxial compression test:
1. Stage 1 (Consolidation): Apply confining pressure σ3 on all faces
2. Stage 2 (Shearing): Keep confining pressure, apply axial compression

Key features:
- Cylindrical mesh (H:D = 2:1) for realistic triaxial conditions
- Drucker-Prager plasticity model
- Incremental loading with history tracking

Author: Li Shiyao (Assisted by Claude)
Date: 2026-02-04
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
from jax_fem.generate_mesh import cylinder_mesh_gmsh, get_meshio_cell_type, Mesh


class TriaxialCylinderDP(Problem):
    """
    Drucker-Prager model for triaxial compression test with cylindrical specimen.
    
    Features:
    - Neumann BC for confining pressure on lateral (curved) surface
    - Dirichlet BC for fixed bottom and controlled top displacement
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
        
        For cylindrical surface: traction = p * n_inward
        where n_inward = -[x/r, y/r, 0] (pointing toward axis)
        """
        confining_p = self.confining_pressure
        
        def lateral_traction(u, point, *args):
            """
            Traction on curved lateral surface of cylinder.
            Normal vector points radially outward, so for inward pressure,
            we apply traction in the -r direction.
            """
            x, y = point[0], point[1]
            r = np.sqrt(x**2 + y**2)
            # Avoid division by zero at center (shouldn't happen on surface)
            r_safe = np.where(r > 1e-10, r, 1.0)
            # Inward normal components (negative of outward)
            nx = -x / r_safe
            ny = -y / r_safe
            # Traction = pressure * inward normal
            return confining_p * np.array([nx, ny, 0.])
        
        # Return list matching location_fns order: [lateral_surface]
        return [lateral_traction]

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
        
        # For triaxial: σ1 = σ_zz (axial), σ2 = σ3 = σ_rr (confining)
        sigma_1 = avg_sigma[2, 2]  # axial stress (z-direction)
        sigma_3 = (avg_sigma[0, 0] + avg_sigma[1, 1]) / 2.0  # confining stress
        
        # Geomechanics convention: compression positive
        p = -(sigma_1 + 2 * sigma_3) / 3.0  # mean stress
        q = -(sigma_1 - sigma_3)  # deviatoric stress
        
        return p, q, -sigma_1, -sigma_3

    def compute_volumetric_strain(self, sol):
        """Compute average volumetric strain."""
        u_grads = self.fe.sol_to_grad(sol)
        # Volumetric strain = trace(epsilon)
        def vol_strain(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return np.trace(eps)
        vmap_vol_strain = jax.vmap(jax.vmap(vol_strain))
        vol_strains = vmap_vol_strain(u_grads)
        # Weight by integration weights
        avg_vol_strain = np.sum(vol_strains.flatten() * self.fe.JxW.flatten()) / np.sum(self.fe.JxW)
        return avg_vol_strain


def run_triaxial_cylinder():
    """
    Run a triaxial compression test simulation with cylindrical specimen.
    
    Test setup:
    - Cylindrical specimen (H:D = 2:1, standard triaxial ratio)
    - Stage 1: Isotropic consolidation under confining pressure
    - Stage 2: Axial compression with constant confining pressure
    """
    
    print("\n" + "="*70)
    print("  TRIAXIAL COMPRESSION TEST - Cylindrical Specimen")
    print("  Drucker-Prager Plasticity Model")
    print("="*70)
    
    # =========================================================================
    # Material Parameters
    # =========================================================================
    E = 70.0e3      # Young's modulus [MPa]
    nu = 0.3        # Poisson's ratio
    alpha = 0.3     # DP friction parameter
    k = 250.0       # DP cohesion parameter [MPa]
    a = 2.5         # Apex regularization
    
    # Confining pressure (σ3)
    confining_pressure = 50.0  # [MPa]
    
    print(f"\nMaterial Parameters:")
    print(f"  E = {E} MPa, ν = {nu}")
    print(f"  α = {alpha}, k = {k} MPa")
    print(f"  Confining Pressure σ3 = {confining_pressure} MPa")
    
    # =========================================================================
    # Cylindrical Mesh Setup (H:D = 2:1)
    # =========================================================================
    R = 2.5   # Radius [mm]
    H = 10.0  # Height [mm] (H:D = 10:5 = 2:1)
    
    # Output directory
    results_dir = os.path.join(project_root, 'results', 'triaxial_cylinder')
    os.makedirs(results_dir, exist_ok=True)
    vtk_dir = os.path.join(results_dir, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    
    print(f"\nCylindrical Specimen: R = {R} mm, H = {H} mm (H:D = 2:1)")
    print(f"Generating mesh...")
    
    # Generate cylindrical mesh
    meshio_mesh = cylinder_mesh_gmsh(
        data_dir=results_dir,
        R=R, H=H,
        circle_mesh=4,   # mesh density on circle
        hight_mesh=10,   # mesh density along height
        rect_ratio=0.4
    )
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'], ele_type='HEX8')
    
    print(f"  Nodes: {mesh.points.shape[0]}, Elements: {mesh.cells.shape[0]}")
    
    # =========================================================================
    # Boundary Conditions
    # =========================================================================
    tol = 1e-5
    
    # Dirichlet BCs
    def bottom(point):
        """Bottom face: z = 0"""
        return np.isclose(point[2], 0., atol=tol)
    
    def top(point):
        """Top face: z = H"""
        return np.isclose(point[2], H, atol=tol)
    
    def center_bottom(point):
        """Center point at bottom - fix x, y to prevent rigid body rotation"""
        return np.logical_and(
            np.logical_and(np.abs(point[0]) < tol, np.abs(point[1]) < tol),
            np.isclose(point[2], 0., atol=tol)
        )
    
    def zero_val(point):
        return 0.
    
    def get_disp_top(disp):
        def val_fn(point):
            return disp
        return val_fn
    
    # Neumann BC: Lateral surface (curved surface of cylinder)
    def lateral_surface(point):
        """Lateral curved surface: r ≈ R"""
        r = np.sqrt(point[0]**2 + point[1]**2)
        z = point[2]
        on_surface = np.isclose(r, R, atol=tol * 10)  # slightly larger tolerance for curved surface
        not_top_bottom = np.logical_and(z > tol, z < H - tol)
        return np.logical_and(on_surface, not_top_bottom)
    
    # Dirichlet: 
    # - Bottom: fix uz=0 (z displacement)
    # - Center bottom: fix ux=uy=0 (prevent rigid body motion)
    # - Top: uz = prescribed displacement
    dirichlet_bc_info = [
        [bottom, center_bottom, center_bottom, top],  # location functions
        [2, 0, 1, 2],                                  # DOF indices
        [zero_val, zero_val, zero_val, get_disp_top(0.)]  # values
    ]
    
    # Neumann: confining pressure on lateral surface
    location_fns = [lateral_surface]
    
    # =========================================================================
    # Problem Setup
    # =========================================================================
    problem = TriaxialCylinderDP(
        mesh, vec=3, dim=3, ele_type='HEX8',
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        E=E, nu=nu, alpha=alpha, k=k, a=a,
        confining_pressure=confining_pressure
    )
    
    solver_options = {
        'petsc_solver': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        }
    }
    
    # =========================================================================
    # Stage 1: Isotropic Consolidation
    # =========================================================================
    print("\n" + "-"*50)
    print("Stage 1: Isotropic Consolidation")
    print("-"*50)
    
    sol_list = solver(problem, solver_options=solver_options)
    problem.update_stress_strain(sol_list[0])
    
    p0, q0, sig1_0, sig3_0 = problem.compute_stress_invariants()
    vol_strain_0 = problem.compute_volumetric_strain(sol_list[0])
    
    print(f"  After consolidation:")
    print(f"    p = {p0:.2f} MPa, q = {q0:.2f} MPa")
    print(f"    σ1 = {sig1_0:.2f} MPa, σ3 = {sig3_0:.2f} MPa")
    print(f"    εv = {vol_strain_0*100:.4f}%")
    
    vtk_path = os.path.join(vtk_dir, 'u_000_consolidation.vtu')
    save_sol(problem.fe, sol_list[0], vtk_path)
    
    # =========================================================================
    # Stage 2: Axial Compression (Shearing)
    # =========================================================================
    print("\n" + "-"*50)
    print("Stage 2: Axial Compression (Shearing)")
    print("-"*50)
    
    # Axial displacement increments (compression = negative)
    max_axial_strain = 0.05  # 5% axial strain
    max_axial_disp = -max_axial_strain * H  # negative for compression
    n_steps = 20
    disps = np.linspace(0., max_axial_disp, n_steps + 1)[1:]  # skip 0
    
    # Storage for results
    p_history = [float(p0)]
    q_history = [float(q0)]
    axial_strain_history = [0.0]
    vol_strain_history = [float(vol_strain_0) * 100]
    
    overall_start = time.time()
    
    for i, disp in enumerate(disps):
        step_start = time.time()
        
        # Update top displacement BC
        dirichlet_bc_info[-1][-1] = get_disp_top(disp)
        problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        
        # Solve
        sol_list = solver(problem, solver_options=solver_options)
        problem.update_stress_strain(sol_list[0])
        
        # Compute results
        p, q, sig1, sig3 = problem.compute_stress_invariants()
        vol_strain = problem.compute_volumetric_strain(sol_list[0])
        
        # Record history
        axial_strain = -disp / H * 100  # percentage, compression positive
        p_history.append(float(p))
        q_history.append(float(q))
        axial_strain_history.append(float(axial_strain))
        vol_strain_history.append(float(vol_strain) * 100)
        
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
    strain_arr = onp.array(axial_strain_history)
    vol_strain_arr = onp.array(vol_strain_history)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Stress-Strain Curve (q vs ε1)
    ax1 = axes[0]
    ax1.plot(strain_arr, q_arr, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Axial Strain ε₁ [%]', fontsize=12)
    ax1.set_ylabel('Deviatoric Stress q [MPa]', fontsize=12)
    ax1.set_title('Stress-Strain Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: q-p Stress Path
    ax2 = axes[1]
    ax2.plot(p_arr, q_arr, 'r-o', linewidth=2, markersize=4, label='Stress Path')
    ax2.scatter(p_arr[0], q_arr[0], s=100, c='green', marker='s', zorder=5, label='Start')
    ax2.scatter(p_arr[-1], q_arr[-1], s=100, c='red', marker='*', zorder=5, label='End')
    
    # Plot DP yield surface approximation
    p_line = onp.linspace(0, max(p_arr) * 1.2, 100)
    M_dp = 6 * alpha / (1 - alpha)
    d_dp = 6 * k / (3 - alpha)
    q_yield = M_dp * p_line + d_dp
    ax2.plot(p_line, q_yield, 'k--', linewidth=1.5, alpha=0.7, label='DP Yield')
    
    ax2.set_xlabel('Mean Stress p [MPa]', fontsize=12)
    ax2.set_ylabel('Deviatoric Stress q [MPa]', fontsize=12)
    ax2.set_title(f'q-p Stress Path (σ₃ = {confining_pressure} MPa)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Volumetric Strain
    ax3 = axes[2]
    ax3.plot(strain_arr, vol_strain_arr, 'g-o', linewidth=2, markersize=4)
    ax3.set_xlabel('Axial Strain ε₁ [%]', fontsize=12)
    ax3.set_ylabel('Volumetric Strain εᵥ [%]', fontsize=12)
    ax3.set_title('Volume Change', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'triaxial_cylinder_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")
    
    # Save data to CSV
    data = onp.column_stack((strain_arr, p_arr, q_arr, vol_strain_arr))
    csv_path = os.path.join(results_dir, 'triaxial_cylinder_data.csv')
    onp.savetxt(csv_path, data, delimiter=',', 
                header='Axial_Strain_pct,Mean_Stress_p_MPa,Deviatoric_Stress_q_MPa,Vol_Strain_pct',
                comments='')
    print(f"  Data saved: {csv_path}")
    
    print("\n" + "="*70)
    print("  SIMULATION COMPLETE")
    print("="*70)
    
    return {
        'p': p_arr,
        'q': q_arr,
        'axial_strain': strain_arr,
        'vol_strain': vol_strain_arr,
        'results_dir': results_dir
    }


if __name__ == "__main__":
    run_triaxial_cylinder()
