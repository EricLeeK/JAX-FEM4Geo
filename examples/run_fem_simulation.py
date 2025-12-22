"""
Run Drucker-Prager FEM Simulation

This script sets up the boundary conditions and runs the finite element simulation
using the DruckerPragerPlasticity model defined in src.models.
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

from src.models.drucker_prager import DruckerPragerPlasticity
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh

def run_simulation():
    # Parameters
    E = 70.0e3
    nu = 0.3
    alpha = 0.3
    k = 250.0
    a = 2.5 # 0.01 * k

    # Mesh
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    
    # Output directory
    results_dir = os.path.join(project_root, 'results', '2025-12-22_verification')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'vtk'), exist_ok=True)

    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 5, 5, 5

    meshio_mesh = box_mesh_gmsh(
        Nx=Nx, Ny=Ny, Nz=Nz,
        domain_x=Lx, domain_y=Ly, domain_z=Lz,
        data_dir=results_dir, ele_type=ele_type,
    )
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # BCs
    def top(point): return np.isclose(point[2], Lz, atol=1e-5)
    def bottom(point): return np.isclose(point[2], 0., atol=1e-5)
    def dirichlet_val_bottom(point): return 0.
    def get_dirichlet_top(disp):
        def val_fn(point): return disp
        return val_fn

    disps = np.hstack((np.linspace(0., -0.15, 16), np.linspace(-0.14, 0., 15)))

    location_fns = [bottom, top]
    value_fns = [dirichlet_val_bottom, get_dirichlet_top(disps[0])]
    vecs = [2, 2]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    # Initialize Problem
    # Note: We pass material params here
    problem = DruckerPragerPlasticity(mesh, vec=3, dim=3, 
                                      dirichlet_bc_info=dirichlet_bc_info,
                                      E=E, nu=nu, alpha=alpha, k=k, a=a)
    
    avg_stresses = []
    
    print("\n" + "="*60)
    print("Running Drucker-Prager Simulation (Refactored)")
    print("="*60)

    overall_start = time.time()

    for i, disp in enumerate(disps):
        if i % 5 == 0:
            print(f"Step {i+1}/{len(disps)}, displacement = {disp:.4f} mm")
        
        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
        problem.fe.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        
        sol_list = solver(problem, solver_options={'petsc_solver': {}}) 
        problem.update_stress_strain(sol_list[0])
        
        avg_stress = problem.compute_avg_stress()
        avg_stresses.append(avg_stress)
        
        vtk_path = os.path.join(results_dir, f'vtk/u_{i:03d}.vtu')
        save_sol(problem.fe, sol_list[0], vtk_path)

    avg_stresses = np.array(avg_stresses)
    
    # Save Data
    strains_zz = disps / Lz
    stresses_zz = avg_stresses[:, 2, 2]
    data = np.column_stack((strains_zz, stresses_zz))
    csv_path = os.path.join(results_dir, "fem_data.csv")
    onp.savetxt(csv_path, data, delimiter=",", header="Strain_ZZ,Stress_ZZ", comments="")
    print(f"Data saved to {csv_path}")

    # Plot
    fig = plt.figure(figsize=(10, 8))
    plt.plot(strains_zz * 100, stresses_zz, 
             color='blue', marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.xlabel('Strain (z-z) [%]')
    plt.ylabel('Stress (z-z) [MPa]')
    plt.title('Drucker-Prager FEM Simulation')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(results_dir, 'stress_strain_curve.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_simulation()
