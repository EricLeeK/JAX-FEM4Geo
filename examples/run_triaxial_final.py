"""
Corrected Standard Triaxial Test - Physics Hardened Version
Author: Li Shiyao (Assisted by Alma)
Date: 2026-01-08

Final Fix: Use JAX-compatible logical operations for BCs.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
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
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh

class StandardTriaxialProblem(Problem):
    def __init__(self, mesh, vec, dim, dirichlet_bc_info, location_fns, 
                 E, nu, alpha, k, a, confining_p):
        self.E, self.nu, self.alpha, self.k, self.a = E, nu, alpha, k, a
        self.confining_p = confining_p
        self.apply_top_pressure = True
        super().__init__(mesh, vec=vec, dim=dim, 
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)

    def custom_init(self):
        self.fe = self.fes[0]
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def get_tensor_map(self):
        E, nu, alpha, k, a = self.E, self.nu, self.alpha, self.k, self.a
        mu = E / (2. * (1. + nu))
        lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
        bulk_k = lmbda + 2./3.*mu
        
        def stress_return_map(u_grad, sigma_old, epsilon_old):
            eps_crt = 0.5 * (u_grad + u_grad.T)
            eps_inc = eps_crt - epsilon_old
            sig_trial = lmbda * np.trace(eps_inc) * np.eye(self.dim) + 2. * mu * eps_inc + sigma_old
            I1 = np.trace(sig_trial)
            s_dev = sig_trial - (I1 / 3.) * np.eye(self.dim)
            J2 = 0.5 * np.sum(s_dev * s_dev)
            f_yield = np.sqrt(J2 + a*a) + alpha * I1 - k
            f_plus = np.where(f_yield > 0., f_yield, 0.)
            # Associated DP Denominator
            denom = mu + 9. * bulk_k * alpha * alpha
            d_lam = f_plus / denom
            sigma = sig_trial - d_lam * (mu * s_dev / np.sqrt(J2 + a*a) + 3. * bulk_k * alpha * np.eye(self.dim))
            return sigma
        return stress_return_map

    def get_surface_maps(self):
        p = self.confining_p
        def left_t(u, x, *args): return np.array([p, 0., 0.])
        def right_t(u, x, *args): return np.array([-p, 0., 0.])
        def front_t(u, x, *args): return np.array([0., p, 0.])
        def back_t(u, x, *args): return np.array([0., -p, 0.])
        def top_t(u, x, *args): 
            val = p if self.apply_top_pressure else 0.
            return np.array([0., 0., -val])
        return [left_t, right_t, front_t, back_t, top_t]

    def update_stress_strain(self, sol):
        u_grads = self.fe.sol_to_grad(sol)
        rm = self.get_tensor_map()
        vmap_rm = jax.vmap(jax.vmap(rm))
        vmap_strain = jax.vmap(jax.vmap(lambda g: 0.5*(g+g.T)))
        self.sigmas_old = vmap_rm(u_grads, self.sigmas_old, self.epsilons_old)
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def compute_avg_stress(self):
        sigma = np.sum(self.sigmas_old.reshape(-1, self.fe.vec, self.dim) * self.fe.JxW.reshape(-1)[:, None, None], axis=0)
        return sigma / np.sum(self.fe.JxW)

def run():
    sigma3 = 50.0
    results_dir = os.path.join(project_root, 'results', 'triaxial_final')
    os.makedirs(results_dir, exist_ok=True)
    meshio_mesh = box_mesh_gmsh(Nx=1, Ny=1, Nz=1, domain_x=10, domain_y=10, domain_z=10, data_dir=results_dir, ele_type='HEX8')
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[get_meshio_cell_type('HEX8')])

    def bottom(p): return np.isclose(p[2], 0.)
    def top_loc(p): return np.isclose(p[2], 10.)
    def left_loc(p): return np.isclose(p[0], 0.)
    def right_loc(p): return np.isclose(p[0], 10.)
    def front_loc(p): return np.isclose(p[1], 0.)
    def back_loc(p): return np.isclose(p[1], 10.)
    
    # Corrected JAX corner function
    def corner(p): 
        return np.logical_and(np.logical_and(np.isclose(p[0], 0.), np.isclose(p[1], 0.)), np.isclose(p[2], 0.))
    
    dir_iso = [[bottom, corner, corner], [2, 0, 1], [lambda p:0., lambda p:0., lambda p:0.]]
    loc_fns = [left_loc, right_loc, front_loc, back_loc, top_loc]
    
    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}}
    problem = StandardTriaxialProblem(mesh, 3, 3, dir_iso, loc_fns, 70e3, 0.3, 0.3, 250.0, 2.5, sigma3)
    
    p_h, q_h = [0.], [0.]
    print("\n--- STAGE 1: Isotropic Consolidation ---")
    sol = solver(problem, solver_options=solver_options)
    problem.update_stress_strain(sol[0])
    avg_s = problem.compute_avg_stress()
    s1, s3 = -avg_s[2,2], -(avg_s[0,0]+avg_s[1,1])/2
    p_h.append(float((s1+2*s3)/3)); q_h.append(float(s1-s3))
    print(f"End Iso: p={p_h[-1]:.1f}, q={q_h[-1]:.1f}")

    print("\n--- STAGE 2: Shearing ---")
    problem.apply_top_pressure = False
    for i in range(15):
        d = (i+1)/15 * (-0.1)
        def get_val(v): return lambda p: v
        new_dir = [[bottom, top_loc, corner, corner], [2, 2, 0, 1], [get_val(0.), get_val(d), get_val(0.), get_val(0.)]]
        problem.fe.update_Dirichlet_boundary_conditions(new_dir)
        sol = solver(problem, solver_options=solver_options)
        problem.update_stress_strain(sol[0])
        avg_s = problem.compute_avg_stress()
        s1, s3 = -avg_s[2,2], -(avg_s[0,0]+avg_s[1,1])/2
        p_h.append(float((s1+2*s3)/3)); q_h.append(float(s1-s3))
        if (i+1)%5==0: print(f"Step {i+1}: p={p_h[-1]:.1f}, q={q_h[-1]:.1f}")

    plt.figure(figsize=(8,6))
    plt.plot(p_h, q_h, 'r-o', label='Simulation Path')
    p_theory = onp.linspace(sigma3, max(p_h)*1.1, 10)
    plt.plot(p_theory, 3*(p_theory - sigma3), 'b--', label='Theory (Slope=3)')
    plt.xlabel('p'); plt.ylabel('q'); plt.title('Corrected Standard Triaxial Path'); plt.legend(); plt.grid()
    plt.savefig(os.path.join(results_dir, 'final_triaxial_path.png'))
    print(f"Done. Graph in {results_dir}")

if __name__ == "__main__":
    run()
