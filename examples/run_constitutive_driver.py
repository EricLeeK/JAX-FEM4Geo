"""
Verification of Drucker-Prager Plasticity Implementation
Comparison between 'Analytical' (Constitutive Driver) and FEM (implied).

This script implements the return mapping algorithm for Drucker-Prager
plasticity on a single material point under UNIAXIAL STRESS conditions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import root
import os

# =============================================================================
# Material Parameters (Must match FEM)
# =============================================================================
E = 70.0e3      # MPa
nu = 0.3
alpha = 0.3
k = 250.0
a = 0.01 * k    # Regularization

# Lame parameters
mu = E / (2. * (1. + nu))
lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))

# =============================================================================
# Constitutive Model Functions (Numpy version of JAX logic)
# =============================================================================

def safe_sqrt(x):
    return np.sqrt(x) if x > 0 else 0.

def elastic_stress(epsilon):
    return lmbda * np.trace(epsilon) * np.eye(3) + 2. * mu * epsilon

def stress_return_map_point(epsilon_inc, sigma_old):
    """
    Computes new stress given strain increment and old stress.
    Returns: sigma_new, internal_vars (not used here)
    """
    sigma_trial = elastic_stress(epsilon_inc) + sigma_old

    I1 = np.trace(sigma_trial)
    s_dev = sigma_trial - (I1 / 3.) * np.eye(3)
    J2 = 0.5 * np.sum(s_dev * s_dev)

    sqrt_J2_reg = np.sqrt(J2 + a * a)
    f_yield = sqrt_J2_reg + alpha * I1 - k

    if f_yield <= 0:
        return sigma_trial
    
    # Plastic correction
    n_dev = s_dev / (2. * sqrt_J2_reg) if sqrt_J2_reg > 1e-12 else np.zeros((3,3))
    delta_lambda = f_yield / (1. + 3. * alpha * alpha)
    
    sigma = sigma_trial - delta_lambda * (n_dev + alpha * np.eye(3))
    
    # Apex return
    if np.trace(sigma) > k / alpha:
        sigma = (k / (3. * alpha)) * np.eye(3)
        
    return sigma

# =============================================================================
# Simulation Loop (Uniaxial Stress Driver)
# =============================================================================

def simulate_uniaxial_stress():
    # Loading schedule (same as FEM)
    disps = np.hstack((np.linspace(0., -0.15, 16), np.linspace(-0.14, 0., 15)))
    Lz = 10.
    target_strains_zz = disps / Lz
    
    # Storage
    stresses_zz = []
    strains_zz = []
    
    # State variables
    sigma_current = np.zeros((3, 3))
    epsilon_current = np.zeros((3, 3))
    
    # Initial guess for lateral strain increment
    lat_strain_inc_guess = 0.0

    print(f"Starting verification simulation...")
    print(f"Total steps: {len(target_strains_zz)}")

    for i, eps_zz_target in enumerate(target_strains_zz):
        if i == 0:
            stresses_zz.append(0.0)
            strains_zz.append(0.0)
            continue
            
        # Calculate target strain increment in ZZ
        d_eps_zz = eps_zz_target - epsilon_current[2, 2]
        
        def residual(d_eps_lat):
            d_eps_lat_val = d_eps_lat[0] if hasattr(d_eps_lat, '__len__') else d_eps_lat
            # Construct increment tensor
            inc_strain = np.diag([d_eps_lat_val, d_eps_lat_val, d_eps_zz])
            sig = stress_return_map_point(inc_strain, sigma_current)
            return sig[0, 0] # We want sigma_xx = 0
            
        # Solve for lateral strain increment
        sol = root(residual, lat_strain_inc_guess, method='hybr')
        
        if not sol.success:
            print(f"Warning: Convergence failed at step {i}")
            
        d_eps_lat_sol = sol.x[0]
        lat_strain_inc_guess = d_eps_lat_sol
        
        # Update state with converged solution
        final_inc_strain = np.diag([d_eps_lat_sol, d_eps_lat_sol, d_eps_zz])
        sigma_current = stress_return_map_point(final_inc_strain, sigma_current)
        epsilon_current = epsilon_current + final_inc_strain
        
        stresses_zz.append(sigma_current[2, 2])
        strains_zz.append(epsilon_current[2, 2])

    # Path handling
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, 'results', '2025-12-22_verification')
    os.makedirs(results_dir, exist_ok=True)

    # Save to CSV
    data = np.column_stack((strains_zz, stresses_zz))
    csv_path = os.path.join(results_dir, "verification_data.csv")
    np.savetxt(csv_path, data, delimiter=",", header="Strain_ZZ,Stress_ZZ", comments="")
    print(f"Data saved to {csv_path}")

    return np.array(strains_zz), np.array(stresses_zz)

if __name__ == "__main__":
    strains, stresses = simulate_uniaxial_stress()
