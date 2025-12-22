"""
Post-processing script to compare FEM vs Constitutive Driver results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

def calculate_slope(strain, stress):
    # Calculate slope using linear regression on the initial elastic part
    peak_idx = np.argmax(np.abs(strain))
    loading_strain = strain[:peak_idx]
    loading_stress = stress[:peak_idx]
    
    # Take first 5 points for elastic modulus (skipping 0,0)
    if len(loading_strain) > 5:
        s_strain = loading_strain[1:6]
        s_stress = loading_stress[1:6]
        slope, _ = np.polyfit(s_strain, s_stress, 1)
        return slope
    return 0.0

def compare_results():
    print("Comparing Verification (Driver) vs FEM Results...")
    
    # Path setup
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, 'results', '2025-12-22_verification')
    
    fem_csv = os.path.join(results_dir, "fem_data.csv")
    ver_csv = os.path.join(results_dir, "verification_data.csv")
    
    try:
        ver_data = np.loadtxt(ver_csv, delimiter=",", skiprows=1)
        fem_data = np.loadtxt(fem_csv, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"Error reading CSV files from {results_dir}: {e}")
        return

    ver_strain = ver_data[:, 0]
    ver_stress = ver_data[:, 1]
    
    fem_strain = fem_data[:, 0]
    fem_stress = fem_data[:, 1]
    
    # Calculate Elastic Slopes
    ver_slope = calculate_slope(ver_strain, ver_stress)
    fem_slope = calculate_slope(fem_strain, fem_stress)
    
    print(f"\nElastic Modulus (Slope of initial loading):")
    print(f"  Verification Driver: {ver_slope:.2f} MPa")
    print(f"  FEM Simulation:      {fem_slope:.2f} MPa")
    print(f"  Difference:          {abs(ver_slope - fem_slope):.2f} MPa")
    
    # Calculate Hardening/Plastic Slope
    peak_idx_v = np.argmax(np.abs(ver_strain))
    peak_idx_f = np.argmax(np.abs(fem_strain))
    
    v_plas_strain = ver_strain[peak_idx_v-3:peak_idx_v]
    v_plas_stress = ver_stress[peak_idx_v-3:peak_idx_v]
    f_plas_strain = fem_strain[peak_idx_f-3:peak_idx_f]
    f_plas_stress = fem_stress[peak_idx_f-3:peak_idx_f]
    
    if len(v_plas_strain) > 1 and len(f_plas_strain) > 1:
        ver_p_slope, _ = np.polyfit(v_plas_strain, v_plas_stress, 1)
        fem_p_slope, _ = np.polyfit(f_plas_strain, f_plas_stress, 1)
        
        print(f"\nPlastic/Tangent Slope (Near peak loading):")
        print(f"  Verification Driver: {ver_p_slope:.2f} MPa")
        print(f"  FEM Simulation:      {fem_p_slope:.2f} MPa")
        print(f"  Difference:          {abs(ver_p_slope - fem_p_slope):.2f} MPa")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(ver_strain*100, ver_stress, 'r-', linewidth=2, label='Verification (Driver)')
    plt.plot(fem_strain*100, fem_stress, 'b--', linewidth=2, marker='o', markersize=4, label='FEM (JAX)')
    
    plt.xlabel('Strain ZZ [%]')
    plt.ylabel('Stress ZZ [MPa]')
    plt.title('Validation: Constitutive Driver vs FEM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(results_dir, "comparison_plot.png")
    plt.savefig(out_path)
    print(f"\nComparison plot saved to {out_path}")

if __name__ == "__main__":
    compare_results()
