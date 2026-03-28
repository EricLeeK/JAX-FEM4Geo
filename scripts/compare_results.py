"""
Post-processing script to compare FEM vs Constitutive Driver results.

Performs:
1. Slope comparison (elastic modulus, plastic tangent)
2. Point-by-point stress comparison via interpolation
3. PASS/FAIL verdict based on configurable tolerance
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Tolerance for PASS/FAIL verdict (MPa)
STRESS_TOL = 1.0  # max allowable point-wise stress difference
SLOPE_REL_TOL = 0.05  # 5% relative tolerance on slopes


def calculate_slope(strain, stress):
    """Calculate elastic slope using linear regression on the initial loading part."""
    peak_idx = np.argmax(np.abs(strain))
    loading_strain = strain[:peak_idx]
    loading_stress = stress[:peak_idx]

    if len(loading_strain) > 5:
        s_strain = loading_strain[1:6]
        s_stress = loading_stress[1:6]
        slope, _ = np.polyfit(s_strain, s_stress, 1)
        return slope
    return 0.0


def pointwise_comparison(ver_strain, ver_stress, fem_strain, fem_stress):
    """Compute point-wise stress error between driver and FEM.

    If strain grids match exactly, uses direct subtraction.
    Otherwise, splits into loading/unloading branches and interpolates each.
    """
    if len(ver_strain) == len(fem_strain) and np.allclose(ver_strain, fem_strain):
        # Grids are identical — direct comparison
        diff = np.abs(ver_stress - fem_stress)
        return diff, fem_stress

    # Non-matching grids: split at peak strain, interpolate each branch
    peak_v = np.argmax(np.abs(ver_strain))
    peak_f = np.argmax(np.abs(fem_strain))

    # Loading branch (0 -> peak)
    load_interp = np.interp(
        ver_strain[:peak_v+1], fem_strain[:peak_f+1], fem_stress[:peak_f+1])
    # Unloading branch (peak -> end), reverse for monotonicity
    unload_v = ver_strain[peak_v:][::-1]
    unload_f_strain = fem_strain[peak_f:][::-1]
    unload_f_stress = fem_stress[peak_f:][::-1]
    unload_interp = np.interp(unload_v, unload_f_strain, unload_f_stress)[::-1]

    fem_interp = np.concatenate([load_interp, unload_interp[1:]])
    ver_combined = np.concatenate([ver_stress[:peak_v+1], ver_stress[peak_v+1:]])
    diff = np.abs(ver_combined - fem_interp)
    return diff, fem_interp


def compare_results(results_dir=None):
    print("Comparing Verification (Driver) vs FEM Results...")
    print("=" * 60)

    if results_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        results_dir = os.path.join(project_root, 'results', '2025-12-22_verification')

    fem_csv = os.path.join(results_dir, "fem_data.csv")
    ver_csv = os.path.join(results_dir, "verification_data.csv")

    try:
        ver_data = np.loadtxt(ver_csv, delimiter=",", skiprows=1)
        fem_data = np.loadtxt(fem_csv, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"Error reading CSV files from {results_dir}: {e}")
        return False

    ver_strain = ver_data[:, 0]
    ver_stress = ver_data[:, 1]
    fem_strain = fem_data[:, 0]
    fem_stress = fem_data[:, 1]

    all_pass = True

    # --- 1. Slope comparison ---
    print("\n[1] Slope Comparison")
    print("-" * 40)

    ver_slope = calculate_slope(ver_strain, ver_stress)
    fem_slope = calculate_slope(fem_strain, fem_stress)
    slope_diff = abs(ver_slope - fem_slope)
    slope_rel = slope_diff / abs(ver_slope) if ver_slope != 0 else float('inf')
    slope_pass = slope_rel < SLOPE_REL_TOL

    print(f"  Elastic Modulus:")
    print(f"    Driver:     {ver_slope:.2f} MPa")
    print(f"    FEM:        {fem_slope:.2f} MPa")
    print(f"    Rel. diff:  {slope_rel:.2e}  {'PASS' if slope_pass else 'FAIL'}")
    if not slope_pass:
        all_pass = False

    peak_idx_v = np.argmax(np.abs(ver_strain))
    peak_idx_f = np.argmax(np.abs(fem_strain))

    v_plas_strain = ver_strain[peak_idx_v-3:peak_idx_v]
    v_plas_stress = ver_stress[peak_idx_v-3:peak_idx_v]
    f_plas_strain = fem_strain[peak_idx_f-3:peak_idx_f]
    f_plas_stress = fem_stress[peak_idx_f-3:peak_idx_f]

    if len(v_plas_strain) > 1 and len(f_plas_strain) > 1:
        ver_p_slope, _ = np.polyfit(v_plas_strain, v_plas_stress, 1)
        fem_p_slope, _ = np.polyfit(f_plas_strain, f_plas_stress, 1)
        p_diff = abs(ver_p_slope - fem_p_slope)
        p_rel = p_diff / abs(ver_p_slope) if ver_p_slope != 0 else float('inf')
        p_pass = p_rel < SLOPE_REL_TOL

        print(f"  Plastic Tangent Slope:")
        print(f"    Driver:     {ver_p_slope:.2f} MPa")
        print(f"    FEM:        {fem_p_slope:.2f} MPa")
        print(f"    Rel. diff:  {p_rel:.2e}  {'PASS' if p_pass else 'FAIL'}")
        if not p_pass:
            all_pass = False

    # --- 2. Point-by-point comparison ---
    print(f"\n[2] Point-by-Point Comparison")
    print("-" * 40)

    diff, fem_interp = pointwise_comparison(ver_strain, ver_stress, fem_strain, fem_stress)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    max_idx = np.argmax(diff)
    pw_pass = max_diff < STRESS_TOL

    print(f"  Points compared:  {len(diff)}")
    print(f"  Max stress diff:  {max_diff:.6f} MPa  (at strain = {ver_strain[max_idx]:.6f})")
    print(f"  Mean stress diff: {mean_diff:.6f} MPa")
    print(f"  Tolerance:        {STRESS_TOL} MPa")
    print(f"  Verdict:          {'PASS' if pw_pass else 'FAIL'}")
    if not pw_pass:
        all_pass = False

    # --- 3. Overall verdict ---
    print(f"\n{'=' * 60}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    # --- 4. Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stress-strain overlay
    ax1 = axes[0]
    ax1.plot(ver_strain*100, ver_stress, 'r-', linewidth=2, label='Driver')
    ax1.plot(fem_strain*100, fem_stress, 'b--', linewidth=2, marker='o', markersize=3, label='FEM')
    ax1.set_xlabel('Strain ZZ [%]')
    ax1.set_ylabel('Stress ZZ [MPa]')
    ax1.set_title('Stress-Strain Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: point-wise error
    ax2 = axes[1]
    ax2.plot(ver_strain*100, diff, 'k-', linewidth=1.5)
    ax2.axhline(y=STRESS_TOL, color='r', linestyle='--', alpha=0.7, label=f'Tolerance ({STRESS_TOL} MPa)')
    ax2.set_xlabel('Strain ZZ [%]')
    ax2.set_ylabel('|Stress Difference| [MPa]')
    ax2.set_title('Point-wise Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "comparison_plot.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out_path}")

    return all_pass


if __name__ == "__main__":
    compare_results()
