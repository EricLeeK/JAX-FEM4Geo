#!/usr/bin/env python
"""
E2: Noise Robustness — Parameter Inversion under Observation Noise

Tests how the two-stage σ_zz inversion degrades as Gaussian noise is added
to synthetic observations.

Noise model: σ_zz_noisy = σ_zz_clean * (1 + rel_std * N(0,1))

Sweep:  rel_std ∈ [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
Trials: 5 per noise level (different random seeds)

Output:
  - Parameter error (E, k) vs noise level
  - Pass/fail at 1% threshold
"""

import jax
import jax.numpy as np
import numpy as onp
import os, sys, json, time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from common import (
    InversionDruckerPrager, create_mesh_and_bc,
    volume_avg_sigma_zz, generate_observation,
    two_stage_inversion, RESULTS_DIR,
)

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(RESULTS_DIR, 'e2_noise_robustness')
os.makedirs(OUT_DIR, exist_ok=True)

E_TRUE, K_TRUE = 70000.0, 50.0
E_INIT, K_INIT = 55000.0, 45.0
DISP_ELASTIC = -0.015
DISP_PLASTIC = -0.028

NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
N_TRIALS = 5


def main():
    print("=" * 70)
    print("E2: NOISE ROBUSTNESS — TWO-STAGE σ_zz INVERSION")
    print("=" * 70)
    t0_total = time.time()

    # --- Generate clean observations ---
    print("\n[1] Generating clean observations...")
    mesh_e, bc_e = create_mesh_and_bc(DISP_ELASTIC)
    prob_e = InversionDruckerPrager(mesh_e, vec=3, dim=3, dirichlet_bc_info=bc_e)
    sigma_e_clean, _ = generate_observation(
        prob_e, bc_e, DISP_ELASTIC, E_TRUE, K_TRUE, SOLVER_OPTIONS)

    mesh_p, bc_p = create_mesh_and_bc(DISP_PLASTIC)
    prob_p = InversionDruckerPrager(mesh_p, vec=3, dim=3, dirichlet_bc_info=bc_p)
    sigma_p_clean, _ = generate_observation(
        prob_p, bc_p, DISP_PLASTIC, E_TRUE, K_TRUE, SOLVER_OPTIONS)

    print(f"  σ_zz_elastic (clean) = {sigma_e_clean:.6f} MPa")
    print(f"  σ_zz_plastic (clean) = {sigma_p_clean:.6f} MPa")

    # --- Noise sweep ---
    print("\n[2] Running noise sweep...")
    rng = onp.random.RandomState(42)
    all_results = []

    for noise_level in NOISE_LEVELS:
        print(f"\n  --- Noise level: {noise_level*100:.1f}% ---")
        trial_results = []

        for trial in range(N_TRIALS):
            # Add noise (multiplicative relative noise)
            if noise_level == 0.0:
                sigma_e_noisy = sigma_e_clean
                sigma_p_noisy = sigma_p_clean
            else:
                sigma_e_noisy = sigma_e_clean * (1.0 + noise_level * rng.randn())
                sigma_p_noisy = sigma_p_clean * (1.0 + noise_level * rng.randn())

            verbose = (trial == 0 and noise_level in [0.0, 0.01, 0.10])
            try:
                result = two_stage_inversion(
                    sigma_e_noisy, sigma_p_noisy,
                    E_init=E_INIT, K_init=K_INIT,
                    solver_options=SOLVER_OPTIONS,
                    disp_elastic=DISP_ELASTIC,
                    disp_plastic=DISP_PLASTIC,
                    verbose=verbose,
                )
                err_E = abs(result['E_final'] - E_TRUE) / E_TRUE
                err_k = abs(result['k_final'] - K_TRUE) / K_TRUE
                trial_results.append({
                    'trial': trial,
                    'E_final': result['E_final'],
                    'k_final': result['k_final'],
                    'err_E': err_E,
                    'err_k': err_k,
                    'n_evals': result['n_evals'],
                    'time_s': result['time_s'],
                    'sigma_e_noisy': sigma_e_noisy,
                    'sigma_p_noisy': sigma_p_noisy,
                    'converged': True,
                })
            except Exception as ex:
                print(f"    Trial {trial} failed: {ex}")
                trial_results.append({
                    'trial': trial, 'err_E': float('nan'), 'err_k': float('nan'),
                    'converged': False,
                })

        # Aggregate
        converged = [r for r in trial_results if r['converged']]
        if converged:
            errs_E = [r['err_E'] for r in converged]
            errs_k = [r['err_k'] for r in converged]
            mean_err_E = onp.mean(errs_E)
            std_err_E = onp.std(errs_E)
            mean_err_k = onp.mean(errs_k)
            std_err_k = onp.std(errs_k)
            mean_evals = onp.mean([r['n_evals'] for r in converged])
            pass_E = mean_err_E < 0.01
            pass_k = mean_err_k < 0.01
        else:
            mean_err_E = std_err_E = mean_err_k = std_err_k = float('nan')
            mean_evals = float('nan')
            pass_E = pass_k = False

        level_result = {
            'noise_level': noise_level,
            'n_converged': len(converged),
            'mean_err_E': float(mean_err_E),
            'std_err_E': float(std_err_E),
            'mean_err_k': float(mean_err_k),
            'std_err_k': float(std_err_k),
            'mean_evals': float(mean_evals),
            'pass_E': bool(pass_E),
            'pass_k': bool(pass_k),
            'trials': trial_results,
        }
        all_results.append(level_result)

        status = 'PASS' if (pass_E and pass_k) else 'FAIL'
        print(f"    E err: {mean_err_E:.6%} ± {std_err_E:.6%}  "
              f"k err: {mean_err_k:.6%} ± {std_err_k:.6%}  [{status}]")

    # --- Plotting ---
    print("\n[3] Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    noise_pcts = [r['noise_level'] * 100 for r in all_results]
    mean_E = [r['mean_err_E'] for r in all_results]
    std_E = [r['std_err_E'] for r in all_results]
    mean_k = [r['mean_err_k'] for r in all_results]
    std_k = [r['std_err_k'] for r in all_results]

    # Panel 1: Error vs noise
    ax = axes[0]
    ax.errorbar(noise_pcts, mean_E, yerr=std_E, fmt='b-s', markersize=6,
                capsize=4, linewidth=1.5, label='E error')
    ax.errorbar(noise_pcts, mean_k, yerr=std_k, fmt='r-^', markersize=6,
                capsize=4, linewidth=1.5, label='k error')
    ax.axhline(0.01, color='k', linestyle='--', alpha=0.5, label='1% threshold')
    ax.set_xlabel('Noise level [%]')
    ax.set_ylabel('Relative error')
    ax.set_title('Parameter Error vs Noise')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Panel 2: Number of evaluations vs noise
    ax = axes[1]
    evals = [r['mean_evals'] for r in all_results]
    ax.plot(noise_pcts, evals, 'g-o', markersize=6, linewidth=1.5)
    ax.set_xlabel('Noise level [%]')
    ax.set_ylabel('# function evaluations')
    ax.set_title('Optimization Cost vs Noise')
    ax.grid(True, alpha=0.3)

    # Panel 3: Scatter of recovered (E, k) colored by noise
    ax = axes[2]
    cmap = plt.cm.viridis
    for i, r in enumerate(all_results):
        color = cmap(i / max(len(all_results) - 1, 1))
        for t in r['trials']:
            if t['converged']:
                ax.scatter(t['E_final'], t['k_final'], color=color, s=30, alpha=0.7)
        # Label
        if r['trials'] and r['trials'][0].get('converged'):
            ax.scatter([], [], color=color, s=30, label=f"{r['noise_level']*100:.1f}%")

    ax.plot(E_TRUE, K_TRUE, 'r*', markersize=15, zorder=10, label='True')
    ax.set_xlabel('E [MPa]')
    ax.set_ylabel('k [MPa]')
    ax.set_title('Recovered Parameters')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle('E2: Noise Robustness of Two-Stage Inversion', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'noise_robustness.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")

    # --- Save results ---
    t_total = time.time() - t0_total

    # Clean up trial results for JSON (remove non-serializable)
    save_results = {
        'E_true': E_TRUE, 'k_true': K_TRUE,
        'noise_levels': NOISE_LEVELS,
        'n_trials': N_TRIALS,
        'total_time_s': t_total,
        'levels': [],
    }
    for r in all_results:
        level_save = {k: v for k, v in r.items() if k != 'trials'}
        level_save['trials'] = []
        for t in r['trials']:
            trial_save = {k: float(v) if isinstance(v, (onp.floating, float)) else v
                          for k, v in t.items()
                          if k not in ('history_E', 'history_k')}
            level_save['trials'].append(trial_save)
        save_results['levels'].append(level_save)

    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("E2 SUMMARY")
    print("=" * 70)
    print(f"  {'Noise%':>8s}  {'E err':>12s}  {'k err':>12s}  {'Evals':>6s}  {'Pass':>5s}")
    for r in all_results:
        status = 'PASS' if (r['pass_E'] and r['pass_k']) else 'FAIL'
        print(f"  {r['noise_level']*100:>7.1f}%  {r['mean_err_E']:>11.6%}  "
              f"{r['mean_err_k']:>11.6%}  {r['mean_evals']:>6.0f}  {status:>5s}")
    print(f"\n  Total time: {t_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
