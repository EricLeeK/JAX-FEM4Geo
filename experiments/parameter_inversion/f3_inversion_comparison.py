#!/usr/bin/env python
"""
F3: Full Inversion Comparison — AD vs FD.

Same twin-experiment setup. Two strategies:
  - AD inversion: jax.grad + scipy L-BFGS-B
  - FD inversion: central-difference grad + same L-BFGS-B

Key output: loss vs wall-clock time (the fairest efficiency comparison).
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
    volume_avg_sigma_zz, fd_gradient, RESULTS_DIR,
)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(RESULTS_DIR, 'f3_inversion_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

E_TRUE, K_TRUE = 70000.0, 50.0
E_INIT, K_INIT = 55000.0, 45.0
DISP_ELASTIC = -0.015
DISP_PLASTIC = -0.028
FD_EPS = [100.0, 1.0]  # step sizes for E, k


def run_two_stage_inversion(label, value_and_grad_E, value_and_grad_k,
                            loss_joint_fn, vg_joint, obs_elastic, obs_plastic):
    """Run two-stage inversion and record wall-clock times per eval."""
    from scipy.optimize import minimize_scalar, minimize as scipy_minimize

    history = {'loss': [], 'wallclock': [], 'E': [], 'k': []}
    t0 = time.time()

    # Stage 1: E
    def track_E(E_val):
        l = float(value_and_grad_E(np.array([E_val]))[0])
        history['loss'].append(l)
        history['wallclock'].append(time.time() - t0)
        history['E'].append(E_val)
        history['k'].append(K_INIT)
        return l

    res_E = minimize_scalar(track_E, bounds=(30000, 120000), method='bounded',
                            options={'xatol': 1.0, 'maxiter': 30})
    E_found = res_E.x

    # Stage 2: k
    def track_k(k_val):
        l = float(value_and_grad_k(np.array([k_val]))[0])
        history['loss'].append(l)
        history['wallclock'].append(time.time() - t0)
        history['E'].append(E_found)
        history['k'].append(k_val)
        return l

    res_k = minimize_scalar(track_k, bounds=(35, 80), method='bounded',
                            options={'xatol': 0.1, 'maxiter': 30})
    k_found = res_k.x

    # Stage 3: Joint refinement
    def track_joint(x):
        p = np.array(x)
        l, g = vg_joint(p)
        lv = float(l)
        gn = onp.array(g, dtype=onp.float64)
        history['loss'].append(lv)
        history['wallclock'].append(time.time() - t0)
        history['E'].append(x[0])
        history['k'].append(x[1])
        return lv, gn

    res_j = scipy_minimize(
        track_joint, x0=onp.array([E_found, k_found]),
        method='L-BFGS-B', jac=True,
        bounds=[(30000., 120000.), (35., 80.)],
        options={'maxiter': 30, 'ftol': 1e-20, 'gtol': 1e-12},
    )

    t_total = time.time() - t0
    E_final, k_final = res_j.x
    err_E = abs(E_final - E_TRUE) / E_TRUE
    err_k = abs(k_final - K_TRUE) / K_TRUE

    print(f"  {label}: E={E_final:.2f} (err {err_E:.4%}), k={k_final:.4f} (err {err_k:.4%}), "
          f"time={t_total:.2f}s, evals={len(history['loss'])}")

    return {
        'label': label,
        'E_final': float(E_final), 'k_final': float(k_final),
        'err_E': float(err_E), 'err_k': float(err_k),
        'total_time': t_total,
        'n_evals': len(history['loss']),
        'history': history,
    }


def main():
    print("=" * 70)
    print("F3: FULL INVERSION COMPARISON — AD vs FD")
    print("=" * 70)

    # --- Setup ---
    # Generate observations
    from common import generate_observation
    mesh_e, bc_e = create_mesh_and_bc(DISP_ELASTIC)
    prob_obs_e = InversionDruckerPrager(mesh_e, vec=3, dim=3, dirichlet_bc_info=bc_e)
    obs_elastic, _ = generate_observation(prob_obs_e, bc_e, DISP_ELASTIC, E_TRUE, K_TRUE, SOLVER_OPTIONS)

    mesh_p, bc_p = create_mesh_and_bc(DISP_PLASTIC)
    prob_obs_p = InversionDruckerPrager(mesh_p, vec=3, dim=3, dirichlet_bc_info=bc_p)
    obs_plastic, _ = generate_observation(prob_obs_p, bc_p, DISP_PLASTIC, E_TRUE, K_TRUE, SOLVER_OPTIONS)

    print(f"  Observations: σ_elastic={obs_elastic:.4f}, σ_plastic={obs_plastic:.4f}")

    # --- AD setup ---
    mesh_e2, bc_e2 = create_mesh_and_bc(DISP_ELASTIC)
    prob_e_ad = InversionDruckerPrager(mesh_e2, vec=3, dim=3, dirichlet_bc_info=bc_e2)
    fwd_e_ad = ad_wrapper(prob_e_ad, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)

    mesh_p2, bc_p2 = create_mesh_and_bc(DISP_PLASTIC)
    prob_p_ad = InversionDruckerPrager(mesh_p2, vec=3, dim=3, dirichlet_bc_info=bc_p2)
    fwd_p_ad = ad_wrapper(prob_p_ad, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)

    def loss_E_ad(params_E):
        E = params_E[0]; k_dum = 200.0
        sol = fwd_e_ad(np.array([E, k_dum]))[0]
        pred = volume_avg_sigma_zz(prob_e_ad.fe, sol, prob_e_ad.sigmas_old, prob_e_ad.epsilons_old, E, k_dum)
        return (pred - obs_elastic) ** 2

    def loss_k_ad(params_k):
        k = params_k[0]; E_fix = 70000.0  # will be overwritten after stage 1
        sol = fwd_p_ad(np.array([E_fix, k]))[0]
        pred = volume_avg_sigma_zz(prob_p_ad.fe, sol, prob_p_ad.sigmas_old, prob_p_ad.epsilons_old, E_fix, k)
        return (pred - obs_plastic) ** 2

    def loss_joint_ad(params):
        E, k = params[0], params[1]
        sol_e = fwd_e_ad(params)[0]
        pred_e = volume_avg_sigma_zz(prob_e_ad.fe, sol_e, prob_e_ad.sigmas_old, prob_e_ad.epsilons_old, E, k)
        sol_p = fwd_p_ad(params)[0]
        pred_p = volume_avg_sigma_zz(prob_p_ad.fe, sol_p, prob_p_ad.sigmas_old, prob_p_ad.epsilons_old, E, k)
        return (pred_e - obs_elastic) ** 2 + (pred_p - obs_plastic) ** 2

    # Warm up AD
    print("\n  Warming up AD...")
    _ = jax.value_and_grad(loss_E_ad)(np.array([E_INIT]))
    _ = jax.value_and_grad(loss_k_ad)(np.array([K_INIT]))
    _ = jax.value_and_grad(loss_joint_ad)(np.array([E_INIT, K_INIT]))

    # --- FD setup (same problems, but gradient via FD) ---
    mesh_e3, bc_e3 = create_mesh_and_bc(DISP_ELASTIC)
    prob_e_fd = InversionDruckerPrager(mesh_e3, vec=3, dim=3, dirichlet_bc_info=bc_e3)
    fwd_e_fd = ad_wrapper(prob_e_fd, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)

    mesh_p3, bc_p3 = create_mesh_and_bc(DISP_PLASTIC)
    prob_p_fd = InversionDruckerPrager(mesh_p3, vec=3, dim=3, dirichlet_bc_info=bc_p3)
    fwd_p_fd = ad_wrapper(prob_p_fd, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)

    def loss_E_fd_raw(params_E):
        E = params_E[0]; k_dum = 200.0
        sol = fwd_e_fd(np.array([E, k_dum]))[0]
        pred = volume_avg_sigma_zz(prob_e_fd.fe, sol, prob_e_fd.sigmas_old, prob_e_fd.epsilons_old, E, k_dum)
        return (pred - obs_elastic) ** 2

    def loss_k_fd_raw(params_k):
        k = params_k[0]; E_fix = 70000.0
        sol = fwd_p_fd(np.array([E_fix, k]))[0]
        pred = volume_avg_sigma_zz(prob_p_fd.fe, sol, prob_p_fd.sigmas_old, prob_p_fd.epsilons_old, E_fix, k)
        return (pred - obs_plastic) ** 2

    def loss_joint_fd_raw(params):
        E, k = params[0], params[1]
        sol_e = fwd_e_fd(params)[0]
        pred_e = volume_avg_sigma_zz(prob_e_fd.fe, sol_e, prob_e_fd.sigmas_old, prob_e_fd.epsilons_old, E, k)
        sol_p = fwd_p_fd(params)[0]
        pred_p = volume_avg_sigma_zz(prob_p_fd.fe, sol_p, prob_p_fd.sigmas_old, prob_p_fd.epsilons_old, E, k)
        return (pred_e - obs_elastic) ** 2 + (pred_p - obs_plastic) ** 2

    # FD value_and_grad wrappers
    def vg_E_fd(params):
        l = loss_E_fd_raw(params)
        g = fd_gradient(loss_E_fd_raw, params, [100.0])
        return l, np.array(g)

    def vg_k_fd(params):
        l = loss_k_fd_raw(params)
        g = fd_gradient(loss_k_fd_raw, params, [1.0])
        return l, np.array(g)

    def vg_joint_fd(params):
        l = loss_joint_fd_raw(params)
        g = fd_gradient(loss_joint_fd_raw, params, FD_EPS)
        return l, np.array(g)

    # Warm up FD
    print("  Warming up FD...")
    _ = vg_E_fd(np.array([E_INIT]))
    _ = vg_k_fd(np.array([K_INIT]))

    # --- Run inversions ---
    print("\n--- AD Inversion ---")
    result_ad = run_two_stage_inversion(
        "AD", jax.value_and_grad(loss_E_ad), jax.value_and_grad(loss_k_ad),
        loss_joint_ad, jax.value_and_grad(loss_joint_ad),
        obs_elastic, obs_plastic,
    )

    print("\n--- FD Inversion ---")
    result_fd = run_two_stage_inversion(
        "FD", vg_E_fd, vg_k_fd,
        loss_joint_fd_raw, vg_joint_fd,
        obs_elastic, obs_plastic,
    )

    # --- Plotting ---
    print("\n[Plot] Generating comparison plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss vs wall-clock time
    ax = axes[0]
    h_ad = result_ad['history']
    h_fd = result_fd['history']
    ax.semilogy(h_ad['wallclock'], h_ad['loss'], 'b-o', markersize=3, label='AD', linewidth=1.5)
    ax.semilogy(h_fd['wallclock'], h_fd['loss'], 'r-s', markersize=3, label='FD', linewidth=1.5)
    ax.set_xlabel('Wall-clock time [s]')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Wall-Clock Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss vs iteration
    ax = axes[1]
    ax.semilogy(h_ad['loss'], 'b-o', markersize=3, label='AD', linewidth=1.5)
    ax.semilogy(h_fd['loss'], 'r-s', markersize=3, label='FD', linewidth=1.5)
    ax.set_xlabel('Function evaluation')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Evaluations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Parameter error vs time
    ax = axes[2]
    err_E_ad = [abs(e - E_TRUE) / E_TRUE for e in h_ad['E']]
    err_k_ad = [abs(k - K_TRUE) / K_TRUE for k in h_ad['k']]
    err_E_fd = [abs(e - E_TRUE) / E_TRUE for e in h_fd['E']]
    err_k_fd = [abs(k - K_TRUE) / K_TRUE for k in h_fd['k']]
    ax.semilogy(h_ad['wallclock'], err_E_ad, 'b-', linewidth=1.5, label='AD: E err')
    ax.semilogy(h_ad['wallclock'], [max(e, 1e-16) for e in err_k_ad], 'b--', linewidth=1.5, label='AD: k err')
    ax.semilogy(h_fd['wallclock'], err_E_fd, 'r-', linewidth=1.5, label='FD: E err')
    ax.semilogy(h_fd['wallclock'], [max(e, 1e-16) for e in err_k_fd], 'r--', linewidth=1.5, label='FD: k err')
    ax.axhline(0.01, color='k', linestyle=':', alpha=0.5, label='1% target')
    ax.set_xlabel('Wall-clock time [s]')
    ax.set_ylabel('Relative parameter error')
    ax.set_title('Convergence Speed')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('F3: AD vs FD Inversion Comparison', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")

    # Save
    save_results = {
        'ad': {k: v for k, v in result_ad.items() if k != 'history'},
        'fd': {k: v for k, v in result_fd.items() if k != 'history'},
        'speedup': result_fd['total_time'] / max(result_ad['total_time'], 1e-10),
    }
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("F3 SUMMARY")
    print("=" * 70)
    print(f"  {'':15s} {'AD':>12s}  {'FD':>12s}")
    print(f"  {'Total time':15s} {result_ad['total_time']:>11.2f}s  {result_fd['total_time']:>11.2f}s")
    print(f"  {'Evaluations':15s} {result_ad['n_evals']:>12d}  {result_fd['n_evals']:>12d}")
    print(f"  {'E error':15s} {result_ad['err_E']:>11.4%}  {result_fd['err_E']:>11.4%}")
    print(f"  {'k error':15s} {result_ad['err_k']:>11.4%}  {result_fd['err_k']:>11.4%}")
    print(f"\n  Speedup (FD time / AD time): {save_results['speedup']:.2f}×")
    print("=" * 70)


if __name__ == "__main__":
    main()
