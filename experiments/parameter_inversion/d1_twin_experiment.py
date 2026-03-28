#!/usr/bin/env python
"""
D1: Twin Experiment — Synthetic Data Parameter Inversion

Strategy: Two-stage inversion (physically motivated)
  Stage 1: Elastic observation determines E (k does not affect elastic response)
  Stage 2: Plastic observation determines k (with E fixed from Stage 1)
  Stage 3: Joint refinement of (E, k) starting from Stage 1+2 result

This avoids the fundamental pitfall where the optimizer pushes k into
the elastic regime (high k → material elastic → dL/dk = 0 → k stuck).

Pass criteria:
  - |E_final - E_true| / E_true < 0.01
  - |k_final - k_true| / k_true < 0.01
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from common import (
    InversionDruckerPrager, create_mesh_and_bc, update_bc,
    volume_avg_sigma_zz, generate_observation, adam_optimize,
    RESULTS_DIR, stress_return_dp,
)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper, solver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
E_TRUE = 70000.0
K_TRUE = 50.0
E_INIT = 55000.0
K_INIT = 45.0

DISP_ELASTIC = -0.015   # elastic regime for all reasonable E, k
DISP_PLASTIC = -0.028   # plastic regime at true params

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

OUT_DIR = os.path.join(RESULTS_DIR, 'd1_twin_experiment')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("D1: TWIN EXPERIMENT — TWO-STAGE PARAMETER INVERSION")
    print("=" * 70)
    print(f"  True params:   E = {E_TRUE}, k = {K_TRUE}")
    print(f"  Initial guess: E = {E_INIT}, k = {K_INIT}")
    print()

    # --- Step 1: Generate synthetic observations ---
    print("[1] Generating synthetic observations...")
    t0_total = time.time()

    mesh_e, bc_e = create_mesh_and_bc(DISP_ELASTIC)
    prob_obs_e = InversionDruckerPrager(mesh_e, vec=3, dim=3, dirichlet_bc_info=bc_e)
    obs_elastic, _ = generate_observation(prob_obs_e, bc_e, DISP_ELASTIC, E_TRUE, K_TRUE, SOLVER_OPTIONS)

    mesh_p, bc_p = create_mesh_and_bc(DISP_PLASTIC)
    prob_obs_p = InversionDruckerPrager(mesh_p, vec=3, dim=3, dirichlet_bc_info=bc_p)
    obs_plastic, _ = generate_observation(prob_obs_p, bc_p, DISP_PLASTIC, E_TRUE, K_TRUE, SOLVER_OPTIONS)

    print(f"  σ_zz_obs (elastic, disp={DISP_ELASTIC}): {obs_elastic:.6f} MPa")
    print(f"  σ_zz_obs (plastic, disp={DISP_PLASTIC}): {obs_plastic:.6f} MPa")

    # --- Stage 1: Invert E from elastic observation ---
    print("\n" + "=" * 70)
    print("STAGE 1: Invert E from elastic observation")
    print("=" * 70)

    mesh_e2, bc_e2 = create_mesh_and_bc(DISP_ELASTIC)
    prob_e = InversionDruckerPrager(mesh_e2, vec=3, dim=3, dirichlet_bc_info=bc_e2)
    fwd_e = ad_wrapper(prob_e, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)

    def loss_E(params):
        """Loss for E only; k fixed at a high value (elastic regime)."""
        E = params[0]
        k_dummy = 200.0  # high enough that elastic everywhere
        full_params = np.array([E, k_dummy])
        sol = fwd_e(full_params)[0]
        pred = volume_avg_sigma_zz(
            prob_e.fe, sol, prob_e.sigmas_old, prob_e.epsilons_old, E, k_dummy,
        )
        return (pred - obs_elastic) ** 2

    vg_E = jax.value_and_grad(loss_E)

    # Warm up
    l0, g0 = vg_E(np.array([E_INIT]))
    print(f"  Initial: loss={float(l0):.6e}, dL/dE={float(g0[0]):.6e}")

    history_E = {'loss': [], 'E': []}
    try:
        from scipy.optimize import minimize_scalar

        def E_loss_scalar(E_val):
            l = float(loss_E(np.array([E_val])))
            history_E['loss'].append(l)
            history_E['E'].append(E_val)
            return l

        res_E = minimize_scalar(E_loss_scalar, bounds=(30000, 120000), method='bounded',
                                options={'xatol': 1.0, 'maxiter': 30})
        E_stage1 = res_E.x
        print(f"  Scipy bounded: E = {E_stage1:.2f} in {res_E.nfev} evaluations")
    except ImportError:
        def vg_E_1d(params):
            return vg_E(params[:1])

        p, h = adam_optimize(vg_E_1d, onp.array([E_INIT]), num_iters=30, lr=2000.,
                             bounds=[(30000, 120000)])
        E_stage1 = float(p[0])
        history_E = {'loss': h['loss'], 'E': [pp[0] for pp in h['params']]}

    err_E_s1 = abs(E_stage1 - E_TRUE) / E_TRUE
    print(f"  Stage 1 result: E = {E_stage1:.2f} (err {err_E_s1:.6%})")

    # --- Stage 2: Invert k from plastic observation (E fixed) ---
    print("\n" + "=" * 70)
    print(f"STAGE 2: Invert k from plastic observation (E fixed at {E_stage1:.0f})")
    print("=" * 70)

    mesh_p2, bc_p2 = create_mesh_and_bc(DISP_PLASTIC)
    prob_p = InversionDruckerPrager(mesh_p2, vec=3, dim=3, dirichlet_bc_info=bc_p2)
    fwd_p = ad_wrapper(prob_p, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)

    E_fixed = E_stage1

    def loss_k(params_k):
        """Loss for k only; E fixed from Stage 1."""
        k = params_k[0]
        full_params = np.array([E_fixed, k])
        sol = fwd_p(full_params)[0]
        pred = volume_avg_sigma_zz(
            prob_p.fe, sol, prob_p.sigmas_old, prob_p.epsilons_old, E_fixed, k,
        )
        return (pred - obs_plastic) ** 2

    vg_k = jax.value_and_grad(loss_k)

    # Warm up
    l0k, g0k = vg_k(np.array([K_INIT]))
    print(f"  Initial: loss={float(l0k):.6e}, dL/dk={float(g0k[0]):.6e}")

    history_k = {'loss': [], 'k': []}
    try:
        from scipy.optimize import minimize_scalar

        def k_loss_scalar(k_val):
            l = float(loss_k(np.array([k_val])))
            history_k['loss'].append(l)
            history_k['k'].append(k_val)
            return l

        res_k = minimize_scalar(k_loss_scalar, bounds=(35, 80), method='bounded',
                                options={'xatol': 0.1, 'maxiter': 30})
        k_stage2 = res_k.x
        print(f"  Scipy bounded: k = {k_stage2:.4f} in {res_k.nfev} evaluations")
    except ImportError:
        def vg_k_1d(params):
            return vg_k(params[:1])

        p, h = adam_optimize(vg_k_1d, onp.array([K_INIT]), num_iters=30, lr=5.,
                             bounds=[(15, 100)])
        k_stage2 = float(p[0])
        history_k = {'loss': h['loss'], 'k': [pp[0] for pp in h['params']]}

    err_k_s2 = abs(k_stage2 - K_TRUE) / K_TRUE
    print(f"  Stage 2 result: k = {k_stage2:.4f} (err {err_k_s2:.6%})")

    # --- Stage 3: Joint refinement ---
    print("\n" + "=" * 70)
    print(f"STAGE 3: Joint refinement from ({E_stage1:.0f}, {k_stage2:.2f})")
    print("=" * 70)

    def loss_joint(params):
        E, k = params[0], params[1]
        sol_e = fwd_e(params)[0]
        pred_e = volume_avg_sigma_zz(
            prob_e.fe, sol_e, prob_e.sigmas_old, prob_e.epsilons_old, E, k,
        )
        sol_p = fwd_p(params)[0]
        pred_p = volume_avg_sigma_zz(
            prob_p.fe, sol_p, prob_p.sigmas_old, prob_p.epsilons_old, E, k,
        )
        return (pred_e - obs_elastic) ** 2 + (pred_p - obs_plastic) ** 2

    vg_joint = jax.value_and_grad(loss_joint)

    p_start = np.array([E_stage1, k_stage2])
    l0j, g0j = vg_joint(p_start)
    print(f"  Start: loss={float(l0j):.6e}, grad=({float(g0j[0]):.2e}, {float(g0j[1]):.2e})")

    history_joint = {'loss': [], 'params': []}
    try:
        from scipy.optimize import minimize as scipy_minimize

        def scipy_joint(x):
            p = np.array(x)
            l, g = vg_joint(p)
            lv = float(l)
            gn = onp.array(g, dtype=onp.float64)
            history_joint['loss'].append(lv)
            history_joint['params'].append(onp.array(x).copy())
            step = len(history_joint['loss'])
            if step <= 3 or step % 5 == 0:
                print(f"  Eval {step:3d}: loss={lv:.6e}  E={x[0]:.1f} k={x[1]:.4f}")
            return lv, gn

        res_j = scipy_minimize(
            scipy_joint,
            x0=onp.array([E_stage1, k_stage2]),
            method='L-BFGS-B',
            jac=True,
            bounds=[(30000., 120000.), (15., 100.)],
            options={'maxiter': 50, 'ftol': 1e-20, 'gtol': 1e-12},
        )
        E_final, k_final = res_j.x
        print(f"  Joint L-BFGS-B: {res_j.nit} iters, converged={res_j.success}")
    except ImportError:
        def callback_j(step, params, loss_val, grad):
            if step % 5 == 0:
                print(f"  Step {step}: loss={loss_val:.6e} E={params[0]:.1f} k={params[1]:.3f}")

        pf, history_joint = adam_optimize(
            vg_joint, onp.array([E_stage1, k_stage2]),
            num_iters=30, lr=50., bounds=[(30000, 120000), (15, 100)],
            callback=callback_j,
        )
        E_final, k_final = float(pf[0]), float(pf[1])

    t_total = time.time() - t0_total
    err_E_final = abs(E_final - E_TRUE) / E_TRUE
    err_k_final = abs(k_final - K_TRUE) / K_TRUE
    print(f"  Final: E = {E_final:.2f} (err {err_E_final:.6%}), k = {k_final:.6f} (err {err_k_final:.6%})")

    # --- Plotting ---
    print("\n[Plot] Generating convergence plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss convergence across all stages
    ax = axes[0]
    n1 = len(history_E['loss'])
    n2 = len(history_k['loss'])
    ax.semilogy(range(n1), history_E['loss'], 'b-o', markersize=3, label='Stage 1 (E)')
    ax.semilogy(range(n1, n1 + n2), history_k['loss'], 'r-o', markersize=3, label='Stage 2 (k)')
    if history_joint.get('loss'):
        n3 = len(history_joint['loss'])
        ax.semilogy(range(n1 + n2, n1 + n2 + n3), history_joint['loss'],
                     'g-o', markersize=3, label='Stage 3 (joint)')
    ax.set_xlabel('Function evaluation')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Parameter trajectories
    ax = axes[1]
    E_traj = history_E['E'] + [E_stage1] * n2
    if history_joint.get('params'):
        E_traj += [p[0] for p in history_joint['params']]
    ax.plot(E_traj, 'b-', linewidth=1.5, label='E')
    ax.axhline(E_TRUE, color='b', linestyle='--', alpha=0.5, label=f'E_true={E_TRUE}')
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('E [MPa]', color='b')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    k_traj = [K_INIT] * n1 + history_k['k']
    if history_joint.get('params'):
        k_traj += [p[1] for p in history_joint['params']]
    ax2.plot(k_traj, 'r-', linewidth=1.5, label='k')
    ax2.axhline(K_TRUE, color='r', linestyle='--', alpha=0.5, label=f'k_true={K_TRUE}')
    ax2.set_ylabel('k [MPa]', color='r')
    ax2.legend(loc='upper right')
    ax.set_title('Parameter Convergence')

    # Parameter space
    ax = axes[2]
    if history_joint.get('params'):
        pj = onp.array(history_joint['params'])
        ax.plot(pj[:, 0], pj[:, 1], 'g.-', markersize=4, linewidth=0.8, label='Joint path')
    ax.plot(E_INIT, K_INIT, 'gs', markersize=10, label='Start')
    ax.plot(E_stage1, K_INIT, 'bs', markersize=8, label='After Stage 1')
    ax.plot(E_stage1, k_stage2, 'rs', markersize=8, label='After Stage 2')
    ax.plot(E_final, k_final, 'kD', markersize=10, label='Final')
    ax.plot(E_TRUE, K_TRUE, 'r*', markersize=15, label='True')
    ax.set_xlabel('E [MPa]')
    ax.set_ylabel('k [MPa]')
    ax.set_title('Parameter Space')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'convergence.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved: {plot_path}")

    # --- Save results ---
    results = {
        'E_true': E_TRUE, 'k_true': K_TRUE,
        'E_init': E_INIT, 'k_init': K_INIT,
        'obs_elastic': obs_elastic, 'obs_plastic': obs_plastic,
        'stage1': {'E': float(E_stage1), 'err_E': float(err_E_s1)},
        'stage2': {'k': float(k_stage2), 'err_k': float(err_k_s2)},
        'final': {
            'E': float(E_final), 'k': float(k_final),
            'err_E': float(err_E_final), 'err_k': float(err_k_final),
        },
        'total_time_s': t_total,
    }
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {json_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Stage 1: E = {E_stage1:.2f}  (true={E_TRUE}, err={err_E_s1:.6%})")
    print(f"  Stage 2: k = {k_stage2:.4f}  (true={K_TRUE}, err={err_k_s2:.6%})")
    print(f"  Final:   E = {E_final:.2f}, k = {k_final:.4f}")
    print(f"           E err = {err_E_final:.6%}, k err = {err_k_final:.6%}")
    print(f"  Total time: {t_total:.1f}s")

    pass_E = err_E_final < 0.01
    pass_k = err_k_final < 0.01
    print(f"\n  E convergence (<1%): {'PASS' if pass_E else 'FAIL'}")
    print(f"  k convergence (<1%): {'PASS' if pass_k else 'FAIL'}")
    overall = pass_E and pass_k
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 70)

    assert overall, f"Inversion failed: E err={err_E_final:.4%}, k err={err_k_final:.4%}"
    return results


if __name__ == "__main__":
    main()
