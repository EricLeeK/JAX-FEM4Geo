#!/usr/bin/env python
"""
D3: Multi-Step Loading Path Inversion

Workflow:
  1. Generate synthetic observations along a 3-step loading path:
     Step 1: -0.015 (elastic)
     Step 2: -0.030 (plastic)
     Step 3: -0.020 (elastic unload)
  2. Observation = σ_zz at the final step (with path-dependent history).
  3. Invert E and k using the same last-step-only AD strategy as the tests.
  4. Compare single-step vs multi-step observation accuracy.

Key insight: multi-step observations carry richer information because the
final stress depends on the entire loading history (path-dependent plasticity).

AD strategy: only the LAST step is differentiated (implicit diff);
history steps are replayed at nominal params (stop-gradient).

Note on unrolled differentiation (D3b):
  Ideally we'd differentiate through ALL load steps. However, JAX-FEM's
  ad_wrapper uses custom_vjp with implicit differentiation at each solve's
  fixed point. The internal variables (sigma_old, epsilon_old) are mutated
  on the problem object as numpy-like arrays — they are NOT part of the JAX
  trace. This means gradients cannot flow through history updates between
  steps. A true unrolled approach would require making internal state part
  of the JAX computation graph (a fundamental JAX-FEM architectural change).

  Instead, D3b uses a two-stage strategy applied to multi-step observations:
    Stage 1: Invert E from step 1 (elastic regime, k-independent)
    Stage 2: Invert k from the final multi-step observation (E fixed)
  This physically decouples the parameters while using richer path information.
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
E_INIT = 65000.0
K_INIT = 45.0

DISPLACEMENTS = [-0.015, -0.030, -0.020]  # elastic → plastic → unload

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

OUT_DIR = os.path.join(RESULTS_DIR, 'd3_multistep')
os.makedirs(OUT_DIR, exist_ok=True)


def run_history_steps(problem, params, dirichlet_bc_info, displacements, solver_options,
                     n_sub=3):
    """Run history steps (all except the last) to build up internal vars.

    params must be concrete (not JAX tracers) — call this OUTSIDE jax.grad.
    Uses sub-increments between displacement steps for robustness.
    """
    E = float(onp.array(params[0]))
    k = float(onp.array(params[1]))
    dim = problem.dim

    prev_disp = 0.0
    prev_sol = None
    for step_idx, disp in enumerate(displacements[:-1]):
        # Sub-increment from prev_disp to disp
        for sub in range(n_sub):
            frac = (sub + 1) / n_sub
            sub_disp = prev_disp + frac * (disp - prev_disp)
            update_bc(problem, dirichlet_bc_info, sub_disp)
            problem.set_params(params)
            step_opts = dict(solver_options)
            if prev_sol is not None:
                step_opts['initial_guess'] = [prev_sol]
            sol_list = solver(problem, solver_options=step_opts)
            sol = sol_list[0]
            prev_sol = sol

            # Update internal vars from solution
            fe = problem.fe
            u_grads = fe.sol_to_grad(sol)

            def one_stress(ug, so, eo):
                return stress_return_dp(ug, so, eo, E, k, dim)

            def one_strain(ug):
                return 0.5 * (ug + ug.T)

            new_sigmas = jax.vmap(jax.vmap(one_stress))(
                u_grads, problem.sigmas_old, problem.epsilons_old,
            )
            new_epsilons = jax.vmap(jax.vmap(one_strain))(u_grads)

            problem.sigmas_old = new_sigmas
            problem.epsilons_old = new_epsilons
            problem.internal_vars[0] = new_sigmas
            problem.internal_vars[1] = new_epsilons

        prev_disp = disp


def generate_multistep_observation(problem, dirichlet_bc_info, displacements,
                                   E_true, k_true, solver_options):
    """Generate observation σ_zz at the final step of a multi-step path."""
    params_true = np.array([E_true, k_true])

    # Reset
    problem.reset_internal_vars()

    # Run all steps
    for step_idx, disp in enumerate(displacements):
        update_bc(problem, dirichlet_bc_info, disp)
        problem.set_params(params_true)
        sol_list = solver(problem, solver_options=solver_options)
        sol = sol_list[0]

        if step_idx < len(displacements) - 1:
            # Update history
            fe = problem.fe
            u_grads = fe.sol_to_grad(sol)
            dim = problem.dim

            def one_stress(ug, so, eo):
                return stress_return_dp(ug, so, eo, E_true, k_true, dim)

            def one_strain(ug):
                return 0.5 * (ug + ug.T)

            problem.sigmas_old = jax.vmap(jax.vmap(one_stress))(
                u_grads, problem.sigmas_old, problem.epsilons_old,
            )
            problem.epsilons_old = jax.vmap(jax.vmap(one_strain))(u_grads)
            problem.internal_vars[0] = problem.sigmas_old
            problem.internal_vars[1] = problem.epsilons_old

    # Compute observation at final step
    sigma_zz = volume_avg_sigma_zz(
        problem.fe, sol, problem.sigmas_old, problem.epsilons_old, E_true, k_true,
    )
    return float(sigma_zz), sol


def run_inversion(label, displacements, obs_value, E_init, k_init, num_iters=50, lr=300.0):
    """Run parameter inversion for a given observation setup."""
    print(f"\n--- Inversion: {label} ---")
    print(f"  Displacements: {displacements}")
    print(f"  Observation σ_zz = {obs_value:.6f} MPa")
    print(f"  Initial: E = {E_init}, k = {k_init}")

    mesh, bc_info = create_mesh_and_bc(displacements[0])
    problem = InversionDruckerPrager(mesh, vec=3, dim=3, dirichlet_bc_info=bc_info)
    fwd_pred = ad_wrapper(problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    params_init_np = np.array([E_init, k_init])

    if len(displacements) == 1:
        # Single-step: straightforward
        def loss_fn(params):
            E, k = params[0], params[1]
            sol = fwd_pred(params)[0]
            pred = volume_avg_sigma_zz(
                problem.fe, sol, problem.sigmas_old, problem.epsilons_old, E, k,
            )
            return (pred - obs_value) ** 2
    else:
        # Multi-step: replay history, then AD only on last step.
        # History replay must happen OUTSIDE jax.grad (concrete params).
        def loss_fn_with_history(params_concrete):
            """Called with concrete params; rebuilds history then does AD last step."""
            problem.reset_internal_vars()
            run_history_steps(problem, params_concrete, bc_info, displacements, SOLVER_OPTIONS)
            update_bc(problem, bc_info, displacements[-1])

            def ad_last_step(params):
                E, k = params[0], params[1]
                sol = fwd_pred(params)[0]
                pred = volume_avg_sigma_zz(
                    problem.fe, sol, problem.sigmas_old, problem.epsilons_old, E, k,
                )
                return (pred - obs_value) ** 2

            return jax.value_and_grad(ad_last_step)(params_concrete)

        loss_fn = None  # will use loss_fn_with_history instead

    if loss_fn is not None:
        value_and_grad_fn = jax.value_and_grad(loss_fn)
    else:
        value_and_grad_fn = loss_fn_with_history

    # Warm up
    loss0, grad0 = value_and_grad_fn(params_init_np)
    print(f"  Initial loss: {float(loss0):.6e}")

    def callback(step, params, loss_val, grad):
        if step % 10 == 0 or step < 3:
            print(f"    Step {step:3d}: loss={loss_val:.6e}  "
                  f"E={params[0]:.1f} k={params[1]:.3f}")

    t0 = time.time()
    params_final, history = adam_optimize(
        value_and_grad_fn,
        onp.array([E_init, k_init]),
        num_iters=num_iters,
        lr=lr,
        bounds=[(30000., 120000.), (35., 80.)],
        callback=callback,
    )
    t_opt = time.time() - t0

    # Try L-BFGS-B refinement
    E_best, k_best = float(params_final[0]), float(params_final[1])
    try:
        from scipy.optimize import minimize as scipy_minimize

        def scipy_fn(x):
            p = np.array(x)
            l, g = value_and_grad_fn(p)
            return float(l), onp.array(g, dtype=onp.float64)

        res = scipy_minimize(scipy_fn, x0=onp.array([E_best, k_best]),
                             method='L-BFGS-B', jac=True,
                             bounds=[(30000., 120000.), (35., 80.)],
                             options={'maxiter': 30, 'ftol': 1e-20, 'gtol': 1e-12})
        E_best, k_best = res.x
        print(f"  L-BFGS-B refined: E={E_best:.2f}, k={k_best:.4f}")
    except ImportError:
        pass

    err_E = abs(E_best - E_TRUE) / E_TRUE
    err_k = abs(k_best - K_TRUE) / K_TRUE

    print(f"  Final: E = {E_best:.2f} (err {err_E:.4%}), k = {k_best:.4f} (err {err_k:.4%})")
    print(f"  Time: {t_opt:.2f}s")

    return {
        'label': label,
        'displacements': displacements,
        'obs_value': obs_value,
        'E_final': E_best, 'k_final': k_best,
        'err_E': err_E, 'err_k': err_k,
        'history': history,
        'time_s': t_opt,
    }


def run_twostage_multistep_inversion(obs_elastic, obs_multistep, displacements,
                                     E_init, k_init, num_iters=50, lr=300.0):
    """Two-stage inversion using elastic + multi-step observations.

    Stage 1: Invert E from elastic-regime observation (step 1).
    Stage 2: Invert k from multi-step final observation, with E fixed
             and history replayed at (E_fixed, k_current).
    """
    from scipy.optimize import minimize_scalar

    print("\n--- Inversion: Two-stage multi-step ---")
    print(f"  Elastic obs σ_zz = {obs_elastic:.6f} MPa")
    print(f"  Multi-step obs σ_zz = {obs_multistep:.6f} MPa")

    t0 = time.time()

    # --- Stage 1: Invert E from elastic observation ---
    print("  Stage 1: Invert E from elastic observation...")
    mesh_e, bc_e = create_mesh_and_bc(displacements[0])
    prob_e = InversionDruckerPrager(mesh_e, vec=3, dim=3, dirichlet_bc_info=bc_e)
    fwd_e = ad_wrapper(prob_e, solver_options=SOLVER_OPTIONS,
                       adjoint_solver_options=SOLVER_OPTIONS)

    def loss_E(params_E):
        E = params_E[0]
        k_dummy = 200.0  # high enough to stay elastic
        sol = fwd_e(np.array([E, k_dummy]))[0]
        pred = volume_avg_sigma_zz(
            prob_e.fe, sol, prob_e.sigmas_old, prob_e.epsilons_old, E, k_dummy,
        )
        return (pred - obs_elastic) ** 2

    # Warm up
    _ = jax.value_and_grad(loss_E)(np.array([E_init]))

    history_E_vals = []

    def track_E(E_val):
        l = float(loss_E(np.array([E_val])))
        history_E_vals.append({'E': E_val, 'loss': l})
        return l

    res_E = minimize_scalar(track_E, bounds=(30000, 120000), method='bounded',
                            options={'xatol': 1.0, 'maxiter': 30})
    E_fixed = res_E.x
    print(f"    E = {E_fixed:.2f} ({res_E.nfev} evals, err {abs(E_fixed - E_TRUE)/E_TRUE:.4%})")

    # --- Stage 2: Invert k from multi-step final observation ---
    print("  Stage 2: Invert k from multi-step observation (E fixed)...")
    mesh_k, bc_k = create_mesh_and_bc(displacements[0])
    prob_k = InversionDruckerPrager(mesh_k, vec=3, dim=3, dirichlet_bc_info=bc_k)
    fwd_k = ad_wrapper(prob_k, solver_options=SOLVER_OPTIONS,
                       adjoint_solver_options=SOLVER_OPTIONS)

    def loss_k_with_history(params_k_concrete):
        """Replay history with (E_fixed, k), then AD on last step."""
        k_val = float(onp.array(params_k_concrete[0]))
        full_params = np.array([E_fixed, k_val])

        # Reset and replay history steps
        prob_k.reset_internal_vars()
        run_history_steps(prob_k, full_params, bc_k, displacements, SOLVER_OPTIONS)

        # AD on last step
        update_bc(prob_k, bc_k, displacements[-1])

        def ad_last(params_k):
            k = params_k[0]
            p = np.array([E_fixed, k])
            sol = fwd_k(p)[0]
            pred = volume_avg_sigma_zz(
                prob_k.fe, sol, prob_k.sigmas_old, prob_k.epsilons_old,
                E_fixed, k,
            )
            return (pred - obs_multistep) ** 2

        return jax.value_and_grad(ad_last)(params_k_concrete)

    # Warm up
    _ = loss_k_with_history(np.array([k_init]))

    history_k_vals = []

    def track_k(k_val):
        l, _ = loss_k_with_history(np.array([k_val]))
        l = float(l)
        history_k_vals.append({'k': k_val, 'loss': l})
        return l

    res_k = minimize_scalar(track_k, bounds=(35, 80), method='bounded',
                            options={'xatol': 0.1, 'maxiter': 30})
    k_best = res_k.x
    print(f"    k = {k_best:.4f} ({res_k.nfev} evals, err {abs(k_best - K_TRUE)/K_TRUE:.4%})")

    t_opt = time.time() - t0
    E_best = E_fixed
    err_E = abs(E_best - E_TRUE) / E_TRUE
    err_k = abs(k_best - K_TRUE) / K_TRUE

    print(f"  Final: E = {E_best:.2f} (err {err_E:.4%}), k = {k_best:.4f} (err {err_k:.4%})")
    print(f"  Time: {t_opt:.2f}s")

    # Build a combined history for plotting (loss across both stages)
    all_loss = [h['loss'] for h in history_E_vals] + [h['loss'] for h in history_k_vals]
    all_params = [[h['E'], k_init] for h in history_E_vals] + \
                 [[E_best, h['k']] for h in history_k_vals]

    return {
        'label': 'Two-stage multi-step',
        'displacements': displacements,
        'obs_value': obs_multistep,
        'E_final': E_best, 'k_final': k_best,
        'err_E': err_E, 'err_k': err_k,
        'history': {'loss': all_loss, 'params': all_params},
        'time_s': t_opt,
    }


def main():
    print("=" * 70)
    print("D3: MULTI-STEP LOADING PATH INVERSION")
    print("=" * 70)
    print(f"  True params: E = {E_TRUE}, k = {K_TRUE}")
    print(f"  Initial:     E = {E_INIT}, k = {K_INIT}")
    print(f"  Full path:   {DISPLACEMENTS}")

    # --- Step 1: Generate observations ---
    print("\n[1] Generating observations...")
    mesh_obs, bc_obs = create_mesh_and_bc(DISPLACEMENTS[0])
    prob_obs = InversionDruckerPrager(mesh_obs, vec=3, dim=3, dirichlet_bc_info=bc_obs)
    obs_multistep, _ = generate_multistep_observation(
        prob_obs, bc_obs, DISPLACEMENTS, E_TRUE, K_TRUE, SOLVER_OPTIONS,
    )
    print(f"  Multi-step observation (final σ_zz): {obs_multistep:.6f} MPa")

    # Single-step observation at the plastic displacement (no history)
    mesh_ss, bc_ss = create_mesh_and_bc(DISPLACEMENTS[1])
    prob_ss = InversionDruckerPrager(mesh_ss, vec=3, dim=3, dirichlet_bc_info=bc_ss)
    params_true = np.array([E_TRUE, K_TRUE])
    prob_ss.set_params(params_true)
    sol_ss = solver(prob_ss, solver_options=SOLVER_OPTIONS)[0]
    obs_single = float(volume_avg_sigma_zz(
        prob_ss.fe, sol_ss, prob_ss.sigmas_old, prob_ss.epsilons_old, E_TRUE, K_TRUE,
    ))
    print(f"  Single-step observation (disp={DISPLACEMENTS[1]}, no history): {obs_single:.6f} MPa")

    # Elastic observation at step 1 (for two-stage variant)
    mesh_el, bc_el = create_mesh_and_bc(DISPLACEMENTS[0])
    prob_el = InversionDruckerPrager(mesh_el, vec=3, dim=3, dirichlet_bc_info=bc_el)
    obs_elastic, _ = generate_observation(
        prob_el, bc_el, DISPLACEMENTS[0], E_TRUE, K_TRUE, SOLVER_OPTIONS,
    )
    print(f"  Elastic observation (disp={DISPLACEMENTS[0]}): {obs_elastic:.6f} MPa")

    # --- Step 2: Run inversions ---
    print("\n[2] Running inversions...")

    # 2a: Single-step (plastic, no history)
    result_single = run_inversion(
        "Single-step (plastic, no history)",
        [DISPLACEMENTS[1]],
        obs_single,
        E_INIT, K_INIT,
        num_iters=50, lr=300.0,
    )

    # 2b: Multi-step (full path with history, last-step AD)
    result_multi = run_inversion(
        "Multi-step (elastic → plastic → unload)",
        DISPLACEMENTS,
        obs_multistep,
        E_INIT, K_INIT,
        num_iters=50, lr=300.0,
    )

    # 2c: Two-stage multi-step (elastic → E, then multi-step → k)
    result_twostage = run_twostage_multistep_inversion(
        obs_elastic, obs_multistep, DISPLACEMENTS,
        E_INIT, K_INIT,
    )

    # --- Step 3: Plotting ---
    print("\n[3] Generating comparison plots...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 3a: Loss convergence comparison
    ax = axes[0]
    ax.semilogy(result_single['history']['loss'], 'b-', linewidth=1.5, label='Single-step')
    ax.semilogy(result_multi['history']['loss'], 'r-', linewidth=1.5, label='Multi-step')
    ax.semilogy(result_twostage['history']['loss'], 'g-', linewidth=1.5, label='Two-stage multi')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Convergence Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3b: E convergence
    ax = axes[1]
    params_s = onp.array(result_single['history']['params'])
    params_m = onp.array(result_multi['history']['params'])
    params_t = onp.array(result_twostage['history']['params'])
    ax.plot(params_s[:, 0], 'b-', linewidth=1.5, label='Single-step')
    ax.plot(params_m[:, 0], 'r-', linewidth=1.5, label='Multi-step')
    ax.plot(params_t[:, 0], 'g-', linewidth=1.5, label='Two-stage multi')
    ax.axhline(E_TRUE, color='k', linestyle='--', alpha=0.5, label=f'E_true={E_TRUE}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E [MPa]')
    ax.set_title('E Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3c: k convergence
    ax = axes[2]
    ax.plot(params_s[:, 1], 'b-', linewidth=1.5, label='Single-step')
    ax.plot(params_m[:, 1], 'r-', linewidth=1.5, label='Multi-step')
    ax.plot(params_t[:, 1], 'g-', linewidth=1.5, label='Two-stage multi')
    ax.axhline(K_TRUE, color='k', linestyle='--', alpha=0.5, label=f'k_true={K_TRUE}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('k [MPa]')
    ax.set_title('k Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved: {plot_path}")

    # --- Step 4: Save results ---
    results = {
        'E_true': E_TRUE, 'k_true': K_TRUE,
        'E_init': E_INIT, 'k_init': K_INIT,
        'displacements': DISPLACEMENTS,
        'single_step': {
            'obs': obs_single,
            'E_final': result_single['E_final'],
            'k_final': result_single['k_final'],
            'err_E': result_single['err_E'],
            'err_k': result_single['err_k'],
            'time_s': result_single['time_s'],
        },
        'multi_step': {
            'obs': obs_multistep,
            'E_final': result_multi['E_final'],
            'k_final': result_multi['k_final'],
            'err_E': result_multi['err_E'],
            'err_k': result_multi['err_k'],
            'time_s': result_multi['time_s'],
        },
        'twostage_multistep': {
            'obs_elastic': obs_elastic,
            'obs_multistep': obs_multistep,
            'E_final': result_twostage['E_final'],
            'k_final': result_twostage['k_final'],
            'err_E': result_twostage['err_E'],
            'err_k': result_twostage['err_k'],
            'time_s': result_twostage['time_s'],
        },
    }
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {json_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("MULTI-STEP INVERSION COMPARISON")
    print("=" * 70)
    print(f"  {'':20s} {'Single-step':>15s}  {'Multi-step':>15s}  {'Two-stage':>15s}")
    print(f"  {'E final':20s} {result_single['E_final']:>15.2f}"
          f"  {result_multi['E_final']:>15.2f}"
          f"  {result_twostage['E_final']:>15.2f}")
    print(f"  {'k final':20s} {result_single['k_final']:>15.4f}"
          f"  {result_multi['k_final']:>15.4f}"
          f"  {result_twostage['k_final']:>15.4f}")
    print(f"  {'E error':20s} {result_single['err_E']:>14.4%}"
          f"  {result_multi['err_E']:>14.4%}"
          f"  {result_twostage['err_E']:>14.4%}")
    print(f"  {'k error':20s} {result_single['err_k']:>14.4%}"
          f"  {result_multi['err_k']:>14.4%}"
          f"  {result_twostage['err_k']:>14.4%}")
    print(f"  {'Time (s)':20s} {result_single['time_s']:>15.1f}"
          f"  {result_multi['time_s']:>15.1f}"
          f"  {result_twostage['time_s']:>15.1f}")

    pass_single_E = result_single['err_E'] < 0.05
    pass_single_k = result_single['err_k'] < 0.05
    pass_multi_E = result_multi['err_E'] < 0.05
    pass_multi_k = result_multi['err_k'] < 0.05
    pass_ts_E = result_twostage['err_E'] < 0.05
    pass_ts_k = result_twostage['err_k'] < 0.05

    print(f"\n  Single-step:      E {'PASS' if pass_single_E else 'FAIL'}, "
          f"k {'PASS' if pass_single_k else 'FAIL'}")
    print(f"  Multi-step:       E {'PASS' if pass_multi_E else 'FAIL'}, "
          f"k {'PASS' if pass_multi_k else 'FAIL'}")
    print(f"  Two-stage multi:  E {'PASS' if pass_ts_E else 'FAIL'}, "
          f"k {'PASS' if pass_ts_k else 'FAIL'}")

    # Two-stage should resolve both params; others may be under-determined
    overall = pass_ts_E and pass_ts_k
    print(f"\n  OVERALL (two-stage both <5%): {'PASS' if overall else 'FAIL'}")
    print("=" * 70)

    if not (pass_single_E and pass_single_k and pass_multi_E and pass_multi_k):
        print("\n  NOTE: Single-observation inversions (single-step, multi-step)")
        print("  with 2 params may have non-unique solutions (under-determined).")
        print("  Two-stage decouples E and k using elastic + plastic observations.")

    return results


if __name__ == "__main__":
    main()
