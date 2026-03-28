#!/usr/bin/env python
"""
E1: Displacement Field Inversion

Compares two loss functions for parameter inversion:
  - L_sigma: (σ_zz_pred - σ_zz_obs)²  (scalar, used in D1)
  - L_disp:  ||u_pred - u_obs||²        (full displacement field)

Key questions:
  1. Does displacement-field loss improve convergence?
  2. Does it improve the condition number (D2 found 3.2M for σ_zz loss)?
  3. Can joint (E, k) optimization work directly (without two-stage)?

The displacement field contains spatial information across all DOFs,
providing much richer gradients than a single scalar σ_zz.
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
    InversionDruckerPrager, create_mesh_and_bc, update_bc,
    volume_avg_sigma_zz, generate_observation, adam_optimize,
    RESULTS_DIR,
)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper, solver

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(RESULTS_DIR, 'e1_displacement_inversion')
os.makedirs(OUT_DIR, exist_ok=True)

E_TRUE, K_TRUE = 70000.0, 50.0
E_INIT, K_INIT = 60000.0, 45.0
DISP_PLASTIC = -0.028


def main():
    print("=" * 70)
    print("E1: DISPLACEMENT FIELD INVERSION")
    print("=" * 70)
    t0_total = time.time()

    # --- Generate observation ---
    print("\n[1] Generating observation (displacement field + σ_zz)...")
    mesh_obs, bc_obs = create_mesh_and_bc(DISP_PLASTIC)
    prob_obs = InversionDruckerPrager(mesh_obs, vec=3, dim=3, dirichlet_bc_info=bc_obs)
    params_true = np.array([E_TRUE, K_TRUE])
    prob_obs.set_params(params_true)
    sol_obs_list = solver(prob_obs, solver_options=SOLVER_OPTIONS)
    u_obs = sol_obs_list[0]  # (num_nodes, 3)
    sigma_zz_obs = float(volume_avg_sigma_zz(
        prob_obs.fe, u_obs, prob_obs.sigmas_old, prob_obs.epsilons_old, E_TRUE, K_TRUE,
    ))

    n_nodes = u_obs.shape[0]
    n_dofs = n_nodes * 3
    print(f"  Observation: {n_nodes} nodes, {n_dofs} DOFs")
    print(f"  σ_zz_obs = {sigma_zz_obs:.6f} MPa")
    print(f"  ||u_obs|| = {float(np.linalg.norm(u_obs)):.6e}")

    # --- Setup problems for inversion ---
    print("\n[2] Setting up AD-wrapped problems...")

    # Problem for displacement loss
    mesh_d, bc_d = create_mesh_and_bc(DISP_PLASTIC)
    prob_disp = InversionDruckerPrager(mesh_d, vec=3, dim=3, dirichlet_bc_info=bc_d)
    fwd_disp = ad_wrapper(prob_disp, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    # Problem for sigma loss
    mesh_s, bc_s = create_mesh_and_bc(DISP_PLASTIC)
    prob_sigma = InversionDruckerPrager(mesh_s, vec=3, dim=3, dirichlet_bc_info=bc_s)
    fwd_sigma = ad_wrapper(prob_sigma, solver_options=SOLVER_OPTIONS,
                           adjoint_solver_options=SOLVER_OPTIONS)

    # --- Loss functions ---
    def loss_disp(params):
        """Displacement field MSE loss."""
        sol = fwd_disp(params)[0]
        diff = sol - u_obs
        return np.sum(diff ** 2) / n_dofs

    def loss_sigma(params):
        """Scalar σ_zz loss (same as D1)."""
        E, k = params[0], params[1]
        sol = fwd_sigma(params)[0]
        pred = volume_avg_sigma_zz(
            prob_sigma.fe, sol, prob_sigma.sigmas_old, prob_sigma.epsilons_old, E, k,
        )
        return (pred - sigma_zz_obs) ** 2

    vg_disp_raw = jax.value_and_grad(loss_disp)
    vg_sigma = jax.value_and_grad(loss_sigma)

    # Wrap vg_disp to handle solver failures gracefully
    _last_good_loss = [1e10]

    def vg_disp(params):
        try:
            l, g = vg_disp_raw(params)
            _last_good_loss[0] = float(l)
            return l, g
        except Exception:
            # Return large loss + zero grad so optimizer backs off
            return np.array(_last_good_loss[0] * 10), np.zeros_like(params)

    # Warm up
    print("\n[3] Warming up...")
    params_init = np.array([E_INIT, K_INIT])
    _ = vg_disp(params_init)
    _ = vg_sigma(params_init)

    # --- Compare gradients at initial point ---
    print("\n[4] Gradient comparison at initial point...")
    l_d, g_d = vg_disp(params_init)
    l_s, g_s = vg_sigma(params_init)
    print(f"  Disp loss:  L={float(l_d):.6e}, dL/dE={float(g_d[0]):.6e}, dL/dk={float(g_d[1]):.6e}")
    print(f"  Sigma loss: L={float(l_s):.6e}, dL/dE={float(g_s[0]):.6e}, dL/dk={float(g_s[1]):.6e}")

    # Gradient ratios (measures parameter coupling)
    r_disp = abs(float(g_d[0]) / float(g_d[1])) if abs(float(g_d[1])) > 1e-30 else float('inf')
    r_sigma = abs(float(g_s[0]) / float(g_s[1])) if abs(float(g_s[1])) > 1e-30 else float('inf')
    print(f"  |dL/dE| / |dL/dk| ratio:  disp={r_disp:.2e}, sigma={r_sigma:.2e}")
    print(f"  (Closer to 1 = better conditioned)")

    # --- Hessian comparison (FD) ---
    print("\n[5] Computing Hessian at true params...")

    def fd_hessian(loss_fn, params_center, eps):
        H = onp.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                p_pp = onp.array(params_center); p_pm = onp.array(params_center)
                p_mp = onp.array(params_center); p_mm = onp.array(params_center)
                p_pp[i] += eps[i]; p_pp[j] += eps[j]
                p_pm[i] += eps[i]; p_pm[j] -= eps[j]
                p_mp[i] -= eps[i]; p_mp[j] += eps[j]
                p_mm[i] -= eps[i]; p_mm[j] -= eps[j]
                L_pp = float(loss_fn(np.array(p_pp)))
                L_pm = float(loss_fn(np.array(p_pm)))
                L_mp = float(loss_fn(np.array(p_mp)))
                L_mm = float(loss_fn(np.array(p_mm)))
                H[i, j] = (L_pp - L_pm - L_mp + L_mm) / (4 * eps[i] * eps[j])
        return H

    H_disp = fd_hessian(loss_disp, [E_TRUE, K_TRUE], [50.0, 0.5])
    H_sigma = fd_hessian(loss_sigma, [E_TRUE, K_TRUE], [50.0, 0.5])

    eig_disp = onp.linalg.eigvalsh(H_disp)
    eig_sigma = onp.linalg.eigvalsh(H_sigma)
    cond_disp = max(abs(eig_disp)) / max(min(abs(eig_disp)), 1e-30)
    cond_sigma = max(abs(eig_sigma)) / max(min(abs(eig_sigma)), 1e-30)

    print(f"  Displacement loss:")
    print(f"    Eigenvalues: [{eig_disp[0]:.4e}, {eig_disp[1]:.4e}]")
    print(f"    Condition number: {cond_disp:.2f}")
    print(f"  Sigma loss:")
    print(f"    Eigenvalues: [{eig_sigma[0]:.4e}, {eig_sigma[1]:.4e}]")
    print(f"    Condition number: {cond_sigma:.2f}")
    print(f"  Improvement factor: {cond_sigma / max(cond_disp, 1e-30):.1f}×")

    # --- Joint inversion with displacement loss (no two-stage needed?) ---
    print("\n[6] Joint (E,k) inversion with displacement loss...")

    try:
        from scipy.optimize import minimize as scipy_minimize

        history_disp = {'loss': [], 'params': [], 'wallclock': []}
        t0 = time.time()

        def track_disp(x):
            p = np.array(x)
            l, g = vg_disp(p)
            lv = float(l)
            gn = onp.array(g, dtype=onp.float64)
            history_disp['loss'].append(lv)
            history_disp['params'].append(onp.array(x).copy())
            history_disp['wallclock'].append(time.time() - t0)
            step = len(history_disp['loss'])
            if step <= 5 or step % 5 == 0:
                print(f"    Eval {step}: loss={lv:.6e}  E={x[0]:.1f} k={x[1]:.4f}")
            return lv, gn

        res_disp = scipy_minimize(
            track_disp, x0=onp.array([E_INIT, K_INIT]),
            method='L-BFGS-B', jac=True,
            bounds=[(30000., 120000.), (35., 80.)],
            options={'maxiter': 50, 'ftol': 1e-20, 'gtol': 1e-14},
        )
        t_disp = time.time() - t0
        E_disp, k_disp = res_disp.x
        err_E_disp = abs(E_disp - E_TRUE) / E_TRUE
        err_k_disp = abs(k_disp - K_TRUE) / K_TRUE
        print(f"  Result: E={E_disp:.2f} (err {err_E_disp:.6%}), k={k_disp:.4f} (err {err_k_disp:.6%})")
        print(f"  Time: {t_disp:.2f}s, {res_disp.nit} iters, converged={res_disp.success}")
        has_scipy = True

    except ImportError:
        print("  scipy not available, using Adam...")
        history_disp = {'loss': [], 'params': [], 'wallclock': []}
        t0 = time.time()

        def cb(step, params, loss_val, grad):
            history_disp['wallclock'].append(time.time() - t0)
            if step % 5 == 0:
                print(f"    Step {step}: loss={loss_val:.6e} E={params[0]:.1f} k={params[1]:.3f}")

        pf, history_disp_raw = adam_optimize(
            vg_disp, onp.array([E_INIT, K_INIT]),
            num_iters=60, lr=100., bounds=[(30000, 120000), (35, 80)],
            callback=cb,
        )
        t_disp = time.time() - t0
        E_disp, k_disp = float(pf[0]), float(pf[1])
        err_E_disp = abs(E_disp - E_TRUE) / E_TRUE
        err_k_disp = abs(k_disp - K_TRUE) / K_TRUE
        history_disp = {'loss': history_disp_raw['loss'],
                        'params': history_disp_raw['params'],
                        'wallclock': history_disp['wallclock']}
        has_scipy = False

    # --- Two-stage displacement inversion ---
    print("\n[6b] Two-stage displacement inversion (elastic→E, plastic→k)...")
    from scipy.optimize import minimize_scalar

    disp_2s_ok = True
    history_disp_2s = {'loss': [], 'params': []}
    try:
        # Generate elastic displacement observation
        DISP_ELASTIC = -0.015
        mesh_de, bc_de = create_mesh_and_bc(DISP_ELASTIC)
        prob_disp_e = InversionDruckerPrager(mesh_de, vec=3, dim=3, dirichlet_bc_info=bc_de)
        prob_disp_e.set_params(params_true)
        u_obs_elastic = solver(prob_disp_e, solver_options=SOLVER_OPTIONS)[0]

        # Stage 1: E from elastic displacement observation
        mesh_de2, bc_de2 = create_mesh_and_bc(DISP_ELASTIC)
        prob_de2 = InversionDruckerPrager(mesh_de2, vec=3, dim=3, dirichlet_bc_info=bc_de2)
        fwd_de2 = ad_wrapper(prob_de2, solver_options=SOLVER_OPTIONS,
                             adjoint_solver_options=SOLVER_OPTIONS)

        n_dofs_e = u_obs_elastic.shape[0] * 3

        def loss_disp_E(E_arr):
            E = E_arr[0]
            sol = fwd_de2(np.array([E, 200.0]))[0]
            return np.sum((sol - u_obs_elastic) ** 2) / n_dofs_e

        _ = jax.value_and_grad(loss_disp_E)(np.array([E_INIT]))

        t0_2s = time.time()

        def track_disp_E(E_val):
            l = float(loss_disp_E(np.array([E_val])))
            history_disp_2s['loss'].append(l)
            history_disp_2s['params'].append([E_val, K_INIT])
            return l

        res_disp_E = minimize_scalar(track_disp_E, bounds=(40000, 110000), method='bounded',
                                      options={'xatol': 1.0, 'maxiter': 30})
        E_disp_2s = res_disp_E.x
        print(f"  Stage 1 (disp): E = {E_disp_2s:.2f} ({res_disp_E.nfev} evals)")

        # Stage 2: k from plastic displacement observation (E fixed)
        mesh_dp2, bc_dp2 = create_mesh_and_bc(DISP_PLASTIC)
        prob_dp2 = InversionDruckerPrager(mesh_dp2, vec=3, dim=3, dirichlet_bc_info=bc_dp2)
        fwd_dp2 = ad_wrapper(prob_dp2, solver_options=SOLVER_OPTIONS,
                             adjoint_solver_options=SOLVER_OPTIONS)

        def loss_disp_k(k_arr):
            k = k_arr[0]
            sol = fwd_dp2(np.array([E_disp_2s, k]))[0]
            return np.sum((sol - u_obs) ** 2) / n_dofs

        # Warm up with try/except
        try:
            _ = jax.value_and_grad(loss_disp_k)(np.array([K_INIT]))
        except Exception:
            pass

        def track_disp_k(k_val):
            try:
                l = float(loss_disp_k(np.array([k_val])))
                if onp.isnan(l) or onp.isinf(l):
                    return 1e20
            except Exception:
                return 1e20
            history_disp_2s['loss'].append(l)
            history_disp_2s['params'].append([E_disp_2s, k_val])
            return l

        res_disp_k = minimize_scalar(track_disp_k, bounds=(35, 80), method='bounded',
                                      options={'xatol': 0.1, 'maxiter': 30})
        k_disp_2s = res_disp_k.x
        t_disp_2s = time.time() - t0_2s
        err_E_disp_2s = abs(E_disp_2s - E_TRUE) / E_TRUE
        err_k_disp_2s = abs(k_disp_2s - K_TRUE) / K_TRUE
        print(f"  Stage 2 (disp): k = {k_disp_2s:.4f} ({res_disp_k.nfev} evals)")
        print(f"  Result: E={E_disp_2s:.2f} (err {err_E_disp_2s:.6%}), k={k_disp_2s:.4f} (err {err_k_disp_2s:.6%})")
        print(f"  Time: {t_disp_2s:.2f}s")

    except Exception as ex:
        print(f"  Two-stage displacement inversion failed: {ex}")
        disp_2s_ok = False
        E_disp_2s, k_disp_2s = float('nan'), float('nan')
        err_E_disp_2s, err_k_disp_2s = float('nan'), float('nan')
        t_disp_2s = float('nan')

    # --- Two-stage sigma inversion (D1 baseline) ---
    print("\n[7] Two-stage σ_zz inversion (baseline from D1)...")
    try:
        from scipy.optimize import minimize_scalar, minimize as scipy_minimize

        history_sigma = {'loss': [], 'params': [], 'wallclock': []}
        t0 = time.time()

        # Stage 1: E from elastic observation
        mesh_e, bc_e = create_mesh_and_bc(-0.015)
        prob_e = InversionDruckerPrager(mesh_e, vec=3, dim=3, dirichlet_bc_info=bc_e)
        fwd_e = ad_wrapper(prob_e, solver_options=SOLVER_OPTIONS, adjoint_solver_options=SOLVER_OPTIONS)

        # Get elastic observation
        prob_e.set_params(params_true)
        sol_e_obs = solver(prob_e, solver_options=SOLVER_OPTIONS)[0]
        sigma_e_obs = float(volume_avg_sigma_zz(
            prob_e.fe, sol_e_obs, prob_e.sigmas_old, prob_e.epsilons_old, E_TRUE, K_TRUE,
        ))

        def loss_E_only(params_E):
            E = params_E[0]; k_dum = 200.0
            sol = fwd_e(np.array([E, k_dum]))[0]
            pred = volume_avg_sigma_zz(prob_e.fe, sol, prob_e.sigmas_old, prob_e.epsilons_old, E, k_dum)
            return (pred - sigma_e_obs) ** 2

        def track_E(E_val):
            l = float(loss_E_only(np.array([E_val])))
            history_sigma['loss'].append(l)
            history_sigma['params'].append(onp.array([E_val, K_INIT]))
            history_sigma['wallclock'].append(time.time() - t0)
            return l

        # Warm up
        _ = jax.value_and_grad(loss_E_only)(np.array([E_INIT]))

        res_E = minimize_scalar(track_E, bounds=(30000, 120000), method='bounded',
                                options={'xatol': 1.0, 'maxiter': 30})
        E_s1 = res_E.x

        # Stage 2: k
        def loss_k_only(params_k):
            k = params_k[0]
            sol = fwd_sigma(np.array([E_s1, k]))[0]
            pred = volume_avg_sigma_zz(prob_sigma.fe, sol, prob_sigma.sigmas_old, prob_sigma.epsilons_old, E_s1, k)
            return (pred - sigma_zz_obs) ** 2

        _ = jax.value_and_grad(loss_k_only)(np.array([K_INIT]))

        def track_k(k_val):
            l = float(loss_k_only(np.array([k_val])))
            history_sigma['loss'].append(l)
            history_sigma['params'].append(onp.array([E_s1, k_val]))
            history_sigma['wallclock'].append(time.time() - t0)
            return l

        res_k = minimize_scalar(track_k, bounds=(35, 80), method='bounded',
                                options={'xatol': 0.1, 'maxiter': 30})
        k_s2 = res_k.x
        t_sigma = time.time() - t0
        err_E_sigma = abs(E_s1 - E_TRUE) / E_TRUE
        err_k_sigma = abs(k_s2 - K_TRUE) / K_TRUE
        print(f"  Result: E={E_s1:.2f} (err {err_E_sigma:.6%}), k={k_s2:.4f} (err {err_k_sigma:.6%})")
        print(f"  Time: {t_sigma:.2f}s")

    except ImportError:
        t_sigma = float('nan')
        err_E_sigma = float('nan')
        err_k_sigma = float('nan')

    # --- Plotting ---
    print("\n[8] Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss convergence
    ax = axes[0, 0]
    if history_disp['loss']:
        ax.semilogy(history_disp['wallclock'], history_disp['loss'],
                     'b-o', markersize=3, label='Disp (joint)', linewidth=1.5)
    if history_sigma['loss']:
        ax.semilogy(history_sigma['wallclock'], history_sigma['loss'],
                     'r-s', markersize=3, label='σ_zz (2-stage)', linewidth=1.5)
    ax.set_xlabel('Wall-clock time [s]')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence: Loss vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Parameter trajectory
    ax = axes[0, 1]
    if history_disp['params']:
        pd = onp.array(history_disp['params'])
        ax.plot(pd[:, 0], pd[:, 1], 'b.-', markersize=4, linewidth=0.8, label='Disp (joint)')
    if history_sigma['params']:
        ps = onp.array(history_sigma['params'])
        ax.plot(ps[:, 0], ps[:, 1], 'r.-', markersize=4, linewidth=0.8, label='σ_zz (2-stage)')
    if history_disp_2s['params']:
        pd2 = onp.array(history_disp_2s['params'])
        ax.plot(pd2[:, 0], pd2[:, 1], 'g.-', markersize=4, linewidth=0.8, label='Disp (2-stage)')
    ax.plot(E_INIT, K_INIT, 'ks', markersize=10, label='Start')
    ax.plot(E_TRUE, K_TRUE, 'r*', markersize=15, label='True')
    ax.set_xlabel('E [MPa]')
    ax.set_ylabel('k [MPa]')
    ax.set_title('Parameter Space Trajectories')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Condition number comparison (bar chart)
    ax = axes[1, 0]
    labels = ['σ_zz loss', 'Disp loss']
    conds = [cond_sigma, cond_disp]
    colors = ['red', 'blue']
    bars = ax.bar(labels, conds, color=colors, alpha=0.7)
    ax.set_ylabel('Condition number')
    ax.set_title('Hessian Condition Number')
    ax.set_yscale('log')
    for bar, c in zip(bars, conds):
        ax.text(bar.get_x() + bar.get_width() / 2, c * 1.5,
                f'{c:.1e}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Error comparison bar chart
    ax = axes[1, 1]
    strategies = ['Disp\n(joint)', 'Disp\n(2-stage)', 'σ_zz\n(2-stage)']
    errs_E = [err_E_disp, err_E_disp_2s, err_E_sigma]
    errs_k = [err_k_disp, err_k_disp_2s, err_k_sigma]
    x = onp.arange(len(strategies))
    w = 0.35
    ax.bar(x - w/2, errs_E, w, label='E error', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, errs_k, w, label='k error', color='coral', alpha=0.8)
    ax.axhline(0.01, color='k', linestyle='--', alpha=0.5, label='1% threshold')
    ax.set_ylabel('Relative error')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_title('Parameter Recovery Errors')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('E1: Displacement Field vs Stress Scalar Inversion', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")

    # --- Save ---
    t_total = time.time() - t0_total
    results = {
        'disp_joint': {
            'E_final': float(E_disp), 'k_final': float(k_disp),
            'err_E': float(err_E_disp), 'err_k': float(err_k_disp),
            'time_s': t_disp,
            'hessian_eigenvalues': eig_disp.tolist(),
            'condition_number': float(cond_disp),
            'strategy': 'joint L-BFGS-B',
        },
        'disp_two_stage': {
            'E_final': float(E_disp_2s), 'k_final': float(k_disp_2s),
            'err_E': float(err_E_disp_2s), 'err_k': float(err_k_disp_2s),
            'time_s': t_disp_2s,
            'strategy': 'two-stage displacement (elastic→E, plastic→k)',
        },
        'sigma_two_stage': {
            'E_final': float(E_s1) if not onp.isnan(err_E_sigma) else None,
            'k_final': float(k_s2) if not onp.isnan(err_k_sigma) else None,
            'err_E': float(err_E_sigma),
            'err_k': float(err_k_sigma),
            'time_s': float(t_sigma),
            'hessian_eigenvalues': eig_sigma.tolist(),
            'condition_number': float(cond_sigma),
            'strategy': 'two-stage σ_zz (elastic→E, plastic→k)',
        },
        'condition_improvement': float(cond_sigma / max(cond_disp, 1e-30)),
        'total_time_s': t_total,
    }
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("E1 SUMMARY")
    print("=" * 70)
    print(f"  {'':25s} {'Disp(joint)':>14s}  {'Disp(2-stg)':>14s}  {'σ_zz(2-stg)':>14s}")
    print(f"  {'E error':25s} {err_E_disp:>13.6%}  {err_E_disp_2s:>13.6%}  {err_E_sigma:>13.6%}")
    print(f"  {'k error':25s} {err_k_disp:>13.6%}  {err_k_disp_2s:>13.6%}  {err_k_sigma:>13.6%}")
    print(f"  {'Time':25s} {t_disp:>13.2f}s  {t_disp_2s:>13.2f}s  {t_sigma:>13.2f}s")
    print(f"  {'Condition number':25s} {cond_disp:>13.1f}  {'N/A':>14s}  {cond_sigma:>13.1f}")

    joint_ok = err_E_disp < 0.01 and err_k_disp < 0.01
    disp2s_ok = disp_2s_ok and err_E_disp_2s < 0.01 and err_k_disp_2s < 0.01
    print(f"\n  Joint disp optimization works: {'YES' if joint_ok else 'NO'}")
    print(f"  Two-stage disp optimization works: {'YES' if disp2s_ok else 'NO'}")
    if not joint_ok and eig_disp[0] < 0:
        print(f"  Note: Joint disp fails due to saddle point (negative Hessian eigenvalue = {eig_disp[0]:.4e})")
    print("=" * 70)


if __name__ == "__main__":
    main()
