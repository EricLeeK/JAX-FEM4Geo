#!/usr/bin/env python
"""
F4: Parameter Scaling — AD vs FD gradient cost for 1, 2, 3, 4 parameters.

Uses a local 4-parameter DP variant (E, k, nu, alpha all via internal_vars)
to demonstrate that AD cost is O(1) w.r.t. #params while FD is O(N).

Does NOT modify the main model in src/models/ — keeps a self-contained
experiment-local Problem subclass.
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
    _safe_divide, stress_return_dp, create_mesh_and_bc,
    benchmark, RESULTS_DIR,
)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper, solver

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(RESULTS_DIR, 'f4_scaling')
os.makedirs(OUT_DIR, exist_ok=True)

DISP = -0.028
N_REPEAT = 5


def stress_return_dp_4p(u_grad, sigma_old, epsilon_old, E, k, nu, alpha, dim):
    """4-parameter DP return map: E, k, nu, alpha all differentiable."""
    a = 0.1 * k
    mu = E / (2. * (1. + nu))
    lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
    bulk_k = lmbda + 2. * mu / 3.

    epsilon_crt = 0.5 * (u_grad + u_grad.T)
    epsilon_inc = epsilon_crt - epsilon_old
    sigma_trial = (lmbda * np.trace(epsilon_inc) * np.eye(dim)
                   + 2. * mu * epsilon_inc + sigma_old)

    I1 = np.trace(sigma_trial)
    s_dev = sigma_trial - (I1 / 3.) * np.eye(dim)
    J2 = 0.5 * np.sum(s_dev * s_dev)

    sqrt_J2_reg = np.sqrt(J2 + a * a)
    f_yield = sqrt_J2_reg + alpha * I1 - k
    f_yield_plus = np.where(f_yield > 0., f_yield, 0.)

    n_dev = _safe_divide(s_dev, sqrt_J2_reg)
    denom = mu + 9. * bulk_k * alpha * alpha
    delta_lambda = _safe_divide(f_yield_plus, denom)
    sigma = sigma_trial - delta_lambda * (
        mu * n_dev + 3. * bulk_k * alpha * np.eye(dim)
    )

    sigma_apex = (k / (3. * alpha)) * np.eye(dim)
    at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
    sigma = np.where(at_apex, sigma_apex, sigma)
    return sigma


class FourParamDP(Problem):
    """DP with E, k, nu, alpha all in internal_vars for full AD."""

    def custom_init(self):
        self.fe = self.fes[0]
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.epsilons_old = np.zeros((nc, nq, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [
            self.sigmas_old, self.epsilons_old,
            np.full((nc, nq, 1), 70000.0),  # E
            np.full((nc, nq, 1), 50.0),     # k
            np.full((nc, nq, 1), 0.3),      # nu
            np.full((nc, nq, 1), 0.3),      # alpha
        ]

    def set_params(self, params):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        # params can be length 1, 2, 3, or 4
        n = len(params)
        if n >= 1:
            self.internal_vars[2] = np.full((nc, nq, 1), params[0])  # E
        if n >= 2:
            self.internal_vars[3] = np.full((nc, nq, 1), params[1])  # k
        if n >= 3:
            self.internal_vars[4] = np.full((nc, nq, 1), params[2])  # nu
        if n >= 4:
            self.internal_vars[5] = np.full((nc, nq, 1), params[3])  # alpha

    def get_tensor_map(self):
        dim = self.dim

        def stress_fn(u_grad, sigma_old, epsilon_old, E_arr, k_arr, nu_arr, alpha_arr):
            return stress_return_dp_4p(
                u_grad, sigma_old, epsilon_old,
                E_arr[0], k_arr[0], nu_arr[0], alpha_arr[0], dim,
            )

        return stress_fn


def make_loss(fwd, problem, n_params, true_params):
    """Build loss function for n_params (1-4), fixing the rest at true values."""
    full_true = np.array(true_params)

    def loss_fn(params):
        # Pad params to 4 by filling with true values
        padded = full_true.at[:len(params)].set(params)
        sol = fwd(padded)[0]
        # Simple loss: volume-avg sigma_zz
        u_grads = problem.fe.sol_to_grad(sol)
        c, q, _, dim = u_grads.shape
        so = problem.sigmas_old
        eo = problem.epsilons_old

        def one_quad(ug, s, e):
            return stress_return_dp_4p(
                ug, s, e, padded[0], padded[1], padded[2], padded[3], dim,
            )

        sig = jax.vmap(jax.vmap(one_quad))(u_grads, so, eo)
        JxW = problem.fe.JxW
        return np.sum(sig[..., 2, 2] * JxW) / np.sum(JxW)

    return loss_fn


def main():
    print("=" * 70)
    print("F4: PARAMETER SCALING — AD vs FD")
    print("=" * 70)

    TRUE_PARAMS = [70000.0, 50.0, 0.3, 0.3]
    PARAM_NAMES = ['E', 'k', 'ν', 'α']
    FD_EPS_MAP = {1: [100.0], 2: [100.0, 1.0], 3: [100.0, 1.0, 0.01], 4: [100.0, 1.0, 0.01, 0.01]}

    results = []

    for n_params in [1, 2, 3, 4]:
        print(f"\n--- {n_params} parameter(s): {PARAM_NAMES[:n_params]} ---")

        mesh, bc = create_mesh_and_bc(DISP)
        problem = FourParamDP(mesh, vec=3, dim=3, dirichlet_bc_info=bc)
        fwd = ad_wrapper(problem, solver_options=SOLVER_OPTIONS,
                         adjoint_solver_options=SOLVER_OPTIONS)

        loss_fn = make_loss(fwd, problem, n_params, TRUE_PARAMS)
        params_test = np.array(TRUE_PARAMS[:n_params])

        # AD timing
        vg = jax.value_and_grad(loss_fn)
        _, t_ad, t_ad_std = benchmark(lambda: vg(params_test), n_warmup=2, n_repeat=N_REPEAT)
        print(f"  t_ad  = {t_ad:.4f} ± {t_ad_std:.4f} s")

        # FD timing
        eps_list = FD_EPS_MAP[n_params]

        def run_fd():
            grad = onp.zeros(n_params)
            for i in range(n_params):
                pp = params_test.at[i].set(params_test[i] + eps_list[i])
                pm = params_test.at[i].set(params_test[i] - eps_list[i])
                grad[i] = (float(loss_fn(pp)) - float(loss_fn(pm))) / (2 * eps_list[i])
            return grad

        _, t_fd, t_fd_std = benchmark(run_fd, n_warmup=2, n_repeat=N_REPEAT)
        print(f"  t_fd  = {t_fd:.4f} ± {t_fd_std:.4f} s  ({2 * n_params} solves)")

        speedup = t_fd / t_ad if t_ad > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}×")

        results.append({
            'n_params': n_params,
            'params': PARAM_NAMES[:n_params],
            't_ad': t_ad, 't_ad_std': t_ad_std,
            't_fd': t_fd, 't_fd_std': t_fd_std,
            'speedup': speedup,
        })

    # --- Plotting ---
    print("\n[Plot] Generating scaling plot...")
    fig, ax = plt.subplots(figsize=(8, 5))

    ns = [r['n_params'] for r in results]
    t_ads = [r['t_ad'] for r in results]
    t_fds = [r['t_fd'] for r in results]
    t_ad_stds = [r['t_ad_std'] for r in results]
    t_fd_stds = [r['t_fd_std'] for r in results]

    ax.errorbar(ns, t_ads, yerr=t_ad_stds, fmt='b-s', markersize=10,
                linewidth=2, capsize=5, label='AD (adjoint)')
    ax.errorbar(ns, t_fds, yerr=t_fd_stds, fmt='r-^', markersize=10,
                linewidth=2, capsize=5, label='FD (central diff)')

    ax.set_xlabel('Number of parameters', fontsize=12)
    ax.set_ylabel('Gradient computation time [s]', fontsize=12)
    ax.set_title('F4: AD vs FD Scaling with Parameter Count', fontsize=13)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n}\n({', '.join(PARAM_NAMES[:n])})" for n in ns], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate speedups
    for r in results:
        ax.annotate(f"{r['speedup']:.1f}×",
                    xy=(r['n_params'], r['t_fd']),
                    xytext=(r['n_params'] + 0.15, r['t_fd']),
                    fontsize=10, color='darkred')

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'scaling.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")

    # Save
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("F4 SUMMARY")
    print("=" * 70)
    print(f"  {'N':>3s}  {'AD [s]':>10s}  {'FD [s]':>10s}  {'Speedup':>8s}")
    for r in results:
        print(f"  {r['n_params']:>3d}  {r['t_ad']:>10.4f}  {r['t_fd']:>10.4f}  {r['speedup']:>7.2f}×")
    print("=" * 70)


if __name__ == "__main__":
    main()
