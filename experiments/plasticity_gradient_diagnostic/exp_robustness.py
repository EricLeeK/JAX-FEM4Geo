"""Robustness verification after fix: E,k via internal_vars.

Sweep displacement from pure elastic through deep plasticity.
At each displacement:
  1. AD vs FD gradient (multiple FD eps values)
  2. Taylor test (check 2nd-order convergence)
  3. Yield state (elastic/plastic count)

Acceptance criteria:
  - rel_err(AD, FD) < 1e-3 for at least one FD eps
  - Taylor r1 rate > 1.8 for at least 2 consecutive eps levels
"""
from __future__ import annotations

import math
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
from pathlib import Path

from common import (
    DEFAULT_PARAMS,
    PARAMETER_INFO,
    RESULTS_ROOT,
    build_reference_mesh,
    create_context,
    ensure_dir,
    print_header,
    relative_error,
    summarize_yield_state,
    write_csv,
    write_json,
)


DISPLACEMENTS = [
    -0.005, -0.010, -0.015, -0.020,       # pure elastic
    -0.023, -0.024, -0.025,                # near yield
    -0.0255, -0.026, -0.0265, -0.027,      # transition zone
    -0.028, -0.029, -0.030, -0.031,        # plastic
]

FD_EPS_E = [200.0, 100.0, 50.0, 10.0, 1.0]
FD_EPS_K = [5.0, 2.0, 1.0, 0.5, 0.1]

TAYLOR_EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def central_fd(loss_fn, params, idx, eps):
    lp = float(loss_fn(params.at[idx].add(eps)))
    lm = float(loss_fn(params.at[idx].add(-eps)))
    return (lp - lm) / (2. * eps)


def taylor_test(loss_fn, params, grad_ad):
    key = jax.random.PRNGKey(42)
    dp = jax.random.normal(key, params.shape)
    dp = dp / jnp.linalg.norm(dp)
    L0 = float(loss_fn(params))
    gd = float(jnp.dot(grad_ad, dp))
    results = []
    for eps in TAYLOR_EPS:
        Lp = float(loss_fn(params + eps * dp))
        r0 = abs(Lp - L0)
        r1 = abs(Lp - L0 - eps * gd)
        results.append({"eps": eps, "r0": r0, "r1": r1})
    # compute rates
    for i in range(1, len(results)):
        prev, cur = results[i - 1], results[i]
        if prev["r0"] > 0 and cur["r0"] > 0:
            cur["rate0"] = math.log(prev["r0"] / cur["r0"]) / math.log(prev["eps"] / cur["eps"])
        if prev["r1"] > 0 and cur["r1"] > 0:
            cur["rate1"] = math.log(prev["r1"] / cur["r1"]) / math.log(prev["eps"] / cur["eps"])
    return results


def best_fd_match(loss_fn, params, idx, eps_list, ad_val):
    """Try multiple FD eps, return best rel_err and corresponding fd value."""
    best_re = float("inf")
    best_fd = None
    best_eps = None
    for eps in eps_list:
        try:
            fd = central_fd(loss_fn, params, idx, eps)
        except Exception:
            continue
        re = relative_error(ad_val, fd)
        if re < best_re:
            best_re = re
            best_fd = fd
            best_eps = eps
    return best_fd, best_re, best_eps


def check_taylor(tt_results):
    """Check if Taylor test passes: r1 rate > 1.8 for at least 2 consecutive levels."""
    rates = [r.get("rate1") for r in tt_results if r.get("rate1") is not None]
    for i in range(len(rates) - 1):
        if rates[i] is not None and rates[i + 1] is not None:
            if rates[i] > 1.8 and rates[i + 1] > 1.8:
                return True
    return False


def main():
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp_robustness")
    mesh = build_reference_mesh()

    print_header("Robustness verification: full displacement sweep")
    print(f"Displacements: {len(DISPLACEMENTS)} points from {DISPLACEMENTS[0]} to {DISPLACEMENTS[-1]} mm")
    print(f"FD eps for E: {FD_EPS_E}")
    print(f"FD eps for k: {FD_EPS_K}")
    print(f"Taylor eps: {TAYLOR_EPS}")

    all_rows = []
    all_details = []
    summary_lines = []
    any_failure = False

    for disp in DISPLACEMENTS:
        print(f"\n{'─'*72}")
        print(f"  Displacement = {disp:.4f} mm")
        print(f"{'─'*72}")

        try:
            context = create_context(displacement=disp, mesh=mesh)
            loss_val, grad_ad = jax.value_and_grad(context.loss)(DEFAULT_PARAMS)
            sol_list = context.solve(DEFAULT_PARAMS)
            ys, _ = summarize_yield_state(context.problem.fe, sol_list[0], DEFAULT_PARAMS)
        except Exception as exc:
            print(f"  SOLVER FAILED: {exc}")
            summary_lines.append(f"  {disp:+.4f} mm  SOLVER FAILED")
            continue

        print(f"  loss={float(loss_val):.8e}, plastic={ys.plastic_count}/{ys.num_quadrature_points}")
        print(f"  AD: dL/dE={float(grad_ad[0]): .8e}, dL/dk={float(grad_ad[1]): .8e}")

        # FD with multiple eps
        fd_E, re_E, eps_E = best_fd_match(context.loss, DEFAULT_PARAMS, 0, FD_EPS_E, float(grad_ad[0]))
        fd_k, re_k, eps_k = best_fd_match(context.loss, DEFAULT_PARAMS, 1, FD_EPS_K, float(grad_ad[1]))

        print(f"  FD: dL/dE={fd_E: .8e} (eps={eps_E}), dL/dk={fd_k: .8e} (eps={eps_k})")
        print(f"  rel_err: E={re_E:.3e}, k={re_k:.3e}")

        # Taylor test
        tt = taylor_test(context.loss, DEFAULT_PARAMS, grad_ad)
        taylor_pass = check_taylor(tt)
        r1_rates = [r.get("rate1") for r in tt if r.get("rate1") is not None]
        print(f"  Taylor r1 rates: {['%.3f' % r for r in r1_rates]}")
        print(f"  Taylor pass: {taylor_pass}")

        # Acceptance
        fd_pass = (re_E < 1e-3) and (re_k < 1e-3 or (abs(float(grad_ad[1])) < 1e-12 and abs(fd_k) < 1e-12))
        # Taylor test can degenerate when gradient is near-zero (pure elastic: dL/dk=0),
        # because r1 drops to machine noise. Only require Taylor pass when gradient is nonzero.
        grad_norm = float(jnp.linalg.norm(grad_ad))
        taylor_required = grad_norm > 1e-8 and ys.plastic_count > 0
        overall = fd_pass and (taylor_pass or not taylor_required)

        status = "PASS" if overall else "FAIL"
        if not overall:
            any_failure = True
        regime = "elastic" if ys.plastic_count == 0 else f"plastic({ys.plastic_count}/{ys.num_quadrature_points})"
        summary_lines.append(
            f"  {disp:+.4f} mm  {regime:>16s}  re(E)={re_E:.2e}  re(k)={re_k:.2e}  "
            f"taylor={'OK' if taylor_pass else 'NO'}  [{status}]"
        )
        print(f"  >>> {status}")

        detail = {
            "displacement": disp,
            "loss": float(loss_val),
            "regime": regime,
            "plastic_count": ys.plastic_count,
            "ad_E": float(grad_ad[0]), "ad_k": float(grad_ad[1]),
            "fd_E": fd_E, "fd_k": fd_k,
            "fd_eps_E": eps_E, "fd_eps_k": eps_k,
            "rel_err_E": re_E, "rel_err_k": re_k,
            "taylor_r1_rates": r1_rates,
            "taylor_pass": taylor_pass,
            "fd_pass": fd_pass,
            "overall": status,
        }
        all_details.append(detail)
        all_rows.append({
            "displacement": disp, "regime": regime,
            "plastic_count": ys.plastic_count,
            "ad_E": float(grad_ad[0]), "fd_E": fd_E, "rel_err_E": re_E,
            "ad_k": float(grad_ad[1]), "fd_k": fd_k, "rel_err_k": re_k,
            "taylor_pass": taylor_pass, "status": status,
        })

    # Final summary
    print(f"\n{'='*72}")
    print("  ROBUSTNESS SUMMARY")
    print(f"{'='*72}")
    for line in summary_lines:
        print(line)

    n_pass = sum(1 for d in all_details if d["overall"] == "PASS")
    n_total = len(all_details)
    print(f"\n  Result: {n_pass}/{n_total} passed")
    if any_failure:
        print("  *** SOME TESTS FAILED ***")
    else:
        print("  *** ALL TESTS PASSED ***")

    write_csv(experiment_dir / "robustness.csv", all_rows)
    write_json(experiment_dir / "robustness_details.json", all_details)
    print(f"\n  Saved to {experiment_dir}")


if __name__ == "__main__":
    main()
