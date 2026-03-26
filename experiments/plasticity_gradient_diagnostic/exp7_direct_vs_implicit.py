"""Experiment 7: Direct AD (bypassing ad_wrapper) vs implicit VJP vs FD.

This experiment tests the hypothesis that the problem is in how ad_wrapper/implicit_vjp
handles the parameter dependency. We compute gradients three ways:

1. FD (ground truth)
2. AD via ad_wrapper (implicit VJP) — the current broken path
3. AD via direct JAX differentiation (no custom VJP, no solver loop) —
   we take the converged solution, assume it's fixed, and differentiate
   only the loss function (volume_avg_sigma_zz) directly w.r.t. params.

If (3) matches FD but (2) doesn't, the problem is definitively in the
implicit differentiation chain.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from common import (
    DEFAULT_DISPLACEMENT,
    DEFAULT_PARAMS,
    PARAMETER_INFO,
    RESULTS_ROOT,
    build_reference_mesh,
    central_difference,
    create_context,
    ensure_dir,
    print_header,
    relative_error,
    volume_avg_sigma_zz,
    write_json,
)


def main() -> None:
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp7_direct_vs_implicit")
    mesh = build_reference_mesh()
    context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh)

    print_header("Experiment 7: Direct AD vs Implicit VJP vs FD")

    # ---- Path A: AD via ad_wrapper (implicit VJP) ----
    loss_val_implicit, grad_implicit = jax.value_and_grad(context.loss)(DEFAULT_PARAMS)
    print(f"\n[A] Implicit VJP (ad_wrapper):")
    print(f"  loss = {float(loss_val_implicit):.10e}")
    print(f"  dL/dE = {float(grad_implicit[0]):.10e}")
    print(f"  dL/dk = {float(grad_implicit[1]):.10e}")

    # ---- Path B: Solve once (no AD), then differentiate loss directly ----
    # Get the converged FEM solution
    sol_list = context.solve(DEFAULT_PARAMS)
    sol_fixed = jax.lax.stop_gradient(sol_list[0])

    def direct_loss(params):
        """Loss that only differentiates the stress computation, not the solver."""
        return volume_avg_sigma_zz(
            context.problem.fe,
            sol_fixed,
            params[0],
            params[1],
            yield_beta=context.yield_beta,
            nu=context.problem.nu,
            alpha=context.problem.alpha,
            a_ratio=context.problem.a_ratio,
        )

    loss_val_direct, grad_direct = jax.value_and_grad(direct_loss)(DEFAULT_PARAMS)
    print(f"\n[B] Direct AD (fixed sol, no solver):")
    print(f"  loss = {float(loss_val_direct):.10e}")
    print(f"  dL/dE = {float(grad_direct[0]):.10e}")
    print(f"  dL/dk = {float(grad_direct[1]):.10e}")

    # ---- Path C: FD (ground truth) ----
    print(f"\n[C] Central finite difference:")
    fd_results = {}
    for param in PARAMETER_INFO:
        # FD on the full loss (including solve)
        grad_fd_full = central_difference(context.loss, DEFAULT_PARAMS, param["index"], param["default_eps"])
        # FD on the direct loss (fixed sol)
        grad_fd_direct = central_difference(direct_loss, DEFAULT_PARAMS, param["index"], param["default_eps"])
        fd_results[param["name"]] = {
            "fd_full": grad_fd_full,
            "fd_direct": grad_fd_direct,
        }
        print(f"  dL/d{param['name']} (full, with re-solve): {grad_fd_full: .10e}")
        print(f"  dL/d{param['name']} (direct, fixed sol):   {grad_fd_direct: .10e}")

    # ---- Comparison ----
    print(f"\n{'='*72}")
    print("Comparison summary:")
    print(f"{'='*72}")
    results = {}
    for param in PARAMETER_INFO:
        name = param["name"]
        idx = param["index"]
        ad_impl = float(grad_implicit[idx])
        ad_direct = float(grad_direct[idx])
        fd_full = fd_results[name]["fd_full"]
        fd_direct = fd_results[name]["fd_direct"]

        err_impl_vs_fd = relative_error(fd_full, ad_impl)
        err_direct_vs_fd = relative_error(fd_direct, ad_direct)
        err_direct_vs_fd_full = relative_error(fd_full, ad_direct)

        print(f"\n  dL/d{name}:")
        print(f"    AD implicit:          {ad_impl: .10e}")
        print(f"    AD direct (fixed u):  {ad_direct: .10e}")
        print(f"    FD full (re-solve):   {fd_full: .10e}")
        print(f"    FD direct (fixed u):  {fd_direct: .10e}")
        print(f"    rel_err(AD_impl vs FD_full):     {err_impl_vs_fd:.3e}")
        print(f"    rel_err(AD_direct vs FD_direct): {err_direct_vs_fd:.3e}")
        print(f"    rel_err(AD_direct vs FD_full):   {err_direct_vs_fd_full:.3e}")

        results[name] = {
            "ad_implicit": ad_impl,
            "ad_direct": ad_direct,
            "fd_full": fd_full,
            "fd_direct": fd_direct,
            "rel_err_impl_vs_fd_full": err_impl_vs_fd,
            "rel_err_direct_vs_fd_direct": err_direct_vs_fd,
            "rel_err_direct_vs_fd_full": err_direct_vs_fd_full,
        }

    write_json(experiment_dir / "exp7_results.json", {
        "displacement": DEFAULT_DISPLACEMENT,
        "params": {"E": float(DEFAULT_PARAMS[0]), "k": float(DEFAULT_PARAMS[1])},
        "loss_implicit": float(loss_val_implicit),
        "loss_direct": float(loss_val_direct),
        "results": results,
    })
    print(f"\nSaved results to {experiment_dir}")


if __name__ == "__main__":
    main()
