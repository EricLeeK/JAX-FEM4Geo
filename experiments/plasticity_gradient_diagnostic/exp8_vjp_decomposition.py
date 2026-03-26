"""Experiment 8: Decompose the VJP to understand what implicit_vjp contributes.

Total gradient: dL/dp = ∂L/∂p_direct + ∂L/∂sol * dsol/dp (implicit part)

We compute:
1. Total gradient (from ad_wrapper)
2. Direct gradient (∂L/∂p with sol fixed)
3. Implicit contribution = total - direct
4. FD estimate of the implicit part (resolve with perturbed params, subtract direct effect)
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
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp8_vjp_decomposition")
    mesh = build_reference_mesh()
    context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh)

    print_header("Experiment 8: VJP decomposition")

    # Total gradient via ad_wrapper
    _, grad_total = jax.value_and_grad(context.loss)(DEFAULT_PARAMS)

    # Direct gradient (∂L/∂p with sol fixed at converged solution)
    sol_list = context.solve(DEFAULT_PARAMS)
    sol_fixed = jax.lax.stop_gradient(sol_list[0])

    def direct_loss(params):
        return volume_avg_sigma_zz(
            context.problem.fe, sol_fixed, params[0], params[1],
            yield_beta=context.yield_beta,
            nu=context.problem.nu, alpha=context.problem.alpha,
            a_ratio=context.problem.a_ratio,
        )

    _, grad_direct = jax.value_and_grad(direct_loss)(DEFAULT_PARAMS)

    # Implicit contribution = total - direct
    grad_implicit = grad_total - grad_direct

    # FD full gradient
    fd_full = {}
    for param in PARAMETER_INFO:
        fd_full[param["name"]] = central_difference(context.loss, DEFAULT_PARAMS, param["index"], param["default_eps"])

    # FD of the implicit part only:
    # The "true" implicit part is how L changes because sol changes when params change.
    # We can estimate this as FD_full - FD_direct
    fd_direct = {}
    for param in PARAMETER_INFO:
        fd_direct[param["name"]] = central_difference(direct_loss, DEFAULT_PARAMS, param["index"], param["default_eps"])

    print("\n" + "=" * 72)
    print("Decomposition of dL/dp:")
    print("=" * 72)

    for param in PARAMETER_INFO:
        name = param["name"]
        idx = param["index"]
        total = float(grad_total[idx])
        direct = float(grad_direct[idx])
        implicit = float(grad_implicit[idx])
        fd_f = fd_full[name]
        fd_d = fd_direct[name]
        fd_impl_est = fd_f - fd_d  # FD estimate of the implicit contribution

        print(f"\n  dL/d{name}:")
        print(f"    Total (ad_wrapper):          {total: .10e}")
        print(f"    Direct (∂L/∂p, sol fixed):   {direct: .10e}")
        print(f"    Implicit (total - direct):   {implicit: .10e}")
        print(f"    FD full:                     {fd_f: .10e}")
        print(f"    FD direct:                   {fd_d: .10e}")
        print(f"    FD implicit (full - direct): {fd_impl_est: .10e}")
        print(f"    ---")
        print(f"    Direct AD vs FD:   rel_err = {relative_error(fd_d, direct):.3e}")
        if abs(fd_impl_est) > 1e-15:
            print(f"    Implicit AD vs FD: rel_err = {relative_error(fd_impl_est, implicit):.3e}")
        else:
            print(f"    Implicit FD ≈ 0 (implicit contribution negligible)")

    write_json(experiment_dir / "exp8_results.json", {
        "grad_total": [float(grad_total[0]), float(grad_total[1])],
        "grad_direct": [float(grad_direct[0]), float(grad_direct[1])],
        "grad_implicit": [float(grad_implicit[0]), float(grad_implicit[1])],
        "fd_full": fd_full,
        "fd_direct": fd_direct,
    })
    print(f"\nSaved to {experiment_dir}")


if __name__ == "__main__":
    main()
