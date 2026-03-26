"""Experiment 9: Check if the residual properly depends on params.

We directly compute ∂c/∂p (derivative of residual w.r.t. params at converged sol)
using both AD and FD, to see if the JIT-compiled kernel correctly captures
the parameter dependency.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp

from common import (
    DEFAULT_DISPLACEMENT,
    DEFAULT_PARAMS,
    RESULTS_ROOT,
    build_reference_mesh,
    create_context,
    ensure_dir,
    print_header,
    write_json,
)

from jax_fem.solver import get_flatten_fn, apply_bc


def main() -> None:
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp9_residual_grad")
    mesh = build_reference_mesh()
    context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh)

    print_header("Experiment 9: Residual gradient w.r.t. params")

    # Get converged solution
    sol_list = context.solve(DEFAULT_PARAMS)
    sol_fixed = sol_list[0]
    dofs_fixed = jax.flatten_util.ravel_pytree(sol_list)[0]

    problem = context.problem

    # Define residual function: c(params) at fixed dofs
    def residual_fn(params):
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs_fixed)

    # Compute residual at base params
    res_base = residual_fn(DEFAULT_PARAMS)
    print(f"\nResidual norm at converged sol: {onp.linalg.norm(onp.array(res_base)):.6e}")

    # AD gradient of residual w.r.t. params
    jac_ad = jax.jacobian(residual_fn)(DEFAULT_PARAMS)
    print(f"Jacobian shape (AD): {jac_ad.shape}")
    print(f"||dc/dE||_2 (AD) = {onp.linalg.norm(onp.array(jac_ad[:, 0])):.6e}")
    print(f"||dc/dk||_2 (AD) = {onp.linalg.norm(onp.array(jac_ad[:, 1])):.6e}")

    # FD gradient of residual w.r.t. params
    eps_E = 1.0
    eps_k = 0.01
    params_Ep = DEFAULT_PARAMS.at[0].add(eps_E)
    params_Em = DEFAULT_PARAMS.at[0].add(-eps_E)
    params_kp = DEFAULT_PARAMS.at[1].add(eps_k)
    params_km = DEFAULT_PARAMS.at[1].add(-eps_k)

    dc_dE_fd = (residual_fn(params_Ep) - residual_fn(params_Em)) / (2 * eps_E)
    dc_dk_fd = (residual_fn(params_kp) - residual_fn(params_km)) / (2 * eps_k)

    print(f"\n||dc/dE||_2 (FD, eps={eps_E}) = {onp.linalg.norm(onp.array(dc_dE_fd)):.6e}")
    print(f"||dc/dk||_2 (FD, eps={eps_k}) = {onp.linalg.norm(onp.array(dc_dk_fd)):.6e}")

    # Check if AD dc/dp is zero (which would be wrong)
    dc_dE_ad = onp.array(jac_ad[:, 0])
    dc_dk_ad = onp.array(jac_ad[:, 1])
    dc_dE_fd_np = onp.array(dc_dE_fd)
    dc_dk_fd_np = onp.array(dc_dk_fd)

    if onp.linalg.norm(dc_dE_ad) < 1e-20:
        print("\n*** WARNING: AD says dc/dE = 0 (params not flowing through residual!) ***")
    else:
        rel_err_E = onp.linalg.norm(dc_dE_ad - dc_dE_fd_np) / max(onp.linalg.norm(dc_dE_fd_np), 1e-30)
        print(f"\nrel_err ||dc/dE|| AD vs FD: {rel_err_E:.6e}")

    if onp.linalg.norm(dc_dk_ad) < 1e-20:
        print("*** WARNING: AD says dc/dk = 0 (params not flowing through residual!) ***")
    else:
        rel_err_k = onp.linalg.norm(dc_dk_ad - dc_dk_fd_np) / max(onp.linalg.norm(dc_dk_fd_np), 1e-30)
        print(f"rel_err ||dc/dk|| AD vs FD: {rel_err_k:.6e}")

    write_json(experiment_dir / "exp9_results.json", {
        "res_norm": float(onp.linalg.norm(onp.array(res_base))),
        "dc_dE_norm_ad": float(onp.linalg.norm(dc_dE_ad)),
        "dc_dE_norm_fd": float(onp.linalg.norm(dc_dE_fd_np)),
        "dc_dk_norm_ad": float(onp.linalg.norm(dc_dk_ad)),
        "dc_dk_norm_fd": float(onp.linalg.norm(dc_dk_fd_np)),
    })
    print(f"\nSaved to {experiment_dir}")


if __name__ == "__main__":
    main()
