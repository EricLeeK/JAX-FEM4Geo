"""Experiment 9b: Detailed check of residual dependency on params.

Check if the residual value changes at all when params are perturbed,
both through the full pipeline and through individual components.
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
)

from jax_fem.solver import get_flatten_fn, apply_bc


def main() -> None:
    mesh = build_reference_mesh()
    context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh)
    problem = context.problem

    print_header("Experiment 9b: Detailed residual param dependency check")

    # Get converged solution
    sol_list = context.solve(DEFAULT_PARAMS)
    dofs_fixed = jax.flatten_util.ravel_pytree(sol_list)[0]

    # Function: raw residual (before BC application)
    def raw_residual(params):
        problem.set_params(params)
        res_list = problem.compute_residual(sol_list)
        return jax.flatten_util.ravel_pytree(res_list)[0]

    # Function: residual with BC
    def bc_residual(params):
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs_fixed)

    # Check raw residual
    res0_raw = raw_residual(DEFAULT_PARAMS)
    print(f"\nRaw residual norm at base params: {onp.linalg.norm(onp.array(res0_raw)):.10e}")

    # Check with perturbed E
    for delta_E in [1.0, 10.0, 100.0, 1000.0]:
        params_p = DEFAULT_PARAMS.at[0].add(delta_E)
        res_p = raw_residual(params_p)
        diff = onp.linalg.norm(onp.array(res_p) - onp.array(res0_raw))
        print(f"  E+{delta_E}: raw res change = {diff:.10e}, raw res norm = {onp.linalg.norm(onp.array(res_p)):.10e}")

    # Check with perturbed k
    for delta_k in [0.01, 0.1, 1.0, 10.0]:
        params_p = DEFAULT_PARAMS.at[1].add(delta_k)
        res_p = raw_residual(params_p)
        diff = onp.linalg.norm(onp.array(res_p) - onp.array(res0_raw))
        print(f"  k+{delta_k}: raw res change = {diff:.10e}, raw res norm = {onp.linalg.norm(onp.array(res_p)):.10e}")

    # Also check: does get_tensor_map() actually use self.E_val?
    print(f"\n--- Checking tensor_map directly ---")
    problem.set_params(DEFAULT_PARAMS)
    tensor_map_1 = problem.get_tensor_map()

    # Get one u_grad from the solution
    fe = problem.fe
    u_grads = fe.sol_to_grad(sol_list[0])
    ug_sample = u_grads[0, 0]  # first cell, first quad
    sig_old = jnp.zeros((3, 3))
    eps_old = jnp.zeros((3, 3))

    stress_1 = tensor_map_1(ug_sample, sig_old, eps_old)
    print(f"  sigma_zz at base params: {float(stress_1[2, 2]):.10e}")

    # Change E_val and check if tensor_map sees it
    problem.E_val = DEFAULT_PARAMS[0] + 1000.0
    stress_2_same_map = tensor_map_1(ug_sample, sig_old, eps_old)
    print(f"  sigma_zz after E+1000 (same tensor_map instance): {float(stress_2_same_map[2, 2]):.10e}")

    # Get a fresh tensor_map
    tensor_map_2 = problem.get_tensor_map()
    stress_2_new_map = tensor_map_2(ug_sample, sig_old, eps_old)
    print(f"  sigma_zz after E+1000 (new tensor_map instance):  {float(stress_2_new_map[2, 2]):.10e}")

    # Reset
    problem.set_params(DEFAULT_PARAMS)

    # Check if kernel (jit-compiled vmap) sees changes
    print(f"\n--- Checking jit-compiled kernel ---")
    # The kernel is self.kernel = jax.jit(jax.vmap(kernel))
    # kernel calls self.get_tensor_map() inside its body
    # When called with different self.E_val, does it recompile?

    problem.set_params(DEFAULT_PARAMS)
    res_a = problem.compute_residual(sol_list)
    res_a_norm = onp.linalg.norm(onp.array(jax.flatten_util.ravel_pytree(res_a)[0]))

    problem.E_val = DEFAULT_PARAMS[0] + 1000.0
    res_b = problem.compute_residual(sol_list)
    res_b_norm = onp.linalg.norm(onp.array(jax.flatten_util.ravel_pytree(res_b)[0]))

    print(f"  Residual norm at E={float(DEFAULT_PARAMS[0])}: {res_a_norm:.10e}")
    print(f"  Residual norm at E={float(DEFAULT_PARAMS[0] + 1000)}: {res_b_norm:.10e}")
    print(f"  Changed? {not onp.isclose(res_a_norm, res_b_norm)}")

    problem.set_params(DEFAULT_PARAMS)


if __name__ == "__main__":
    main()
