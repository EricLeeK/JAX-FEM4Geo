"""Experiment 10: Check if jax.vjp trace of the constraint_fn correctly sees params.

The hypothesis is: when jax.vjp traces through constraint_fn,
it may or may not re-trace the JIT-compiled kernel. We verify by
checking the Jacobian computed via jax.vjp vs FD.

Also compare with a "non-JIT" residual computation.
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
    stress_return_dp,
)

from jax_fem.solver import get_flatten_fn, apply_bc


def main() -> None:
    mesh = build_reference_mesh()
    context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh)
    problem = context.problem

    print_header("Experiment 10: jax.vjp trace analysis")

    sol_list = context.solve(DEFAULT_PARAMS)

    # ---- Method 1: constraint_fn through problem.compute_residual (JIT kernel) ----
    def constraint_fn_jit(params):
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        return res_fn(dofs)

    # Compute VJP via the same path as implicit_vjp
    v_test = jnp.ones(problem.num_total_dofs_all_vars)
    primals_1, vjp_fn_1 = jax.vjp(constraint_fn_jit, DEFAULT_PARAMS)
    vjp_1, = vjp_fn_1(v_test)
    print(f"\n[Method 1] VJP through problem.compute_residual (JIT kernel):")
    print(f"  vjp w.r.t. E: {float(vjp_1[0]):.10e}")
    print(f"  vjp w.r.t. k: {float(vjp_1[1]):.10e}")

    # ---- Method 2: bypass compute_residual, build residual from scratch ----
    fe = problem.fe

    def constraint_fn_manual(params):
        """Recompute residual manually without going through JIT kernel."""
        E, k = params[0], params[1]
        u_grads = fe.sol_to_grad(sol_list[0])
        cells, quads, vec, dim = u_grads.shape
        sigma_old = jnp.zeros((cells, quads, dim, dim))
        epsilon_old = jnp.zeros_like(sigma_old)

        def one_quad(ug, so, eo):
            return stress_return_dp(
                ug, so, eo, E, k, dim,
                nu=problem.nu, alpha=problem.alpha, a_ratio=problem.a_ratio,
            )

        # Compute stress at all quadrature points
        sigma = jax.vmap(jax.vmap(one_quad))(u_grads, sigma_old, epsilon_old)

        # Compute weak form: ∫ σ : ∇v dΩ
        # shape_grads: (num_cells, num_quads, num_nodes, dim)
        # sigma: (num_cells, num_quads, vec, dim)
        # v_grads_JxW: (num_cells, num_quads, num_nodes, 1, dim)
        v_grads_JxW = fe.v_grads_JxW  # (num_cells, num_quads, num_nodes, 1, dim)
        # res = σ_ij * ∂v/∂x_j * JxW, summed over quads
        # sigma[:, :, None, :, :] has shape (cells, quads, 1, vec, dim)
        # v_grads_JxW has shape (cells, quads, nodes, 1, dim)
        # result: (cells, quads, nodes, vec) from sum over dim
        val = jnp.sum(sigma[:, :, None, :, :] * v_grads_JxW, axis=-1)  # (cells, quads, nodes, vec)
        val = jnp.sum(val, axis=1)  # (cells, nodes, vec) - sum over quads

        # Assemble to global
        num_nodes_total = fe.num_total_nodes
        res = jnp.zeros((num_nodes_total, fe.vec))
        cells_array = fe.cells  # (num_cells, num_nodes)
        for c in range(cells):
            for n in range(cells_array.shape[1]):
                node_idx = cells_array[c, n]
                res = res.at[node_idx].add(val[c, n])
        return res.reshape(-1)

    primals_2, vjp_fn_2 = jax.vjp(constraint_fn_manual, DEFAULT_PARAMS)
    vjp_2, = vjp_fn_2(v_test[:fe.num_total_nodes * fe.vec])
    print(f"\n[Method 2] VJP through manual residual (no JIT kernel):")
    print(f"  vjp w.r.t. E: {float(vjp_2[0]):.10e}")
    print(f"  vjp w.r.t. k: {float(vjp_2[1]):.10e}")

    # ---- FD check on Method 2 ----
    def res_scalar_E(params):
        r = constraint_fn_manual(params)
        return jnp.sum(r * v_test[:fe.num_total_nodes * fe.vec])

    eps = 0.01
    fd_E = (res_scalar_E(DEFAULT_PARAMS.at[0].add(eps)) - res_scalar_E(DEFAULT_PARAMS.at[0].add(-eps))) / (2 * eps)
    fd_k = (res_scalar_E(DEFAULT_PARAMS.at[1].add(eps)) - res_scalar_E(DEFAULT_PARAMS.at[1].add(-eps))) / (2 * eps)
    print(f"\n[FD] Manual residual VJP via FD (eps={eps}):")
    print(f"  vjp w.r.t. E: {float(fd_E):.10e}")
    print(f"  vjp w.r.t. k: {float(fd_k):.10e}")

    print(f"\n{'='*72}")
    print("If Method 1 gives zero but Method 2 / FD give non-zero,")
    print("it confirms the JIT kernel doesn't pass params through.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
