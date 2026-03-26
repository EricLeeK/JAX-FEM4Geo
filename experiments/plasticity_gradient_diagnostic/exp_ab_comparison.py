"""A/B comparison: Old (self-closure) vs New (internal_vars) parameter passing.

Runs the exact same problem configuration with both implementations,
comparing AD gradient, FD gradient, and Taylor test in a single script.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import sys, os, math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JAX_FEM_PATH = PROJECT_ROOT / "jax-fem-main"
if str(JAX_FEM_PATH) not in sys.path:
    sys.path.append(str(JAX_FEM_PATH))

from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper, solver

# ---- Shared config ----
DEFAULT_PARAMS = jnp.array([70000.0, 50.0])
DISPLACEMENT = -0.03
NU = 0.3
ALPHA = 0.3
A_RATIO = 0.1
RESULTS_ROOT = PROJECT_ROOT / "results" / "plasticity_gradient_diagnostic"


def solver_options():
    return {"petsc_solver": {"ksp_type": "preonly", "pc_type": "lu"}}


def stress_return_dp(u_grad, sigma_old, epsilon_old, E, k, dim,
                     nu=NU, alpha=ALPHA, a_ratio=A_RATIO):
    a = a_ratio * k
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    bulk_k = lmbda + 2.0 * mu / 3.0

    epsilon_crt = 0.5 * (u_grad + u_grad.T)
    epsilon_inc = epsilon_crt - epsilon_old
    sigma_trial = lmbda * jnp.trace(epsilon_inc) * jnp.eye(dim) + 2.0 * mu * epsilon_inc + sigma_old

    I1 = jnp.trace(sigma_trial)
    s_dev = sigma_trial - (I1 / 3.0) * jnp.eye(dim)
    J2 = 0.5 * jnp.sum(s_dev * s_dev)
    sqrt_J2_reg = jnp.sqrt(J2 + a * a)
    f_yield = sqrt_J2_reg + alpha * I1 - k

    f_yield_plus = jnp.where(f_yield > 0.0, f_yield, 0.0)
    y_safe = jnp.where(jnp.abs(sqrt_J2_reg) < 1e-30, 1.0, sqrt_J2_reg)
    n_dev = jnp.where(jnp.abs(sqrt_J2_reg) < 1e-30, 0.0, s_dev / y_safe)
    denom_safe = jnp.where(jnp.abs(mu + 9.0 * bulk_k * alpha * alpha) < 1e-30, 1.0, mu + 9.0 * bulk_k * alpha * alpha)
    delta_lambda = jnp.where(jnp.abs(mu + 9.0 * bulk_k * alpha * alpha) < 1e-30, 0.0, f_yield_plus / denom_safe)

    sigma = sigma_trial - delta_lambda * (mu * n_dev + 3.0 * bulk_k * alpha * jnp.eye(dim))
    sigma_apex = (k / (3.0 * alpha)) * jnp.eye(dim)
    at_apex = jnp.logical_and(f_yield > 0.0, I1 > k / alpha)
    sigma = jnp.where(at_apex, sigma_apex, sigma)
    return sigma


def volume_avg_sigma_zz(fe, sol, E, k):
    u_grads = fe.sol_to_grad(sol)
    cells, quads, _, dim = u_grads.shape
    sigma_old = jnp.zeros((cells, quads, dim, dim))
    epsilon_old = jnp.zeros_like(sigma_old)

    def one_quad(ug, so, eo):
        return stress_return_dp(ug, so, eo, E, k, dim)

    sig = jax.vmap(jax.vmap(one_quad))(u_grads, sigma_old, epsilon_old)
    jxw = fe.JxW
    return jnp.sum(sig[..., 2, 2] * jxw) / jnp.sum(jxw)


# ========================= OLD MODEL (self-closure) =========================
class OldModel(Problem):
    def __init__(self, mesh, vec, dim, dirichlet_bc_info):
        super().__init__(mesh, vec=vec, dim=dim, dirichlet_bc_info=dirichlet_bc_info)

    def custom_init(self):
        self.fe = self.fes[0]
        self.E_val = DEFAULT_PARAMS[0]
        self.k_val = DEFAULT_PARAMS[1]
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.sigmas_old = jnp.zeros((nc, nq, self.fe.vec, self.dim))
        self.epsilons_old = jnp.zeros_like(self.sigmas_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def set_params(self, params):
        self.E_val = params[0]
        self.k_val = params[1]

    def get_tensor_map(self):
        dim = self.dim
        def stress_fn(u_grad, sigma_old, epsilon_old):
            return stress_return_dp(u_grad, sigma_old, epsilon_old,
                                    self.E_val, self.k_val, dim)
        return stress_fn


# ========================= NEW MODEL (internal_vars) =========================
class NewModel(Problem):
    def __init__(self, mesh, vec, dim, dirichlet_bc_info):
        super().__init__(mesh, vec=vec, dim=dim, dirichlet_bc_info=dirichlet_bc_info)

    def custom_init(self):
        self.fe = self.fes[0]
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.sigmas_old = jnp.zeros((nc, nq, self.fe.vec, self.dim))
        self.epsilons_old = jnp.zeros_like(self.sigmas_old)
        E_field = jnp.full((nc, nq, 1), DEFAULT_PARAMS[0])
        k_field = jnp.full((nc, nq, 1), DEFAULT_PARAMS[1])
        self.internal_vars = [self.sigmas_old, self.epsilons_old, E_field, k_field]

    def set_params(self, params):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        self.internal_vars[2] = jnp.full((nc, nq, 1), params[0])
        self.internal_vars[3] = jnp.full((nc, nq, 1), params[1])

    def get_tensor_map(self):
        dim = self.dim
        def stress_fn(u_grad, sigma_old, epsilon_old, E_arr, k_arr):
            return stress_return_dp(u_grad, sigma_old, epsilon_old,
                                    E_arr[0], k_arr[0], dim)
        return stress_fn


# ========================= Test harness =========================
def build_mesh():
    mesh_dir = RESULTS_ROOT / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    ele_type = "HEX8"
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(Nx=2, Ny=2, Nz=2, domain_x=10., domain_y=10., domain_z=10.,
                                 data_dir=str(mesh_dir), ele_type=ele_type)
    return Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


def make_bc(displacement, height=10.0):
    def bottom(point): return jnp.isclose(point[2], 0.)
    def top(point): return jnp.isclose(point[2], height)
    def corner(point):
        return jnp.logical_and(jnp.logical_and(jnp.isclose(point[0], 0.), jnp.isclose(point[1], 0.)),
                               jnp.isclose(point[2], 0.))
    return [[bottom, top, corner, corner], [2, 2, 0, 1],
            [lambda p: 0., lambda p: displacement, lambda p: 0., lambda p: 0.]]


def central_fd(loss_fn, params, index, eps):
    lp = loss_fn(params.at[index].add(eps))
    lm = loss_fn(params.at[index].add(-eps))
    return float((lp - lm) / (2. * eps))


def taylor_test(loss_fn, params, grad_ad, eps_list):
    """Returns list of (eps, r0, r1, rate0, rate1)."""
    jnp_key = jax.random.PRNGKey(42)
    dp = jax.random.normal(jnp_key, params.shape)
    dp = dp / jnp.linalg.norm(dp)
    L0 = float(loss_fn(params))
    gd = float(jnp.dot(grad_ad, dp))
    results = []
    prev_r0 = prev_r1 = prev_eps = None
    for eps in eps_list:
        Lp = float(loss_fn(params + eps * dp))
        r0 = abs(Lp - L0)
        r1 = abs(Lp - L0 - eps * gd)
        rate0 = math.log(prev_r0 / r0) / math.log(prev_eps / eps) if prev_r0 and r0 > 0 else None
        rate1 = math.log(prev_r1 / r1) / math.log(prev_eps / eps) if prev_r1 and r1 > 0 else None
        results.append((eps, r0, r1, rate0, rate1))
        prev_r0, prev_r1, prev_eps = r0, r1, eps
    return results


def run_test(label, ModelClass, mesh):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    problem = ModelClass(mesh, vec=3, dim=3, dirichlet_bc_info=make_bc(DISPLACEMENT))
    opts = solver_options()
    fwd_pred = ad_wrapper(problem, solver_options=opts, adjoint_solver_options=opts)

    def loss(params):
        sol_list = fwd_pred(params)
        return volume_avg_sigma_zz(problem.fe, sol_list[0], params[0], params[1])

    # AD gradient
    loss_val, grad_ad = jax.value_and_grad(loss)(DEFAULT_PARAMS)
    print(f"\n  Loss = {float(loss_val):.10e}")
    print(f"  AD  dL/dE = {float(grad_ad[0]): .10e}")
    print(f"  AD  dL/dk = {float(grad_ad[1]): .10e}")

    # FD gradient
    fd_E = central_fd(loss, DEFAULT_PARAMS, 0, 100.0)
    fd_k = central_fd(loss, DEFAULT_PARAMS, 1, 1.0)
    print(f"  FD  dL/dE = {fd_E: .10e}  (eps=100)")
    print(f"  FD  dL/dk = {fd_k: .10e}  (eps=1)")

    def rel_err(a, b):
        s = max(abs(a), abs(b), 1e-30)
        return abs(a - b) / s

    re_E = rel_err(float(grad_ad[0]), fd_E)
    re_k = rel_err(float(grad_ad[1]), fd_k)
    print(f"  rel_err(E) = {re_E:.3e}")
    print(f"  rel_err(k) = {re_k:.3e}")

    # Taylor test
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    tt = taylor_test(loss, DEFAULT_PARAMS, grad_ad, eps_list)
    print(f"\n  Taylor test:")
    for eps, r0, r1, rate0, rate1 in tt:
        r0s = f"{rate0:.3f}" if rate0 else "  -  "
        r1s = f"{rate1:.3f}" if rate1 else "  -  "
        print(f"    eps={eps:.0e}: r0={r0:.6e} (rate {r0s}), r1={r1:.6e} (rate {r1s})")

    return {
        "loss": float(loss_val),
        "ad_E": float(grad_ad[0]), "ad_k": float(grad_ad[1]),
        "fd_E": fd_E, "fd_k": fd_k,
        "rel_err_E": re_E, "rel_err_k": re_k,
        "taylor_r1_rates": [t[4] for t in tt if t[4] is not None],
    }


def main():
    mesh = build_mesh()
    print("\n" + "#" * 72)
    print("#  A/B Comparison: displacement = -0.03 mm, plastic regime")
    print("#  Same mesh, same params, same loss function")
    print("#" * 72)

    res_old = run_test("OLD: E,k via self-closure (BUG)", OldModel, mesh)
    res_new = run_test("NEW: E,k via internal_vars (FIX)", NewModel, mesh)

    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"                    OLD (self-closure)      NEW (internal_vars)")
    print(f"  AD  dL/dE:      {res_old['ad_E']: .8e}      {res_new['ad_E']: .8e}")
    print(f"  FD  dL/dE:      {res_old['fd_E']: .8e}      {res_new['fd_E']: .8e}")
    print(f"  rel_err(E):     {res_old['rel_err_E']:.3e}              {res_new['rel_err_E']:.3e}")
    print(f"  AD  dL/dk:      {res_old['ad_k']: .8e}      {res_new['ad_k']: .8e}")
    print(f"  FD  dL/dk:      {res_old['fd_k']: .8e}      {res_new['fd_k']: .8e}")
    print(f"  rel_err(k):     {res_old['rel_err_k']:.3e}              {res_new['rel_err_k']:.3e}")

    old_rates = res_old["taylor_r1_rates"]
    new_rates = res_new["taylor_r1_rates"]
    print(f"  Taylor r1 rates (OLD): {['%.3f' % r for r in old_rates]}")
    print(f"  Taylor r1 rates (NEW): {['%.3f' % r for r in new_rates]}")

    old_ok = all(r > 1.8 for r in old_rates[:3]) if old_rates else False
    new_ok = all(r > 1.8 for r in new_rates[:3]) if new_rates else False
    print(f"\n  OLD Taylor test passed? {'YES' if old_ok else 'NO'}")
    print(f"  NEW Taylor test passed? {'YES' if new_ok else 'NO'}")
    print(f"  OLD rel_err < 1e-3?     {'YES' if res_old['rel_err_E'] < 1e-3 and res_old['rel_err_k'] < 1e-3 else 'NO'}")
    print(f"  NEW rel_err < 1e-3?     {'YES' if res_new['rel_err_E'] < 1e-3 and res_new['rel_err_k'] < 1e-3 else 'NO'}")


if __name__ == "__main__":
    main()
