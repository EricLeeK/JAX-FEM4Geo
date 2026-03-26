from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JAX_FEM_PATH = PROJECT_ROOT / "jax-fem-main"
if str(JAX_FEM_PATH) not in sys.path:
    sys.path.append(str(JAX_FEM_PATH))

from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper, solver

DEFAULT_PARAMS = jnp.array([70000.0, 50.0])
DEFAULT_DISPLACEMENT = -0.03
DEFAULT_NU = 0.3
DEFAULT_ALPHA = 0.3
DEFAULT_A_RATIO = 0.1
FD_SWEEP_EPS_VALUES = [500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
TAYLOR_EPS_VALUES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
DISPLACEMENT_SWEEP_VALUES = [round(x, 3) for x in onp.arange(-0.020, -0.035 - 1e-12, -0.001)]
RESULTS_ROOT = PROJECT_ROOT / "results" / "plasticity_gradient_diagnostic"
PARAMETER_INFO = [
    {"index": 0, "name": "E", "default_eps": 100.0},
    {"index": 1, "name": "k", "default_eps": 1.0},
]


def solver_options() -> dict[str, dict[str, str]]:
    return {"petsc_solver": {"ksp_type": "preonly", "pc_type": "lu"}}


def ensure_dir(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (jnp.ndarray, onp.ndarray)):
        return onp.asarray(value).tolist()
    if isinstance(value, (jnp.floating, onp.floating, float)):
        return float(value)
    if isinstance(value, (jnp.integer, onp.integer, int)):
        return int(value)
    if isinstance(value, (jnp.bool_, onp.bool_, bool)):
        return bool(value)
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


class DiagnosticDruckerPrager(Problem):
    def __init__(
        self,
        mesh: Mesh,
        vec: int,
        dim: int,
        dirichlet_bc_info: list[Any],
        nu: float = DEFAULT_NU,
        alpha: float = DEFAULT_ALPHA,
        a_ratio: float = DEFAULT_A_RATIO,
        yield_beta: float | None = None,
    ):
        self.nu = nu
        self.alpha = alpha
        self.a_ratio = a_ratio
        self.yield_beta = yield_beta
        super().__init__(mesh, vec=vec, dim=dim, dirichlet_bc_info=dirichlet_bc_info)

    def custom_init(self) -> None:
        self.fe = self.fes[0]
        num_cells = len(self.fe.cells)
        num_quads = self.fe.num_quads
        self.epsilons_old = jnp.zeros((num_cells, num_quads, self.fe.vec, self.dim))
        self.sigmas_old = jnp.zeros_like(self.epsilons_old)
        E_field = jnp.full((num_cells, num_quads, 1), DEFAULT_PARAMS[0])
        k_field = jnp.full((num_cells, num_quads, 1), DEFAULT_PARAMS[1])
        self.internal_vars = [self.sigmas_old, self.epsilons_old, E_field, k_field]

    def set_params(self, params: jnp.ndarray) -> None:
        num_cells = len(self.fe.cells)
        num_quads = self.fe.num_quads
        E_field = jnp.full((num_cells, num_quads, 1), params[0])
        k_field = jnp.full((num_cells, num_quads, 1), params[1])
        self.internal_vars[2] = E_field
        self.internal_vars[3] = k_field

    def get_tensor_map(self):
        dim = self.dim
        nu = self.nu
        alpha = self.alpha
        a_ratio = self.a_ratio
        yield_beta = self.yield_beta

        def stress_return_map(u_grad, sigma_old, epsilon_old, E_arr, k_arr):
            return stress_return_dp(
                u_grad,
                sigma_old,
                epsilon_old,
                E_arr[0],
                k_arr[0],
                dim,
                nu=nu,
                alpha=alpha,
                a_ratio=a_ratio,
                yield_beta=yield_beta,
            )

        return stress_return_map


@dataclass
class DiagnosticContext:
    mesh: Mesh
    problem: DiagnosticDruckerPrager
    fwd_pred: Any
    displacement: float
    yield_beta: float | None
    options: dict[str, Any]

    def loss(self, params: jnp.ndarray) -> jnp.ndarray:
        sol_list = self.fwd_pred(params)
        return volume_avg_sigma_zz(
            self.problem.fe,
            sol_list[0],
            params[0],
            params[1],
            yield_beta=self.yield_beta,
            nu=self.problem.nu,
            alpha=self.problem.alpha,
            a_ratio=self.problem.a_ratio,
        )

    def solve(self, params: jnp.ndarray):
        self.problem.set_params(params)
        return solver(self.problem, solver_options=self.options)


@dataclass
class YieldSummary:
    num_quadrature_points: int
    plastic_count: int
    apex_count: int
    elastic_count: int
    f_yield_min: float
    f_yield_max: float
    f_yield_mean: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_quadrature_points": self.num_quadrature_points,
            "plastic_count": self.plastic_count,
            "apex_count": self.apex_count,
            "elastic_count": self.elastic_count,
            "f_yield_min": self.f_yield_min,
            "f_yield_max": self.f_yield_max,
            "f_yield_mean": self.f_yield_mean,
        }



def write_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    if not rows:
        path_obj.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path_obj.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def write_json(path: Path | str, data: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")



def save_line_plot(
    path: Path | str,
    x_values: list[float] | onp.ndarray,
    series: list[dict[str, Any]],
    xlabel: str,
    ylabel: str,
    title: str,
    xscale: str = "linear",
    yscale: str = "linear",
) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)

    fig, ax = plt.subplots(figsize=(8, 5))
    for item in series:
        ax.plot(
            x_values,
            item["y"],
            marker=item.get("marker", "o"),
            linestyle=item.get("linestyle", "-"),
            label=item["label"],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    if any(item.get("label") for item in series):
        ax.legend()
    fig.tight_layout()
    fig.savefig(path_obj, dpi=200)
    plt.close(fig)



def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)



def copy_params_with_delta(params: jnp.ndarray, index: int, delta: float) -> jnp.ndarray:
    return params.at[index].add(delta)



def central_difference(loss_fn, params: jnp.ndarray, index: int, eps: float) -> float:
    params_plus = copy_params_with_delta(params, index, eps)
    params_minus = copy_params_with_delta(params, index, -eps)
    loss_plus = loss_fn(params_plus)
    loss_minus = loss_fn(params_minus)
    return float((loss_plus - loss_minus) / (2.0 * eps))



def relative_error(reference: float, candidate: float, floor: float = 1e-12) -> float:
    scale = max(abs(reference), abs(candidate), floor)
    return abs(reference - candidate) / scale



def convergence_rate(prev_eps: float, prev_residual: float, eps: float, residual: float) -> float | None:
    if prev_residual <= 0.0 or residual <= 0.0:
        return None
    return math.log(prev_residual / residual) / math.log(prev_eps / eps)



def safe_divide(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    tiny = 1e-30
    y_safe = jnp.where(jnp.abs(y) < tiny, 1.0, y)
    return jnp.where(jnp.abs(y) < tiny, 0.0, x / y_safe)



def yield_activation(f_yield: jnp.ndarray, beta: float | None) -> jnp.ndarray:
    if beta is None:
        return jnp.where(f_yield > 0.0, f_yield, 0.0)
    return jax.nn.softplus(beta * f_yield) / beta



def stress_return_dp(
    u_grad: jnp.ndarray,
    sigma_old: jnp.ndarray,
    epsilon_old: jnp.ndarray,
    E: jnp.ndarray,
    k: jnp.ndarray,
    dim: int,
    nu: float = DEFAULT_NU,
    alpha: float = DEFAULT_ALPHA,
    a_ratio: float = DEFAULT_A_RATIO,
    yield_beta: float | None = None,
) -> jnp.ndarray:
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

    f_yield_plus = yield_activation(f_yield, yield_beta)
    n_dev = safe_divide(s_dev, sqrt_J2_reg)
    denom = mu + 9.0 * bulk_k * alpha * alpha
    delta_lambda = safe_divide(f_yield_plus, denom)
    sigma = sigma_trial - delta_lambda * (
        mu * n_dev + 3.0 * bulk_k * alpha * jnp.eye(dim)
    )

    sigma_apex = (k / (3.0 * alpha)) * jnp.eye(dim)
    at_apex = jnp.logical_and(f_yield > 0.0, I1 > k / alpha)
    sigma = jnp.where(at_apex, sigma_apex, sigma)
    return sigma



def yield_diagnostics_from_u_grad(
    u_grad: jnp.ndarray,
    sigma_old: jnp.ndarray,
    epsilon_old: jnp.ndarray,
    E: jnp.ndarray,
    k: jnp.ndarray,
    dim: int,
    nu: float = DEFAULT_NU,
    alpha: float = DEFAULT_ALPHA,
    a_ratio: float = DEFAULT_A_RATIO,
) -> dict[str, jnp.ndarray]:
    a = a_ratio * k
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    epsilon_crt = 0.5 * (u_grad + u_grad.T)
    epsilon_inc = epsilon_crt - epsilon_old
    sigma_trial = lmbda * jnp.trace(epsilon_inc) * jnp.eye(dim) + 2.0 * mu * epsilon_inc + sigma_old

    I1 = jnp.trace(sigma_trial)
    s_dev = sigma_trial - (I1 / 3.0) * jnp.eye(dim)
    J2 = 0.5 * jnp.sum(s_dev * s_dev)
    sqrt_J2_reg = jnp.sqrt(J2 + a * a)
    f_yield = sqrt_J2_reg + alpha * I1 - k
    at_apex = jnp.logical_and(f_yield > 0.0, I1 > k / alpha)
    return {
        "f_yield": f_yield,
        "I1": I1,
        "J2": J2,
        "sqrt_J2_reg": sqrt_J2_reg,
        "is_plastic": f_yield > 0.0,
        "at_apex": at_apex,
    }



def volume_avg_sigma_zz(
    fe,
    sol: jnp.ndarray,
    E: jnp.ndarray,
    k: jnp.ndarray,
    yield_beta: float | None = None,
    nu: float = DEFAULT_NU,
    alpha: float = DEFAULT_ALPHA,
    a_ratio: float = DEFAULT_A_RATIO,
) -> jnp.ndarray:
    u_grads = fe.sol_to_grad(sol)
    cells, quads, _, dim = u_grads.shape
    sigma_old = jnp.zeros((cells, quads, dim, dim))
    epsilon_old = jnp.zeros_like(sigma_old)

    def one_quad(ug, so, eo):
        return stress_return_dp(
            ug,
            so,
            eo,
            E,
            k,
            dim,
            nu=nu,
            alpha=alpha,
            a_ratio=a_ratio,
            yield_beta=yield_beta,
        )

    sig = jax.vmap(jax.vmap(one_quad))(u_grads, sigma_old, epsilon_old)
    jxw = fe.JxW
    return jnp.sum(sig[..., 2, 2] * jxw) / jnp.sum(jxw)



def quadrature_yield_diagnostics(
    fe,
    sol: jnp.ndarray,
    E: jnp.ndarray,
    k: jnp.ndarray,
    nu: float = DEFAULT_NU,
    alpha: float = DEFAULT_ALPHA,
    a_ratio: float = DEFAULT_A_RATIO,
) -> dict[str, onp.ndarray]:
    u_grads = fe.sol_to_grad(sol)
    cells, quads, _, dim = u_grads.shape
    sigma_old = jnp.zeros((cells, quads, dim, dim))
    epsilon_old = jnp.zeros_like(sigma_old)

    def one_quad(ug, so, eo):
        return yield_diagnostics_from_u_grad(
            ug,
            so,
            eo,
            E,
            k,
            dim,
            nu=nu,
            alpha=alpha,
            a_ratio=a_ratio,
        )

    diagnostics = jax.vmap(jax.vmap(one_quad))(u_grads, sigma_old, epsilon_old)
    return {name: onp.asarray(value) for name, value in diagnostics.items()}



def summarize_yield_state(
    fe,
    sol: jnp.ndarray,
    params: jnp.ndarray,
    nu: float = DEFAULT_NU,
    alpha: float = DEFAULT_ALPHA,
    a_ratio: float = DEFAULT_A_RATIO,
) -> tuple[YieldSummary, dict[str, onp.ndarray]]:
    diagnostics = quadrature_yield_diagnostics(
        fe,
        sol,
        params[0],
        params[1],
        nu=nu,
        alpha=alpha,
        a_ratio=a_ratio,
    )
    is_plastic = diagnostics["is_plastic"].astype(bool)
    at_apex = diagnostics["at_apex"].astype(bool)
    f_yield = diagnostics["f_yield"]
    summary = YieldSummary(
        num_quadrature_points=int(is_plastic.size),
        plastic_count=int(is_plastic.sum()),
        apex_count=int(at_apex.sum()),
        elastic_count=int(is_plastic.size - is_plastic.sum()),
        f_yield_min=float(f_yield.min()),
        f_yield_max=float(f_yield.max()),
        f_yield_mean=float(f_yield.mean()),
    )
    return summary, diagnostics



def representative_u_grads(fe, sol: jnp.ndarray, params: jnp.ndarray) -> dict[str, Any]:
    u_grads = onp.asarray(fe.sol_to_grad(sol))
    jxw = onp.asarray(fe.JxW)
    weights = jxw[..., None, None]
    weighted_mean = onp.sum(u_grads * weights, axis=(0, 1)) / onp.sum(jxw)

    _, diagnostics = summarize_yield_state(fe, sol, params)
    f_yield = diagnostics["f_yield"]
    max_index = onp.unravel_index(int(f_yield.argmax()), f_yield.shape)
    return {
        "volume_weighted_mean": weighted_mean,
        "most_plastic_quad": u_grads[max_index],
        "most_plastic_index": [int(max_index[0]), int(max_index[1])],
        "most_plastic_f_yield": float(f_yield[max_index]),
    }



def switched_quadrature_points(mask_a: onp.ndarray, mask_b: onp.ndarray) -> list[list[int]]:
    indices = onp.argwhere(mask_a.astype(bool) != mask_b.astype(bool))
    return [[int(cell), int(quad)] for cell, quad in indices]



def build_reference_mesh(mesh_dir: Path | None = None) -> Mesh:
    mesh_dir = ensure_dir(mesh_dir or (RESULTS_ROOT / "mesh"))
    ele_type = "HEX8"
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(
        Nx=2,
        Ny=2,
        Nz=2,
        domain_x=10.0,
        domain_y=10.0,
        domain_z=10.0,
        data_dir=str(mesh_dir),
        ele_type=ele_type,
    )
    return Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])



def make_dirichlet_bc(displacement: float, height: float = 10.0) -> list[Any]:
    def bottom(point):
        return jnp.isclose(point[2], 0.0)

    def top(point):
        return jnp.isclose(point[2], height)

    def corner(point):
        return jnp.logical_and(
            jnp.logical_and(jnp.isclose(point[0], 0.0), jnp.isclose(point[1], 0.0)),
            jnp.isclose(point[2], 0.0),
        )

    return [
        [bottom, top, corner, corner],
        [2, 2, 0, 1],
        [lambda point: 0.0, lambda point: displacement, lambda point: 0.0, lambda point: 0.0],
    ]



def create_context(
    displacement: float = DEFAULT_DISPLACEMENT,
    mesh: Mesh | None = None,
    yield_beta: float | None = None,
) -> DiagnosticContext:
    mesh = mesh or build_reference_mesh()
    problem = DiagnosticDruckerPrager(
        mesh,
        vec=3,
        dim=3,
        dirichlet_bc_info=make_dirichlet_bc(displacement),
        yield_beta=yield_beta,
    )
    options = solver_options()
    fwd_pred = ad_wrapper(problem, solver_options=options, adjoint_solver_options=options)
    return DiagnosticContext(
        mesh=mesh,
        problem=problem,
        fwd_pred=fwd_pred,
        displacement=displacement,
        yield_beta=yield_beta,
        options=options,
    )
