from __future__ import annotations

import jax
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
    representative_u_grads,
    stress_return_dp,
    write_csv,
    write_json,
    yield_diagnostics_from_u_grad,
)


def single_point_sigma_zz(u_grad: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    sigma = stress_return_dp(
        u_grad,
        jnp.zeros((3, 3)),
        jnp.zeros((3, 3)),
        params[0],
        params[1],
        dim=3,
    )
    return sigma[2, 2]



def main() -> None:
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp3_single_point")
    mesh = build_reference_mesh()
    context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh)

    print_header("Experiment 3: single-point AD vs FD")
    print(f"Displacement: {DEFAULT_DISPLACEMENT:.3f} mm")

    fem_sol = context.solve(DEFAULT_PARAMS)
    representatives = representative_u_grads(context.problem.fe, fem_sol[0], DEFAULT_PARAMS)

    rows: list[dict[str, float | str]] = []
    details = {
        "displacement": DEFAULT_DISPLACEMENT,
        "base_params": {"E": float(DEFAULT_PARAMS[0]), "k": float(DEFAULT_PARAMS[1])},
        "representatives": {},
    }

    for label, u_grad_value in representatives.items():
        if label == "most_plastic_index" or label == "most_plastic_f_yield":
            continue

        u_grad = jnp.asarray(u_grad_value)
        loss_fn = lambda params: single_point_sigma_zz(u_grad, params)
        sigma_zz, grad_ad = jax.value_and_grad(loss_fn)(DEFAULT_PARAMS)
        diagnostics = yield_diagnostics_from_u_grad(
            u_grad,
            jnp.zeros((3, 3)),
            jnp.zeros((3, 3)),
            DEFAULT_PARAMS[0],
            DEFAULT_PARAMS[1],
            dim=3,
        )

        print(f"\nRepresentative point: {label}")
        print(f"  sigma_zz={float(sigma_zz):.8e}")
        print(
            f"  trial f_yield={float(diagnostics['f_yield']):.8e}, "
            f"plastic={bool(diagnostics['is_plastic'])}, apex={bool(diagnostics['at_apex'])}"
        )

        details["representatives"][label] = {
            "u_grad": u_grad_value.tolist(),
            "sigma_zz": float(sigma_zz),
            "f_yield": float(diagnostics["f_yield"]),
            "is_plastic": bool(diagnostics["is_plastic"]),
            "at_apex": bool(diagnostics["at_apex"]),
            "grad_ad": {"E": float(grad_ad[0]), "k": float(grad_ad[1])},
        }

        for param in PARAMETER_INFO:
            grad_fd = central_difference(loss_fn, DEFAULT_PARAMS, param["index"], param["default_eps"])
            rel_err = relative_error(float(grad_ad[param["index"]]), grad_fd)
            rows.append(
                {
                    "representative": label,
                    "parameter": param["name"],
                    "eps": param["default_eps"],
                    "sigma_zz": float(sigma_zz),
                    "grad_ad": float(grad_ad[param["index"]]),
                    "grad_fd": grad_fd,
                    "abs_error": abs(grad_fd - float(grad_ad[param["index"]])),
                    "rel_error": rel_err,
                }
            )
            print(
                f"  dL/d{param['name']}: AD={float(grad_ad[param['index']]): .8e}, "
                f"FD={grad_fd: .8e}, rel_err={rel_err:.3e}"
            )

    details["representatives"]["most_plastic_index"] = representatives["most_plastic_index"]
    details["representatives"]["most_plastic_f_yield"] = representatives["most_plastic_f_yield"]
    write_csv(experiment_dir / "single_point_gradients.csv", rows)
    write_json(experiment_dir / "single_point_details.json", details)
    print(f"\nSaved results to {experiment_dir}")


if __name__ == "__main__":
    main()
