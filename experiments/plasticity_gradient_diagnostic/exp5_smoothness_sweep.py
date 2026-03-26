from __future__ import annotations

import jax

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
    save_line_plot,
    summarize_yield_state,
    write_csv,
    write_json,
)

SMOOTH_BETAS = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]



def main() -> None:
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp5_smoothness_sweep")
    mesh = build_reference_mesh()

    print_header("Experiment 5: smoothness sweep")
    print(f"Displacement: {DEFAULT_DISPLACEMENT:.3f} mm")

    rows: list[dict[str, float | str | int]] = []
    details = {
        "displacement": DEFAULT_DISPLACEMENT,
        "base_params": {"E": float(DEFAULT_PARAMS[0]), "k": float(DEFAULT_PARAMS[1])},
        "variants": {},
    }

    variant_specs = [("hard", None)] + [(f"softplus_beta_{beta:g}", beta) for beta in SMOOTH_BETAS]
    for label, beta in variant_specs:
        context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh, yield_beta=beta)
        loss_val, grad_ad = jax.value_and_grad(context.loss)(DEFAULT_PARAMS)
        sol_list = context.solve(DEFAULT_PARAMS)
        yield_summary, _ = summarize_yield_state(context.problem.fe, sol_list[0], DEFAULT_PARAMS)

        variant_detail = {
            "yield_beta": beta,
            "loss": float(loss_val),
            "grad_ad": {"E": float(grad_ad[0]), "k": float(grad_ad[1])},
            "yield_summary": yield_summary.as_dict(),
        }
        print(f"\nVariant {label}:")
        print(f"  Loss={float(loss_val):.8e}")
        print(f"  AD gradient: dL/dE={float(grad_ad[0]):.8e}, dL/dk={float(grad_ad[1]):.8e}")

        for param in PARAMETER_INFO:
            grad_fd = central_difference(context.loss, DEFAULT_PARAMS, param["index"], param["default_eps"])
            rel_err = relative_error(float(grad_ad[param["index"]]), grad_fd)
            rows.append(
                {
                    "variant": label,
                    "yield_beta": "hard" if beta is None else beta,
                    "parameter": param["name"],
                    "loss": float(loss_val),
                    "grad_ad": float(grad_ad[param["index"]]),
                    "grad_fd": grad_fd,
                    "abs_error": abs(grad_fd - float(grad_ad[param["index"]])),
                    "rel_error": rel_err,
                    "plastic_count": yield_summary.plastic_count,
                    "apex_count": yield_summary.apex_count,
                }
            )
            variant_detail[f"grad_fd_{param['name']}"] = grad_fd
            variant_detail[f"rel_error_{param['name']}"] = rel_err
            print(
                f"  dL/d{param['name']}: FD={grad_fd: .8e}, rel_err={rel_err:.3e}"
            )
        details["variants"][label] = variant_detail

    beta_rows = [row for row in rows if row["yield_beta"] != "hard"]
    save_line_plot(
        experiment_dir / "smoothness_error_E.png",
        SMOOTH_BETAS,
        [
            {
                "label": "rel err dL/dE",
                "y": [row["rel_error"] for row in beta_rows if row["parameter"] == "E"],
            },
            {
                "label": "rel err dL/dk",
                "y": [row["rel_error"] for row in beta_rows if row["parameter"] == "k"],
            },
        ],
        xlabel="softplus beta",
        ylabel="Relative error",
        title="AD vs FD relative error under smooth yield activation",
        xscale="log",
        yscale="log",
    )
    save_line_plot(
        experiment_dir / "smoothness_gradients_k.png",
        SMOOTH_BETAS,
        [
            {
                "label": "AD dL/dk",
                "y": [row["grad_ad"] for row in beta_rows if row["parameter"] == "k"],
            },
            {
                "label": "FD dL/dk",
                "y": [row["grad_fd"] for row in beta_rows if row["parameter"] == "k"],
            },
        ],
        xlabel="softplus beta",
        ylabel="Gradient wrt k",
        title="Gradient trend for k under smooth yield activation",
        xscale="log",
        yscale="linear",
    )

    write_csv(experiment_dir / "smoothness_sweep.csv", rows)
    write_json(experiment_dir / "smoothness_details.json", details)
    print(f"\nSaved results to {experiment_dir}")


if __name__ == "__main__":
    main()
