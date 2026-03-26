from __future__ import annotations

import jax

from common import (
    DEFAULT_PARAMS,
    DISPLACEMENT_SWEEP_VALUES,
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


def main() -> None:
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp6_displacement_sweep")
    mesh = build_reference_mesh()

    print_header("Experiment 6: displacement sweep")
    print(
        f"Sweeping displacement from {DISPLACEMENT_SWEEP_VALUES[0]:.3f} to "
        f"{DISPLACEMENT_SWEEP_VALUES[-1]:.3f} mm"
    )

    rows: list[dict[str, float | int]] = []
    details = {
        "base_params": {"E": float(DEFAULT_PARAMS[0]), "k": float(DEFAULT_PARAMS[1])},
        "displacements": [],
    }

    for displacement in DISPLACEMENT_SWEEP_VALUES:
        try:
            context = create_context(displacement=displacement, mesh=mesh)
            loss_val, grad_ad = jax.value_and_grad(context.loss)(DEFAULT_PARAMS)
            sol_list = context.solve(DEFAULT_PARAMS)
            yield_summary, _ = summarize_yield_state(context.problem.fe, sol_list[0], DEFAULT_PARAMS)
        except Exception as exc:
            print(f"\nDisplacement {displacement:.3f} mm: SOLVER FAILED ({exc}), skipping.")
            continue

        displacement_detail = {
            "displacement": displacement,
            "loss": float(loss_val),
            "grad_ad": {"E": float(grad_ad[0]), "k": float(grad_ad[1])},
            "yield_summary": yield_summary.as_dict(),
            "fd": {},
        }

        print(
            f"\nDisplacement {displacement:.3f} mm: loss={float(loss_val):.8e}, "
            f"plastic={yield_summary.plastic_count}/{yield_summary.num_quadrature_points}"
        )
        for param in PARAMETER_INFO:
            try:
                grad_fd = central_difference(context.loss, DEFAULT_PARAMS, param["index"], param["default_eps"])
            except Exception:
                grad_fd = float("nan")
            rel_err = relative_error(float(grad_ad[param["index"]]), grad_fd)
            rows.append(
                {
                    "displacement": displacement,
                    "parameter_index": param["index"],
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
            displacement_detail["fd"][param["name"]] = {
                "grad_fd": grad_fd,
                "rel_error": rel_err,
            }
            print(
                f"  dL/d{param['name']}: AD={float(grad_ad[param['index']]): .8e}, "
                f"FD={grad_fd: .8e}, rel_err={rel_err:.3e}"
            )

        details["displacements"].append(displacement_detail)

    swept_disps = sorted(set(row["displacement"] for row in rows))
    save_line_plot(
        experiment_dir / "displacement_rel_error.png",
        swept_disps,
        [
            {
                "label": "rel err dL/dE",
                "y": [row["rel_error"] for row in rows if row["parameter"] == "E"],
            },
            {
                "label": "rel err dL/dk",
                "y": [row["rel_error"] for row in rows if row["parameter"] == "k"],
            },
        ],
        xlabel="Displacement (mm)",
        ylabel="Relative error",
        title="AD vs FD relative error over displacement sweep",
        xscale="linear",
        yscale="log",
    )
    save_line_plot(
        experiment_dir / "displacement_plastic_count.png",
        swept_disps,
        [
            {
                "label": "plastic quadrature points",
                "y": [row["plastic_count"] for row in rows if row["parameter"] == "E"],
            },
        ],
        xlabel="Displacement (mm)",
        ylabel="Plastic quadrature point count",
        title="Plastic activation along the displacement sweep",
        xscale="linear",
        yscale="linear",
    )

    write_csv(experiment_dir / "displacement_sweep.csv", rows)
    write_json(experiment_dir / "displacement_sweep_details.json", details)
    print(f"\nSaved results to {experiment_dir}")


if __name__ == "__main__":
    main()
