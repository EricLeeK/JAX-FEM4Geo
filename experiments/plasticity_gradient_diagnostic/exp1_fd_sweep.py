from __future__ import annotations

import jax
import numpy as onp

from common import (
    DEFAULT_DISPLACEMENT,
    DEFAULT_PARAMS,
    FD_SWEEP_EPS_VALUES,
    PARAMETER_INFO,
    RESULTS_ROOT,
    build_reference_mesh,
    central_difference,
    create_context,
    ensure_dir,
    print_header,
    relative_error,
    save_line_plot,
    write_csv,
    write_json,
)


def main() -> None:
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp1_fd_sweep")
    mesh = build_reference_mesh()
    context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh)

    print_header("Experiment 1: FD step sweep")
    loss_val, grad_ad = jax.value_and_grad(context.loss)(DEFAULT_PARAMS)
    grad_ad_np = onp.asarray(grad_ad)
    print(f"Base displacement: {DEFAULT_DISPLACEMENT:.3f} mm")
    print(f"Base loss: {float(loss_val):.8e}")
    print(f"AD gradient: dL/dE={grad_ad_np[0]:.8e}, dL/dk={grad_ad_np[1]:.8e}")

    rows: list[dict[str, float | str]] = []
    summary = {
        "displacement": DEFAULT_DISPLACEMENT,
        "loss": float(loss_val),
        "grad_ad": {"E": float(grad_ad_np[0]), "k": float(grad_ad_np[1])},
    }

    for param in PARAMETER_INFO:
        fd_values = []
        print(f"\nParameter {param['name']}:")
        for eps in FD_SWEEP_EPS_VALUES:
            grad_fd = central_difference(context.loss, DEFAULT_PARAMS, param["index"], eps)
            fd_values.append(grad_fd)
            row = {
                "parameter": param["name"],
                "eps": eps,
                "grad_fd": grad_fd,
                "grad_ad": float(grad_ad_np[param["index"]]),
                "abs_error": abs(grad_fd - float(grad_ad_np[param["index"]])),
                "rel_error": relative_error(float(grad_ad_np[param["index"]]), grad_fd),
            }
            rows.append(row)
            print(
                f"  eps={eps:8.3f} -> FD={grad_fd: .8e}, "
                f"rel_err={row['rel_error']:.3e}"
            )

        save_line_plot(
            experiment_dir / f"fd_sweep_{param['name']}.png",
            FD_SWEEP_EPS_VALUES,
            [
                {"label": f"FD dL/d{param['name']}", "y": fd_values},
                {
                    "label": f"AD dL/d{param['name']}",
                    "y": [float(grad_ad_np[param['index']])] * len(FD_SWEEP_EPS_VALUES),
                    "linestyle": "--",
                },
            ],
            xlabel="FD step size",
            ylabel=f"Gradient wrt {param['name']}",
            title=f"FD sweep for {param['name']} at displacement {DEFAULT_DISPLACEMENT:.3f} mm",
            xscale="log",
            yscale="linear",
        )

    write_csv(experiment_dir / "fd_sweep.csv", rows)
    write_json(experiment_dir / "summary.json", summary)
    print(f"\nSaved results to {experiment_dir}")


if __name__ == "__main__":
    main()
