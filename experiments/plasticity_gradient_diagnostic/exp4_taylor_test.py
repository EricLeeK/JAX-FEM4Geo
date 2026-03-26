from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as onp

from common import (
    DEFAULT_DISPLACEMENT,
    DEFAULT_PARAMS,
    RESULTS_ROOT,
    TAYLOR_EPS_VALUES,
    build_reference_mesh,
    convergence_rate,
    create_context,
    ensure_dir,
    print_header,
    save_line_plot,
    write_csv,
    write_json,
)


def main() -> None:
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp4_taylor_test")
    mesh = build_reference_mesh()
    context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh)

    print_header("Experiment 4: Taylor test")
    loss_0, grad_ad = jax.value_and_grad(context.loss)(DEFAULT_PARAMS)
    rng = onp.random.default_rng(0)
    raw_direction = rng.normal(size=2)
    direction = jnp.asarray(jnp.array([100.0, 1.0]) * raw_direction / onp.linalg.norm(raw_direction))
    grad_dot_direction = float(jnp.dot(grad_ad, direction))

    print(f"Displacement: {DEFAULT_DISPLACEMENT:.3f} mm")
    print(f"Base loss: {float(loss_0):.8e}")
    print(f"Direction: [{float(direction[0]):.8e}, {float(direction[1]):.8e}]")
    print(f"grad_ad dot direction: {grad_dot_direction:.8e}")

    rows: list[dict[str, float | None]] = []
    prev_eps = None
    prev_r0 = None
    prev_r1 = None
    for eps in TAYLOR_EPS_VALUES:
        loss_eps = context.loss(DEFAULT_PARAMS + eps * direction)
        r0 = float(abs(loss_eps - loss_0))
        r1 = float(abs(loss_eps - loss_0 - eps * grad_dot_direction))
        row = {
            "eps": eps,
            "loss_eps": float(loss_eps),
            "r0": r0,
            "r1": r1,
            "rate_r0": None,
            "rate_r1": None,
        }
        if prev_eps is not None and prev_r0 is not None and prev_r1 is not None:
            row["rate_r0"] = convergence_rate(prev_eps, prev_r0, eps, r0)
            row["rate_r1"] = convergence_rate(prev_eps, prev_r1, eps, r1)
        rows.append(row)
        prev_eps = eps
        prev_r0 = r0
        prev_r1 = r1
        rate_r0_str = "-" if row["rate_r0"] is None else f"{row['rate_r0']:.3f}"
        rate_r1_str = "-" if row["rate_r1"] is None else f"{row['rate_r1']:.3f}"
        print(
            f"  eps={eps:.1e}: r0={r0:.8e} (rate {rate_r0_str}), "
            f"r1={r1:.8e} (rate {rate_r1_str})"
        )

    save_line_plot(
        experiment_dir / "taylor_residuals.png",
        TAYLOR_EPS_VALUES,
        [
            {"label": "r0 = |L(p + eps d) - L(p)|", "y": [row["r0"] for row in rows]},
            {
                "label": "r1 = |L(p + eps d) - L(p) - eps grad.d|",
                "y": [row["r1"] for row in rows],
            },
        ],
        xlabel="eps",
        ylabel="Residual",
        title="Taylor test residuals",
        xscale="log",
        yscale="log",
    )

    write_csv(experiment_dir / "taylor_test.csv", rows)
    write_json(
        experiment_dir / "summary.json",
        {
            "displacement": DEFAULT_DISPLACEMENT,
            "loss_0": float(loss_0),
            "grad_ad": {"E": float(grad_ad[0]), "k": float(grad_ad[1])},
            "direction": {"E": float(direction[0]), "k": float(direction[1])},
            "grad_dot_direction": grad_dot_direction,
        },
    )
    print(f"\nSaved results to {experiment_dir}")


if __name__ == "__main__":
    main()
