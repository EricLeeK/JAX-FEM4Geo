from __future__ import annotations

from common import (
    DEFAULT_DISPLACEMENT,
    DEFAULT_PARAMS,
    PARAMETER_INFO,
    RESULTS_ROOT,
    build_reference_mesh,
    copy_params_with_delta,
    create_context,
    ensure_dir,
    print_header,
    summarize_yield_state,
    switched_quadrature_points,
    write_csv,
    write_json,
)


def main() -> None:
    experiment_dir = ensure_dir(RESULTS_ROOT / "exp2_active_set")
    mesh = build_reference_mesh()
    context = create_context(displacement=DEFAULT_DISPLACEMENT, mesh=mesh)

    print_header("Experiment 2: active-set comparison")
    print(f"Displacement: {DEFAULT_DISPLACEMENT:.3f} mm")

    rows: list[dict[str, float | int | str]] = []
    details: dict[str, object] = {
        "displacement": DEFAULT_DISPLACEMENT,
        "base_params": {"E": float(DEFAULT_PARAMS[0]), "k": float(DEFAULT_PARAMS[1])},
        "comparisons": {},
    }

    for param in PARAMETER_INFO:
        comparison_key = param["name"]
        details["comparisons"][comparison_key] = {}
        variants = {
            "minus": copy_params_with_delta(DEFAULT_PARAMS, param["index"], -param["default_eps"]),
            "base": DEFAULT_PARAMS,
            "plus": copy_params_with_delta(DEFAULT_PARAMS, param["index"], param["default_eps"]),
        }
        masks = {}

        print(f"\nPerturbing {comparison_key} by +/- {param['default_eps']}")
        for label, params in variants.items():
            sol_list = context.solve(params)
            summary, diagnostics = summarize_yield_state(context.problem.fe, sol_list[0], params)
            masks[label] = {
                "is_plastic": diagnostics["is_plastic"],
                "at_apex": diagnostics["at_apex"],
            }
            details["comparisons"][comparison_key][label] = {
                "params": {"E": float(params[0]), "k": float(params[1])},
                "summary": summary.as_dict(),
            }
            rows.append(
                {
                    "comparison": comparison_key,
                    "variant": label,
                    "E": float(params[0]),
                    "k": float(params[1]),
                    **summary.as_dict(),
                }
            )
            print(
                f"  {label:>5}: plastic={summary.plastic_count:2d}/{summary.num_quadrature_points}, "
                f"apex={summary.apex_count:2d}, f_yield in [{summary.f_yield_min:.4f}, {summary.f_yield_max:.4f}]"
            )

        base_plastic = masks["base"]["is_plastic"]
        base_apex = masks["base"]["at_apex"]
        for label in ["minus", "plus"]:
            plastic_switches = switched_quadrature_points(base_plastic, masks[label]["is_plastic"])
            apex_switches = switched_quadrature_points(base_apex, masks[label]["at_apex"])
            details["comparisons"][comparison_key][f"switches_vs_base_{label}"] = {
                "plastic_switch_count": len(plastic_switches),
                "plastic_switch_points": plastic_switches,
                "apex_switch_count": len(apex_switches),
                "apex_switch_points": apex_switches,
            }
            print(
                f"  switches base->{label}: plastic={len(plastic_switches)}, apex={len(apex_switches)}"
            )

    write_csv(experiment_dir / "active_set_summary.csv", rows)
    write_json(experiment_dir / "active_set_details.json", details)
    print(f"\nSaved results to {experiment_dir}")


if __name__ == "__main__":
    main()
