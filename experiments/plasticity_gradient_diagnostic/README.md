# Plasticity Gradient Diagnostic

This directory implements the diagnostic plan from `ai过程文档/塑性阶段梯度诊断规划_2026-03-25.md`.

## Layout

- `common.py`: shared FEM setup, Drucker-Prager return map, diagnostics, plotting helpers.
- `exp1_fd_sweep.py`: center-difference step sweep for `E` and `k` at `u = -0.03 mm`.
- `exp2_active_set.py`: compares quadrature-point plastic/apex activity for `params +/- eps`.
- `exp3_single_point.py`: isolates the return map on representative `u_grad` samples from the FEM solution.
- `exp4_taylor_test.py`: first-order Taylor test for the full FEM loss.
- `exp5_smoothness_sweep.py`: replaces the hard plastic switch with `softplus(beta * f) / beta` and scans `beta`.
- `exp6_displacement_sweep.py`: sweeps displacement from `-0.020` to `-0.035 mm`.

## Defaults

The scripts use the same baseline values as the planning note:

- `E = 70000 MPa`
- `k = 50 MPa`
- `nu = 0.3`
- `alpha = 0.3`
- `a = 0.1 * k`
- `displacement = -0.03 mm`
- mesh: `2 x 2 x 2` HEX8 block with the same corner-stabilized boundary conditions as `tests/test_diff_dp.py`

## Run

From the repository root:

```bash
./.venv/bin/python experiments/plasticity_gradient_diagnostic/exp1_fd_sweep.py
./.venv/bin/python experiments/plasticity_gradient_diagnostic/exp2_active_set.py
./.venv/bin/python experiments/plasticity_gradient_diagnostic/exp3_single_point.py
./.venv/bin/python experiments/plasticity_gradient_diagnostic/exp4_taylor_test.py
./.venv/bin/python experiments/plasticity_gradient_diagnostic/exp5_smoothness_sweep.py
./.venv/bin/python experiments/plasticity_gradient_diagnostic/exp6_displacement_sweep.py
```

## Outputs

Each script writes CSV, JSON, and plot files to `results/plasticity_gradient_diagnostic/<experiment_name>/`.

The most useful artifacts are:

- `exp1_fd_sweep/fd_sweep.csv`: FD gradients across step sizes.
- `exp2_active_set/active_set_details.json`: quadrature points that switch active sets.
- `exp3_single_point/single_point_gradients.csv`: AD vs FD on isolated return-map samples.
- `exp4_taylor_test/taylor_test.csv`: `r0` and `r1` residuals with observed rates.
- `exp5_smoothness_sweep/smoothness_sweep.csv`: AD/FD mismatch under different softplus smoothness levels.
- `exp6_displacement_sweep/displacement_sweep.csv`: how gradient mismatch evolves with displacement.
