# FEM-JAX-GEO: Differentiable Geomechanics with JAX-FEM

## Project Overview

A differentiable geomechanics finite element package built on JAX-FEM. Implements Drucker-Prager plasticity with full gradient support (AD) for parameter inversion and gradient-based optimization.

Key capabilities:
- Drucker-Prager elastoplastic constitutive model with return mapping
- Differentiable FEM via `internal_vars` parameter passing (compatible with JAX JIT)
- Incremental loading with path-dependent state tracking
- AD vs FD verified from elastic through plastic regime, single-step and multi-step

## Directory Structure

```
src/models/
    drucker_prager.py       # Main DP model (internal_vars mode, gradient-safe)
examples/
    run_fem_simulation.py   # 3D block compression (incremental load/unload)
    run_constitutive_driver.py  # Single-point constitutive verification
    run_triaxial_final.py   # Standard triaxial test (cubic specimen)
    run_triaxial_cylinder.py    # Triaxial test (cylindrical specimen, H:D=2:1)
tests/
    test_diff_dp.py         # AD vs FD for DP model (elastic + plastic)
    test_diff_dp_multistep.py   # Multi-step incremental loading gradient test
    test_diff_elastic.py    # AD vs FD for linear elasticity
scripts/
    compare_results.py      # FEM vs driver comparison (slopes + point-by-point)
docs/                       # Theory notes, reports, thesis documents
results/                    # Simulation outputs (VTK, CSV, PNG)
experiments/                # Gradient diagnostic experiments (archived)
```

## Getting Started

### Prerequisites
- Python 3.10+
- JAX, JAX-FEM (in `jax-fem-main/`)
- NumPy, SciPy, Matplotlib

### Running Simulations
```bash
python examples/run_fem_simulation.py       # Full FEM simulation
python examples/run_constitutive_driver.py  # Single-point verification
```

### Running Tests
```bash
python -m pytest tests/ -v                  # All tests (4 tests)
python tests/test_diff_dp.py               # DP gradient test only
```

### Comparing Results
```bash
python scripts/compare_results.py           # FEM vs driver (PASS/FAIL verdict)
```
