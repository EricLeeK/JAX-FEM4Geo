# FEM-JAX-GEO: Differentiable Geomechanics with JAX-FEM

## Project Overview
This project implements constitutive models (specifically Drucker-Prager plasticity) and finite element simulations using the `jax-fem` library. It focuses on differentiability for gradient-based optimization in geomechanics.

## Directory Structure

*   `src/`: Core source code.
    *   `src/models/`: Constitutive model definitions (e.g., `drucker_prager.py`).
*   `examples/`: Runnable simulation scripts.
    *   `run_fem_simulation.py`: Full FEM simulation of a 3D block.
    *   `run_constitutive_driver.py`: Single-point verification driver.
*   `tests/`: Unit and integration tests.
    *   `test_diff_dp.py`: Differentiability (AD vs FD) tests for the DP model.
*   `scripts/`: Utility scripts.
    *   `compare_results.py`: Post-processing to compare FEM vs Analytical results.
*   `docs/`: Documentation and reports.
*   `results/`: Simulation outputs (images, CSVs, VTK).

## Getting Started

### Prerequisites
*   Python 3.10+ (3.11 recommended)
*   JAX, JAX-FEM (located in `jax-fem-main`)
*   NumPy, SciPy, Matplotlib, meshio, gmsh, fenics-basix

### Setup (macOS / CPU-only, PETSc-free)

The project ships a patched `jax-fem` in which PETSc is **optional** (see
`jax-fem-main/jax_fem/solver.py`): a scipy-backed `SparseMat` replaces the
PETSc matrix container, so all solvers except `petsc_solver` work without
`petsc4py`. This makes the project run on macOS out of the box. On Linux/GPU
you can instead install `petsc4py` and use `petsc_solver` + MUMPS for large
problems.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-mac.txt
uv pip install -e jax-fem-main/
```

### Running the FEM Simulation
```bash
python examples/run_fem_simulation.py
```

### Running the Verification Driver
```bash
python examples/run_constitutive_driver.py
```

### Comparing Results
```bash
python scripts/compare_results.py
```

### Running Tests
```bash
python tests/test_diff_dp.py
```

### Gradient Check (AD vs FD) — P0-3

Verifies that the adjoint-based automatic differentiation through the
Drucker-Prager FEM solve matches a finite-difference reference, on macOS with
the `umfpack` (scipy) backend:

```bash
python tests/test_diff_dp_mac.py
```

### Parameter Inversion — P0-1 / P0-2

The differentiable-FEM payoff: recover Drucker-Prager `(E, k)` from synthetic
triaxial reaction data by gradient-based optimization (direct differentiation,
log-space L-BFGS). Generates truth data, inverts from a perturbed initial
guess, and runs a robustness sweep over multiple starts.

```bash
python -m src.inversion.run_inversion
```
