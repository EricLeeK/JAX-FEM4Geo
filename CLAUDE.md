# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FEM-JAX-GEO is a differentiable geomechanics finite element package built on top of JAX-FEM. It implements constitutive models (primarily Drucker-Prager plasticity) for gradient-based optimization in geomechanics using automatic differentiation.

The project structure includes:
- Custom constitutive models built on JAX-FEM's Problem class
- Full FEM simulations with path-dependent material behavior
- Differentiability testing via AD vs FD comparison
- Post-processing utilities for comparing analytical vs numerical results

## JAX-FEM Integration

This project relies on the JAX-FEM library located in `jax-fem-main/`. JAX-FEM is a differentiable finite element package based on JAX that provides:
- AD + FEM capabilities with GPU acceleration
- 2D/3D elements (quad, triangle, hex, tetrahedron)
- First and second order elements
- Linear and nonlinear analysis (heat, elasticity, hyperelasticity, plasticity)
- Multi-physics support
- Integration with PETSc for solver options

### Key JAX-FEM Classes

- **`Problem`** (`jax_fem/problem.py`): Base class for defining FEM problems. Handles one or more coupled FE variables, mesh management, and boundary conditions.
- **`FiniteElement`** (`jax_fem/fe.py`): Core FE implementation managing mesh, shape functions, quadrature.
- **`Mesh`** (`jax_fem/generate_mesh.py`): Mesh data structure and generation utilities.
- **`solver`** (`jax_fem/solver.py`): Nonlinear solver with multiple linear solver backends (JAX, scipy UMFPACK, PETSc).

## Architecture

### Constitutive Model Implementation Pattern

Custom material models in `src/models/` inherit from `jax_fem.problem.Problem`:

1. **Initialization**: Set material parameters (E, nu, alpha, k, etc.) and pass mesh/BC info to Problem
2. **`custom_init()`**: Initialize internal variables (stress/strain history arrays)
3. **`get_tensor_map()`**: Return the stress computation function for the solver
4. **`get_maps()`**: Define the core constitutive relations:
   - `strain(u_grad)`: Compute strain from displacement gradient
   - `stress_return_map(u_grad, sigma_old, epsilon_old)`: Compute stress using return mapping algorithm
5. **`update_stress_strain(sol)`**: Update internal variables after each load step
6. **Helper methods**: Problem-specific utilities like `compute_avg_stress()`

The stress return mapping must handle:
- Elastic trial stress
- Yield function evaluation
- Plastic correction (return to yield surface)
- Safe mathematical operations (avoiding division by zero, negative square roots)

### Path Setup Pattern

Scripts use this pattern to ensure `jax-fem-main` is in the Python path:

```python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
```

## Common Development Commands

### Running Simulations

```bash
# Full FEM simulation of 3D block under compression
python examples/run_fem_simulation.py

# Single-point constitutive verification driver
python examples/run_constitutive_driver.py
```

### Running Tests

Tests verify differentiability (AD vs FD) for various constitutive model implementations:

```bash
# Run specific test
python tests/test_diff_dp.py                    # Basic Drucker-Prager differentiability
python tests/test_diff_elastic.py               # Elastic model
python tests/test_diff_dp_strategy1_visco.py    # Viscoplastic regularization
python tests/test_diff_dp_strategy2_direct.py   # Direct differentiation
python tests/test_diff_dp_strategy3_smooth.py   # Smooth yield surface
python tests/test_diff_dp_hardening.py          # Hardening model
python tests/test_diff_dp_bc_fix.py             # BC handling

# Run all tests in the directory
python -m pytest tests/
```

### Post-Processing

```bash
# Compare FEM results with analytical solutions
python scripts/compare_results.py
```

## Boundary Condition Patterns

JAX-FEM uses a three-element list structure for Dirichlet BCs:

```python
dirichlet_bc_info = [location_fns, vecs, value_fns]
```

Where:
- `location_fns`: List of callables that identify BC locations (return bool for each point)
- `vecs`: List of component indices (0=x, 1=y, 2=z for 3D problems)
- `value_fns`: List of callables that return the prescribed value

Example for compression test:
```python
def bottom(point): return np.isclose(point[2], 0., atol=1e-5)
def top(point): return np.isclose(point[2], Lz, atol=1e-5)

location_fns = [bottom, top]
vecs = [2, 2]  # z-component for both
value_fns = [lambda p: 0., lambda p: displacement]
```

## Solver Configuration

The `solver()` function accepts options for different linear solver backends:

```python
# PETSc solver (recommended for large problems)
solver_options = {'petsc_solver': {}}
sol_list = solver(problem, solver_options=solver_options)

# JAX solver (iterative, GPU-compatible)
solver_options = {'jax_solver': {}}

# Scipy UMFPACK (direct solver)
solver_options = {'umfpack_solver': {}}
```

## Incremental Loading Pattern

For path-dependent materials, simulations use incremental loading:

1. Define displacement history (load/unload cycles)
2. Loop over load steps
3. Update boundary conditions for current step
4. Solve equilibrium equations
5. Update internal variables (stress/strain history)
6. Save results (VTK files, average stress)

## Output Organization

Results are typically saved in `results/YYYY-MM-DD_description/`:
- `vtk/`: VTK files for visualization (one per load step)
- `fem_data.csv`: Stress-strain or other tabular data
- `*.png`: Plots generated during post-processing

## JAX-FEM Application Examples

The `jax-fem-main/applications/` directory contains reference implementations:
- `stokes/`: Stokes flow (multi-variable problem)
- `wave/`: Wave equation (time-dependent)
- `crystal_plasticity/`: Crystal plasticity models
- `phase_field_fracture/`: Phase field fracture mechanics
- `thermal_mechanical/`: Coupled thermomechanical analysis
- `periodic_bc/`: Periodic boundary conditions

These can serve as templates for new problem types.
