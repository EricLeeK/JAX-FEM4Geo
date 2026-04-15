#!/usr/bin/env python
"""
H5: First heterogeneous inversion experiment — Two-region E field.

Setup:
- 20×20 QUAD4 mesh (400 elements), 10×10 domain
- True E: left half = 50000 MPa, right half = 90000 MPa
- Fixed k = 50 MPa
- Observation: full-field displacement, no noise
- Optimizer: L-BFGS-B in log-E space
- Initial guess: uniform E = 70000 MPa

This validates that the heterogeneous inversion framework can recover
a sharp two-region interface from displacement data.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from common import (
    InversionHeterogeneousDP2D,
    create_2d_mesh_traction_bc,
    generate_synthetic_observation,
    make_heterogeneous_loss,
    smoothness_regularizer,
    two_region_E_field,
    uniform_E_field,
    log_to_E,
    E_to_log,
    get_cell_centroids,
    plot_E_field,
    save_results,
    RESULTS_DIR,
)

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
from jax_fem.solver import ad_wrapper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
Nx, Ny = 20, 20
Lx, Ly = 10., 10.
TRACTION = -50.0  # Compressive traction on top [MPa]
K_FIXED = 500.0
E_LEFT, E_RIGHT = 50000., 90000.
E_INIT = 70000.
REG_WEIGHT = 0.0  # No regularization for first test (sharp interface)

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(RESULTS_DIR, 'h5_two_region')
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CELLS = Nx * Ny


def main():
    print("=" * 70)
    print("H5: HETEROGENEOUS INVERSION — TWO-REGION E FIELD")
    print(f"  Mesh: {Nx}×{Ny} = {NUM_CELLS} elements")
    print(f"  True E: left={E_LEFT}, right={E_RIGHT}")
    print(f"  Fixed k = {K_FIXED}")
    print(f"  Traction = {TRACTION} MPa")
    print("=" * 70)
    t0_total = time.time()

    # --- Step 1: Generate synthetic observation ---
    print("\n[1] Generating synthetic observation...")
    E_true = two_region_E_field(Nx, Ny, E_LEFT, E_RIGHT)
    print(f"  E_true: min={E_true.min():.0f}, max={E_true.max():.0f}")

    obs_data = generate_synthetic_observation(
        E_true_field=E_true,
        k_fixed=K_FIXED,
        traction=TRACTION,
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        obs_node_indices=None,  # full-field
        noise_level=0.0,
        solver_options=SOLVER_OPTIONS,
    )
    u_obs = obs_data['u_obs']
    obs_indices = obs_data['obs_indices']
    u_full_true = obs_data['u_full']
    print(f"  Observation: {u_obs.shape[0]} nodes, ||u_obs|| = {float(np.linalg.norm(u_obs)):.6e}")

    # --- Step 2: Set up inversion problem ---
    print("\n[2] Setting up inversion problem...")
    mesh, bc_info, loc_fns, _ = create_2d_mesh_traction_bc(
        TRACTION, Lx, Ly, Nx, Ny)

    inv_problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info,
        location_fns=loc_fns,
        E=E_INIT, nu=0.3, alpha=0.3, k=K_FIXED,
        traction_value=TRACTION,
    )
    fwd_pred = ad_wrapper(inv_problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    # Build loss function (in log-E space)
    reg_fn = smoothness_regularizer(Nx, Ny) if REG_WEIGHT > 0 else None
    loss_fn = make_heterogeneous_loss(
        fwd_pred, u_obs, obs_indices,
        regularizer=reg_fn, reg_weight=REG_WEIGHT,
    )

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    # --- Step 3: Warm up JIT ---
    print("\n[3] Warming up JIT...")
    log_E_init = E_to_log(uniform_E_field(Nx, Ny, E_INIT))
    t_warmup_start = time.time()
    loss_val, grad_val = value_and_grad_fn(np.array(log_E_init))
    t_warmup = time.time() - t_warmup_start
    print(f"  Initial loss: {float(loss_val):.6e}")
    print(f"  ||grad||: {float(np.linalg.norm(grad_val)):.6e}")
    print(f"  JIT warmup: {t_warmup:.1f}s")

    # --- Step 4: L-BFGS-B optimization ---
    print("\n[4] Running L-BFGS-B optimization...")
    from scipy.optimize import minimize as scipy_minimize

    history = {'loss': [], 'grad_norm': [], 'wallclock': []}
    t_opt_start = time.time()

    # Bounds in log-space: E in [1000, 500000] -> log_E in [log(1000), log(500000)]
    log_E_lo = float(onp.log(1000.))
    log_E_hi = float(onp.log(500000.))
    bounds = [(log_E_lo, log_E_hi)] * NUM_CELLS

    def objective(x):
        """Scipy-compatible objective: returns (loss, grad) as float64 arrays."""
        log_E = np.array(x)
        loss, grad = value_and_grad_fn(log_E)
        loss_f = float(loss)
        grad_np = onp.array(grad, dtype=onp.float64)
        grad_norm = float(onp.linalg.norm(grad_np))

        history['loss'].append(loss_f)
        history['grad_norm'].append(grad_norm)
        history['wallclock'].append(time.time() - t_opt_start)

        step = len(history['loss'])
        if step <= 5 or step % 10 == 0:
            E_cur = onp.exp(x)
            print(f"    Eval {step:4d}: loss={loss_f:.6e}  "
                  f"||grad||={grad_norm:.3e}  "
                  f"E range=[{E_cur.min():.0f}, {E_cur.max():.0f}]")

        return loss_f, grad_np

    result = scipy_minimize(
        objective,
        x0=onp.array(log_E_init, dtype=onp.float64),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-20, 'gtol': 1e-12},
    )

    t_opt = time.time() - t_opt_start
    log_E_final = result.x
    E_final = onp.exp(log_E_final)

    print(f"\n  Optimization completed:")
    print(f"    Converged: {result.success}")
    print(f"    Message: {result.message}")
    print(f"    Iterations: {result.nit}")
    print(f"    Function evaluations: {result.nfev}")
    print(f"    Time: {t_opt:.1f}s")
    print(f"    Final loss: {result.fun:.6e}")

    # --- Step 5: Evaluate results ---
    print("\n[5] Evaluating results...")
    E_true_arr = onp.array(E_true)
    E_error = onp.abs(E_final - E_true_arr)
    E_rel_error = E_error / E_true_arr
    l2_error = float(onp.sqrt(onp.mean(E_error ** 2)))
    l2_rel_error = float(onp.sqrt(onp.mean(E_rel_error ** 2)))
    max_error = float(onp.max(E_error))
    max_rel_error = float(onp.max(E_rel_error))
    mean_error = float(onp.mean(E_error))

    print(f"  E field recovery:")
    print(f"    E_true:  [{E_true_arr.min():.0f}, {E_true_arr.max():.0f}]")
    print(f"    E_inv:   [{E_final.min():.0f}, {E_final.max():.0f}]")
    print(f"    L2 error (abs): {l2_error:.2f} MPa")
    print(f"    L2 error (rel): {l2_rel_error:.4%}")
    print(f"    Max error (abs): {max_error:.2f} MPa")
    print(f"    Max error (rel): {max_rel_error:.4%}")
    print(f"    Mean error (abs): {mean_error:.2f} MPa")

    # Left/right region accuracy
    left_mask = E_true_arr < (E_LEFT + E_RIGHT) / 2
    right_mask = ~left_mask
    left_err = float(onp.mean(onp.abs(E_final[left_mask] - E_LEFT) / E_LEFT))
    right_err = float(onp.mean(onp.abs(E_final[right_mask] - E_RIGHT) / E_RIGHT))
    print(f"    Left region (E={E_LEFT:.0f}) mean rel error: {left_err:.4%}")
    print(f"    Right region (E={E_RIGHT:.0f}) mean rel error: {right_err:.4%}")

    # --- Step 6: Plotting ---
    print("\n[6] Generating plots...")
    vmin = min(E_LEFT, E_final.min()) * 0.95
    vmax = max(E_RIGHT, E_final.max()) * 1.05

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # True E field
    plot_E_field(E_true, Nx, Ny, ax=axes[0, 0], title='True E field',
                 vmin=vmin, vmax=vmax)

    # Recovered E field
    plot_E_field(E_final, Nx, Ny, ax=axes[0, 1], title='Recovered E field',
                 vmin=vmin, vmax=vmax)

    # Error field
    plot_E_field(E_error, Nx, Ny, ax=axes[0, 2], title='|E_true - E_inv| error')

    # Convergence curve
    ax = axes[1, 0]
    ax.semilogy(history['wallclock'], history['loss'], 'b-', linewidth=1.5)
    ax.set_xlabel('Wall-clock time [s]')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence: Loss vs Time')
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 1]
    ax.semilogy(range(1, len(history['grad_norm']) + 1),
                history['grad_norm'], 'r-', linewidth=1.5)
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('||grad||')
    ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3)

    # Line cut at mid-height (iy = Ny//2)
    ax = axes[1, 2]
    iy_mid = Ny // 2
    E_true_line = E_true_arr.reshape(Nx, Ny)[:, iy_mid]
    E_inv_line = E_final.reshape(Nx, Ny)[:, iy_mid]
    x_centers = onp.linspace(Lx / (2 * Nx), Lx - Lx / (2 * Nx), Nx)
    ax.plot(x_centers, E_true_line, 'k-', linewidth=2, label='True')
    ax.plot(x_centers, E_inv_line, 'b--o', markersize=3, linewidth=1.5, label='Inverted')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('E [MPa]')
    ax.set_title(f'Line cut at y = {Ly/2:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'H5: Two-Region E Field Inversion ({Nx}×{Ny} mesh, '
                 f'L2 rel err = {l2_rel_error:.2%})', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'h5_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")
    plt.close()

    # --- Step 7: Save results ---
    t_total = time.time() - t0_total
    results = {
        'config': {
            'Nx': Nx, 'Ny': Ny, 'num_cells': NUM_CELLS,
            'Lx': Lx, 'Ly': Ly,
            'traction': TRACTION,
            'E_left': E_LEFT, 'E_right': E_RIGHT,
            'E_init': E_INIT, 'k_fixed': K_FIXED,
            'reg_weight': REG_WEIGHT,
        },
        'optimization': {
            'method': 'L-BFGS-B',
            'converged': bool(result.success),
            'message': str(result.message),
            'n_iterations': int(result.nit),
            'n_function_evals': int(result.nfev),
            'final_loss': float(result.fun),
            'time_s': t_opt,
        },
        'errors': {
            'l2_abs': l2_error,
            'l2_rel': l2_rel_error,
            'max_abs': max_error,
            'max_rel': max_rel_error,
            'mean_abs': mean_error,
            'left_region_mean_rel': left_err,
            'right_region_mean_rel': right_err,
        },
        'timing': {
            'jit_warmup_s': t_warmup,
            'optimization_s': t_opt,
            'total_s': t_total,
        },
        'E_true': E_true.tolist(),
        'E_recovered': E_final.tolist(),
    }
    json_path = save_results(OUT_DIR, results)
    print(f"  Saved: {json_path}")

    # Also save numpy arrays for post-processing
    onp.save(os.path.join(OUT_DIR, 'E_true.npy'), E_true)
    onp.save(os.path.join(OUT_DIR, 'E_recovered.npy'), E_final)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("H5 SUMMARY")
    print("=" * 70)
    print(f"  Mesh: {Nx}×{Ny} = {NUM_CELLS} elements")
    print(f"  Converged: {result.success}")
    print(f"  L2 relative error: {l2_rel_error:.4%}")
    print(f"  Left region error:  {left_err:.4%}")
    print(f"  Right region error: {right_err:.4%}")
    print(f"  Total time: {t_total:.1f}s")
    status = "PASS" if l2_rel_error < 0.10 else "NEEDS IMPROVEMENT"
    print(f"  Status: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
