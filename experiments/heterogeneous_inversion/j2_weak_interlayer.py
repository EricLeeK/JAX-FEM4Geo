"""
J2: Weak interlayer (软弱夹层) inversion experiment.

Tests whether the differentiable FEM framework can identify a thin weak zone
embedded in a strong background material.

Setup:
- 2D rectangular domain 10×10, 30×30 QUAD4 mesh (900 elements)
- True E field: background E=80000, diagonal weak interlayer E=20000
  - Interlayer: 45° band through domain center, width ≈ 1.5 units
  - Region defined by |cy - cx| < width/2
- Fixed k=50
- Top compression BC (displacement=-0.1), full-field observation + 1% noise
- TV regularization (λ=0.01, E_ref=70000)
- L-BFGS-B optimizer, maxiter=100

Expected: recover the diagonal weak interlayer geometry.
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from common import (
    InversionHeterogeneousDP2D,
    create_2d_mesh_and_bc,
    generate_synthetic_observation,
    displacement_loss,
    optimize_lbfgsb,
    plot_E_field_comparison,
    compute_inversion_metrics,
    RESULTS_DIR,
)
from jax_fem.solver import ad_wrapper
from src.regularization import build_structured_neighbor_pairs, tv_regularizer


def build_interlayer_E_field(Nx, Ny, Lx, Ly, angle_deg=45, width=1.5,
                             E_bg=80000.0, E_weak=20000.0):
    """Build E field with a diagonal weak interlayer.

    The interlayer is a band at `angle_deg` passing through the domain center.
    For 45°, the interlayer region is where |cy - cx| < width/2.

    Cell ordering: cell_idx = ix * Ny + iy (matches rectangle_mesh).

    Parameters
    ----------
    Nx, Ny : int, mesh divisions
    Lx, Ly : float, domain size
    angle_deg : float, interlayer angle (currently supports 45°)
    width : float, interlayer width
    E_bg : float, background Young's modulus
    E_weak : float, interlayer Young's modulus

    Returns
    -------
    E_field : (num_cells,) JAX array
    """
    nc = Nx * Ny
    E_field = onp.full(nc, E_bg)
    half_w = width / 2.0
    for ix in range(Nx):
        for iy in range(Ny):
            cx = (ix + 0.5) * Lx / Nx
            cy = (iy + 0.5) * Ly / Ny
            if abs(cy - cx) < half_w:
                cell_idx = ix * Ny + iy
                E_field[cell_idx] = E_weak
    return np.array(E_field)


def run_interlayer_experiment(Nx=30, Ny=30, Lx=10.0, Ly=10.0,
                              displacement=-0.1, k_fixed=50.0,
                              angle_deg=45, width=1.5,
                              E_bg=80000.0, E_weak=20000.0,
                              noise_level=0.01, reg_weight=0.01,
                              E_ref=70000.0, maxiter=100):
    """Run the weak interlayer inversion experiment."""
    print("=" * 70)
    print("J2: Weak Interlayer (软弱夹层) Inversion")
    print("=" * 70)

    nc = Nx * Ny

    # --- Build true E field ---
    E_true = build_interlayer_E_field(Nx, Ny, Lx, Ly, angle_deg=angle_deg,
                                      width=width, E_bg=E_bg, E_weak=E_weak)
    interlayer_mask = onp.array(E_true) < (E_bg + E_weak) / 2.0
    n_interlayer = int(interlayer_mask.sum())
    print(f"\nProblem setup:")
    print(f"  Mesh: {Nx}×{Ny} = {nc} QUAD4 elements")
    print(f"  Domain: {Lx}×{Ly}")
    print(f"  Displacement: {displacement}")
    print(f"  k (fixed): {k_fixed}")
    print(f"  E_bg={E_bg:.0f}, E_weak={E_weak:.0f}")
    print(f"  Interlayer: {angle_deg}° diagonal, width={width}")
    print(f"  Interlayer cells: {n_interlayer}/{nc} "
          f"({100 * n_interlayer / nc:.1f}%)")
    print(f"  Noise level: {noise_level * 100:.0f}%")
    print(f"  TV regularization: λ={reg_weight}, E_ref={E_ref}")
    print(f"  Optimizer: L-BFGS-B, maxiter={maxiter}")

    # --- Create mesh and problem ---
    mesh, bc_info = create_2d_mesh_and_bc(displacement, Lx=Lx, Ly=Ly,
                                          Nx=Nx, Ny=Ny)
    problem = InversionHeterogeneousDP2D(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=bc_info, E_init=70000.0, k=k_fixed,
    )

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options,
                          adjoint_solver_options=solver_options)

    # --- Generate synthetic observation with noise ---
    print("\nGenerating synthetic observation...")
    t0 = time.time()
    key = jax.random.PRNGKey(42)
    obs_data, obs_indices, sol_true = generate_synthetic_observation(
        problem, fwd_pred, E_true,
        noise_level=noise_level, key=key,
    )
    print(f"  Forward solve: {time.time() - t0:.2f}s")
    print(f"  Observation: {obs_data.shape[0]} nodes × {obs_data.shape[1]} components")
    print(f"  Max displacement: {float(np.max(np.abs(sol_true))):.6e}")

    # --- Build TV regularization ---
    neighbor_pairs = build_structured_neighbor_pairs(Nx, Ny)
    print(f"  Neighbor pairs: {len(neighbor_pairs)}")
    neighbor_pairs_jax = np.array(neighbor_pairs)

    regularizer = lambda E: tv_regularizer(E, neighbor_pairs_jax, E_ref=E_ref)

    # --- Define loss ---
    def loss_fn(E_field):
        return displacement_loss(E_field, problem, fwd_pred, obs_data, obs_indices,
                                 regularizer=regularizer, reg_weight=reg_weight)

    # --- Sanity checks ---
    loss_at_true = float(loss_fn(E_true))
    print(f"\n  Loss at true E: {loss_at_true:.6e}")

    E_init = np.full(nc, 70000.0)
    loss_at_init = float(loss_fn(E_init))
    print(f"  Loss at init E: {loss_at_init:.6e}")

    # --- Gradient check ---
    print("\nGradient check...")
    t0 = time.time()
    grad = jax.grad(loss_fn)(E_init)
    print(f"  Grad computed in {time.time() - t0:.2f}s")
    print(f"  |grad|: {float(np.linalg.norm(grad)):.4e}")
    print(f"  grad range: [{float(np.min(grad)):.4e}, {float(np.max(grad)):.4e}]")

    # --- Run inversion ---
    print(f"\nRunning L-BFGS-B optimization (maxiter={maxiter})...")
    result = optimize_lbfgsb(loss_fn, E_init, E_min=10000.0, E_max=200000.0,
                              maxiter=maxiter)

    E_final = result['E_final']

    # --- Metrics ---
    metrics = compute_inversion_metrics(E_true, E_final)
    print(f"\nInversion results:")
    print(f"  Final loss: {result['loss_history'][-1]:.6e}")
    print(f"  L2 relative error: {metrics['L2_relative_error']:.4e}")
    print(f"  Mean relative error: {metrics['mean_relative_error']:.4e}")
    print(f"  Max relative error: {metrics['max_relative_error']:.4e}")
    print(f"  Time: {result['time_s']:.1f}s")
    print(f"  Iterations: {result['nit']}")

    # --- Interlayer identification metrics ---
    E_final_np = onp.array(E_final)
    E_true_np = onp.array(E_true)
    threshold = (E_bg + E_weak) / 2.0  # midpoint between weak and background

    detected_weak = E_final_np < threshold
    bg_mask = ~interlayer_mask

    # Interlayer detection rate (sensitivity / recall)
    true_positives = onp.sum(detected_weak & interlayer_mask)
    detection_rate = float(true_positives) / max(n_interlayer, 1)

    # False positive rate (background cells misidentified as weak)
    false_positives = onp.sum(detected_weak & bg_mask)
    n_bg = int(bg_mask.sum())
    false_positive_rate = float(false_positives) / max(n_bg, 1)

    print(f"\n  Interlayer identification (threshold={threshold:.0f}):")
    print(f"    Detection rate: {detection_rate:.1%} "
          f"({true_positives}/{n_interlayer} interlayer cells)")
    print(f"    False positive rate: {false_positive_rate:.1%} "
          f"({false_positives}/{n_bg} background cells)")

    # Region statistics
    print(f"\n  Interlayer region (true={E_weak:.0f}):")
    print(f"    mean={E_final_np[interlayer_mask].mean():.1f}, "
          f"std={E_final_np[interlayer_mask].std():.1f}")
    print(f"  Background region (true={E_bg:.0f}):")
    print(f"    mean={E_final_np[bg_mask].mean():.1f}, "
          f"std={E_final_np[bg_mask].std():.1f}")

    # --- Save results ---
    exp_dir = os.path.join(RESULTS_DIR, 'j2_weak_interlayer')
    os.makedirs(exp_dir, exist_ok=True)

    plot_E_field_comparison(
        E_true, E_final, Nx, Ny, Lx, Ly,
        title=f'J2: Weak Interlayer Inversion ({Nx}×{Ny}, TV λ={reg_weight})',
        save_path=os.path.join(exp_dir, 'E_field.png'),
    )

    # Convergence plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(result['loss_history'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('J2: Convergence')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'convergence.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        pass

    # Save metrics to JSON
    save_data = {
        'Nx': Nx, 'Ny': Ny, 'num_cells': nc,
        'Lx': Lx, 'Ly': Ly,
        'displacement': displacement, 'k_fixed': k_fixed,
        'E_bg': E_bg, 'E_weak': E_weak,
        'angle_deg': angle_deg, 'width': width,
        'n_interlayer_cells': n_interlayer,
        'noise_level': noise_level,
        'reg_type': 'tv', 'reg_weight': reg_weight, 'E_ref': E_ref,
        'optimizer': 'lbfgs', 'maxiter': maxiter,
        'metrics': metrics,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'time_s': result['time_s'],
        'nit': result['nit'],
        'final_loss': result['loss_history'][-1],
        'loss_at_true': loss_at_true,
        'interlayer_mean': float(E_final_np[interlayer_mask].mean()),
        'interlayer_std': float(E_final_np[interlayer_mask].std()),
        'background_mean': float(E_final_np[bg_mask].mean()),
        'background_std': float(E_final_np[bg_mask].std()),
    }
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to {exp_dir}")
    print("=" * 70)

    return result, metrics


if __name__ == "__main__":
    run_interlayer_experiment(
        Nx=30, Ny=30, Lx=10.0, Ly=10.0,
        displacement=-0.1, k_fixed=50.0,
        angle_deg=45, width=1.5,
        E_bg=80000.0, E_weak=20000.0,
        noise_level=0.01, reg_weight=0.01,
        E_ref=70000.0, maxiter=100,
    )
