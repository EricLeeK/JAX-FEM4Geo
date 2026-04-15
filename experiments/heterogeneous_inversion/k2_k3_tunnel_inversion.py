"""
K2 + K3a-e: Tunnel excavation inversion experiments.

K2: Observation point setup
  - Full-field: all nodes
  - Sparse: tunnel perimeter + surface settlement points

K3a: Uniform rock mass, full-field observation (baseline)
K3b: Uniform rock mass, sparse observation (realistic)
K3c: Two-zone rock mass (soft near-tunnel zone)
K3d: Graded E field (weathering gradient)
K3e: Noise robustness test (1%, 3%, 5% noise on K3a)
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_fem.solver import solver, ad_wrapper
from k1_tunnel_mesh import (
    generate_tunnel_mesh, TunnelInversionDP2D, two_zone_E_field_tunnel,
)
from common import log_to_E, E_to_log, save_results, RESULTS_DIR

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
K_FIXED = 500.0
Lx, Ly = 20., 20.
cx, cy, R = 10., 10., 2.5

OUT_DIR = os.path.join(RESULTS_DIR, 'k3_tunnel_inversion')
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# K2: Observation point utilities
# ═══════════════════════════════════════════════════════════════════════════

def get_tunnel_perimeter_nodes(mesh, cx=10., cy=10., R=2.5, atol=0.5):
    """Select nodes near tunnel perimeter (simulating convergence gauges)."""
    pts = mesh.points
    dist = onp.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    mask = onp.abs(dist - R) < atol
    return onp.where(mask)[0]


def get_surface_nodes(mesh, Ly=20., atol=1e-5):
    """Select nodes on top surface (simulating settlement monitoring)."""
    return onp.where(onp.isclose(mesh.points[:, 1], Ly, atol=atol))[0]


def get_sparse_observation_indices(mesh, cx=10., cy=10., R=2.5, Ly=20.):
    """K2: Combine tunnel perimeter + surface nodes for sparse observation."""
    perim = get_tunnel_perimeter_nodes(mesh, cx, cy, R, atol=0.5)
    surface = get_surface_nodes(mesh, Ly)
    combined = onp.unique(onp.concatenate([perim, surface]))
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# E field generators
# ═══════════════════════════════════════════════════════════════════════════

def uniform_E_field(mesh, E_val=70000.):
    return onp.full(len(mesh.cells), E_val)


def graded_E_field(mesh, cx=10., cy=10., R=2.5, E_near=40000., E_far=90000.,
                   max_dist=10.):
    """Radially graded E: linearly increases from tunnel face outward."""
    cell_pts = onp.take(mesh.points, mesh.cells, axis=0)
    centroids = onp.mean(cell_pts, axis=1)
    dist = onp.sqrt((centroids[:, 0] - cx)**2 + (centroids[:, 1] - cy)**2) - R
    dist = onp.clip(dist, 0, max_dist)
    frac = dist / max_dist
    return E_near + frac * (E_far - E_near)


# ═══════════════════════════════════════════════════════════════════════════
# Shared infrastructure
# ═══════════════════════════════════════════════════════════════════════════

def build_tunnel_bc(Lx=20., Ly=20., displacement=-0.05):
    """Standard tunnel BCs: bottom u_y=0, left/right u_x=0, top u_y=disp."""
    def bottom(p): return np.isclose(p[1], 0., atol=1e-5)
    def left(p):   return np.isclose(p[0], 0., atol=1e-5)
    def right(p):  return np.isclose(p[0], Lx, atol=1e-5)
    def top(p):    return np.isclose(p[1], Ly, atol=1e-5)

    return [
        [bottom, left, right, top],
        [1, 0, 0, 1],
        [lambda p: 0., lambda p: 0., lambda p: 0.,
         lambda p, _d=displacement: _d],
    ]


def generate_observation(mesh, E_true, bc_info, noise_level=0.0):
    """Generate synthetic displacement observation."""
    nc = len(mesh.cells)
    prob = TunnelInversionDP2D(
        mesh, vec=2, dim=2, ele_type='TRI3',
        dirichlet_bc_info=bc_info, E=70000., k=K_FIXED,
    )
    prob.set_params(np.array(E_true))
    sol = solver(prob, solver_options=SOLVER_OPTIONS)
    u_full = sol[0]

    if noise_level > 0:
        max_disp = float(np.max(np.abs(u_full)))
        noise = noise_level * max_disp * jax.random.normal(
            jax.random.PRNGKey(42), u_full.shape)
        u_full = u_full + noise

    return u_full


def run_inversion(mesh, bc_info, u_obs, obs_indices, maxiter=150):
    """Run L-BFGS-B inversion in log-E space."""
    nc = len(mesh.cells)

    inv_problem = TunnelInversionDP2D(
        mesh, vec=2, dim=2, ele_type='TRI3',
        dirichlet_bc_info=bc_info, E=70000., k=K_FIXED,
    )
    fwd_pred = ad_wrapper(inv_problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    n_obs_dofs = u_obs[obs_indices].shape[0] * u_obs[obs_indices].shape[1]
    u_obs_sel = u_obs[obs_indices]

    def loss_fn(log_E):
        E_field = np.exp(log_E)
        sol = fwd_pred(E_field)[0]
        u_pred = sol[obs_indices]
        return np.sum((u_pred - u_obs_sel) ** 2) / n_obs_dofs

    value_and_grad_fn = jax.value_and_grad(loss_fn)
    log_E_init = onp.array(E_to_log(np.full(nc, 70000.)), dtype=onp.float64)

    # Warmup
    _ = value_and_grad_fn(np.array(log_E_init))

    bounds = [(float(onp.log(5000.)), float(onp.log(300000.)))] * nc
    loss_history = []

    def objective(x):
        loss, grad = value_and_grad_fn(np.array(x))
        loss_f = float(loss)
        loss_history.append(loss_f)
        return loss_f, onp.array(grad, dtype=onp.float64)

    from scipy.optimize import minimize as scipy_minimize
    t0 = time.time()
    result = scipy_minimize(
        objective, x0=log_E_init,
        method='L-BFGS-B', jac=True, bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12},
    )
    elapsed = time.time() - t0
    E_final = onp.exp(result.x)

    return {
        'E_final': E_final,
        'loss_history': loss_history,
        'elapsed': elapsed,
        'nit': int(result.nit),
        'converged': bool(result.success),
        'final_loss': float(result.fun),
    }


def compute_metrics(E_final, E_true):
    """Compute L2 and mean relative error."""
    E_f = onp.array(E_final)
    E_t = onp.array(E_true)
    l2_rel = float(onp.sqrt(onp.mean(((E_f - E_t) / E_t) ** 2)))
    mean_rel = float(onp.mean(onp.abs(E_f - E_t) / E_t))
    return {'l2_rel': l2_rel, 'mean_rel': mean_rel}


# ═══════════════════════════════════════════════════════════════════════════
# Sub-experiments
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(name, mesh, bc_info, E_true, obs_indices, noise_level=0.0,
                   maxiter=150):
    """Run a single tunnel inversion experiment."""
    nc = len(mesh.cells)
    n_obs = len(obs_indices)
    print(f"\n  [{name}] E range: [{E_true.min():.0f}, {E_true.max():.0f}], "
          f"obs: {n_obs}/{len(mesh.points)} nodes, noise: {noise_level:.0%}")

    u_obs = generate_observation(mesh, E_true, bc_info, noise_level)
    result = run_inversion(mesh, bc_info, u_obs, obs_indices, maxiter)
    metrics = compute_metrics(result['E_final'], E_true)

    print(f"  [{name}] L2 rel: {metrics['l2_rel']:.4f} ({metrics['l2_rel']:.2%}), "
          f"time: {result['elapsed']:.1f}s, iters: {result['nit']}")

    return {
        'name': name,
        'n_cells': nc,
        'n_obs': n_obs,
        'noise_level': noise_level,
        'E_true_range': [float(E_true.min()), float(E_true.max())],
        **metrics,
        'time_s': result['elapsed'],
        'nit': result['nit'],
        'converged': result['converged'],
        'final_loss': result['final_loss'],
        'E_final': result['E_final'],
        'loss_history': result['loss_history'],
    }


def main():
    print("=" * 70)
    print("K2-K3: TUNNEL EXCAVATION INVERSION EXPERIMENTS")
    print("=" * 70)

    # --- Generate mesh ---
    mesh = generate_tunnel_mesh(Lx, Ly, cx, cy, R,
                                 mesh_size_tunnel=0.8, mesh_size_far=2.5)
    nc = len(mesh.cells)
    nn = len(mesh.points)
    bc_info = build_tunnel_bc(Lx, Ly, displacement=-0.05)
    print(f"  Mesh: {nc} TRI3 elements, {nn} nodes")

    # --- K2: Observation points ---
    full_obs = onp.arange(nn)
    sparse_obs = get_sparse_observation_indices(mesh, cx, cy, R, Ly)
    print(f"  Full-field obs: {len(full_obs)} nodes")
    print(f"  Sparse obs: {len(sparse_obs)} nodes "
          f"(perimeter + surface)")

    # --- E fields ---
    E_uniform = uniform_E_field(mesh, 70000.)
    E_two_zone = two_zone_E_field_tunnel(mesh, cx, cy, R,
                                          E_near=50000., E_far=90000., zone_R=5.0)
    E_graded = graded_E_field(mesh, cx, cy, R,
                               E_near=40000., E_far=90000., max_dist=8.)

    all_results = []

    # --- K3a: Uniform, full-field ---
    r = run_experiment('K3a_uniform_full', mesh, bc_info,
                       E_two_zone, full_obs, noise_level=0.0, maxiter=150)
    all_results.append(r)

    # --- K3b: Uniform, sparse observation ---
    r = run_experiment('K3b_uniform_sparse', mesh, bc_info,
                       E_two_zone, sparse_obs, noise_level=0.0, maxiter=150)
    all_results.append(r)

    # --- K3c: Two-zone (soft near-tunnel) ---
    E_soft_zone = two_zone_E_field_tunnel(mesh, cx, cy, R,
                                           E_near=30000., E_far=80000., zone_R=4.0)
    r = run_experiment('K3c_soft_zone', mesh, bc_info,
                       E_soft_zone, full_obs, noise_level=0.0, maxiter=150)
    all_results.append(r)

    # --- K3d: Graded E field ---
    r = run_experiment('K3d_graded', mesh, bc_info,
                       E_graded, full_obs, noise_level=0.0, maxiter=150)
    all_results.append(r)

    # --- K3e: Noise robustness ---
    for noise in [0.01, 0.03, 0.05]:
        r = run_experiment(f'K3e_noise_{noise:.0%}', mesh, bc_info,
                           E_two_zone, full_obs, noise_level=noise, maxiter=150)
        all_results.append(r)

    # --- Summary ---
    print(f"\n{'═' * 70}")
    print("K3 SUMMARY")
    print(f"{'═' * 70}")
    print(f"{'Experiment':<25s} | {'Obs':<6s} | {'Noise':<6s} | "
          f"{'L2 rel':<8s} | {'Time':<7s} | {'Iters':<6s}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<25s} | {r['n_obs']:<6d} | {r['noise_level']:<6.0%} | "
              f"{r['l2_rel']:<8.2%} | {r['time_s']:<6.1f}s | {r['nit']:<6d}")

    # --- Save JSON ---
    save_data = []
    for r in all_results:
        entry = {k: v for k, v in r.items()
                 if k not in ('E_final', 'loss_history')}
        save_data.append(entry)
    json_path = os.path.join(OUT_DIR, 'k3_results.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # --- Plots ---
    _plot_results(mesh, all_results)

    return all_results


def _plot_results(mesh, all_results):
    """Generate K3 comparison plots."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    # Row 0: E field maps (true, K3a inverted, K3b inverted)
    vmin, vmax = 25000, 95000
    for col, name in enumerate(['K3a_uniform_full', 'K3b_uniform_sparse', 'K3c_soft_zone']):
        r = next((x for x in all_results if x['name'] == name), None)
        if r is None:
            continue
        ax = axes[0, col]
        sc = ax.tripcolor(mesh.points[:, 0], mesh.points[:, 1], mesh.cells,
                           facecolors=r['E_final'], cmap='viridis',
                           vmin=vmin, vmax=vmax)
        circle = plt.Circle((cx, cy), R, fill=True, color='white', ec='k', lw=1)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(f"{name}\nL2={r['l2_rel']:.2%}", fontsize=10)
        plt.colorbar(sc, ax=ax, label='E [MPa]', shrink=0.8)

    # Row 1: K3d graded + observation point map + convergence
    # K3d inverted
    ax = axes[1, 0]
    r_graded = next((x for x in all_results if x['name'] == 'K3d_graded'), None)
    if r_graded:
        sc = ax.tripcolor(mesh.points[:, 0], mesh.points[:, 1], mesh.cells,
                           facecolors=r_graded['E_final'], cmap='viridis',
                           vmin=vmin, vmax=vmax)
        circle = plt.Circle((cx, cy), R, fill=True, color='white', ec='k', lw=1)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(f"K3d graded\nL2={r_graded['l2_rel']:.2%}", fontsize=10)
        plt.colorbar(sc, ax=ax, label='E [MPa]', shrink=0.8)

    # Observation point map
    ax = axes[1, 1]
    sparse_obs = get_sparse_observation_indices(mesh, cx, cy, R, Ly)
    ax.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells,
               lw=0.3, color='lightgray')
    ax.plot(mesh.points[sparse_obs, 0], mesh.points[sparse_obs, 1],
            'r^', ms=6, label=f'Sparse obs ({len(sparse_obs)} pts)')
    circle = plt.Circle((cx, cy), R, fill=True, color='#f0f0f0', ec='k', lw=1.5)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title('K2: Observation Points', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, Lx + 0.5)
    ax.set_ylim(-0.5, Ly + 0.5)

    # Convergence curves
    ax = axes[1, 2]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3']
    for i, r in enumerate(all_results[:5]):
        if r['loss_history']:
            ax.semilogy(r['loss_history'], lw=1.5, alpha=0.8, color=colors[i % len(colors)],
                        label=r['name'].replace('_', ' '))
    ax.set_xlabel('Function Evaluation')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence')
    ax.legend(fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)

    # Row 2: L2 error bar chart + noise robustness + summary text
    # Bar chart
    ax = axes[2, 0]
    names_short = [r['name'].split('_', 1)[1] for r in all_results]
    l2_vals = [r['l2_rel'] * 100 for r in all_results]
    colors_bar = ['#55A868' if v < 5 else '#DD8452' if v < 10 else '#C44E52' for v in l2_vals]
    ax.barh(range(len(all_results)), l2_vals, color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(all_results)))
    ax.set_yticklabels(names_short, fontsize=8)
    ax.set_xlabel('L2 Relative Error [%]')
    ax.axvline(5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_title('All Experiments: Accuracy')
    ax.invert_yaxis()

    # Noise robustness
    ax = axes[2, 1]
    noise_results = [r for r in all_results if 'noise' in r['name']]
    r_baseline = next((r for r in all_results if r['name'] == 'K3a_uniform_full'), None)
    if noise_results and r_baseline:
        noises = [0.0] + [r['noise_level'] for r in noise_results]
        errs = [r_baseline['l2_rel'] * 100] + [r['l2_rel'] * 100 for r in noise_results]
        ax.plot([n * 100 for n in noises], errs, 'o-', color='#4C72B0', lw=2, ms=8)
        ax.set_xlabel('Noise Level [%]')
        ax.set_ylabel('L2 Relative Error [%]')
        ax.set_title('K3e: Noise Robustness')
        ax.grid(True, alpha=0.3)

    # Time comparison
    ax = axes[2, 2]
    times = [r['time_s'] for r in all_results]
    ax.barh(range(len(all_results)), times, color='#4C72B0', alpha=0.7)
    ax.set_yticks(range(len(all_results)))
    ax.set_yticklabels(names_short, fontsize=8)
    ax.set_xlabel('Time [s]')
    ax.set_title('Computation Time')
    ax.invert_yaxis()

    fig.suptitle('K2-K3: Tunnel Excavation Inversion Experiments', fontsize=15, y=0.99)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, '隧道反演实验结果.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


if __name__ == "__main__":
    main()
