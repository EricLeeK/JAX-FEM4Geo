"""
K1: Tunnel excavation mesh generation and AD gradient verification.

Generates an unstructured TRI3 mesh for a 2D tunnel cross-section:
  - Rectangular domain 20×20 m
  - Circular excavation (radius R=2.5 m) at center (10, 10)
  - Graded mesh: fine near tunnel face, coarser away

Verifies:
  1. Forward solve on unstructured mesh works correctly
  2. AD gradient matches FD gradient (key technical risk: T1 in roadmap)
  3. Heterogeneous E-field inversion on unstructured mesh

This resolves technical risk T1: "AD gradient correctness on unstructured meshes."
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

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver, ad_wrapper
from src.models.drucker_prager_2d import DruckerPragerPlasticity2D

SOLVER_OPTIONS = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
OUT_DIR = os.path.join(project_root, 'results', 'heterogeneous_inversion', 'k1_tunnel_mesh')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Tunnel mesh generation via gmsh
# ---------------------------------------------------------------------------

def generate_tunnel_mesh(Lx=20., Ly=20., cx=10., cy=10., R=2.5,
                         mesh_size_tunnel=0.5, mesh_size_far=2.0):
    """Generate an unstructured TRI3 mesh with a circular tunnel excavation.

    Parameters
    ----------
    Lx, Ly : float
        Domain dimensions.
    cx, cy : float
        Tunnel center coordinates.
    R : float
        Tunnel radius.
    mesh_size_tunnel : float
        Mesh element size near tunnel face.
    mesh_size_far : float
        Mesh element size at domain boundary.

    Returns
    -------
    mesh : Mesh
        JAX-FEM Mesh object (TRI3 elements).
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("tunnel")

    # Outer boundary (rectangle)
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size_far)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0, mesh_size_far)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0, mesh_size_far)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0, mesh_size_far)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    # Inner boundary (circle = tunnel excavation)
    pc = gmsh.model.geo.addPoint(cx, cy, 0, mesh_size_tunnel)
    pr = gmsh.model.geo.addPoint(cx + R, cy, 0, mesh_size_tunnel)
    pt = gmsh.model.geo.addPoint(cx, cy + R, 0, mesh_size_tunnel)
    pl = gmsh.model.geo.addPoint(cx - R, cy, 0, mesh_size_tunnel)
    pb = gmsh.model.geo.addPoint(cx, cy - R, 0, mesh_size_tunnel)

    c1 = gmsh.model.geo.addCircleArc(pr, pc, pt)
    c2 = gmsh.model.geo.addCircleArc(pt, pc, pl)
    c3 = gmsh.model.geo.addCircleArc(pl, pc, pb)
    c4 = gmsh.model.geo.addCircleArc(pb, pc, pr)

    inner_loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])

    # Surface = outer - inner (tunnel is a hole)
    surf = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Extract mesh data
    import meshio
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = coords.reshape(-1, 3)[:, :2]  # 2D

    # Reindex: gmsh tags may not be contiguous
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    cells = []
    for et, nt in zip(elem_types, elem_node_tags):
        n_nodes = len(nt) // len(elem_tags[0]) if len(elem_tags) > 0 else 3
        # TRI3 = 3 nodes per element
        nt_arr = onp.array(nt, dtype=int)
        n_elem = len(nt_arr) // 3
        tri_nodes = nt_arr.reshape(n_elem, 3)
        for tri in tri_nodes:
            cells.append([tag_to_idx[int(t)] for t in tri])

    cells = onp.array(cells, dtype=onp.int32)

    gmsh.finalize()

    points = onp.array(points)

    # Remove unused nodes (e.g., tunnel center point)
    used_nodes = onp.unique(cells.flatten())
    if len(used_nodes) < len(points):
        old_to_new = onp.full(len(points), -1, dtype=int)
        old_to_new[used_nodes] = onp.arange(len(used_nodes))
        points = points[used_nodes]
        cells = old_to_new[cells]

    # Fix cell orientation: TRI3 requires counter-clockwise ordering
    new_cells = []
    for c in cells:
        p = points[c]
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:  # clockwise -> flip to CCW
            new_cells.append(c[[0, 2, 1]])
        else:
            new_cells.append(c)
    cells = onp.stack(new_cells)

    mesh = Mesh(points, cells, ele_type='TRI3')
    return mesh


# ---------------------------------------------------------------------------
# InversionHeterogeneousDP2D adapted for TRI3
# ---------------------------------------------------------------------------

class TunnelInversionDP2D(DruckerPragerPlasticity2D):
    """2D plane-strain DP with per-element E field on unstructured TRI3 mesh."""

    def __init__(self, mesh, vec=2, dim=2, ele_type='TRI3',
                 dirichlet_bc_info=None, location_fns=None,
                 E=70.0e3, nu=0.3, alpha=0.3, k=500.0, a=None,
                 traction_value=None):
        self.traction_value = traction_value
        self.E = E
        self.nu = nu
        self.alpha = alpha
        self.k = k
        self.a = a if a is not None else 0.01 * k
        self._a_ratio = self.a / self.k
        if traction_value is not None:
            self._setup_traction(traction_value)
        from jax_fem.problem import Problem
        Problem.__init__(self, mesh, vec=vec, dim=dim, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)

    def _setup_traction(self, traction_value):
        traction = traction_value

        def get_surface_maps(self_ignored=None):
            def traction_fn(u, point):
                return np.array([0., traction])
            return [traction_fn]

        self.get_surface_maps = get_surface_maps

    def set_params(self, E_field):
        nc, nq = len(self.fe.cells), self.fe.num_quads
        E_quad = np.repeat(E_field[:, None, None], nq, axis=1)
        self.internal_vars[2] = E_quad


# ---------------------------------------------------------------------------
# E field generators for tunnel mesh
# ---------------------------------------------------------------------------

def two_zone_E_field_tunnel(mesh, cx=10., cy=10., R=2.5,
                             E_near=50000., E_far=90000., zone_R=5.0):
    """Create a two-zone E field: near tunnel = E_near, far = E_far.

    Elements with centroid within zone_R of tunnel center get E_near.
    """
    cell_points = onp.take(mesh.points, mesh.cells, axis=0)
    centroids = onp.mean(cell_points, axis=1)
    dist = onp.sqrt((centroids[:, 0] - cx)**2 + (centroids[:, 1] - cy)**2)

    E_field = onp.where(dist < zone_R, E_near, E_far)
    return E_field


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_k1():
    print("=" * 70)
    print("K1: TUNNEL MESH GENERATION & AD GRADIENT VERIFICATION")
    print("=" * 70)

    Lx, Ly = 20., 20.
    cx, cy, R = 10., 10., 2.5
    k_fixed = 500.0
    traction = -50.0

    # --- Step 1: Generate mesh ---
    print("\n[1] Generating tunnel mesh...")
    mesh = generate_tunnel_mesh(Lx, Ly, cx, cy, R,
                                 mesh_size_tunnel=0.8, mesh_size_far=2.5)
    nc = len(mesh.cells)
    nn = len(mesh.points)
    print(f"  Nodes: {nn}, Elements (TRI3): {nc}")

    # --- Step 2: Set up forward problem and test solve ---
    print("\n[2] Testing forward solve on tunnel mesh...")

    def bottom(p):
        return np.isclose(p[1], 0., atol=1e-5)

    def left(p):
        return np.isclose(p[0], 0., atol=1e-5)

    def right(p):
        return np.isclose(p[0], Lx, atol=1e-5)

    def top(p):
        return np.isclose(p[1], Ly, atol=1e-5)

    # Use pure displacement BCs first to test mesh validity
    # Bottom u_y=0, left u_x=0, right u_x=0, top u_y=displacement
    displacement = -0.05  # 0.5 mm compression
    dirichlet_bc_info = [
        [bottom, left, right, top],
        [1, 0, 0, 1],
        [lambda p: 0., lambda p: 0., lambda p: 0.,
         lambda p: displacement],
    ]
    location_fns = None

    problem = TunnelInversionDP2D(
        mesh, vec=2, dim=2, ele_type='TRI3',
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        E=70000., k=k_fixed,
    )

    # Set uniform E field for initial test
    E_uniform = np.full(nc, 70000.)
    problem.set_params(E_uniform)

    t0 = time.time()
    sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
    t_solve = time.time() - t0
    u_sol = sol_list[0]
    print(f"  Forward solve: {t_solve:.2f}s")
    print(f"  Max displacement: {float(np.max(np.abs(u_sol))):.6e}")
    print(f"  Solution shape: {u_sol.shape}")

    # --- Step 3: AD gradient verification ---
    print("\n[3] AD gradient verification on unstructured mesh...")

    # Create synthetic observation with two-zone E field
    E_true = two_zone_E_field_tunnel(mesh, cx, cy, R,
                                      E_near=50000., E_far=90000., zone_R=5.0)
    print(f"  E_true: min={E_true.min():.0f}, max={E_true.max():.0f}")
    print(f"  Near-tunnel cells (E=50000): {(E_true < 70000).sum()}")
    print(f"  Far cells (E=90000): {(E_true >= 70000).sum()}")

    # Generate observation
    truth_problem = TunnelInversionDP2D(
        mesh, vec=2, dim=2, ele_type='TRI3',
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        E=70000., k=k_fixed,
    )
    truth_problem.set_params(np.array(E_true))
    sol_true = solver(truth_problem, solver_options=SOLVER_OPTIONS)
    u_obs = sol_true[0]
    obs_indices = onp.arange(u_obs.shape[0])

    # Build inversion problem with AD wrapper
    inv_problem = TunnelInversionDP2D(
        mesh, vec=2, dim=2, ele_type='TRI3',
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        E=70000., k=k_fixed,
    )
    fwd_pred = ad_wrapper(inv_problem, solver_options=SOLVER_OPTIONS,
                          adjoint_solver_options=SOLVER_OPTIONS)

    n_obs_dofs = u_obs.shape[0] * u_obs.shape[1]

    def loss_fn(log_E):
        E_field = np.exp(log_E)
        sol = fwd_pred(E_field)[0]
        u_pred = sol[obs_indices]
        return np.sum((u_pred - u_obs) ** 2) / n_obs_dofs

    log_E_init = np.log(np.full(nc, 70000.))

    # AD gradient
    print("  Computing AD gradient...")
    t0 = time.time()
    ad_grad = jax.grad(loss_fn)(log_E_init)
    ad_grad_np = onp.array(ad_grad)
    t_ad = time.time() - t0
    print(f"  AD gradient: {t_ad:.2f}s, ||grad|| = {float(np.linalg.norm(ad_grad)):.4e}")

    # FD gradient (sample 20 components for speed)
    print("  Computing FD gradient (20 samples)...")
    n_sample = min(20, nc)
    sample_idx = onp.linspace(0, nc - 1, n_sample, dtype=int)
    eps = 0.01
    fd_grad_samples = onp.zeros(n_sample)

    t0 = time.time()
    for j, i in enumerate(sample_idx):
        e_i = np.zeros(nc).at[i].set(1.0)
        f_plus = float(loss_fn(log_E_init + eps * e_i))
        f_minus = float(loss_fn(log_E_init - eps * e_i))
        fd_grad_samples[j] = (f_plus - f_minus) / (2 * eps)
    t_fd = time.time() - t0
    print(f"  FD gradient ({n_sample} samples): {t_fd:.2f}s")

    # Compare
    ad_samples = ad_grad_np[sample_idx]
    abs_diff = onp.abs(ad_samples - fd_grad_samples)
    rel_diff = abs_diff / (onp.abs(ad_samples) + 1e-30)

    cos_sim = float(onp.dot(ad_samples, fd_grad_samples) /
                    (onp.linalg.norm(ad_samples) * onp.linalg.norm(fd_grad_samples) + 1e-30))
    max_rel_err = float(onp.max(rel_diff))
    mean_rel_err = float(onp.mean(rel_diff))

    print(f"\n  AD vs FD gradient comparison:")
    print(f"    Cosine similarity: {cos_sim:.8f}")
    print(f"    Max relative error: {max_rel_err:.4e}")
    print(f"    Mean relative error: {mean_rel_err:.4e}")

    grad_ok = cos_sim > 0.9999 and max_rel_err < 1e-3
    print(f"    Status: {'✅ PASS' if grad_ok else '❌ FAIL'}")

    # --- Step 4: Quick inversion test ---
    print("\n[4] Quick inversion test (maxiter=100)...")
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    _ = value_and_grad_fn(log_E_init)  # warmup

    log_E_lo = float(onp.log(5000.))
    log_E_hi = float(onp.log(300000.))
    bounds = [(log_E_lo, log_E_hi)] * nc

    loss_history = []

    def objective(x):
        log_E = np.array(x)
        loss, grad = value_and_grad_fn(log_E)
        loss_f = float(loss)
        loss_history.append(loss_f)
        return loss_f, onp.array(grad, dtype=onp.float64)

    from scipy.optimize import minimize as scipy_minimize
    t0 = time.time()
    result = scipy_minimize(
        objective,
        x0=onp.array(log_E_init, dtype=onp.float64),
        method='L-BFGS-B', jac=True, bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-20, 'gtol': 1e-12},
    )
    t_inv = time.time() - t0

    E_final = onp.exp(result.x)
    E_true_np = onp.array(E_true)
    l2_rel = float(onp.sqrt(onp.mean(((E_final - E_true_np) / E_true_np) ** 2)))
    mean_rel = float(onp.mean(onp.abs(E_final - E_true_np) / E_true_np))

    print(f"  Inversion time: {t_inv:.1f}s")
    print(f"  Iterations: {result.nit}")
    print(f"  L2 relative error: {l2_rel:.4f} ({l2_rel:.2%})")
    print(f"  Mean relative error: {mean_rel:.4f} ({mean_rel:.2%})")
    print(f"  Converged: {result.success}")

    # --- Step 5: Plots ---
    print("\n[5] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Mesh visualization
    ax = axes[0, 0]
    ax.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells, lw=0.3, color='gray')
    circle = plt.Circle((cx, cy), R, fill=True, color='white', ec='black', lw=1.5)
    ax.add_patch(circle)
    ax.set_xlim(-0.5, Lx + 0.5)
    ax.set_ylim(-0.5, Ly + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Tunnel Mesh ({nc} TRI3 elements)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    # (b) True E field
    ax = axes[0, 1]
    centroids = onp.mean(onp.take(mesh.points, mesh.cells, axis=0), axis=1)
    sc = ax.tripcolor(mesh.points[:, 0], mesh.points[:, 1], mesh.cells,
                       facecolors=E_true_np, cmap='viridis',
                       vmin=45000, vmax=95000)
    circle = plt.Circle((cx, cy), R, fill=True, color='white', ec='black', lw=1.5)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title('True E Field')
    plt.colorbar(sc, ax=ax, label='E [MPa]')

    # (c) Inverted E field
    ax = axes[1, 0]
    sc = ax.tripcolor(mesh.points[:, 0], mesh.points[:, 1], mesh.cells,
                       facecolors=E_final, cmap='viridis',
                       vmin=45000, vmax=95000)
    circle = plt.Circle((cx, cy), R, fill=True, color='white', ec='black', lw=1.5)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title(f'Inverted E Field (L2 rel = {l2_rel:.2%})')
    plt.colorbar(sc, ax=ax, label='E [MPa]')

    # (d) AD vs FD gradient
    ax = axes[1, 1]
    ax.plot(ad_samples, fd_grad_samples, 'o', ms=5, alpha=0.7)
    lim = [min(ad_samples.min(), fd_grad_samples.min()),
           max(ad_samples.max(), fd_grad_samples.max())]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.5, label='y=x')
    ax.set_xlabel('AD gradient')
    ax.set_ylabel('FD gradient')
    ax.set_title(f'AD vs FD Gradient (cos sim = {cos_sim:.6f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('K1: Tunnel Mesh — AD Gradient Verification on Unstructured Grid',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, '隧道网格_AD梯度验证.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # --- Save results ---
    results = {
        'mesh': {'n_nodes': nn, 'n_cells': nc, 'ele_type': 'TRI3'},
        'domain': {'Lx': Lx, 'Ly': Ly, 'cx': cx, 'cy': cy, 'R': R},
        'gradient_check': {
            'cosine_similarity': cos_sim,
            'max_relative_error': max_rel_err,
            'mean_relative_error': mean_rel_err,
            'n_samples': n_sample,
            'pass': grad_ok,
        },
        'inversion': {
            'l2_rel': l2_rel,
            'mean_rel': mean_rel,
            'time_s': t_inv,
            'nit': int(result.nit),
            'converged': bool(result.success),
        },
    }
    json_path = os.path.join(OUT_DIR, 'k1_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("K1 SUMMARY")
    print("=" * 70)
    print(f"  Mesh: {nc} TRI3 elements (unstructured)")
    print(f"  Forward solve: ✅ works")
    print(f"  AD gradient vs FD: {'✅ PASS' if grad_ok else '❌ FAIL'} "
          f"(cos sim = {cos_sim:.6f})")
    print(f"  Inversion L2 rel error: {l2_rel:.2%}")
    print(f"  Technical risk T1: {'RESOLVED ✅' if grad_ok else 'NOT RESOLVED ❌'}")
    print("=" * 70)


if __name__ == "__main__":
    run_k1()
