"""P0-1 / P0-2: Parameter inversion for Drucker-Prager plasticity.

End-to-end inversion pipeline:

    1. Generate synthetic "truth" data with known parameters (E*, k*) by
       running the forward triaxial model at several axial displacement levels.
    2. Recover (E, k) from a *perturbed* initial guess by minimizing the
       sum-of-squared reaction-stress error against the truth, using gradient
       descent (Adam) and/or L-BFGS. Gradients come from direct (fixed-u)
       differentiation (see src/inversion/forward.py).

This closes the loop that the project's thesis hinges on: differentiable FEM
enables gradient-based parameter inversion of a plasticity model.

Run:
    python -m src.inversion.run_inversion
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inversion.forward import (
    TriaxialForwardProblem, make_box_mesh, solve_displacement,
    reaction_loss, total_loss, SOLVER_OPTIONS)

# Reusable mesh: shared across all load levels for consistency.
MESH_DIR = os.path.join(project_root, 'results', 'inversion')
MESH = None  # lazily built so import is cheap


def get_mesh():
    global MESH
    if MESH is None:
        MESH = make_box_mesh(Nx=2, Ny=2, Nz=2, data_dir=MESH_DIR)
    return MESH


# ---------------------------------------------------------------------------
# Step 1: synthetic ground-truth data generation

def generate_truth_data(E_true, k_true, confining_p=100.0,
                        axial_disps=None):
    """Run the forward model at truth parameters to produce target reactions.

    Returns
    -------
    problems : list[TriaxialForwardProblem]  (one per load level)
    sols     : list[array]                   (solved displacement fields)
    targets  : list[float]                   (volume-averaged sigma_zz per level)
    """
    if axial_disps is None:
        # Span elastic and plastic regimes so both E and k are constrained.
        axial_disps = [-0.02, -0.1, -0.3, -0.6, -1.0]

    mesh = get_mesh()
    problems, sols, targets = [], [], []
    params_true = np.array([E_true, k_true])

    print(f"\n[Truth] E*={E_true:.1f}, k*={k_true:.1f}, confining_p={confining_p}")
    print(f"[Truth] load levels (axial disp): {axial_disps}")
    for d in axial_disps:
        prob = TriaxialForwardProblem(mesh, confining_p=confining_p, axial_disp=d)
        sol = solve_displacement(prob, params_true)
        # Reaction = volume-averaged sigma_zz evaluated at truth params.
        # Loss is 0 here by construction; we extract the target stress.
        avg_sigma = _eval_avg_sigma_zz(prob, sol, params_true)
        problems.append(prob)
        sols.append(sol)
        targets.append(float(avg_sigma))
        print(f"   d={d:+.3f}  ->  sigma_zz_avg = {avg_sigma:+.3f} MPa")
    return problems, sols, targets


def _eval_avg_sigma_zz(problem, sol, params):
    """Volume-averaged sigma_zz (the reaction observable), non-differentiable helper."""
    from src.inversion.forward import dp_stress
    E, k = params[0], params[1]
    nu, alpha, dim = problem.nu, problem.alpha, problem.dim
    fe = problem.fe
    u_grads = fe.sol_to_grad(sol)
    strains = 0.5 * (u_grads + np.transpose(u_grads, (0, 1, 3, 2)))
    vmap_stress = jax.vmap(jax.vmap(
        lambda eps: dp_stress(eps, E, k, nu, alpha, dim)))
    sigmas = vmap_stress(strains)
    weighted = sigmas.reshape(-1, dim, dim) * fe.JxW.reshape(-1)[:, None, None]
    avg_sigma = np.sum(weighted, axis=0) / np.sum(fe.JxW)
    return avg_sigma[2, 2]


# ---------------------------------------------------------------------------
# Step 2: inversion (gradient-based recovery)

def run_inversion(problems, sols, targets,
                  E_init, k_init,
                  method='lbfgs', max_iter=500, lr=1e-2, tol=1e-14,
                  log_every=20):
    """Minimize total reaction loss to recover (E, k) from a perturbed guess.

    Parameters are optimized in **log-space** ``z = [log E, log k]``. This
    (a) enforces positivity automatically, and (b) balances the very different
    scales of E (~1e4) and k (~1e2) so a single learning rate / L-BFGS works
    without per-parameter tuning. The chain-rule Jacobian ``dE/dz = E`` is
    handled automatically by JAX.

    Each iteration only re-evaluates the constitutive law on the precomputed
    strain fields — *no* FEM re-solve is needed during optimization (direct
    differentiation), so iterations are cheap.

    Parameters
    ----------
    problems, sols, targets : truth data from ``generate_truth_data``
    E_init, k_init : float
        Perturbed starting guess (must be > 0).
    method : 'adam' | 'lbfgs'
    """
    # Loss in log-space: z = [log E, log k], so params = exp(z).
    def loss_z(z):
        return total_loss(problems, sols, np.exp(z), targets)

    loss_and_grad = jax.value_and_grad(loss_z)

    z0 = np.log(np.array([E_init, k_init]))
    print(f"\n[Invert] method={method}, init E={E_init:.1f}, k={k_init:.1f} "
          f"(z0={np.array(onp.array(z0))})")
    print("-" * 60)

    loss0, _ = loss_and_grad(z0)
    print(f"  iter    0: loss={float(loss0):.6e}  "
          f"E={float(np.exp(z0)[0]):.2f}  k={float(np.exp(z0)[1]):.3f}")

    if method == 'adam':
        z = _adam(loss_and_grad, z0, max_iter, lr, tol, log_every)
    elif method == 'lbfgs':
        z = _lbfgs(loss_and_grad, z0, max_iter, tol, log_every)
    else:
        raise ValueError(f"unknown method: {method}")

    return np.exp(z)


def _adam(loss_and_grad, z0, max_iter, lr, tol, log_every):
    """Plain Adam optimizer (no extra deps) in log-param space z."""
    z = z0
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    for i in range(1, max_iter + 1):
        loss, g = loss_and_grad(z)
        if float(loss) < tol:
            print(f"  iter {i:4d}: loss={float(loss):.6e} (converged)")
            break
        m = b1 * m + (1. - b1) * g
        v = b2 * v + (1. - b2) * g * g
        mhat = m / (1. - b1 ** i)
        vhat = v / (1. - b2 ** i)
        z = z - lr * mhat / (np.sqrt(vhat) + eps)
        if i % log_every == 0:
            p = np.exp(z)
            print(f"  iter {i:4d}: loss={float(loss):.6e}  "
                  f"E={float(p[0]):.2f}  k={float(p[1]):.3f}")
    return z


def _lbfgs(loss_and_grad, z0, max_iter, tol, log_every):
    """L-BFGS-B via scipy, wrapping the JAX loss/grad with onp conversion."""
    from scipy.optimize import minimize

    def fg(x):
        l, g = loss_and_grad(np.array(x))
        return float(l), onp.array(g)

    res = minimize(fun=fg, x0=onp.array(z0), jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'disp': False,
                            'gtol': 1e-12, 'ftol': 1e-16})
    p = np.exp(np.array(res.x))
    print(f"  L-BFGS done in {res.nit} iters: loss={res.fun:.6e}  "
          f"E={float(p[0]):.2f}  k={float(p[1]):.3f}")
    return np.array(res.x)


# ---------------------------------------------------------------------------
# Step 3: verification harness (P0-2 closure)

def _recover_single(problems, sols, targets, E_init, k_init, quiet=True):
    """One inversion run; returns (E_rec, k_rec). Quiet mode suppresses logs."""
    import io, contextlib
    buf = io.StringIO() if quiet else None
    with contextlib.redirect_stdout(buf or sys.stdout):
        rec = run_inversion(problems, sols, targets, E_init, k_init,
                            method='lbfgs', max_iter=500)
    return float(rec[0]), float(rec[1])


def robustness_check(problems, sols, targets, E_true, k_true):
    """Recover parameters from several perturbed initial guesses.

    Confirms the recovery is not an artefact of a lucky starting point: the
    basin of attraction should cover reasonable ±50% perturbations of both
    parameters.
    """
    import itertools
    print("\n" + "#" * 60)
    print("ROBUSTNESS CHECK: recovery from multiple perturbed starts")
    print("#" * 60)
    # Multiplicative perturbations of the truth (avoids sign/zero issues).
    perturbs = [0.5, 0.7, 1.5, 2.0]
    results = []
    for fE, fk in itertools.product(perturbs, perturbs):
        E0, k0 = E_true * fE, k_true * fk
        E_rec, k_rec = _recover_single(problems, sols, targets, E0, k0, quiet=True)
        errE = abs(E_rec - E_true) / E_true * 100
        errk = abs(k_rec - k_true) / k_true * 100
        ok = errE < 1. and errk < 1.
        results.append(ok)
        print(f"  start E0={E0:.0f} ({fE:.1f}x) k0={k0:.0f} ({fk:.1f}x)  "
              f"->  E={E_rec:.0f} ({errE:.2f}%)  k={k_rec:.1f} ({errk:.2f}%)  "
              f"{'OK' if ok else 'FAIL'}")
    n_ok = sum(results)
    print(f"\n  {n_ok}/{len(results)} starts recovered within 1%.")
    return n_ok == len(results)


def main():
    os.makedirs(MESH_DIR, exist_ok=True)
    t0 = time.time()

    # --- Ground truth ---
    E_true, k_true = 70.0e3, 250.0
    problems, sols, targets = generate_truth_data(E_true, k_true)

    # --- Inversion from a perturbed initial guess ---
    E_init, k_init = 50.0e3, 180.0   # ~-28% / -28% off the truth
    rec_lbfgs = run_inversion(problems, sols, targets, E_init, k_init,
                              method='lbfgs', max_iter=500)

    # --- Report ---
    print("\n" + "=" * 60)
    print("INVERSION RESULT")
    print("=" * 60)
    E_rec, k_rec = float(rec_lbfgs[0]), float(rec_lbfgs[1])
    print(f"  Truth:        E* = {E_true:.1f}    k* = {k_true:.3f}")
    print(f"  Recovered:    E  = {E_rec:.1f}    k  = {k_rec:.3f}")
    print(f"  Error:        dE = {abs(E_rec-E_true)/E_true*100:.2f}%   "
          f"dk = {abs(k_rec-k_true)/k_true*100:.2f}%")

    # --- Robustness: multiple perturbed starts ---
    robust_ok = robustness_check(problems, sols, targets, E_true, k_true)

    print(f"\n  Total time: {time.time()-t0:.1f}s")
    ok = abs(E_rec - E_true) / E_true < 0.01 and abs(k_rec - k_true) / k_true < 0.01
    if ok and robust_ok:
        print("  -> SUCCESS: parameters recovered within 1%, robust across starts.")
    elif ok:
        print("  -> SUCCESS (single start); robustness: see table above.")
    else:
        print("  -> PARTIAL: see errors above.")

    # Save result
    out_path = os.path.join(MESH_DIR, 'inversion_result.csv')
    onp.savetxt(out_path,
                onp.array([[E_true, k_true, E_rec, k_rec]]),
                delimiter=',',
                header='E_true,k_true,E_recovered,k_recovered',
                comments='')
    print(f"  Saved: {out_path}")
    return rec_lbfgs


if __name__ == "__main__":
    main()
