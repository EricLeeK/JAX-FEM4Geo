"""P2-9: Optimizer ablation — L-BFGS vs Adam for parameter inversion.

Compares the two optimizers on the (noise-free) DP inversion in terms of:
    * final parameter error
    * number of loss evaluations / iterations to converge
    * sensitivity to the learning rate / step size

Both run in log-parameter space (z = [log E, log k]) on the reaction-force loss
with direct (fixed-u) differentiation, so gradients are exact and cheap. The
fair comparison metric is loss-evaluations to reach a target accuracy, since
each evaluation costs the same for both methods.

Outputs:
    results/inversion/optimizer_ablation.json
    results/inversion/optimizer_ablation.png

Run:
    python -m src.inversion.optimizer_ablation
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import sys
import time
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scipy.optimize import minimize as sp_min

from src.inversion.forward import reaction_loss
from src.inversion.run_inversion import generate_truth_data, MESH_DIR


# ---------------------------------------------------------------------------
# Optimizers (both in log-space z = [log E, log k])

def run_lbfgs(loss_fn, E0, k0, max_iter=200):
    """L-BFGS-B. Records (n_eval, loss) trajectory."""
    loss_and_grad = jax.value_and_grad(lambda z: loss_fn(np.exp(z)))
    n_eval = [0]
    traj = []

    def fg(z):
        n_eval[0] += 1
        l, g = loss_and_grad(np.array(z))
        traj.append((n_eval[0], float(l)))
        return float(l), onp.array(g)

    t0 = time.time()
    res = sp_min(fun=fg, x0=onp.log([E0, k0]), jac=True, method='L-BFGS-B',
                 options={'maxiter': max_iter, 'gtol': 1e-14, 'ftol': 1e-18})
    E_rec, k_rec = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    return E_rec, k_rec, traj, time.time() - t0, res.nit


def run_adam(loss_fn, E0, k0, lr, max_iter=2000):
    """Adam. Records (iter, loss) trajectory."""
    loss_and_grad = jax.value_and_grad(lambda z: loss_fn(np.exp(z)))
    z = np.log(np.array([E0, k0]))
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(z); v = np.zeros_like(z)
    traj = []
    t0 = time.time()
    for i in range(1, max_iter + 1):
        l, g = loss_and_grad(z)
        traj.append((i, float(l)))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1 ** i); vhat = v / (1 - b2 ** i)
        z = z - lr * mhat / (np.sqrt(vhat) + eps)
        if float(l) < 1e-16:
            break
    E_rec, k_rec = float(np.exp(z[0])), float(np.exp(z[1]))
    return E_rec, k_rec, traj, time.time() - t0, len(traj)


def main():
    print("=" * 64)
    print("P2-9: Optimizer ablation — L-BFGS vs Adam")
    print("=" * 64)

    E_true, k_true = 70.0e3, 250.0
    problems, sols, targets = generate_truth_data(E_true, k_true)

    def loss_fn(params):
        return sum(reaction_loss(p, s, params, t)
                   for p, s, t in zip(problems, sols, targets))

    E0, k0 = 50.0e3, 180.0
    result = {'truth': {'E': E_true, 'k': k_true}, 'start': {'E': E0, 'k': k0}}

    # --- L-BFGS ---
    print("\n[L-BFGS]")
    Er, kr, traj_b, t_b, nit_b = run_lbfgs(loss_fn, E0, k0)
    print(f"  E={Er:.1f} ({abs(Er-E_true)/E_true*100:.4f}%)  "
          f"k={kr:.3f} ({abs(kr-k_true)/k_true*100:.4f}%)  "
          f"[{nit_b} iters, {len(traj_b)} evals, {t_b:.1f}s]")
    result['lbfgs'] = {'E': Er, 'k': kr, 'iters': nit_b, 'n_eval': len(traj_b),
                       'time': t_b, 'traj': traj_b}

    # --- Adam at several learning rates ---
    print("\n[Adam] learning-rate sweep")
    result['adam'] = {}
    for lr in [1e-2, 1e-1, 1e0, 1e1]:
        Er, kr, traj_a, t_a, n_a = run_adam(loss_fn, E0, k0, lr=lr, max_iter=500)
        print(f"  lr={lr:<5}: E={Er:.1f} ({abs(Er-E_true)/E_true*100:.3f}%)  "
              f"k={kr:.2f} ({abs(kr-k_true)/k_true*100:.3f}%)  "
              f"[{n_a} iters, {t_a:.1f}s]")
        result['adam'][f'lr_{lr}'] = {'E': Er, 'k': kr, 'iters': n_a,
                                       'time': t_a, 'traj': traj_a}

    _plot(result, E_true, k_true)
    out_json = os.path.join(MESH_DIR, 'optimizer_ablation.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_json}")
    print(f"Saved: {os.path.join(MESH_DIR, 'optimizer_ablation.png')}")
    return result


def _plot(result, E_true, k_true):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # L-BFGS trajectory (vs eval count)
    tb = result['lbfgs']['traj']
    xb, yb = zip(*tb)
    ax.semilogy(xb, yb, 'k-', linewidth=2.5, label=f"L-BFGS ({len(tb)} evals)")

    colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
    for (key, col) in zip(result['adam'], colors):
        ta = result['adam'][key]['traj']
        xa, ya = zip(*ta)
        lr = key.split('_')[1]
        ax.semilogy(xa, ya, color=col, linewidth=1.5, alpha=0.85,
                    label=f"Adam lr={lr} ({len(ta)} iters)")

    ax.set_xlabel('loss evaluation count')
    ax.set_ylabel('loss (log scale)')
    ax.set_title(f'P2-9: Optimizer convergence\n(truth E*={E_true:.0f}, k*={k_true:.0f})')
    ax.set_ylim(bottom=1e-20)
    ax.legend(); ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(MESH_DIR, 'optimizer_ablation.png'), dpi=140)


if __name__ == "__main__":
    main()
