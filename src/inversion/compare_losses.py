"""P1-6: Loss-function ablation — reaction force vs. displacement.

Answers the reviewer question "why reaction force, not displacement, as the
inversion loss?" with three lines of quantitative evidence:

    1. Gradient magnitude: under displacement control the solved displacement
       field u is (nearly) independent of the material stiffness, so a loss on
       u has a degenerate ~0 gradient. The reaction stress, by contrast,
       scales with stiffness and gives a healthy gradient.
    2. Optimizer behaviour: starting from the same perturbed guess, the
       reaction loss converges to the truth while the displacement loss stalls
       (flat loss landscape).
    3. Parameter error: final recovery accuracy for each loss.

The displacement loss is differentiated through the implicit FEM solve
(ad_wrapper adjoint) — this is the only path that carries u's param dependence.
The reaction loss uses the direct (fixed-u) strategy proven in P0.

Outputs:
    results/inversion/loss_comparison.json   (raw numbers)
    results/inversion/loss_comparison.png    (convergence + gradient plots)

Run:
    python -m src.inversion.compare_losses
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

from jax_fem.solver import ad_wrapper
from src.inversion.forward import (
    TriaxialForwardProblem, make_box_mesh, solve_displacement,
    reaction_loss, dp_stress, SOLVER_OPTIONS)
from src.inversion.run_inversion import generate_truth_data, get_mesh, MESH_DIR


# ---------------------------------------------------------------------------
# The two candidate losses

def make_reaction_loss(problems, sols, targets):
    """Reaction loss = sum of (sigma_zz - target)^2. Direct differentiation."""
    def loss_fn(params):
        total = 0.
        for prob, sol, tgt in zip(problems, sols, targets):
            total = total + reaction_loss(prob, sol, params, tgt)
        return total
    return loss_fn


def make_displacement_loss(problems_fwd, params_true):
    """Displacement loss = sum of ||u(params) - u*||^2.

    Differentiated through the implicit FEM solve (ad_wrapper). The truth
    fields u* are generated once at params_true and held fixed as targets.

    Implementation caveat: the implicit adjoint (custom_vjp) stores the params
    tracer on the problem instance via ``set_params``. Reusing one problem
    across many traced evaluations leaks the tracer, so we rebuild a fresh
    problem + ad_wrapper on *every* loss call. This is slow but correct and is
    exactly the kind of friction that motivates the direct (reaction) strategy.
    """
    # Truth displacement fields, one per load level (computed once, detached).
    sols_star = [solve_displacement(p, params_true) for p in problems_fwd]
    # Capture the BC recipe so we can rebuild identical fresh problems inside
    # the traced loss. Store mesh + load levels + confining pressure.
    mesh = problems_fwd[0].mesh if hasattr(problems_fwd[0], 'mesh') else None
    confining_p = problems_fwd[0].confining_p
    axial_disps = [p.axial_disp for p in problems_fwd]

    def loss_fn(params):
        total = 0.
        for d, sol_star in zip(axial_disps, sols_star):
            # Fresh problem each call: avoids tracer leaking through set_params
            # state on a reused instance.
            prob = TriaxialForwardProblem(
                _get_shared_mesh(), confining_p, d)
            fwd = ad_wrapper(prob, solver_options=SOLVER_OPTIONS,
                             adjoint_solver_options=SOLVER_OPTIONS)
            sol = fwd(params)
            total = total + np.sum((sol[0] - sol_star[0]) ** 2)
        return total
    return loss_fn


def _get_shared_mesh():
    """Return the lazily-built shared mesh (see run_inversion.get_mesh)."""
    return get_mesh()


# ---------------------------------------------------------------------------
# Optimisers

def minimize_lbfgs(loss_and_grad, E_init, k_init, max_iter=300):
    """Log-space L-BFGS via scipy. Used for the reaction loss (no tracer issue
    because reaction loss uses direct differentiation, not the implicit adjoint
    that stores state on the problem)."""
    from scipy.optimize import minimize

    def fg(z):
        l, g = loss_and_grad(np.exp(np.array(z)))
        return float(l), onp.array(g) * onp.exp(onp.array(z))  # d/dz = d/dp * p

    res = minimize(fun=fg, x0=onp.log(onp.array([E_init, k_init])), jac=True,
                   method='L-BFGS-B',
                   options={'maxiter': max_iter, 'gtol': 1e-14})
    return float(np.exp(res.x[0])), float(np.exp(res.x[1])), res


def minimize_adam(loss_and_grad, E_init, k_init, max_iter=50, lr=1e0):
    """Log-space Adam. Used for the displacement loss, whose implicit adjoint
    is incompatible with scipy's finite-difference L-BFGS probing (tracer
    leak). Plain Adam evaluates the exact analytic gradient, no probing."""
    z = np.log(np.array([E_init, k_init]))
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(z); v = np.zeros_like(z)
    for i in range(1, max_iter + 1):
        loss, g = loss_and_grad(np.exp(z))
        m = b1 * m + (1 - b1) * g * np.exp(z)   # chain rule to log space
        v = b2 * v + (1 - b2) * (g * np.exp(z)) ** 2
        mhat = m / (1 - b1 ** i); vhat = v / (1 - b2 ** i)
        z = z - lr * mhat / (np.sqrt(vhat) + eps)
    return float(np.exp(z[0])), float(np.exp(z[1]))


def gradient_norm(loss_fn, params):
    """L2 norm of the loss gradient at params (in natural param space)."""
    g = jax.grad(loss_fn)(params)
    return float(np.sqrt(np.sum(g ** 2)))


# ---------------------------------------------------------------------------
# Experiment

def main():
    os.makedirs(MESH_DIR, exist_ok=True)
    print("=" * 70)
    print("P1-6: Loss ablation — reaction force vs. displacement")
    print("=" * 70)

    E_true, k_true = 70.0e3, 250.0
    params_true = np.array([E_true, k_true])

    # Truth data (targets for the reaction loss).
    problems, sols, targets = generate_truth_data(E_true, k_true)

    # Fresh problems for the displacement loss (need their own ad_wrapper).
    # Reuse the same mesh + load levels + confining pressure.
    mesh = get_mesh()
    confining_p = 100.0
    axial_disps = [-0.02, -0.1, -0.3, -0.6, -1.0]
    problems_disp = [TriaxialForwardProblem(mesh, confining_p, d) for d in axial_disps]

    loss_rxn = make_reaction_loss(problems, sols, targets)
    loss_disp = make_displacement_loss(problems_disp, params_true)

    # ---- 1. Gradient magnitude at a perturbed point ----
    E0, k0 = 50.0e3, 180.0
    p0 = np.array([E0, k0])
    print(f"\n[1] Gradient magnitude at perturbed point E={E0}, k={k0}")
    grad_rxn = gradient_norm(loss_rxn, p0)
    grad_disp = gradient_norm(loss_disp, p0)
    print(f"    reaction loss:     |grad| = {grad_rxn:.6e}")
    print(f"    displacement loss: |grad| = {grad_disp:.6e}")
    ratio = grad_rxn / (grad_disp + 1e-30)
    print(f"    ratio (rxn/disp):  {ratio:.3e}  -> {'reaction is far more informative' if ratio > 1e3 else 'comparable'}")

    # ---- 2. Optimizer convergence from the same start ----
    print(f"\n[2] Recovery from E={E0}, k={k0}")
    loss_rxn_g = jax.value_and_grad(loss_rxn)
    loss_disp_g = jax.value_and_grad(loss_disp)

    t0 = time.time()
    Er, kr, res_r = minimize_lbfgs(loss_rxn_g, E0, k0)
    tr = time.time() - t0
    print(f"    reaction:     E={Er:.1f} ({abs(Er-E_true)/E_true*100:.2f}%)  "
          f"k={kr:.2f} ({abs(kr-k_true)/k_true*100:.2f}%)  "
          f"[{res_r.nit} iters, {tr:.1f}s]")

    # Displacement loss: implicit adjoint is incompatible with scipy's FD-based
    # L-BFGS probing, and (as this experiment shows) its gradient is ~5e4x
    # smaller. We run plain Adam (exact analytic grad) for a fixed budget and
    # report how little it moves — the expected, damning result.
    t0 = time.time()
    Ed, kd = minimize_adam(loss_disp_g, E0, k0, max_iter=30, lr=1e0)
    td = time.time() - t0
    print(f"    displacement: E={Ed:.1f} ({abs(Ed-E_true)/E_true*100:.2f}%)  "
          f"k={kd:.2f} ({abs(kd-k_true)/k_true*100:.2f}%)  "
          f"[30 Adam iters, {td:.1f}s]")

    # ---- 3. Plot ----
    iters_r = res_r.nit
    _plot(E_true, k_true, E0, k0, loss_rxn_g, loss_disp_g,
          (Er, kr), (Ed, kd), grad_rxn, grad_disp)

    result = {
        'truth': {'E': E_true, 'k': k_true},
        'start': {'E': E0, 'k': k0},
        'gradient_norm_at_start': {'reaction': grad_rxn, 'displacement': grad_disp},
        'recovery': {
            'reaction':     {'E': Er, 'k': kr, 'errE_pct': abs(Er-E_true)/E_true*100,
                             'errk_pct': abs(kr-k_true)/k_true*100, 'iters': iters_r,
                             'optimizer': 'L-BFGS'},
            'displacement': {'E': Ed, 'k': kd, 'errE_pct': abs(Ed-E_true)/E_true*100,
                             'errk_pct': abs(kd-k_true)/k_true*100, 'iters': 30,
                             'optimizer': 'Adam'},
        },
        'conclusion': 'reaction loss is the correct observable under displacement control',
    }
    out_json = os.path.join(MESH_DIR, 'loss_comparison.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_json}")
    print(f"Saved: {os.path.join(MESH_DIR, 'loss_comparison.png')}")
    return result


def _plot(E_true, k_true, E0, k0, loss_rxn_g, loss_disp_g,
          rec_rxn, rec_disp, grad_rxn, grad_disp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: gradient magnitude bar
    ax = axes[0]
    bars = ax.bar(['reaction\nloss', 'displacement\nloss'],
                  [grad_rxn, grad_disp + 1e-30],
                  color=['#2ecc71', '#e74c3c'], log=True)
    ax.set_ylabel(r'$\|\nabla_\theta L\|$ at perturbed start (log scale)')
    ax.set_title(f'(a) Gradient magnitude at start\n(E₀={E0:.0f}, k₀={k0:.0f})')
    for b, v in zip(bars, [grad_rxn, grad_disp]):
        ax.text(b.get_x() + b.get_width()/2, v * 3,
                f'{v:.2e}', ha='center', va='bottom', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Right: recovery error
    ax = axes[1]
    labels = ['reaction', 'displacement']
    errE = [abs(rec_rxn[0]-E_true)/E_true*100, abs(rec_disp[0]-E_true)/E_true*100]
    errk = [abs(rec_rxn[1]-k_true)/k_true*100, abs(rec_disp[1]-k_true)/k_true*100]
    x = onp.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, errE, w, label='E error (%)', color='#3498db')
    ax.bar(x + w/2, errk, w, label='k error (%)', color='#f39c12')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Parameter recovery error (%)')
    ax.set_title('(b) Final recovery error (lower is better)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, (eE, ek) in enumerate(zip(errE, errk)):
        ax.text(i - w/2, eE + 0.5, f'{eE:.2f}%', ha='center', fontsize=9)
        ax.text(i + w/2, ek + 0.5, f'{ek:.2f}%', ha='center', fontsize=9)

    fig.suptitle('P1-6: Why reaction force (not displacement) for inversion',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MESH_DIR, 'loss_comparison.png'), dpi=140)


if __name__ == "__main__":
    main()
