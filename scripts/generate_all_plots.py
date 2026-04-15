#!/usr/bin/env python
"""
Generate publication-quality plots for ALL experiment results.

Reads JSON result files and produces polished matplotlib figures
with English labels and Chinese filenames.
"""

import json
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RESULTS = '/Users/chrisv/Projects/JAX-FEM/results'
PARAM_INV = os.path.join(RESULTS, 'parameter_inversion')
HETERO_INV = os.path.join(RESULTS, 'heterogeneous_inversion')
DIAG = os.path.join(RESULTS, 'plasticity_gradient_diagnostic')


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# D1: Twin Experiment — Convergence Summary
# ═══════════════════════════════════════════════════════════════════════════
def plot_d1():
    d = load_json(os.path.join(PARAM_INV, 'd1_twin_experiment/results.json'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Bar chart of errors
    ax = axes[0]
    stages = ['Stage 1\n(E only)', 'Stage 2\n(k only)', 'Joint\nRefinement']
    err_E = [d['stage1']['err_E'], d['stage1']['err_E'], d['final']['err_E']]
    err_k = [float('nan'), d['stage2']['err_k'], d['final']['err_k']]
    x = np.arange(len(stages))
    w = 0.35
    bars_E = ax.bar(x - w/2, [max(e, 1e-16) for e in err_E], w,
                    label='E error', color='#4C72B0', alpha=0.85)
    valid_k = [(i, e) for i, e in enumerate(err_k) if not np.isnan(e)]
    if valid_k:
        ax.bar([x[i] + w/2 for i, _ in valid_k], [max(e, 1e-16) for _, e in valid_k],
               w, label='k error', color='#DD8452', alpha=0.85)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel('Relative Error')
    ax.set_title('Parameter Recovery Accuracy')
    ax.axhline(0.01, color='gray', ls='--', lw=0.8, alpha=0.6, label='1% threshold')
    ax.legend()
    ax.set_ylim(1e-16, 1)

    # Summary text panel
    ax = axes[1]
    ax.axis('off')
    info = (
        f"D1: Two-Stage Twin Experiment\n\n"
        f"True values:   E = {d['E_true']:.0f} MPa,  k = {d['k_true']:.1f} MPa\n"
        f"Initial guess: E = {d['E_init']:.0f} MPa,  k = {d['k_init']:.1f} MPa\n\n"
        f"Stage 1 (elastic → E):\n"
        f"    E = {d['stage1']['E']:.2f}  (err = {d['stage1']['err_E']:.2e})\n\n"
        f"Stage 2 (plastic → k):\n"
        f"    k = {d['stage2']['k']:.6f}  (err = {d['stage2']['err_k']:.2e})\n\n"
        f"Joint refinement:\n"
        f"    E = {d['final']['E']:.6f}  (err = {d['final']['err_E']:.2e})\n"
        f"    k = {d['final']['k']:.10f}  (err = {d['final']['err_k']:.2e})\n\n"
        f"Total time: {d['total_time_s']:.1f} s"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))

    plt.suptitle('D1: Two-Stage Parameter Inversion (Twin Experiment)', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'd1_twin_experiment', '双阶段孪生实验结果.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ D1: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# D2: Sensitivity Analysis — Loss Landscape + Hessian
# ═══════════════════════════════════════════════════════════════════════════
def plot_d2():
    d = load_json(os.path.join(PARAM_INV, 'd2_sensitivity/results.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Hessian info
    ax = axes[0]
    ax.axis('off')
    H = np.array(d['hessian'])
    eig = d['eigenvalues']
    cond = d['condition_number']
    info = (
        f"D2: Sensitivity Analysis at True Parameters\n\n"
        f"Hessian matrix (∂²L / ∂θᵢ∂θⱼ):\n"
        f"     [{H[0,0]:12.4e}  {H[0,1]:12.4e}]\n"
        f"     [{H[1,0]:12.4e}  {H[1,1]:12.4e}]\n\n"
        f"Eigenvalues:\n"
        f"     λ₁ = {eig[0]:.4e}  (E direction)\n"
        f"     λ₂ = {eig[1]:.4e}  (k direction)\n\n"
        f"Condition number: {cond:,.0f}\n"
        f"     ≈ {cond:.2e}\n\n"
        f"Locally convex: {d['locally_convex']}\n"
        f"Loss range: [{d['loss_min']:.1f}, {d['loss_max']:.1f}]"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))

    # Eigenvalue bar chart
    ax = axes[1]
    labels = ['λ₁ (E-sensitive)', 'λ₂ (k-sensitive)']
    vals = [abs(eig[0]), abs(eig[1])]
    colors = ['#4C72B0', '#DD8452']
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('Eigenvalue Magnitude')
    ax.set_title(f'Hessian Eigenvalues (κ = {cond:.2e})')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 2, f'{v:.2e}',
                ha='center', fontsize=10)

    plt.suptitle('D2: Loss Landscape Sensitivity Analysis (σ_zz Loss)', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'd2_sensitivity', '损失函数灵敏度分析.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ D2: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# D3: Multi-step Inversion Comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_d3():
    d = load_json(os.path.join(PARAM_INV, 'd3_multistep/results.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    strategies = ['Single-step\njoint', 'Multi-step\njoint', 'Two-stage\n+ multi-step']
    keys = ['single_step', 'multi_step', 'twostage_multistep']
    colors_E = ['#4C72B0', '#55A868', '#C44E52']
    colors_k = ['#6C8EBF', '#7CC47C', '#D4726C']

    # Error comparison
    ax = axes[0]
    err_E = [d[k]['err_E'] for k in keys]
    err_k = [d[k]['err_k'] for k in keys]
    x = np.arange(len(strategies))
    w = 0.35
    ax.bar(x - w/2, err_E, w, label='E relative error', color='#4C72B0', alpha=0.85)
    ax.bar(x + w/2, err_k, w, label='k relative error', color='#DD8452', alpha=0.85)
    ax.axhline(0.01, color='gray', ls='--', lw=0.8, alpha=0.6, label='1% threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel('Relative Error')
    ax.set_yscale('log')
    ax.set_title('Parameter Recovery Accuracy')
    ax.legend(fontsize=8)

    # Time comparison
    ax = axes[1]
    times = [d[k]['time_s'] for k in keys]
    bars = ax.bar(strategies, times, color=['#4C72B0', '#55A868', '#C44E52'], alpha=0.85)
    ax.set_ylabel('Time [s]')
    ax.set_title('Computation Time')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, t + 0.2, f'{t:.1f}s',
                ha='center', fontsize=9)

    plt.suptitle('D3: Multi-step Loading Inversion Strategies', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'd3_multistep', '多步加载反演策略对比.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ D3: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# E1: Displacement vs Stress Loss Comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_e1():
    d = load_json(os.path.join(PARAM_INV, 'e1_displacement_inversion/results.json'))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Error comparison
    ax = axes[0]
    strategies = ['Disp\n(joint)', 'Disp\n(2-stage)', 'σ_zz\n(2-stage)']
    keys = ['disp_joint', 'disp_two_stage', 'sigma_two_stage']
    err_E = [d[k]['err_E'] for k in keys]
    err_k = [d[k]['err_k'] for k in keys]
    x = np.arange(len(strategies))
    w = 0.35
    ax.bar(x - w/2, [max(e, 1e-16) for e in err_E], w,
           label='E error', color='#4C72B0', alpha=0.85)
    ax.bar(x + w/2, [max(e, 1e-16) for e in err_k], w,
           label='k error', color='#DD8452', alpha=0.85)
    ax.axhline(0.01, color='gray', ls='--', lw=0.8, alpha=0.6, label='1% threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel('Relative Error')
    ax.set_title('Parameter Recovery Accuracy')
    ax.set_yscale('log')
    ax.set_ylim(1e-16, 10)
    ax.legend(fontsize=8)

    # Condition number comparison
    ax = axes[1]
    cond_disp = d['disp_joint']['condition_number']
    cond_sigma = d['sigma_two_stage']['condition_number']
    labels = ['Displacement\nloss', 'σ_zz\nloss']
    conds = [cond_disp, cond_sigma]
    colors = ['#4C72B0', '#DD8452']
    bars = ax.bar(labels, conds, color=colors, alpha=0.85, width=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('Hessian Condition Number')
    ax.set_title('Loss Function Conditioning')
    for bar, c in zip(bars, conds):
        ax.text(bar.get_x() + bar.get_width()/2, c * 2, f'{c:.1e}',
                ha='center', fontsize=9)

    # Eigenvalue spectrum
    ax = axes[2]
    eig_d = d['disp_joint']['hessian_eigenvalues']
    eig_s = d['sigma_two_stage']['hessian_eigenvalues']
    x_pos = [0, 1, 3, 4]
    vals = [abs(eig_d[0]), abs(eig_d[1]), abs(eig_s[0]), abs(eig_s[1])]
    colors = ['#4C72B0', '#6C8EBF', '#DD8452', '#E8A67C']
    bars = ax.bar(x_pos, [max(v, 1e-16) for v in vals], color=colors, alpha=0.85)
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(['Disp loss', 'σ_zz loss'])
    ax.set_yscale('log')
    ax.set_ylabel('|Eigenvalue|')
    ax.set_title('Hessian Eigenvalue Spectrum')
    # Mark negative eigenvalues
    for i, (xp, ev) in enumerate(zip(x_pos, [eig_d[0], eig_d[1], eig_s[0], eig_s[1]])):
        sign = '−' if ev < 0 else '+'
        ax.text(xp, max(abs(ev), 1e-16) * 3, f'{sign}{abs(ev):.1e}',
                ha='center', fontsize=7, rotation=45)

    plt.suptitle('E1: Displacement Field vs Stress Scalar Inversion', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'e1_displacement_inversion', '位移场与应力标量反演对比.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ E1: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# E2: Noise Robustness
# ═══════════════════════════════════════════════════════════════════════════
def plot_e2():
    d = load_json(os.path.join(PARAM_INV, 'e2_noise_robustness/results.json'))
    levels = d['levels']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    noise_pct = [lv['noise_level'] * 100 for lv in levels]
    mean_E = [lv['mean_err_E'] for lv in levels]
    std_E = [lv['std_err_E'] for lv in levels]
    mean_k = [lv['mean_err_k'] for lv in levels]
    std_k = [lv['std_err_k'] for lv in levels]

    # Panel 1: Error vs noise
    ax = axes[0]
    ax.errorbar(noise_pct, mean_E, yerr=std_E, fmt='-s', markersize=6,
                capsize=4, lw=1.5, label='E error', color='#4C72B0')
    ax.errorbar(noise_pct, mean_k, yerr=std_k, fmt='-^', markersize=6,
                capsize=4, lw=1.5, label='k error', color='#DD8452')
    ax.axhline(0.01, color='gray', ls='--', lw=0.8, alpha=0.6, label='1% threshold')
    ax.set_xlabel('Observation Noise Level [%]')
    ax.set_ylabel('Relative Error')
    ax.set_title('Parameter Error vs Noise')
    ax.set_yscale('log')
    ax.legend()

    # Panel 2: Convergence rate
    ax = axes[1]
    n_converged = [lv['n_converged'] for lv in levels]
    ax.bar(noise_pct, n_converged, width=0.8, color='#55A868', alpha=0.85)
    ax.axhline(d['n_trials'], color='gray', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel('Observation Noise Level [%]')
    ax.set_ylabel('# Converged Trials')
    ax.set_title(f'Convergence Rate (out of {d["n_trials"]} trials)')
    ax.set_ylim(0, d['n_trials'] + 0.5)

    # Panel 3: Recovered parameters scatter
    ax = axes[2]
    cmap = plt.cm.viridis
    for i, lv in enumerate(levels):
        c = cmap(i / max(len(levels)-1, 1))
        Es = [t['E_final'] for t in lv['trials'] if t['converged'] and 'E_final' in t]
        ks = [t['k_final'] for t in lv['trials'] if t['converged'] and 'k_final' in t]
        if Es:
            ax.scatter(Es, ks, color=c, s=40, alpha=0.7, edgecolors='white', linewidth=0.5,
                       label=f'{lv["noise_level"]*100:.0f}%')
    ax.plot(d['E_true'], d['k_true'], 'r*', markersize=15, zorder=10, label='True')
    ax.set_xlabel('E [MPa]')
    ax.set_ylabel('k [MPa]')
    ax.set_title('Recovered Parameters')
    ax.legend(fontsize=7, ncol=2, title='Noise')

    plt.suptitle('E2: Noise Robustness of Two-Stage Inversion', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'e2_noise_robustness', '噪声鲁棒性分析.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ E2: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# E3: Triaxial Inversion Summary
# ═══════════════════════════════════════════════════════════════════════════
def plot_e3():
    d = load_json(os.path.join(PARAM_INV, 'e3_triaxial_inversion/results.json'))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.axis('off')
    info = (
        f"E3: Triaxial Cylinder Inversion\n\n"
        f"Configuration:\n"
        f"    Confining pressure: {d['confining_pressure']:.1f} MPa\n"
        f"    Load steps: {d['n_load_steps']}\n"
        f"    Elastic disp: {d['disp_elastic']}\n"
        f"    Plastic disp: {d['disp_plastic']}\n\n"
        f"Observations:\n"
        f"    σ_zz (elastic): {d['obs_elastic']:.6f} MPa\n"
        f"    σ_zz (plastic): {d['obs_plastic']:.6f} MPa\n\n"
        f"Stage 1 (E recovery):\n"
        f"    E = {d['stage1']['E']:.2f} MPa  (error = {d['stage1']['err_E']:.2e})\n"
        f"    Function evaluations: {d['stage1']['nfev']}\n\n"
        f"Stage 2 (k recovery):\n"
        f"    k = {d['stage2']['k']:.6f} MPa  (error = {d['stage2']['err_k']:.2e})\n"
        f"    Function evaluations: {d['stage2']['nfev']}\n\n"
        f"Final result:\n"
        f"    E error: {d['final']['err_E']:.2e}   ({'PASS' if d['final']['err_E'] < 0.01 else 'FAIL'})\n"
        f"    k error: {d['final']['err_k']:.2e}   ({'PASS' if d['final']['err_k'] < 0.01 else 'FAIL'})\n"
        f"    Total time: {d['total_time_s']:.1f} s"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4e8', alpha=0.8))
    plt.title('E3: Triaxial Cylinder Inversion Results', fontsize=14)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'e3_triaxial_inversion', '三轴反演结果.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ E3: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# F1: Gradient Accuracy — V-curve
# ═══════════════════════════════════════════════════════════════════════════
def plot_f1():
    d = load_json(os.path.join(PARAM_INV, 'f1_gradient_accuracy/results.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, pname in enumerate(['E', 'k']):
        ax = axes[idx]
        for regime, color, marker, ls in [('elastic', '#4C72B0', 'o', '-'),
                                           ('plastic', '#C44E52', 's', '--')]:
            data = d[regime]['fd_sweep']
            eps = [dd['rel_eps'] for dd in data]
            errs = [max(dd[f'err_{pname}'], 1e-16) for dd in data]
            ax.loglog(eps, errs, color=color, marker=marker, ms=5, lw=1.5,
                      ls=ls, label=f'FD ({regime})')

        ax.axhline(1e-14, color='#55A868', ls=':', lw=1.5, alpha=0.8,
                   label='Machine precision')
        ax.axhspan(0, 1e-14, alpha=0.05, color='green')
        ax.set_xlabel('Relative Perturbation ε')
        ax.set_ylabel(f'Relative Error in ∂L/∂{pname}')
        ax.set_title(f'FD Accuracy: ∂L/∂{pname}')
        ax.legend(fontsize=8)
        ax.invert_xaxis()
        ax.set_ylim(1e-16, 1e2)

    plt.suptitle('F1: FD Gradient Accuracy vs Step Size (AD = Ground Truth)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'f1_gradient_accuracy', 'AD与FD梯度精度V曲线.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ F1: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# F2: Gradient Timing — AD vs FD
# ═══════════════════════════════════════════════════════════════════════════
def plot_f2():
    d = load_json(os.path.join(PARAM_INV, 'f2_gradient_timing/results.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    meshes = [r['mesh'] for r in d]
    ncells = [r['n_cells'] for r in d]
    t_fwd = [r['t_fwd'] * 1000 for r in d]
    t_ad = [r['t_ad'] * 1000 for r in d]
    t_fd_2p = [r['t_fd_2p'] * 1000 for r in d]
    speedup = [r['speedup_2p'] for r in d]
    overhead = [r['ad_overhead'] for r in d]

    # Timing bars
    ax = axes[0]
    x = np.arange(len(meshes))
    w = 0.25
    ax.bar(x - w, t_fwd, w, label='Forward only', color='#55A868', alpha=0.85)
    ax.bar(x, t_ad, w, label='AD (adjoint)', color='#4C72B0', alpha=0.85)
    ax.bar(x + w, t_fd_2p, w, label='FD (2-param)', color='#DD8452', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(meshes)
    ax.set_xlabel('Mesh Size')
    ax.set_ylabel('Time [ms]')
    ax.set_title('Gradient Computation Time')
    ax.legend(fontsize=8)

    # Speedup + overhead
    ax = axes[1]
    ax.plot(ncells, speedup, 'o-', color='#4C72B0', lw=2, ms=8, label='AD/FD speedup')
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('FD / AD Speedup', color='#4C72B0')
    ax.tick_params(axis='y', labelcolor='#4C72B0')
    ax2 = ax.twinx()
    ax2.plot(ncells, overhead, 's--', color='#C44E52', lw=1.5, ms=6, label='AD overhead')
    ax2.set_ylabel('AD / Forward Ratio', color='#C44E52')
    ax2.tick_params(axis='y', labelcolor='#C44E52')
    ax.set_title('AD Speedup over FD')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.suptitle('F2: AD vs FD Gradient Timing', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'f2_gradient_timing', 'AD与FD梯度计算时间对比.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ F2: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# F3: AD vs FD Inversion Comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_f3():
    d = load_json(os.path.join(PARAM_INV, 'f3_inversion_comparison/results.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = ['AD', 'FD']
    keys = ['ad', 'fd']

    # Accuracy
    ax = axes[0]
    err_E = [d[k]['err_E'] for k in keys]
    err_k = [d[k]['err_k'] for k in keys]
    x = np.arange(2)
    w = 0.3
    ax.bar(x - w/2, [max(e, 1e-16) for e in err_E], w,
           label='E error', color='#4C72B0', alpha=0.85)
    ax.bar(x + w/2, [max(e, 1e-16) for e in err_k], w,
           label='k error', color='#DD8452', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Relative Error')
    ax.set_yscale('log')
    ax.set_title('Inversion Accuracy')
    ax.legend()
    ax.set_ylim(1e-16, 1)

    # Time + Evals
    ax = axes[1]
    times = [d[k]['total_time'] for k in keys]
    evals = [d[k]['n_evals'] for k in keys]
    colors = ['#4C72B0', '#DD8452']
    bars = ax.bar(methods, times, color=colors, alpha=0.85, width=0.5)
    ax.set_ylabel('Total Time [s]')
    ax.set_title(f'Inversion Time (Speedup: {d["speedup"]:.1f}×)')
    for bar, t, ne in zip(bars, times, evals):
        ax.text(bar.get_x() + bar.get_width()/2, t + 0.1,
                f'{t:.2f}s\n({ne} evals)', ha='center', fontsize=9)

    plt.suptitle('F3: AD vs FD Full Inversion Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'f3_inversion_comparison', 'AD与FD完整反演对比.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ F3: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# F4: Scaling — AD vs FD with increasing parameters
# ═══════════════════════════════════════════════════════════════════════════
def plot_f4():
    d = load_json(os.path.join(PARAM_INV, 'f4_scaling/results.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_params = [r['n_params'] for r in d]
    t_ad = [r['t_ad'] * 1000 for r in d]
    t_fd = [r['t_fd'] * 1000 for r in d]
    speedups = [r['speedup'] for r in d]
    param_labels = [', '.join(r['params']) for r in d]

    # Timing
    ax = axes[0]
    ax.plot(n_params, t_ad, 'o-', color='#4C72B0', lw=2, ms=8, label='AD')
    ax.plot(n_params, t_fd, 's-', color='#DD8452', lw=2, ms=8, label='FD')
    ax.fill_between(n_params, t_ad, t_fd, alpha=0.1, color='#DD8452')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Gradient Time [ms]')
    ax.set_title('Gradient Time: AD vs FD')
    ax.set_xticks(n_params)
    ax.legend()

    # Speedup
    ax = axes[1]
    bars = ax.bar(n_params, speedups, color='#55A868', alpha=0.85, width=0.6)
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('FD / AD Speedup')
    ax.set_title('AD Speedup (O(1) vs O(n))')
    ax.set_xticks(n_params)
    for bar, s, lbl in zip(bars, speedups, param_labels):
        ax.text(bar.get_x() + bar.get_width()/2, s + 0.05,
                f'{s:.1f}×\n({lbl})', ha='center', fontsize=7.5, va='bottom')
    # Theoretical O(n) reference
    ax.plot(n_params, n_params, 'k--', alpha=0.4, label='Theoretical O(n)')
    ax.legend(fontsize=8)

    plt.suptitle('F4: AD vs FD Scaling with Number of Parameters', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'f4_scaling', 'AD与FD参数数量缩放性.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ F4: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# F5: FD Step-Size Sensitivity Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_f5():
    d = load_json(os.path.join(PARAM_INV, 'f5_fd_sensitivity/results.json'))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    err_grid = np.array(d['err_grid'])
    eps_E = d['eps_E_list']
    eps_k = d['eps_k_list']

    # Replace NaN for display
    err_display = np.where(np.isnan(err_grid), 0, err_grid)
    mask = np.isnan(err_grid)

    im = ax.imshow(err_display, origin='lower', aspect='auto',
                   cmap='RdYlGn_r', interpolation='nearest')
    # Mark NaN cells
    for i in range(err_grid.shape[0]):
        for j in range(err_grid.shape[1]):
            if mask[i, j]:
                ax.text(j, i, '✗', ha='center', va='center', fontsize=14, color='red')
            else:
                ax.text(j, i, f'{err_grid[i,j]:.4f}', ha='center', va='center',
                        fontsize=8)

    ax.set_xticks(range(len(eps_k)))
    ax.set_xticklabels([f'{e}' for e in eps_k])
    ax.set_yticks(range(len(eps_E)))
    ax.set_yticklabels([f'{e}' for e in eps_E])
    ax.set_xlabel('ε_k (k perturbation)')
    ax.set_ylabel('ε_E (E perturbation)')
    ax.set_title(f'FD Inversion Error vs Step Size\n'
                 f'(AD baseline err = {d["ad_baseline"]["err"]:.4f})')
    plt.colorbar(im, ax=ax, label='k relative error')

    plt.suptitle('F5: FD Step-Size Sensitivity Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PARAM_INV, 'f5_fd_sensitivity', 'FD步长灵敏度热力图.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ F5: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# I3: Regularization Study — λ vs Error curves
# ═══════════════════════════════════════════════════════════════════════════
def plot_i3():
    data = load_json(os.path.join(HETERO_INV, 'i3_regularization_study/results.json'))

    noise_levels = sorted(set(r['noise_level'] for r in data if 'L2_relative_error' in r))
    reg_types = ['tv', 'laplacian']

    fig, axes = plt.subplots(len(noise_levels), 2, figsize=(14, 4 * len(noise_levels)))
    if len(noise_levels) == 1:
        axes = axes[np.newaxis, :]

    for row, noise in enumerate(noise_levels):
        for col, rtype in enumerate(reg_types):
            ax = axes[row, col]
            subset = [r for r in data
                      if r['noise_level'] == noise and r['reg_type'] == rtype
                      and 'L2_relative_error' in r]
            if not subset:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            lams = [r['lambda'] for r in subset]
            l2 = [r['L2_relative_error'] for r in subset]
            mean_err = [r['mean_relative_error'] for r in subset]

            ax.semilogx(lams, l2, 'o-', color='#4C72B0', lw=2, ms=6, label='L2 rel error')
            ax.semilogx(lams, mean_err, 's--', color='#55A868', lw=1.5, ms=5,
                        label='Mean rel error')
            ax.set_xlabel('Regularization Weight λ')
            ax.set_ylabel('Relative Error')
            ax.set_title(f'{rtype.upper()} regularizer, Noise = {noise*100:.0f}%')
            ax.legend(fontsize=7)
            ax.set_ylim(0, min(max(l2) * 1.2, 1.0))

    # Also add no-reg baseline
    no_reg = [r for r in data if r['reg_type'] == 'none' and 'L2_relative_error' in r]
    if no_reg:
        for row, noise in enumerate(noise_levels):
            nr = [r for r in no_reg if r['noise_level'] == noise]
            if nr:
                for col in range(2):
                    axes[row, col].axhline(nr[0]['L2_relative_error'],
                                           color='gray', ls='--', lw=1, alpha=0.6,
                                           label=f'No reg (L2={nr[0]["L2_relative_error"]:.3f})')
                    axes[row, col].legend(fontsize=7)

    plt.suptitle('I3: Regularization Effect on Heterogeneous E-Field Inversion',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(HETERO_INV, 'i3_regularization_study', '正则化参数影响分析.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ I3: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# J1: Layered E-field Inversion — Layer-by-layer accuracy
# ═══════════════════════════════════════════════════════════════════════════
def plot_j1():
    d = load_json(os.path.join(HETERO_INV, 'j1_layered/results.json'))
    layers = d['layer_stats']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Layer-by-layer bar chart
    ax = axes[0]
    layer_names = [f"Layer {l['layer']}\n({l['y_range'][0]:.0f}-{l['y_range'][1]:.0f})" for l in layers]
    E_true = [l['E_true'] / 1000 for l in layers]
    E_inv = [l['mean_inverted'] / 1000 for l in layers]
    E_std = [l['std_inverted'] / 1000 for l in layers]
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w/2, E_true, w, label='True E', color='#4C72B0', alpha=0.85)
    ax.bar(x + w/2, E_inv, w, yerr=E_std, label='Inverted E (±σ)',
           color='#DD8452', alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel("Young's Modulus E [GPa]")
    ax.set_title('Layer-by-Layer E Recovery')
    ax.legend()

    # Relative error per layer
    ax = axes[1]
    rel_errs = [l['relative_error'] * 100 for l in layers]
    colors = ['#55A868' if e < 5 else '#DD8452' if e < 10 else '#C44E52' for e in rel_errs]
    bars = ax.bar(layer_names, rel_errs, color=colors, alpha=0.85)
    ax.axhline(5, color='gray', ls='--', lw=0.8, alpha=0.6, label='5% threshold')
    ax.set_ylabel('Mean Relative Error [%]')
    ax.set_title(f'Layer Recovery Error (L2 rel = {d["metrics"]["L2_relative_error"]:.2%})')
    ax.legend()
    for bar, e in zip(bars, rel_errs):
        ax.text(bar.get_x() + bar.get_width()/2, e + 0.3, f'{e:.1f}%',
                ha='center', fontsize=9)

    plt.suptitle(f'J1: Four-Layer E-Field Inversion ({d["Nx"]}×{d["Ny"]} mesh, '
                 f'noise={d["noise_level"]*100:.0f}%)', fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(HETERO_INV, 'j1_layered', '四层E场反演结果.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ J1: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# J2: Weak Interlayer Detection
# ═══════════════════════════════════════════════════════════════════════════
def plot_j2():
    d = load_json(os.path.join(HETERO_INV, 'j2_weak_interlayer/metrics.json'))
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.axis('off')

    info = (
        f"J2: Weak Interlayer Detection\n\n"
        f"Configuration:\n"
        f"    Mesh: {d['Nx']}×{d['Ny']} = {d['num_cells']} elements\n"
        f"    Background E: {d['E_bg']/1000:.0f} GPa\n"
        f"    Weak zone E:  {d['E_weak']/1000:.0f} GPa\n"
        f"    Interlayer angle: {d['angle_deg']}°, width: {d['width']} m\n"
        f"    Noise: {d['noise_level']*100:.0f}%, Reg: {d['reg_type']} (λ={d['reg_weight']})\n\n"
        f"Detection Performance:\n"
        f"    Detection rate:      {d['detection_rate']:.1%}\n"
        f"    False positive rate:  {d['false_positive_rate']:.1%}\n\n"
        f"Accuracy Metrics:\n"
        f"    L2 relative error:   {d['metrics']['L2_relative_error']:.2%}\n"
        f"    Mean relative error:  {d['metrics']['mean_relative_error']:.2%}\n"
        f"    Max relative error:   {d['metrics']['max_relative_error']:.2%}\n\n"
        f"Zone Statistics:\n"
        f"    Interlayer: mean={d['interlayer_mean']/1000:.1f} GPa, "
        f"σ={d['interlayer_std']/1000:.1f} GPa\n"
        f"    Background: mean={d['background_mean']/1000:.1f} GPa, "
        f"σ={d['background_std']/1000:.1f} GPa\n\n"
        f"Optimization: {d['nit']} iterations, {d['time_s']:.1f}s"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', alpha=0.9))
    plt.title('J2: Weak Interlayer Detection Results', fontsize=14)
    plt.tight_layout()
    out = os.path.join(HETERO_INV, 'j2_weak_interlayer', '弱夹层检测结果.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ J2: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# J3: Random Field Inversion
# ═══════════════════════════════════════════════════════════════════════════
def plot_j3():
    d = load_json(os.path.join(HETERO_INV, 'j3_random_field/metrics.json'))
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.axis('off')

    info = (
        f"J3: Random Field Inversion\n\n"
        f"Configuration:\n"
        f"    Mesh: {d['Nx']}×{d['Ny']} = {d['num_cells']} elements\n"
        f"    Noise: {d['noise_level']*100:.0f}%\n"
        f"    Regularization: {d['reg_type']} (λ={d['reg_weight']})\n"
        f"    Optimizer: {d['optimizer']}, max iter: {d['maxiter']}\n\n"
        f"Accuracy Metrics:\n"
        f"    L2 relative error:    {d['metrics']['L2_relative_error']:.4f}  ({d['metrics']['L2_relative_error']:.2%})\n"
        f"    Mean relative error:   {d['metrics']['mean_relative_error']:.4f}  ({d['metrics']['mean_relative_error']:.2%})\n"
        f"    Max relative error:    {d['metrics']['max_relative_error']:.4f}  ({d['metrics']['max_relative_error']:.2%})\n"
        f"    Correlation coeff:     {d['correlation_coefficient']:.4f}\n\n"
        f"Field Statistics:\n"
        f"    True field:     mean={d['true_field_stats']['mean']/1000:.1f} GPa, "
        f"σ={d['true_field_stats']['std']/1000:.1f} GPa\n"
        f"    Inverted field: mean={d['inverted_field_stats']['mean']/1000:.1f} GPa, "
        f"σ={d['inverted_field_stats']['std']/1000:.1f} GPa\n\n"
        f"Optimization:\n"
        f"    Iterations: {d['nit']}, Time: {d['time_s']:.1f}s\n"
        f"    Final loss: {d['final_loss']:.4e} (at true: {d['loss_at_true']:.4e})"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', alpha=0.9))
    plt.title('J3: Random Field Inversion Results', fontsize=14)
    plt.tight_layout()
    out = os.path.join(HETERO_INV, 'j3_random_field', '随机场反演结果.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ J3: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# J4: Scaling — AD speedup for heterogeneous inversion
# ═══════════════════════════════════════════════════════════════════════════
def plot_j4():
    d = load_json(os.path.join(HETERO_INV, 'j4_scaling/scaling_results.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    N = [r['N'] for r in d]
    t_ad = [r['ad_time_s'] for r in d]
    fd_times = [r['fd_time_est_s'] for r in d if r['fd_time_est_s'] is not None]
    N_fd = [r['N'] for r in d if r['fd_time_est_s'] is not None]
    speedups = [r['speedup'] for r in d if r['speedup'] is not None]
    N_sp = [r['N'] for r in d if r['speedup'] is not None]

    # Timing comparison
    ax = axes[0]
    ax.loglog(N, t_ad, 'o-', color='#4C72B0', lw=2, ms=8, label='AD gradient')
    if fd_times:
        ax.loglog(N_fd, fd_times, 's-', color='#DD8452', lw=2, ms=8, label='FD gradient (est.)')
    ax.set_xlabel('Number of Parameters (cells)')
    ax.set_ylabel('Gradient Time [s]')
    ax.set_title('Gradient Computation Time')
    ax.legend()

    # Speedup
    ax = axes[1]
    if speedups:
        ax.loglog(N_sp, speedups, 'o-', color='#55A868', lw=2, ms=8, label='Measured')
        # Theoretical O(N)
        ax.loglog(N_sp, N_sp, 'k--', alpha=0.4, label='Theoretical O(N)')
        ax.set_xlabel('Number of Parameters (cells)')
        ax.set_ylabel('AD / FD Speedup')
        ax.set_title('AD Speedup Scaling')
        ax.legend()
        for n, s in zip(N_sp, speedups):
            ax.annotate(f'{s:.0f}×', (n, s), textcoords='offset points',
                        xytext=(10, 5), fontsize=9)

    plt.suptitle('J4: AD Gradient Scaling for Heterogeneous E-Field Inversion',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(HETERO_INV, 'j4_scaling', 'AD梯度缩放性分析.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ J4: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Robustness Diagnostic: AD vs FD across displacements
# ═══════════════════════════════════════════════════════════════════════════
def plot_robustness():
    d = load_json(os.path.join(DIAG, 'exp_robustness/robustness_details.json'))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    disps = [r['displacement'] for r in d]
    ad_E = [r['ad_E'] for r in d]
    fd_E = [r['fd_E'] for r in d]
    ad_k = [r['ad_k'] for r in d]
    fd_k = [r['fd_k'] for r in d]
    rel_err_E = [r['rel_err_E'] for r in d]
    rel_err_k = [r['rel_err_k'] for r in d]
    plastic_count = [r['plastic_count'] for r in d]
    regimes = [r['regime'] for r in d]

    # Gradient ∂L/∂E
    ax = axes[0, 0]
    ax.plot(disps, ad_E, 'o-', color='#4C72B0', lw=1.5, ms=5, label='AD')
    ax.plot(disps, fd_E, 's--', color='#DD8452', lw=1, ms=4, label='FD')
    ax.set_xlabel('Applied Displacement')
    ax.set_ylabel('∂L/∂E')
    ax.set_title('Gradient ∂L/∂E')
    ax.legend()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

    # Gradient ∂L/∂k
    ax = axes[0, 1]
    ax.plot(disps, ad_k, 'o-', color='#4C72B0', lw=1.5, ms=5, label='AD')
    ax.plot(disps, fd_k, 's--', color='#DD8452', lw=1, ms=4, label='FD')
    ax.set_xlabel('Applied Displacement')
    ax.set_ylabel('∂L/∂k')
    ax.set_title('Gradient ∂L/∂k')
    ax.legend()

    # Relative errors
    ax = axes[1, 0]
    ax.semilogy(disps, [max(e, 1e-16) for e in rel_err_E], 'o-', color='#4C72B0',
                lw=1.5, ms=5, label='∂L/∂E error')
    ax.semilogy(disps, [max(e, 1e-16) for e in rel_err_k], 's-', color='#DD8452',
                lw=1.5, ms=5, label='∂L/∂k error')
    ax.axhline(1e-4, color='gray', ls='--', lw=0.8, alpha=0.5, label='10⁻⁴ threshold')
    ax.set_xlabel('Applied Displacement')
    ax.set_ylabel('Relative Error |FD - AD| / |AD|')
    ax.set_title('AD vs FD Agreement')
    ax.legend(fontsize=8)

    # Plastic quadrature points
    ax = axes[1, 1]
    colors = ['#55A868' if r == 'elastic' else '#C44E52' for r in regimes]
    ax.bar(range(len(disps)), plastic_count, color=colors, alpha=0.85)
    ax.set_xticks(range(len(disps)))
    ax.set_xticklabels([f'{d:.3f}' for d in disps], rotation=45, fontsize=7)
    ax.set_xlabel('Applied Displacement')
    ax.set_ylabel('# Plastic Quadrature Points')
    ax.set_title('Material Regime (green=elastic, red=plastic)')

    plt.suptitle('Gradient Robustness: AD vs FD Across Loading Regimes',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(DIAG, 'exp_robustness', '梯度鲁棒性_不同加载状态.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Robustness: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Grand Summary: All Experiments Overview
# ═══════════════════════════════════════════════════════════════════════════
def plot_grand_summary():
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Homogeneous inversion accuracy (D1, E3)
    ax = fig.add_subplot(gs[0, 0])
    d1 = load_json(os.path.join(PARAM_INV, 'd1_twin_experiment/results.json'))
    e3 = load_json(os.path.join(PARAM_INV, 'e3_triaxial_inversion/results.json'))
    exps = ['D1\nBlock', 'E3\nTriaxial']
    err_E = [d1['final']['err_E'], e3['final']['err_E']]
    err_k = [d1['final']['err_k'], e3['final']['err_k']]
    x = np.arange(2)
    w = 0.3
    ax.bar(x - w/2, [max(e, 1e-16) for e in err_E], w, label='E', color='#4C72B0')
    ax.bar(x + w/2, [max(e, 1e-16) for e in err_k], w, label='k', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(exps)
    ax.set_yscale('log')
    ax.set_ylabel('Relative Error')
    ax.set_title('Homogeneous Inversion')
    ax.legend(fontsize=8)
    ax.set_ylim(1e-16, 1)

    # 2. Noise robustness summary
    ax = fig.add_subplot(gs[0, 1])
    e2 = load_json(os.path.join(PARAM_INV, 'e2_noise_robustness/results.json'))
    noise_pct = [lv['noise_level'] * 100 for lv in e2['levels']]
    me = [lv['mean_err_E'] for lv in e2['levels']]
    mk = [lv['mean_err_k'] for lv in e2['levels']]
    ax.semilogy(noise_pct, me, 'o-', color='#4C72B0', lw=1.5, ms=5, label='E err')
    ax.semilogy(noise_pct, mk, 's-', color='#DD8452', lw=1.5, ms=5, label='k err')
    ax.axhline(0.01, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Noise [%]')
    ax.set_ylabel('Rel Error')
    ax.set_title('Noise Robustness (E2)')
    ax.legend(fontsize=8)

    # 3. AD vs FD speedup scaling (F4)
    ax = fig.add_subplot(gs[0, 2])
    f4 = load_json(os.path.join(PARAM_INV, 'f4_scaling/results.json'))
    n_p = [r['n_params'] for r in f4]
    sp = [r['speedup'] for r in f4]
    ax.bar(n_p, sp, color='#55A868', alpha=0.85, width=0.6)
    ax.plot(n_p, n_p, 'k--', alpha=0.4)
    ax.set_xlabel('# Parameters')
    ax.set_ylabel('Speedup')
    ax.set_title('AD Speedup (F4)')

    # 4. Condition number (E1)
    ax = fig.add_subplot(gs[1, 0])
    e1 = load_json(os.path.join(PARAM_INV, 'e1_displacement_inversion/results.json'))
    labels = ['Disp loss', 'σ_zz loss']
    conds = [e1['disp_joint']['condition_number'], e1['sigma_two_stage']['condition_number']]
    bars = ax.bar(labels, conds, color=['#4C72B0', '#DD8452'], alpha=0.85, width=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('Condition #')
    ax.set_title('Hessian Conditioning (E1)')
    for bar, c in zip(bars, conds):
        ax.text(bar.get_x() + bar.get_width()/2, c * 2, f'{c:.1e}',
                ha='center', fontsize=8)

    # 5. Heterogeneous scaling (J4)
    ax = fig.add_subplot(gs[1, 1])
    j4 = load_json(os.path.join(HETERO_INV, 'j4_scaling/scaling_results.json'))
    N = [r['N'] for r in j4]
    t_ad = [r['ad_time_s'] for r in j4]
    fd_t = [(r['fd_time_est_s'] if r['fd_time_est_s'] else None) for r in j4]
    ax.loglog(N, t_ad, 'o-', color='#4C72B0', lw=2, ms=6, label='AD')
    N_fd = [n for n, t in zip(N, fd_t) if t is not None]
    fd_valid = [t for t in fd_t if t is not None]
    if fd_valid:
        ax.loglog(N_fd, fd_valid, 's-', color='#DD8452', lw=2, ms=6, label='FD (est.)')
    ax.set_xlabel('# Cells')
    ax.set_ylabel('Time [s]')
    ax.set_title('Hetero Gradient Scaling (J4)')
    ax.legend(fontsize=8)

    # 6. Layer inversion (J1)
    ax = fig.add_subplot(gs[1, 2])
    j1 = load_json(os.path.join(HETERO_INV, 'j1_layered/results.json'))
    layers = j1['layer_stats']
    layer_names = [f"L{l['layer']}" for l in layers]
    E_true = [l['E_true']/1000 for l in layers]
    E_inv = [l['mean_inverted']/1000 for l in layers]
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w/2, E_true, w, label='True', color='#4C72B0')
    ax.bar(x + w/2, E_inv, w, label='Inverted', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel('E [GPa]')
    ax.set_title('Layered Inversion (J1)')
    ax.legend(fontsize=8)

    # 7. FD V-curve (F1)
    ax = fig.add_subplot(gs[2, 0])
    f1 = load_json(os.path.join(PARAM_INV, 'f1_gradient_accuracy/results.json'))
    for regime, c, m in [('elastic', '#4C72B0', 'o'), ('plastic', '#C44E52', 's')]:
        eps = [dd['rel_eps'] for dd in f1[regime]['fd_sweep']]
        errs = [max(dd['err_E'], 1e-16) for dd in f1[regime]['fd_sweep']]
        ax.loglog(eps, errs, c=c, marker=m, ms=4, lw=1.2, label=f'{regime}')
    ax.axhline(1e-14, color='#55A868', ls=':', lw=1)
    ax.set_xlabel('ε')
    ax.set_ylabel('∂L/∂E Error')
    ax.set_title('FD V-curve (F1)')
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    # 8. Regularization effect (I3)
    ax = fig.add_subplot(gs[2, 1])
    i3 = load_json(os.path.join(HETERO_INV, 'i3_regularization_study/results.json'))
    for rtype, c, m in [('tv', '#4C72B0', 'o'), ('laplacian', '#DD8452', 's')]:
        subset = [r for r in i3 if r['noise_level'] == 0.0 and r['reg_type'] == rtype
                  and 'L2_relative_error' in r]
        if subset:
            lams = [r['lambda'] for r in subset]
            errs = [r['L2_relative_error'] for r in subset]
            ax.semilogx(lams, errs, f'{m}-', color=c, lw=1.5, ms=5, label=rtype.upper())
    no_reg = [r for r in i3 if r['noise_level'] == 0.0 and r['reg_type'] == 'none'
              and 'L2_relative_error' in r]
    if no_reg:
        ax.axhline(no_reg[0]['L2_relative_error'], color='gray', ls='--', lw=0.8,
                   alpha=0.6, label='No reg')
    ax.set_xlabel('λ')
    ax.set_ylabel('L2 Rel Error')
    ax.set_title('Regularization (I3, 0% noise)')
    ax.legend(fontsize=8)

    # 9. AD vs FD inversion (F3)
    ax = fig.add_subplot(gs[2, 2])
    f3 = load_json(os.path.join(PARAM_INV, 'f3_inversion_comparison/results.json'))
    methods = ['AD', 'FD']
    times = [f3['ad']['total_time'], f3['fd']['total_time']]
    bars = ax.bar(methods, times, color=['#4C72B0', '#DD8452'], alpha=0.85, width=0.5)
    ax.set_ylabel('Time [s]')
    ax.set_title(f'Inversion Time ({f3["speedup"]:.1f}× speedup)')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, t + 0.1, f'{t:.2f}s',
                ha='center', fontsize=9)

    fig.suptitle('JAX-FEM-GEO: Differentiable Geomechanics Inversion — Complete Results Overview',
                 fontsize=16, y=0.98)
    out = os.path.join(RESULTS, '全部实验结果总览.png')
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Grand Summary: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Generating publication-quality plots for all experiments")
    print("=" * 60)

    plot_d1()
    plot_d2()
    plot_d3()
    plot_e1()
    plot_e2()
    plot_e3()
    plot_f1()
    plot_f2()
    plot_f3()
    plot_f4()
    plot_f5()
    plot_i3()
    plot_j1()
    plot_j2()
    plot_j3()
    plot_j4()
    plot_robustness()
    plot_grand_summary()

    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 60)
