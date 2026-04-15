"""
L3: AD vs FD vs PSO — Combined comparison plots for paper.

Reads L1 (AD vs FD) and L2 (AD vs PSO) results and produces
publication-quality figures:
  1. Parameter dimension vs total inversion time (3 curves)
  2. Parameter dimension vs inversion accuracy (L2 relative error)
  3. Forward evaluations comparison (bar chart)
  4. Summary table figure
"""

import json
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = '/Users/chrisv/Projects/JAX-FEM/results/heterogeneous_inversion'
OUT_DIR = os.path.join(RESULTS, 'l3_method_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
with open(os.path.join(RESULTS, 'l1_fd_baseline/l1_results.json')) as f:
    l1_data = json.load(f)
with open(os.path.join(RESULTS, 'l2_pso_comparison/l2_results.json')) as f:
    l2_data = json.load(f)

# Build unified data structure
# AD data from L1 (all mesh sizes)
ad_Ns = [r['nc'] for r in l1_data]
ad_times = [r['ad_time'] for r in l1_data]
ad_errs = [r['ad_l2_rel'] for r in l1_data]
ad_nfev = [r['ad_nfev'] for r in l1_data]

# FD data from L1 (where available)
fd_Ns = [r['nc'] for r in l1_data if r['fd_time'] is not None]
fd_times = [r['fd_time'] for r in l1_data if r['fd_time'] is not None]
fd_errs = [r['fd_l2_rel'] for r in l1_data if r['fd_l2_rel'] is not None]
fd_nfev = [r['fd_nfev'] for r in l1_data if r['fd_nfev'] is not None]

# PSO data from L2
pso_Ns = [r['nc'] for r in l2_data]
pso_times = [r['pso_time'] for r in l2_data]
pso_errs = [r['pso_l2_rel'] for r in l2_data]
pso_nfev = [r['pso_nfev'] for r in l2_data]


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: The main paper figure (2×2 grid)
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Color scheme
C_AD = '#4C72B0'
C_FD = '#DD8452'
C_PSO = '#55A868'

# ── (a) Time vs #params ──
ax = axes[0, 0]
ax.semilogy(ad_Ns, ad_times, 'o-', color=C_AD, lw=2.5, ms=9, label='AD (L-BFGS-B)', zorder=3)
ax.semilogy(fd_Ns, fd_times, 's-', color=C_FD, lw=2.5, ms=9, label='FD (L-BFGS-B)', zorder=2)
ax.semilogy(pso_Ns, pso_times, '^-', color=C_PSO, lw=2.5, ms=9, label='PSO', zorder=2)
ax.set_xlabel('Number of Parameters N')
ax.set_ylabel('Total Inversion Time [s]')
ax.set_title('(a) Inversion Time vs Problem Size')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(15, 500)

# ── (b) Accuracy vs #params ──
ax = axes[0, 1]
ax.semilogy(ad_Ns, [e * 100 for e in ad_errs], 'o-', color=C_AD, lw=2.5, ms=9,
            label='AD (L-BFGS-B)', zorder=3)
ax.semilogy(fd_Ns, [e * 100 for e in fd_errs], 's-', color=C_FD, lw=2.5, ms=9,
            label='FD (L-BFGS-B)', zorder=2)
ax.semilogy(pso_Ns, [e * 100 for e in pso_errs], '^-', color=C_PSO, lw=2.5, ms=9,
            label='PSO', zorder=2)
ax.axhline(5, color='gray', ls='--', lw=1, alpha=0.5, label='5% target')
ax.set_xlabel('Number of Parameters N')
ax.set_ylabel('L2 Relative Error [%]')
ax.set_title('(b) Inversion Accuracy vs Problem Size')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(15, 500)

# ── (c) Forward evaluations (bar chart at common mesh sizes) ──
ax = axes[1, 0]
# Use mesh sizes where all three methods have data: 25, 64, 100
common_Ns = [25, 64, 100]
common_labels = ['25\n(5×5)', '64\n(8×8)', '100\n(10×10)']

ad_evals_common = [ad_nfev[ad_Ns.index(n)] for n in common_Ns]
fd_evals_common = [fd_nfev[fd_Ns.index(n)] for n in common_Ns]
pso_evals_common = [pso_nfev[pso_Ns.index(n)] for n in common_Ns]

x = np.arange(len(common_Ns))
w = 0.25
ax.bar(x - w, ad_evals_common, w, label='AD', color=C_AD, alpha=0.85)
ax.bar(x, fd_evals_common, w, label='FD', color=C_FD, alpha=0.85)
ax.bar(x + w, pso_evals_common, w, label='PSO', color=C_PSO, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(common_labels)
ax.set_xlabel('Number of Parameters N')
ax.set_ylabel('Forward Evaluations')
ax.set_title('(c) Computational Cost (Forward Solves)')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── (d) Speedup vs #params ──
ax = axes[1, 1]
# FD speedup from L1
fd_speedups_N = [r['nc'] for r in l1_data if r.get('speedup') is not None]
fd_speedups = [r['speedup'] for r in l1_data if r.get('speedup') is not None]
# PSO speedup from L2
pso_speedups_N = [r['nc'] for r in l2_data]
pso_speedups = [r['speedup'] for r in l2_data]

ax.plot(fd_speedups_N, fd_speedups, 's-', color=C_FD, lw=2.5, ms=9,
        label='vs FD')
ax.plot(pso_speedups_N, pso_speedups, '^-', color=C_PSO, lw=2.5, ms=9,
        label='vs PSO')
ax.set_xlabel('Number of Parameters N')
ax.set_ylabel('AD Speedup (×)')
ax.set_title('(d) AD Speedup Factor')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
for n, s in zip(fd_speedups_N, fd_speedups):
    ax.annotate(f'{s:.0f}×', (n, s), textcoords='offset points',
                xytext=(8, 5), fontsize=9, color=C_FD)
for n, s in zip(pso_speedups_N, pso_speedups):
    ax.annotate(f'{s:.0f}×', (n, s), textcoords='offset points',
                xytext=(8, -12), fontsize=9, color=C_PSO)

fig.suptitle('Method Comparison: AD vs FD vs PSO for Heterogeneous E-Field Inversion',
             fontsize=15, y=0.99)
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, '方法对比_AD_FD_PSO.png')
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Main figure: {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Summary table as figure (for paper/presentation)
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Build table data
col_labels = ['N (Mesh)', 'AD Time', 'AD L2%', 'FD Time', 'FD L2%',
              'PSO Time', 'PSO L2%', 'AD vs FD', 'AD vs PSO']

table_data = []
for n in [25, 64, 100, 225, 400]:
    row = [f'{n}']
    # AD
    i_ad = ad_Ns.index(n)
    row.append(f'{ad_times[i_ad]:.1f}s')
    row.append(f'{ad_errs[i_ad]:.2%}')
    # FD
    if n in fd_Ns:
        i_fd = fd_Ns.index(n)
        row.append(f'{fd_times[i_fd]:.1f}s')
        row.append(f'{fd_errs[i_fd]:.2%}')
    else:
        row.extend(['—', '—'])
    # PSO
    if n in pso_Ns:
        i_pso = pso_Ns.index(n)
        row.append(f'{pso_times[i_pso]:.1f}s')
        row.append(f'{pso_errs[i_pso]:.2%}')
    else:
        row.extend(['—', '—'])
    # Speedups
    l1_entry = next((r for r in l1_data if r['nc'] == n), None)
    l2_entry = next((r for r in l2_data if r['nc'] == n), None)
    row.append(f'{l1_entry["speedup"]:.1f}×' if l1_entry and l1_entry.get('speedup') else '—')
    row.append(f'{l2_entry["speedup"]:.1f}×' if l2_entry else '—')
    table_data.append(row)

table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

# Style header
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#4C72B0')
    table[(0, j)].set_text_props(color='white', weight='bold')

# Color code cells
for i in range(len(table_data)):
    # AD columns — blue tint
    table[(i + 1, 1)].set_facecolor('#E8EEF6')
    table[(i + 1, 2)].set_facecolor('#E8EEF6')
    # FD columns — orange tint
    table[(i + 1, 3)].set_facecolor('#FBE8D8')
    table[(i + 1, 4)].set_facecolor('#FBE8D8')
    # PSO columns — green tint
    table[(i + 1, 5)].set_facecolor('#E0F0E0')
    table[(i + 1, 6)].set_facecolor('#E0F0E0')

ax.set_title('Method Comparison Summary: AD vs FD vs PSO',
             fontsize=14, pad=20, weight='bold')
plt.tight_layout()
table_path = os.path.join(OUT_DIR, '方法对比汇总表.png')
plt.savefig(table_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Summary table: {table_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Save combined JSON
# ═══════════════════════════════════════════════════════════════════════════

combined = {
    'description': 'L3: AD vs FD vs PSO method comparison for heterogeneous E-field inversion',
    'benchmark': 'Two-region E field (E_left=50000, E_right=90000), traction=-50 MPa, k=500',
    'ad': [{'N': n, 'time': t, 'l2_rel': e, 'nfev': nf}
           for n, t, e, nf in zip(ad_Ns, ad_times, ad_errs, ad_nfev)],
    'fd': [{'N': n, 'time': t, 'l2_rel': e, 'nfev': nf}
           for n, t, e, nf in zip(fd_Ns, fd_times, fd_errs, fd_nfev)],
    'pso': [{'N': n, 'time': t, 'l2_rel': e, 'nfev': nf}
            for n, t, e, nf in zip(pso_Ns, pso_times, pso_errs, pso_nfev)],
    'key_findings': [
        'AD time nearly constant (~13s) across 25-400 parameters',
        'FD time grows linearly with N (64s at N=25, 196s at N=225)',
        'PSO fails completely at all tested dimensions (L2 > 50%)',
        'AD speedup vs FD: 10-15x',
        'AD speedup vs PSO: 5-12x (but PSO accuracy is useless)',
        'AD gradient matches FD gradient with cosine similarity = 1.0',
    ],
}

json_path = os.path.join(OUT_DIR, 'l3_combined.json')
with open(json_path, 'w') as f:
    json.dump(combined, f, indent=2)
print(f"✓ Combined data: {json_path}")

print("\nL3 complete — all method comparison plots generated.")
