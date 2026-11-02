"""
Fig 9: Protein-interface model comparison.
Shows all 4 models at protein-proximal vs non-proximal positions.
"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/glt/rnaxgb/RNKDB'
with open(f'{BASE}/fair_eval_v3/review_experiment_results.json') as f:
    data = json.load(f)

exp1 = data['experiment_1_protein_interface']
models = ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Panel A: Accuracy comparison ──
ax = axes[0]
x = np.arange(len(models))
width = 0.35

prox_vals = [exp1[m]['proximal_accuracy'] for m in models]
nonprox_vals = [exp1[m]['nonproximal_accuracy'] for m in models]

bars1 = ax.bar(x - width/2, prox_vals, width, label='Protein-proximal', color='#4477AA')
bars2 = ax.bar(x + width/2, nonprox_vals, width, label='Non-proximal', color='#CCCCCC')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Accuracy')
ax.set_title('Protein-interface accuracy by model')
ax.legend(fontsize=8)
ax.axhline(0.25, color='black', ls='--', alpha=0.3, label='Random')
ax.set_ylim(0, 1.0)

# Add delta labels
for i, m in enumerate(models):
    delta = exp1[m]['delta']
    color = 'green' if delta > 0 else 'red'
    y = max(prox_vals[i], nonprox_vals[i]) + 0.03
    ax.text(i, y, f'{delta:+.3f}', ha='center', fontsize=9, fontweight='bold', color=color)
    # Add n labels
    ax.text(i, 0.05, f'n={exp1[m]["proximal_n"]}', ha='center', fontsize=6, color='#4477AA')
    ax.text(i, 0.15, f'n={exp1[m]["nonproximal_n"]}', ha='center', fontsize=6, color='#999999')

# ── Panel B: Enrichment OR forest plot ──
ax = axes[1]

# Compute ORs for protein-proximal enrichment per model
ors = []
cis = []
for m in models:
    a = exp1[m]['proximal_accuracy'] * exp1[m]['proximal_n']
    b = exp1[m]['proximal_n'] - a
    c = exp1[m]['nonproximal_accuracy'] * exp1[m]['nonproximal_n']
    d = exp1[m]['nonproximal_n'] - c
    from scipy.stats import fisher_exact
    try:
        or_val, p = fisher_exact([[int(a), int(b)], [int(c), int(d)]])
        log_or = np.log(or_val)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d) if a>0 and b>0 and c>0 and d>0 else 0.1
        ci_lo = np.exp(log_or - 1.96*se)
        ci_hi = np.exp(log_or + 1.96*se)
    except:
        or_val, ci_lo, ci_hi = 1, 1, 1
    ors.append(or_val)
    cis.append((ci_lo, ci_hi))

colors_or = ['#4477AA', '#CC6677', '#88CCEE', '#AA7744']
for i, (m, or_val, (ci_lo, ci_hi), c) in enumerate(zip(models, ors, cis, colors_or)):
    ax.errorbar(or_val, i, xerr=[[or_val - ci_lo], [ci_hi - or_val]],
                fmt='o', color=c, capsize=4, markersize=10, lw=2)
    sig = '***' if ci_lo > 1 or ci_hi < 1 else ''
    ax.text(or_val + 0.05, i, f'{or_val:.2f} {sig}', va='center', fontsize=10, fontweight='bold')

ax.axvline(1.0, color='black', ls='--', alpha=0.5)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models)
ax.set_xlabel('Odds Ratio (proximal vs non-proximal)')
ax.set_title('Protein-interface enrichment (OR)')
ax.set_xlim(0.5, 2.5)
ax.text(1.0, -0.8, '← Depleted | Enriched →', ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig('/home/glt/rnaxgb/Nature-comms-v2/figures/fig9_protein_interface.pdf',
            dpi=150, bbox_inches='tight')
plt.savefig('/home/glt/rnaxgb/Nature-comms-v2/figures/fig9_protein_interface.png',
            dpi=150, bbox_inches='tight')
print('Saved fig9_protein_interface')
