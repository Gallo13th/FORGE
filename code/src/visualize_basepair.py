"""
Base-pair state prediction visualization — Nature Methods submission style.
Compares GeoRNA, RNAfold MFE, and majority baseline against DSSR ground truth.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
OUT_DIR = BASE / 'stress_test' / 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# NM palette
NM_BLUE   = '#2F6B9A'
NM_RED    = '#B94A48'
NM_GREEN  = '#4F8A5B'
NM_ORANGE = '#D8903D'
NM_PURPLE = '#7B5EA7'
NM_GREY   = '#6F7378'
NM_DARK   = '#20242A'
NM_CYAN   = '#3A8FBF'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})


def label_panel(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            ha='left', va='top', fontsize=9, fontweight='bold')


def savefig(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT_DIR / f'{stem}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'  {stem}')


# ── Load data ──
bp_results = json.load(open(BASE / 'fair_eval_v3' / 'basepair_results.json'))
rf_results = json.load(open(BASE / 'fair_eval_v3' / 'rnafold_baseline_results.json'))

geo_acc = bp_results['accuracy']
geo_auc = bp_results['auc']
geo_bacc = bp_results['balanced_acc']
geo_rec_paired = bp_results['recall_paired']
geo_rec_unpaired = bp_results['recall_unpaired']
geo_prec_paired = bp_results['precision_paired']
geo_f1 = bp_results['f1']

rf_acc = rf_results['accuracy']
rf_auc = rf_results['auc']
rf_bacc = rf_results['balanced_accuracy']
rf_rec_paired = rf_results['recall_paired']
rf_rec_unpaired = rf_results['recall_unpaired']
rf_prec_paired = rf_results['precision_paired']
rf_f1 = rf_results['f1']
rf_paired_frac = rf_results['rnafold_paired_fraction']
true_paired_frac = rf_results['true_paired_fraction']
rf_maj = rf_results['majority_baseline']

fig = plt.figure(figsize=(7.2, 4.8))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.42)

# ── (a) Accuracy bar: majority vs RNAfold vs GeoRNA ──
ax = fig.add_subplot(gs[0, 0])
methods = ['Majority\nbaseline', 'RNAfold\n(MFE)', 'GeoRNA\n(geometry)']
accs = [rf_maj * 100, rf_acc * 100, geo_acc * 100]
deltas = [0, (rf_acc - rf_maj) * 100, (geo_acc - rf_maj) * 100]
colors = [NM_GREY, NM_CYAN, NM_BLUE]
x = np.arange(3)
bars = ax.bar(x, accs, color=colors, edgecolor='white', linewidth=0.3, width=0.55)

for i, (a, d, c) in enumerate(zip(accs, deltas, colors)):
    if d > 0:
        ax.text(i, a + 1.2, f'+{d:.1f}pp', ha='center', fontsize=7, fontweight='bold', color=c)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=6.5)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(55, 88)
ax.set_title('Base-pair prediction accuracy', fontweight='bold')
label_panel(ax, 'a')

# ── (b) AUC comparison ──
ax = fig.add_subplot(gs[0, 1])
auc_methods = ['RNAfold\n(MFE)', 'GeoRNA\n(geometry)']
auc_vals = [rf_auc, geo_auc]
auc_colors = [NM_CYAN, NM_BLUE]
x = np.arange(2)
ax.bar(x, auc_vals, color=auc_colors, edgecolor='white', linewidth=0.3, width=0.45)
ax.axhline(y=0.5, color=NM_GREY, ls='--', lw=0.7, alpha=0.5, label='Random AUC=0.5')
for i, v in enumerate(auc_vals):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold', color=auc_colors[i])
ax.set_xticks(x)
ax.set_xticklabels(auc_methods, fontsize=6.5)
ax.set_ylabel('AUC')
ax.set_ylim(0.4, 1.0)
ax.set_title('Discrimination (AUC)', fontweight='bold')
ax.legend(frameon=False, fontsize=5.5)
label_panel(ax, 'b')

# ── (c) Paired vs unpaired recall ──
ax = fig.add_subplot(gs[1, 0])
metrics = ['Recall\npaired', 'Recall\nunpaired', 'Precision\npaired']
geo_vals = [geo_rec_paired, geo_rec_unpaired, geo_prec_paired]
rf_vals  = [rf_rec_paired, rf_rec_unpaired, rf_prec_paired]
x = np.arange(len(metrics))
w = 0.28
ax.bar(x - w/2, [v*100 for v in rf_vals], w, color=NM_CYAN, alpha=0.85,
       edgecolor='white', linewidth=0.3, label='RNAfold')
ax.bar(x + w/2, [v*100 for v in geo_vals], w, color=NM_BLUE, alpha=0.85,
       edgecolor='white', linewidth=0.3, label='GeoRNA')

# Highlight the big gap
for i, (g, r) in enumerate(zip(geo_vals, rf_vals)):
    delta = (g - r) * 100
    if abs(delta) > 3:
        ax.annotate(f'Δ={delta:+.1f}pp', xy=(i + w/2, g*100 + 1),
                   ha='center', fontsize=6.5, fontweight='bold',
                   color=NM_GREEN if delta > 0 else NM_RED)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=6.5)
ax.set_ylabel('Rate (%)')
ax.set_ylim(0, 100)
ax.set_title('Per-class breakdown', fontweight='bold')
ax.legend(frameon=False, fontsize=6, loc='lower left')
label_panel(ax, 'c')

# ── (d) RNAfold over-pairing analysis ──
ax = fig.add_subplot(gs[1, 1])
pair_data = [
    ('True\n(DSSR)', true_paired_frac * 100, NM_GREY),
    ('RNAfold\npredicted', rf_paired_frac * 100, NM_CYAN),
]
x = np.arange(2)
ax.bar(x, [d[1] for d in pair_data], color=[d[2] for d in pair_data],
       edgecolor='white', linewidth=0.3, width=0.45)
for i, (label, v, c) in enumerate(pair_data):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold', color=c)

overpair = (rf_paired_frac - true_paired_frac) * 100
ax.annotate(f'Over-pairing:\n+{overpair:.1f}pp',
            xy=(0.5, max(true_paired_frac, rf_paired_frac)*100 + 8),
            ha='center', fontsize=7, fontweight='bold', color=NM_RED,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5F5', edgecolor=NM_RED, alpha=0.7, lw=0.5))

ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in pair_data], fontsize=6.5)
ax.set_ylabel('Paired fraction (%)')
ax.set_ylim(0, 70)
ax.set_title('RNAfold over-predicts base-pairing', fontweight='bold')
label_panel(ax, 'd')

fig.suptitle('Base-pair state prediction: GeoRNA vs RNAfold vs DSSR ground truth',
             fontweight='bold', x=0.02, ha='left', fontsize=9.5)
savefig(fig, 'fig_basepair_comparison')
print('Done.')
