"""
Four-task comparison figures — Nature Methods submission style.
English labels, consistent NM colour palette, clean journal aesthetic.
Run from RNKDB/:
    python ../src/visualize_four_tasks.py
"""
import json, os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
OUT_DIR = BASE / 'stress_test' / 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Nature Methods colour palette ──
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


# ── Compute DMS binary classification metrics from test data ──
def _compute_dms_binary():
    y_true = np.load(BASE / 'fair_eval_v3' / 'dms_full_test_ytrue.npy')
    y_pred = np.load(BASE / 'fair_eval_v3' / 'dms_full_test_ypred.npy')
    # Median split: above median = high reactivity (1), below = low (0)
    median = np.median(y_true)
    y_bin_true = (y_true >= median).astype(int)
    # Predicted scores for AUC; for accuracy, threshold predictions at same median
    y_bin_pred = (y_pred >= median).astype(int)
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y_bin_true, y_bin_pred)
    baseline = max(y_bin_true.mean(), 1 - y_bin_true.mean())
    auc = roc_auc_score(y_bin_true, y_pred)
    return float(acc), float(baseline), float(auc)

_dms_bin_acc, _dms_bin_bl, _dms_bin_auc = _compute_dms_binary()
_dms_bin_delta = (_dms_bin_acc - _dms_bin_bl) * 100

# ── Task data ──
# (label, acc, baseline, auc, r2, delta_pp, color, desc)
TASKS = [
    ('Nucleotide\n(4-class)',     0.646, 0.250, 0.95,  None, 39.6, NM_GREEN,  'A/C/G/U'),
    ('Base-pair\n(DSSR)',         0.792, 0.631, 0.880, None, 16.1, NM_BLUE,   'paired / unpaired'),
    ('DMS (binary)\n(median split)', _dms_bin_acc, _dms_bin_bl, _dms_bin_auc, None, _dms_bin_delta, NM_ORANGE, 'high/low reactivity'),
    ('DMS\nreactivity',           None,  None,  None,  0.329, None, NM_ORANGE, 'regression (R²)'),
    ('Protein binding\n(<6Å)', 0.629, 0.577, 0.678, None,  5.2, NM_RED,    'bound / unbound'),
]
DMS_EXTRA = {'r_pearson': 0.576, 'rho': 0.516, 'per_chain_r_mean': 0.400, 'per_chain_r_median': 0.419}


def fig1_four_tasks_metrics():
    """Panel a: classification accuracy vs baseline (4 tasks, DMS as binary).
       Panel b: AUC bars for all classification tasks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))

    # Panel a: classification accuracy bars — 4 tasks (nucleotide, base-pair, DMS binary, protein binding)
    cls_labels = ['Nucleotide\n(4-class)', 'Base-pair\n(DSSR)', 'DMS (binary)\n(median split)', 'Protein binding\n(<6Å)']
    cls_tasks = [t for t in TASKS if t[1] is not None and t[0] != 'DMS\nreactivity']
    names = [t[0] for t in cls_tasks]
    accs  = [t[1]*100 for t in cls_tasks]
    bls   = [t[2]*100 for t in cls_tasks]
    deltas = [t[5] for t in cls_tasks]
    colors = [t[6] for t in cls_tasks]
    x = np.arange(len(cls_tasks))
    w = 0.28

    ax1.bar(x - w/2, bls, w, color=NM_GREY, alpha=0.45, edgecolor='white', linewidth=0.3, label='Baseline')
    bars = ax1.bar(x + w/2, accs, w, color=colors, edgecolor='white', linewidth=0.3, label='GeoRNA')

    for i, (a, d, c) in enumerate(zip(accs, deltas, colors)):
        ax1.text(i + w/2, a + 1.5, f'+{d:.0f}pp', ha='center', fontsize=7, fontweight='bold', color=c)

    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace('\n', ' ') for n in names], fontsize=6.5, rotation=30, ha='right')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Classification: model vs baseline', fontweight='bold')
    ax1.legend(frameon=False, fontsize=6)
    ax1.set_ylim(0, 100)
    label_panel(ax1, 'a')

    # Panel b: AUC bars — all classification tasks (incl. DMS binary)
    metric_vals = []
    metric_names = []
    metric_colors = []
    # Manually order: nucleotide, base-pair, DMS binary, protein binding
    for t in TASKS:
        if t[3] is not None and t[0] != 'DMS\nreactivity':
            metric_vals.append(t[3])
            metric_names.append(t[0].replace('\n', ''))
            metric_colors.append(t[6])

    x2 = np.arange(len(metric_vals))
    ax2.bar(x2, metric_vals, color=metric_colors, edgecolor='white', linewidth=0.3, width=0.45)
    ax2.axhline(y=0.5, color=NM_GREY, ls='--', lw=0.7, alpha=0.5, label='Random AUC=0.5')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_names, fontsize=6.5, rotation=30, ha='right')
    ax2.set_ylabel('AUC')
    ax2.set_title('Classification: AUC', fontweight='bold')
    ax2.legend(frameon=False, fontsize=6)
    ax2.set_ylim(0.4, 1.05)

    # DMS regression R² annotation
    dms = [t for t in TASKS if t[4] is not None][0]
    ax2.text(3.5, 0.55,
             f'DMS regression\nR² = {dms[4]:.3f}\nr = {DMS_EXTRA["r_pearson"]:.3f}',
             ha='center', fontsize=6.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8F0', edgecolor=NM_ORANGE, alpha=0.8, lw=0.5))
    label_panel(ax2, 'b')

    fig.suptitle('Four prediction tasks from identical backbone geometry features',
                 fontweight='bold', x=0.02, ha='left', fontsize=9.5)
    savefig(fig, 'fig_four_tasks_metrics')


def fig2_information_gradient():
    """Horizontal bar: delta-pp over baseline, with AUC annotations + explanatory text."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    data = [
        ('Nucleotide type\n4-class (A/C/G/U)',   39.6, 0.95, NM_GREEN,  'Geometry directly determines\nA-form helical constraints'),
        ('Base-pair state (DSSR)\npaired / unpaired', 16.1, 0.88, NM_BLUE,   'Base-pairing alters backbone\nconformation measurably'),
        ('DMS reactivity\nregression R²=0.33',   5.7,  None, NM_ORANGE, 'Solvent accessibility partially\ncorrelates with geometry'),
        ('Protein binding\n<6Å binary',           5.2,  0.68, NM_RED,    'Side-chain + electrostatics;\nbackbone geometry insufficient'),
    ]

    y_pos = [3, 2, 1, 0]
    labels   = [d[0] for d in data]
    deltas   = [d[1] for d in data]
    aucs     = [d[2] for d in data]
    colors   = [d[3] for d in data]
    explains  = [d[4] for d in data]

    ax.barh(y_pos, deltas, color=colors, edgecolor='white', linewidth=0.3, height=0.5, alpha=0.88)

    for i, (d, c) in enumerate(zip(deltas, colors)):
        ax.text(d + 0.5, y_pos[i], f'+{d:.0f}pp', va='center', fontweight='bold', color=c, fontsize=9)
        if aucs[i]:
            ax.text(d + 0.5, y_pos[i] - 0.22, f'AUC={aucs[i]:.2f}', va='center', fontsize=6.5, color=NM_GREY)

    for i, exp in enumerate(explains):
        ax.text(44, y_pos[i], exp, va='center', fontsize=6, color=NM_GREY, style='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Improvement over baseline (pp)', fontsize=7.5)
    ax.set_xlim(0, 58)
    ax.invert_yaxis()

    # Arrow showing information decay
    ax.annotate('Information\ndecay', xy=(53, 3.3), xytext=(53, -0.4),
                arrowprops=dict(arrowstyle='->', color=NM_GREY, lw=1.2, alpha=0.6),
                fontsize=6.5, color=NM_GREY, ha='center', va='center')

    ax.set_title('Backbone geometry information gradient: from direct conformation to indirect environment',
                 fontweight='bold', fontsize=9, loc='left')
    savefig(fig, 'fig_information_gradient')


def fig3_confidence_orthogonality():
    """GeoRNA nucleotide confidence is orthogonal to all structural properties."""
    fig, ax = plt.subplots(figsize=(7.2, 2.8))

    data = [
        ('Nucleotide\ntype',      0.95, NM_GREEN,  'confidence = accuracy'),
        ('Base-pair\nstate',      0.49, NM_BLUE,   'conf AUC = 0.49'),
        ('DMS\nreactivity',       0.50, NM_ORANGE, 'r(conf, DMS) = 0.03'),
        ('Protein\nbinding',      0.47, NM_RED,    'r(conf, bind) = −0.03'),
    ]

    x = np.arange(len(data))
    vals   = [d[1] for d in data]
    colors = [d[2] for d in data]
    labels = [f'{d[0]}\n({d[3]})' for d in data]

    ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.3, width=0.5)
    ax.axhline(y=0.5, color=NM_GREY, ls='--', lw=0.7, alpha=0.5, label='Random baseline')

    for i, (v, c) in enumerate(zip(vals, colors)):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontweight='bold', color=c, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel('AUC / Correlation', fontsize=7.5)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title('GeoRNA confidence vs structural properties: fully orthogonal',
                 fontweight='bold', fontsize=9, loc='left')
    savefig(fig, 'fig_confidence_orthogonal')


def fig4_dms_detail():
    """DMS prediction detail: scatter plot + per-chain r distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))

    # Load data
    y_true = np.load(BASE / 'fair_eval_v3' / 'dms_full_test_ytrue.npy')
    y_pred = np.load(BASE / 'fair_eval_v3' / 'dms_full_test_ypred.npy')

    # Panel a: scatter (subsample)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_true), min(20000, len(y_true)), replace=False)
    ax1.scatter(y_true[idx], y_pred[idx], s=1.5, alpha=0.25, color=NM_ORANGE, rasterized=True)
    ax1.plot([0, 1], [0, 1], '--', color=NM_DARK, lw=0.7, alpha=0.5)
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    r  = np.corrcoef(y_true, y_pred)[0, 1]
    ax1.set_xlabel('True DMS')
    ax1.set_ylabel('Predicted DMS')
    ax1.set_title(f'R² = {r2:.3f}    r = {r:.3f}', fontweight='bold', loc='left', fontsize=7.5)
    label_panel(ax1, 'a')

    # Panel b: per-chain r distribution
    chains = []
    with open(BASE / 'fair_eval' / 'chain_metadata.csv') as f:
        for row in csv.DictReader(f):
            if row['split_group'] == 'post2025' and int(row['length']) < 500:
                chains.append((row['pdb_head'], int(row['length'])))

    from scipy.stats import pearsonr
    crs = []; off = 0
    for cn, L in chains:
        dms_path = BASE / 'fair_eval_v3' / 'dms_cache' / f'{cn}_dms.npy'
        if not dms_path.exists():
            continue
        Lc = min(L, len(np.load(dms_path).ravel()))
        if Lc >= 5 and off + Lc <= len(y_true):
            crs.append(pearsonr(y_true[off:off+Lc], y_pred[off:off+Lc])[0])
        off += Lc
    crs = np.array(crs)

    ax2.hist(crs, bins=40, color=NM_GREY, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax2.axvline(x=crs.mean(), color=NM_ORANGE, lw=1.2, label=f'Mean = {crs.mean():.3f}')
    ax2.axvline(x=0, color=NM_GREY, ls='--', lw=0.7, alpha=0.5)
    ax2.set_xlabel('Per-chain Pearson r')
    ax2.set_ylabel('Chain count')
    ax2.set_title(f'Median = {np.median(crs):.3f}    r>0.5: {(crs>0.5).sum()}/{len(crs)} ({(crs>0.5).sum()/len(crs)*100:.0f}%)',
                  fontweight='bold', loc='left', fontsize=7.5)
    ax2.legend(frameon=False, fontsize=6)
    label_panel(ax2, 'b')

    fig.suptitle('Full-dataset DMS reactivity prediction from backbone geometry',
                 fontweight='bold', x=0.02, ha='left', fontsize=9.5)
    savefig(fig, 'fig_dms_detail')


if __name__ == '__main__':
    print("Generating four-task figures (Nature Methods style)...")
    fig1_four_tasks_metrics()
    fig2_information_gradient()
    fig3_confidence_orthogonality()
    fig4_dms_detail()
    print(f"Done. Saved to {OUT_DIR}/")
