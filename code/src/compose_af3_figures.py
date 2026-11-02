"""Compose the AF3 comparison figures in the manuscript style (v2).

Fig 1: PyMOL cartoon alignments (native grey translucent, AF3 solid blue).
Fig 2: waterfall/delta bars (real sequence -> RNet-opt / FORGE) with hatched
       improvement/degradation segments + explicit legends; plus SS F1 and
       contact recovery panels. Layout uses constrained_layout (matplotlib
       resolves conflicts by computation); legend placement is verified to
       stay within the figure bounding box.
Fig 3: per-residue P-RMSD profiles.

Run from RNKDB/
"""
import json
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox

SUBMISSION_STYLE = Path('/home/glt/rnaxgb/Nature-Methods-submission/figures')
sys.path.insert(0, str(SUBMISSION_STYLE))
from figure_style import apply_style, save_figure, label_panel, despine, COLORS

apply_style()

STRESS = Path('/home/glt/rnaxgb/RNKDB/stress_test')
OUT = STRESS / 'af3results'
FIG = OUT / 'figures'
fine = json.load(open(OUT / 'fine_compare.json'))

ORDER = ['10zt', '10zu', '11eh', '11ag']
NAME = {'10zt': '10ZT (gRNAde)', '10zu': '10ZU (MPNN-fixbb)', '11eh': '11EH (Struct2Seq-SHAPE)', '11ag': '11AG (MPNN-RFdiff)'}
VARIANTS = ['real', 'rnetopt', 'forge']
VARLABEL = {'real': 'real sequence', 'rnetopt': 'RNet-optimized', 'forge': 'FORGE sequence'}
JOB = {}
for k in ORDER:
    if k == '10zt':
        JOB['10zt_real'] = '10zt_real_scaffold_519nt'
        JOB['10zt_rnetopt'] = '10zt_r1_37mut_scaffold_519nt'
        JOB['10zt_forge'] = '10zt_forge_scaffold_519nt'
    else:
        for v in VARIANTS:
            JOB[f'{k}_{v}'] = f'{k}_{v}_scaffold_5{18 if k in ("10zu", "11ag") else 19}nt'


def jobname(k, v):
    return JOB[f'{k}_{v}']


def best_val(k, v):
    return fine[jobname(k, v)]['best']


def rng(k, v, field):
    mods = fine[jobname(k, v)]['all_models']
    return min(m[field] for m in mods), max(m[field] for m in mods)


def legend_inside_figure(fig, texts=()):
    """Verify all legends/text artist bboxes are inside the figure box."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    ok = True
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(renderer)
        inside = (bb.x0 >= fb.x0 - 1 and bb.y0 >= fb.y0 - 1
                  and bb.x1 <= fb.x1 + 1 and bb.y1 <= fb.y1 + 1)
        if not inside:
            ok = False
            print(f'LEGEND OUT OF FIGURE: {bb}')
    return ok


# ---------------- Fig 1 ----------------
fig, axes = plt.subplots(4, 4, figsize=(9.4, 10.6),
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.30, 'width_ratios': [1, 1, 1, 0.14]})
for i, key in enumerate(ORDER):
    for j, var in enumerate(VARIANTS):
        job = jobname(key, var)
        ax = axes[i][j]
        ax.imshow(imread(FIG / f'pymol_{job}.png'))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.7); s.set_edgecolor('#AAAAAA')
        bm = best_val(key, var)
        ax.set_title(f'{NAME[key]} | {VARLABEL[var]}\nP-RMSD {bm["P_RMSD"]:.1f} A  (model {bm["model"]})', fontsize=7)
        label_panel(ax, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'][i * 3 + j], x=-0.03, y=1.05)
for ax in axes[:, 3]:
    ax.set_axis_off()
axleg = axes[1][3]
axleg.legend(handles=[mpatches.Patch(color=COLORS['blue'], label='AF3 (cartoon)'),
                      mpatches.Patch(color='#A0A0A0', alpha=0.55, label='cryo-EM (cartoon, translucent)')],
             frameon=False, fontsize=7, loc='center')
save_figure(fig, FIG, 'fig1_af3_alignment')

# ---------------- Fig 2 ----------------
fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6), constrained_layout=True)
hatch_imp, hatch_dec = '////', '\\\\\\\\'
col_imp, col_dec = COLORS['correct'], COLORS['error']
x = np.arange(len(ORDER))

def waterfall(ax, target_var, title):
    real_vals = [best_val(k, 'real')['P_RMSD'] for k in ORDER]
    tgt_vals = [best_val(k, target_var)['P_RMSD'] for k in ORDER]
    ax.bar(x, real_vals, color=COLORS['blue'], width=0.5, label='real sequence (best model)')
    changes = []
    for i, (r0, r1) in enumerate(zip(real_vals, tgt_vals)):
        delta = r1 - r0
        changes.append(delta)
        if delta > 0:
            ax.bar(i, delta, bottom=r0, color=col_dec, width=0.5, hatch=hatch_dec, edgecolor='white', linewidth=0)
        else:
            ax.bar(i, -delta, bottom=r1, color=col_imp, width=0.5, hatch=hatch_imp, edgecolor='white', linewidth=0)
        ax.plot([i - 0.25, i + 0.25], [r1, r1], color='k', lw=1.0)
        (lo, hi) = rng(ORDER[i], 'real', 'P_RMSD')
        ax.plot([i + 0.25, i + 0.25], [lo, hi], color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([k.upper() for k in ORDER])
    ax.set_ylabel('P-RMSD to cryo-EM (A)')
    ax.set_title(title)
    autolabels = ['%+.1f' % d for d in changes]
    for i, t in enumerate(autolabels):
        ax.text(i, max(real_vals[i], tgt_vals[i]) + 1.2, t, ha='center', fontsize=6.5,
                color='k' if abs(changes[i]) < 3 else (col_imp if changes[i] < 0 else col_dec))
    handles = [
        mpatches.Patch(facecolor=COLORS['blue'], label='real-sequence P-RMSD'),
        mpatches.Patch(facecolor=col_imp, hatch=hatch_imp, edgecolor='white', label=f'{VARLABEL[target_var]} improvement'),
        mpatches.Patch(facecolor=col_dec, hatch=hatch_dec, edgecolor='white', label=f'{VARLABEL[target_var]} degradation'),
        mpatches.Patch(facecolor='white', edgecolor='k', label=f'{VARLABEL[target_var]} (line = value)'),
    ]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6.5)

waterfall(axes[0][0], 'rnetopt', 'RNet-optimized sequence vs real sequence')
waterfall(axes[0][1], 'forge', 'FORGE sequence vs real sequence')

for j, (field, ylab) in enumerate(zip(['SS_F1', 'contactRec'], ['SS F1', 'native contact recovery'])):
    ax = axes[1][j]
    for v, col in zip(VARIANTS, [COLORS['blue'], COLORS['correct'], COLORS['error']]):
        vals = [best_val(k, v)[field] for k in ORDER]
        ax.bar(x + [-0.25 + 0.25 * VARIANTS.index(v)], vals, width=0.22, color=col,
               label=VARLABEL[v])
    ax.set_xticks(x); ax.set_xticklabels([k.upper() for k in ORDER])
    ax.set_ylim(0, 1.05); ax.set_ylabel(ylab)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=6.5)
    ax.set_title('Secondary-structure recovery' if field == 'SS_F1' else 'Tertiary contact recovery')
for axr in axes.ravel():
    despine(axr)
label_panel(axes[0][0], 'a'); label_panel(axes[0][1], 'b')
label_panel(axes[1][0], 'c'); label_panel(axes[1][1], 'd')
assert legend_inside_figure(fig), 'legend layout conflict detected'
save_figure(fig, FIG, 'fig2_af3_stats')

# ---------------- Fig 3 ----------------
fig, axes = plt.subplots(4, 3, figsize=(8.6, 8.4), constrained_layout=True)
colors = {v: c for v, c in zip(VARIANTS, [COLORS['blue'], COLORS['correct'], COLORS['error']])}
for i, key in enumerate(ORDER):
    for j, var in enumerate(VARIANTS):
        job = jobname(key, var)
        bm = fine[job]['best_model']
        rms = fine[job]['best']['per_residue_rmsd']
        ax = axes[i][j]
        ax.plot(np.arange(1, len(rms) + 1), rms, color=colors[var], lw=0.8)
        ax.axhline(np.mean(rms), lw=0.6, ls='--', color='k', label=f'mean {np.mean(rms):.1f} A')
        ax.set_ylim(0, max(rms) * 1.12)
        ax.set_title(f'{NAME[key]} | {VARLABEL[var]} (model {bm})', fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 2:
            ax.set_xlabel('residue', fontsize=6.5)
        if j == 0:
            ax.set_ylabel('P-RMSD (A)', fontsize=6.5)
        if i == 0 and j == 0:
            ax.legend(fontsize=5.5, loc='upper left')
for axr in axes.ravel():
    despine(axr)
label_panel(axes[0][0], 'a')
save_figure(fig, FIG, 'fig3_af3_perresidue')
print('figures saved')
