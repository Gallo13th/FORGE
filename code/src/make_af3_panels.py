"""Split all AF3 comparison panels into individual figures (panels/).

Panels:
  align_{job}.(pdf|png|svg)          - PyMOL cartoon alignments (from pymol_*.png)
  stats_{name}.(pdf|png|svg)         - 2 waterfall bars + SS F1 + contact recovery
  perres_{job}.(pdf|png|svg)         - per-residue P-RMSD profiles (12)
  dms_{key}_{variant}.(pdf|png|svg)  - target SS vs RNet DMS prediction (4x3)

Run from RNKDB/
"""
import json, sys, shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SUBMISSION_STYLE = Path('/home/glt/rnaxgb/Nature-Methods-submission/figures')
sys.path.insert(0, str(SUBMISSION_STYLE))
from figure_style import apply_style, save_figure, label_panel, despine, COLORS

apply_style()
STRESS = Path('/home/glt/rnaxgb/RNKDB/stress_test')
OUT = STRESS / 'af3results'
FIG = OUT / 'figures'
PANELS = FIG / 'panels'
PANELS.mkdir(exist_ok=True)
fine = json.load(open(OUT / 'fine_compare.json'))

ORDER = ['10zt', '10zu', '11eh', '11ag']
NAME = {'10zt': '10ZT (gRNAde)', '10zu': '10ZU (MPNN-fixbb)', '11eh': '11EH (Struct2Seq-SHAPE)', '11ag': '11AG (MPNN-RFdiff)'}
VARIANTS = ['real', 'rnetopt', 'forge']
VARLABEL = {'real': 'real sequence', 'rnetopt': 'RNet-optimized', 'forge': 'FORGE sequence'}
KEYS = {'10zt': '10ZT', '10zu': '10ZU', '11eh': '11EH', '11ag': '11AG'}
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


def legend_inside(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    ok = True
    legs = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)
    for leg in legs:
        bb = leg.get_window_extent(r)
        if not (fb.x0 - 1 <= bb.x0 and bb.x1 <= fb.x1 + 1 and fb.y0 - 1 <= bb.y0 and bb.y1 <= fb.y1 + 1):
            ok = False
            print('LEGEND OUT:', bb)
    return ok


# ---------------- 1. alignment panels (copy PyMOL PNGs) ----------------
for k in ORDER:
    for v in VARIANTS:
        src = FIG / f'pymol_{jobname(k, v)}.png'
        if src.exists():
            shutil.copy(src, PANELS / f'align_{jobname(k, v)}.png')
print('alignment panels:', len(list(PANELS.glob('align_*.png'))))

# ---------------- 2. stats panels ----------------
hatch_imp, hatch_dec = '////', '\\\\\\\\'
col_imp, col_dec = COLORS['correct'], COLORS['error']
x = np.arange(len(ORDER))


def waterfall(target_var, stem):
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
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
        ax.plot([i - 0.25, i + 0.25], [r1, r1], color='0.35', lw=0.9)
        (lo, hi) = rng(ORDER[i], 'real', 'P_RMSD')
        ax.errorbar(i, real_vals[i], yerr=[[real_vals[i] - lo], [hi - real_vals[i]]],
                    fmt='none', ecolor='0.62', elinewidth=0.8, capsize=2, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[k] for k in ORDER], fontsize=6, rotation=0)
    ax.set_ylabel('P-RMSD to cryo-EM (A)')
    ax.set_title(f'{VARLABEL[target_var]} vs real sequence')
    handles = [
        mpatches.Patch(facecolor=COLORS['blue'], label='real-sequence P-RMSD'),
        mpatches.Patch(facecolor=col_imp, hatch=hatch_imp, edgecolor='white', label='improvement'),
        mpatches.Patch(facecolor=col_dec, hatch=hatch_dec, edgecolor='white', label='degradation'),
        mpatches.Patch(facecolor='none', edgecolor='0.35', label='target value (line) / 5-model range (grey bar)'),
    ]
    fig.subplots_adjust(bottom=0.30)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=6)
    assert legend_inside(fig)
    save_figure(fig, PANELS, stem, svg=True, png=True)


waterfall('rnetopt', 'stats_rmsd_rnetopt_vs_real')
waterfall('forge', 'stats_rmsd_forge_vs_real')

for field, ylab in zip(['SS_F1', 'contactRec'], ['SS F1', 'native contact recovery']):
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    for v, col in zip(VARIANTS, [COLORS['blue'], COLORS['correct'], COLORS['error']]):
        vals = [best_val(ORDER[i], v)[field] for i in range(len(ORDER))]
        off = 0.25 * (VARIANTS.index(v) - 1)
        ax.bar(x + off, vals, width=0.22, color=col, label=VARLABEL[v])
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[k] for k in ORDER], fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylab)
    fig.subplots_adjust(bottom=0.30)
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=3, fontsize=6)
    assert legend_inside(fig)
    save_figure(fig, PANELS, f'stats_{field}', svg=True, png=True)

# ---------------- 3. per-residue panels ----------------
colors = {v: c for v, c in zip(VARIANTS, [COLORS['blue'], COLORS['correct'], COLORS['error']])}
for k in ORDER:
    for v in VARIANTS:
        job = jobname(k, v)
        bm = fine[job]['best_model']
        rms = fine[job]['best']['per_residue_rmsd']
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        ax.plot(np.arange(1, len(rms) + 1), rms, color=colors[v], lw=0.9)
        ax.axhline(np.mean(rms), lw=0.7, ls='--', color='k', label=f'mean {np.mean(rms):.1f} A')
        ax.set_ylim(0, max(rms) * 1.12)
        ax.set_xlabel('residue')
        ax.set_ylabel('P-RMSD (A)')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_title(f'{NAME[k]} | {VARLABEL[v]} (model {bm})', fontsize=7.5)
        save_figure(fig, PANELS, f'perres_{jobname(k, v)}', svg=True, png=True)

# ---------------- 4. DMS vs target panels ----------------
import torch
from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining

perpos = json.load(open(STRESS / 'per_position_data.json'))
rnet_opt = {}
for k in ['10zt', '10zu', '11eh', '11ag']:
    if k == '10zt':
        rnet_opt[k] = json.load(open(OUT / 'rnet_refined_10zt.json'))['rnet_refined_seq']
    else:
        rnet_opt[k] = json.load(open(OUT / f'rnet_opt_{k.upper()}.json'))['best_seq']

SEQ_BY = {}
for k in ORDER:
    key = key = KEYS[k]
    real = json.load(open(OUT / f'{OUT.name}_null')) if False else None

seed = {}
for k in ORDER:
    ke = KEYS[k]
    realseq = json.load(open(OUT / f'rnet_opt_{ke}.json'))['native_seq'] if k != '10zt' else 'UGCCACGGGGAGCGAUAGCUCCGGGUGAAAACCCUGGGGGCCGUGGCUAGGCUGGAGGCCCCCGGCAUGCCGUACACGAAAGUGUACCCUCCAGCC'
    forge = ''.join(e['geo_pred'] for e in perpos[ke + '_H_A'])
    seed[k] = {'real': realseq, 'rnetopt': rnet_opt[k], 'forge': forge}

tok = RnaTokenizer.from_pretrained('multimolecule/ribonanzanet')
model = RibonanzaNetForPreTraining.from_pretrained('multimolecule/ribonanzanet')
model.eval()


def dms_of(seq):
    with torch.no_grad():
        out = model(**tok(seq, return_tensors='pt'))
    return out.logits_dms.squeeze().numpy()


for k in ORDER:
    keyi = KEYS[k]
    npairs = fine[jobname(k, 'real')]['native_pairs_1b']
    n = fine[jobname(k, 'real')]['seqlen']
    for v in VARIANTS:
        seq = seed[k][v]
        dms = dms_of(seq)
        tgt = np.ones(n)
        for (a, b) in npairs:
            tgt[a - 1] = 0
            tgt[b - 1] = 0
        fig, ax = plt.subplots(figsize=(5.2, 2.9))
        ax.bar(np.arange(1, n + 1), dms, color=COLORS['blue'], alpha=0.75, width=0.85, label='RNet DMS prediction')
        ax.plot(np.arange(1, n + 1), tgt, color='k', ls='--', lw=0.8, label='target SS (1=unpaired, 0=paired)')
        for (a, b) in npairs:
            ax.axvspan(a - 0.5, a + 0.5, color='0.88', zorder=0)
            ax.axvspan(b - 0.5, b + 0.5, color='0.88', zorder=0)
        ax.axhline(0.125, color='k', lw=0.5, ls=':')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, n + 1)
        ax.set_xlabel('residue')
        ax.set_ylabel('DMS')
        fig.subplots_adjust(bottom=0.30)
        fig.legend(handles=[
            mpatches.Patch(color=COLORS['blue'], label='RNet DMS prediction'),
            plt.Line2D([], [], color='k', ls='--', lw=0.8, label='target SS (binary)'),
        ], loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=6)
        ax.set_title(f'{NAME[k]} | {VARLABEL[v]}', fontsize=7.5)
        save_figure(fig, PANELS, f'dms_{k}_{v}', svg=True, png=True)

print('panel files:', len(list(PANELS.iterdir())))
