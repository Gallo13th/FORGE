"""Fig 4 (v2): dotplot secondary-structure comparison + RNet reactivity.

Rows = the four AI-designed pseudoknot constructs.
Col 0: dotplot of base pairs - upper triangle: native cryo-EM pairs (black),
       lower triangle: AF3 best-model WC pairs (blue).
Col 1: RibonanzaNet DMS prediction per position with the target SS profile
       (grey bands = paired positions, dashed line = ideal profile).
Col 2: 10ZT mutation series dotplots (real / G1 24mut / R1 37mut / FORGE 44mut).

Output: fig4_af3_2d_dms (pdf/png/svg)
"""
import json
import sys
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SUBMISSION_STYLE = Path('/home/glt/rnaxgb/Nature-Methods-submission/figures')
sys.path.insert(0, str(SUBMISSION_STYLE))
from figure_style import apply_style, save_figure, label_panel, despine, COLORS

apply_style()

STRESS = Path('/home/glt/rnaxgb/RNKDB/stress_test')
OUT = STRESS / 'af3results'
FIG = OUT / 'figures'
fine = json.load(open(OUT / 'fine_compare.json'))

NAME = {'10ZT': '10ZT (gRNAde)', '10ZU': '10ZU (MPNN-fixbb)', '11EH': '11EH (Struct2Seq-SHAPE)', '11AG': '11AG (MPNN-RFdiff)'}
JOB_REAL = {'10ZT': '10zt_real_scaffold_519nt', '10ZU': '10zu_real_scaffold_518nt',
            '11EH': '11eh_real_scaffold_519nt', '11AG': '11ag_real_scaffold_518nt'}
SEQS = {
    '10ZT': 'UGCCACGGGGAGCGAUAGCUCCGGGUGAAAACCCUGGGGGCCGUGGCUAGGCUGGAGGCCCCCGGCAUGCCGUACACGAAAGUGUACCCUCCAGCC',
    '10ZU': 'GGACGCCCGGCGCUGUAGCGCCGGCCGAAAGGCCACGCCCAGGCGUCCUGGCGGGCCAGGGCGGUCGCGACGGGCCCGAAAGGGCCCAGGCCCGCC',
    '11EH': 'CCCUCUCUACACCUAUCGGUGUACCCAUAAGGGUCGCCCCAGAGAGGGCCCACUCACCGGGGCCACAUGUGCCGCCCCAAAGGGCGGAGUGAGUGG',
    '11AG': 'GCGCCGGAGGCCCUAAAGGGCCAAGCGAAAGCUUAGGGGGACCGGCGCAGGGAUAGUACCCCCCACCGGUGCGGGCCUAUAGGCCCGAACUAUCCCGGAUC',
}

CFG_RANGES = {'10ZT': [(2, 68), (486, 514)], '10ZU': [(1, 67), (485, 513)],
              '11EH': [(2, 68), (486, 514)], '11AG': [(1, 67), (485, 518)]}
JOBDIR = {'10zt_real_scaffold_519nt': '/tmp/opencode/af3zt', '10zt_g1_24mut_scaffold_519nt': '/tmp/opencode/af3g1',
          '10zt_r1_37mut_scaffold_519nt': '/tmp/opencode/af3rn', '10zt_forge_scaffold_519nt': '/tmp/opencode/af3f1'}

WC_TESTS = {
    ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')],
    ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],
    ('A', 'U'): [('N1', 'N3'), ('N6', 'O4')],
    ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')],
    ('G', 'U'): [('N2', 'O2'), ('O6', 'N3')],
    ('U', 'G'): [('O2', 'N2'), ('N3', 'O6')],
}


def cif_atoms(path):
    lines = open(path).read().split('\n')
    cols = {}
    for l in lines:
        if l.startswith('_atom_site.'):
            cols[l.split('.')[-1].strip()] = len(cols)
    res = defaultdict(dict)
    for f in [l.split() for l in lines if l.startswith('ATOM')]:
        k = (f[cols['label_asym_id']], int(f[cols['label_seq_id']]))
        res[k]['_comp'] = f[cols['label_comp_id']]
        res[k][f[cols['label_atom_id']].strip('"')] = 1
    return res


def wc(res, keys):
    cands = []
    for a in range(len(keys)):
        for b in range(a + 3, len(keys)):
            i, j = keys[a], keys[b]
            tests = WC_TESTS.get((res[i]['_comp'], res[j]['_comp']), [])
            for (t1, t2) in tests:
                if t1 in res[i] and t2 in res[j]:
                    cands.append((a, b))
                    break
    used = set()
    out = set()
    for a, b in cands:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        out.add((a, b))
    return out


def af3_pairs(key, job):
    bm = fine[job]
    jobdir = JOBDIR.get(job, f'/tmp/opencode/af3batch/{job}')
    cif = glob.glob(f'{jobdir}/*model_{bm["best_model"]}.cif')[0]
    res = cif_atoms(cif)
    ids = []
    for (a, b) in CFG_RANGES[key]:
        ids.extend(range(a, b + 1))
    keys = [('A', s) for s in ids]
    return wc(res, keys)


def dotplot(ax, key, job, npairs):
    n = fine[job]['seqlen']
    sA = af3_pairs(key, job)
    sN = set((a - 1, b - 1) for (a, b) in npairs)
    X = np.array([[b, a] for (a, b) in sN]).reshape(-1, 2)
    Y = np.array(list(sA)).reshape(-1, 2) if sA else np.zeros((0, 2))
    if len(X):
        ax.plot(X[:, 1] + 0.5, X[:, 0] + 0.5, 's', color='k', ms=1.4, mew=0)
    if len(Y):
        ax.plot(Y[:, 0] + 0.5, Y[:, 1] + 0.5, 's', color=COLORS['blue'], ms=1.4, mew=0)
    ax.set_xlim(0, n + 1)
    ax.set_ylim(n + 1, 0)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.tick_params(labelsize=5.5)
    ax.axhline(0, color='k', lw=0.3)
    ax.axvline(0, color='k', lw=0.3)


def run_dms_all():
    import torch
    from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining
    tok = RnaTokenizer.from_pretrained('multimolecule/ribonanzanet')
    model = RibonanzaNetForPreTraining.from_pretrained('multimolecule/ribonanzanet')
    model.eval()
    dms_all = {}
    for key, seq in SEQS.items():
        with torch.no_grad():
            out = model(**tok(seq, return_tensors='pt'))
        dms_all[key] = out.logits_dms.squeeze().numpy()
    return dms_all


def main():
    dms_all = run_dms_all()
    ZT_SERIES = ['10zt_real_scaffold_519nt', '10zt_g1_24mut_scaffold_519nt',
                 '10zt_r1_37mut_scaffold_519nt', '10zt_forge_scaffold_519nt']
    ZT_LBL = ['real sequence', 'G1 (24 mut, FORGE+SS)', 'R1 (37 mut, RNet-opt)', 'FORGE sequence (44 mut)']
    fig, axes = plt.subplots(4, 3, figsize=(9.2, 9.0), gridspec_kw={'hspace': 0.42, 'wspace': 0.35})
    for i, key in enumerate(['10ZU', '11EH', '11AG', '10ZT']):
        job = JOB_REAL[key]
        n = fine[job]['seqlen']
        npairs = fine[job]['native_pairs_1b']
        # col0 dotplot
        ax = axes[i][0]
        dotplot(ax, key, job, npairs)
        ax.set_xticks([1, n // 2, n]); ax.set_yticks([1, n // 2, n])
        ax.set_title(f'{NAME[key]} | real sequence\nP-RMSD {fine[job]["best"]["P_RMSD"]:.1f} A, SS F1 {fine[job]["best"]["SS_F1"]:.2f}',
                     fontsize=7)
        # col1 DMS
        ax = axes[i][1]
        tgt = np.ones(n)
        for (a, b) in npairs:
            tgt[a - 1] = 0
            tgt[b - 1] = 0
        dms = dms_all[key]
        x = np.arange(1, n + 1)
        ax.bar(x, dms, color=COLORS['blue'], alpha=0.75, width=0.85, label='RNet DMS pred.')
        ax.plot(x, tgt, 'k--', lw=0.7, label='target SS (0=paired, 1=unpaired)')
        for (a, b) in npairs:
            ax.axvspan(a - 0.5, a + 0.5, color='0.88', zorder=0)
            ax.axvspan(b - 0.5, b + 0.5, color='0.88', zorder=0)
        ax.axhline(0.125, color='k', lw=0.4, ls=':')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, n + 1)
        ax.tick_params(labelsize=5.5)
        if i == 0:
            ax.set_title('RNet DMS prediction vs target', fontsize=7)
            ax.legend(fontsize=5.5, loc='lower left', bbox_to_anchor=(0.02, 0.02), framealpha=0.85)
        ax.set_ylabel('DMS', fontsize=6.5)
        # col2 10ZT series
        ax = axes[i][2]
        job2 = ZT_SERIES[i]
        dotplot(ax, '10ZT', job2, fine['10zt_real_scaffold_519nt']['native_pairs_1b'])
        ax.set_xticks([1, 48, 96]); ax.set_yticks([1, 48, 96])
        ax.set_title(f'10ZT | {ZT_LBL[i]}\nP-RMSD {fine[job2]["best"]["P_RMSD"]:.1f} A, SS F1 {fine[job2]["best"]["SS_F1"]:.2f}', fontsize=7)
    label_panel(axes[0][0], 'a'); label_panel(axes[0][1], 'b')
    save_figure(fig, FIG, 'fig4_af3_2d_dms')


if __name__ == '__main__':
    main()
