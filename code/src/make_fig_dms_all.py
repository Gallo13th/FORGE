"""Single combined DMS figure: 4 PDBs x 3 sequence classes.

Rows = 10ZT / 10ZU / 11EH / 11AG (PDB ID only).
Cols = real sequence / FORGE sequence / FORGE (R-Net optimized).
Each panel: RibonanzaNet DMS prediction (blue bars) vs target secondary
structure (grey bands + dashed binary profile, 0 = paired / 1 = unpaired).
Legend appears exactly once (figure bottom).

Output: fig_dms_all.(pdf|png|svg) in af3results/figures/
"""
import json
import sys
import numpy as np
from pathlib import Path

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
fine = json.load(open(OUT / 'fine_compare.json'))

PDB = ['10ZT', '10ZU', '11EH', '11AG']
NATIVE_SEQ = {
    '10ZT': 'UGCCACGGGGAGCGAUAGCUCCGGGUGAAAACCCUGGGGGCCGUGGCUAGGCUGGAGGCCCCCGGCAUGCCGUACACGAAAGUGUACCCUCCAGCC',
    '10ZU': 'GGACGCCCGGCGCUGUAGCGCCGGCCGAAAGGCCACGCCCAGGCGUCCUGGCGGGCCAGGGCGGUCGCGACGGGCCCGAAAGGGCCCAGGCCCGCC',
    '11EH': 'CCCUCUCUACACCUAUCGGUGUACCCAUAAGGGUCGCCCCAGAGAGGGCCCACUCACCGGGGCCACAUGUGCCGCCCCAAAGGGCGGAGUGAGUGG',
    '11AG': 'GCGCCGGAGGCCCUAAAGGGCCAAGCGAAAGCUUAGGGGGACCGGCGCAGGGAUAGUACCCCCCACCGGUGCGGGCCUAUAGGCCCGAACUAUCCCGGAUC',
}
REAL_JOB = {'10ZT': '10zt_real_scaffold_519nt', '10ZU': '10zu_real_scaffold_518nt',
            '11EH': '11eh_real_scaffold_519nt', '11AG': '11ag_real_scaffold_518nt'}


def main():
    import torch
    from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining
    perpos = json.load(open(STRESS / 'per_position_data.json'))

    seqs = {}
    for pdb in PDB:
        key = pdb if pdb != '10ZT' else '10ZT'
        rnet = json.load(open(OUT / f'rnet_opt_from_forge_{pdb}.json'))['best_seq']
        forge = ''.join(e['geo_pred'] for e in perpos[pdb + '_H_A'])
        seqs[pdb] = {'real': NATIVE_SEQ[pdb], 'forge': forge, 'rnet': rnet}

    tok = RnaTokenizer.from_pretrained('multimolecule/ribonanzanet')
    model = RibonanzaNetForPreTraining.from_pretrained('multimolecule/ribonanzanet')
    model.eval()
    dms = {}
    for pdb, cls in seqs.items():
        dms[pdb] = {}
        for v, seq in cls.items():
            with torch.no_grad():
                out = model(**tok(seq, return_tensors='pt'))
            dms[pdb][v] = out.logits_dms.squeeze().numpy()
            print(pdb, v, len(seq))

    cols = [('real', 'real sequence'), ('forge', 'FORGE sequence'), ('rnet', 'FORGE (R-Net optimized)')]
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 9.2), gridspec_kw={'wspace': 0.28, 'hspace': 0.42})
    for i, pdb in enumerate(PDB):
        job = REAL_JOB[pdb]
        npairs = fine[job]['native_pairs_1b']
        n = fine[job]['seqlen']
        for j, (v, vlab) in enumerate(cols):
            ax = axes[i][j]
            tgt = np.ones(n)
            for (a, b) in npairs:
                tgt[a - 1] = 0
                tgt[b - 1] = 0
            ax.bar(np.arange(1, n + 1), dms[pdb][v], color=COLORS['blue'], alpha=0.75, width=0.85)
            ax.plot(np.arange(1, n + 1), tgt, color='k', ls='--', lw=0.8)
            for (a, b) in npairs:
                ax.axvspan(a - 0.5, a + 0.5, color='0.88', zorder=0)
                ax.axvspan(b - 0.5, b + 0.5, color='0.88', zorder=0)
            ax.axhline(0.125, color='k', lw=0.5, ls=':')
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, n + 1)
            ax.set_xticks([1, n // 2, n])
            ax.tick_params(labelsize=6)
            despine(ax)
            if i == 3:
                ax.set_xlabel('residue', fontsize=6.5)
            if j == 0:
                ax.set_ylabel('DMS', fontsize=7)
            if i == 0:
                ax.set_title(vlab, fontsize=7.5)
            label_panel(ax, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'][i * 3 + j], x=-0.02, y=1.07)
        axes[i][0].set_ylabel('DMS', fontsize=7)
        axes[i][0].text(-0.52, 0.5, pdb, transform=axes[i][0].transAxes, rotation=90,
                        ha='center', va='center', fontsize=9, fontweight='bold')
    fig.subplots_adjust(bottom=0.13)
    fig.legend(handles=[
        mpatches.Patch(color=COLORS['blue'], label='RNet DMS prediction'),
        plt.Line2D([], [], color='k', ls='--', lw=0.8, label='target secondary structure (0 = paired, 1 = unpaired)'),
        plt.Line2D([], [], color='k', ls=':', lw=0.6, label='reactivity threshold 0.125'),
        mpatches.Patch(color='0.88', label='paired positions (from cryo-EM structure)'),
    ], loc='lower center', bbox_to_anchor=(0.5, 0.004), ncol=4, fontsize=7)
    save_figure(fig, FIG, 'fig_dms_all')


if __name__ == '__main__':
    main()
