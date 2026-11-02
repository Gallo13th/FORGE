"""Figures for the AF3 forward-folding vs cryo-EM comparison.

Reads fine_compare.json (batch) + hard-coded 10ZT series numbers (computed
earlier from /tmp/opencode/af3zt, af3d10zt, af3g1, af3f1, af3rn, af3r2).

Figures:
  fig1_af3_alignment.png  - native vs best-model 2D-projection superpositions
                            (12 panels: 4 structures x {REAL, RNETOPT, FORGE})
  fig2_af3_stats.png      - P-RMSD / SS F1 / contact recovery / shifted pairs
  fig3_af3_perresidue.png - per-residue P-RMSD profiles (best model)

Run from RNKDB/
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

STRESS = Path('/home/glt/rnaxgb/RNKDB/stress_test')
OUT = STRESS / 'af3results'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
fine = json.load(open(OUT / 'fine_compare.json'))

ORDER = {'10zu': 96, '11eh': 96, '11ag': 101}
JOB_CFG = {'10zu': {'ranges': [(1, 67), (485, 513)], 'npy': '10ZU_H_A.npy', 'pdb': '10ZU'},
           '11eh': {'ranges': [(2, 68), (486, 514)], 'npy': '11EH_H_A.npy', 'pdb': '11EH'},
           '11ag': {'ranges': [(1, 67), (485, 518)], 'npy': '11AG_H_A.npy', 'pdb': '11AG'}}

# 10ZT series (computed in prior work): pdb, variant, min, max, ss_f1
EXTRA = [
    # P-RMSD min, P-RMSD max, SS F1 min, SS F1 max, label
    (6.59, 7.66, 0.987, 0.987, 'REAL 0mut'),
    (22.32, 24.48, 0.312, 0.406, 'G1 24mut'),
    (15.95, 23.21, 0.620, 0.648, 'R2 29mut'),
    (23.86, 33.83, 0.625, 0.667, 'R1 37mut'),
    (26.73, 27.56, 0.000, 0.030, 'FORGE 44mut'),
]


def load_best_coords(jobname):
    """Return (native_96or101x7x3, af3_rotated x7x3, seqfull, seq_ids)"""
    key = next(k for k in ORDER if k in jobname)
    c = JOB_CFG[key]
    n = ORDER[key]
    import glob
    req = json.load(open(glob.glob(f'/tmp/opencode/af3batch/{jobname}/*job_request.json')[0]))[0]
    seqfull = req['sequences'][0]['rnaSequence']['sequence']
    seq_ids = []
    for (a, b) in c['ranges']:
        seq_ids.extend(range(a, b + 1))
    af3keys = [('A', s) for s in seq_ids]

    def cif_atoms(path):
        lines = open(path).read().split('\n')
        cols = {}
        for l in lines:
            if l.startswith('_atom_site.'):
                cols[l.split('.')[-1].strip()] = len(cols)
        res = defaultdict(dict)
        for f in [l.split() for l in lines if l.startswith('ATOM')]:
            keyk = (f[cols['label_asym_id']], int(f[cols['label_seq_id']]))
            res[keyk][f[cols['label_atom_id']].strip('"')] = (
                float(f[cols['Cartn_x']]), float(f[cols['Cartn_y']]), float(f[cols['Cartn_z']]))
        return res

    bm = fine[jobname]['best_model']
    res = cif_atoms(glob.glob(f'/tmp/opencode/af3batch/{jobname}/*model_{bm}.cif')[0])
    cu = np.zeros((n, 7, 3))
    for di, pos in enumerate(af3keys):
        A = res[pos]
        atoms = ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']
        atoms.append('N9' if seqfull[pos[1] - 1] in 'AG' else 'N1')
        for i, nm in enumerate(atoms):
            cu[di, i] = A[nm]
    nat = np.load(STRESS / c['npy'])
    # kabsch align on P atoms
    P, Q = cu[:, 0, :], nat[:, 0, :]
    pc, qc = P.mean(0), Q.mean(0)
    Pc, Qc = P - pc, Q - qc
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    rot = (cu - pc) @ R.T + qc
    return nat, rot


def projection_2d(nat, rot):
    """Common SVD plane for both structures."""
    comb = np.vstack([nat[:, 0, :], rot[:, 0, :]])
    comb -= comb.mean(0)
    U, S, Vt = np.linalg.svd(comb, full_matrices=False)
    plane = Vt[:2]
    p_n = nat[:, 0, :] @ plane.T
    p_a = rot[:, 0, :] @ plane.T
    return p_n, p_a


# ---------------- Fig 1: alignment projections ----------------
variants = ['real', 'rnetopt', 'forge']
labels = ['REAL seq', 'RNet-optimized', 'FORGE seq']
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i, key in enumerate(['10zu', '11eh', '11ag']):
    for j, var in enumerate(variants):
        job = f'{key}_{var}_scaffold_5{18 if key == "10zu" or key == "11ag" else 19}nt'
        nat, rot = load_best_coords(job)
        p_n, p_a = projection_2d(nat, rot)
        ax = axes[i][j]
        ax.plot(p_n[:, 0], p_n[:, 1], '-', color='0.3', lw=0.9, label='cryo-EM')
        sc = ax.scatter(p_a[:, 0], p_a[:, 1], c=np.arange(p_a.shape[0]), cmap='viridis', s=8, zorder=3)
        ax.plot(p_a[:, 0], p_a[:, 1], '-', color='crimson', lw=0.6, alpha=0.6)
        bm = fine[job]['best_model']
        ax.set_title(f'{key.upper()} {labels[j]} | P-RMSD {fine[job]["best"]["P_RMSD"]:.1f} A (m{bm})', fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(FIG / 'fig1_af3_alignment.png', dpi=200)
plt.close()

# ---------------- Fig 2: stats ----------------
keys = ['10zu', '11eh', '11ag']
names = {'10zu': '10ZU\nMPNN-fixbb', '11eh': '11EH\nStruct2Seq-SHAPE', '11ag': '11AG\nMPNN-RFdiff'}
colors = {'real': '#4C72B0', 'rnetopt': '#55A868', 'forge': '#C44E52'}
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
# top row: P-RMSD & SS F1 per structure (include 10ZT EXTRA in separate right panel)
for j, var in enumerate(variants):
    rms_lo, rms_hi, ss_lo, ss_hi, rec_lo, rec_hi, sh = [], [], [], [], [], [], []
    for key in keys:
        job = f'{key}_{var}_scaffold_5{18 if key == "10zu" or key == "11ag" else 19}nt'
        mods = fine[job]['all_models']
        rms_lo.append(min(m['P_RMSD'] for m in mods)); rms_hi.append(max(m['P_RMSD'] for m in mods))
        ss_lo.append(min(m['SS_F1'] for m in mods)); ss_hi.append(max(m['SS_F1'] for m in mods))
        rec_lo.append(min(m['contactRec'] for m in mods)); rec_hi.append(max(m['contactRec'] for m in mods))
        sh.append(np.mean([m['shifted_match'] for m in mods]))
    x = np.arange(3)
    ax = axes[0][j]
    ax.bar(x, rms_lo, color=colors[var], alpha=0.85, width=0.5)
    ax.bar(x, [(hi - lo) for lo, hi in zip(rms_lo, rms_hi)], bottom=rms_lo, color=colors[var], alpha=0.45, width=0.5)
    ax.set_xticks(x); ax.set_xticklabels([names[k] for k in keys])
    ax.set_title(f'{labels[j]}', fontsize=11)
    if j == 0: ax.set_ylabel('design P-RMSD (A)')
    ax2 = ax.twinx()
    ax2.plot(x, ss_lo, 's-', color='k', ms=4, lw=1, label='SS F1 min')
    ax2.plot(x, ss_hi, '^:', color='k', ms=4, lw=1, label='SS F1 max')
    ax2.set_ylim(0, 1.05); ax2.set_ylabel('SS F1')
    if j == 0: ax2.legend(loc='lower left', fontsize=7)
# bottom row: contact recovery + shifted pairs, then 10ZT panel
for j, var in enumerate(variants):
    rec_lo, rec_hi = [], []
    for key in keys:
        job = f'{key}_{var}_scaffold_5{18 if key == "10zu" or key == "11ag" else 19}nt'
        mods = fine[job]['all_models']
        rec_lo.append(min(m['contactRec'] for m in mods)); rec_hi.append(max(m['contactRec'] for m in mods))
    x = np.arange(3)
    ax = axes[1][j]
    ax.bar(x, rec_lo, color=colors[var], alpha=0.85, width=0.5)
    ax.bar(x, [(hi - lo) for lo, hi in zip(rec_lo, rec_hi)], bottom=rec_lo, color=colors[var], alpha=0.45, width=0.5)
    ax.set_xticks(x); ax.set_xticklabels([names[k] for k in keys])
    ax.set_ylim(0, 1)
    if j == 0: ax.set_ylabel('native contact recovery')
    ax.set_title(f'{labels[j]} contact recovery', fontsize=10)
ax = axes[1][2]
x = np.arange(len(EXTRA))
rms = [e[0] for e in EXTRA]; rms_u = [e[1] for e in EXTRA]
ss = [e[2] for e in EXTRA]; ss_u = [e[3] for e in EXTRA]
ax.bar(x, rms, color='#8172B3', alpha=0.9, width=0.55)
ax.bar(x, [(u - l) for l, u in zip(rms, rms_u)], bottom=rms, color='#8172B3', alpha=0.4, width=0.55)
ax2 = ax.twinx()
ax2.plot(x, ss, 'sk', ms=4)
ax2.plot(x, ss_u, '^:', color='k', ms=4, lw=1)
ax2.set_ylim(-0.05, 1.05)
ax.set_xticks(x); ax.set_xticklabels([e[4] for e in EXTRA], rotation=40, fontsize=7)
ax.set_ylabel('P-RMSD (A)'); ax2.set_ylabel('SS F1')
ax.set_title('10ZT mutation series', fontsize=10)
plt.tight_layout()
plt.savefig(FIG / 'fig2_af3_stats.png', dpi=200)
plt.close()

# ---------------- Fig 3: per-residue profiles ----------------
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for i, key in enumerate(keys):
    for j, var in enumerate(variants):
        job = f'{key}_{var}_scaffold_5{18 if key == "10zu" or key == "11ag" else 19}nt'
        bm = fine[job]['best_model']
        rms = fine[job]['best']['per_residue_rmsd']
        ax = axes[i][j]
        ax.plot(np.arange(1, len(rms) + 1), rms, color='tab:blue', lw=0.9)
        ax.axhline(np.mean(rms), lw=0.7, ls='--', color='tab:red', label=f'mean {np.mean(rms):.1f} A')
        ax.set_ylim(0, max(rms) + 2)
        ax.set_title(f'{key.upper()} {labels[j]} (model {bm})', fontsize=9)
        ax.set_xlabel('position'); ax.set_ylabel('P-RMSD (A)')
        ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(FIG / 'fig3_af3_perresidue.png', dpi=200)
plt.close()
print('saved', list(FIG.glob('*.png')))
