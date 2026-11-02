"""PyMOL renderings of the AF3 predictions superposed on cryo-EM structures.

For each batch job, the best AF3 model (by design P-RMSD) and the native
deposited structure are drawn as nucleic-acid cartoons of the designed
segment; native is grey, AF3 is colored by residue index. Transparent
background, ray-traced PNG for later matplotlib composition (PDF output).

Run from RNKDB/:
    python ../src/pymol_af3_alignment.py
"""
import glob
import numpy as np
from pathlib import Path

import pymol
from pymol import cmd

pymol.finish_launching(['pymol', '-cq'])

STRESS = Path('/home/glt/rnaxgb/RNKDB/stress_test')
OUT = STRESS / 'af3results' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
fine = __import__('json').load(open(STRESS / 'af3results' / 'fine_compare.json'))

CFG = {
    '10zu': {'ranges': [(1, 67), (485, 513)], 'deposit': '10ZU', 'npy': '10ZU_H_A.npy'},
    '11eh': {'ranges': [(2, 68), (486, 514)], 'deposit': '11EH', 'npy': '11EH_H_A.npy'},
    '11ag': {'ranges': [(1, 67), (485, 518)], 'deposit': '11AG', 'npy': '11AG_H_A.npy'},
}

CFG['10zt'] = {'ranges': [(2, 68), (486, 514)], 'deposit': '10ZT', 'npy': '10ZT_H_A.npy'}
ZT_JOBS = ['10zt_real_scaffold_519nt', '10zt_g1_24mut_scaffold_519nt',
           '10zt_r2_29mut_scaffold_519nt', '10zt_r1_37mut_scaffold_519nt',
           '10zt_forge_scaffold_519nt']
ZT_DIR = {'10zt_real_scaffold_519nt': 'af3zt', '10zt_g1_24mut_scaffold_519nt': 'af3g1',
          '10zt_r2_29mut_scaffold_519nt': 'af3r2', '10zt_r1_37mut_scaffold_519nt': 'af3rn',
          '10zt_forge_scaffold_519nt': 'af3f1'}

VARIANTS = ['real', 'rnetopt', 'forge']
LABEL = {'real': 'REAL', 'rnetopt': 'RNet-opt', 'forge': 'FORGE'}


def resi_expr(ranges, n):
    return '(resi ' + ' or '.join(f'{a}-{b}' for a, b in ranges) + ')'


def resi_sel():
    return 'chain A and resi 1-67 and (index of first part) or chain A and resi 485-513'


def hex_for_t(i, n, cmap='viridis_r'):
    from matplotlib import cm
    return '%02X%02X%02X' % tuple(int(255 * c) for c in cm.get_cmap(cmap)(i / max(n - 1, 1))[:3])


from collections import defaultdict


def cif_atoms(path, deposit=False):
    lines = open(path).read().split('\n')
    cols = {}
    for l in lines:
        if l.startswith('_atom_site.'):
            cols[l.split('.')[-1].strip()] = len(cols)
    res = defaultdict(dict)
    for f in [l.split() for l in lines if l.startswith('ATOM')]:
        _at = f[cols['label_atom_id']].strip('"')
        key = (f[cols['auth_asym_id']], int(f[cols['auth_seq_id']])) if deposit else (f[cols['label_asym_id']], int(f[cols['label_seq_id']]))
        res[key][_at] = (float(f[cols['Cartn_x']]), float(f[cols['Cartn_y']]), float(f[cols['Cartn_z']]))
    return res


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    Pc, Qc = P - pc, Q - qc
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, pc, qc


def transform_matrix(R, pc, qc):
    t = qc - R @ pc
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return [float(v) for row in M for v in row]


def render_one(job, key, variant):
    cfg = CFG[key]
    ranges = cfg['ranges']
    bm = fine[job]['best_model']
    if key == '10zt':
        jobdir = f'/tmp/opencode/{ZT_DIR[job]}'
    else:
        jobdir = f'/tmp/opencode/af3batch/{job}'
    af3_cif = glob.glob(f'{jobdir}/*model_{bm}.cif')[0]
    nat_cif = f'/tmp/opencode/{cfg["deposit"]}.cif'

    # Kabsch from P atoms of the design segments
    af3r = cif_atoms(af3_cif, deposit=False)
    natr = cif_atoms(nat_cif, deposit=True)
    rngs = []
    for (a, b) in ranges:
        rngs.extend(range(a, b + 1))
    P_af3 = np.array([af3r[('A', s)]['P'] for s in rngs])
    P_nat = np.array([natr[('A', s)]['P'] for s in rngs])
    R, pc, qc = kabsch(P_af3, P_nat)
    M = transform_matrix(R, pc, qc)

    cmd.reinitialize()
    cmd.load(af3_cif, 'af3')
    cmd.load(nat_cif, 'nat')
    cmd.transform_object('af3', M, state=0, homogenous=1)

    sel_expr = 'chain A and (' + ' or '.join(f'resi {a}-{b}' for a, b in ranges) + ')'
    cmd.select('af3seg', f'af3 and {sel_expr}')
    cmd.select('natseg', f'nat and {sel_expr}')
    rngs = []
    for (a, b) in ranges:
        rngs.extend(range(a, b + 1))

    cmd.hide('everything', 'all')
    cmd.show('cartoon', 'af3seg')
    cmd.show('cartoon', 'natseg')
    cmd.set('cartoon_trace_atoms', 0)
    cmd.set('cartoon_side_chain_helper', 0)
    cmd.set('cartoon_flat_cycles', 1)
    cmd.set('cartoon_ring_mode', 0)
    cmd.set('cartoon_transparency', 0.30, 'natseg')
    cmd.color('grey35', 'natseg')
    cmd.orient('natseg')

    cmd.color('0x0F4D92', 'af3seg')
    cmd.set('cartoon_transparency', 0.0, 'af3seg')

    cmd.set('ray_opaque_background', 0)
    cmd.set('ray_shadows', 0)
    cmd.set('antialias', 1)
    cmd.bg_color('white')
    cmd.ray(640, 640)
    cmd.png(str(OUT / f'pymol_{job}.png'))
    cmd.delete('all')


def main():
    for key in ['10zt', '10zu', '11eh', '11ag']:
        if key == '10zt':
            for job in ZT_JOBS:
                render_one(job, key, 'zt')
                print('rendered', job)
        else:
            for var in VARIANTS:
                job = f'{key}_{var}_scaffold_5{18 if key in ("10zu", "11ag") else 19}nt'
                render_one(job, key, var)
                print('rendered', job)


if __name__ == '__main__':
    main()
