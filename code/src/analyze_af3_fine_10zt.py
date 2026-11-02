"""Fine analysis extension for the 10ZT series (A1/D_S0, G1, R2, R1, F1).

Reuses the analyze_af3_fine pipeline over the auxiliary job directories
under /tmp/opencode/, appending results to fine_compare.json so that the
figure scripts can treat all four structures uniformly.
"""
import json, glob
import numpy as np
from collections import defaultdict
from pathlib import Path

STRESS = Path('/home/glt/rnaxgb/RNKDB/stress_test')
OUT = STRESS / 'af3results'
fine = json.load(open(OUT / 'fine_compare.json'))

WC_TESTS = {
    ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')],
    ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],
    ('A', 'U'): [('N1', 'N3'), ('N6', 'O4')],
    ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')],
    ('G', 'U'): [('N2', 'O2'), ('O6', 'N3')],
    ('U', 'G'): [('O2', 'N2'), ('N3', 'O6')],
}
CFG = {'ranges': [(2, 68), (486, 514)], 'npy': '10ZT_H_A.npy', 'deposit': '10ZT'}

JOBS = {
    '10zt_real_scaffold_519nt': '/tmp/opencode/af3zt',          # A1 (native, seed 1249034137)
    '10zt_g1_24mut_scaffold_519nt': '/tmp/opencode/af3g1',
    '10zt_r2_29mut_scaffold_519nt': '/tmp/opencode/af3r2',
    '10zt_r1_37mut_scaffold_519nt': '/tmp/opencode/af3rn',
    '10zt_forge_scaffold_519nt': '/tmp/opencode/af3f1',
}


def cif_atoms(path, deposit=False):
    lines = open(path).read().split('\n')
    cols = {}
    for l in lines:
        if l.startswith('_atom_site.'):
            cols[l.split('.')[-1].strip()] = len(cols)
    res = defaultdict(dict)
    for f in [l.split() for l in lines if l.startswith('ATOM')]:
        key = (f[cols['auth_asym_id']], int(f[cols['auth_seq_id']])) if deposit else (f[cols['label_asym_id']], int(f[cols['label_seq_id']]))
        res[key]['_comp'] = f[cols['label_comp_id']]
        res[key][f[cols['label_atom_id']].strip('"')] = (float(f[cols['Cartn_x']]), float(f[cols['Cartn_y']]), float(f[cols['Cartn_z']]))
    return res


def wc(res, keys):
    cands = []
    for a in range(len(keys)):
        for b in range(a + 3, len(keys)):
            i, j = keys[a], keys[b]
            tests = WC_TESTS.get((res[i]['_comp'], res[j]['_comp']), [])
            best = 99
            for (t1, t2) in tests:
                if t1 in res[i] and t2 in res[j]:
                    d = float(np.linalg.norm(np.array(res[i][t1]) - np.array(res[j][t2])))
                    best = min(best, d)
            if best < 4.2:
                cands.append((best, a, b))
    cands.sort()
    used = set()
    out = set()
    for d, a, b in cands:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        out.add((a, b))
    return out


def contacts(cu):
    out = set()
    n = cu.shape[0]
    for i in range(n):
        for j in range(i + 3, n):
            if np.min(np.linalg.norm(cu[i, :, None, :] - cu[j][None, :, :], axis=2)) < 7.5:
                out.add((i, j))
    return out


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    Pc, Qc = P - pc, Q - qc
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T, pc, qc


def main():
    n = 96
    seq_ids = []
    for (a, b) in CFG['ranges']:
        seq_ids.extend(range(a, b + 1))
    nat = np.load(STRESS / CFG['npy'])
    natc = contacts(nat)
    natcif = cif_atoms(f"/tmp/opencode/{CFG['deposit']}.cif", deposit=True)
    natkeys = [('A', s) for s in seq_ids]
    snat = wc(natcif, natkeys)
    snat_1b = {(a + 1, b + 1) for (a, b) in snat}

    for job, jobdir in JOBS.items():
        if job in fine:
            print('skip', job)
            continue
        req = json.load(open(glob.glob(jobdir + '/*job_request.json')[0]))[0]
        seqfull = req['sequences'][0]['rnaSequence']['sequence']
        af3keys = [('A', s) for s in seq_ids]
        per_model = []
        for m in range(5):
            res = cif_atoms(glob.glob(jobdir + f'/*model_{m}.cif')[0])
            cu = np.zeros((n, 7, 3))
            for di, pos in enumerate(af3keys):
                A = res[pos]
                atoms = ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']
                atoms.append('N9' if seqfull[pos[1] - 1] in 'AG' else 'N1')
                for i, nm in enumerate(atoms):
                    cu[di, i] = A[nm]
            R, pc, qc = kabsch(cu[:, 0, :], nat[:, 0, :])
            rot = (cu - pc) @ R.T + qc
            res_rmsd = np.sqrt(((rot[:, 0] - nat[:, 0]) ** 2).sum(1))
            rmsdP = float(res_rmsd.mean())
            sA = wc(res, af3keys)
            shifted = 0
            used = set()
            for (a, b) in sorted(snat_1b):
                best = None
                for (a2, b2) in sorted(sA):
                    if (a2, b2) in used:
                        continue
                    if abs(a2 - a) <= 2 and abs(b2 - b) <= 2:
                        best = (a2, b2)
                        break
                if best:
                    used.add(best)
                    shifted += 1
            afc = contacts(rot)
            extra = afc - natc
            lr = sum(1 for (i, j) in extra if abs(i - j) > 12)
            per_model.append({
                'model': m, 'P_RMSD': round(rmsdP, 3),
                'RMSD7': round(float(np.sqrt(((rot - nat) ** 2).sum(2)).mean()), 3),
                'contactRec': round(len(natc & afc) / len(natc), 3),
                'n_native_pairs': len(snat), 'n_af3_pairs': len(sA),
                'SS_F1': round(2 * len(sA & snat) / (len(sA) + len(snat)) if sA or snat else 0, 3),
                'shifted_match': shifted, 'n_extra_contacts': len(extra), 'n_extra_longrange': lr,
                'per_residue_rmsd': [round(float(x), 2) for x in res_rmsd],
                'worst5': [(int(i + 1), round(float(res_rmsd[i]), 2)) for i in np.argsort(-res_rmsd)[:5]],
            })
        best = min(per_model, key=lambda x: x['P_RMSD'])
        fine[job] = {'seqlen': n, 'n_job_chains': len(req['sequences']), 'best_model': best['model'],
                     'best': best, 'all_models': per_model,
                     'native_pairs_1b': sorted([list(p) for p in snat_1b])}
        print(f'{job:>34}: best m{best["model"]} P-RMSD {best["P_RMSD"]} | SS F1 {best["SS_F1"]} | rec {best["contactRec"]} | shifted {best["shifted_match"]}/{len(snat)}')
    json.dump(fine, open(OUT / 'fine_compare.json', 'w'), indent=1)
    print('merged -> fine_compare.json')


if __name__ == '__main__':
    main()
