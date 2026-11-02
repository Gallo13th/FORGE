"""Fine analysis for the FORGE->RNet AF3 batch (2026-09-04).

Appends 8 jobs (scaffold + free variants for 10ZT/10ZU/11EH/11AG) to
fine_compare.json with the standard metric suite, so all subsequent figure
scripts see the FORGE->RNet series alongside REAL and FORGE columns.
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
CFG = {
    '10zt': {'ranges': [(2, 68), (486, 514)], 'npy': '10ZT_H_A.npy', 'deposit': '10ZT'},
    '10zu': {'ranges': [(1, 67), (485, 513)], 'npy': '10ZU_H_A.npy', 'deposit': '10ZU'},
    '11eh': {'ranges': [(2, 68), (486, 514)], 'npy': '11EH_H_A.npy', 'deposit': '11EH'},
    '11ag': {'ranges': [(1, 67), (485, 518)], 'npy': '11AG_H_A.npy', 'deposit': '11AG'},
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
    new_jobs = sorted([d.name for d in Path('/tmp/opencode/af3fr').iterdir() if d.is_dir()])
    for job in new_jobs:
        if job in fine:
            continue
        key = next(k for k in CFG if k in job)
        cfg = CFG[key]
        ranges = cfg['ranges']
        n = int(json.load(open(Path(f'/tmp/opencode/af3fr/{job}') / [f for f in (Path(f'/tmp/opencode/af3fr/{job}')).iterdir() if 'job_request' in f.name][0].name))[0]['sequences'][0]['rnaSequence']['sequence'].__len__()) if False else None
        req = json.load(open(glob.glob(f'/tmp/opencode/af3fr/{job}/*job_request.json')[0]))[0]
        seqfull = req['sequences'][0]['rnaSequence']['sequence']
        n = sum(b - a + 1 for (a, b) in ranges)
        scaffold_ids = []
        for (a, b) in ranges:
            scaffold_ids.extend(range(a, b + 1))
        if 'free_' in job:
            seq_ids = list(range(1, n + 1))
        else:
            seq_ids = scaffold_ids
        af3keys = [('A', s) for s in seq_ids]
        nat = np.load(STRESS / cfg['npy'])
        natc = contacts(nat)
        natcif = cif_atoms(f'/tmp/opencode/{cfg["deposit"]}.cif', deposit=True)
        natkeys = [('A', s) for s in scaffold_ids]
        snat = wc(natcif, natkeys)
        snat_1b = {(a + 1, b + 1) for (a, b) in snat}
        per_model = []
        for m in range(5):
            res = cif_atoms(glob.glob(f'/tmp/opencode/af3fr/{job}/*model_{m}.cif')[0])
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
                match = None
                for (a2, b2) in sorted(sA):
                    if (a2, b2) in used:
                        continue
                    if abs(a2 - a) <= 2 and abs(b2 - b) <= 2:
                        match = (a2, b2)
                        break
                if match:
                    used.add(match)
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
        print(f'{job:>40}: best m{best["model"]} P-RMSD {best["P_RMSD"]} | SS F1 {best["SS_F1"]} | rec {best["contactRec"]} | shifted {best["shifted_match"]}/{len(snat)}')
    json.dump(fine, open(OUT / 'fine_compare.json', 'w'), indent=1)
    print('saved')


if __name__ == '__main__':
    main()
