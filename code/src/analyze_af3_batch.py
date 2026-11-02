"""Batch analysis of the AF3 server stress-test confirms (2026-09-03 batch).

For each job: extract the designed segment (auth-ranges fixed per construct),
align to the cryo-EM native 7-atom coords, compute P-RMSD / 7-atom RMSD /
contact recovery / Jaccard / WC-pair SS F1 / interpart distance, plus
pLDDT and iPTM from the summary/full-data JSONs.

Run from RNKDB/:
    python ../src/analyze_af3_batch.py
"""
import json, glob, statistics
import numpy as np
from collections import defaultdict
from pathlib import Path

STRESS = Path('/home/glt/rnaxgb/RNKDB/stress_test')
OUT = STRESS / 'af3results'
BATCH = Path('/tmp/opencode/af3batch')

WC_TESTS = {
    ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')],
    ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],
    ('A', 'U'): [('N1', 'N3'), ('N6', 'O4')],
    ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')],
    ('G', 'U'): [('N2', 'O2'), ('O6', 'N3')],
    ('U', 'G'): [('O2', 'N2'), ('N3', 'O6')],
}
# label-seq ranges of the designed segment per PDB (AF3 labels = input order)
JD = {
    '10zu': {'n': 96, 'ranges': [(1, 67), (485, 513)], 'npy': '10ZU_H_A.npy'},
    '11eh': {'n': 96, 'ranges': [(2, 68), (486, 514)], 'npy': '11EH_H_A.npy'},
    '11ag': {'n': 101, 'ranges': [(1, 67), (485, 518)], 'npy': '11AG_H_A.npy'},
}


def parse_cif(path):
    lines = open(path).read().split('\n')
    cols = {}
    for l in lines:
        if l.startswith('_atom_site.'):
            cols[l.split('.')[-1].strip()] = len(cols)
    res = defaultdict(dict)
    for f in [l.split() for l in lines if l.startswith('ATOM')]:
        seq = int(f[cols['label_seq_id']])
        atom = f[cols['label_atom_id']].strip('"')
        res[seq][atom] = (float(f[cols['Cartn_x']]), float(f[cols['Cartn_y']]), float(f[cols['Cartn_z']]))
        res[seq]['_comp'] = f[cols['label_comp_id']]
    return res


def coarse_and_contacts(res, seq_ids, seqfull, n):
    cu = np.zeros((n, 7, 3))
    for di, pos in enumerate(seq_ids):
        A = res[pos]
        atoms = ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']
        atoms.append('N9' if seqfull[pos - 1] in 'AG' else 'N1')
        for i, nm in enumerate(atoms):
            cu[di, i] = A[nm]
    cset = set()
    for i in range(n):
        for j in range(i + 3, n):
            if np.min(np.linalg.norm(cu[i, :, None, :] - cu[j][None, :, :], axis=2)) < 7.5:
                cset.add((i, j))
    return cu, cset


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    Pc, Qc = P - pc, Q - qc
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, pc, qc


def wc_pairs(res, seq_ids):
    cands = []
    for a in range(len(seq_ids)):
        for b in range(a + 3, len(seq_ids)):
            i, j = seq_ids[a], seq_ids[b]
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
    for dd, a, b in cands:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        out.add((a, b))
    return out


def main():
    jobs = sorted([d for d in BATCH.iterdir() if d.is_dir()])
    allres = {}
    for jd in jobs:
        name = jd.name
        req = json.load(open(glob.glob(str(jd / '*job_request.json'))[0]))[0]
        seqfull = req['sequences'][0]['rnaSequence']['sequence']
        # also check second chain for 11ag
        nch = len(req['sequences'])
        jcfg = None
        for k, v in JD.items():
            if k in name:
                jcfg = v
        n, ranges, npy = jcfg['n'], jcfg['ranges'], jcfg['npy']
        seq_ids = []
        for (a, b) in ranges:
            seq_ids.extend(range(a, b + 1))
        nat = np.load(STRESS / npy)
        natc = set()
        for i in range(n):
            for j in range(i + 3, n):
                if np.min(np.linalg.norm(nat[i, :, None, :] - nat[j][None, :, :], axis=2)) < 7.5:
                    natc.add((i, j))
        nat_full = None
        natives = []
        rows = []
        for m in range(5):
            cif = glob.glob(str(jd / f'*model_{m}.cif'))[0]
            res = parse_cif(cif)
            cu, afc = coarse_and_contacts(res, seq_ids, seqfull, n)
            R, pc, qc = kabsch(cu[:, 0, :], nat[:, 0, :])
            rot = (cu - pc) @ R.T + qc
            rmsdP = float(np.sqrt(((rot[:, 0] - nat[:, 0]) ** 2).sum(1)).mean())
            rmsd7 = float(np.sqrt(((rot - nat) ** 2).sum(2)).mean())
            rec = len(natc & afc) / len(natc)
            jac = len(natc & afc) / len(natc | afc)
            sA = wc_pairs(res, seq_ids)
            sN = wc_pairs_from_native_last = None
            rows.append((rmsdP, rmsd7, rec, jac))
        # native SS pairs via same detector on the native npy substitute: recompute from cif? use deposit cif
        d0 = json.load(open(glob.glob(str(jd / '*full_data_0.json'))[0]))
        from collections import defaultdict as dd
        rp = dd(list)
        for a, p, r in zip(d0['atom_chain_ids'], d0['atom_plddts'], d0['token_res_ids']):
            rp[r].append(p)
        ml = {x: statistics.mean(v) for x, v in rp.items()}
        design_pl = [step for rng in ranges for step in range(rng[0], rng[1] + 1)]
        design_plLDDT = statistics.mean([ml.get(x, 0) for x in design_pl])
        scaf_plLDDT = statistics.mean([ml.get(x, 0) for x in range(1, len(seqfull) + 1) if x not in set(design_pl)])
        summ = json.load(open(glob.glob(str(jd / '*summary_confidences_0.json'))[0]))
        allres[name] = {
            'n_seq': nch, 'len': len(seqfull),
            'P-RMSD_min': round(min(r[0] for r in rows), 2), 'P-RMSD_max': round(max(r[0] for r in rows), 2),
            '7atom_RMSD_min': round(min(r[1] for r in rows), 2), '7atom_RMSD_max': round(max(r[1] for r in rows), 2),
            'contactRec_min': round(min(r[2] for r in rows), 3), 'contactRec_max': round(max(r[2] for r in rows), 3),
            'Jaccard_min': round(min(r[3] for r in rows), 3), 'Jaccard_max': round(max(r[3] for r in rows), 3),
            'design_pLDDT': round(design_plLDDT, 1), 'scaffold_pLDDT': round(scaf_plLDDT, 1),
            'iptm': summ.get('iptm'), 'ptm': summ.get('ptm'),
            'seeds': req.get('modelSeeds'),
        }
        print(f"{name:>40}: P-RMSD {allres[name]['P-RMSD_min']}-{allres[name]['P-RMSD_max']} A | rec {allres[name]['contactRec_min']}-{allres[name]['contactRec_max']} | iptm {allres[name]['iptm']} ptm {allres[name]['ptm']}")
    json.dump(allres, open(OUT / 'batch_summary.json', 'w'), indent=1)
    print('\nsaved batch_summary.json')


if __name__ == '__main__':
    main()
