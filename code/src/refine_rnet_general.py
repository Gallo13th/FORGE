"""Generic Das-style RNet-guided refinement for the remaining stress-test
pseudoknot constructs (10ZU, 11EH, 11AG).

For each target:
  - target secondary structure: WC base pairs from the deposited cryo-EM
    model (same geometric detector as the AF3 SS comparisons; for 11AG the
    inter-molecular P4 stem of the dimer is included in the paired flags)
  - starting sequence: native design sequence
  - locked positions: FORGE high-confidence (geo_conf > 0.7)
  - objective: RNet DMS+2A3 reactivity profile agreement with paired/unpaired
    target classification (0 = paired -> low reactivity, 1 = unpaired -> high)
  - greedy search: single-nucleotide swaps on unlocked positions plus
    compensating WC pair swaps, up to 3 changes/round, 8 rounds

Outputs (into RNKDB/stress_test/af3results/):
  rnet_opt_{pdb}.json, rnet_opt_{pdb}.fasta

Run from RNKDB/:
    python ../src/refine_rnet_general.py
"""

import json, os, subprocess
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining

BASE_DIR = Path(os.path.dirname(__file__)).parent
STRESS = BASE_DIR / 'RNKDB' / 'stress_test'
OUT_DIR = STRESS / 'af3results'
OUT_DIR.mkdir(exist_ok=True)
PERPOS = json.load(open(STRESS / 'per_position_data.json'))

WC_TESTS = {
    ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')],
    ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],
    ('A', 'U'): [('N1', 'N3'), ('N6', 'O4')],
    ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')],
    ('G', 'U'): [('N2', 'O2'), ('O6', 'N3')],
    ('U', 'G'): [('O2', 'N2'), ('N3', 'O6')],
}
LETTERS = 'ACGU'
WC_PAIRS_S = {'GC', 'CG', 'AU', 'UA', 'GU', 'UG'}

# auth residue ranges of the designed segment per PDB (verified from the
# deposited models): chain -> list of (start, end)
TARGETS = {
    '10ZT': {'chains': ['A'], 'design_auth': {'A': [(2, 68), (486, 514)]}},
    '10ZU': {'chains': ['A'], 'design_auth': {'A': [(1, 67), (485, 513)]}},
    '11EH': {'chains': ['A'], 'design_auth': {'A': [(2, 68), (486, 514)]}},
    '11AG': {'chains': ['A', 'C'], 'design_auth': {'A': [(1, 67), (485, 518)], 'C': [(1, 67), (485, 518)]}},
}


def cif_atoms(path):
    lines = open(path).read().split('\n')
    cols = {}
    for l in lines:
        if l.startswith('_atom_site.'):
            cols[l.split('.')[-1].strip()] = len(cols)
    rows = [l.split() for l in lines if l.startswith('ATOM')]
    res = {}
    for f in rows:
        key = (f[cols['auth_asym_id']], int(f[cols['auth_seq_id']]))
        res.setdefault(key, {'comp': f[cols['label_comp_id']]})
        res[key].setdefault(f[cols['label_atom_id']].strip('"'),
                            (float(f[cols['Cartn_x']]), float(f[cols['Cartn_y']]), float(f[cols['Cartn_z']])))
    return res


def design_resids(target_cfg, cif_path):
    """Return per chain: ordered list of (auth_seq, design_idx) and design length."""
    atoms = cif_atoms(cif_path)
    chain_design = {}
    for ch in target_cfg['chains']:
        ranges = target_cfg['design_auth'][ch]
        seq_ids = []
        for (a, b) in ranges:
            seq_ids.extend(range(a, b + 1))
        chain_design[ch] = seq_ids
    return atoms, chain_design


def wc_pair_distance(res_i, res_j):
    tests = WC_TESTS.get((res_i['comp'], res_j['comp']), [])
    best = 99.0
    for (t1, t2) in tests:
        if t1 in res_i and t2 in res_j:
            d = float(np.linalg.norm(np.array(res_i[t1]) - np.array(res_j[t2])))
            best = min(best, d)
    return best


def target_pairs(cif_path, target_cfg):
    atoms, cd = design_resids(target_cfg, cif_path)
    n = len(cd[target_cfg['chains'][0]])
    # global positions: chain A first, then C
    global_ids = []
    k = 0
    for ch in target_cfg['chains']:
        for di, auth in enumerate(cd[ch]):
            global_ids.append((ch, di, auth, k))
            k += 1
    N = k
    cands = []
    for (ch1, d1, auth1, g1) in global_ids:
        for (ch2, d2, auth2, g2) in global_ids:
            if (ch2, d2, g2) == (ch1, d1, g1):
                continue
            if (ch2, d2, g2) < (ch1, d1, g1):
                continue
            if ch1 == ch2 and abs(d1 - d2) < 3:
                continue
            r1 = atoms[(ch1, auth1)]
            r2 = atoms[(ch2, auth2)]
            d = wc_pair_distance(r1, r2)
            if d < 4.2:
                cands.append((d, g1, g2))
    cands.sort()
    used = set()
    pairs = []
    for d, a, b in cands:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        pairs.append((a, b))
    intra = [(a, b) for (a, b) in pairs if (a < n and b < n) or (a >= n and b >= n)]
    pair_idx = set(pairs)
    return pairs, intra, pair_idx, [tuple(ids) for ids in cd.values()]


def score(dms, a3, t):
    return -float(np.mean(0.5 * np.abs(dms - t) + 0.5 * np.abs(a3 - t)))


def binary_agreement(dms, a3, t, thresh=0.125):
    return float(np.mean(((dms > thresh).astype(int) == t) * 0.5 + ((a3 > thresh).astype(int) == t) * 0.5))


def react(model, tokenizer, seq):
    with torch.no_grad():
        out = model(**tokenizer(seq, return_tensors='pt'))
    return out.logits_dms.squeeze().numpy(), out.logits_2a3.squeeze().numpy()


def rssim(mfe, target_set):
    st = []
    ps = set()
    for i, ch in enumerate(mfe):
        if ch == '(':
            st.append(i)
        elif ch == ')':
            if st:
                ps.add((st.pop() + 1, i + 1))
    inter = len(ps & target_set)
    union = len(ps | target_set)
    return (2 * inter / union) if union else 0.0


def refine(target, model, tokenizer, start='native'):
    cfg = TARGETS[target]
    per = PERPOS[{'10ZT': '10ZT_H_A', '10ZU': '10ZU_H_A', '11EH': '11EH_H_A', '11AG': '11AG_H_A'}[target]]
    native = ''.join(e['nt'] for e in per)
    forge_pred = ''.join(e['geo_pred'] for e in per)
    high_conf = {i + 1 for i, e in enumerate(per) if e['geo_conf'] > 0.7}
    n = len(native)
    cif = f'/tmp/opencode/{target}.cif'
    pairs, intra, pair_idx, _ = target_pairs(cif, cfg)
    t = np.ones(n)
    for (a, b) in pairs:
        if a < n:
            t[a] = 0
        elif a >= n:
            t[a - n] = 0
    # for 11AG chain A positions paired with chain C counted too (a or b beyond offset)
    t2 = np.ones(n)
    for (a, b) in pairs:
        t2[a % n] = 0
        t2[b % n] = 0
    t = t2

    print(f'== {target}: n={n} native-geom pairs={len(pairs)} (intra={len(intra)}) high-conf={sorted(high_conf)}', flush=True)
    nat_dms_sc, nat_a3_sc = react(model, tokenizer, native)
    forg_dms_sc, forg_a3_sc = react(model, tokenizer, forge_pred)
    print(f'   native consistency {score(nat_dms_sc, nat_a3_sc, t):.4f} (binary {binary_agreement(nat_dms_sc, nat_a3_sc, t):.3f}) | '
          f'FORGE {score(forg_dms_sc, forg_a3_sc, t):.4f}', flush=True)

    start_seq = forge_pred if start == 'forge' else native
    seq = list(start_seq)
    if start == 'native':
        score0 = score(nat_dms_sc, nat_a3_sc, t)
    else:
        score0 = score(forg_dms_sc, forg_a3_sc, t)
    current = score0
    best_score = current
    best_seq = list(seq)
    traj = [{'round': 0, 'score': round(current, 4), 'sequence': start_seq}]
    cand_pos = [p for p in range(1, n + 1) if p not in high_conf]
    partner = {}
    for (a, b) in intra:
        if a < n:
            partner[a] = b
            partner[b] = a
    max_rounds = int(os.environ.get('RREFINE_ROUNDS', '8'))
    for rnd in range(1, max_rounds + 1):
        improvements = []
        for p in cand_pos:
            cur = seq[p - 1]
            for nt in LETTERS:
                if nt == cur:
                    continue
                s = list(seq)
                s[p - 1] = nt
                dms, a3 = react(model, tokenizer, ''.join(s))
                sc = score(dms, a3, t)
                if sc > current + 1e-5:
                    improvements.append((sc, p, nt, None, None))
            if p in partner:
                q = partner[p]
                if q in cand_pos and q > p:
                    cur2 = seq[q - 1]
                    best_combo = None
                    for x in LETTERS:
                        for y in LETTERS:
                            if x == cur and y == cur2:
                                continue
                            if x + y in WC_PAIRS_S:
                                s = list(seq)
                                s[p - 1] = x
                                s[q - 1] = y
                                dms, a3 = react(model, tokenizer, ''.join(s))
                                sc = score(dms, a3, t)
                                if best_combo is None or sc > best_combo[0]:
                                    best_combo = (sc, x, y)
                    if best_combo and best_combo[0] > current + 1e-5:
                        improvements.append((best_combo[0], p, best_combo[1], q, best_combo[2]))
        improvements = [x for x in improvements if x[0] is not None]
        improvements.sort(key=lambda x: -x[0])
        applied = 0
        usedp = set()
        for sc, p, nt, q, qnt in improvements:
            if applied >= 3:
                break
            if p in usedp or (q is not None and q in usedp):
                continue
            seq[p - 1] = nt
            if q is not None:
                seq[q - 1] = qnt
            usedp.add(p)
            if q is not None:
                usedp.add(q)
            applied += 1
        if applied == 0:
            print(f'   round {rnd}: convergence', flush=True)
            break
        seqstr = ''.join(seq)
        dms, a3 = react(model, tokenizer, seqstr)
        current = score(dms, a3, t)
        if current > best_score:
            best_score = current
            best_seq = list(seq)
        n_mut = sum(1 for a, b in zip(native, seqstr) if a != b)
        traj.append({'round': rnd, 'score': round(current, 4), 'n_mut_from_native': n_mut, 'sequence': seqstr})
        print(f'   round {rnd}: score {current:.4f}, {n_mut}/{n} mutated', flush=True)

    best = ''.join(best_seq)
    lock_ref = native
    assert all(best[int(p) - 1] == start_seq[int(p) - 1] for p in high_conf), 'high-conf changed!'
    # mild variant: shortest-distance trajectory point with score > -0.21 (or the best <= ceil(0.31*n) mut)
    mild = None
    limit = int(round(0.31 * n))
    for tr in traj:
        if tr.get('n_mut_from_native', 0) <= limit and tr['score'] > -0.21:
            mild = tr
    if mild is None:
        mild = traj[0]
    mild_seq = mild['sequence']
    best_dms, best_a3 = react(model, tokenizer, best)
    mild_dms, mild_a3 = react(model, tokenizer, mild_seq)
    p = subprocess.run(['RNAfold'], input=best + '\n', capture_output=True, text=True)
    folded = [l.split()[0] for l in p.stdout.split('\n') if l.strip() and l.strip()[0] in '.([']
    mfe = folded[0] if folded else ''
    result = {
        'target': target,
        'native_seq': native,
        'forge_seq': forge_pred,
        'best_seq': best,
        'mild_seq': mild_seq,
        'n_mut_best': sum(1 for a, b in zip(native, best) if a != b),
        'n_mut_mild': sum(1 for a, b in zip(native, mild_seq) if a != b),
        'score_native': round(float(score(nat_dms_sc, nat_a3_sc, t)), 4),
        'score_forge': round(float(score(forg_dms_sc, forg_a3_sc, t)), 4),
        'score_best': round(float(score(best_dms, best_a3, t)), 4),
        'binary_best': round(float(binary_agreement(best_dms, best_a3, t)), 4),
        'mfe_bracket': mfe,
        'sssim_vs_target': round(float(rssim(mfe, set(intra) | set(pairs))), 4),
        'high_conf': sorted(high_conf),
        'start': start,
        'recovery_vs_native': round(float(sum(1 for a, b in zip(native, best) if a == b) / n), 4),
        'recovery_start_vs_native': round(float(sum(1 for a, b in zip(native, start_seq) if a == b) / n), 4),
        'trajectory': traj,
    }
    json.dump(result, open(OUT_DIR / f'rnet_opt_{target}.json', 'w'), indent=2)
    return result


def main():
    import sys
    only = sys.argv[1:] if len(sys.argv) > 1 else ['10ZT', '10ZU', '11EH', '11AG']
    start = 'forge'
    tokenizer = RnaTokenizer.from_pretrained('multimolecule/ribonanzanet')
    model = RibonanzaNetForPreTraining.from_pretrained('multimolecule/ribonanzanet')
    model.eval()
    results = {}
    for tg in only:
        results[tg] = refine(tg, model, tokenizer, start=start)
        outjson = OUT_DIR / f'rnet_opt_from_forge_{tg}.json'
        json.dump(results[tg], open(outjson, 'w'), indent=2)
        r = results[tg]
        print(f'{tg}: start={r["start"]} | FORGE-baseline recovery {r["recovery_start_vs_native"]:.3f} '
              f'-> after RNet {r["recovery_vs_native"]:.3f} | muts {r["n_mut_best"]}')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'trajectory'} for k, v in results.items()}, indent=1))


if __name__ == '__main__':
    main()
