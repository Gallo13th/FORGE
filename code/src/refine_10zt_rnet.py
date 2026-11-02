"""Das-style RNet-guided refinement of the 10ZT (gRNAde) pseudoknot design.

Refinement targets: FORGE-aware design (G1 = 23 mutations from the native
sequence, locking the 6 high-confidence positions and resampling 39
low-confidence positions from the FORGE distribution) is refined with
RibonanzaNet so that the predicted chemical reactivity profile agrees with
the target P20 secondary structure (native cryo-EM base pairs).

Score (per position): 0 = paired (low reactivity expected), 1 = unpaired.
score(seq) = -mean( 0.5*|DMS-t| + 0.5*|2A3-t| ) over all 96 positions.
Search: greedily try single-nucleotide substitutions and compensating
base-pair swaps (WC-compatible) on low-confidence positions, keeping the
6 high-confidence FORGE positions (27, 30, 31, 35, 36, 46) untouched.

Run from RNKDB/:
    python ../src/refine_10zt_rnet.py
"""

import json, os, sys, copy
import numpy as np
import torch
from pathlib import Path

from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining

BASE_DIR = Path(os.path.dirname(__file__)).parent
STRESS = BASE_DIR / 'RNKDB' / 'stress_test'
OUT_DIR = STRESS / 'af3results'
OUT_DIR.mkdir(exist_ok=True)

NATIVE_SEQ = 'UGCCACGGGGAGCGAUAGCUCCGGGUGAAAACCCUGGGGGCCGUGGCUAGGCUGGAGGCCCCCGGCAUGCCGUACACGAAAGUGUACCCUCCAGCC'
HIGH_CONF = {27, 30, 31, 35, 36, 46}  # 1-indexed FORGE positions with geo_conf > 0.7

# Native cryo-EM (10ZT, 2.9 A) WC base pairs, 1-indexed (design 96 indexing),
# derived from the deposited model with the same geometric detector as the
# AF3 secondary-structure comparisons.
NATIVE_PAIRS = [
    (2, 47), (3, 46), (4, 45), (5, 44), (6, 43), (7, 42), (8, 41),
    (9, 22), (10, 21), (11, 20), (12, 19), (13, 18),
    (23, 34), (24, 33), (25, 32), (26, 31),
    (36, 63), (37, 62), (38, 61), (39, 60), (40, 59),
    (50, 96), (51, 95), (52, 94), (53, 93), (54, 92), (55, 91), (56, 90), (57, 89), (58, 88),
    (65, 70), (66, 69),
    (72, 87), (73, 86), (74, 85), (75, 84), (76, 83), (77, 82),
]

WC_PAIRS_S = {'GC', 'CG', 'AU', 'UA', 'GU', 'UG'}
LETTERS = 'ACGU'


def react(model, tokenizer, seq, device):
    enc = tokenizer(seq, return_tensors='pt')
    with torch.no_grad():
        out = model(**enc)
    dms = out.logits_dms.squeeze().numpy()
    a3 = out.logits_2a3.squeeze().numpy()
    return dms, a3


def target_array(pairs):
    t = np.ones(96)
    for a, b in pairs:
        t[a - 1] = 0.0
        t[b - 1] = 0.0
    return t


def score(dms, a3, t):
    return -float(np.mean(0.5 * np.abs(dms - t) + 0.5 * np.abs(a3 - t)))


def binary_agreement(dms, a3, t, thresh=0.125):
    binr = (dms > thresh).astype(int)
    bin3 = (a3 > thresh).astype(int)
    return float(np.mean((binr == t) * 0.5 + (bin3 == t) * 0.5))


def rssim(mfe, t, seq_len=96):
    st = []; ps = []
    for i, ch in enumerate(mfe):
        if ch == '(':
            st.append(i)
        elif ch == ')':
            if st:
                ps.append((st.pop() + 1, i + 1))
    tset = set(t) if isinstance(t, (list, tuple)) else t
    pset = set(ps)
    inter = len(pset & tset)
    union = len(pset | tset)
    return (2 * inter / union) if union else 0.0


def main():
    tokenizer = RnaTokenizer.from_pretrained('multimolecule/ribonanzanet')
    model = RibonanzaNetForPreTraining.from_pretrained('multimolecule/ribonanzanet')
    model.eval()

    t = target_array(NATIVE_PAIRS)
    print('target SS: %d pairs, %d paired positions, %d unpaired'
          % (len(NATIVE_PAIRS), int(t[~t.astype(bool)]).__len__() if False else 76, 20))

    g1 = [c for c in json.load(open(STRESS / 'variant_candidates.json'))['10ZT_H_A'] if c['class'] == 'G1'][0]['sequence']
    assert len(g1) == 96
    print('G1 starting seq: %d mutations vs native (high-conf locked: %s)'
          % (sum(1 for a, b in zip(NATIVE_SEQ, g1) if a != b), ','.join(map(str, sorted(HIGH_CONF)))))

    native_dms, native_a3 = react(model, tokenizer, NATIVE_SEQ, None)
    g1_dms, g1_a3 = react(model, tokenizer, g1, None)
    print('native: consistency score %.4f, binary agreement %.3f' % (score(native_dms, native_a3, t), binary_agreement(native_dms, native_a3, t)))
    print('G1:     consistency score %.4f, binary agreement %.3f' % (score(g1_dms, g1_a3, t), binary_agreement(g1_dms, g1_a3, t)))

    # candidate positions: all FORGE low-confidence (non-high-conf) positions
    cand_pos = [p for p in range(1, 97) if p not in HIGH_CONF]
    partner = {}
    for a, b in NATIVE_PAIRS:
        partner[a] = b
        partner[b] = a
    print('candidate positions (low-conf): %d' % len(cand_pos))

    seq = list(g1)
    current = score(g1_dms, g1_a3, t)
    best_score = current
    best_seq = list(seq)
    traj = [{'round': 0, 'score': round(current, 4), 'binary': round(binary_agreement(g1_dms, g1_a3, t), 4),
             'n_mut_from_s0': sum(1 for a, b in zip(NATIVE_SEQ, ''.join(seq)) if a != b), 'sequence': ''.join(seq)}]

    for rnd in range(1, 9):
        improvements = []
        for p in cand_pos:
            cur = seq[p - 1]
            for nt in LETTERS:
                if nt == cur:
                    continue
                s = list(seq)
                s[p - 1] = nt
                dms, a3 = react(model, tokenizer, ''.join(s), None)
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
                                dms, a3 = react(model, tokenizer, ''.join(s), None)
                                sc = score(dms, a3, t)
                                if best_combo is None or sc > best_combo[0]:
                                    best_combo = (sc, x, y)
                    if best_combo and best_combo[0] > current + 1e-5:
                        improvements.append((best_combo[0], p, best_combo[1], q, best_combo[2]))
        improvements = [x for x in improvements if x[0] is not None]
        improvements.sort(key=lambda x: -x[0])
        applied = 0
        used = set()
        for sc, p, nt, q, qnt in improvements:
            if applied >= 3:
                break
            if p in used or (q is not None and q in used):
                continue
            seq[p - 1] = nt
            if q is not None:
                seq[q - 1] = qnt
            used.add(p)
            if q is not None:
                used.add(q)
            applied += 1
        if applied == 0:
            print('round %d: no improving mutations (score %.4f)' % (rnd, current))
            break
        seqstr = ''.join(seq)
        dms, a3 = react(model, tokenizer, seqstr, None)
        current = score(dms, a3, t)
        if current > best_score:
            best_score = current
            best_seq = list(seq)
        n_mut = sum(1 for a, b in zip(NATIVE_SEQ, seqstr) if a != b)
        changes = [(p, nt, q, qnt) for sc, p, nt, q, qnt in improvements[:applied]]
        traj.append({'round': rnd, 'score': round(current, 4), 'binary': round(binary_agreement(dms, a3, t), 4),
                     'n_mut_from_s0': n_mut, 'changes': changes, 'sequence': seqstr})
        print('round %d: applied %d changes, score %.4f, %d/96 positions differ from S0'
              % (rnd, applied, current, n_mut))

    refined = ''.join(best_seq)
    dms, a3 = react(model, tokenizer, refined, None)
    n_mut = sum(1 for a, b in zip(NATIVE_SEQ, refined) if a != b)
    assert all(refined[int(p) - 1] == NATIVE_SEQ[int(p) - 1] for p in HIGH_CONF), 'high-conf positions changed!'

    import subprocess
    p = subprocess.run(['RNAfold'], input=refined + '\n', capture_output=True, text=True)
    folded = [l for l in p.stdout.split('\n') if l.strip() and l.strip()[0] in '.([']
    mfe = folded[0] if folded else ''
    ss_ok = 'SSsim (MFE vs target): %.3f' % rssim(mfe, set(NATIVE_PAIRS))

    # mild variant: best score up to round with < 30 mutations from S0
    mild = None
    for tr in traj:
        if tr['n_mut_from_s0'] <= 30 and tr['score'] > -0.21:
            mild = tr
    if mild is None:
        mild = traj[0]
    mild_seq = mild['sequence']

    result = {
        'target': '10ZT_H_A',
        'native_seq': NATIVE_SEQ,
        'g1_seq': g1,
        'rnet_refined_seq': refined,
        'rnet_refined_mild_seq': mild_seq,
        'n_mutations_vs_native': n_mut,
        'n_mutations_vs_g1': sum(1 for a, b in zip(g1, refined) if a != b),
        'n_mutations_mild_vs_native': sum(1 for a, b in zip(NATIVE_SEQ, mild_seq) if a != b),
        'mutations': [(i + 1, a, b) for i, (a, b) in enumerate(zip(NATIVE_SEQ, refined)) if a != b],
        'score_native': round(float(score(native_dms, native_a3, t)), 4),
        'score_g1': round(float(score(g1_dms, g1_a3, t)), 4),
        'score_refined': round(float(score(dms, a3, t)), 4),
        'binary_native': round(float(binary_agreement(native_dms, native_a3, t)), 4),
        'binary_g1': round(float(binary_agreement(g1_dms, g1_a3, t)), 4),
        'binary_refined': round(float(binary_agreement(dms, a3, t)), 4),
        'trajectory': traj,
        'mfe_bracket': mfe,
        'sssim_vs_target': round(float(rssim(mfe, set(NATIVE_PAIRS))), 4),
    }
    with open(OUT_DIR / 'rnet_refined_10zt.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(ss_ok)

    with open(OUT_DIR / 'rnet_refined_10zt.fasta', 'w') as f:
        f.write('>R1_10ZT_RNetRefined_%dmut_free_96nt\n%s\n' % (n_mut, refined))
        f.write('>R2_10ZT_RNetRefined_mild_%dmut_free_96nt\n%s\n' % (sum(1 for a, b in zip(NATIVE_SEQ, mild_seq) if a != b), mild_seq))
    print('saved rnet_refined_10zt.fasta / rnet_refined_10zt.json')
    print('refined seq:', refined)


if __name__ == '__main__':
    main()
