"""
Cross-evaluate our XGBoost models on RiboDiffusion's 8 test splits.

Maps RiboDiffusion chain IDs (1f27_A) to our pdb_head format (1F27_H_A),
extracts features from fair_eval_v3 .npy files, and evaluates all 3 models.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/cross_eval_ribodiffusion.py
"""
import os, json, pickle, gc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import defaultdict

FAIR_EVAL = "/home/glt/rnaxgb/RNKDB/fair_eval"
FAIR_EVAL_V3 = "/home/glt/rnaxgb/RNKDB/fair_eval_v3"
RIBODIFF_DIR = "/home/glt/rnaxgb/RiboDiffusion/split_data"
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


def rd_to_pdb_head(rd_chain):
    """Map RiboDiffusion chain ID to our pdb_head format.
    RiboDiffusion: {pdb_id}_{chain} (e.g., 1f27_A)
    Our format: {PDB_ID}_{H|L|X}_{CHAIN} (e.g., 1F27_H_A)
    """
    parts = rd_chain.split('_', 1)
    if len(parts) != 2:
        return None
    pdb_id, chain = parts
    pdb_upper = pdb_id.upper()
    letters = [c for c in chain if c.isalpha()]
    if letters:
        if all(c.isupper() for c in letters):
            prefix = 'H'
        elif all(c.islower() for c in letters):
            prefix = 'L'
        else:
            prefix = 'X'
    else:
        prefix = 'X'
    return f'{pdb_upper}_{prefix}_{chain}'


def build_global_chain_index():
    """Build pdb_head -> (X_path, y_path, offset, nres) for ALL splits.
    Sources: train (pre2025_train), val (pre2025_val), pre2025_test, post2025_test.
    """
    chain_index = {}
    # Pre-2025 splits from CSV (CSV order matches .npy)
    for csv_name, feat_prefix in [
        ("cluster_split_train.csv", "train"),
        ("cluster_split_val.csv", "val"),
        ("cluster_split_test.csv", "pre2025_test"),
    ]:
        df = pd.read_csv(os.path.join(FAIR_EVAL, csv_name))
        offset = 0
        for _, row in df.iterrows():
            pdb_head = row['pdb_head']
            nres = int(row['n_residues'])
            chain_index[pdb_head] = (
                os.path.join(FAIR_EVAL_V3, f"{feat_prefix}_X.npy"),
                os.path.join(FAIR_EVAL_V3, f"{feat_prefix}_y.npy"),
                offset, nres)
            offset += nres

    # Post-2025 from chain_metadata (metadata order matches .npy)
    meta = pd.read_csv(os.path.join(FAIR_EVAL, "chain_metadata.csv"))
    post = meta[meta['split_group'] == 'post2025']
    offset = 0
    for _, row in post.iterrows():
        pdb_head = row['pdb_head']
        nres = int(row['length'])
        chain_index[pdb_head] = (
            os.path.join(FAIR_EVAL_V3, "post2025_test_X.npy"),
            os.path.join(FAIR_EVAL_V3, "post2025_test_y.npy"),
            offset, nres)
        offset += nres

    return chain_index


def load_ribodiffusion_splits():
    """Load all 8 RiboDiffusion test splits.
    Returns dict: split_name -> list of (rd_chain, pdb_head) pairs.
    """
    splits = {}
    for ts in sorted(os.listdir(RIBODIFF_DIR)):
        if not ts.endswith('.tsv'):
            continue
        name = ts.replace('.tsv', '')
        df = pd.read_csv(os.path.join(RIBODIFF_DIR, ts), sep='\t')
        test = df[df['SET'] == 'TEST']['PDB_ID'].tolist()
        splits[name] = test
    return splits


def extract_for_chains(rd_chains, chain_index):
    """Extract X, y for a list of RiboDiffusion chain IDs.
    Returns X, y, and a list of (rd_chain, pdb_head, nres) for chains found.
    """
    by_file = defaultdict(list)  # (fx, fy) -> list of (pdb_head, offset, nres)

    for rd_chain in rd_chains:
        pdb_head = rd_to_pdb_head(rd_chain)
        if pdb_head and pdb_head in chain_index:
            fx, fy, offset, nres = chain_index[pdb_head]
            by_file[(fx, fy)].append((rd_chain, pdb_head, offset, nres))

    X_list, y_list = [], []
    x_cache, y_cache = {}, {}
    chain_info = []

    for (fx, fy), entries in sorted(by_file.items()):
        if fx not in x_cache:
            x_cache[fx] = np.load(fx, mmap_mode='r')
            y_cache[fy] = np.load(fy, mmap_mode='r')
        for rd_chain, pdb_head, offset, nres in entries:
            # Use .copy() to load into memory; astype for consistency
            Xc = x_cache[fx][offset:offset + nres].copy().astype(np.float32)
            yc = y_cache[fy][offset:offset + nres].copy()
            X_list.append(Xc)
            y_list.append(yc)
            chain_info.append((rd_chain, pdb_head, nres))

    if not X_list:
        return None, None, []

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, chain_info


def evaluate_model(model, X, y):
    pred = model.predict(X)
    acc = float(accuracy_score(y, pred))
    pc = {}
    for lb, b in IDX_TO_BASE.items():
        mk = y == lb
        if mk.sum():
            pc[b] = float(accuracy_score(y[mk], pred[mk]))
    return acc, pc


def main():
    # Build global chain index
    print("Building global chain index...")
    chain_index = build_global_chain_index()
    print(f"  Total chains: {len(chain_index)}")

    # Load RiboDiffusion splits
    print("Loading RiboDiffusion splits...")
    rd_splits = load_ribodiffusion_splits()
    for name, chains in rd_splits.items():
        print(f"  {name}: {len(chains)} test chains")

    # Get our training chains (for leakage analysis)
    train_csv = pd.read_csv(os.path.join(FAIR_EVAL, "cluster_split_train.csv"))
    our_train = set(train_csv['pdb_head'])
    print(f"  Our training chains: {len(our_train)}")

    # Load all 3 models
    print("Loading models...")
    models = {}
    models['existing_pre2025'] = pickle.load(
        open(os.path.join(FAIR_EVAL_V3, "model.pkl"), "rb"))
    models['das_split_retrained'] = pickle.load(
        open(os.path.join(FAIR_EVAL_V3, "model_das_split.pkl"), "rb"))
    models['das_cluster_excluded'] = pickle.load(
        open(os.path.join(FAIR_EVAL_V3, "model_das_cluster_excluded.pkl"), "rb"))
    print(f"  Loaded {len(models)} models")

    # Evaluate on each RiboDiffusion split
    all_results = {}
    print("\n" + "=" * 80)
    print("XGBoost on RiboDiffusion Test Splits")
    print("=" * 80)

    for split_name, rd_chains in rd_splits.items():
        print(f"\n── {split_name} ──")

        X, y, chain_info = extract_for_chains(rd_chains, chain_index)
        if X is None:
            print(f"  No chains found!")
            continue

        # Leakage analysis
        our_chains_in_split = {ci[1] for ci in chain_info}
        leaked = our_chains_in_split & our_train
        clean = our_chains_in_split - our_train

        n_total = X.shape[0]
        print(f"  Found: {len(chain_info)} chains, {n_total:,} nt")
        print(f"  Leaked (in our train): {len(leaked)} chains")
        print(f"  Clean (not in train): {len(clean)} chains")

        fold_results = {
            'split_name': split_name,
            'n_chains_total': len(chain_info),
            'n_chains_leaked': len(leaked),
            'n_chains_clean': len(clean),
            'n_nt': int(n_total),
            'models': {},
        }

        for model_name, model in models.items():
            acc, pc = evaluate_model(model, X, y)

            # Evaluate on clean subset only
            if clean:
                clean_mask = np.zeros(n_total, dtype=bool)
                offset = 0
                for rd_c, pdb_h, nres in chain_info:
                    if pdb_h in clean:
                        clean_mask[offset:offset + nres] = True
                    offset += nres
                X_clean = X[clean_mask]
                y_clean = y[clean_mask]
                acc_clean, pc_clean = evaluate_model(model, X_clean, y_clean)
            else:
                acc_clean, pc_clean = None, {}

            fold_results['models'][model_name] = {
                'acc_all': float(acc),
                'acc_clean': float(acc_clean) if acc_clean is not None else None,
                'per_class_all': {k: float(v) for k, v in pc.items()},
                'per_class_clean': {k: float(v) for k, v in pc_clean.items()} if acc_clean is not None else None,
            }

        all_results[split_name] = fold_results

        # Print summary
        print(f"  {'Model':<25s} {'All':>6s} {'Clean':>6s}")
        print(f"  {'-'*38}")
        for mn, mr in fold_results['models'].items():
            acc_str = f"{mr['acc_all']:.4f}" if mr['acc_all'] else "N/A"
            clean_str = f"{mr['acc_clean']:.4f}" if mr['acc_clean'] else "N/A"
            print(f"  {mn:<25s} {acc_str:>6s} {clean_str:>6s}")

        del X, y
        gc.collect()

    # Aggregate across folds
    print(f"\n{'='*80}")
    print("AGGREGATE ACROSS ALL 8 FOLDS")
    print(f"{'='*80}")

    # Summarize per model
    print(f"\n── Overall (mean ± std across folds) ──")
    print(f"  {'Model':<25s} {'All acc':>14s} {'Clean acc':>14s}")
    print(f"  {'-'*55}")
    for model_name in models:
        all_vals = [all_results[s]['models'][model_name]['acc_all']
                    for s in all_results]
        clean_vals = [all_results[s]['models'][model_name]['acc_clean']
                      for s in all_results
                      if all_results[s]['models'][model_name]['acc_clean'] is not None]
        all_mean = np.mean(all_vals)
        all_std = np.std(all_vals)
        print(f"  {model_name:<25s} {all_mean:.4f}±{all_std:.4f}", end="")
        if clean_vals:
            clean_mean = np.mean(clean_vals)
            clean_std = np.std(clean_vals)
            print(f" {clean_mean:.4f}±{clean_std:.4f}")
        else:
            print()

    # Global: concatenate all clean chains from all folds (dedup by pdb_head)
    print(f"\n── Global (all clean chains across folds, deduplicated) ──")
    seen_clean = set()
    X_clean_all, y_clean_all = [], []
    X_leaked_all, y_leaked_all = [], []

    x_cache, y_cache = {}, {}
    for split_name, fold_r in all_results.items():
        _, _, chain_info = extract_for_chains(rd_splits[split_name], chain_index)

        by_file = defaultdict(list)
        for rd_chain, pdb_head, nres in chain_info:
            ci = chain_index[pdb_head]  # (fx, fy, offset, nres)
            fx, fy, offset = ci[0], ci[1], ci[2]
            by_file[(fx, fy)].append((pdb_head, offset, nres))

        for (fx, fy), entries in by_file.items():
            if fx not in x_cache:
                x_cache[fx] = np.load(fx, mmap_mode='r')
                y_cache[fy] = np.load(fy, mmap_mode='r')
            for pdb_head, offset, nres in entries:
                if pdb_head in seen_clean:
                    continue
                seen_clean.add(pdb_head)
                Xc = x_cache[fx][offset:offset + nres].copy().astype(np.float32)
                yc = y_cache[fy][offset:offset + nres].copy()
                if pdb_head in our_train:
                    X_leaked_all.append(Xc)
                    y_leaked_all.append(yc)
                else:
                    X_clean_all.append(Xc)
                    y_clean_all.append(yc)

    print(f"  Total unique chains: {len(seen_clean)}")
    print(f"  Leaked (in our train): {len(y_leaked_all)}")
    print(f"  Clean: {len(y_clean_all)}")

    if X_clean_all:
        Xc = np.concatenate(X_clean_all)
        yc = np.concatenate(y_clean_all)
        print(f"  Clean nt: {Xc.shape[0]:,}")
        for model_name in models:
            acc, pc = evaluate_model(models[model_name], Xc, yc)
            print(f"  {model_name} clean: {acc:.4f}, pc={pc}")

    if X_leaked_all:
        Xl = np.concatenate(X_leaked_all)
        yl = np.concatenate(y_leaked_all)
        print(f"  Leaked nt: {Xl.shape[0]:,}")
        for model_name in models:
            acc, pc = evaluate_model(models[model_name], Xl, yl)
            print(f"  {model_name} leaked: {acc:.4f}, pc={pc}")

    # Save results
    output_path = os.path.join(FAIR_EVAL_V3, "cross_eval_ribodiffusion.json")
    # Clean up for JSON
    results_out = {
        'per_fold': {
            sn: {
                'n_chains_total': fr['n_chains_total'],
                'n_chains_leaked': fr['n_chains_leaked'],
                'n_chains_clean': fr['n_chains_clean'],
                'n_nt': fr['n_nt'],
                'models': fr['models'],
            }
            for sn, fr in all_results.items()
        },
    }
    # Add global
    results_out['global_unique_chains'] = len(seen_clean)
    results_out['global_leaked'] = len(y_leaked_all)
    results_out['global_clean'] = len(y_clean_all)

    with open(output_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
