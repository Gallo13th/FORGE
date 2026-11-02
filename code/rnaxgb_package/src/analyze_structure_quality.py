"""
Structure quality assessment: correlate GeoRNA per-position confidence
with experimental B-factors from mmCIF files. Demonstrates broad utility
of GeoRNA as an RNA model quality assessment tool.

Run from RNKDB/: python ../src/analyze_structure_quality.py
"""
import os, sys, json, time, pickle, gzip, re
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path
from collections import defaultdict
from urllib.request import urlopen, Request
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
import torch
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std
import xgboost as xgb

COORDS_DIR = "./coords"
SEQ_DIR = "./seq"
MODEL_PATH = "./fair_eval_v3/model.pkl"
OUTPUT_DIR = "./quality_assessment"
META_PATH = "./fair_eval/chain_metadata.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper: parse mmCIF for B-factors ────────────────────────────────
def fetch_mmcif(pdb_id):
    """Download mmCIF from RCSB, return file content as string."""
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        req = Request(url, headers={"User-Agent": "GeoRNA/1.0"})
        with urlopen(req, timeout=30) as f:
            content = f.read().decode("utf-8")
        return content
    except Exception:
        return None

def parse_bfactors_from_cif(cif_text):
    """Extract per-atom B-factors for RNA backbone atoms from mmCIF text.

    Returns: dict mapping chain_label -> list of per-residue dicts with
             {residue_num, atom_name, bfactor}
    """
    # Parse _atom_site records
    results = defaultdict(list)

    in_site = False
    headers = {}
    for line in cif_text.split("\n"):
        line = line.strip()

        if line.startswith("_atom_site."):
            parts = line.split()
            field = parts[0].replace("_atom_site.", "")
            headers[field] = len(headers)
            in_site = True
            continue

        if in_site and (line.startswith("_") or line.startswith("loop_")):
            in_site = len(headers) == 0
            continue

        if in_site and line and not line.startswith("#") and not line.startswith("_"):
            parts = line.split()
            if len(parts) < max(headers.values()) + 1:
                continue

            def get_val(key):
                if key in headers:
                    idx = headers[key]
                    return parts[idx] if idx < len(parts) else "?"
                return "?"

            group_pdb = get_val("group_PDB")
            if group_pdb != "ATOM":
                continue

            # Only RNA backbone atoms (P, OP1, OP2, O5', C5', C4', C3', O3', N1, N9)
            atom_name = get_val("auth_atom_id") or get_val("label_atom_id")
            comp_id = get_val("auth_comp_id") or get_val("label_comp_id")
            chain_id = get_val("auth_asym_id") or get_val("label_asym_id")
            entity_id = get_val("label_entity_id")
            res_num = get_val("auth_seq_id") or get_val("label_seq_id")
            bfactor = get_val("B_iso_or_equiv")

            # RNA residues
            rna_bases = {"A", "C", "G", "U", "DA", "DC", "DG", "DT"}
            if comp_id.strip() not in rna_bases:
                continue

            try:
                bf = float(bfactor)
                res = int(res_num)
            except (ValueError, TypeError):
                continue

            results[chain_id].append({
                "residue_num": res,
                "comp_id": comp_id.strip(),
                "atom_name": atom_name,
                "bfactor": bf,
            })

    return dict(results)

def aggregate_bfactors_by_residue(chain_atoms):
    """Average B-factors per residue (mean over backbone atoms)."""
    by_res = defaultdict(list)
    for a in chain_atoms:
        by_res[a["residue_num"]].append(a["bfactor"])
    return {res: np.mean(bfs) for res, bfs in sorted(by_res.items())}


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("GeoRNA Structure Quality Assessment")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Load metadata, pick diverse post-2025 chains
    meta = pd.read_csv(META_PATH)
    post2025 = meta[meta['split_group'] == 'post2025'].copy()
    # Get unique PDB IDs
    post2025['pdb_id'] = post2025['pdb_head'].str.split('_').str[0]
    unique_pdbs = post2025['pdb_id'].drop_duplicates().tolist()

    # Sample ~100 PDBs across the alphabet for diversity
    np.random.seed(42)
    sample_pdbs = sorted(np.random.choice(unique_pdbs, size=min(100, len(unique_pdbs)), replace=False))
    # Plus the application demo PDB and a few high-profile ones
    extra_pdbs = ["10PX", "9M0P", "8ZFI", "7AM2"]
    for p in extra_pdbs:
        if p not in sample_pdbs:
            sample_pdbs.append(p)

    print(f"Sampled {len(sample_pdbs)} PDBs for quality assessment")

    # ── Phase 1: Download mmCIF and extract B-factors ──
    print("\n" + "-" * 50)
    print("Phase 1: Fetching mmCIF and extracting B-factors")
    print("-" * 50)

    bfactor_data = {}     # pdb_id -> {chain_id -> {res_num: mean_bfactor}}
    resolution_data = {}  # pdb_id -> resolution
    failed_fetch = 0

    for pdb_id in tqdm(sample_pdbs, desc="Fetching mmCIF"):
        cif_text = fetch_mmcif(pdb_id)
        if cif_text is None:
            failed_fetch += 1
            continue

        # Extract resolution
        res_match = re.search(r'_refine\.ls_d_res_high\s+(\d+\.?\d*)', cif_text)
        if not res_match:
            res_match = re.search(r'_em_3d_reconstruction\.resolution\s+(\d+\.?\d*)', cif_text)
        if res_match:
            resolution_data[pdb_id] = float(res_match.group(1))

        # Extract B-factors
        chain_atoms = parse_bfactors_from_cif(cif_text)
        for chain_id, atoms in chain_atoms.items():
            if len(atoms) < 5:
                continue
            bfactor_data[f"{pdb_id}_{chain_id}"] = aggregate_bfactors_by_residue(atoms)

    print(f"  Fetched: {len(sample_pdbs) - failed_fetch}/{len(sample_pdbs)} PDBs")
    print(f"  Chains with B-factors: {len(bfactor_data)}")
    print(f"  Structures with resolution: {len(resolution_data)}")

    # ── Phase 2: Run GeoRNA predictions with per-position confidence ──
    print("\n" + "-" * 50)
    print("Phase 2: GeoRNA per-position confidence")
    print("-" * 50)

    confidence_data = {}  # pdb_head -> {residue_offset: confidence}

    # Get post-2025 chains that have B-factors
    analysis_chains = []
    for pdb_head in post2025['pdb_head']:
        pdb_id = pdb_head.split('_')[0]
        chain_id = '_'.join(pdb_head.split('_')[1:])
        # Try matching: pdb_head format is like "100D_H_A" but B-factor keys are "A"
        bf_key_base = f"{pdb_id}_{chain_id}"
        # Also try just the last part as chain
        chain_letter = pdb_head.split('_')[-1]
        bf_key_short = f"{pdb_id}_{chain_letter}"

        if bf_key_base in bfactor_data:
            analysis_chains.append((pdb_head, bf_key_base))
        elif bf_key_short in bfactor_data:
            analysis_chains.append((pdb_head, bf_key_short))

    print(f"Chains with both predictions and B-factors: {len(analysis_chains)}")

    failed_pred = 0
    for pdb_head, bf_key in tqdm(analysis_chains, desc="Predicting"):
        coord_file = os.path.join(COORDS_DIR, f"{pdb_head}.npy")
        seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
        if not os.path.exists(coord_file) or not os.path.exists(seq_file):
            failed_pred += 1
            continue
        try:
            coords = np.load(coord_file)
            L = coords.shape[0]
            if L > 500 or L < 5:
                failed_pred += 1
                continue
            data = build_graph(torch.from_numpy(np.nan_to_num(coords[:, :7])).float())
            x = np.nan_to_num(data['x'].numpy())
            edge_attr = data['edge_attr'].numpy()
            edge_index = data['edge_index'].numpy()
            num_nodes = x.shape[0]
            if edge_attr.shape[0] > 0 and edge_index.shape[1] > 0:
                edge_mean = aggregate_features_mean(edge_attr, edge_index[1], num_nodes)
                edge_std = aggregate_features_std(edge_attr, edge_index[1], num_nodes)
            else:
                edge_mean = np.zeros((num_nodes, edge_attr.shape[1]))
                edge_std = np.zeros((num_nodes, edge_attr.shape[1]))
            features = np.concatenate([x, edge_mean, edge_std], axis=-1).astype(np.float32)

            with open(seq_file, 'r') as fh:
                seq = fh.readlines()[-1].strip()
            base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
            labels = np.array([base_map.get(s, -1) for s in seq], dtype=np.int8)
            valid = (labels != -1) & (~np.isnan(features).any(axis=1))
            if valid.sum() < 5:
                failed_pred += 1
                continue

            proba = model.predict_proba(features[valid])
            confidence = np.max(proba, axis=1)

            confidence_data[pdb_head] = {
                'bf_key': bf_key,
                'confidence': confidence.tolist(),
                'n_residues': int(valid.sum()),
            }
        except Exception:
            failed_pred += 1

    print(f"  Successful predictions: {len(confidence_data)}, failed: {failed_pred}")

    # ── Phase 3: Correlate confidence with B-factors ──
    print("\n" + "-" * 50)
    print("Phase 3: Correlation analysis")
    print("-" * 50)

    per_chain_corrs = []
    all_confidences = []
    all_bfactors = []
    per_resolution_bin = defaultdict(lambda: {'conf': [], 'bf': [], 'corrs': []})

    for pdb_head, cdata in confidence_data.items():
        bf_key = cdata['bf_key']
        if bf_key not in bfactor_data:
            continue

        bf_by_res = bfactor_data[bf_key]
        confidences = cdata['confidence']
        n_res = cdata['n_residues']

        # Align by residue index
        bf_list = sorted(bf_by_res.items())
        if len(bf_list) < 5:
            continue

        # Take min length between B-factor list and prediction
        n_common = min(n_res, len(bf_list))
        bf_values = np.array([bf for _, bf in bf_list[:n_common]])
        conf_values = np.array(confidences[:n_common])

        # Per-chain Spearman correlation
        if len(bf_values) >= 10:
            sp_r, sp_p = spearmanr(conf_values, bf_values)
            pr_r, pr_p = pearsonr(conf_values, bf_values)
            per_chain_corrs.append({
                'pdb_head': pdb_head,
                'n_residues': n_common,
                'spearman_r': float(sp_r),
                'spearman_p': float(sp_p),
                'pearson_r': float(pr_r),
                'pearson_p': float(pr_p),
            })

            all_confidences.extend(conf_values)
            all_bfactors.extend(bf_values)

            # Bin by resolution
            pdb_id = pdb_head.split('_')[0]
            if pdb_id in resolution_data:
                res = resolution_data[pdb_id]
                res_bin = f"{res:.1f}" if res < 4 else "4.0+"
                per_resolution_bin[res_bin]['conf'].extend(conf_values)
                per_resolution_bin[res_bin]['bf'].extend(bf_values)
                per_resolution_bin[res_bin]['corrs'].append(sp_r)

    # ── Overall statistics ──
    all_conf = np.array(all_confidences)
    all_bf = np.array(all_bfactors)

    overall_sp_r, overall_sp_p = spearmanr(all_conf, all_bf)
    overall_pr_r, overall_pr_p = pearsonr(all_conf, all_bf)

    print(f"\nOverall (n={len(all_conf):,} positions):")
    print(f"  Spearman r = {overall_sp_r:.4f}, p = {overall_sp_p:.2e}")
    print(f"  Pearson  r = {overall_pr_r:.4f}, p = {overall_pr_p:.2e}")

    # Mean per-chain correlation
    mean_sp = np.mean([c['spearman_r'] for c in per_chain_corrs])
    n_neg = sum(1 for c in per_chain_corrs if c['spearman_r'] < 0)
    n_sig = sum(1 for c in per_chain_corrs if c['spearman_p'] < 0.05)
    print(f"\nPer-chain (n={len(per_chain_corrs)} chains):")
    print(f"  Mean Spearman r = {mean_sp:.4f}")
    print(f"  Negative correlations: {n_neg}/{len(per_chain_corrs)} ({n_neg/len(per_chain_corrs)*100:.1f}%)")
    print(f"  Significant (p<0.05): {n_sig}/{len(per_chain_corrs)} ({n_sig/len(per_chain_corrs)*100:.1f}%)")

    # Per resolution bin
    print(f"\nPer resolution bin:")
    print(f"{'Resolution':<15} {'N positions':>12} {'Mean Conf':>10} {'Mean Sp r':>10}")
    print("-" * 50)
    res_bin_stats = {}
    for res_bin in sorted(per_resolution_bin.keys()):
        data = per_resolution_bin[res_bin]
        mean_conf = np.mean(data['conf'])
        mean_sp_r = np.mean(data['corrs']) if data['corrs'] else 0
        print(f"{res_bin:<15} {len(data['conf']):>12,} {mean_conf:>10.4f} {mean_sp_r:>10.4f}")
        res_bin_stats[res_bin] = {
            'n_positions': len(data['conf']),
            'mean_confidence': float(mean_conf),
            'mean_per_chain_spearman': float(mean_sp_r),
        }

    # ── Phase 4: Confidence binned by B-factor ──
    print(f"\nConfidence by B-factor bin:")
    bf_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 200)]
    bf_bin_stats = {}
    for lo, hi in bf_bins:
        mask = (all_bf >= lo) & (all_bf < hi)
        if mask.sum() > 10:
            mean_conf = all_conf[mask].mean()
            print(f"  B-factor {lo}-{hi}: n={mask.sum():,}, mean_conf={mean_conf:.4f}")
            bf_bin_stats[f"{lo}-{hi}"] = {
                'n': int(mask.sum()),
                'mean_confidence': float(mean_conf),
            }

    # ── Save ──
    results = {
        'n_pdbs_sampled': len(sample_pdbs),
        'n_chains_with_bfactors': len(bfactor_data),
        'n_chains_analyzed': len(confidence_data),
        'n_chains_correlated': len(per_chain_corrs),
        'n_total_positions': int(len(all_conf)),
        'overall_spearman_r': float(overall_sp_r),
        'overall_spearman_p': float(overall_sp_p),
        'overall_pearson_r': float(overall_pr_r),
        'overall_pearson_p': float(overall_pr_p),
        'mean_per_chain_spearman_r': float(mean_sp),
        'n_negative_correlations': n_neg,
        'n_significant_chains': n_sig,
        'resolution_bin_stats': res_bin_stats,
        'bfactor_bin_stats': bf_bin_stats,
        'per_chain_corrs': per_chain_corrs[:50],  # top 50 for file size
    }

    with open(os.path.join(OUTPUT_DIR, 'quality_assessment_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/quality_assessment_results.json")


if __name__ == '__main__':
    main()
