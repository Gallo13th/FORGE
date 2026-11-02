"""
Build RNA Geometric Feature Ontology.
Classifies all 935 features into a hierarchical taxonomy and computes
per-category discriminability (F-statistics) for each nucleotide pair.

Run from RNKDB/: python ../src/build_feature_ontology.py
"""
import os, sys, json, time, pickle
import numpy as np
from collections import defaultdict, OrderedDict
from scipy.stats import f_oneway
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import torch
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

OUTPUT_DIR = "./feature_ontology"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Ontology definition ──────────────────────────────────────────────
# Hierarchical classification of all feature types

ONTOLOGY = OrderedDict({
    "I. Intra-nucleotide backbone geometry": {
        "description": "Features computed from the 7 atoms within a single nucleotide",
        "subcategories": OrderedDict({
            "I.A. Covalent bond lengths": {
                "pattern": r"^bond_",
                "description": "Bond distances between consecutive backbone atoms (P-O5', O5'-C5', C5'-C4', C4'-C3', C3'-O3', O3'-N)"
            },
            "I.B. Non-bonded short-range distances": {
                "pattern": r"^dist_(P|O5|C5|C4|C3|O3|N)-(C[345]|O[35]|P|N)$",
                "description": "Through-space distances between non-adjacent backbone atoms (e.g., P-C4, C5-O3)"
            },
            "I.C. Sequential shift distances": {
                "pattern": r"^dist_.*_shift\d+$",
                "description": "Distances between atoms in sequentially adjacent nucleotides (i→i+k shifts, k=1,2,3)"
            },
            "I.D. Sequential shift angles": {
                "pattern": r"^angle_.*_span\d+$",
                "description": "Angles formed by atom triplets spanning sequential nucleotides"
            },
            "I.E. Coordinate displacements (δ, δ²)": {
                "pattern": r"^(delta|delta2)_",
                "description": "Per-atom displacement from local center and squared displacement, 7 atoms × 3 axes × 2 stats = 42 features"
            },
            "I.F. Per-atom backbone curvature": {
                "pattern": r"^[A-Z]a?_kappa",
                "description": "Frenet-style curvature κ per backbone atom; τ torsion where applicable"
            },
            "I.G. Sugar ring geometry": {
                "pattern": r"^(sugar_planarity|puckering_|ring_distort)",
                "description": "Sugar pucker parameters: planarity, puckering amplitude/phase (2 dims), ring distortion"
            },
        })
    },
    "II. Glycosidic bond geometry": {
        "description": "Geometry of the C4'-N1/N9 glycosidic bond connecting backbone to base",
        "subcategories": OrderedDict({
            "II.A. Glycosidic angles and orientation": {
                "pattern": r"^gly_",
                "description": "Glycosidic bond angle (C4'-N-base), cross-product norm, sugar-base angle, sugar-base plane cosine"
            },
            "II.B. Base orientation consistency": {
                "pattern": r"^base_norm_cons_",
                "description": "Windowed consistency of base normal vector orientation (5 and 11-nt windows)"
            },
            "II.C. Base stacking": {
                "pattern": r"^(stacking|base_stack_cons_|stack_density)",
                "description": "Base stacking score, windowed stacking consistency (3/5/7/11/13-nt), stacking density (5/11-nt)"
            },
        })
    },
    "III. Helical and backbone conformational parameters": {
        "description": "Helical parameters and backbone dihedral-based conformational features",
        "subcategories": OrderedDict({
            "III.A. Helical parameters": {
                "pattern": r"^helical_",
                "description": "Per-nucleotide helical rise, twist, roll, tilt; windowed standard deviations (3/7-nt)"
            },
            "III.B. Backbone dihedrals": {
                "pattern": r"^(dihedral_|torsion_)",
                "description": "Backbone dihedral sin/cos (α-ζ, χ, η, θ); windowed torsion variance"
            },
            "III.C. Backbone direction consistency": {
                "pattern": r"^(backbone_dir_cons|pp_dir_cons|base_norm_cons)",
                "description": "Windowed backbone/PP-direction vector consistency across multiple scales"
            },
            "III.D. Ribose plane consistency": {
                "pattern": r"^ribose_plane_cons_",
                "description": "Windowed ribose plane normal consistency (3/5/7/11/13-nt windows)"
            },
        })
    },
    "IV. Multi-scale local neighborhood statistics": {
        "description": "Features capturing the nucleotide's spatial context at multiple distance scales",
        "subcategories": OrderedDict({
            "IV.A. Solvent accessibility proxies": {
                "pattern": r"^sasa_",
                "description": "Approximate solvent-accessible surface area at 4/6/8/12 Å probe radii"
            },
            "IV.B. K-nearest-neighbor density metrics": {
                "pattern": r"^knn",
                "description": "Mean/std/kth distance and local density at k=16,32,64,128 neighbors; cross-scale ratios"
            },
            "IV.C. Graph-theoretic centrality": {
                "pattern": r"^graph_",
                "description": "Clustering coefficient, betweenness, eigenvector centrality, closeness, degree, triangles, local efficiency"
            },
            "IV.D. Local conformational windows": {
                "pattern": r"_(cons|smooth|sigma)_?\d+$|_(cv|std)_\d+$|_var\d+$",
                "description": "Windowed consistency/variance of various geometric properties across 3/5/7/11/13-nt windows"
            },
        })
    },
    "V. Inter-nucleotide pairwise geometry (edge-derived)": {
        "description": "Features computed on edges of the k-NN graph, aggregated via mean/std pooling",
        "subcategories": OrderedDict({
            "V.A. Inter-atomic RBF distances": {
                "pattern": r"^dist_.*_rbf\d+$",
                "description": "RBF-expanded pairwise distances between all 7×7=49 atom pairs (32 Gaussians each → 1568 raw edge features, aggregated to 344 mean+344 std per node)"
            },
            "V.B. Relative spatial encoding": {
                "pattern": r"^(direct_|quat_|pe_)",
                "description": "Direction vectors (3), quaternions (4), and sinusoidal positional encodings (32) between neighboring nodes"
            },
            "V.C. Rosetta-inspired inter-nucleotide geometry": {
                "pattern": r"^rosetta_",
                "description": "40-dimensional Rosetta-style base-backbone interaction features between nucleotide pairs"
            },
            "V.D. Edge dihedral and trihedral angles": {
                "pattern": r"^(dih_|tri_|pair_angle_|cec_)",
                "description": "Dihedral angles (7 types × sin/cos = 14), trihedral angles (7 types × sin/cos = 14), pair angles (4), CEC features (8)"
            },
            "V.E. Base-pairing and stacking flags": {
                "pattern": r"^stack_flag",
                "description": "Binary stacking interaction flag from edge geometry"
            },
        })
    },
    "VI. Global and center-referenced features": {
        "description": "Features referencing the global chain coordinate system or overall shape",
        "subcategories": OrderedDict({
            "VI.A. Center and inertia": {
                "pattern": r"^(center_|inertia_|dist_global|dist_main_dim)",
                "description": "Chain center of mass (3), moment of inertia (min/max/ratio), distance to global center, distance along main dimension"
            },
            "VI.B. Local curvature and sequential turns": {
                "pattern": r"(local_curvature|sequential_turn|C4_local_curvature)",
                "description": "Local C4' curvature (5-nt window), sequential turn span angle"
            },
            "VI.C. Electrostatic and dynamics proxies": {
                "pattern": r"^(elec_|dyn_|local_helix|hydration|ss_tendency|local_density|contact_num)",
                "description": "Electrostatic cluster features, dynamic betweenness, local helix propensity, hydration, SS tendency, contact number"
            },
            "VI.D. Base-pair geometry": {
                "pattern": r"^bp_",
                "description": "Base-pair geometry best/top-3 scores, number of good pairs, partner distance"
            },
        })
    },
})


def get_feature_names():
    """Get all 935 feature names from a single-chain featurization."""
    coords_dir = "./coords"
    for f in sorted(os.listdir(coords_dir)):
        if not f.endswith('.npy'):
            continue
        coords = np.load(os.path.join(coords_dir, f))
        if coords.shape[0] < 5:
            continue
        data = build_graph(torch.from_numpy(np.nan_to_num(coords[:, :7])).float())
        node_names = list(data['node_feat_name'])
        edge_names = list(data['edge_feat_name'])
        # Feature layout: [node_0..node_N, edge_mean_0..edge_mean_M, edge_std_0..edge_std_M]
        full_names = (list(node_names) +
                      [f"{n}_mean" for n in edge_names] +
                      [f"{n}_std" for n in edge_names])
        return node_names, edge_names, full_names
    raise RuntimeError("No coordinate files found")


def match_feature(name, pattern):
    """Check if a feature name matches a regex pattern."""
    import re
    return bool(re.match(pattern, name))


def classify_features(node_names, edge_names):
    """Assign each of the 935 features to ontology categories."""
    n_node = len(node_names)
    n_edge = len(edge_names)

    # Build full name list
    all_names = list(node_names) + [f"{n}_mean" for n in edge_names] + [f"{n}_std" for n in edge_names]

    classification = {}  # feature_index -> (major_category, subcategory)
    cat_counts = defaultdict(int)
    unclassified = []

    for idx, name in enumerate(all_names):
        # Strip _mean/_std suffix for edge features when matching
        match_name = name.replace('_mean', '').replace('_std', '')

        classified = False
        for major_cat, cat_info in ONTOLOGY.items():
            for sub_cat, sub_info in cat_info['subcategories'].items():
                if match_feature(match_name, sub_info['pattern']):
                    classification[idx] = (major_cat, sub_cat)
                    cat_counts[(major_cat, sub_cat)] += 1
                    classified = True
                    break
            if classified:
                break

        if not classified:
            unclassified.append((idx, name))

    return classification, cat_counts, unclassified, all_names


def compute_pairwise_fstat(features, labels, feature_indices):
    """Compute F-statistic for each nucleotide pair using specified feature indices.

    Returns dict mapping (nt1, nt2) -> mean F across features
    """
    pair_fstats = defaultdict(list)
    nt_pairs = [('A','G'), ('U','C'), ('A','C'), ('A','U'), ('G','C'), ('G','U')]

    for idx in feature_indices:
        feat_vals = features[:, idx]
        for nt1, nt2 in nt_pairs:
            mask1 = labels == (ord(nt1) - ord('A')) if nt1 in 'ACGU' else None
            # Use numeric labels: A=0, C=1, G=2, U=3
            base_to_idx = {'A':0, 'C':1, 'G':2, 'U':3}
            mask = (labels == base_to_idx[nt1]) | (labels == base_to_idx[nt2])
            if mask.sum() < 10:
                continue
            group1 = feat_vals[labels == base_to_idx[nt1]]
            group2 = feat_vals[labels == base_to_idx[nt2]]
            if len(group1) < 5 or len(group2) < 5:
                continue
            try:
                f_val, _ = f_oneway(group1, group2)
                pair_fstats[(nt1, nt2)].append(f_val)
            except Exception:
                continue

    return {pair: np.mean(fs) for pair, fs in pair_fstats.items()}


def main():
    print("=" * 70)
    print("RNA Geometric Feature Ontology")
    print("=" * 70)

    # ── Step 1: Get feature names ──
    print("\n[1/4] Extracting feature names...")
    node_names, edge_names, all_names = get_feature_names()
    print(f"  Node features: {len(node_names)}")
    print(f"  Edge features: {len(edge_names)}")
    print(f"  Total after aggregation: {len(all_names)}")

    # ── Step 2: Classify features ──
    print("\n[2/4] Classifying features into ontology...")
    classification, cat_counts, unclassified, _ = classify_features(node_names, edge_names)

    print(f"\n  Feature classification:")
    for major_cat in ONTOLOGY:
        major_total = sum(v for (m, s), v in cat_counts.items() if m == major_cat)
        print(f"  {major_cat}: {major_total} features")
        for sub_cat in ONTOLOGY[major_cat]['subcategories']:
            n = cat_counts.get((major_cat, sub_cat), 0)
            if n > 0:
                print(f"    {sub_cat}: {n}")

    if unclassified:
        print(f"\n  ⚠ Unclassified features: {len(unclassified)}")
        for idx, name in unclassified[:10]:
            print(f"    [{idx}] {name}")

    # ── Step 3: Load features and compute discriminability ──
    print("\n[3/4] Computing per-category discriminability...")

    # Load a representative subset of post-2025 features
    feature_dir = "./fair_eval_v3"
    import glob
    chunk_files = sorted(glob.glob(os.path.join(feature_dir, "post2025_test_X_chunk_*.npy")))
    label_files = sorted(glob.glob(os.path.join(feature_dir, "post2025_test_y_chunk_*.npy")))

    if not chunk_files:
        # Try alternative location
        chunk_files = sorted(glob.glob(os.path.join(feature_dir, "*_X_*.npy")))
        label_files = sorted(glob.glob(os.path.join(feature_dir, "*_y_*.npy")))

    if chunk_files:
        print(f"  Loading {len(chunk_files)} feature chunks...")
        X = np.vstack([np.load(f) for f in chunk_files[:3]])  # First 3 chunks (~1500 chains)
        y = np.hstack([np.load(f) for f in label_files[:3]])
    else:
        # Generate features on the fly from coordinate files
        print("  No pre-computed features found. Generating on-the-fly from 200 chains...")
        import pandas as pd
        meta = pd.read_csv("./fair_eval/chain_metadata.csv")
        post2025 = meta[meta['split_group'] == 'post2025']
        chains = post2025['pdb_head'].tolist()[:200]

        X_parts, y_parts = [], []
        for pdb_head in chains:
            coord_file = f"./coords/{pdb_head}.npy"
            seq_file = f"./seq/{pdb_head}.fasta"
            if not os.path.exists(coord_file) or not os.path.exists(seq_file):
                continue
            try:
                coords = np.load(coord_file)
                if coords.shape[0] < 5 or coords.shape[0] > 500:
                    continue
                data = build_graph(torch.from_numpy(np.nan_to_num(coords[:, :7])).float())
                x = np.nan_to_num(data['x'].numpy())
                ea = data['edge_attr'].numpy()
                ei = data['edge_index'].numpy()
                nn = x.shape[0]
                em = aggregate_features_mean(ea, ei[1], nn) if ea.shape[0] > 0 else np.zeros((nn, ea.shape[1]))
                es = aggregate_features_std(ea, ei[1], nn) if ea.shape[0] > 0 else np.zeros((nn, ea.shape[1]))
                feats = np.concatenate([x, em, es], axis=-1).astype(np.float32)
                with open(seq_file) as fh:
                    seq = fh.readlines()[-1].strip()
                bm = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
                labs = np.array([bm.get(s, -1) for s in seq], dtype=np.int8)
                valid = (labs != -1) & (~np.isnan(feats).any(axis=1))
                X_parts.append(feats[valid])
                y_parts.append(labs[valid])
            except Exception:
                continue

        X = np.vstack(X_parts).astype(np.float32)
        y = np.hstack(y_parts).astype(np.int8)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Labels: {y.shape}")

    # ── Step 4: Compute discriminability per category ──
    print("\n[4/4] Computing per-category F-statistics...")

    nt_pairs = [('A','G'), ('U','C'), ('A','C'), ('A','U'), ('G','C'), ('G','U')]
    pair_labels = ['A↔G (purine)', 'U↔C (pyrimidine)', 'A↔C', 'A↔U', 'G↔C', 'G↔U']

    # Group feature indices by category
    cat_indices = defaultdict(list)
    for idx in range(X.shape[1]):
        if idx in classification:
            major, sub = classification[idx]
            cat_indices[(major, sub)].append(idx)

    # Compute F-stats per category
    ontology_results = {}
    for major_cat in ONTOLOGY:
        ontology_results[major_cat] = {
            'description': ONTOLOGY[major_cat]['description'],
            'n_features': sum(v for (m, s), v in cat_counts.items() if m == major_cat),
            'subcategories': {},
            'pairwise_fstats': {},
        }
        for sub_cat in ONTOLOGY[major_cat]['subcategories']:
            key = (major_cat, sub_cat)
            indices = cat_indices.get(key, [])
            if not indices:
                continue

            pair_fs = compute_pairwise_fstat(X, y, indices)
            # Compute "chemically ambiguous" score = mean of A/G and U/C F-stats
            ambig_f = (pair_fs.get(('A','G'), 0) + pair_fs.get(('U','C'), 0)) / 2
            # Compute "chemically unambiguous" score = mean of other 4 pairs
            unambig_f = np.mean([pair_fs.get(p, 0) for p in [('A','C'),('A','U'),('G','C'),('G','U')]])

            ontology_results[major_cat]['subcategories'][sub_cat] = {
                'description': ONTOLOGY[major_cat]['subcategories'][sub_cat]['description'],
                'n_features': len(indices),
                'pairwise_fstats': {f"{p[0]}/{p[1]}": float(v) for p, v in pair_fs.items()},
                'ambiguous_mean_F': float(ambig_f),
                'unambiguous_mean_F': float(unambig_f),
                'ambiguity_ratio': float(ambig_f / unambig_f) if unambig_f > 0 else 0,
            }

    # ── Print results ──
    print(f"\n{'='*90}")
    print("FEATURE ONTOLOGY: PER-CATEGORY DISCRIMINABILITY")
    print(f"{'='*90}")

    for major_cat, cat_data in ontology_results.items():
        print(f"\n{'-'*80}")
        print(f"{major_cat}  ({cat_data['n_features']} features)")
        print(f"  {cat_data['description']}")
        print(f"  {'Subcategory':<45} {'n':>4} {'A/G':>8} {'U/C':>8} {'ambig':>8} {'unamb':>8} {'ratio':>7}")
        print(f"  {'':->45} {'':->4} {'':->8} {'':->8} {'':->8} {'':->8} {'':->7}")

        for sub_cat, sub_data in cat_data['subcategories'].items():
            short_name = sub_cat.split('. ', 1)[1] if '. ' in sub_cat else sub_cat
            pfs = sub_data['pairwise_fstats']
            ag = pfs.get('A/G', 0)
            uc = pfs.get('U/C', 0)
            amb = sub_data['ambiguous_mean_F']
            unamb = sub_data['unambiguous_mean_F']
            ratio = sub_data['ambiguity_ratio']
            print(f"  {short_name:<45} {sub_data['n_features']:>4} {ag:>8.1f} {uc:>8.1f} {amb:>8.1f} {unamb:>8.1f} {ratio:>7.2f}")

    # ── Top discriminators for chemically ambiguous pairs ──
    print(f"\n{'='*90}")
    print("TOP FEATURE CATEGORIES FOR CHEMICALLY AMBIGUOUS PAIRS")
    print(f"{'='*90}")

    # Flatten and rank
    all_subcats = []
    for major_cat, cat_data in ontology_results.items():
        for sub_cat, sub_data in cat_data['subcategories'].items():
            ag = sub_data['pairwise_fstats'].get('A/G', 0)
            uc = sub_data['pairwise_fstats'].get('U/C', 0)
            all_subcats.append({
                'category': sub_cat,
                'major': major_cat,
                'A/G_F': ag,
                'U/C_F': uc,
                'ambig_mean': (ag + uc) / 2,
                'n_features': sub_data['n_features'],
            })

    print(f"\n  Best for A↔G discrimination (purine ambiguity):")
    for sc in sorted(all_subcats, key=lambda x: -x['A/G_F'])[:8]:
        print(f"    {sc['category']}: F={sc['A/G_F']:.1f} ({sc['n_features']} features)")

    print(f"\n  Best for U↔C discrimination (pyrimidine ambiguity):")
    for sc in sorted(all_subcats, key=lambda x: -x['U/C_F'])[:8]:
        print(f"    {sc['category']}: F={sc['U/C_F']:.1f} ({sc['n_features']} features)")

    # ── Save ──
    output = {
        'ontology': ONTOLOGY,
        'feature_counts': {f"{m} / {s}": n for (m, s), n in cat_counts.items()},
        'total_classified': sum(cat_counts.values()),
        'total_features': len(all_names) if 'all_names' in dir() else 935,
        'unclassified': [(idx, name) for idx, name in unclassified],
        'discriminability': ontology_results,
        'n_samples': int(X.shape[0]),
    }

    with open(os.path.join(OUTPUT_DIR, 'feature_ontology.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nOntology saved to {OUTPUT_DIR}/feature_ontology.json")


if __name__ == '__main__':
    main()
