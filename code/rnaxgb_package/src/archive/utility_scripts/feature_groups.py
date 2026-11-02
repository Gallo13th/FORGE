"""
Feature group definitions for ablation analysis.
935 features = 247 node + 344 edge_mean + 344 edge_std.

Verified against generate_torch_features_v0.py build_graph() construction order:
  node: extract_residue_features + extract_enrich_features +
        extract_angle_features + extract_rolling_features
  edge: RNAEdgeFeatures.build() → [rbf(224), direct(3), quat(4), sin_pe(32),
         rosetta(40), cec_pe(8), pair_angles(4), dih_tri(28), stack(1)]
"""

# ── Individual node sub-block column ranges (0-indexed) ──
NODE_BASE = list(range(0, 37))       # 37: bonds, distances, angles, planarity
NODE_DELTA = list(range(37, 79))     # 42: delta + delta2 (7 atoms x 3 axes x 2)
NODE_SPATIAL = list(range(79, 105))  # 26: curvature, SASA, inertia, P-dist stats
NODE_ENRICHED = list(range(105, 195)) # 90: helical, kNN, graph topo, base-pair, glycosidic
NODE_DIHEDRAL = list(range(195, 202)) # 7: backbone trihedral + vector-vector angles
NODE_ROLLING = list(range(202, 247))  # 45: 5 windows x 9 consistency types

# ── Edge sub-block layout (positions within one aggregation block) ──
# Each edge aggregation (mean or std) has this internal layout:
_EDGE_LAYOUT = [
    ("rbf",           224),  # 7 atom-pair RBF distances x 32 bins
    ("direct_vec",      3),  # P_i → P_j unit vector
    ("quat",            4),  # inter-residue orientation quaternion
    ("sin_pe",         32),  # sinusoidal positional encoding
    ("rosetta",        40),  # trRosetta-style features
    ("cec_pe",          8),  # CEC positional encoding
    ("pair_angles",     4),  # pair angles (k=1..4)
    ("dih_tri",        28),  # edge dihedral (14) + trihedral (14)
    ("stack_flag",      1),  # base stacking flag
]
assert sum(n for _, n in _EDGE_LAYOUT) == 344, f"Edge layout sums to {sum(n for _, n in _EDGE_LAYOUT)}"


def _edge_range(name, edge_base=0):
    """Return column indices for an edge feature block within one aggregation."""
    start = edge_base
    for n, count in _EDGE_LAYOUT:
        if n == name:
            return list(range(start, start + count))
        start += count
    raise KeyError(f"Unknown edge block: {name}")


# Edge MEAN block offsets (mean starts at col 247)
EDGE_MEAN_BASE = 247

# Edge STD block offsets (std starts at col 591)
EDGE_STD_BASE = 591

# Verify: 247 + 344 = 591, 591 + 344 = 935
assert EDGE_MEAN_BASE + 344 == EDGE_STD_BASE
assert EDGE_STD_BASE + 344 == 935


# ── Level 2 ablation groups (11 groups, edge mean+std combined) ──
LEVEL2_GROUPS = {
    # Node groups
    "node_base":           NODE_BASE,
    "node_delta":          NODE_DELTA,
    "node_spatial":        NODE_SPATIAL,
    "node_enriched":       NODE_ENRICHED,
    "node_dihedral":       NODE_DIHEDRAL,
    "node_rolling":        NODE_ROLLING,
    # Edge groups (mean + std combined)
    "edge_rbf":            (_edge_range("rbf", EDGE_MEAN_BASE)
                            + _edge_range("rbf", EDGE_STD_BASE)),
    "edge_direct_quat":    (_edge_range("direct_vec", EDGE_MEAN_BASE)
                            + _edge_range("quat", EDGE_MEAN_BASE)
                            + _edge_range("direct_vec", EDGE_STD_BASE)
                            + _edge_range("quat", EDGE_STD_BASE)),
    "edge_pe":             (_edge_range("sin_pe", EDGE_MEAN_BASE)
                            + _edge_range("cec_pe", EDGE_MEAN_BASE)
                            + _edge_range("sin_pe", EDGE_STD_BASE)
                            + _edge_range("cec_pe", EDGE_STD_BASE)),
    "edge_rosetta":        (_edge_range("rosetta", EDGE_MEAN_BASE)
                            + _edge_range("rosetta", EDGE_STD_BASE)),
    "edge_angles_dih":     (_edge_range("pair_angles", EDGE_MEAN_BASE)
                            + _edge_range("dih_tri", EDGE_MEAN_BASE)
                            + _edge_range("stack_flag", EDGE_MEAN_BASE)
                            + _edge_range("pair_angles", EDGE_STD_BASE)
                            + _edge_range("dih_tri", EDGE_STD_BASE)
                            + _edge_range("stack_flag", EDGE_STD_BASE)),
}

# ── Level 1 coarse groups (3 groups) ──
LEVEL1_GROUPS = {
    "node_all":       list(range(0, 247)),
    "edge_mean_all":  list(range(247, 591)),
    "edge_std_all":   list(range(591, 935)),
}

# ── Feature names generation ──
def generate_feature_names():
    """Generate 935 human-readable feature names matching the column order."""
    names = []

    # Node: base geometry
    for i in range(len(NODE_BASE)):
        names.append(f"node_base_{i}")

    # Node: delta
    for i in range(len(NODE_DELTA)):
        names.append(f"node_delta_{i}")

    # Node: spatial
    for i in range(len(NODE_SPATIAL)):
        names.append(f"node_spatial_{i}")

    # Node: enriched
    for i in range(len(NODE_ENRICHED)):
        names.append(f"node_enriched_{i}")

    # Node: dihedral
    for i in range(len(NODE_DIHEDRAL)):
        names.append(f"node_dihedral_{i}")

    # Node: rolling
    for i in range(len(NODE_ROLLING)):
        names.append(f"node_rolling_{i}")

    # Edge mean
    for name, count in _EDGE_LAYOUT:
        for i in range(count):
            names.append(f"edge_mean_{name}_{i}")

    # Edge std
    for name, count in _EDGE_LAYOUT:
        for i in range(count):
            names.append(f"edge_std_{name}_{i}")

    return names


def validate_groups():
    """Verify all groups are non-overlapping and cover indices 0-934."""
    all_indices = set()
    for name, cols in LEVEL2_GROUPS.items():
        for c in cols:
            if c in all_indices:
                overlap = [g for g, cs in LEVEL2_GROUPS.items() if c in cs]
                raise ValueError(f"Column {c} in both '{name}' and {overlap}")
            all_indices.add(c)

    expected = set(range(935))
    missing = expected - all_indices
    extra = all_indices - expected

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)[:20]}...")
    if extra:
        raise ValueError(f"Extra columns (should not happen): {sorted(extra)[:20]}...")

    # Check all indices are valid
    for name, cols in LEVEL2_GROUPS.items():
        for c in cols:
            assert 0 <= c <= 934, f"Column {c} out of range in group '{name}'"

    print(f"Feature groups validated: {len(LEVEL2_GROUPS)} groups, {len(expected)} total features, no overlaps")


def get_feature_category_map():
    """Return a dict mapping each column index to its LEVEL2 category name."""
    cat_map = {}
    for group_name, cols in LEVEL2_GROUPS.items():
        for c in cols:
            cat_map[c] = group_name
    return cat_map


if __name__ == '__main__':
    validate_groups()
    print("All checks passed.")
    print(f"Level 1 groups: {list(LEVEL1_GROUPS.keys())}")
    print(f"Level 2 groups ({len(LEVEL2_GROUPS)}):")
    for name, cols in LEVEL2_GROUPS.items():
        print(f"  {name:<25} {len(cols):>4} cols  [{cols[0]}-{cols[-1]}]")
