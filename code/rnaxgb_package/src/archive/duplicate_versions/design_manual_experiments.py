"""
Manual experiments addressing review weaknesses.
Designed for user to run locally after fixing infrastructure.

Experiments:
  1. 3-atom baseline (P, C4', N) vs 7-atom features
  2. DSSR validation of 3D SS classifier
  3. Three-element bridge analysis (base/phosphate/ribose + ligand/ion/water)
"""
import json
import numpy as np
from collections import defaultdict
import os, sys

BASE = '/home/glt/rnaxgb/RNKDB'

# ============================================================================
# Experiment 1: 3-atom baseline
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: 3-atom (P, C4', N) baseline vs 7-atom features")
print("=" * 70)
print("""
RATIONALE: The baseline DL models (gRNAde, RiboDiffusion, RDesign) operate on
a 3-bead backbone (P, C4', N1/N9). Our XGBoost uses 7 atoms. This creates an
input information asymmetry. To quantify how much of our advantage comes from
the richer atom set, we train XGBoost on features computed from only the 3 atoms
available to baselines.

APPROACH: Re-run feature generation with only atoms [0, 3, 6] (P, C4', N),
keeping the same feature definitions where applicable. The 3-atom features
include:
  - 3 pairwise distances among P, C4', N (vs 21 for 7-atom)
  - 1 bond angle triplet (vs 35)
  - Backbone dihedrals that can be computed from 3 atoms per residue + neighbors
  - Spatial statistics computed from C4' positions only

The key comparison: XGBoost-3atom vs XGBoost-7atom (Δ = atom set advantage)
                      XGBoost-3atom vs gRNAde/RiboDiffusion (Δ = feature+architecture advantage)

IMPLEMENTATION STATUS: Script framework ready. Requires running feature generation
with 3-atom coordinate subset then XGBoost training.

To run:
  1. Modify generate_torch_features_v0.py to accept --atom_subset flag
  2. Generate features with --atom_subset 0,3,6
  3. Train XGBoost with run_xgb_model_v2.py
  4. Compare results

Expected result: 3-atom XGBoost should be between baselines (30-41%) and
7-atom XGBoost (63%). The gap quantifies atom-set contribution.
""")

# ============================================================================
# Experiment 2: DSSR Validation of SS Classifier
# ============================================================================
print("=" * 70)
print("EXPERIMENT 2: DSSR validation of 3D SS classifier")
print("=" * 70)
print("""
RATIONALE: Our SS classification uses custom 3D geometry-based base-pair
detection (C4'-C4' ≤ 16Å, glycosidic convergence < 90°, base normal alignment
< 60°). DSSR is the gold standard for RNA structure annotation. We need to
quantify agreement between our classifier and DSSR on the 200 annotated chains.

APPROACH: For each of the 200 annotated chains:
  1. Get the PDB ID from the chain name
  2. Submit to DSSR for analysis
  3. Extract per-nucleotide SS assignments from DSSR output
  4. Compare with our 3D geometry-based assignments
  5. Compute Cohen's κ and confusion matrix

DSSR SS categories to map:
  DSSR 'WC' (Watson-Crick) → stem
  DSSR 'Hoogsteen', 'Wobble' → stem (non-canonical pairs)
  DSSR 'helix' → stem
  DSSR 'junction', 'internal_loop' → junction, internal_loop, bulge
  DSSR unpaired → loop, single_stranded, hairpin_loop

The mapping is not 1:1 — our classifier has 8 categories, DSSR has different
annotation conventions. We need a reasonable mapping to compare.

DSSR ACCESS NOTE: The dssrwebapi package at /home/glt/rnaxgb/dssrwebapi/
has import issues (expects to be installed as package). Fix with:
  pip install -e /home/glt/rnaxgb/dssrwebapi/
Or use the CLI directly:
  python3 /home/glt/rnaxgb/dssrwebapi/cli.py analyze <pdb_id>
""")

# Check which PDB IDs are in our 200 annotated chains
with open(f'{BASE}/fair_eval_v3/extended_annotations.json') as f:
    annotations = json.load(f)

annotated_chains = list(annotations['protein_binding']['per_chain'].keys())
pdb_ids = sorted(set(c.split('_')[0] for c in annotated_chains))
print(f"Annotated chains: {len(annotated_chains)} from {len(pdb_ids)} unique PDBs")
print(f"Sample PDB IDs: {pdb_ids[:10]}")
print(f"\nTo validate: run DSSR on these {len(pdb_ids)} PDBs and compare SS assignments.")

# ============================================================================
# Experiment 3: Three-element bridge analysis
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Three-element bridge analysis")
print("=" * 70)
print("""
RATIONALE: Current binding-site analysis is too coarse (just "binding" vs
"non-binding"). RNA structure is stabilized by three-element bridge systems:
  - Base moiety (N1/N9, N2, N3, O2, O4, O6, N4, N6, N7)
  - Phosphate/sugar backbone (P, O5', O3', C4', C3')
  - Environmental molecules (water, metal ions, small-molecule ligands)

These form stabilizing bridges: e.g., Mg2+ bridging phosphate→base, water-mediated
base→phosphate contacts, ligand π-stacking with bases + H-bonding to backbone.

APPROACH: For each nucleotide position in the 200 annotated chains:
  1. Classify environmental contact:
     - Water-mediated: C4' or base atom within 3.5Å of water oxygen
     - Ion-coordinated: C4' or base atom within 3.5Å of metal (Mg2+, K+, Na+, etc.)
     - Ligand-contacting: C4' or base atom within 4.0Å of non-polymer ligand
  2. Classify RNA moiety involved:
     - Base contact: N1/N9 within 3.5Å of environmental molecule
     - Phosphate contact: P within 3.5Å
     - Ribose contact: C4' within 3.5Å (not via phosphate)
  3. Classify bridge type:
     - Two-element bridge: base+phosphate, base+ribose, phosphate+ribose
     - Three-element bridge: base+phosphate+ribose all engaged
     - Single contact: only one moiety involved
  4. Compute XGBoost accuracy and XGB-dominant enrichment per bridge type

This provides mechanistic granularity beyond "binding vs non-binding."
""")

# Analyze existing binding annotations for moiety-level detail
with open(f'{BASE}/fair_eval_v3/binding_annotations_v2.json') as f:
    binding = json.load(f)

print("Current binding annotation detail level:")
print(f"  Fields available per chain: {list(binding[list(binding.keys())[0]].keys())}")
print(f"\nTo extend to three-element bridge analysis:")
print("  1. Re-run binding annotation with per-position moiety flags")
print("  2. Script: src/annotate_binding_sites.py needs extension")
print("  3. Add flags: base_contact, phosphate_contact, ribose_contact")
print("  4. Classify bridge type from combination of flags")

# ============================================================================
# Experiment 4 (Bonus): Feature-level binding perturbation signature
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4 (Bonus): Feature perturbation at three-element bridges")
print("=" * 70)
print("""
RATIONALE: If three-element bridges produce distinct geometric signatures,
specific features should show systematic shifts at bridge positions vs
non-bridge positions. This is a more targeted version of the KS test analysis.

APPROACH: For each bridge type (base-only, phosphate-only, base+phosphate bridge,
base+phosphate+ribose bridge), compute the top-10 SHAP features' distribution
shift (Cohen's d) relative to non-binding positions. Test whether bridge
positions show larger shifts than single-contact positions.

PREDICTION: Three-element bridges should show the largest feature shifts,
consistent with the hypothesis that multi-moiety contacts produce the most
distinctive backbone geometric perturbations.
""")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Manual Experiment Queue")
print("=" * 70)
print("""
Priority order (recommended by user):

  [HIGH]    Exp 2: DSSR validation
            - Fix dssrwebapi import OR use CLI
            - Run on ~100 unique PDBs from annotated chains
            - Compare SS assignments, compute Cohen's κ
            - EFFORT: 1-2 days

  [MEDIUM]  Exp 3: Three-element bridge analysis
            - Extend annotate_binding_sites.py
            - Add per-position moiety contact flags
            - Classify bridge types, compute enrichment
            - EFFORT: 2-3 days

  [LOW]     Exp 1: 3-atom baseline
            - Modify feature generation for atom subset
            - Train XGBoost on 3-atom features
            - Compare with 7-atom and baseline results
            - EFFORT: 1-2 days (mostly compute)

  [BONUS]   Exp 4: Feature perturbation at bridges
            - Use Exp 3 output
            - Compute Cohen's d per bridge type
            - EFFORT: 0.5 day (analysis only)
""")
