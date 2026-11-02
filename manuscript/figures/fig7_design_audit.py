"""Nature Communications Fig. 7: design-score and interface audit."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nc")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, "/home/glt/rnaxgb/NMI-submission/figures")
import nmi_style as S
from nmi_style import COLORS as COL

S.apply_style()
OUT = Path("/home/glt/rnaxgb/NMI-submission/analysis/out")
FIG = Path("/home/glt/rnaxgb/NC-submission/figures")
O = json.load(open(OUT / "openknot_within.json"))
H = json.load(open(OUT / "openknot_heldout_baseline.json"))
C = json.load(open(OUT / "context_cards.json"))

fig = plt.figure(figsize=(7.2, 4.8))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.38)

# a: pooled vs within vs held-out
ax = fig.add_subplot(gs[0, 0])
vals = [O["pooled_spearman"], O["within_puzzle_normalized_rank"], O["heldout_target_spearman_mean"]]
labs = ["Pooled", "Within\npuzzle", "Held-out\ntarget"]
bars = ax.bar(range(3), vals, color=[COL["heat_mid"], COL["georna"], COL["dms"]], width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.0)
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=5.8)
ax.set_ylabel("Spearman $\\rho$"); ax.set_ylim(0, 0.36)
ax.set_title("OpenKnot association by evaluation scheme", fontsize=6.4)
S.label_panel(ax, "a", x=-0.22, y=1.10)

# b: per-puzzle baseline vs +CE
ax = fig.add_subplot(gs[0, 1])
P = pd.DataFrame(H["per_puzzle"])
ax.scatter(P.baseline_rho, P.plus_ce_rho, s=14, color=COL["georna"], edgecolor="white", linewidth=0.4)
lim = [-0.45, 0.6]
ax.plot(lim, lim, ls="--", lw=0.7, color=COL["neutral"])
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("MFE+GC+length $\\rho$"); ax.set_ylabel("+ FORGE CE $\\rho$")
ax.set_title("Held-out-target performance by puzzle", fontsize=6.4)
S.label_panel(ax, "b", x=-0.22, y=1.10)

# c: AF3 P-RMSD matrix (3 constructs with derived/refined data)
ax = fig.add_subplot(gs[1, 0])
constructs = ["10ZU", "11EH", "11AG"]
true = [15.0, 17.4, 24.5]
forge = [23.2, 26.5, 20.8]
ref = [23.9, 9.9, 23.7]
x = np.arange(len(constructs)); w = 0.26
ax.bar(x - w, true, w, color=COL["baseline"], label="true")
ax.bar(x, forge, w, color=COL["georna"], label="FORGE")
ax.bar(x + w, ref, w, color=COL["ribodiffusion"], label="refined")
ax.set_xticks(x); ax.set_xticklabels(constructs, fontsize=5.8)
ax.set_ylabel("P-RMSD ($\\AA$)"); ax.set_ylim(0, 30)
ax.legend(frameon=False, fontsize=5.0, loc="upper left", handlelength=1.0)
ax.set_title("AF3 P-RMSD by construct and sequence class", fontsize=6.4)
S.label_panel(ax, "c", x=-0.22, y=1.10)

# d: 11AG interface accuracy
ax = fig.add_subplot(gs[1, 1])
x = np.arange(2); w = 0.34
iface = [C["11AG_A"]["interface_acc"], C["11AG_C"]["interface_acc"]]
non = [C["11AG_A"]["noninterface_acc"], C["11AG_C"]["noninterface_acc"]]
ax.bar(x - w/2, np.array(iface)*100, w, color=COL["error"], label="interface")
ax.bar(x + w/2, np.array(non)*100, w, color=COL["georna"], label="non-interface")
for i in range(2):
    ax.text(x[i]-w/2, iface[i]*100+1.5, f"{iface[i]*100:.0f}", ha="center", fontsize=5.4)
    ax.text(x[i]+w/2, non[i]*100+1.5, f"{non[i]*100:.0f}", ha="center", fontsize=5.4)
ax.set_xticks(x); ax.set_xticklabels(["chain A", "chain C"], fontsize=5.8)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 70)
ax.legend(frameon=False, fontsize=5.0, loc="upper right", handlelength=1.0)
ax.set_title("11AG interface vs non-interface accuracy", fontsize=6.4)
S.label_panel(ax, "d", x=-0.22, y=1.10)

fig.savefig(FIG / "fig7_design_audit.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig7_design_audit.png", dpi=400, bbox_inches="tight")
print("wrote fig7")
