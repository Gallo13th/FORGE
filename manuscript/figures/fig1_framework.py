"""Draft of a new main Figure 1: FORGE framework and central results."""
import os, sys
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nc")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

sys.path.insert(0, "/home/glt/rnaxgb/NMI-submission/figures")
import nmi_style as S
from nmi_style import COLORS as COL

S.apply_style()
OUT = Path("/home/glt/rnaxgb/overleaf/figures")
OUT.mkdir(exist_ok=True)

fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.36)


def panel_label(ax, s, x=-0.10, y=1.10):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold")


# a — schematic workflow
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("From backbone geometry to task-specific readouts", fontsize=6.8, pad=2)


def flow_box(x, y, w, h, title, sub, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc="white", ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=6.0, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
            fontsize=4.8, color=COL["neutral"], linespacing=1.1)


flow_box(0.35, 4.15, 3.1, 1.05, "7-atom backbone",
         "P, O5, C5, C4\nC3, O3, N1/N9", COL["baseline"])
ax.add_patch(FancyArrowPatch((1.9, 4.15), (1.9, 3.85), arrowstyle="-|>",
                             mutation_scale=7, color=COL["dark"], lw=0.8))
flow_box(0.35, 2.80, 3.1, 1.00, "935 descriptors",
         "247 node + 688 edge\nk=32 residue graph", COL["georna"])
ax.add_patch(FancyArrowPatch((1.9, 2.80), (1.9, 2.50), arrowstyle="-|>",
                             mutation_scale=7, color=COL["dark"], lw=0.8))
flow_box(0.35, 1.45, 3.1, 1.00, "XGBoost", "gradient-boosted trees", COL["dms"])
# readout chips, 2 columns x 3 rows
chips = [("identity", 0), ("pairing", 1), ("stacking", 2),
         ("SS class", 3), ("SASA", 4), ("reactivity", 5)]
for name, k in chips:
    col = 0 if k < 3 else 1
    row = k % 3
    x = 4.55 + col * 2.55
    y = 4.55 - row * 1.10
    ax.add_patch(FancyBboxPatch((x, y), 2.30, 0.72, boxstyle="round,pad=0.04",
                                fc=COL["heat_mid"], ec="none", alpha=0.35))
    ax.text(x + 1.15, y + 0.36, name, ha="center", va="center", fontsize=5.8)
ax.text(7.0, 5.55, "task heads", ha="center", fontsize=5.4, color=COL["neutral"])
ax.add_patch(FancyArrowPatch((3.55, 1.95), (4.45, 1.95), arrowstyle="-|>",
                             mutation_scale=7, color=COL["dark"], lw=0.8))
ax.text(5.0, 0.90, "interpretable features + controlled evaluation",
        ha="center", fontsize=5.2, color=COL["neutral"])
panel_label(ax, "a")

# b — decomposition, with retraining separated
sub = gs[0, 1].subgridspec(2, 1, hspace=0.75, height_ratios=[1.8, 0.9])
ax = fig.add_subplot(sub[0])
labels = ["All", "Exact\nrepeats", "Sequence-\nnovel", "<80%\nidentity"]
vals = [64.6, 87.1, 55.1, 48.9]
colors = [COL["baseline"], COL["error"], COL["georna"], COL["ablation_mid"]]
y = np.arange(len(vals))[::-1]
ax.barh(y, vals, color=colors, height=0.62)
for yy, v in zip(y, vals):
    ax.text(v + 1.4, yy, f"{v:.1f}", va="center", fontsize=5.8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.4)
ax.set_xlim(0, 100); ax.set_xlabel("Nucleotide recovery (%)", fontsize=5.4)
ax.set_title("Temporal recovery by sequence population", fontsize=6.5, pad=2)
panel_label(ax, "b")

ax = fig.add_subplot(sub[1])
labels2 = ["Family-\npermissive", "Family-\ndisjoint"]
vals2 = [70.0, 50.4]
y2 = np.arange(len(vals2))[::-1]
ax.barh(y2, vals2, color=[COL["ribodiffusion"], COL["dms"]], height=0.55)
for yy, v in zip(y2, vals2):
    ax.text(v + 1.4, yy, f"{v:.1f}", va="center", fontsize=5.6)
ax.set_yticks(y2); ax.set_yticklabels(labels2, fontsize=5.2)
ax.set_xlim(0, 100); ax.set_xlabel("Recovery (%)", fontsize=5.2)
ax.set_title("Separate post-2025 training evaluation", fontsize=5.8, pad=1)

# c — information spectrum
ax = fig.add_subplot(gs[1, 0])
tasks = [("identity", 64.6), ("pairing", 79.2), ("stacking", 81.8),
         ("SS class", 78.5), ("SASA (r×100)", 94.8),
         ("reactivity (r×100)", 57.6), ("pseudoknot (AUC×100)", 75.0)]
y = np.arange(len(tasks))[::-1]
for yy, (name, v) in zip(y, tasks):
    ax.plot([0, v], [yy, yy], color=COL["light"], lw=2.0, solid_capstyle="round")
    ax.scatter([v], [yy], s=22, color=COL["georna"], zorder=3)
    ax.text(v + 1.5, yy, f"{v:.1f}", va="center", fontsize=5.2)
ax.set_yticks(y); ax.set_yticklabels([t[0] for t in tasks], fontsize=5.2)
ax.set_xlim(0, 108); ax.set_xlabel("Metric value (not commensurate)", fontsize=5.4)
ax.set_title("Task-specific information spectrum", fontsize=6.8, pad=2)
panel_label(ax, "c")

# d — design evidence is not interchangeable
ax = fig.add_subplot(gs[1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Evidence types for a design claim", fontsize=6.8, pad=2)
boxes = [
    (0.2, "Backbone\ncompatibility", "native vs decoy\nscore", COL["georna"]),
    (3.55, "Forward-model\nconsistency", "AlphaFold 3 fold\nof designed sequence", COL["heat_mid"]),
    (6.9, "Experimental\nvalidation", "solved structure or\nfunctional assay", COL["ribodiffusion"]),
]
for x, title, sub, c in boxes:
    ax.add_patch(FancyBboxPatch((x, 3.05), 2.75, 1.55, boxstyle="round,pad=0.05",
                                fc="white", ec=c, lw=0.9))
    ax.text(x + 1.375, 4.05, title, ha="center", va="center", fontsize=5.4,
            fontweight="bold", linespacing=1.0)
    ax.text(x + 1.375, 3.42, sub, ha="center", va="center", fontsize=4.9,
            color=COL["neutral"], linespacing=1.1)
for x in (2.98, 6.33):
    ax.add_patch(FancyArrowPatch((x, 3.82), (x + 0.52, 3.82), arrowstyle="-|>",
                                 mutation_scale=7, color=COL["dark"], lw=0.8))
ax.text(5.0, 2.45, "not interchangeable", ha="center", fontsize=6.0,
        color=COL["error"], fontweight="bold")
ax.text(5.0, 1.35, "solved AI-designed pseudoknots: generation $\\neq$ read-back $\\neq$ foldability",
        ha="center", fontsize=5.1, color=COL["neutral"])
panel_label(ax, "d")

fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.07)
fig.savefig(OUT / "fig1_framework.pdf")
fig.savefig(OUT / "fig1_framework.png", dpi=400, bbox_inches="tight")
print("wrote fig1_framework")
