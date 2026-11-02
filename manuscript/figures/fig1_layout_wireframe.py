"""Rough layout wireframe for the Figure 1 art plan (not final art)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("/home/glt/rnaxgb/overleaf/figures")
INK = "#14213D"; TEAL = "#2A9D8F"; AMBER = "#E9C46A"; CORAL = "#E76F51"; NAVY = "#0B1020"

fig = plt.figure(figsize=(7.2, 7.5), facecolor="#FAFAF7")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 183); ax.set_ylim(0, 190)
ax.axis("off")

# panels
ax.add_patch(FancyBboxPatch((8, 102), 50, 74, boxstyle="round,pad=1.2",
                            fc="white", ec="#B8B8B8", lw=0.8))
ax.text(33, 170, "representation", ha="center", fontsize=7, color=INK, fontweight="bold")
ax.text(33, 158, "7-atom nucleotide\n247 node + 688 edge\nk = 32 graph\n935 descriptors", ha="center",
        va="top", fontsize=5.5, color="#555555", linespacing=1.5)

ax.add_patch(FancyBboxPatch((63, 66), 82, 110, boxstyle="round,pad=1.2",
                            fc=NAVY, ec="none"))
ax.text(104, 168, "HERO: real RNA ribbon → translucent lens → six beams", ha="center",
        fontsize=7, color="white", fontweight="bold")
ax.text(104, 154, "faint topographic information landscape\nconfidence gradient along the ribbon",
        ha="center", va="top", fontsize=5.5, color="#CFCFCF", linespacing=1.5)
# beams
for i, c in enumerate([TEAL, AMBER, CORAL, TEAL, AMBER, CORAL]):
    ax.plot([104, 136], [140 - i * 7, 140 - i * 7], color=c, lw=3, alpha=0.85)
    ax.text(136, 140 - i * 7, ["identity", "pairing", "stacking", "SS class", "SASA", "reactivity"][i],
            va="center", fontsize=5.0, color="white")

ax.add_patch(FancyBboxPatch((150, 103), 27, 63, boxstyle="round,pad=1.0",
                            fc="white", ec="#B8B8B8", lw=0.8))
ax.text(163.5, 160, "recovery", ha="center", fontsize=6.5, color=INK, fontweight="bold")
ax.text(163.5, 152, "64.6 / 87.1\n55.1 / 48.9\n+ separate training\nevaluation 70.0/50.4",
        ha="center", va="top", fontsize=5.2, color="#555555", linespacing=1.4)

ax.add_patch(FancyBboxPatch((150, 68), 27, 30, boxstyle="round,pad=1.0",
                            fc="white", ec="#B8B8B8", lw=0.8))
ax.text(163.5, 91, "confidence\ntransport", ha="center", fontsize=6.5, color=INK, fontweight="bold")
ax.text(163.5, 82, "51.9 / 90.1\n79.9 / 95.1\n28.0 / 80.8", ha="center", va="top", fontsize=5.2,
        color="#555555", linespacing=1.4)

# bottom gates
y0 = 14
for i, (label, sub) in enumerate([("compatibility", "native vs decoy"),
                                  ("forward model", "AF3 consistency"),
                                  ("experiment", "structure / function")]):
    x = 18 + i * 52
    ax.add_patch(FancyBboxPatch((x, y0), 42, 30, boxstyle="round,pad=1.0",
                                fc="white", ec=[TEAL, AMBER, CORAL][i], lw=1.2))
    ax.text(x + 21, y0 + 20, label, ha="center", fontsize=6.5, color=INK, fontweight="bold")
    ax.text(x + 21, y0 + 11, sub, ha="center", fontsize=5.2, color="#555555")
    if i < 2:
        ax.add_patch(FancyArrowPatch((x + 42, y0 + 15), (x + 51, y0 + 15),
                                     arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK))
ax.text(91.5, 6, "d  evidence gates: not interchangeable", ha="center", fontsize=6.5,
        color=INK, fontweight="bold")

# panel labels
for s, x, y in [("a", 8, 178), ("b", 150, 178), ("c", 150, 64), ("d", 8, 47)]:
    ax.text(x, y, s, fontsize=9, fontweight="bold", color=INK)

fig.savefig(OUT / "fig1_layout_wireframe.png", dpi=300, facecolor=fig.get_facecolor())
fig.savefig(OUT / "fig1_layout_wireframe.pdf", facecolor=fig.get_facecolor())
print("wrote wireframe")
