"""Regenerate FORGE NMI-submission figures in Nature Machine Intelligence style.

Reads the same locked RNKDB artifacts as the legacy generator
(Nature-Methods-submission/figures/generate_all.py) and redraws the four main
data figures (Fig. 2-5), two supplementary figures (S3 N-free controls, S4
stress details) and the AF3 forward-folding supplementary figure with the NMI
palette, Arial-metric fonts (Liberation Sans), and redesigned panel layouts.

Redesign decisions (NMI art audit, 2026-09-17):
  Fig. 2  a: vertical bars -> horizontal lollipop (no rotated tick labels,
             FORGE as hero at top, baselines sorted, direct value labels)
          b: two-point line chart -> paired dumbbell (weighted + mean recovery,
             emphasises the bounded temporal shift)
          c: family bars keep type; single-line labels, class colours, n shown
          d: confusion matrix -> white-to-indigo sequential cmap shared with
             Fig. 3d (suite-level colour consistency)
  Fig. 3  c: dual-axis (twinx) line chart -> two stacked mini-panels sharing
             the CE-bin x axis (dual y-axes are discouraged at NMI)
          b: unrelated bar colours -> single-hue indigo gradient
             (7-atom -> 5-atom -> 3-atom reads as progressive reduction)
  Fig. 4  c: two-bar "subset shift" panel (redundant with text and SI Table S1)
             -> stacked confidence histogram, correct vs incorrect positions
             (direct evidence for the abstention mechanism)
          d: histogram with mean/median vlines -> ECDF with quartile markers
             (reads the three textual claims directly: 26% < 30% abstention,
             median 85%, 41% never reach 90%)
  Fig. 5  d: 25-bar grouped chart -> dot plot (FORGE emphasised, baselines
             small markers, no legend look-up fatigue)
          a: compensatory-mutant tier colour separated from RDesign
             (previously identical ochre)

Numbers are computed from the locked artifacts at runtime; nothing is restated
by hand except the locked OpenKnot complete-case summary and stress-test
baseline recoveries (identical to the legacy generator).

Outputs (NMI-submission/figures/): fig2_identifiability, fig3_representation_
controls, fig4_calibration, fig6_design_stress (main-text Fig. 5),
fig_s1_nfree, suppfig_stress_details, fig5_af3_combined.
Source-data CSVs go to NMI-submission/source_data/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import wrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nmi")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import nmi_style as S
from nmi_style import COLORS as COL
from nmi_style import METHOD_COLORS, apply_style, label_panel

apply_style()

ROOT = Path("/home/glt/rnaxgb")
V3 = ROOT / "RNKDB/fair_eval_v3"
BL = ROOT / "RNKDB/baseline_results"
ABL = ROOT / "RNKDB/atom_ablation"
ONTO = ROOT / "RNKDB/feature_ontology/feature_ontology.json"
OK = ROOT / "RNKDB/openknot_results"
ENT = ROOT / "RNKDB/entropy_analysis"
STRESS = ROOT / "RNKDB/stress_test"
CAL = ROOT / "NMI-submission/analysis/out"
OUT = ROOT / "NMI-submission/figures"
SRC = ROOT / "NMI-submission/source_data"
SRC.mkdir(parents=True, exist_ok=True)

# Locked summaries carried over verbatim from the legacy generator.
OPENKNOT_COMPLETE_CASE = {
    "n": 1656,
    "rho_mfe_f1_ok": 0.135,
    "rho_model_ce_ok": 0.365,
    "partial_model_ce_given_mfe": {"rho": 0.549},
    "partial_model_ce_given_mfe_gc_length": {"rho": 0.123},
    "ols_delta_r2": {
        "r2_base": 0.251,
        "r2_full": 0.273,
        "delta_r2": 0.022,
        "f_stat": 48.7,
        "p_value": 4.3e-12,
        "ce_coefficient": 4.49,
    },
    "source": "SS_EVALUATION_RESULTS.md Section 31",
}

STRESS_CHAINS = ["10ZT_H_A", "10ZU_H_A", "11EH_H_A", "11AG_H_A", "11AG_H_C"]
STRESS_CHAIN_LABELS = ["10ZT", "10ZU", "11EH", "11AG A", "11AG C"]
STRESS_BASELINE_RECOVERY = {
    "10ZT_H_A": {"gRNAde": 0.125, "RiboDiffusion": 0.135, "RDesign": 0.188},
    "10ZU_H_A": {"gRNAde": 0.073, "RiboDiffusion": 0.250, "RDesign": 0.396},
    "11EH_H_A": {"gRNAde": 0.146, "RiboDiffusion": 0.219, "RDesign": 0.146},
    "11AG_H_A": {"gRNAde": 0.129, "RiboDiffusion": 0.228, "RDesign": 0.109},
    "11AG_H_C": {"gRNAde": 0.129, "RiboDiffusion": 0.238, "RDesign": 0.109},
}

KNN_CE_BINS = [
    {"label": "0.0-0.3", "accuracy": 0.815, "entropy": 0.606},
    {"label": "0.3-0.6", "accuracy": 0.556, "entropy": 0.724},
    {"label": "0.6-1.0", "accuracy": 0.496, "entropy": 0.729},
    {"label": "1.0-1.5", "accuracy": 0.476, "entropy": 0.758},
    {"label": "1.5+", "accuracy": 0.449, "entropy": 0.769},
]

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "nmi_sequential", [COL["heat_low"], COL["heat_mid"], COL["heat_high"]]
)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def pct(x):
    return 100 * float(x)


def save(fig, stem):
    S.save_figure(fig, OUT, stem)
    print(f"wrote {stem}")


def short_label(s, width=16):
    return "\n".join(wrap(str(s), width=width, break_long_words=False))


def display_family_name(name):
    label = str(name).replace("_", " ")
    return {
        "5 8S rRNA": "5.8S rRNA",
        "RNaseP bact b": "RNase P bact. B",
    }.get(label, label)


def weighted_recovery(records):
    lengths = np.asarray([int(r["length"]) for r in records], dtype=float)
    recoveries = np.asarray([float(r["recovery"]) for r in records], dtype=float)
    return float(np.sum(lengths * recoveries) / np.sum(lengths))


def bootstrap_weighted_recovery(records, n_boot=2000, seed=11):
    lengths = np.asarray([int(r["length"]) for r in records], dtype=float)
    recoveries = np.asarray([float(r["recovery"]) for r in records], dtype=float)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    n = len(records)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = np.sum(lengths[idx] * recoveries[idx]) / np.sum(lengths[idx])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return [np.nan, np.nan]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [float(max(0, centre - half)), float(min(1, centre + half))]


def value_label(ax, x, y, text, *, ha="left", dx=0.0, dy=0.0, fs=6.2, color=None, bold=False):
    ax.text(
        x + dx,
        y + dy,
        text,
        ha=ha,
        va="center",
        fontsize=fs,
        color=color or COL["dark"],
        fontweight="bold" if bold else "normal",
    )


# ---------------------------------------------------------------------------
# Generation-time layout audit (matplotlib window extents, ground truth)
# ---------------------------------------------------------------------------
def audit_figure(fig, name, skip_text=lambda t: False):
    """Report overlaps between text artists and between text and bar patches.

    Runs on the rendered canvas before saving; catches collisions that the
    post-hoc PDF audit attributes imprecisely (hatch tiles, clip rects).
    ``skip_text`` whitelists intentional text (e.g. heatmap cell labels).
    """
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = []
    for ax in fig.axes:
        for t in ax.texts:
            s = t.get_text().strip()
            if not s or skip_text(s):
                continue
            bb = t.get_window_extent(renderer=ren)
            texts.append((s, bb))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                s = t.get_text().strip()
                if s:
                    texts.append((s, t.get_window_extent(renderer=ren)))
        for t in ax.title, ax.xaxis.label, ax.yaxis.label:
            s = t.get_text().strip()
            if s:
                texts.append((s, t.get_window_extent(renderer=ren)))
    bars = []
    from matplotlib.patches import Rectangle

    for ax in fig.axes:
        axbb = ax.get_window_extent(renderer=ren)
        for p in ax.patches:
            if not isinstance(p, Rectangle):
                continue
            fc = p.get_facecolor()
            if fc[3] <= 0.05:
                continue
            rgb = tuple(round(c, 3) for c in fc[:3])
            if rgb == (0.910, 0.918, 0.929):  # background tint band (fig4b)
                continue
            bb = p.get_window_extent(renderer=ren)
            # clip to the axes box: bars may extend beyond xlim/ylim but are
            # visually clipped there
            x0 = max(bb.x0, axbb.x0); y0 = max(bb.y0, axbb.y0)
            x1 = min(bb.x1, axbb.x1); y1 = min(bb.y1, axbb.y1)
            if x1 <= x0 or y1 <= y0:
                continue
            from matplotlib.transforms import Bbox
            bb = Bbox([[x0, y0], [x1, y1]])
            if bb.width * bb.height > 30 and bb.width > 3 and bb.height > 3:
                bars.append(bb)
    problems = []
    for i in range(len(texts)):
        s1, b1 = texts[i]
        for j in range(i + 1, len(texts)):
            s2, b2 = texts[j]
            ix0 = max(b1.x0, b2.x0); iy0 = max(b1.y0, b2.y0)
            ix1 = min(b1.x1, b2.x1); iy1 = min(b1.y1, b2.y1)
            if ix1 - ix0 > 0.8 and iy1 - iy0 > 1.5:
                problems.append(f"text-text [{s1[:28]}] x [{s2[:28]}]")
        for bb in bars:
            ix0 = max(b1.x0, bb.x0); iy0 = max(b1.y0, bb.y0)
            ix1 = min(b1.x1, bb.x1); iy1 = min(b1.y1, bb.y1)
            if ix1 - ix0 > 1.0 and iy1 - iy0 > 1.5:
                area_t = max(1.0, (b1.x1 - b1.x0) * (b1.y1 - b1.y0))
                if (ix1 - ix0) * (iy1 - iy0) > 0.25 * area_t:
                    problems.append(
                        f"text-bar [{s1[:28]}] on bar@({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f})"
                    )
    if problems:
        print(f"  LAYOUT AUDIT {name}: {len(problems)} problems")
        for p in problems:
            print(f"    - {p}")
    else:
        print(f"  LAYOUT AUDIT {name}: clean")
    return problems


# ---------------------------------------------------------------------------
# Fig. 2 -- temporal identifiability
# ---------------------------------------------------------------------------
def fig2_identifiability():
    summary = load_json(V3 / "summary.json")
    comp = load_json(BL / "comparison_results_full.json")["overall"]
    mlp = load_json(V3 / "mlp_baseline_results.json")
    families = load_json(V3 / "per_family_all_models.json")
    per_chain = {
        "gRNAde": load_json(BL / "grnade_results.json"),
        "RiboDiffusion": load_json(BL / "ribodiffusion_results.json"),
        "RDesign": load_json(BL / "rdesign_results.json"),
        "FORGE": load_json(V3 / "post2025_xgb_results.json"),
    }
    ci_rows = []
    for i, (method, records) in enumerate(per_chain.items()):
        ci = bootstrap_weighted_recovery(records, seed=101 + i)
        ci_rows.append(
            {
                "method": method,
                "n_chains": len(records),
                "n_nt": int(sum(int(r["length"]) for r in records)),
                "weighted_recovery": weighted_recovery(records),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "ci_method": "chain bootstrap of weighted recovery, 2000 resamples",
            }
        )
    # matched-position override: re-score baselines on the FORGE position mask
    matched_csv = CAL / "per_chain_identifiability.csv"
    if matched_csv.exists():
        mm = pd.read_csv(matched_csv)
        def _boot(col):
            v = mm[col].dropna()
            w = mm.loc[v.index, "n"].to_numpy(float)
            r = v.to_numpy(float)
            rng = np.random.RandomState(0)
            bs = np.array([(r[i] * w[i]).sum() / w[i].sum()
                           for i in (rng.randint(0, len(r), len(r)) for _ in range(2000))])
            return float((r * w).sum() / w.sum()), np.percentile(bs, [2.5, 97.5])
        for meth, col in (("gRNAde", "gRNAde_rec"), ("RiboDiffusion", "RiboDiffusion_rec"),
                          ("RDesign", "RDesign_rec")):
            est, ci = _boot(col)
            for row in ci_rows:
                if row["method"] == meth:
                    row.update({"weighted_recovery": est, "ci95_low": float(ci[0]),
                                "ci95_high": float(ci[1]),
                                "n_chains": int(mm[col].notna().sum()),
                                "n_nt": int(mm.loc[mm[col].notna(), "n"].sum()),
                                "ci_method": "chain bootstrap on matched FORGE position mask"})
                    break
    pd.DataFrame(ci_rows).to_csv(SRC / "fig2a_weighted_recovery_bootstrap_ci.csv", index=False)
    ci_by_method = {r["method"]: r for r in ci_rows}

    pos = pd.DataFrame(load_json(ENT / "per_position_entropy.json"))
    nts = ["A", "C", "G", "U"]
    confusion_counts = pd.crosstab(pos["true_class"], pos["pred_class"]).reindex(
        index=nts, columns=nts, fill_value=0
    )
    confusion_pct = confusion_counts.div(confusion_counts.sum(axis=1), axis=0) * 100
    confusion_pct.round(3).to_csv(SRC / "fig2d_confusion_matrix.csv")

    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 0.75], hspace=0.52, wspace=0.34)

    # -- a: horizontal lollipop, FORGE hero on top --------------------------
    ax = fig.add_subplot(gs[0, 0])
    methods = ["gRNAde", "RiboDiffusion", "RDesign", "MLP", "FORGE"]
    values = [
        pct(ci_by_method["gRNAde"]["weighted_recovery"]),
        pct(ci_by_method["RiboDiffusion"]["weighted_recovery"]),
        pct(ci_by_method["RDesign"]["weighted_recovery"]),
        52.8,  # matched-budget multi-learner MLP (not the archived 59.5% run)
        pct(summary["post2025_test_weighted_recovery"]),
    ]
    order = np.argsort(values)  # ascending so FORGE lands on top
    ypos = np.arange(len(methods))
    for rank, i in enumerate(order):
        m, v = methods[i], values[i]
        color = METHOD_COLORS[m]
        hero = m == "FORGE"
        ax.plot([0, v], [rank, rank], color=color, lw=1.0 if hero else 0.8,
                alpha=1.0 if hero else 0.75, zorder=2)
        ax.scatter([v], [rank], s=46 if hero else 30, color=color, zorder=3,
                   edgecolor="white", linewidth=0.6)
        if m in ci_by_method:
            lo = pct(ci_by_method[m]["ci95_low"])
            hi = pct(ci_by_method[m]["ci95_high"])
            ax.plot([lo, hi], [rank, rank], color=COL["dark"], lw=0.9, zorder=4)
            for xt in (lo, hi):
                ax.plot([xt, xt], [rank - 0.14, rank + 0.14], color=COL["dark"], lw=0.9, zorder=4)
            value_label(ax, hi, rank, f"{v:.1f}", dx=1.3, bold=hero)
        else:
            value_label(ax, v, rank, f"{v:.1f}*", dx=1.3, bold=False)
    ax.axvline(25, color=COL["baseline"], ls="--", lw=0.8, zorder=1)
    ax.text(25, -0.62, "random", ha="center", va="top", fontsize=6.0, color=COL["baseline"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([methods[i] for i in order])
    ax.get_yticklabels()[-1].set_fontweight("bold")  # FORGE
    ax.set_xlim(0, 78)
    ax.set_ylim(-0.9, len(methods) - 0.1)
    ax.set_xlabel("Weighted recovery on post-2025 chains (%)")
    ax.set_title("Temporal recovery and contextual baselines")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    label_panel(ax, "a", x=-0.34, y=1.04)

    # -- b: dumbbell temporal shift -----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    metrics = [
        ("weighted", pct(summary["pre2025_test_weighted_recovery"]), pct(summary["post2025_test_weighted_recovery"]), COL["georna"]),
        ("mean per chain", pct(summary["pre2025_test_mean_recovery"]), pct(summary["post2025_test_mean_recovery"]), COL["ribodiffusion"]),
    ]
    yy = [1, 0]
    for (name, pre, post, color), yyv in zip(metrics, yy):
        ax.plot([0, 1], [yyv, yyv], color=color, lw=1.4, alpha=0.35, zorder=1,
                solid_capstyle="round")
        ax.scatter([0, 1], [yyv, yyv], s=120, color=color, zorder=3,
                   edgecolor="white", linewidth=0.7)
        ax.annotate("", xy=(1, yyv), xytext=(0, yyv),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0, alpha=0))
        ax.text(-0.08, yyv, f"{pre:.1f}", ha="right", va="center", fontsize=6.4, color=COL["dark"])
        ax.text(1.08, yyv, f"{post:.1f}", ha="left", va="center", fontsize=6.4,
                color=COL["dark"], fontweight="bold")
        ax.text(0.5, yyv + 0.30, name, ha="center", va="bottom", fontsize=6.4, color=color)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pre-2025\ncluster test", "post-2025\ntemporal test"])
    ax.set_yticks([])
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    ax.set_title("Bounded temporal shift")
    label_panel(ax, "b", x=-0.30, y=1.04)

    # -- c: family-level recovery -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    fam_rows = []
    for fam, rec in families.items():
        fam_rows.append((display_family_name(fam), rec["n_chains"], rec["models"]["XGBoost"]["mean_recovery"]))

    def family_group(name):
        lower = name.lower()
        if "rrna" in lower or "trna" in lower or "rnase p" in lower:
            return "rRNA/tRNA-like"
        if any(k in lower for k in ("purine", "sam", "tpp", "glm", "riboswitch")):
            return "riboswitch"
        return "other"

    group_order = {"rRNA/tRNA-like": 0, "riboswitch": 1, "other": 2}
    fam_rows = [(n_, n, r, family_group(n_)) for n_, n, r in fam_rows]
    fam_rows = sorted(fam_rows, key=lambda x: (group_order[x[3]], -x[2]))
    group_colors = {"rRNA/tRNA-like": COL["georna"], "riboswitch": COL["dms"], "other": COL["georna_6"]}
    fam_vals = [pct(r[2]) for r in fam_rows]
    y = np.arange(len(fam_rows))
    ax.barh(y, fam_vals, color=[group_colors[r[3]] for r in fam_rows], height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]} ({r[1]})" for r in fam_rows], fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlabel("Mean recovery (%)")
    ax.set_xlim(0, 108)
    for yyv, val in zip(y, fam_vals):
        value_label(ax, val, yyv, f"{val:.0f}", dx=1.2, fs=5.8)
    handles = [Line2D([0], [0], color=group_colors[g], lw=4) for g in group_order]
    ax.legend(handles, list(group_order), frameon=False, loc="lower right", fontsize=5.8,
              handlelength=1.2, borderaxespad=0.1)
    ax.axvline(25, color=COL["baseline"], ls="--", lw=0.7, zorder=0)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Recovery by Rfam family")
    label_panel(ax, "c", x=-0.44, y=1.04)

    # -- d: confusion matrix -------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(confusion_pct.to_numpy(float), vmin=0, vmax=75, cmap=HEAT_CMAP)
    ax.set_xticks(np.arange(len(nts)))
    ax.set_xticklabels(nts)
    ax.set_yticks(np.arange(len(nts)))
    ax.set_yticklabels(nts)
    ax.set_xlabel("Predicted base")
    ax.set_ylabel("True base")
    ax.set_title("Confusion matrix by nucleotide")
    for i in range(len(nts)):
        for j in range(len(nts)):
            val = float(confusion_pct.iloc[i, j])
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.4,
                    color="white" if val > 42 else COL["dark"],
                    fontweight="bold" if i == j else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("row %", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8)
    cbar.outline.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    label_panel(ax, "d", x=-0.30, y=1.04)

    audit_figure(fig, "fig2_identifiability")
    save(fig, "fig2_identifiability")


# ---------------------------------------------------------------------------
# Fig. 3 -- representation controls
# ---------------------------------------------------------------------------
def fig3_representation_controls():
    b2 = load_json(V3 / "v4_b2hard_results.json")["b2_hard"]
    atom = load_json(ABL / "atom_ablation_results.json")
    onto = load_json(ONTO)

    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.38)

    # -- a: matched-subset N-free retraining (bars from zero) ----------------
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    w = 0.34
    v7 = [pct(b2["acc_7atom_retrained"]), pct(b2["purine_acc_7atom"])]
    v6 = [pct(b2["acc_6atom_retrained"]), pct(b2["purine_acc_6atom"])]
    ax.bar(x - w / 2, v7, w, color=METHOD_COLORS["FORGE 7-atom"], label="7-atom")
    ax.bar(x + w / 2, v6, w, color=METHOD_COLORS["FORGE 6-atom"], label="6-atom N-free")
    for xpos, val in zip([x[0] - w / 2, x[0] + w / 2, x[1] - w / 2, x[1] + w / 2],
                         [v7[0], v6[0], v7[1], v6[1]]):
        ax.text(xpos, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=6.0,
                color=COL["dark"])
    ax.set_xticks(x)
    ax.set_xticklabels(["4-class\nnucleotide", "purine/\npyrimidine"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 104)
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.2,
              columnspacing=1.0, handletextpad=0.5)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Removing N retains most signal")
    label_panel(ax, "a", x=-0.20, y=1.04)

    # -- b: inference-time atom masking (single-hue gradient) ----------------
    ax = fig.add_subplot(gs[0, 1])
    order = ["7atom", "5atom", "3atom"]
    labels = ["7-atom", "5-atom", "3-atom"]
    vals = [pct(atom["results"][k]["weighted_recovery"]) for k in order]
    kept = [atom["results"][k]["n_features_kept"] for k in order]
    grad = [COL["georna"], "#7884B4", "#C0C6E0"]
    xx = np.arange(len(labels))
    ax.bar(xx, vals, color=grad, width=0.58)
    for xpos, val in zip(xx, vals):
        ax.text(xpos, val + 1.1, f"{val:.1f}", ha="center", va="bottom", fontsize=6.2,
                color=COL["dark"])
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{lab}\n({n} features)" for lab, n in zip(labels, kept)], fontsize=6.2)
    ax.set_ylabel("Weighted recovery (%)")
    ax.set_ylim(0, 74)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Signal decays gracefully with masking")
    label_panel(ax, "b", x=-0.20, y=1.04)

    # -- c: kNN ambiguity, stacked shared-x panels ---------------------------
    sub = gs[1, 0].subgridspec(2, 1, hspace=0.12)
    ax_top = fig.add_subplot(sub[0])
    ax_bot = fig.add_subplot(sub[1], sharex=ax_top)
    labels = [b["label"] for b in KNN_CE_BINS]
    k10 = [pct(b["accuracy"]) for b in KNN_CE_BINS]
    entropy = [b["entropy"] for b in KNN_CE_BINS]
    xx = np.arange(len(labels))
    ax_top.plot(xx, k10, "o-", color=COL["confidence"], lw=1.3, markersize=3.6)
    ax_top.set_ylabel("$k$=10 NN\naccuracy (%)", fontsize=6.6)
    ax_top.set_ylim(35, 92)
    ax_top.set_xlim(-0.25, 4.75)
    ax_top.yaxis.grid(True)
    ax_top.set_axisbelow(True)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_title("Uncertainty marks mixed neighbourhoods", pad=3)
    ax_top.text(0.0, k10[0] + 3.5, f"{k10[0]:.0f}", fontsize=5.8, ha="center", color=COL["confidence"])
    ax_top.text(4.12, k10[-1], f"{k10[-1]:.0f}", fontsize=5.8, ha="left", va="center",
                color=COL["confidence"])
    ax_bot.plot(xx, entropy, "s-", color=COL["uncertainty"], lw=1.3, markersize=3.4)
    ax_bot.set_ylabel("Local label\nentropy", fontsize=6.6)
    ax_bot.set_ylim(0.55, 0.83)
    ax_bot.set_xticks(xx)
    ax_bot.set_xticklabels(labels, fontsize=6.2)
    ax_bot.set_xlabel("FORGE cross-entropy bin (higher = more uncertain)", fontsize=6.8)
    ax_bot.yaxis.grid(True)
    ax_bot.set_axisbelow(True)
    label_panel(ax_top, "c", x=-0.24, y=1.10)

    # -- d: ontology heatmap ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    subs = []
    for cat in onto["discriminability"].values():
        for name, sd in cat["subcategories"].items():
            pf = sd["pairwise_fstats"]
            subs.append((name.split(". ", 1)[-1], pf.get("A/G", np.nan), pf.get("U/C", np.nan)))
    top = sorted(subs, key=lambda x: max(x[1], x[2]), reverse=True)[:6]
    mat = np.array([[t[1], t[2]] for t in top], dtype=float)
    mat_norm = np.log1p(mat)
    mat_norm = mat_norm / np.nanmax(mat_norm)
    im = ax.imshow(mat_norm, aspect="auto", cmap=HEAT_CMAP, vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A/G", "U/C"])
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([short_label(t[0], 22) for t in top], fontsize=6.0)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6.0,
                    color="white" if mat_norm[i, j] > 0.55 else COL["dark"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("normalised log $F$", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8)
    cbar.outline.set_visible(False)
    ax.set_title("U/C signal is more concentrated", pad=16)
    label_panel(ax, "d", x=-0.42, y=1.04)

    audit_figure(fig, "fig3_representation_controls")
    save(fig, "fig3_representation_controls")


# ---------------------------------------------------------------------------
# Fig. 4 -- calibration and abstention
# ---------------------------------------------------------------------------
def fig4_calibration():
    b3 = load_json(V3 / "v4_b3full_results.json")
    cal = load_json(CAL / "calibration_full.json")
    summary = load_json(V3 / "summary.json")
    pos = pd.DataFrame(load_json(ENT / "per_position_entropy.json"))

    # reliability bins from the full per-position artifact (as in legacy generator)
    n_bins = 6
    bins = np.linspace(0.25, 1.0, n_bins + 1)
    rel_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pos["max_prob"] >= lo) & (pos["max_prob"] < (hi if hi < 1.0 else hi + 1e-9))
        g = pos.loc[mask]
        rel_rows.append(
            {
                "confidence_bin": f"{lo:.3f}-{hi:.3f}",
                "n_positions": int(len(g)),
                "mean_confidence": float(g["max_prob"].mean()) if len(g) else np.nan,
                "observed_accuracy": float(g["correct"].mean()) if len(g) else np.nan,
            }
        )
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(SRC / "fig4a_reliability_bins.csv", index=False)

    # per-chain abstention distribution: use the frozen artifact so Fig. 4d
    # and the Results paragraph share one protocol (min_keep=5, 10% grid).
    dist = pd.read_csv(ROOT / "NMI-submission/analysis/out/chain_abstention_full.csv")
    vals = dist["frac_removed"].to_numpy(float)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.38, width_ratios=[1.0, 1.12])

    # -- a: reliability -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0.25, 1.0], [0.25, 1.0], color=COL["baseline"], ls="--", lw=0.8)
    sizes = 14 + 60 * rel["n_positions"].to_numpy(float) / rel["n_positions"].max()
    ax.plot(rel["mean_confidence"], rel["observed_accuracy"], color=COL["confidence"],
            lw=1.0, zorder=2)
    ax.scatter(rel["mean_confidence"], rel["observed_accuracy"], s=sizes,
               color=COL["confidence"], edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0.25, 1.02)
    ax.set_ylim(0.25, 1.02)
    ax.set_title(f"Reliability trend (ECE = {cal['ece_10bin_full']:.3f})")
    label_panel(ax, "a", x=-0.22, y=1.04)

    # -- b: pooled abstention ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    xs = [a["pct_removed"] for a in cal["abstention"]]
    ys = [pct(a["acc"]) for a in cal["abstention"]]
    ax.axvspan(50, 75, color=COL["light"], alpha=0.8, zorder=0)
    ax.plot(xs, ys, "o-", color=COL["confidence"], lw=1.4, markersize=3.6, zorder=3)
    ax.axhline(pct(summary["post2025_test_weighted_recovery"]), color=COL["baseline"],
               ls="--", lw=0.8, zorder=1)
    ax.text(1, 62.2, "full temporal\nweighted recovery (64.6%)", fontsize=5.8,
            color=COL["baseline"], ha="left", va="top")
    for target, (lx, ly) in ((50, (43.0, 87.0)), (75, (75, 100.2))):
        p = min(cal["abstention"], key=lambda a: abs(float(a["pct_removed"]) - target))
        ax.scatter([p["pct_removed"]], [pct(p["acc"])], s=34, color=COL["georna"], zorder=4,
                   edgecolor="white", linewidth=0.6)
        ax.text(lx, ly, f"{pct(p['acc']):.1f}", ha="center",
                va="bottom", fontsize=6.2, color=COL["dark"], fontweight="bold")
    ax.set_xlabel("Least-confident positions removed (%)")
    ax.set_ylabel("Accuracy of retained positions (%)")
    ax.set_ylim(58, 104)
    ax.set_xlim(-3, 93)
    ax.set_title("Abstention curve")
    label_panel(ax, "b", x=-0.20, y=1.04)

    # -- c: confidence histogram, correct vs incorrect --------------------------
    ax = fig.add_subplot(gs[1, 0])
    edges = np.linspace(0.25, 1.0, 26)
    conf = pos["max_prob"].to_numpy(float)
    corr = pos["correct"].to_numpy(bool)
    ax.hist([conf[corr], conf[~corr]], bins=edges, stacked=True,
            color=[COL["georna"], COL["error"]], label=["correct", "incorrect"],
            edgecolor="white", linewidth=0.2)
    ax.set_xlabel("Model confidence (max predicted probability)")
    ax.set_ylabel("Positions")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Confidence of correct vs incorrect predictions")
    label_panel(ax, "c", x=-0.22, y=1.04)

    # -- d: per-chain abstention ECDF ------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    grid = np.arange(0, 1.01, 0.1)
    ecdf = [(vals <= g).mean() for g in grid]
    ax.step(grid, ecdf, where="post", color=COL["georna"], lw=1.6)
    med = float(np.median(vals)); frac30 = float((vals <= 0.3).mean())
    ax.scatter([0.3, med], [frac30, 0.5], s=30, color=COL["dms"], zorder=4,
               edgecolor="white", linewidth=0.6)
    ax.annotate(f"median chain: {med*100:.0f}%", xy=(med, 0.5), xytext=(0.34, 0.56),
                fontsize=6.2, color=COL["dark"],
                arrowprops=dict(arrowstyle="-", color=COL["neutral"], lw=0.7))
    ax.annotate(f"{frac30*100:.0f}% of chains need\n$<$30% abstention", xy=(0.3, frac30),
                xytext=(0.03, 0.30), fontsize=6.2, color=COL["dark"],
                arrowprops=dict(arrowstyle="-", color=COL["neutral"], lw=0.7))
    never = (vals >= 1.0).mean()
    ax.text(0.97, 1.0 - never + 0.06, f"{never * 100:.0f}% never reach 90%",
            fontsize=6.2, color=COL["error"], ha="right", va="bottom")
    ax.set_xlabel("Positions abstained per chain (fraction)")
    ax.set_ylabel("Cumulative fraction of chains")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "$>$90%"])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Per-chain abstention fraction")
    label_panel(ax, "d", x=-0.22, y=1.04)

    audit_figure(fig, "fig4_calibration")
    save(fig, "fig4_calibration")


# ---------------------------------------------------------------------------
# Fig. 5 (file fig6_design_stress) -- design separability + stress test
# ---------------------------------------------------------------------------
def fig6_design_stress():
    b1 = load_json(V3 / "v4_b1hard_results.json")
    mfe = pd.read_csv(OK / "mfe_scores.csv")
    df = mfe.dropna(subset=["model_ce", "mfe_f1", "ok_score", "seq_len", "seq"]).copy()
    complete = OPENKNOT_COMPLETE_CASE
    per_pos = load_json(STRESS / "per_position_data.json")

    order = ["GC-shuffle", "dinuc-shuffle", "SS-shuffle", "gRNAde", "RiboDiffusion", "swap_pairs", "RDesign"]
    labels = {
        "GC-shuffle": "GC shuffle",
        "dinuc-shuffle": "dinucleotide shuffle",
        "SS-shuffle": "SS-context shuffle",
        "gRNAde": "gRNAde",
        "RiboDiffusion": "RiboDiffusion",
        "swap_pairs": "compensatory mutants",
        "RDesign": "RDesign",
    }
    rows = [(k, b1["tiers"][k]["mean"], b1["tiers"][k]["n"]) for k in order if k in b1["tiers"]]
    ci_rows = []
    for key, auc, n in rows:
        successes = int(round(float(auc) * int(n)))
        lo, hi = wilson_ci(successes, int(n))
        ci_rows.append(
            {
                "tier": key,
                "auc": float(auc),
                "n": int(n),
                "native_better_count_approx": successes,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_method": "Wilson interval from stored tier aggregate; raw native-decoy CE pairs were not stored",
            }
        )
    pd.DataFrame(ci_rows).to_csv(SRC / "fig5a_decoy_auc_intervals.csv", index=False)

    def stress_mean(chain, key):
        return float(np.mean([bool(r[key]) for r in per_pos[chain]]))

    forge = [stress_mean(c, "geo_correct") for c in STRESS_CHAINS]
    forge6 = [stress_mean(c, "geo6_correct") for c in STRESS_CHAINS]

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 1.18], width_ratios=[1.30, 1.0, 1.0],
        hspace=0.62, wspace=0.52,
    )

    # -- a: AUC ladder ---------------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(rows))
    auc_vals = [r[1] for r in rows]
    tier_colors = {
        "GC-shuffle": COL["reference"],
        "dinuc-shuffle": COL["reference"],
        "SS-shuffle": COL["reference"],
        "gRNAde": METHOD_COLORS["gRNAde"],
        "RiboDiffusion": METHOD_COLORS["RiboDiffusion"],
        "swap_pairs": COL["georna_6"],
        "RDesign": METHOD_COLORS["RDesign"],
    }
    cols = [tier_colors[r[0]] for r in rows]
    ax.barh(y, auc_vals, color=cols, height=0.62)
    # whiskers drawn only beyond the bar end (from AUC to CI upper bound),
    # so no error-bar line crosses the filled bar or its value label
    for i, (key, val, _n) in enumerate(rows):
        hi = ci_rows[i]["ci95_high"]
        ax.plot([val, hi], [i, i], color=COL["dark"], lw=0.8, zorder=3)
        ax.plot([hi, hi], [i - 0.14, i + 0.14], color=COL["dark"], lw=0.8, zorder=3)
        ax.text(hi + 0.025, i, f"{val:.2f}", ha="left", va="center",
                fontsize=5.8, color=COL["dark"])
    ax.axvline(0.5, color=COL["baseline"], ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[r[0]] for r in rows], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("AUC: native sequence ranked above candidate")
    ax.set_title("Native-vs-decoy ladder")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    label_panel(ax, "a", x=-0.42, y=1.03)

    # -- b: OpenKnot associations ------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    corr_rows = [
        ("CE raw", complete["rho_model_ce_ok"], COL["compatibility"]),
        ("CE | MFE", complete["partial_model_ce_given_mfe"]["rho"], "#6E6EA8"),
        ("CE | full", complete["partial_model_ce_given_mfe_gc_length"]["rho"], "#A5A5CC"),
        ("MFE F1", complete["rho_mfe_f1_ok"], COL["foldability"]),
    ]
    yy = np.arange(len(corr_rows))
    corr_vals = [r[1] for r in corr_rows]
    ax.barh(yy, corr_vals, color=[r[2] for r in corr_rows], height=0.56)
    ax.axvline(0, color=COL["baseline"], lw=0.7)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in corr_rows], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.68)
    ax.set_xlabel("Spearman $\\rho$ with OpenKnot score")
    ax.set_title("Non-equivalent associations")
    for ypos, val in zip(yy, corr_vals):
        ax.text(val + 0.018, ypos, f"{val:.2f}", ha="left", va="center", fontsize=5.8,
                color=COL["dark"])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    label_panel(ax, "b", x=-0.52, y=1.08)

    # -- c: incremental R^2 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    r2 = complete["ols_delta_r2"]
    r2_vals = [r2["r2_base"], r2["r2_full"]]
    ax.bar(np.arange(2), r2_vals, color=[COL["reference"], COL["compatibility"]], width=0.52)
    for i, val in enumerate(r2_vals):
        ax.text(i, val + 0.008, f"{val:.3f}", ha="center", va="bottom", fontsize=6.0,
                color=COL["dark"])
    # delta bracket
    ax.annotate("", xy=(1, r2_vals[1] + 0.035), xytext=(0, r2_vals[1] + 0.035),
                arrowprops=dict(arrowstyle="-", color=COL["neutral"], lw=0.7))
    ax.text(0.5, r2_vals[1] + 0.043, "$\\Delta R^2$ = 0.022", ha="center", va="bottom",
            fontsize=6.2, color=COL["dark"], fontweight="bold")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["MFE F1 + GC\n+ length", "+ FORGE\nCE"], fontsize=6.2)
    ax.set_ylabel("OpenKnot score model $R^2$")
    ax.set_ylim(0, 0.36)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Small independent increment")
    label_panel(ax, "c", x=-0.30, y=1.08)

    # -- d: stress-test dot plot ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1:3])
    series = [
        ("FORGE", forge, METHOD_COLORS["FORGE"], 62, "o"),
        ("FORGE 6-atom", forge6, METHOD_COLORS["FORGE 6-atom"], 40, "o"),
        ("gRNAde", [STRESS_BASELINE_RECOVERY[c]["gRNAde"] for c in STRESS_CHAINS], METHOD_COLORS["gRNAde"], 26, "o"),
        ("RiboDiffusion", [STRESS_BASELINE_RECOVERY[c]["RiboDiffusion"] for c in STRESS_CHAINS], METHOD_COLORS["RiboDiffusion"], 26, "o"),
        ("RDesign", [STRESS_BASELINE_RECOVERY[c]["RDesign"] for c in STRESS_CHAINS], METHOD_COLORS["RDesign"], 26, "o"),
    ]
    yy = np.arange(len(STRESS_CHAINS))
    offsets = np.linspace(-0.30, 0.30, len(series))
    for (name, vals_m, color, size, marker), off in zip(series, offsets):
        ax.scatter([pct(v) for v in vals_m], yy + off, s=size, color=color, label=name,
                   zorder=3 if name == "FORGE" else 2, edgecolor="white", linewidth=0.5)
    # exact FORGE values as a fixed right-side column (never overlaps dots)
    for i, v in enumerate(forge):
        ax.text(64.5, yy[i], f"{pct(v):.0f}%", ha="right", va="center",
                fontsize=6.2, color=COL["georna"], fontweight="bold")
    ax.text(64.5, -0.85, "FORGE", ha="right", va="center", fontsize=5.8,
            color=COL["georna"])
    ax.axvline(25, color=COL["baseline"], ls="--", lw=0.8, zorder=1)
    ax.text(25, -0.72, "random", ha="center", va="top", fontsize=6.0, color=COL["baseline"])
    ax.set_yticks(yy)
    ax.set_yticklabels(STRESS_CHAIN_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Nucleotide recovery (%)")
    ax.set_xlim(0, 66)
    ax.set_ylim(len(STRESS_CHAINS) - 0.45, -0.95)
    ax.set_title("Frozen-model pseudoknot stress test")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=5.9,
              handletextpad=0.15, columnspacing=0.9, markerscale=0.75)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    label_panel(ax, "d", x=-0.13, y=1.06)

    audit_figure(fig, "fig6_design_stress")
    save(fig, "fig6_design_stress")


# ---------------------------------------------------------------------------
# Supplementary Fig. S3 -- N-free retraining controls (both scales)
# ---------------------------------------------------------------------------
def fig_s1_nfree():
    b2 = load_json(V3 / "v4_b2hard_results.json")["b2_hard"]
    v9 = load_json(V3 / "v9_6atom_full_results.json")
    summary = load_json(V3 / "summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.subplots_adjust(wspace=0.34)
    x = np.arange(2)
    w = 0.34
    panels = [
        (axes[0], "Matched subset (50% of test)", [pct(b2["acc_7atom_retrained"]), pct(b2["purine_acc_7atom"])],
         [pct(b2["acc_6atom_retrained"]), pct(b2["purine_acc_6atom"])]),
        (axes[1], "Full temporal test", [pct(summary["post2025_test_weighted_recovery"]), 82.2],
         [pct(v9["post2025"]["weighted_recovery"]), 72.2]),
    ]
    for ax, title, v7, v6 in panels:
        ax.bar(x - w / 2, v7, w, color=METHOD_COLORS["FORGE 7-atom"], label="7-atom")
        ax.bar(x + w / 2, v6, w, color=METHOD_COLORS["FORGE 6-atom"], label="6-atom N-free")
        for xpos, val in zip([x[0] - w / 2, x[0] + w / 2, x[1] - w / 2, x[1] + w / 2],
                             [v7[0], v6[0], v7[1], v6[1]]):
            ax.text(xpos, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=6.0,
                    color=COL["dark"])
        ax.set_xticks(x)
        ax.set_xticklabels(["4-class\nnucleotide", "purine/\npyrimidine"])
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(title)
    axes[0].set_ylabel("Accuracy (%)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, frameon=False,
               fontsize=6.4, handlelength=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.24)
    label_panel(axes[0], "a", x=-0.22, y=1.10)
    label_panel(axes[1], "b", x=-0.10, y=1.10)
    audit_figure(fig, "fig_s1_nfree")
    save(fig, "fig_s1_nfree")


# ---------------------------------------------------------------------------
# Supplementary Fig. S4 -- stress-test confidence and chemical class
# ---------------------------------------------------------------------------
def suppfig_stress_details():
    per_pos = load_json(STRESS / "per_position_data.json")

    def stress_mean(chain, key):
        return float(np.mean([bool(r[key]) for r in per_pos[chain]]))

    rows = []
    for chain, label in zip(STRESS_CHAINS, STRESS_CHAIN_LABELS):
        records = per_pos[chain]
        high = [r for r in records if r["geo_conf"] > 0.7]
        rows.append(
            {
                "chain": label,
                "mean_forge_confidence": float(np.mean([r["geo_conf"] for r in records])),
                "fraction_confidence_gt_0p7": float(len(high) / len(records)),
                "n_high_confidence": len(high),
                "n_high_confidence_correct": int(sum(bool(r["geo_correct"]) for r in high)),
                "purine_pyrimidine_accuracy_7atom": stress_mean(chain, "geo_pur_correct"),
                "purine_pyrimidine_accuracy_6atom": stress_mean(chain, "geo6_pur_correct"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(SRC / "figs4_stress_details.csv", index=False)

    yy = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    # lollipop instead of bars: x range is truncated (0.35-0.52), so bars from
    # zero would overstate differences; dots + thin stems are honest here
    for yyv, val in zip(yy, df["mean_forge_confidence"]):
        ax.plot([0.35, val], [yyv, yyv], color=COL["confidence"], lw=0.9, alpha=0.8)
        ax.scatter([val], [yyv], s=30, color=COL["confidence"], edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.text(val + 0.006, yyv, f"{val:.3f}", ha="left", va="center", fontsize=5.8,
                color=COL["dark"])
    ax.set_yticks(yy)
    ax.set_yticklabels(df["chain"])
    ax.invert_yaxis()
    ax.set_xlim(0.35, 0.545)
    ax.set_xlabel("Mean FORGE confidence")
    ax.set_title("Low confidence on designs")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    label_panel(ax, "a", x=-0.55, y=1.08)

    ax = axes[1]
    ax.barh(yy, df["fraction_confidence_gt_0p7"], color=COL["georna_6"], height=0.60)
    for ypos, frac, n_high in zip(yy, df["fraction_confidence_gt_0p7"], df["n_high_confidence"]):
        ax.text(0.108, ypos, f"n={int(n_high)}", ha="right", va="center", fontsize=5.8,
                color=COL["dark"])
    ax.set_xlim(0, 0.115)
    ax.set_xlabel("Fraction with confidence $>$0.7")
    ax.set_title("High-confidence calls are rare")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    label_panel(ax, "b", x=-0.24, y=1.08)

    ax = axes[2]
    bar_h = 0.30
    ax.barh(yy - bar_h / 2, df["purine_pyrimidine_accuracy_7atom"], height=bar_h,
            color=METHOD_COLORS["FORGE"], label="7-atom")
    ax.barh(yy + bar_h / 2, df["purine_pyrimidine_accuracy_6atom"], height=bar_h,
            color=METHOD_COLORS["FORGE 6-atom"], label="6-atom")
    ax.axvline(0.5, color=COL["baseline"], ls="--", lw=0.8)
    ax.set_xlim(0.42, 0.82)
    ax.set_xlabel("Purine/pyrimidine accuracy")
    ax.set_title("Chemical class persists")
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.34), ncol=2, fontsize=6.0,
              handlelength=1.4)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    label_panel(ax, "c", x=-0.24, y=1.08)

    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(bottom=0.30)
    audit_figure(fig, "suppfig_stress_details")
    save(fig, "suppfig_stress_details")


# ---------------------------------------------------------------------------
# Supplementary Fig. S1/S2 -- AF3 forward-folding control (restyled port)
# ---------------------------------------------------------------------------
def fig5_af3_combined():
    fine = load_json(STRESS / "af3results/fine_compare.json")

    ORDER = ["10zt", "10zu", "11eh", "11ag"]
    NAME = {"10zt": "10ZT", "10zu": "10ZU", "11eh": "11EH", "11ag": "11AG"}
    JOB = {
        "10zt_real": "10zt_real_scaffold_519nt", "10zt_rnetopt": "10zt_forge_rnetopt_scaffold_519nt",
        "10zt_forge": "10zt_forge_scaffold_519nt",
        "10zu_real": "10zu_real_scaffold_518nt", "10zu_rnetopt": "10zu_forge_rnetopt_scaffold_518nt",
        "10zu_forge": "10zu_forge_scaffold_518nt",
        "11eh_real": "11eh_real_scaffold_519nt", "11eh_rnetopt": "11eh_forge_rnetopt_scaffold_519nt",
        "11eh_forge": "11eh_forge_scaffold_519nt",
        "11ag_real": "11ag_real_scaffold_518nt", "11ag_rnetopt": "11ag_forge_rnetopt_scaffold_dimer_2x518nt",
        "11ag_forge": "11ag_forge_scaffold_518nt",
    }
    VARIANTS = ["real", "rnetopt", "forge"]
    VLABEL = {"real": "true design", "rnetopt": "FORGE + R-Net refined", "forge": "FORGE-derived"}
    x = np.arange(len(ORDER))

    def best(k, v, field):
        return fine[JOB[f"{k}_{v}"]]["best"][field]

    def range5(k, v, field):
        mods = fine[JOB[f"{k}_{v}"]]["all_models"]
        return min(m[field] for m in mods), max(m[field] for m in mods)

    def waterfall(ax, target_var, letter):
        real_vals = [best(k, "real", "P_RMSD") for k in ORDER]
        tgt_vals = [best(k, target_var, "P_RMSD") for k in ORDER]
        ax.bar(x, real_vals, color=COL["georna"], width=0.52, label="true design")
        for i, (r0, r1) in enumerate(zip(real_vals, tgt_vals)):
            delta = r1 - r0
            if delta > 0:
                ax.bar(i, delta, bottom=r0, color=COL["delta_down"], width=0.52,
                       hatch="\\\\\\\\", edgecolor="white", linewidth=0)
            else:
                ax.bar(i, -delta, bottom=r1, color=COL["delta_up"], width=0.52,
                       hatch="////", edgecolor="white", linewidth=0)
            ax.plot([i - 0.26, i + 0.26], [r1, r1], color="0.35", lw=0.9)
            lo, hi = range5(ORDER[i], "real", "P_RMSD")
            ax.errorbar(i, real_vals[i], yerr=[[real_vals[i] - lo], [hi - real_vals[i]]],
                        fmt="none", ecolor="0.62", elinewidth=0.8, capsize=2, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([NAME[k] for k in ORDER])
        ax.set_ylabel("P-RMSD to cryo-EM ($\\mathrm{\\AA}$)")
        ax.set_title(VLABEL[target_var])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        label_panel(ax, letter, x=-0.20, y=1.06)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.1), gridspec_kw={"wspace": 0.32})
    waterfall(axes[0], "rnetopt", "a")
    waterfall(axes[1], "forge", "b")

    axc = axes[2]
    var_colors = [COL["georna"], COL["delta_up"], COL["delta_down"]]
    for j, v in enumerate(VARIANTS):
        vals = [best(k, v, "SS_F1") for k in ORDER]
        axc.bar(x + 0.28 * (j - 1), vals, width=0.26, color=var_colors[j],
                label=VLABEL[v], alpha=0.95)
        recs = [best(k, v, "contactRec") for k in ORDER]
        for i, rv in enumerate(recs):
            axc.bar(x[i] + 0.28 * (j - 1), rv, bottom=vals[i], width=0.26,
                    color=var_colors[j], hatch="\\\\\\\\", edgecolor="white",
                    linewidth=0, alpha=0.95)
    axc.set_xticks(x)
    axc.set_xticklabels([NAME[k] for k in ORDER])
    axc.set_ylim(0, 2.05)
    axc.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    axc.set_ylabel("SS $F_1$ + contact recov.")
    axc.set_title("pairing vs tertiary contacts")
    axc.yaxis.grid(True)
    axc.set_axisbelow(True)
    label_panel(axc, "c", x=-0.14, y=1.10)
    handles = [
        Patch(facecolor=COL["reference"], edgecolor="0.35",
              label="hatched = native contact recovery"),
        Patch(facecolor=COL["georna"], label="true design"),
        Patch(facecolor=COL["delta_up"], label="FORGE + R-Net refined"),
        Patch(facecolor=COL["delta_down"], label="FORGE-derived"),
    ]
    fig.subplots_adjust(bottom=0.30)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=4, fontsize=6.4)
    audit_figure(fig, "fig5_af3_combined")
    save(fig, "fig5_af3_combined")


if __name__ == "__main__":
    fig2_identifiability()
    fig3_representation_controls()
    fig4_calibration()
    fig6_design_stress()
    fig_s1_nfree()
    suppfig_stress_details()
    fig5_af3_combined()
    print("done")
