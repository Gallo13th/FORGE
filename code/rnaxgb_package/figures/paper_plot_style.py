"""Shared matplotlib style for paper figures — ICLR 2026 submission."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

FONT_SIZE = 10
DPI = 300
FORMAT = 'pdf'
FIG_DIR = 'figures'

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

# Colorblind-safe palette (IBM / Wong, 2011)
COLORS = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#009E73']
GRAY = '#888888'

# Model color mapping
MODEL_COLORS = {
    'XGBoost (ours)': COLORS[0],
    'gRNAde': COLORS[2],
    'RiboDiffusion': COLORS[3],
    'RDesign': COLORS[1],
    'RiboSphere': COLORS[4],
}

# Per-class colors
CLASS_COLORS = {'A': '#648FFF', 'C': '#785EF0', 'G': '#DC267F', 'U': '#FE6100'}


def save_fig(fig, name, fmt=FORMAT):
    """Save figure to FIG_DIR with consistent naming."""
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f'{name}.{fmt}')
    fig.savefig(path)
    print(f'  Saved: {path}')
    plt.close(fig)


def set_figsize(width_ratio=0.48, aspect=0.75):
    """Set figure size for LaTeX column widths.

    Args:
        width_ratio: 0.48 for single column, 0.95 for full width
        aspect: height/width ratio
    """
    # ICLR text width ≈ 5.5 inches for single column
    textwidth = 5.5
    width = textwidth * width_ratio / 0.48  # normalize to actual inches
    height = width * aspect
    return plt.subplots(figsize=(width, height))
