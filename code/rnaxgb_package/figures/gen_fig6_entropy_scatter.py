"""Figure 6: Model CE vs Rfam entropy scatter — multi-panel."""
from paper_plot_style import *
import json, os

# Load correlation data
corr_file = 'RNKDB/entropy_analysis/rfam_comparison/correlation_summary.json'
with open(corr_file) as f:
    corr_data = json.load(f)

# Families with data and their Pearson r
families = ['5_8S_rRNA', '5S_rRNA', 'tRNA', 'RNaseP_bact_b', 'Intron_gpI',
            'GOLLD', 'ROOL', 'Purine', 'K10_TLS']

pearson_r = []
for fam in families:
    r = corr_data.get(fam, {}).get('pearson_r', None)
    pearson_r.append(r if r is not None else float('nan'))

# Load column-level data for scatter plots
data_dir = 'RNKDB/entropy_analysis'
col_dir = os.path.join(data_dir, 'rfam_comparison')

# We'll use the existing combined_correlation.png as reference
# and generate simplified scatter plots from the column correlation files

fig, axes = plt.subplots(3, 3, figsize=(10, 9))
axes = axes.flatten()

for idx, (fam, ax) in enumerate(zip(families, axes)):
    col_file = os.path.join(col_dir, f'{fam}_column_correlation.json')
    if os.path.exists(col_file):
        with open(col_file) as f:
            col_data = json.load(f)
        rfam_ent = np.array([float(d['rfam_entropy'] or 0.0) for d in col_data])
        model_ce = np.array([float(d['mean_model_ce'] or 0.0) for d in col_data])

        ax.scatter(rfam_ent, model_ce, alpha=0.4, s=8, color=COLORS[0], edgecolors='none')
        ax.set_title(f'{fam} (r={pearson_r[idx]:.3f})', fontsize=FONT_SIZE-1, fontweight='bold')

        # Trend line
        mask = ~np.isnan(rfam_ent) & ~np.isnan(model_ce)
        if mask.sum() > 2:
            from numpy.polynomial.polynomial import polyfit
            try:
                b, m = polyfit(rfam_ent[mask], model_ce[mask], 1)
                xs = np.linspace(rfam_ent[mask].min(), rfam_ent[mask].max(), 50)
                ax.plot(xs, b + m * xs, color=COLORS[2], linewidth=1, alpha=0.6)
            except Exception:
                pass
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                color=GRAY, fontsize=FONT_SIZE-1)
        ax.set_title(f'{fam} (N/A)', fontsize=FONT_SIZE-1, fontweight='bold')

    ax.set_xlabel('Rfam Column Entropy', fontsize=FONT_SIZE-3)
    ax.set_ylabel('Model CE', fontsize=FONT_SIZE-3)

fig.suptitle('Figure 6: Per-position model cross-entropy vs. Rfam evolutionary entropy',
             fontsize=FONT_SIZE+2, fontweight='bold', y=1.01)

plt.tight_layout()
save_fig(fig, 'fig6_entropy_scatter')
