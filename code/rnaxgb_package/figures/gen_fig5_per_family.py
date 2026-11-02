"""Figure 5: Per-family accuracy heatmap — all models × 11 families."""
from paper_plot_style import *
import json, os

# Data from paper draft §3.6 + summary.json
families = ['5_8S_rRNA', '5S_rRNA', 'tRNA', 'Fungi_U3', 'AdoCbl-II',
            'Intron_gpI', 'RNaseP_bact_b', 'ROOL', 'GOLLD', 'Purine', 'K10_TLS']

# Per-family accuracy: [XGBoost, RDesign, gRNAde, RiboDiffusion]
data = np.array([
    [94.4, 41.9, 27.8, 50.9],
    [88.1, 68.0, 27.4, 53.6],
    [75.7, 61.2, 56.0, 39.6],
    [72.1, 46.9, 27.4, 32.3],
    [64.4, 58.6, 38.1, 25.0],
    [58.0, 41.4, 26.2, 24.2],
    [49.9, 43.9, 18.2, 18.4],
    [43.3, 41.7, 27.5, 26.2],
    [43.1, 41.1, 24.9, 26.1],
    [37.4, 38.2, 30.2, 26.6],
    [29.0, 15.9, 42.9, 42.9],
])

model_names = ['XGBoost\n(ours)', 'RDesign', 'gRNAde', 'RiboDiffusion']

fig, ax = plt.subplots(figsize=(7.5, 5))

im = ax.imshow(data.T, cmap='RdYlGn', vmin=15, vmax=95, aspect='auto')

ax.set_xticks(range(len(families)))
ax.set_xticklabels(families, rotation=45, ha='right', fontsize=FONT_SIZE-2)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=FONT_SIZE-2)

# Add text annotations
for i in range(len(families)):
    for j in range(len(model_names)):
        val = data[i, j]
        color = 'white' if val < 40 or val > 75 else 'black'
        ax.text(i, j, f'{val:.1f}', ha='center', va='center',
                fontsize=FONT_SIZE-3, fontweight='bold', color=color)

# Bold the max per family
for i in range(len(families)):
    best_j = np.argmax(data[i])
    ax.add_patch(plt.Rectangle((i - 0.5, best_j - 0.5), 1, 1,
                                fill=False, edgecolor='black', linewidth=2.5))

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('Sequence Recovery (%)', fontsize=FONT_SIZE-1)

ax.set_title('Figure 5: Per-family accuracy (post-2025 OOD, Rfam-annotated chains)',
             fontsize=FONT_SIZE+1, fontweight='bold')

# Add chain count annotations
chain_counts = [146, 365, 540, 11, 29, 12, 78, 39, 32, 22, 12]
for i, (fam, cnt) in enumerate(zip(families, chain_counts)):
    ax.text(i, 4.5, f'n={cnt}', ha='center', fontsize=FONT_SIZE-4, color=GRAY)

save_fig(fig, 'fig5_per_family_heatmap')
