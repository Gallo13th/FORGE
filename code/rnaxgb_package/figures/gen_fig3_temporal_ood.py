"""Figure 3: Temporal OOD comparison — all models on post-2025 OOD."""
from paper_plot_style import *

# Data from paper draft §3.3
models = ['XGBoost\n(ours)', 'RDesign', 'RiboDiffusion', 'gRNAde']
accuracy_2025 = [61.1, 41.3, 32.5, 31.2]
accuracy_2026 = [66.7, 41.8, 33.2, 26.9]
accuracy_overall = [63.11, 41.44, 32.79, 29.67]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(7, 4))

bars1 = ax.bar(x - width, accuracy_2025, width, label='2025 releases',
               color=COLORS[0], edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, accuracy_2026, width, label='2026 releases',
               color=COLORS[1], edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, accuracy_overall, width, label='Overall',
               color=COLORS[4], edgecolor='white', linewidth=0.5)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=FONT_SIZE-3, fontweight='bold')

ax.axhline(y=25.0, color=GRAY, linestyle='--', linewidth=1, alpha=0.7)
ax.text(4.1, 25.0, 'Random (25%)', fontsize=FONT_SIZE-2, color=GRAY, va='bottom', ha='right')

ax.set_ylabel('Mean Sequence Recovery (%)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(frameon=False, loc='upper right')
ax.set_ylim(0, 78)

ax.set_title('Figure 3: All models on post-2025 OOD test (4,135 chains, 286,698 nt)',
             fontsize=FONT_SIZE+1, fontweight='bold')

save_fig(fig, 'fig3_temporal_ood')
