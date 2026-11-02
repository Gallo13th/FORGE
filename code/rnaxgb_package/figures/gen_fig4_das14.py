"""Figure 4: Das 14 leakage analysis + baseline comparison."""
from paper_plot_style import *

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# --- Left panel: Leakage analysis ---
model_variants = ['model.pkl\n(trained on all\npre-2025)',
                  'das_split\n(Das clusters\nexcluded)',
                  'das_cluster\nexcluded\n(post-2025 incl)']
clean_acc = [53.3, 55.7, 50.1]
leaked_acc = [96.5, 59.5, 49.3]

x = np.arange(len(model_variants))
width = 0.35

bars_clean = ax1.bar(x - width/2, clean_acc, width, label='Clean (unseen chains)',
                     color=COLORS[0], edgecolor='white', linewidth=0.5)
bars_leaked = ax1.bar(x + width/2, leaked_acc, width, label='Leaked (seen chains)',
                      color=COLORS[2], edgecolor='white', linewidth=0.5)

for bar, val in zip(bars_clean, clean_acc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
             ha='center', va='bottom', fontsize=FONT_SIZE-3, fontweight='bold')
for bar, val in zip(bars_leaked, leaked_acc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
             ha='center', va='bottom', fontsize=FONT_SIZE-3, fontweight='bold')

# Leakage gap annotations
gaps = [43.2, 3.8, -0.8]
for i, gap in enumerate(gaps):
    y = max(clean_acc[i], leaked_acc[i]) + 10
    ax1.annotate(f'Gap: {gap:+.1f}pp', xy=(x[i], y), ha='center',
                fontsize=FONT_SIZE-2, fontweight='bold',
                color=COLORS[2] if gap > 10 else GRAY)

ax1.set_ylabel('Accuracy (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(model_variants, fontsize=FONT_SIZE-2)
ax1.legend(frameon=False, fontsize=FONT_SIZE-2)
ax1.set_ylim(0, 112)
ax1.set_title('Das 14: Leakage-Controlled Evaluation', fontweight='bold')

# --- Right panel: Published baselines comparison ---
methods = ['RiboSphere', 'XGBoost\n(ours, clean)', 'RIDiffusion', 'gRNAde', 'RDesign']
accs = [63.0, 55.7, 53.3, 52.9, 41.6]
method_colors = [COLORS[4], COLORS[0], COLORS[3], COLORS[2], COLORS[1]]

bars = ax2.barh(methods, accs, color=method_colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, accs):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
             va='center', fontsize=FONT_SIZE-2, fontweight='bold')

ax2.axvline(x=25.0, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
ax2.text(25.5, -0.6, 'Random', fontsize=FONT_SIZE-2, color=GRAY)
ax2.set_xlabel('Sequence Recovery (%)')
ax2.set_xlim(0, 75)
ax2.set_title('Das 14 Benchmark (Clean)', fontweight='bold')

fig.suptitle('Figure 4: Das 14 cluster benchmark — leakage and baseline comparison.',
             fontsize=FONT_SIZE+1, fontweight='bold', y=1.01)
save_fig(fig, 'fig4_das14_analysis')
