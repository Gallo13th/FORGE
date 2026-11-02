"""Figure 7: SHAP feature importance — global top-20 bar chart + per-class top-5 table."""
from paper_plot_style import *
import csv, os

# Load SHAP importance
shap_file = 'RNKDB/results/CL00111_outside_shap_shap_importance.csv'
features = []
importances = []
with open(shap_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        features.append(row['feature'])
        importances.append(float(row['importance']))

# Top 20
top_n = 20
top_features = features[:top_n]
top_importances = importances[:top_n]

fig, ax = plt.subplots(figsize=(7, 5.5))

y_pos = range(top_n)
bars = ax.barh(y_pos, top_importances[::-1], color=COLORS[0], edgecolor='white', linewidth=0.5)

# Feature names (reversed for horizontal bar)
feat_names = [f.replace('_', r'\_') for f in top_features[::-1]]
ax.set_yticks(y_pos)
ax.set_yticklabels(feat_names, fontsize=FONT_SIZE-3)
ax.set_xlabel('Mean |SHAP| Value', fontsize=FONT_SIZE-1)
ax.invert_yaxis()

# Value labels
for bar, val in zip(bars, top_importances[::-1]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=FONT_SIZE-4)

ax.set_title('Figure 7: Top-20 SHAP feature importance (global)',
             fontsize=FONT_SIZE+1, fontweight='bold')

save_fig(fig, 'fig7_shap_importance')
