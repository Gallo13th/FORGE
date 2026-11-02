"""
Fig 10: Chemical information ceiling — Nature style (no text walls).
Clean visual: schematic molecules + data panels.
"""
import sys
sys.path.insert(0, '/home/glt/rnaxgb/Nature-comms-v2/figures')
from paper_plot_style import *
import json, numpy as np

BASE = '/home/glt/rnaxgb/RNKDB'
with open(f'{BASE}/fair_eval_v3/feature_information_analysis.json') as f:
    fi = json.load(f)

fig = plt.figure(figsize=(8, 6.5))

# ── Left: Chemical schematic (clean, minimal text) ──
ax_s = fig.add_axes([0.02, 0.05, 0.30, 0.92])
ax_s.set_xlim(0, 10)
ax_s.set_ylim(0, 14)
ax_s.axis('off')

# Purine pair (top)
# Simplified as colored blocks, not text walls
ax_s.text(5, 13.5, 'Purines', ha='center', fontsize=10, fontweight='bold', color=PURINE_COLOR)

# A block — just molecule name and key difference
a_rect = plt.Rectangle((0.3, 9.5), 4.3, 3.5, facecolor='#E8F0FE', edgecolor=XGB_COLOR, lw=1.5, alpha=0.6)
ax_s.add_patch(a_rect)
ax_s.text(2.45, 12.5, 'Adenine', ha='center', fontsize=9, fontweight='bold', color=XGB_COLOR)
ax_s.text(2.45, 11.7, 'N6-amine', ha='center', fontsize=8)
ax_s.text(2.45, 11.0, '(discriminating atom)', ha='center', fontsize=6, color='#666666', fontstyle='italic')

# G block
g_rect = plt.Rectangle((5.4, 9.5), 4.3, 3.5, facecolor='#FEE8E8', edgecolor=GRNADE_COLOR, lw=1.5, alpha=0.6)
ax_s.add_patch(g_rect)
ax_s.text(7.55, 12.5, 'Guanine', ha='center', fontsize=9, fontweight='bold', color=GRNADE_COLOR)
ax_s.text(7.55, 11.7, 'O6-carbonyl', ha='center', fontsize=8)
ax_s.text(7.55, 11.0, '(discriminating atom)', ha='center', fontsize=6, color='#666666', fontstyle='italic')

# Ambiguity arrow
ax_s.annotate('', xy=(5, 10.5), xytext=(5.5, 11.5),
              arrowprops=dict(arrowstyle='<->', color='#888888', lw=1.5))
ax_s.text(4.7, 11, 'A↔G', fontsize=7, color='#888888', ha='center')
ax_s.text(5, 9.8, '23.6% confusion', ha='center', fontsize=7, color=GRNADE_COLOR)

# 7-atom note
ax_s.text(5, 8.8, '7-atom backbone: sees only N9', ha='center', fontsize=7, color='#FF6600', fontweight='bold')

# Pyrimidine pair (bottom)
ax_s.text(5, 7.5, 'Pyrimidines', ha='center', fontsize=10, fontweight='bold', color=PYRIMIDINE_COLOR)

u_rect = plt.Rectangle((0.3, 3.5), 4.3, 3.5, facecolor='#E8F0FE', edgecolor=U_COLOR, lw=1.5, alpha=0.6)
ax_s.add_patch(u_rect)
ax_s.text(2.45, 6.5, 'Uridine', ha='center', fontsize=9, fontweight='bold', color=U_COLOR)
ax_s.text(2.45, 5.7, 'O4-carbonyl', ha='center', fontsize=8)
ax_s.text(2.45, 5.0, '(discriminating atom)', ha='center', fontsize=6, color='#666666', fontstyle='italic')

c_rect = plt.Rectangle((5.4, 3.5), 4.3, 3.5, facecolor='#FEE8E8', edgecolor=C_COLOR, lw=1.5, alpha=0.6)
ax_s.add_patch(c_rect)
ax_s.text(7.55, 6.5, 'Cytidine', ha='center', fontsize=9, fontweight='bold', color=C_COLOR)
ax_s.text(7.55, 5.7, 'N4-amine', ha='center', fontsize=8)
ax_s.text(7.55, 5.0, '(discriminating atom)', ha='center', fontsize=6, color='#666666', fontstyle='italic')

ax_s.annotate('', xy=(5, 4.5), xytext=(5.5, 5.5),
              arrowprops=dict(arrowstyle='<->', color='#888888', lw=1.5))
ax_s.text(4.7, 5, 'U↔C', fontsize=7, color='#888888', ha='center')
ax_s.text(5, 3.8, '18.1% confusion', ha='center', fontsize=7, color=GRNADE_COLOR)
ax_s.text(5, 2.8, '7-atom backbone: sees only N1', ha='center', fontsize=7, color='#FF6600', fontweight='bold')

# Summary box at bottom
summary_rect = plt.Rectangle((0.3, 0.3), 9.4, 2.2, facecolor='#FFF9E0', edgecolor='#FFAA00', lw=1.5, alpha=0.5)
ax_s.add_patch(summary_rect)
ax_s.text(5, 2.1, '7-atom backbone IS the information bottleneck', ha='center',
         fontsize=8, fontweight='bold', color='#FF6600')
ax_s.text(5, 1.5, 'No architecture can overcome missing base-edge atoms', ha='center',
         fontsize=7, color='#666666')
ax_s.text(5, 1.0, '→ Full-atom models | Multi-state ensembles | Evolutionary hybrids', ha='center',
         fontsize=7, fontweight='bold')

# ── Right top: Per-class accuracy bar chart ──
ax_b = fig.add_axes([0.38, 0.52, 0.28, 0.45])
classes = ['A', 'C', 'G', 'U']
class_acc = [fi['per_class_ce'][c]['accuracy'] for c in classes]
class_ce = [fi['per_class_ce'][c]['mean_ce'] for c in classes]
bar_colors = [CLASS_COLORS[c] for c in classes]

x = np.arange(len(classes))
bars = ax_b.bar(x, class_acc, 0.5, color=bar_colors, edgecolor='white', lw=0.5)
for i, (acc, ce) in enumerate(zip(class_acc, class_ce)):
    ax_b.text(i, acc/2, f'{acc:.0%}', ha='center', fontweight='bold', fontsize=8, color='white')
    ax_b.text(i, acc + 0.015, f'CE={ce:.2f}', ha='center', fontsize=6, color='#666666')

ax_b.set_xticks(x); ax_b.set_xticklabels(classes)
ax_b.set_ylabel('Accuracy', fontsize=8)
ax_b.set_ylim(0, 1.0)
ax_b.axhline(0.25, color=RANDOM_COLOR, ls='--', alpha=0.5, lw=0.8)
ax_b.text(3.3, 0.26, 'Random', fontsize=6, color=RANDOM_COLOR)
ax_b.text(1.5, 0.98, 'Per-class accuracy', ha='center', fontsize=9, fontweight='bold',
         transform=ax_b.transAxes)

# ── Right bottom: 1-NN degradation ──
ax_c = fig.add_axes([0.38, 0.05, 0.28, 0.40])
nn = fi['1nn_by_CE_bin']
bins = list(nn.keys())
accs = [nn[b]['mean_accuracy'] for b in bins]
ns = [nn[b]['n'] for b in bins]
labels = ['0–0.3', '0.3–0.6', '0.6–1.0', '1.0–1.5', '1.5+']

# Gradient color from easy to hard
bar_cols = ['#44AA44', '#88CC66', '#CCCC44', '#DDAA44', '#DD4444']
bx = np.arange(len(bins))
ax_c.bar(bx, accs, 0.55, color=bar_cols, edgecolor='white', lw=0.5)
ax_c.set_xticks(bx); ax_c.set_xticklabels(labels, fontsize=6)
ax_c.set_ylabel('1-NN accuracy', fontsize=8)
ax_c.set_ylim(0, 1.0)
ax_c.axhline(0.25, color=RANDOM_COLOR, ls='--', alpha=0.5, lw=0.8)
for i, (acc, n) in enumerate(zip(accs, ns)):
    ax_c.text(i, acc/2, f'{acc:.0%}', ha='center', fontweight='bold', fontsize=7, color='white')
ax_c.text(2, 0.98, '1-NN by CE bin', ha='center', fontsize=9, fontweight='bold',
         transform=ax_c.transAxes)

# ── Far right: Estimated ceiling ──
ax_d = fig.add_axes([0.70, 0.25, 0.28, 0.50])
ax_d.set_xlim(0, 100)
ax_d.set_ylim(0, 100)
ax_d.axis('off')

# Simple horizontal bar visualization
purine_ceiling = 70
pyrimidine_ceiling = 60
current = 64

# 100% reference bar
ax_d.barh(80, 100, 2, color='#EEEEEE', edgecolor='#CCCCCC', lw=0.5)
ax_d.barh(60, purine_ceiling, 2, color=XGB_COLOR, alpha=0.5, edgecolor=XGB_COLOR, lw=1.5)
ax_d.barh(40, pyrimidine_ceiling, 2, color=GRNADE_COLOR, alpha=0.5, edgecolor=GRNADE_COLOR, lw=1.5)
ax_d.barh(20, current, 2, color='#666666', alpha=0.4, edgecolor='#444444', lw=1)

ax_d.text(0, 88, 'Theoretical maximum: 100%', fontsize=7, color='#CCCCCC')
ax_d.text(0, 68, f'Purine ceiling ~{purine_ceiling}%', fontsize=8, fontweight='bold', color=XGB_COLOR)
ax_d.text(0, 48, f'Pyrimidine ceiling ~{pyrimidine_ceiling}%', fontsize=8, fontweight='bold', color=GRNADE_COLOR)
ax_d.text(0, 28, f'Current XGBoost: {current}%', fontsize=8, color='#666666')

# Gap annotation
ax_d.annotate('Remaining\ngap', xy=(68, 25), xytext=(85, 55),
              fontsize=6, color='#FF6600', fontweight='bold', ha='center',
              arrowprops=dict(arrowstyle='->', color='#FF6600', lw=1.2))

plt.savefig('/home/glt/rnaxgb/Nature-comms-v2/figures/fig10_chemical_ceiling.pdf', dpi=DPI, bbox_inches='tight')
plt.savefig('/home/glt/rnaxgb/Nature-comms-v2/figures/fig10_chemical_ceiling.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('Saved fig10_chemical_ceiling (clean version)')
