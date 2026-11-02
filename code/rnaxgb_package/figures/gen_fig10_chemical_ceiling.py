"""
Fig 10: Chemical information ceiling — why the 7-atom backbone cannot distinguish A↔G and U↔C.
Combined schematic + per-class accuracy + 1-NN degradation.
"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})

fig = plt.figure(figsize=(14, 7))

# ── Panel A: Chemical schematic (axes spanning left third) ──
ax_schem = fig.add_axes([0.02, 0.05, 0.32, 0.90])
ax_schem.set_xlim(0, 12)
ax_schem.set_ylim(0, 14)
ax_schem.axis('off')
ax_schem.set_title('A. The 7-atom representation cannot chemically distinguish A↔G or U↔C',
                   fontsize=11, fontweight='bold', loc='left')

# Purine comparison (A vs G)
ax_schem.text(0, 12.5, 'Purines: Adenine (A) vs Guanine (G)', fontsize=10, fontweight='bold')

# Adenine box
a_box = FancyBboxPatch((0.2, 8), 5, 4, boxstyle='round,pad=0.1', fill=True,
                        facecolor='#E8F0FE', edgecolor='#4477AA', lw=2)
ax_schem.add_patch(a_box)
ax_schem.text(2.7, 11.5, 'Adenine', ha='center', fontsize=9, fontweight='bold', color='#4477AA')
ax_schem.text(0.5, 10.8, 'Has: N6-amine (NH₂)', fontsize=8, color='#225588')
ax_schem.text(0.5, 10.2, '7-atom sees: N9 only', fontsize=8, color='#666666')
ax_schem.text(0.5, 9.6, 'Missing: N6 position', fontsize=8, color='red', fontweight='bold')
ax_schem.text(0.5, 9.0, '→ Confused with G', fontsize=8, color='#CC6677')
ax_schem.text(0.5, 8.3, 'A→G error: 23.6%', fontsize=8, color='#CC6677', fontweight='bold')

# Guanine box
g_box = FancyBboxPatch((6.5, 8), 5, 4, boxstyle='round,pad=0.1', fill=True,
                        facecolor='#FEE8E8', edgecolor='#CC6677', lw=2)
ax_schem.add_patch(g_box)
ax_schem.text(9, 11.5, 'Guanine', ha='center', fontsize=9, fontweight='bold', color='#CC6677')
ax_schem.text(6.8, 10.8, 'Has: O6-carbonyl (C=O)', fontsize=8, color='#882222')
ax_schem.text(6.8, 10.2, '7-atom sees: N9 only', fontsize=8, color='#666666')
ax_schem.text(6.8, 9.6, 'Missing: O6 position', fontsize=8, color='red', fontweight='bold')
ax_schem.text(6.8, 9.0, '→ Correctly predicted 71%', fontsize=8, color='#4477AA')
ax_schem.text(6.8, 8.3, 'G recall: 79.0%', fontsize=8, color='#4477AA', fontweight='bold')

# Pyrimidine comparison (U vs C)
ax_schem.text(0, 7, 'Pyrimidines: Uridine (U) vs Cytidine (C)', fontsize=10, fontweight='bold')

u_box = FancyBboxPatch((0.2, 3.3), 5, 3.5, boxstyle='round,pad=0.1', fill=True,
                        facecolor='#E8F0FE', edgecolor='#4477AA', lw=2)
ax_schem.add_patch(u_box)
ax_schem.text(2.7, 6.3, 'Uridine', ha='center', fontsize=9, fontweight='bold', color='#4477AA')
ax_schem.text(0.5, 5.7, 'Has: O4-carbonyl (C=O)', fontsize=8, color='#225588')
ax_schem.text(0.5, 5.1, '7-atom sees: N1 only', fontsize=8, color='#666666')
ax_schem.text(0.5, 4.5, 'Missing: O4 position', fontsize=8, color='red', fontweight='bold')
ax_schem.text(0.5, 3.9, '→ Confused with C', fontsize=8, color='#CC6677')
ax_schem.text(0.5, 3.5, 'U→C error: 18.1%', fontsize=8, color='#CC6677', fontweight='bold')

c_box = FancyBboxPatch((6.5, 3.3), 5, 3.5, boxstyle='round,pad=0.1', fill=True,
                        facecolor='#FEE8E8', edgecolor='#CC6677', lw=2)
ax_schem.add_patch(c_box)
ax_schem.text(9, 6.3, 'Cytidine', ha='center', fontsize=9, fontweight='bold', color='#CC6677')
ax_schem.text(6.8, 5.7, 'Has: N4-amine (NH₂)', fontsize=8, color='#882222')
ax_schem.text(6.8, 5.1, '7-atom sees: N1 only', fontsize=8, color='#666666')
ax_schem.text(6.8, 4.5, 'Missing: N4 position', fontsize=8, color='red', fontweight='bold')
ax_schem.text(6.8, 3.9, '→ Correctly predicted 75%', fontsize=8, color='#4477AA')
ax_schem.text(6.8, 3.5, 'C recall: 75.0%', fontsize=8, color='#4477AA', fontweight='bold')

# Key insight box
key_box = FancyBboxPatch((0.2, 0.3), 11.3, 2.5, boxstyle='round,pad=0.15', fill=True,
                          facecolor='#FFF9E0', edgecolor='#FFAA00', lw=2)
ax_schem.add_patch(key_box)
ax_schem.text(0.5, 2.3, 'Key insight: The 7-atom coarse-grained backbone (P, O5\', C5\', C4\', C3\', O3\', N1/N9) deliberately omits',
             fontsize=7.5, fontweight='bold')
ax_schem.text(0.5, 1.8, 'base-identifying atoms (N6-amine of A, O6-carbonyl of G, O4-carbonyl of U, N4-amine of C).',
             fontsize=7.5, fontweight='bold')
ax_schem.text(0.5, 1.3, 'No model architecture can overcome these missing atoms. The 7-atom representation IS the bottleneck.',
             fontsize=7.5, fontweight='bold', color='red')
ax_schem.text(0.5, 0.8, 'Progress requires: full-atom featurisation, multi-state ensembles, or hybrid evolutionary approaches.',
             fontsize=7.5, color='#666666')

# ── Panel B: Per-class accuracy + 1-NN degradation ──
ax_b = fig.add_axes([0.38, 0.05, 0.30, 0.90])

# Load feature information data
BASE = '/home/glt/rnaxgb/RNKDB'
with open(f'{BASE}/fair_eval_v3/feature_information_analysis.json') as f:
    fi = json.load(f)

# Per-class accuracy
classes = ['A', 'C', 'G', 'U']
class_acc = [fi['per_class_ce'][c]['accuracy'] for c in classes]
class_ce = [fi['per_class_ce'][c]['mean_ce'] for c in classes]
class_colors = ['#4477AA', '#88CCEE', '#CC6677', '#AA7744']

ax_b.set_title('B. Per-class performance', fontsize=11, fontweight='bold', loc='left')
x = np.arange(len(classes))
width = 0.35
bars = ax_b.bar(x, class_acc, width, color=class_colors, edgecolor='white')
for i, (acc, ce) in enumerate(zip(class_acc, class_ce)):
    ax_b.text(i, acc/2, f'{acc:.1%}', ha='center', fontweight='bold', fontsize=10, color='white')
    ax_b.text(i, acc + 0.02, f'CE={ce:.2f}', ha='center', fontsize=7, color='#666666')
ax_b.set_xticks(x)
ax_b.set_xticklabels(classes)
ax_b.set_ylabel('Accuracy')
ax_b.set_ylim(0, 1.0)
ax_b.axhline(0.25, color='black', ls='--', alpha=0.3)

# ── Panel C: 1-NN degradation ──
ax_c = fig.add_axes([0.72, 0.05, 0.27, 0.43])
ax_c.set_title('C. 1-NN accuracy by CE bin', fontsize=10, fontweight='bold', loc='left')

nn_data = fi['1nn_by_CE_bin']
bins = list(nn_data.keys())
accs = [nn_data[b]['mean_accuracy'] for b in bins]
ns = [nn_data[b]['n'] for b in bins]
bin_labels = ['0-0.3\n(Easy)', '0.3-0.6', '0.6-1.0', '1.0-1.5\n(Hard)', '1.5+']
bin_colors = ['#44AA44', '#88CC66', '#CCCC44', '#DDAA44', '#DD4444']

ax_c.bar(range(len(bins)), accs, color=bin_colors, edgecolor='white')
ax_c.set_xticks(range(len(bins)))
ax_c.set_xticklabels(bin_labels, fontsize=7)
ax_c.axhline(0.25, color='black', ls='--', alpha=0.3)
for i, (acc, n) in enumerate(zip(accs, ns)):
    ax_c.text(i, acc/2, f'{acc:.1%}', ha='center', fontweight='bold', fontsize=8, color='white')
    ax_c.text(i, acc + 0.02, f'n={n:,}', ha='center', fontsize=6, color='#666666')
ax_c.set_ylabel('1-NN accuracy')
ax_c.set_ylim(0, 1.0)

# ── Panel D: Estimated ceiling ──
ax_d = fig.add_axes([0.72, 0.55, 0.27, 0.40])
ax_d.set_title('D. Estimated 7-atom ceiling', fontsize=10, fontweight='bold', loc='left')
ax_d.axis('off')

ceiling_data = [
    ('Purines\n(A, G)', 0.70, '#4477AA'),
    ('Pyrimidines\n(C, U)', 0.60, '#88CCEE'),
]
y_pos = [1.8, 0.6]
for i, (label, ceiling, color) in enumerate(ceiling_data):
    ax_d.barh(y_pos[i], ceiling, height=0.5, color=color, alpha=0.5, edgecolor=color, lw=2)
    ax_d.barh(y_pos[i], 1.0, height=0.5, color='white', alpha=0.3, edgecolor='gray', lw=1, zorder=0)
    ax_d.text(0.02, y_pos[i], f'{label}: ~{ceiling:.0%}', va='center', fontsize=9, fontweight='bold')
    ax_d.text(ceiling + 0.01, y_pos[i], f'{ceiling:.0%}', va='center', fontsize=8, color=color, fontweight='bold')

ax_d.set_xlim(0, 1.1)
ax_d.set_ylim(0, 2.5)
ax_d.text(0.5, -0.5, 'Remaining gap: base-identifying atoms\n+ multi-state ensembles', ha='center',
          fontsize=7, fontstyle='italic', color='gray')

plt.savefig('/home/glt/rnaxgb/Nature-comms-v2/figures/fig10_chemical_ceiling.pdf',
            dpi=150, bbox_inches='tight')
plt.savefig('/home/glt/rnaxgb/Nature-comms-v2/figures/fig10_chemical_ceiling.png',
            dpi=150, bbox_inches='tight')
print('Saved fig10_chemical_ceiling')
