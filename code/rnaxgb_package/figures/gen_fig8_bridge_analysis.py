"""
Fig 8: Three-Element Bridge Analysis.
Shows bridge type schematic + accuracy by bridge type for all models.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = '/home/glt/rnaxgb/RNKDB'
with open(f'{BASE}/fair_eval_v3/bridge_analysis_results.json') as f:
    data = json.load(f)

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9})

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# ── Panel A: Accuracy by bridge type ──
ax = axes[0]
btypes = ['none', 'base_only', 'phosphate_only', 'base_phosphate', 'base_ribose', 'three_element']
labels = ['No contact', 'Base\nonly', 'Phosphate\nonly', 'Base+\nPhosphate', 'Base+\nRibose', 'Three-\nelement']
colors = ['#cccccc', '#4477AA', '#AA7744', '#CC6677', '#88CCEE', '#AA3377']

acc_data = data['accuracy_by_bridge']
x = np.arange(len(btypes))
width = 0.2

for i, model in enumerate(['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']):
    vals = []
    for bt in btypes:
        d = acc_data[model].get(bt, {})
        vals.append(d.get('acc', 0) if d.get('n', 0) > 0 else 0)
    bars = ax.bar(x + (i-1.65)*width, vals, width, label=model, zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Per-nucleotide accuracy')
ax.set_title('Model accuracy by bridge type')
ax.legend(fontsize=7, ncol=2)
ax.axhline(0.25, color='black', ls='--', alpha=0.3)
ax.text(5.5, 0.255, 'Random (25%)', fontsize=6, color='gray')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.2)

# Add n labels
for j, bt in enumerate(btypes):
    n = acc_data['XGBoost'].get(bt, {}).get('n', 0)
    ax.text(j, 0.98, f'n={n}', ha='center', fontsize=6, color='#333333')

# ── Panel B: Bridge diagram ──
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Three-element bridge concept', fontsize=11)

# Draw schematic RNA nucleotide with base (left), ribose (center), phosphate (right)
# Base
base_circle = plt.Circle((2, 7), 0.9, fill=True, color='#4477AA', alpha=0.7, ec='#225588', lw=2)
ax.add_patch(base_circle)
ax.text(2, 7, 'Base', ha='center', fontsize=8, fontweight='bold', color='white')
ax.text(2, 6, 'N1/N9, O2, N6...', ha='center', fontsize=6, color='#225588')

# Ribose
ribose_circle = plt.Circle((5, 7), 0.7, fill=True, color='#88CC66', alpha=0.7, ec='#669944', lw=2)
ax.add_patch(ribose_circle)
ax.text(5, 7, 'Ribose', ha='center', fontsize=8, fontweight='bold', color='white')
ax.text(5, 6.1, "C4', O2'...", ha='center', fontsize=6, color='#669944')

# Phosphate
phos_circle = plt.Circle((8, 7), 0.7, fill=True, color='#AA7744', alpha=0.7, ec='#885522', lw=2)
ax.add_patch(phos_circle)
ax.text(8, 7, 'P', ha='center', fontsize=8, fontweight='bold', color='white')
ax.text(8, 6.1, 'OP1, OP2', ha='center', fontsize=6, color='#885522')

# Environmental molecule (water/ion) in center
env_circle = plt.Circle((5, 4), 0.6, fill=True, color='#FF6644', alpha=0.8, ec='#CC4422', lw=2)
ax.add_patch(env_circle)
ax.text(5, 4, 'H₂O\n/ Ion', ha='center', fontsize=7, fontweight='bold', color='white')

# Bridge lines
ax.plot([2.5, 4.5], [6.3, 4.5], '--', color='gray', lw=1.5, alpha=0.6)
ax.plot([5, 5], [6.3, 4.6], '--', color='gray', lw=1.5, alpha=0.6)
ax.plot([7.5, 5.5], [6.3, 4.5], '--', color='gray', lw=1.5, alpha=0.6)

# Labels for bridge types
ax.text(10.5, 7, 'Three-element\nbridge', fontsize=7, color='#AA3377', fontweight='bold', rotation=0)
ax.annotate('', xy=(10.5, 6.2), xytext=(10.5, 4.8),
            arrowprops=dict(arrowstyle='<->', color='#AA3377', lw=2))
ax.text(10.5, 5.5, 'Base+\nPhosphate\nbridge', fontsize=7, color='#CC6677', ha='center')

# Legend for mediator types
ax.text(0.5, 3.5, 'Mediator types:', fontsize=7, fontweight='bold')
ax.text(0.5, 2.8, '● Water', fontsize=6, color='#FF6644')
ax.text(0.5, 2.3, '● Mg²⁺, K⁺, Na⁺', fontsize=6, color='#888888')
ax.text(0.5, 1.8, '● Ligand', fontsize=6, color='#666666')

# Reference
ax.text(5, 0.5, 'Inspired by Kretsch et al. (2025) Nature 642, 250-259',
        ha='center', fontsize=6, fontstyle='italic', color='gray')

plt.tight_layout()
plt.savefig('/home/glt/rnaxgb/Nature-comms-v2/figures/fig8_bridge_analysis.pdf',
            dpi=150, bbox_inches='tight')
plt.savefig('/home/glt/rnaxgb/Nature-comms-v2/figures/fig8_bridge_analysis.png',
            dpi=150, bbox_inches='tight')
print('Saved fig8_bridge_analysis')
