"""
Train XGBoost with backbone dihedrals only to test architecture confound.

The reviewer correctly noted that comparing XGBoost+935_features against
GNN+raw_coordinates changes two variables simultaneously (features AND architecture).
This experiment tests: what if we give XGBoost only basic backbone dihedral features?

Feature index mapping (from generate_torch_features_v0.py):
- Indices 0-20: 21 bond lengths
- Indices 21-55: 35 bond angles
- Indices 56-69: 14 dihedral angles
- These 70 features are the "basic geometry" subset

We train and evaluate:
1. Dihedral-only (14 features: indices 56-69)
2. Basic geometry (70 features: indices 0-69)
3. Node features only (211 features: indices 0-210)
4. Full features (935 features) — reference
"""
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'RNKDB'))

print("=" * 60)
print("XGBoost with Backbone Dihedrals Only")
print("Testing Architecture vs. Feature Confound")
print("=" * 60)

# Load data
print("\n[1] Loading data...")
X_train = np.load('fair_eval_v3/train_X.npy')
y_train = np.load('fair_eval_v3/train_y.npy')
X_val = np.load('fair_eval_v3/val_X.npy')
y_val = np.load('fair_eval_v3/val_y.npy')
X_test = np.load('fair_eval_v3/post2025_test_X.npy')
y_test = np.load('fair_eval_v3/post2025_test_y.npy')

print(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Feature subsets
configs = {
    'Dihedrals only (14)': list(range(56, 70)),
    'Bond lengths + angles (56)': list(range(0, 56)),
    'Basic geometry (70)': list(range(0, 70)),
    'Node features only (211)': list(range(0, 211)),
    'Full features (935)': list(range(0, 935))
}

# Model params (same as main model)
params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 50,
    'random_state': 42
}

results = {}

for name, indices in configs.items():
    print(f"\n[2] Training: {name} (n_features={len(indices)})...")

    X_tr = X_train[:, indices]
    X_v = X_val[:, indices]
    X_te = X_test[:, indices]

    model = XGBClassifier(**params)
    model.fit(
        X_tr, y_train,
        eval_set=[(X_v, y_val)],
        verbose=False
    )

    # Evaluate
    y_pred_test = model.predict(X_te)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')

    y_pred_val = model.predict(X_v)
    val_acc = accuracy_score(y_val, y_pred_val)

    # Per-class
    per_class = {}
    for nt, nt_name in zip([0, 1, 2, 3], ['A', 'C', 'G', 'U']):
        mask = y_test == nt
        if mask.sum() > 0:
            per_class[nt_name] = {
                'accuracy': float(accuracy_score(y_test[mask], y_pred_test[mask])),
                'n': int(mask.sum())
            }

    results[name] = {
        'n_features': len(indices),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1),
        'per_class': per_class,
        'n_estimators_used': model.best_iteration + 1 if hasattr(model, 'best_iteration') else 500
    }

    print(f"    Val: {val_acc*100:.2f}%, Test: {test_acc*100:.2f}%, F1: {test_f1*100:.2f}%")
    print(f"    Per-class: ", {nt: f"{res['accuracy']*100:.1f}%" for nt, res in per_class.items()})

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("SUMMARY: Feature Subset Comparison")
print(f"{'='*60}")
print(f"\n{'Config':<30s} {'N_feat':>6s} {'ValAcc':>8s} {'TestAcc':>8s} {'MacroF1':>8s}")
print(f"{'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
for name, res in results.items():
    print(f"{name:<30s} {res['n_features']:>6d} {res['val_accuracy']*100:>7.2f}% {res['test_accuracy']*100:>7.2f}% {res['test_macro_f1']*100:>7.2f}%")

# Interpretation
dihe_acc = results['Dihedrals only (14)']['test_accuracy']
full_acc = results['Full features (935)']['test_accuracy']
ratio = dihe_acc / full_acc

print(f"\nInterpretation:")
print(f"  Dihedral-only accuracy: {dihe_acc*100:.2f}%")
print(f"  Full-feature accuracy:  {full_acc*100:.2f}%")
print(f"  Dihedrals capture {ratio*100:.1f}% of full model performance")
print(f"  Information gain from full features: {full_acc*100 - dihe_acc*100:.1f}pp")

if ratio > 0.85:
    print(f"\n  Most discriminative information is in backbone dihedrals.")
    print(f"  The feature engineering contribution beyond basic dihedrals is modest.")
    print(f"  This suggests XGBoost's advantage may be partly architecture-driven.")
else:
    print(f"\n  Substantial information exists beyond backbone dihedrals.")
    print(f"  The full 935 features provide meaningful gains over basic geometry.")
    print(f"  This supports the explicit feature engineering contribution.")

# Save
output = {
    'description': 'XGBoost feature subset comparison for architecture confound analysis',
    'results': results,
    'params': {k: str(v) if not isinstance(v, (int, float, bool, list)) else v for k, v in params.items()}
}

with open('fair_eval_v3/dihedral_baseline_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to fair_eval_v3/dihedral_baseline_results.json")
