"""
Fast XGBoost with backbone dihedrals only (subsampled data).
Tests architecture confound: XGBoost vs GNN, dihedrals vs full features.
"""
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import os, sys

os.chdir(os.path.join(os.path.dirname(__file__), '..', 'RNKDB'))

print("Loading data...", flush=True)
X_train = np.load('fair_eval_v3/train_X.npy')
y_train = np.load('fair_eval_v3/train_y.npy')
X_val = np.load('fair_eval_v3/val_X.npy')
y_val = np.load('fair_eval_v3/val_y.npy')
X_test = np.load('fair_eval_v3/post2025_test_X.npy')
y_test = np.load('fair_eval_v3/post2025_test_y.npy')

# Subsample: use 100K training + full validation + full test
np.random.seed(42)
n_sub = 100000
idx = np.random.choice(len(X_train), n_sub, replace=False)
X_train = X_train[idx]
y_train = y_train[idx]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)

configs = [
    ('Dihedrals only (14)', list(range(56, 70))),
    ('Bond lengths+angles (56)', list(range(0, 56))),
    ('Basic geometry (70)', list(range(0, 70))),
    ('Node features (211)', list(range(0, 211))),
    ('Full features (935)', list(range(0, 935))),
]

# Faster params
params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 30,
    'random_state': 42,
    'n_jobs': -1
}

results = {}
for name, indices in configs:
    print(f"\nTraining: {name} ({len(indices)} features)...", flush=True)
    X_tr = X_train[:, indices].astype(np.float32)
    X_v = X_val[:, indices].astype(np.float32)
    X_te = X_test[:, indices].astype(np.float32)

    model = XGBClassifier(**params)
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)

    y_pred_test = model.predict(X_te)
    test_acc = accuracy_score(y_test, y_pred_test)
    y_pred_val = model.predict(X_v)
    val_acc = accuracy_score(y_val, y_pred_val)

    per_class = {}
    for nt, nt_name in zip([0,1,2,3], ['A','C','G','U']):
        mask = y_test == nt
        if mask.sum() > 0:
            per_class[nt_name] = float(accuracy_score(y_test[mask], y_pred_test[mask]))

    n_trees = model.best_iteration + 1 if hasattr(model, 'best_iteration') else params['n_estimators']
    results[name] = {
        'n_features': len(indices),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'per_class': per_class,
        'n_trees': n_trees
    }
    print(f"  Val: {val_acc*100:.2f}%, Test: {test_acc*100:.2f}%, Trees: {n_trees}", flush=True)

# Summary
print(f"\n{'='*60}")
print(f"{'Config':<30s} {'Feats':>5s} {'ValAcc':>8s} {'TestAcc':>8s}")
print(f"{'-'*60}")
for name, res in results.items():
    print(f"{name:<30s} {res['n_features']:>5d} {res['val_accuracy']*100:>7.2f}% {res['test_accuracy']*100:>7.2f}%")

dihe_acc = results['Dihedrals only (14)']['test_accuracy']
full_acc = results['Full features (935)']['test_accuracy']
print(f"\nDihedrals capture {dihe_acc/full_acc*100:.1f}% of full model performance")
print(f"Absolute gain from full features: {(full_acc-dihe_acc)*100:.1f}pp")

output = {'results': results, 'note': 'Subsampled 100K training, 200 trees, max_depth=6'}
with open('fair_eval_v3/dihedral_baseline_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to fair_eval_v3/dihedral_baseline_results.json")
