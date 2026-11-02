"""
Train MLP on same 935 features to test architecture-independence of bottleneck.

If MLP ≈ XGBoost on same features → architecture-independence → bottleneck is features
If MLP > XGBoost → XGBoost is suboptimal → bottleneck is model
If MLP < XGBoost → XGBoost is a good choice but may not be optimal
"""
import json
import numpy as np
import os, sys
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'RNKDB'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("=" * 60)
print("MLP vs XGBoost: Architecture-Independence Test")
print("=" * 60)

# Load data
print("\n[1] Loading data...", flush=True)
X_train = np.load('fair_eval_v3/train_X.npy').astype(np.float32)
y_train = np.load('fair_eval_v3/train_y.npy').astype(np.int64)
X_val = np.load('fair_eval_v3/val_X.npy').astype(np.float32)
y_val = np.load('fair_eval_v3/val_y.npy').astype(np.int64)
X_test = np.load('fair_eval_v3/post2025_test_X.npy').astype(np.float32)
y_test = np.load('fair_eval_v3/post2025_test_y.npy').astype(np.int64)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)

# Standardize
print("[2] Standardizing...", flush=True)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Convert to torch
X_train_t = torch.from_numpy(X_train_s)
y_train_t = torch.from_numpy(y_train)
X_val_t = torch.from_numpy(X_val_s)
y_val_t = torch.from_numpy(y_val)
X_test_t = torch.from_numpy(X_test_s)
y_test_t = torch.from_numpy(y_test)

# MLP architecture
class MLP(nn.Module):
    def __init__(self, input_dim=935, hidden=512, n_classes=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes)
        )

    def forward(self, x):
        return self.net(x)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[3] Training on {device}...", flush=True)

model = MLP(input_dim=935, hidden=512, n_classes=4, dropout=0.3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

batch_size = 4096
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

best_val_acc = 0
best_state = None
patience = 15
patience_counter = 0

for epoch in range(100):
    model.train()
    total_loss = 0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for Xb, yb in val_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)
    val_acc = val_correct / val_total
    scheduler.step(1 - val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}: loss={total_loss/len(train_dl):.4f}, val_acc={val_acc*100:.2f}%", flush=True)

    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch}", flush=True)
        break

# Load best model and evaluate
model.load_state_dict(best_state)
model.eval()

# Test evaluation
print("\n[4] Evaluating on test set...", flush=True)
model.eval()
all_preds = []
with torch.no_grad():
    for i in range(0, len(X_test_t), batch_size * 2):
        Xb = X_test_t[i:i+batch_size*2].to(device)
        logits = model(Xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
y_pred = np.concatenate(all_preds)

test_acc = accuracy_score(y_test, y_pred)
print(f"  MLP Test Accuracy: {test_acc*100:.2f}%", flush=True)

# Per-class
for nt, nt_name in zip([0,1,2,3], ['A','C','G','U']):
    mask = y_test == nt
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"  {nt_name}: {acc*100:.2f}%", flush=True)

# Also test on training set for overfitting check
print("\n[5] Training set accuracy...", flush=True)
train_preds = []
with torch.no_grad():
    for i in range(0, len(X_train_t), batch_size * 2):
        Xb = X_train_t[i:i+batch_size*2].to(device)
        logits = model(Xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        train_preds.append(preds)
y_train_pred = np.concatenate(train_preds)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"  MLP Train Accuracy: {train_acc*100:.2f}%", flush=True)

# Comparison with XGBoost (from fast dihedral experiment)
# XGBoost full features (100K subsampled): ~58.15% test
# XGBoost full features (full data, 500 trees): ~64.10% test
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"  MLP (935 features, full train): {test_acc*100:.2f}%")
print(f"  XGBoost (935 features, full train): 64.10% (published)")
print(f"  XGBoost (935 features, 100K sub): 58.15% (fast ablation)")

# Save
output = {
    'model': 'MLP-512-256',
    'train_accuracy': float(train_acc),
    'val_accuracy': float(best_val_acc),
    'test_accuracy': float(test_acc),
    'per_class': {
        nt: float(accuracy_score(y_test[y_test==i], y_pred[y_test==i]))
        for i, nt in enumerate(['A','C','G','U'])
    },
    'comparison': {
        'xgboost_full': 0.6410,
        'xgboost_sub_100k': 0.5815,
        'mlp_full': float(test_acc)
    },
    'interpretation': ''
}

if test_acc > 0.61:
    output['interpretation'] = 'MLP ≈ XGBoost: bottleneck is representation, not architecture'
elif test_acc > 0.54:
    output['interpretation'] = 'MLP slightly < XGBoost: XGBoost is better but both limited by features'
else:
    output['interpretation'] = 'MLP << XGBoost: XGBoost architecture matters significantly'

with open('fair_eval_v3/mlp_baseline_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nInterpretation: {output['interpretation']}")
print(f"Results saved to fair_eval_v3/mlp_baseline_results.json")
