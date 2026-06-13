import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Load features
b0 = torch.load("nct_features_student_b0_kd.pt")
dn = torch.load("nct_features_densenet121.pt")

X_b0, y = b0["X"].numpy(), b0["y"].numpy()
X_dn, y2 = dn["X"].numpy(), dn["y"].numpy()
assert np.array_equal(y, y2)

idx = np.arange(len(y))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=y
)

# Split base features
Xb0_train, Xb0_test = X_b0[train_idx], X_b0[test_idx]
Xdn_train, Xdn_test = X_dn[train_idx], X_dn[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Level-0 models
lr_b0 = LogisticRegression(max_iter=3000, n_jobs=-1)
lr_dn = LogisticRegression(max_iter=3000, n_jobs=-1)

lr_b0.fit(Xb0_train, y_train)
lr_dn.fit(Xdn_train, y_train)

# Get probabilities (meta-features)
p_b0_train = lr_b0.predict_proba(Xb0_train)
p_dn_train = lr_dn.predict_proba(Xdn_train)

p_b0_test = lr_b0.predict_proba(Xb0_test)
p_dn_test = lr_dn.predict_proba(Xdn_test)

# Meta input = concat probabilities
meta_X_train = np.concatenate([p_b0_train, p_dn_train], axis=1)
meta_X_test  = np.concatenate([p_b0_test,  p_dn_test], axis=1)

# Level-1 meta-classifier
meta = LogisticRegression(max_iter=3000, n_jobs=-1)
meta.fit(meta_X_train, y_train)

y_pred = meta.predict(meta_X_test)

acc = accuracy_score(y_test, y_pred)
mf1 = f1_score(y_test, y_pred, average="macro")

print(f"[Stacking (LR→LR meta)] Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
