import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def fit_lr_and_predict_proba(X_train, y_train, X_test):
    lr = LogisticRegression(max_iter=3000, n_jobs=-1)
    lr.fit(X_train, y_train)
    return lr.predict_proba(X_test)

# Load features
b0 = torch.load("nct_features_student_b0_kd.pt")
dn = torch.load("nct_features_densenet121.pt")

X_b0, y = b0["X"].numpy(), b0["y"].numpy()
X_dn, y2 = dn["X"].numpy(), dn["y"].numpy()

assert np.array_equal(y, y2), "Label mismatch!"

# Same split for both
idx = np.arange(len(y))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=y
)

Xb0_train, Xb0_test = X_b0[train_idx], X_b0[test_idx]
Xdn_train, Xdn_test = X_dn[train_idx], X_dn[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Train two LR models and get probabilities
proba_b0 = fit_lr_and_predict_proba(Xb0_train, y_train, Xb0_test)
proba_dn = fit_lr_and_predict_proba(Xdn_train, y_train, Xdn_test)

# Ensemble (average)
proba_ens = 0.5 * proba_b0 + 0.5 * proba_dn
y_pred = np.argmax(proba_ens, axis=1)

acc = accuracy_score(y_test, y_pred)
mf1 = f1_score(y_test, y_pred, average="macro")

print(f"[LR Ensemble (B0KD + DenseNet)] Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
