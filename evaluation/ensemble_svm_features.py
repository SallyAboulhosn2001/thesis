import torch
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def fit_svm_proba(X_train, y_train, X_test):
    # LinearSVC does not output probabilities, so we calibrate it
    base = LinearSVC()
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)

b0 = torch.load("nct_features_student_b0_kd.pt")
dn = torch.load("nct_features_densenet121.pt")

X_b0, y = b0["X"].numpy(), b0["y"].numpy()
X_dn, y2 = dn["X"].numpy(), dn["y"].numpy()
assert np.array_equal(y, y2)

idx = np.arange(len(y))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=y
)

Xb0_train, Xb0_test = X_b0[train_idx], X_b0[test_idx]
Xdn_train, Xdn_test = X_dn[train_idx], X_dn[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

proba_b0 = fit_svm_proba(Xb0_train, y_train, Xb0_test)
proba_dn = fit_svm_proba(Xdn_train, y_train, Xdn_test)

proba_ens = 0.5 * proba_b0 + 0.5 * proba_dn
y_pred = np.argmax(proba_ens, axis=1)

acc = accuracy_score(y_test, y_pred)
mf1 = f1_score(y_test, y_pred, average="macro")

print(f"[SVM Ensemble (B0KD + DenseNet)] Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
