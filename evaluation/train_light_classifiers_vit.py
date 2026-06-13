import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

data = torch.load("nct_features_vit.pt")
X = data["X"].numpy()
y = data["y"].numpy()

idx = np.arange(len(y))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=y
)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Logistic Regression
lr = LogisticRegression(max_iter=3000, n_jobs=-1)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average="macro")

print(f"[ViT + LR] Accuracy: {acc_lr:.4f} | Macro-F1: {f1_lr:.4f}")

# Linear SVM
svm = LinearSVC(max_iter=5000)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average="macro")

print(f"[ViT + SVM] Accuracy: {acc_svm:.4f} | Macro-F1: {f1_svm:.4f}")
