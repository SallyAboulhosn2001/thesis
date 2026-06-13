import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = torch.load("nct_features_fused_b0kd_densenet.pt")
X = data["X"].numpy()
y = data["y"].numpy()

seeds = [1, 2, 3, 4, 5]

acc_list = []

for seed in seeds:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(max_iter=3000, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)

mean_acc = np.mean(acc_list)
std_acc = np.std(acc_list)

print("===== FUSED MODEL (5 Repeated Runs) =====")
print(f"Accuracy Mean: {mean_acc:.4f}")
print(f"Accuracy Std:  {std_acc:.4f}")
