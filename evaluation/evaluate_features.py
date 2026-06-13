import sys
import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

def load_xy(data):
    # Support multiple key styles
    if "X" in data and "y" in data:
        X = data["X"]
        y = data["y"]
    elif "features" in data and "labels" in data:
        X = data["features"]
        y = data["labels"]
    else:
        raise KeyError(f"Feature file keys not recognized. Found keys: {list(data.keys())}")

    # Torch tensors -> numpy
    if hasattr(X, "cpu"):
        X = X.cpu().numpy()
    else:
        X = np.array(X)

    if hasattr(y, "cpu"):
        y = y.cpu().numpy()
    else:
        y = np.array(y)

    return X, y

def macro_fnr(cm):
    fnr_list = []
    for i in range(cm.shape[0]):
        FN = cm[i, :].sum() - cm[i, i]
        TP = cm[i, i]
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        fnr_list.append(fnr)
    return float(np.mean(fnr_list))

def compute_roc_auc(y_true, y_proba, classes):
    """
    - Binary: roc_auc_score(y_true, proba[:,1])
    - Multiclass: roc_auc_score(binarized_y, proba, multi_class="ovr", average="macro")
    """
    unique = np.unique(y_true)
    if len(unique) == 2:
        # binary
        return roc_auc_score(y_true, y_proba[:, 1])
    else:
        y_bin = label_binarize(y_true, classes=classes)
        return roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluation/evaluate_features.py <feature_file>")
        sys.exit(1)

    feature_file = sys.argv[1]

    # Safer load for pure tensors (works if your torch version supports it)
    try:
        data = torch.load(feature_file, weights_only=False)
    except TypeError:
        data = torch.load(feature_file)

    X, y = load_xy(data)

    seeds = [1, 2, 3, 4, 5]
    all_metrics = []

    classes = np.unique(y)

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
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        roc = compute_roc_auc(y_test, y_proba, classes=classes)

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        fnr = macro_fnr(cm)

        all_metrics.append([acc, prec, rec, f1, roc, mcc, fnr])

    all_metrics = np.array(all_metrics)
    means = np.mean(all_metrics, axis=0)
    stds = np.std(all_metrics, axis=0)

    print("\n===== RESULTS =====")
    print(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}")
    print(f"Precision: {means[1]:.4f} ± {stds[1]:.4f}")
    print(f"Recall:    {means[2]:.4f} ± {stds[2]:.4f}")
    print(f"F1:        {means[3]:.4f} ± {stds[3]:.4f}")
    print(f"ROC-AUC:   {means[4]:.4f} ± {stds[4]:.4f}")
    print(f"MCC:       {means[5]:.4f} ± {stds[5]:.4f}")
    print(f"FNR:       {means[6]:.4f} ± {stds[6]:.4f}")

if __name__ == "__main__":
    main()