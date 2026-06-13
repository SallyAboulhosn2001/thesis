import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


# -------------------------
# Load fused features
# -------------------------

data = torch.load("crcval_features_fused.pt")
X = data["X"].numpy()
y = data["y"].numpy()

print("Features shape:", X.shape)
print("Labels shape:", y.shape)


# -------------------------
# Evaluation function
# -------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    # ROC-AUC
    classes = np.unique(y)

    if len(classes) == 2:
        roc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    # FNR
    cm = confusion_matrix(y_test, y_pred)

    fnr_list = []

    for i in range(len(cm)):
        FN = np.sum(cm[i, :]) - cm[i, i]
        TP = cm[i, i]

        if (FN + TP) > 0:
            fnr = FN / (FN + TP)
        else:
            fnr = 0

        fnr_list.append(fnr)

    fnr_macro = np.mean(fnr_list)

    return acc, prec, rec, f1, roc, mcc, fnr_macro


# -------------------------
# Multiple seed evaluation
# -------------------------

seeds = [1, 2, 3, 4, 5]

lr_metrics = []
svm_metrics = []

for seed in seeds:

    idx = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        stratify=y,
        random_state=seed
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Logistic Regression
    lr = LogisticRegression(max_iter=3000, n_jobs=-1)

    lr_metrics.append(
        evaluate_model(lr, X_train, X_test, y_train, y_test)
    )

    # SVM
    svm = SVC(probability=True)

    svm_metrics.append(
        evaluate_model(svm, X_train, X_test, y_train, y_test)
    )


# -------------------------
# Print results
# -------------------------

def summarize(metrics):

    m = np.array(metrics)

    mean = np.mean(m, axis=0)
    std = np.std(m, axis=0)

    return mean, std


lr_mean, lr_std = summarize(lr_metrics)
svm_mean, svm_std = summarize(svm_metrics)


print("\n===== FUSION + LOGISTIC REGRESSION =====")

print(f"Accuracy: {lr_mean[0]:.4f} ± {lr_std[0]:.4f}")
print(f"Precision: {lr_mean[1]:.4f} ± {lr_std[1]:.4f}")
print(f"Recall: {lr_mean[2]:.4f} ± {lr_std[2]:.4f}")
print(f"F1: {lr_mean[3]:.4f} ± {lr_std[3]:.4f}")
print(f"ROC-AUC: {lr_mean[4]:.4f} ± {lr_std[4]:.4f}")
print(f"MCC: {lr_mean[5]:.4f} ± {lr_std[5]:.4f}")
print(f"FNR: {lr_mean[6]:.4f} ± {lr_std[6]:.4f}")


print("\n===== FUSION + SVM =====")

print(f"Accuracy: {svm_mean[0]:.4f} ± {svm_std[0]:.4f}")
print(f"Precision: {svm_mean[1]:.4f} ± {svm_std[1]:.4f}")
print(f"Recall: {svm_mean[2]:.4f} ± {svm_std[2]:.4f}")
print(f"F1: {svm_mean[3]:.4f} ± {svm_std[3]:.4f}")
print(f"ROC-AUC: {svm_mean[4]:.4f} ± {svm_std[4]:.4f}")
print(f"MCC: {svm_mean[5]:.4f} ± {svm_std[5]:.4f}")
print(f"FNR: {svm_mean[6]:.4f} ± {svm_std[6]:.4f}")