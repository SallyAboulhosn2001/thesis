import torch
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)

# =============================
# 🔁 CHANGE THIS LINE ONLY
# =============================
feature_file = "glas_features_densenet121.pt"
# For GlaS → "glas_features_densenet121.pt"

# =============================
# LOAD DATA
# =============================
data = torch.load(feature_file)

X = data["X"]
y = data["y"]

# =============================
# LOAD MODELS
# =============================
svm = joblib.load("svm_literature.pkl")
lr = joblib.load("lr_literature.pkl")

models = {
    "SVM": svm,
    "Logistic Regression": lr
}

# =============================
# LC25000 → BINARY MAPPING
# =============================
def map_to_binary(y):
    # LC25000 classes:
    # 0: colon_aca (malignant)
    # 1: colon_n (benign)
    # 2: lung_aca (malignant)
    # 3: lung_n (benign)
    # 4: lung_scc (malignant)

    malignant = {0, 2, 4}
    return np.array([1 if i in malignant else 0 for i in y])


# =============================
# METRICS
# =============================
def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix → FNR
    cm = confusion_matrix(y_true, y_pred)
    fnr_list = []

    for i in range(len(cm)):
        FN = np.sum(cm[i, :]) - cm[i, i]
        TP = cm[i, i]
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        fnr_list.append(fnr)

    fnr = np.mean(fnr_list)

    # ROC-AUC
    roc = None
    try:
        if y_proba is not None:
            if y_proba.shape[1] == 2:
                roc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except:
        pass

    return acc, prec, rec, f1, roc, mcc, fnr


# =============================
# DETECT IF BINARY DATASET
# =============================
is_binary_dataset = len(np.unique(y)) == 2

# =============================
# EVALUATION LOOP
# =============================
for name, model in models.items():

    y_pred_raw = model.predict(X)

    # If evaluating on binary dataset (like GlaS)
    if is_binary_dataset:
        y_true = y
        y_pred = map_to_binary(y_pred_raw)
    else:
        y_true = y
        y_pred = y_pred_raw

    # Probabilities if available
    if hasattr(model, "predict_proba"):
        y_proba_raw = model.predict_proba(X)

        if is_binary_dataset:
            # Convert to binary probabilities
            malignant_idx = [0, 2, 4]
            benign_idx = [1, 3]

            malignant_prob = y_proba_raw[:, malignant_idx].sum(axis=1)
            benign_prob = y_proba_raw[:, benign_idx].sum(axis=1)

            y_proba = np.stack([benign_prob, malignant_prob], axis=1)
        else:
            y_proba = y_proba_raw
    else:
        y_proba = None

    acc, prec, rec, f1, roc, mcc, fnr = compute_metrics(
        y_true, y_pred, y_proba
    )

    print(f"\n===== {name} RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {roc if roc is not None else 'N/A'}")
    print(f"MCC: {mcc:.4f}")
    print(f"FNR: {fnr:.4f}")