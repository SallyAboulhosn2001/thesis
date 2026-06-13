import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize

def compute_all_metrics(y_true, y_pred, y_proba=None):
    metrics = {}

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision_macro"] = precision_score(y_true, y_pred, average="macro")
    metrics["Recall_macro"] = recall_score(y_true, y_pred, average="macro")
    metrics["F1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    if y_proba is not None:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        metrics["ROC_AUC_macro"] = roc_auc_score(
            y_true_bin, y_proba, average="macro", multi_class="ovr"
        )

    cm = confusion_matrix(y_true, y_pred)
    metrics["Confusion_Matrix"] = cm

    # False Negative Rate (macro)
    fnr_list = []
    for i in range(len(cm)):
        FN = sum(cm[i, :]) - cm[i, i]
        TP = cm[i, i]
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        fnr_list.append(fnr)

    metrics["FNR_macro"] = np.mean(fnr_list)

    return metrics
