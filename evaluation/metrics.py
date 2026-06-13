# evaluation/metrics.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred, average="macro"):
    """
    Compute classification metrics.
    """
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).argmax(dim=1).numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average),
    }

    return metrics
