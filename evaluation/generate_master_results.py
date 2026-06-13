import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

# All feature files
models = {
    "EffNet-B3": "nct_features_efficientnet_b3.pt",
    "B0-NoKD": "nct_features_b0_no_kd.pt",
    "B0-KD": "nct_features_student_b0_kd.pt",
    "MobileNetV2": "nct_features_mobilenet.pt",
    "ResNet50": "nct_features_resnet50.pt",
    "DenseNet121": "nct_features_densenet121.pt",
    "ViT-B16": "nct_features_vit.pt",
    "Fusion (B0KD+DenseNet)": "nct_features_fused_b0kd_densenet.pt",
}

seeds = [1, 2, 3, 4, 5]

rows = []

for model_name, file_path in models.items():

    data = torch.load(file_path)
    X = data["X"].numpy()
    y = data["y"].numpy()

    all_metrics = []

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
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)

        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        roc_auc = roc_auc_score(
            y_test_bin, y_proba, average="macro", multi_class="ovr"
        )

        cm = confusion_matrix(y_test, y_pred)
        fnr_list = []
        for i in range(len(cm)):
            FN = sum(cm[i, :]) - cm[i, i]
            TP = cm[i, i]
            fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
            fnr_list.append(fnr)

        fnr_macro = np.mean(fnr_list)

        all_metrics.append([acc, prec, rec, f1, roc_auc, mcc, fnr_macro])

    all_metrics = np.array(all_metrics)
    means = np.mean(all_metrics, axis=0)
    stds = np.std(all_metrics, axis=0)

    rows.append({
        "Model": model_name,
        "Accuracy": means[0],
        "Accuracy_std": stds[0],
        "F1": means[3],
        "F1_std": stds[3],
        "ROC-AUC": means[4],
        "MCC": means[5],
        "FNR": means[6]
    })

df = pd.DataFrame(rows)
df = df.sort_values(by="Accuracy", ascending=False)

df.to_csv("results_cross_dataset.csv", index=False)

print("\n=== FINAL RANKING ===")
print(df)
print("\nSaved to results_cross_dataset.csv")