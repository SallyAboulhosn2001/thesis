# evaluation/evaluate_checkpoint_on_dataset.py

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import densenet121, efficientnet_b3

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)


# -----------------------------
# Metrics
# -----------------------------
def compute_macro_fnr(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fnr_list = []

    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        denom = tp + fn
        fnr_list.append(fn / denom if denom > 0 else 0)

    return float(np.mean(fnr_list))


def compute_metrics(y_true, y_pred, y_proba, n_classes):

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)
    fnr = compute_macro_fnr(y_true, y_pred, n_classes)

    if n_classes == 2:
        roc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

    return acc, f1, roc, mcc, fnr


# -----------------------------
# Inference
# -----------------------------
def infer(model, loader, device, n_classes):

    model.eval()
    softmax = nn.Softmax(dim=1)

    all_y = []
    all_pred = []
    all_proba = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)

            logits = model(xb)

            probs = softmax(logits).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_proba.append(probs)
            all_pred.append(preds)
            all_y.append(yb.numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_proba = np.concatenate(all_proba)

    return compute_metrics(y_true, y_pred, y_proba, n_classes)


# -----------------------------
# Generic folder dataset
# -----------------------------
class FolderDataset(Dataset):

    def __init__(self, root, class_to_idx, transform):

        self.samples = []
        self.transform = transform

        for cls_name, cls_idx in class_to_idx.items():

            cls_dir = os.path.join(root, cls_name)

            if not os.path.isdir(cls_dir):
                continue

            for fn in os.listdir(cls_dir):

                if fn.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):

                    path = os.path.join(cls_dir, fn)

                    self.samples.append((path, cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, y = self.samples[idx]

        img = Image.open(path).convert("RGB")

        x = self.transform(img)

        return x, y


# -----------------------------
# GlaS Dataset
# -----------------------------
class GlasDataset(Dataset):

    def __init__(self, root, csv_path, transform):

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        self.samples = []
        self.transform = transform

        for _, row in df.iterrows():

            name = row["name"]

            label_raw = str(row["grade (GlaS)"]).lower()

            if "benign" in label_raw:
                label = 0

            elif "malignant" in label_raw:
                label = 1

            else:
                continue

            img_path = os.path.join(root, f"{name}.bmp")

            if os.path.exists(img_path):
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError("No GlaS images loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, y = self.samples[idx]

        img = Image.open(path).convert("RGB")

        x = self.transform(img)

        return x, y


# -----------------------------
# Model builder
# -----------------------------
def build_model(model_name, num_classes, checkpoint_path, device):

    if model_name == "densenet121":

        model = densenet121(weights=None)

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )

    elif model_name == "efficientnet_b3":

        model = efficientnet_b3(weights=None)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    else:
        raise ValueError("Unsupported model")

    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Fix EfficientNet classifier mismatch
    if model_name == "efficientnet_b3":

        if "classifier.weight" in state:
            state["classifier.1.weight"] = state.pop("classifier.weight")

        if "classifier.bias" in state:
            state["classifier.1.bias"] = state.pop("classifier.bias")

    model.load_state_dict(state, strict=False)

    model.to(device)

    return model


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["lc_test", "crcval", "glas"]
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["densenet121", "efficientnet_b3"]
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lc_root",
        default="datasets/LC25000/lung_colon_image_set/Test Set"
    )

    parser.add_argument(
        "--crcval_root",
        default="datasets/CRC-VAL-HE-7K/CRC-VAL-HE-7K"
    )

    parser.add_argument(
        "--glas_root",
        default="datasets/Glas/Warwick_QU_Dataset"
    )

    parser.add_argument(
        "--glas_csv",
        default="datasets/Glas/Warwick_QU_Dataset/Grade.csv"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------
    # Dataset selection
    # -----------------------------

    if args.dataset == "lc_test":

        class_to_idx = {
            "colon_aca": 0,
            "colon_n": 1,
            "lung_aca": 2,
            "lung_n": 3,
            "lung_scc": 4
        }

        dataset = FolderDataset(
            args.lc_root,
            class_to_idx,
            transform
        )

        num_classes = 5

    elif args.dataset == "crcval":

        class_to_idx = {
            "ADI": 0,
            "BACK": 1,
            "DEB": 2,
            "LYM": 3,
            "MUC": 4,
            "MUS": 5,
            "NORM": 6,
            "STR": 7,
            "TUM": 8
        }

        dataset = FolderDataset(
            args.crcval_root,
            class_to_idx,
            transform
        )

        num_classes = 9

    else:

        dataset = GlasDataset(
            args.glas_root,
            args.glas_csv,
            transform
        )

        num_classes = 2

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    model = build_model(
        args.model,
        num_classes,
        args.checkpoint,
        device
    )

    acc, f1, roc, mcc, fnr = infer(
        model,
        loader,
        device,
        num_classes
    )

    print("\n===== OFFICIAL RESULTS (Inference Only) =====")
    print(f"Accuracy: {acc}")
    print(f"Macro-F1: {f1}")
    print(f"ROC-AUC: {roc}")
    print(f"MCC: {mcc}")
    print(f"FNR: {fnr}")


if __name__ == "__main__":
    main()