# evaluation/lit_hybrid_featurefusion_lr.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import densenet121, resnet50

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix


class LC25000Dataset(Dataset):
    CLASS_TO_IDX = {
        "colon_aca": 0,
        "colon_n": 1,
        "lung_aca": 2,
        "lung_n": 3,
        "lung_scc": 4,
    }

    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for cls, idx in self.CLASS_TO_IDX.items():
            cls_dir = self.root / cls
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                for p in cls_dir.glob(ext):
                    self.samples.append((str(p), idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No LC25000 images found in {self.root}")

        print(f"[LC25000] {self.root.name} Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


class CRCVALBinaryDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for cls_dir in self.root.iterdir():
            if not cls_dir.is_dir():
                continue

            y = 1 if cls_dir.name.upper() == "TUM" else 0

            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                for p in cls_dir.glob(ext):
                    self.samples.append((str(p), y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No CRC-VAL images found in {self.root}")

        print(f"[CRC-VAL Binary] Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def build_densenet_feature(ckpt, device):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 5)

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)

    model.classifier = nn.Identity()
    model.to(device)
    model.eval()
    return model


def build_resnet_feature(ckpt, device):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)

    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_fused_features(densenet, resnet, loader, device):
    X_list = []
    y_list = []

    for xb, yb in loader:
        xb = xb.to(device)

        fd = densenet(xb).cpu().numpy()
        fr = resnet(xb).cpu().numpy()

        fused = np.concatenate([fd, fr], axis=1)

        X_list.append(fused)
        y_list.append(np.array(yb))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


def macro_fnr(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fnrs = []

    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fnrs.append(fn / (tp + fn) if (tp + fn) > 0 else 0.0)

    return float(np.mean(fnrs))


def print_metrics(title, y_true, y_pred, y_proba):
    labels = np.unique(y_true)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    fnr = macro_fnr(y_true, y_pred, labels)

    if len(labels) == 2:
        roc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

    print(f"\n===== {title} =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"FNR:       {fnr:.4f}")


def map_lc25000_pred_to_binary(pred):
    malignant = {0, 2, 4}
    return np.array([1 if p in malignant else 0 for p in pred], dtype=np.int64)


def map_lc25000_proba_to_binary(proba5):
    malignant_idx = [0, 2, 4]
    benign_idx = [1, 3]

    p_mal = proba5[:, malignant_idx].sum(axis=1)
    p_ben = proba5[:, benign_idx].sum(axis=1)

    return np.stack([p_ben, p_mal], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--densenet_ckpt", default="densenet121_lc25000.pth")
    parser.add_argument("--resnet_ckpt", default="resnet50_lc25000.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    train_ds = LC25000Dataset(
        "datasets/LC25000/lung_colon_image_set/Train and Validation Set",
        transform
    )

    test_ds = LC25000Dataset(
        "datasets/LC25000/lung_colon_image_set/Test Set",
        transform
    )

    crc_ds = CRCVALBinaryDataset(
        "datasets/CRC-VAL-HE-7K/CRC-VAL-HE-7K",
        transform
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    crc_loader = DataLoader(crc_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    densenet = build_densenet_feature(args.densenet_ckpt, device)
    resnet = build_resnet_feature(args.resnet_ckpt, device)

    print("\nExtracting LC25000 train features...")
    X_train, y_train = extract_fused_features(densenet, resnet, train_loader, device)

    print("Extracting LC25000 test features...")
    X_test, y_test = extract_fused_features(densenet, resnet, test_loader, device)

    print("Extracting CRC-VAL features...")
    X_crc, y_crc = extract_fused_features(densenet, resnet, crc_loader, device)

    print("\nTraining Logistic Regression on LC25000 fused features only...")
    clf = LogisticRegression(max_iter=3000, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    y_proba_test = clf.predict_proba(X_test)

    print_metrics(
        "LITERATURE HYBRID #2: FEATURE FUSION + LR on LC25000 TEST",
        y_test,
        y_pred_test,
        y_proba_test
    )

    y_pred_crc_5 = clf.predict(X_crc)
    y_proba_crc_5 = clf.predict_proba(X_crc)

    y_pred_crc_bin = map_lc25000_pred_to_binary(y_pred_crc_5)
    y_proba_crc_bin = map_lc25000_proba_to_binary(y_proba_crc_5)

    print_metrics(
        "LITERATURE HYBRID #2: FEATURE FUSION + LR on CRC-VAL BINARY",
        y_crc,
        y_pred_crc_bin,
        y_proba_crc_bin
    )


if __name__ == "__main__":
    main()