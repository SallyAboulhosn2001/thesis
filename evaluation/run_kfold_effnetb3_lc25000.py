# evaluation/run_kfold_effnetb3_lc25000.py
"""
5-Fold Cross-Validation for EfficientNet-B3 on LC25000 (FULL dataset)
- Uses StratifiedKFold (80/20 per fold)
- Trains EfficientNet-B3 (ImageNet pretrained)
- Early stopping on validation loss
- Saves best checkpoint per fold: kfold_effnetb3_fold{fold}.pth
- Prints fold training logs + summary

Run:
  python evaluation/run_kfold_effnetb3_lc25000.py

Optional args:
  --epochs 50 --batch_size 32 --lr 1e-4 --patience 7 --num_workers 4
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms as T
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from sklearn.model_selection import StratifiedKFold


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset: LC25000 FULL
# -----------------------------
class LC25000FullDataset(Dataset):
    """
    Expects LC25000 structure (as you have used before):

    datasets/LC25000/lung_colon_image_set/
        Train and Validation Set/
            colon_aca/
            colon_n/
            lung_aca/
            lung_n/
            lung_scc/
        Test Set/
            ... (not used for k-fold)

    This dataset loads ONLY Train and Validation Set for k-fold.
    """

    CLASS_TO_IDX = {
        "colon_aca": 0,
        "colon_n": 1,
        "lung_aca": 2,
        "lung_n": 3,
        "lung_scc": 4,
    }

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise RuntimeError(f"LC25000 root_dir not found: {self.root_dir}")

        # Collect all images
        self.image_paths = []
        self.labels = []

        for cls_name, cls_idx in self.CLASS_TO_IDX.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                # Some users have the folder names slightly different; fail loudly.
                raise RuntimeError(f"Missing class folder: {cls_dir}")

            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                for p in cls_dir.glob(ext):
                    self.image_paths.append(p)
                    self.labels.append(cls_idx)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under: {self.root_dir}")

        self.labels = np.array(self.labels, dtype=np.int64)

        print(f"[LC25000-FULL] Samples: {len(self.image_paths)}")
        print(f"[LC25000-FULL] Classes: {self.CLASS_TO_IDX}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        y = int(self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, y


# -----------------------------
# Transforms
# -----------------------------
def get_train_transform():
    return T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform():
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ]
    )


# -----------------------------
# Model
# -----------------------------
def build_effnet_b3(num_classes: int = 5):
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running += loss.item() * xb.size(0)
        n += xb.size(0)

    return running / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running = 0.0
    correct = 0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        running += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        n += xb.size(0)

    val_loss = running / max(n, 1)
    val_acc = correct / max(n, 1)
    return val_loss, val_acc


# -----------------------------
# Main K-Fold
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="datasets/LC25000/lung_colon_image_set/Train and Validation Set")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # FULL dataset for split indices
    full_dataset_for_split = LC25000FullDataset(args.data_root, transform=None)
    y_all = full_dataset_for_split.labels  # numpy array

    # We'll create fold-specific datasets with different transforms:
    # train uses augmentation, val uses deterministic transform.
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), start=1):
        print(f"\n===== FOLD {fold}/5 =====")

        # Build fold datasets
        train_ds = LC25000FullDataset(args.data_root, transform=train_transform)
        val_ds = LC25000FullDataset(args.data_root, transform=val_transform)

        train_subset = Subset(train_ds, train_idx.tolist())
        val_subset = Subset(val_ds, val_idx.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = build_effnet_b3(num_classes=5).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        ckpt_path = f"kfold_effnetb3_fold{fold}.pth"

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)

            print(
                f"Fold {fold} | Epoch {epoch:02d} | "
                f"Train Loss {tr_loss:.4f} | Val Loss {va_loss:.4f} | Val Acc {va_acc:.4f}"
            )

            if va_loss < best_val_loss - 1e-6:
                best_val_loss = va_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best val loss {best_val_loss:.4f} @ epoch {best_epoch})"
                )
                break

        fold_results.append(
            {"fold": fold, "best_val_loss": best_val_loss, "best_epoch": best_epoch, "best_ckpt": ckpt_path}
        )

    print("\n===== K-FOLD DONE =====")
    for r in fold_results:
        print(r)


if __name__ == "__main__":
    main()