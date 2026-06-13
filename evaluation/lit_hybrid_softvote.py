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


# ---------- datasets ----------
class LC25000TestDataset(Dataset):
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
            if not cls_dir.exists():
                raise RuntimeError(f"Missing: {cls_dir}")
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                for p in cls_dir.glob(ext):
                    self.samples.append((str(p), idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root}")

        print(f"[LC25000 TEST] Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


class CRCVALBinaryDataset(Dataset):
    """
    root should be:
      datasets/CRC-VAL-HE-7K/CRC-VAL-HE-7K
    Binary mapping:
      TUM -> 1 (malignant)
      everything else -> 0 (benign)
    """
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if not self.root.exists():
            raise RuntimeError(f"Missing dataset root: {self.root}")

        class_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        if len(class_dirs) == 0:
            raise RuntimeError(f"No class folders in {self.root}")

        for d in class_dirs:
            y = 1 if d.name.upper() == "TUM" else 0
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                for p in d.glob(ext):
                    self.samples.append((str(p), y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}")

        print(f"[CRC-VAL BIN] Samples: {len(self.samples)} | tumor={sum(y for _,y in self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


# ---------- models ----------
def build_densenet121(num_classes=5):
    m = densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def build_resnet50(num_classes=5):
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def predict_softvote(models, loader, device):
    all_y = []
    all_pred = []

    softmax = nn.Softmax(dim=1)

    for xb, yb in loader:
        xb = xb.to(device)
        probs_sum = None

        for m in models:
            logits = m(xb)
            probs = softmax(logits)
            probs_sum = probs if probs_sum is None else probs_sum + probs

        probs_avg = probs_sum / len(models)
        pred = torch.argmax(probs_avg, dim=1).cpu().numpy()

        all_pred.append(pred)
        all_y.append(np.array(yb))

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["lc25000_test", "crcval_binary"])
    ap.add_argument("--dn_ckpt", default="densenet121_lc25000.pth")
    ap.add_argument("--rn_ckpt", default="resnet50_lc25000.pth")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    if args.mode == "lc25000_test":
        ds = LC25000TestDataset("datasets/LC25000/lung_colon_image_set/Test Set", tfm)
        num_classes = 5
    else:
        ds = CRCVALBinaryDataset("datasets/CRC-VAL-HE-7K/CRC-VAL-HE-7K", tfm)
        num_classes = 5  # models are 5-class, but dataset labels are binary (0/1)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    dn = build_densenet121(num_classes=5).to(device)
    rn = build_resnet50(num_classes=5).to(device)

    # robust load (no safe-mode issues)
    try:
        dn_state = torch.load(args.dn_ckpt, map_location=device, weights_only=False)
    except TypeError:
        dn_state = torch.load(args.dn_ckpt, map_location=device)

    try:
        rn_state = torch.load(args.rn_ckpt, map_location=device, weights_only=False)
    except TypeError:
        rn_state = torch.load(args.rn_ckpt, map_location=device)

    dn.load_state_dict(dn_state, strict=True)
    rn.load_state_dict(rn_state, strict=True)

    dn.eval()
    rn.eval()

    y_true, y_pred = predict_softvote([dn, rn], loader, device)

    # For CRC-VAL binary: convert 5-class prediction to binary by malignant-vs-benign mapping:
    # malignant = {0,2,4}, benign = {1,3}
    if args.mode == "crcval_binary":
        malignant = {0, 2, 4}
        y_pred = np.array([1 if p in malignant else 0 for p in y_pred], dtype=np.int64)

    acc = accuracy(y_true, y_pred)
    print("\n===== LITERATURE HYBRID #1: SOFT-VOTING (DenseNet+ResNet) =====")
    print("Mode:", args.mode)
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()