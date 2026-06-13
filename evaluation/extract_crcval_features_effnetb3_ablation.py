import sys
from pathlib import Path

# Make thesis_code/ the import root so "preprocessing" works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b3

from preprocessing.transforms import get_eval_transforms


class FolderDataset(Dataset):
    """
    Expects:
      datasets/CRC-VAL-HE-7K/CRC-VAL-HE-7K/
        ADI/
        BACK/
        DEB/
        ...
        TUM/
    """
    def __init__(self, root_dir: str, transform):
        self.root = Path(root_dir)
        self.transform = transform

        if not self.root.exists():
            raise RuntimeError(f"Dataset path not found: {self.root}")

        class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if len(class_names) == 0:
            raise RuntimeError(f"No class folders found in: {self.root}")

        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.samples = []

        for c in class_names:
            cdir = self.root / c
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                for p in cdir.glob(ext):
                    self.samples.append((str(p), self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {self.root}")

        print(f"[CRC-VAL] Samples: {len(self.samples)}")
        print(f"[CRC-VAL] Classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="datasets/CRC-VAL-HE-7K/CRC-VAL-HE-7K")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--use_stain_norm", type=int, default=0)  # 0/1
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Use stain norm:", bool(args.use_stain_norm))

    tfm = get_eval_transforms(use_stain_norm=bool(args.use_stain_norm))

    ds = FolderDataset(args.data_root, tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Build model + load checkpoint (5-class head), then turn into feature extractor
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)

    # feature extractor output: 1536-D
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()

    feats, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        f = model(xb).detach().cpu().numpy().astype(np.float32)
        feats.append(f)
        labels.append(np.array(yb, dtype=np.int64))

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)

    torch.save({"X": X, "y": y}, args.out)
    print(f"\n✅ Saved: {args.out}")
    print("X shape:", X.shape, "| y shape:", y.shape)


if __name__ == "__main__":
    main()