import argparse
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import efficientnet_b3


# -----------------------------
# LC25000 Test Set Dataset
# -----------------------------
class LC25000TestDataset(Dataset):
    CLASS_TO_IDX = {
        "colon_aca": 0,
        "colon_n": 1,
        "lung_aca": 2,
        "lung_n": 3,
        "lung_scc": 4,
    }

    def __init__(self, test_root, transform):
        self.test_root = Path(test_root)
        self.transform = transform
        self.samples = []

        for cls_name, cls_idx in self.CLASS_TO_IDX.items():
            cls_dir = self.test_root / cls_name
            if not cls_dir.exists():
                raise RuntimeError(f"Missing class folder: {cls_dir}")

            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                for p in cls_dir.glob(ext):
                    self.samples.append((p, cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in: {self.test_root}")

        print(f"[TEST SET] Samples: {len(self.samples)}")
        print(f"[TEST SET] Classes: {self.CLASS_TO_IDX}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, y


def get_eval_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    all_y = []
    all_pred = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        pred = torch.argmax(logits, dim=1)

        correct += (pred == yb).sum().item()
        total += yb.size(0)

        all_y.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    acc = correct / max(total, 1)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    # Macro-F1, MCC, ROC-AUC, FNR can be added if you want,
    # but accuracy is enough for sanity; you already have full metric scripts.
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--test_root", type=str,
                        default="datasets/LC25000/lung_colon_image_set/Test Set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = get_eval_transforms()
    ds = LC25000TestDataset(args.test_root, tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Build EfficientNet-B3 EXACTLY
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    state = torch.load(args.checkpoint, map_location=device)

    # Your k-fold checkpoints are state_dict only, so this should load cleanly:
    model.load_state_dict(state, strict=True)

    model.to(device)

    acc = evaluate(model, loader, device)

    print("\n===== OFFICIAL TEST RESULTS (EffNet-B3) =====")
    print("Checkpoint:", args.checkpoint)
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()