import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from torch.utils.data import DataLoader
from PIL import Image

from preprocessing.transforms import get_eval_transforms
from models.efficientnet import get_efficientnet_b0
from models.densenet import get_densenet121


# -------------------------------
# Test Dataset Loader
# -------------------------------
class LC25000TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.image_paths = []
        self.labels = []

        for cls in self.class_names:
            cls_dir = self.root_dir / cls
            for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
                for p in cls_dir.glob(ext):
                    self.image_paths.append(p)
                    self.labels.append(self.class_to_idx[cls])

        print(f"[TEST SET] Samples: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# -------------------------------
# Fusion Evaluation
# -------------------------------
def evaluate(b0, dn, loader, device):
    b0.eval()
    dn.eval()

    preds = []
    targets = []
    probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            f1 = b0.features(x)
            f2 = dn.features(x)

            f1 = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f1, 1), 1)
            f2 = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f2, 1), 1)

            fused = torch.cat([f1, f2], dim=1)

            # simple linear fusion head
            out = torch.nn.Linear(fused.shape[1], 5).to(device)(fused)

            p = torch.softmax(out, dim=1)

            preds.extend(torch.argmax(p, dim=1).cpu().numpy())
            targets.extend(y.numpy())
            probs.extend(p.cpu().numpy())

    return np.array(preds), np.array(targets), np.array(probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LC25000TestDataset(
        root_dir="datasets/LC25000/lung_colon_image_set/Test Set",
        transform=get_eval_transforms(224)
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    b0 = get_efficientnet_b0(num_classes=5)
    b0.load_state_dict(torch.load("student_efficientnet_b0_kd_lc25000.pth", map_location=device))

    dn = get_densenet121(num_classes=5)
    dn.load_state_dict(torch.load("kfold_densenet121_fold1.pth", map_location=device))

    b0.to(device)
    dn.to(device)

    preds, targets, probs = evaluate(b0, dn, loader, device)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    mcc = matthews_corrcoef(targets, preds)
    roc = roc_auc_score(targets, probs, multi_class="ovr")

    cm = confusion_matrix(targets, preds)
    fnr = 1 - np.diag(cm) / cm.sum(axis=1)
    fnr = fnr.mean()

    print("\n===== FUSION TEST RESULTS =====")
    print("Accuracy:", acc)
    print("Macro-F1:", f1)
    print("ROC-AUC:", roc)
    print("MCC:", mcc)
    print("FNR:", fnr)


if __name__ == "__main__":
    main()