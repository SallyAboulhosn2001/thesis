import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from preprocessing.transforms import get_eval_transforms
from models.densenet import get_densenet121


# -------------------------------
# Official Test Dataset Loader
# -------------------------------
class LC25000TestDataset(Dataset):
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
        print(f"[TEST SET] Classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# -------------------------------
# Evaluation
# -------------------------------
def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = torch.softmax(out, dim=1)

            preds.extend(torch.argmax(p, dim=1).cpu().numpy())
            targets.extend(y.numpy())
            probs.extend(p.cpu().numpy())

    return np.array(preds), np.array(targets), np.array(probs)


def main():
    checkpoint = sys.argv[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = LC25000TestDataset(
        root_dir="datasets/LC25000/lung_colon_image_set/Test Set",
        transform=get_eval_transforms(224)
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = get_densenet121(num_classes=5)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    preds, targets, probs = evaluate(model, test_loader, device)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    mcc = matthews_corrcoef(targets, preds)
    roc = roc_auc_score(targets, probs, multi_class="ovr")

    cm = confusion_matrix(targets, preds)
    fnr = 1 - np.diag(cm) / cm.sum(axis=1)
    fnr = fnr.mean()

    print("\n===== OFFICIAL TEST RESULTS =====")
    print("Accuracy:", acc)
    print("Macro-F1:", f1)
    print("ROC-AUC:", roc)
    print("MCC:", mcc)
    print("FNR:", fnr)


if __name__ == "__main__":
    main()