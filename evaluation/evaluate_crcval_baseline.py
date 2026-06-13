import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from torch.utils.data import DataLoader
from datasets_code.dataset_crc_val import CRCVALDataset
from preprocessing.transforms import get_eval_transforms
from models.densenet import get_densenet121


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CRCVALDataset()

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = get_densenet121(num_classes=9)
    model.load_state_dict(torch.load("densenet121_lc25000.pth", map_location=device))
    model.to(device)

    preds, targets, probs = evaluate(model, loader, device)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    mcc = matthews_corrcoef(targets, preds)
    roc = roc_auc_score(targets, probs, multi_class="ovr")

    cm = confusion_matrix(targets, preds)
    fnr = 1 - np.diag(cm) / cm.sum(axis=1)
    fnr = fnr.mean()

    print("\n===== CRC-VAL BASELINE RESULTS =====")
    print("Accuracy:", acc)
    print("Macro-F1:", f1)
    print("ROC-AUC:", roc)
    print("MCC:", mcc)
    print("FNR:", fnr)


if __name__ == "__main__":
    main()