import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets_code.dataset_nct_crc import NCTCRCDataset
from models.densenet import get_densenet121
from models.student import get_efficientnet_b0_student

def evaluate_ensemble_nct(
    model_a, model_b, loader, device, w_a=0.5, w_b=0.5
):
    model_a.eval()
    model_b.eval()

    correct = 0
    total = 0

    # confusion-free macro-f1
    num_classes = 9
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits_a = model_a(images)         # DenseNet logits (9)
            logits_b = model_b(images)         # B0 KD logits (9)

            probs_a = F.softmax(logits_a, dim=1)
            probs_b = F.softmax(logits_b, dim=1)

            probs = w_a * probs_a + w_b * probs_b
            preds = probs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels, preds):
                conf[t.long(), p.long()] += 1

    acc = correct / total

    # macro-f1
    f1s = []
    for c in range(num_classes):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s)

    return acc, macro_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # NCT data: 9 classes
    nct_ds = NCTCRCDataset()
    loader = DataLoader(nct_ds, batch_size=16, shuffle=False)

    # ---- DenseNet121 ----
    densenet = get_densenet121(num_classes=5)
    densenet.load_state_dict(
        torch.load("densenet121_lc25000.pth", map_location=device)
    )

    # IMPORTANT: swap head to 9-class head for NCT evaluation
    # We'll reuse feature extractor and add new 9-class head
    # (frozen / linear probing style)
    densenet.classifier = torch.nn.Linear(
        densenet.classifier.in_features, 9
    ).to(device)

    densenet = densenet.to(device)

    # ---- B0 KD student ----
    b0 = get_efficientnet_b0_student(num_classes=5)
    b0.load_state_dict(
        torch.load("student_efficientnet_b0_kd_lc25000.pth", map_location=device)
    )

    b0.classifier = torch.nn.Linear(
        b0.classifier.in_features, 9
    ).to(device)

    b0 = b0.to(device)

    # NOTE: these new 9-class heads are randomly initialized.
    # So we MUST train them quickly before ensembling.
    print("❗ Heads are new (9-class). Train them first using extracted features (next step).")

    # For now we stop here to avoid misleading results.
    # We'll do the correct approach next: linear-probe training of both heads.

if __name__ == "__main__":
    main()
