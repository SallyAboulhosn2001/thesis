import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from datasets_code.dataset_crc_val import CRCVALDataset
from models.student import get_efficientnet_b0_student
from models.densenet import get_densenet121

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = CRCVALDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load models
b0 = get_efficientnet_b0_student(num_classes=5)
b0.load_state_dict(torch.load("student_efficientnet_b0_kd_lc25000.pth", map_location=device))
b0 = b0.to(device).eval()

dn = get_densenet121(num_classes=5)
dn.load_state_dict(torch.load("densenet121_lc25000.pth", map_location=device))
dn = dn.to(device).eval()

features = []
labels = []

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(device)

        f1 = b0.features(imgs)
        f1 = torch.flatten(f1, 1)

        f2 = dn.features(imgs)
        f2 = torch.flatten(f2, 1)

        fused = torch.cat([f1, f2], dim=1)

        features.append(fused.cpu())
        labels.append(lbls)

X = torch.cat(features)
y = torch.cat(labels)

torch.save({"X": X, "y": y}, "crcval_features_fused.pt")

print("CRC-VAL fused features saved.")
print("Shape:", X.shape)