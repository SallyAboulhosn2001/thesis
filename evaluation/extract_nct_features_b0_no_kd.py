import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets_code.dataset_nct_crc import NCTCRCDataset
from models.student import get_efficientnet_b0_student
from evaluation.extract_features import extract_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- LOAD B0 NO-KD ----
model = get_efficientnet_b0_student(num_classes=5)
model.load_state_dict(
    torch.load("efficientnet_b0_no_kd_lc25000.pth", map_location=device)
)
model = model.to(device)

# remove classifier → feature extractor
model.classifier = torch.nn.Identity()
model.eval()

print("B0 (NO-KD) model loaded.")

# ---- NCT DATA ----
nct_ds = NCTCRCDataset()
nct_loader = DataLoader(nct_ds, batch_size=16, shuffle=False)

# ---- FEATURE EXTRACTION ----
features, labels = extract_features(model, nct_loader, device)

print("B0 NO-KD features shape:", features.shape)
print("Labels shape:", labels.shape)

# ---- SAVE ----
torch.save(
    {"X": features.cpu(), "y": labels.cpu()},
    "nct_features_b0_no_kd.pt"
)

print("B0 NO-KD NCT features saved.")
