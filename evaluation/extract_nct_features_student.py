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

# ---- LOAD STUDENT ----
student = get_efficientnet_b0_student(num_classes=5)
student.load_state_dict(
    torch.load("student_efficientnet_b0_kd_lc25000.pth", map_location=device)
)
student = student.to(device)

# remove classifier → feature extractor
student.classifier = torch.nn.Identity()
student.eval()

print("KD student loaded.")

# ---- NCT DATA ----
nct_ds = NCTCRCDataset()
nct_loader = DataLoader(nct_ds, batch_size=16, shuffle=False)

# ---- FEATURE EXTRACTION ----
features, labels = extract_features(student, nct_loader, device)

print("Student features shape:", features.shape)
print("Labels shape:", labels.shape)

# ---- SAVE ----
torch.save(
    {"X": features.cpu(), "y": labels.cpu()},
    "nct_features_student_b0_kd.pt"
)

print("KD student NCT features saved.")
