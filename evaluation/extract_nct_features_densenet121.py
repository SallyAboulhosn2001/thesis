import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets_code.dataset_nct_crc import NCTCRCDataset
from models.densenet import get_densenet121
from evaluation.extract_features import extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- LOAD MODEL ----
model = get_densenet121(num_classes=5)
model.load_state_dict(torch.load("densenet121_lc25000.pth", map_location=device))

# remove classifier
model.classifier = torch.nn.Identity()
model = model.to(device)
model.eval()

print("DenseNet121 loaded.")

# ---- NCT DATA ----
nct_ds = NCTCRCDataset()
nct_loader = DataLoader(nct_ds, batch_size=16, shuffle=False)

features, labels = extract_features(model, nct_loader, device)

torch.save(
    {"X": features.cpu(), "y": labels.cpu()},
    "nct_features_densenet121.pt"
)

print("DenseNet121 NCT features saved.")
