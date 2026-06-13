import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets_code.dataset_nct_crc import NCTCRCDataset
from models.resnet import get_resnet50
from evaluation.extract_features import extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = get_resnet50(num_classes=5)
model.load_state_dict(torch.load("resnet50_lc25000.pth", map_location=device))
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

print("ResNet50 loaded.")

nct_ds = NCTCRCDataset()
nct_loader = DataLoader(nct_ds, batch_size=16, shuffle=False)

features, labels = extract_features(model, nct_loader, device)

torch.save(
    {"X": features.cpu(), "y": labels.cpu()},
    "nct_features_resnet50.pt"
)

print("ResNet50 NCT features saved.")
