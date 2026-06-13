import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from datasets_code.dataset_nct_crc import NCTCRCDataset
from models.mobilenet import get_mobilenet_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_mobilenet_v2(5)
model.load_state_dict(torch.load("mobilenet_lc25000.pth", map_location=device))

# Remove classifier only
model.classifier = torch.nn.Identity()
model = model.to(device)
model.eval()

dataset = NCTCRCDataset()
loader = DataLoader(dataset, batch_size=64)

features = []
labels = []

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        feats = model(imgs)
        features.append(feats.cpu())
        labels.append(lbls)

features = torch.cat(features)
labels = torch.cat(labels)

torch.save({"X": features, "y": labels}, "nct_features_mobilenet.pt")
print("MobileNet NCT features saved.")
