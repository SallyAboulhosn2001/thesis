import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from datasets_code.dataset_nct_crc import NCTCRCDataset
from models.vit import get_vit_b16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_vit_b16(5)
model.load_state_dict(torch.load("vit_b16_lc25000.pth", map_location=device))

# Remove classification head
model.heads = torch.nn.Identity()
model = model.to(device)
model.eval()

dataset = NCTCRCDataset()
loader = DataLoader(dataset, batch_size=32)

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

torch.save({"X": features, "y": labels}, "nct_features_vit.pt")
print("ViT NCT features saved.")
