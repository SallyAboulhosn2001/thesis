import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets_code.dataset_lc25000 import LC25000Dataset
from datasets_code.dataset_nct_crc import NCTCRCDataset
from models.backbone import get_efficientnet_b3
from training.trainer import train_one_epoch, evaluate
from evaluation.extract_features import extract_features


print("Thesis pipeline initialized.")

# ---- DEVICE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- LC25000 DATA ----
from preprocessing.transforms import get_basic_transforms, get_eval_transforms

USE_STAIN_NORM = True   # <-- change to False for baseline

train_tfms = get_basic_transforms(train=True, use_stain_norm=USE_STAIN_NORM)
eval_tfms  = get_eval_transforms(use_stain_norm=USE_STAIN_NORM)

train_ds = LC25000Dataset(split="train", transform=train_tfms)
test_ds  = LC25000Dataset(split="test",  transform=eval_tfms)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# ---- MODEL ----
num_classes = 5
model = get_efficientnet_b3(num_classes=num_classes)
model = model.to(device)

# ---- TRAINING ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 3
for epoch in range(epochs):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_loss, val_acc = evaluate(
        model, test_loader, criterion, device
    )
    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# ---- SAVE BASELINE ----
torch.save(model.state_dict(), "baseline_efficientnet_b3_lc25000.pth")
print("Baseline model saved.")

# ---- CROSS-DATASET FEATURE EXTRACTION ----
print("\n--- Cross-dataset evaluation (NCT-CRC) ---")

# remove classifier → feature extractor
model.classifier = torch.nn.Identity()
model.eval()

nct_ds = NCTCRCDataset()
nct_loader = DataLoader(nct_ds, batch_size=16, shuffle=False)

features, labels = extract_features(model, nct_loader, device)

print("Extracted features shape:", features.shape)
print("Labels shape:", labels.shape)

import torch

torch.save(
    {"X": features, "y": labels},
    "nct_features_efficientnet_b3.pt"
)
print("NCT features saved.")
import torch

torch.save(
    {"X": features.cpu(), "y": labels.cpu()},
    "nct_features_efficientnet_b3.pt"
)
print("NCT features saved to disk.")
