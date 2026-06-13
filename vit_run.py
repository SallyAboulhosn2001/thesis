import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))

import torch
from models.vit import get_vit_b16
from datasets_code.dataset_lc25000 import LC25000Dataset
from torch.utils.data import DataLoader
from training.trainer import train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = LC25000Dataset(split="train")
val_ds = LC25000Dataset(split="test")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = get_vit_b16(5).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

epochs = 5

print("ViT-B/16 initialized.")

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"[ViT] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "vit_b16_lc25000.pth")
print("✅ ViT-B/16 saved.")
