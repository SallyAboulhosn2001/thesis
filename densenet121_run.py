import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets_code.dataset_lc25000 import LC25000Dataset
from models.densenet import get_densenet121
from training.trainer import train_one_epoch
from evaluation.utils_eval import evaluate_ce_acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_ds = LC25000Dataset(split="train")
test_ds = LC25000Dataset(split="test")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

model = get_densenet121(num_classes=5).to(device)
print("DenseNet121 initialized.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 5
for epoch in range(epochs):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_loss, val_acc = evaluate_ce_acc(model, test_loader, device)

    print(
        f"[DenseNet121] Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

torch.save(model.state_dict(), "densenet121_lc25000.pth")
print("✅ DenseNet121 baseline saved.")
