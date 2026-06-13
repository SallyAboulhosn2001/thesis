import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # thesis_code/
sys.path.insert(0, str(ROOT))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from datasets_code.dataset_lc25000 import LC25000Dataset


def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, crit, device):
    model.eval()
    total, n, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = crit(out, yb)
        total += loss.item() * xb.size(0)
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        n += xb.size(0)
    return total / max(n, 1), correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_stain_norm", type=int, default=0)  # 0/1
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Use stain norm:", bool(args.use_stain_norm))

    train_ds = LC25000Dataset(split="train", use_stain_norm=bool(args.use_stain_norm))
    test_ds  = LC25000Dataset(split="test",  use_stain_norm=bool(args.use_stain_norm))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_loss = float("inf")
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, crit, device)
        va_loss, va_acc = eval_one_epoch(model, test_loader, crit, device)
        print(f"Epoch [{ep}/{args.epochs}] | Train Loss: {tr:.4f} | Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.4f}")

        if va_loss < best_loss:
            best_loss = va_loss
            torch.save(model.state_dict(), args.out)

    print("✅ Saved:", args.out)


if __name__ == "__main__":
    main()