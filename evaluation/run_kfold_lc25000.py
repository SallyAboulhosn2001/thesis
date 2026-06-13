import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from datasets_code.dataset_lc25000 import build_lc25000_datasets
from training.trainer import train_one_epoch, evaluate
from training.early_stopping import EarlyStopping

# import your model getters
from models.densenet import get_densenet121
from models.backbone import get_efficientnet_b3
from models.resnet import get_resnet50
from models.mobilenet import get_mobilenet_v2
from models.vit import get_vit_b16  # if you have it


def get_model(name: str, num_classes=5):
    name = name.lower()
    if name == "densenet121":
        return get_densenet121(num_classes=num_classes)
    if name == "efficientnet_b3":
        return get_efficientnet_b3(num_classes=num_classes)
    if name == "resnet50":
        return get_resnet50(num_classes=num_classes)
    if name == "mobilenetv2":
        return get_mobilenet_v2(num_classes=num_classes)
    if name == "vit_b16":
        return get_vit_b16(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ===== settings (match methodology) =====
    model_name = "densenet121"   # <- change to run other baselines
    img_size = 224
    batch_size = 32
    lr = 1e-4
    max_epochs = 50
    patience = 7
    n_splits = 5
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build datasets
    train_ds_full, eval_ds_full = build_lc25000_datasets(img_size=img_size)

    y = np.array(train_ds_full.labels)  # same labels ordering in both
    idxs = np.arange(len(train_ds_full))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(idxs, y), start=1):
        print(f"\n===== FOLD {fold}/{n_splits} =====")

        tr_subset = Subset(train_ds_full, tr_idx)
        va_subset = Subset(eval_ds_full, va_idx)

        tr_loader = DataLoader(tr_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(va_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = get_model(model_name, num_classes=5).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        stopper = EarlyStopping(patience=patience, min_delta=0.0)

        best_path = f"kfold_{model_name}_fold{fold}.pth"
        best_val = float("inf")

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, tr_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, va_loader, criterion, device)

            print(f"Fold {fold} | Epoch {epoch:02d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)

            stopper.step(val_loss)
            if stopper.should_stop:
                print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                break

        fold_results.append({"fold": fold, "best_val_loss": best_val, "best_ckpt": best_path})

    print("\n===== K-FOLD DONE =====")
    for r in fold_results:
        print(r)


if __name__ == "__main__":
    main()