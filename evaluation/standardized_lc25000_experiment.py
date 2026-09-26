#!/usr/bin/env python3
"""
Standardized LC25000 architecture experiment for the thesis.

Design goals
------------
1) One persisted 5-fold stratified manifest shared by all architectures.
2) Same preprocessing, optimizer, loss, learning rate, batch size, early stopping,
   and checkpoint rule across architectures.
3) Cross-validation reports HELD-OUT FOLD metrics (not repeated evaluation of
   the same packaged test set).
4) The packaged LC25000 Test Set is touched only in --mode final, after the
   final source model has been trained for a pre-declared number of epochs
   derived from CV (median best epoch).
5) Every evaluation saves per-image predictions/probabilities for later paired
   statistical analysis.
6) Epoch-level resume checkpoints make Colab disconnects recoverable.

Expected LC25000 structure
--------------------------
<DATA_ROOT>/
    Train and Validation Set/
        colon_aca/
        colon_n/
        lung_aca/
        lung_n/
        lung_scc/
    Test Set/
        colon_aca/
        colon_n/
        lung_aca/
        lung_n/
        lung_scc/

Example
-------
python standardized_lc25000_experiment.py \
  --mode cv \
  --model densenet121 \
  --data_root "/content/drive/MyDrive/.../lung_colon_image_set" \
  --out_dir "/content/drive/MyDrive/thesis_standardized_runs"

After all 5 folds for a model:
python standardized_lc25000_experiment.py \
  --mode final \
  --model densenet121 \
  --data_root "/content/drive/MyDrive/.../lung_colon_image_set" \
  --out_dir "/content/drive/MyDrive/thesis_standardized_runs"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from torchvision import models
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B3_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)

from sklearn import __version__ as sklearn_version
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SUPPORTED_MODELS = (
    "densenet121",
    "efficientnet_b3",
    "resnet50",
    "mobilenetv2",
    "vit_b16",
)

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Sample:
    rel_path: str
    class_name: str
    label: int
    fold: int


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is more important than maximal throughput for this study.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        pass

    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state(loader_generator: Optional[torch.Generator] = None) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def restore_rng_state(state: dict, loader_generator: Optional[torch.Generator] = None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if loader_generator is not None and "loader_generator" in state:
        loader_generator.set_state(state["loader_generator"])


# -----------------------------------------------------------------------------
# Data manifest
# -----------------------------------------------------------------------------

def sorted_image_paths(class_dir: Path) -> List[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def scan_trainval(data_root: Path) -> Tuple[List[str], List[str], List[int], Dict[str, int]]:
    trainval_dir = data_root / "Train and Validation Set"
    if not trainval_dir.exists():
        raise FileNotFoundError(f"Missing folder: {trainval_dir}")

    class_names = sorted(d.name for d in trainval_dir.iterdir() if d.is_dir())
    if len(class_names) != 5:
        raise RuntimeError(
            f"Expected 5 class folders under {trainval_dir}; found {len(class_names)}: {class_names}"
        )

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    rel_paths: List[str] = []
    classes: List[str] = []
    labels: List[int] = []

    for class_name in class_names:
        class_dir = trainval_dir / class_name
        paths = sorted_image_paths(class_dir)
        if not paths:
            raise RuntimeError(f"No images found in class folder: {class_dir}")
        for path in paths:
            rel_paths.append(path.relative_to(data_root).as_posix())
            classes.append(class_name)
            labels.append(class_to_idx[class_name])

    return rel_paths, classes, labels, class_to_idx


def dataset_signature(rel_paths: Sequence[str], labels: Sequence[int]) -> str:
    h = hashlib.sha256()
    for rel_path, label in zip(rel_paths, labels):
        h.update(f"{rel_path}\t{label}\n".encode("utf-8"))
    return h.hexdigest()


def create_or_validate_manifest(
    data_root: Path,
    manifest_dir: Path,
    n_splits: int,
    seed: int,
) -> Tuple[List[Sample], Dict[str, int]]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifest_dir / "lc25000_five_fold_manifest.csv"
    meta_json = manifest_dir / "lc25000_five_fold_manifest_meta.json"

    rel_paths, classes, labels, class_to_idx = scan_trainval(data_root)
    signature = dataset_signature(rel_paths, labels)

    if manifest_csv.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        expected = {
            "n_splits": n_splits,
            "seed": seed,
            "dataset_signature": signature,
            "num_samples": len(rel_paths),
            "class_to_idx": class_to_idx,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(
                    f"Existing manifest mismatch for '{key}'.\n"
                    f"Existing: {meta.get(key)}\nCurrent:  {value}\n"
                    "Do not overwrite silently. Use a new output directory if intentionally changing the protocol."
                )

        samples: List[Sample] = []
        with manifest_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(
                    Sample(
                        rel_path=row["rel_path"],
                        class_name=row["class_name"],
                        label=int(row["label"]),
                        fold=int(row["fold"]),
                    )
                )

        if len(samples) != len(rel_paths):
            raise RuntimeError("Manifest row count does not match current dataset.")
        missing = [s.rel_path for s in samples if not (data_root / s.rel_path).exists()]
        if missing:
            raise RuntimeError(f"Manifest contains missing files. First missing: {missing[0]}")
        print(f"[MANIFEST] Reusing verified manifest: {manifest_csv}")
        return samples, class_to_idx

    labels_np = np.asarray(labels, dtype=np.int64)
    indices = np.arange(len(labels_np))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_assignment = np.zeros(len(labels_np), dtype=np.int64)
    for fold_idx, (_, val_idx) in enumerate(skf.split(indices, labels_np), start=1):
        fold_assignment[val_idx] = fold_idx

    samples = [
        Sample(rel_path=rp, class_name=cn, label=int(y), fold=int(fold))
        for rp, cn, y, fold in zip(rel_paths, classes, labels, fold_assignment)
    ]

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rel_path", "class_name", "label", "fold"])
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))

    fold_counts = {
        str(fold): int(sum(s.fold == fold for s in samples))
        for fold in range(1, n_splits + 1)
    }
    meta = {
        "protocol": "stratified_image_level_five_fold",
        "n_splits": n_splits,
        "seed": seed,
        "dataset_signature": signature,
        "num_samples": len(samples),
        "class_to_idx": class_to_idx,
        "fold_counts": fold_counts,
        "lineage_limit": (
            "LC25000 contains offline-augmented derivatives; reliable source-image lineage IDs are not available. "
            "This manifest guarantees exact file-level disjointness between train/validation folds, not source-lineage grouping."
        ),
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[MANIFEST] Created once and locked: {manifest_csv}")
    return samples, class_to_idx


# -----------------------------------------------------------------------------
# Dataset / transforms
# -----------------------------------------------------------------------------

def train_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(15),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        samples: Sequence[Sample],
        transform: T.Compose,
    ):
        self.data_root = data_root
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        path = self.data_root / sample.rel_path
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, sample.label, sample.rel_path


class TestDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        class_to_idx: Dict[str, int],
        transform: T.Compose,
    ):
        self.data_root = data_root
        self.transform = transform
        test_dir = data_root / "Test Set"
        if not test_dir.exists():
            raise FileNotFoundError(f"Missing folder: {test_dir}")

        samples: List[Tuple[str, int]] = []
        for class_name, label in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
            class_dir = test_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing Test Set class folder: {class_dir}")
            for path in sorted_image_paths(class_dir):
                samples.append((path.relative_to(data_root).as_posix(), label))
        if not samples:
            raise RuntimeError(f"No test images found in {test_dir}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, label = self.samples[idx]
        image = Image.open(self.data_root / rel_path).convert("RGB")
        image = self.transform(image)
        return image, label, rel_path


# -----------------------------------------------------------------------------
# Models: explicit ImageNet-1K V1 initialization, five-class replacement head
# -----------------------------------------------------------------------------

def build_model(name: str, num_classes: int = 5) -> nn.Module:
    name = name.lower()

    if name == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        # Preserve EfficientNet's native dropout; replace only the final linear layer.
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenetv2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if name == "vit_b16":
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {name}. Choose one of {SUPPORTED_MODELS}")


# -----------------------------------------------------------------------------
# Metrics / evaluation
# -----------------------------------------------------------------------------

def macro_fnr_from_confusion(cm: np.ndarray) -> float:
    # Each row is one true class: FN = row sum - TP.
    true_counts = cm.sum(axis=1)
    tp = np.diag(cm)
    fn = true_counts - tp
    fnr = np.divide(fn, true_counts, out=np.zeros_like(fn, dtype=float), where=true_counts > 0)
    return float(np.mean(fnr))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, num_classes: int) -> dict:
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fnr_macro": macro_fnr_from_confusion(cm),
        "confusion_matrix": cm.tolist(),
    }

    try:
        y_bin = label_binarize(y_true, classes=labels)
        metrics["roc_auc_macro_ovr"] = float(
            roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc_macro_ovr"] = None

    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    num_classes: int,
) -> Tuple[float, dict, List[dict]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    probs_all: List[np.ndarray] = []
    rows: List[dict] = []

    autocast_enabled = amp and device.type == "cuda"

    for images, labels, rel_paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        probs_np = probs.float().cpu().numpy()

        y_true_all.extend(labels_np.tolist())
        y_pred_all.extend(preds_np.tolist())
        probs_all.extend(list(probs_np))

        for rel_path, y, pred, p in zip(rel_paths, labels_np, preds_np, probs_np):
            row = {
                "rel_path": rel_path,
                "true_label": int(y),
                "pred_label": int(pred),
            }
            for class_idx in range(num_classes):
                row[f"prob_class_{class_idx}"] = float(p[class_idx])
            rows.append(row)

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)
    probs_arr = np.asarray(probs_all, dtype=np.float64)
    metrics = compute_metrics(y_true, y_pred, probs_arr, num_classes=num_classes)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, metrics, rows


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    autocast_enabled = amp and device.type == "cuda"

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if autocast_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def save_rows_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_epoch_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def environment_info(device: torch.device) -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "sklearn": sklearn_version,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)
    return info


def parse_folds(text: str, n_splits: int) -> List[int]:
    if text.strip().lower() == "all":
        return list(range(1, n_splits + 1))
    folds = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    invalid = [f for f in folds if f < 1 or f > n_splits]
    if invalid:
        raise ValueError(f"Invalid folds {invalid}; valid range is 1..{n_splits}")
    return folds


# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------

def build_loaders_for_fold(
    data_root: Path,
    samples: Sequence[Sample],
    fold: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    loader_seed: int,
) -> Tuple[DataLoader, DataLoader, torch.Generator]:
    train_samples = [s for s in samples if s.fold != fold]
    val_samples = [s for s in samples if s.fold == fold]

    train_ds = ManifestDataset(data_root, train_samples, train_transform(image_size))
    val_ds = ManifestDataset(data_root, val_samples, eval_transform(image_size))

    generator = torch.Generator()
    generator.manual_seed(loader_seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, generator


def run_cv_fold(
    args: argparse.Namespace,
    data_root: Path,
    out_dir: Path,
    samples: Sequence[Sample],
    fold: int,
    device: torch.device,
) -> dict:
    fold_dir = out_dir / args.model / "cv" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = fold_dir / "fold_metrics.json"
    if metrics_path.exists() and not args.restart:
        print(f"[SKIP] {args.model} fold {fold}: completed metrics already exist.")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    fold_seed = args.seed + fold
    loader_seed = args.seed * 1000 + fold
    set_global_seed(fold_seed)

    train_loader, val_loader, loader_generator = build_loaders_for_fold(
        data_root=data_root,
        samples=samples,
        fold=fold,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loader_seed=loader_seed,
    )

    model = build_model(args.model, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    best_path = fold_dir / "best_model.pt"
    last_path = fold_dir / "last_state.pt"
    epoch_log = fold_dir / "epoch_log.csv"

    start_epoch = 1
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    if last_path.exists() and not args.restart:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_val_loss = float(state["best_val_loss"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        restore_rng_state(state.get("rng_state", {}), loader_generator)
        print(f"[RESUME] {args.model} fold {fold} from epoch {start_epoch}")

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.amp, scaler
        )
        val_loss, val_metrics, _ = evaluate_model(
            model, val_loader, criterion, device, args.amp, num_classes=5
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model,
                    "fold": fold,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "protocol": "standardized_lc25000_v1",
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - t0
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr": optimizer.param_groups[0]["lr"],
            "best_val_loss_so_far": best_val_loss,
            "best_epoch_so_far": best_epoch,
            "seconds": elapsed,
        }
        append_epoch_log(epoch_log, log_row)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if (args.amp and device.type == "cuda") else None,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "rng_state": capture_rng_state(loader_generator),
            },
            last_path,
        )

        print(
            f"[{args.model}][fold {fold}] epoch {epoch:02d} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"acc {val_metrics['accuracy']*100:.3f}% | f1 {val_metrics['f1_macro']*100:.3f}% | "
            f"best epoch {best_epoch} | no-improve {epochs_without_improvement}/{args.patience}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"[EARLY STOP] fold {fold} at epoch {epoch}; best epoch={best_epoch}")
            break

    if not best_path.exists():
        raise RuntimeError(f"Best checkpoint was not created for fold {fold}.")

    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    val_loss, val_metrics, prediction_rows = evaluate_model(
        model, val_loader, criterion, device, args.amp, num_classes=5
    )

    save_rows_csv(fold_dir / "validation_predictions.csv", prediction_rows)

    result = {
        "model": args.model,
        "fold": fold,
        "best_epoch": int(best_state["epoch"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "final_validation_loss_at_best_checkpoint": float(val_loss),
        **val_metrics,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize_cv(args: argparse.Namespace, out_dir: Path) -> Optional[dict]:
    model_cv_dir = out_dir / args.model / "cv"
    fold_results: List[dict] = []
    for fold in range(1, args.n_splits + 1):
        p = model_cv_dir / f"fold_{fold}" / "fold_metrics.json"
        if not p.exists():
            return None
        fold_results.append(json.loads(p.read_text(encoding="utf-8")))

    scalar_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "mcc",
        "fnr_macro",
        "roc_auc_macro_ovr",
    ]
    summary = {
        "model": args.model,
        "n_splits": args.n_splits,
        "folds": fold_results,
        "best_epochs": [int(r["best_epoch"]) for r in fold_results],
    }
    summary["median_best_epoch"] = int(statistics.median(summary["best_epochs"]))

    for key in scalar_keys:
        vals = [r[key] for r in fold_results if r.get(key) is not None]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    summary_path = out_dir / args.model / "cv" / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== CV SUMMARY =====")
    print(f"Model: {args.model}")
    print(f"Best epochs: {summary['best_epochs']} -> median={summary['median_best_epoch']}")
    print(
        f"Accuracy: {summary['accuracy_mean']*100:.3f}% ± {summary['accuracy_std']*100:.3f}%"
    )
    print(
        f"Macro-F1: {summary['f1_macro_mean']*100:.3f}% ± {summary['f1_macro_std']*100:.3f}%"
    )
    return summary


# -----------------------------------------------------------------------------
# Final standardized source model + one-time packaged Test Set evaluation
# -----------------------------------------------------------------------------

def all_trainval_samples(samples: Sequence[Sample]) -> List[Sample]:
    return list(samples)


def run_final_model(
    args: argparse.Namespace,
    data_root: Path,
    out_dir: Path,
    samples: Sequence[Sample],
    class_to_idx: Dict[str, int],
    device: torch.device,
) -> dict:
    cv_summary_path = out_dir / args.model / "cv" / "cv_summary.json"
    if not cv_summary_path.exists():
        raise RuntimeError(
            f"Missing complete CV summary for {args.model}: {cv_summary_path}\n"
            "Complete all five folds before final training."
        )
    cv_summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))
    fixed_epochs = int(cv_summary["median_best_epoch"])
    if fixed_epochs < 1:
        raise RuntimeError(f"Invalid median best epoch: {fixed_epochs}")

    final_dir = out_dir / args.model / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    result_path = final_dir / "final_test_metrics.json"
    if result_path.exists() and not args.restart:
        print(f"[SKIP] Final run for {args.model} already completed.")
        return json.loads(result_path.read_text(encoding="utf-8"))

    set_global_seed(args.seed + 10_000)
    train_ds = ManifestDataset(
        data_root,
        all_trainval_samples(samples),
        train_transform(args.image_size),
    )
    test_ds = TestDataset(data_root, class_to_idx, eval_transform(args.image_size))

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed * 1000 + 999)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )

    model = build_model(args.model, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    last_path = final_dir / "last_state.pt"
    model_path = final_dir / "final_model.pt"
    epoch_log = final_dir / "epoch_log.csv"
    start_epoch = 1

    if last_path.exists() and not args.restart:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        restore_rng_state(state.get("rng_state", {}), loader_generator)
        print(f"[RESUME FINAL] {args.model} from epoch {start_epoch}/{fixed_epochs}")

    # Important: no packaged Test Set evaluation occurs inside this training loop.
    for epoch in range(start_epoch, fixed_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.amp, scaler
        )
        elapsed = time.time() - t0
        append_epoch_log(
            epoch_log,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": elapsed,
            },
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if (args.amp and device.type == "cuda") else None,
                "rng_state": capture_rng_state(loader_generator),
                "fixed_epochs_from_cv": fixed_epochs,
            },
            last_path,
        )
        print(
            f"[{args.model}][FINAL] epoch {epoch:02d}/{fixed_epochs} | "
            f"train loss {train_loss:.4f}"
        )

    torch.save(
        {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "fixed_epochs_from_cv": fixed_epochs,
            "cv_best_epochs": cv_summary["best_epochs"],
            "protocol": "standardized_lc25000_v1",
        },
        model_path,
    )

    # One-time packaged Test Set evaluation, only after the final model is fixed.
    test_loss, test_metrics, prediction_rows = evaluate_model(
        model, test_loader, criterion, device, args.amp, num_classes=5
    )
    save_rows_csv(final_dir / "test_predictions.csv", prediction_rows)

    result = {
        "model": args.model,
        "training_epochs": fixed_epochs,
        "cv_best_epochs": cv_summary["best_epochs"],
        "test_loss": float(test_loss),
        **test_metrics,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n===== FINAL TEST RESULT =====")
    print(f"Model: {args.model}")
    print(f"Training epochs fixed from CV median: {fixed_epochs}")
    print(f"Accuracy: {result['accuracy']*100:.3f}%")
    print(f"Macro-F1: {result['f1_macro']*100:.3f}%")
    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standardized LC25000 architecture experiment")
    p.add_argument("--mode", choices=["cv", "final"], required=True)
    p.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    p.add_argument("--data_root", required=True, help="Path containing 'Train and Validation Set' and 'Test Set'")
    p.add_argument("--out_dir", required=True, help="Persistent output directory, ideally on Google Drive")
    p.add_argument("--folds", default="all", help="For CV: 'all' or comma-separated folds, e.g. 1,2")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision. If used, use it for every architecture in the comparison.",
    )
    p.add_argument(
        "--restart",
        action="store_true",
        help="Ignore completed/resume state and restart requested work from scratch.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_splits != 5:
        raise ValueError("This thesis protocol is locked to five folds (n_splits=5).")
    if args.image_size != 224:
        raise ValueError("This thesis protocol is locked to 224x224 inputs.")
    if args.batch_size != 16:
        raise ValueError("This standardized rerun protocol is locked to batch_size=16.")
    if not math.isclose(args.lr, 1e-4, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("This standardized rerun protocol is locked to learning_rate=1e-4.")
    if args.max_epochs != 50:
        raise ValueError("This standardized rerun protocol is locked to max_epochs=50.")
    if args.patience != 7:
        raise ValueError("This standardized rerun protocol is locked to patience=7.")
    if args.seed != 42:
        raise ValueError("This standardized rerun protocol is locked to seed=42.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARNING] CUDA GPU not detected. Deep training on CPU is not recommended.")

    manifest_dir = out_dir / "protocol"
    samples, class_to_idx = create_or_validate_manifest(
        data_root=data_root,
        manifest_dir=manifest_dir,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    run_config = {
        "protocol_name": "standardized_lc25000_v1",
        "mode": args.mode,
        "model": args.model,
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "loss": "CrossEntropyLoss",
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.patience,
        "checkpoint_criterion": "minimum_validation_loss",
        "n_splits": args.n_splits,
        "splitter": "StratifiedKFold(shuffle=True, random_state=42)",
        "seed": args.seed,
        "stain_normalization": False,
        "train_augmentation": ["RandomHorizontalFlip(p=0.5)", "RandomVerticalFlip(p=0.5)", "RandomRotation(15 degrees)"],
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "pretraining": "ImageNet-1K V1",
        "amp": bool(args.amp),
        "class_to_idx": class_to_idx,
        "environment": environment_info(device),
        "interpretation_boundary": (
            "LC25000 cross-validation is stratified at image-file level. Because reliable source-image lineage IDs for offline-augmented derivatives are unavailable, source-grouped independence cannot be guaranteed."
        ),
    }
    model_config_dir = out_dir / args.model
    model_config_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_config_dir / "run_config.json"
    if config_path.exists() and not args.restart:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        # Prevent silent protocol drift between sessions for comparable fields.
        locked_keys = [
            "image_size", "batch_size", "optimizer", "learning_rate", "loss",
            "max_epochs", "early_stopping_patience", "checkpoint_criterion",
            "n_splits", "splitter", "seed", "stain_normalization", "pretraining", "amp",
        ]
        drift = {
            k: (existing.get(k), run_config.get(k))
            for k in locked_keys
            if existing.get(k) != run_config.get(k)
        }
        if drift:
            raise RuntimeError(
                f"Protocol drift detected for {args.model}: {drift}\n"
                "Use the same settings for all runs, or intentionally start a new output directory."
            )
    else:
        config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("===== STANDARDIZED LC25000 PROTOCOL =====")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Samples in Train+Validation manifest: {len(samples)}")
    print(f"Batch size: {args.batch_size} | LR: {args.lr} | AMP: {args.amp}")

    if args.mode == "cv":
        requested_folds = parse_folds(args.folds, args.n_splits)
        for fold in requested_folds:
            run_cv_fold(args, data_root, out_dir, samples, fold, device)
        summary = summarize_cv(args, out_dir)
        if summary is None:
            print("[INFO] Not all five folds are complete yet; CV summary will be created automatically when they are.")
    else:
        run_final_model(args, data_root, out_dir, samples, class_to_idx, device)


if __name__ == "__main__":
    main()
