# evaluation/standardized_target_probe.py
"""
Standardized cross-dataset frozen-feature probing for the thesis.

Protocol:
- Load an LC25000-trained final model.
- Freeze the backbone and remove/replace the 5-class LC25000 head.
- Extract native pooled features from a target ImageFolder dataset.
- Evaluate Logistic Regression over five stratified 80/20 splits
  with seeds 1,2,3,4,5.
- Metrics: accuracy, macro precision/recall/F1, macro OVR ROC-AUC,
  MCC, and macro false-negative rate.
- Optionally evaluate a matched ImageNet-only DenseNet121 control.

Expected LC25000 final checkpoints:
  runs/standardized_lc25000/<model>/final/final_model.pt

Supported LC25000-trained sources:
  densenet121, efficientnet_b3, resnet50, mobilenetv2, vit_b16

Target dataset layout:
  <dataset_root>/
      class_a/
          image1.png
          ...
      class_b/
          ...

The script can also resolve one obvious nested ImageFolder root automatically.
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import random
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import (
    DenseNet121_Weights,
    densenet121,
    efficientnet_b3,
    mobilenet_v2,
    resnet50,
    vit_b_16,
)

import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SOURCES = [
    "densenet121",
    "efficientnet_b3",
    "resnet50",
    "mobilenetv2",
    "vit_b16",
]

EXPECTED_FEATURE_DIMS = {
    "densenet121": 1024,
    "efficientnet_b3": 1536,
    "resnet50": 2048,
    "mobilenetv2": 1280,
    "vit_b16": 768,
    "imagenet_densenet121": 1024,
}

SEEDS = [1, 2, 3, 4, 5]
TEST_SIZE = 0.20
LR_MAX_ITER = 3000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".ppm", ".pgm",
}


def set_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True, warn_only=False)

    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def has_images(directory: Path) -> bool:
    try:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except OSError:
        return False
    return False


def looks_like_imagefolder_root(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    class_dirs = [p for p in directory.iterdir() if p.is_dir()]
    image_class_dirs = [p for p in class_dirs if has_images(p)]
    return len(image_class_dirs) >= 2


def resolve_imagefolder_root(root: Path) -> Path:
    root = root.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if looks_like_imagefolder_root(root):
        return root

    candidates = []
    for child in root.iterdir():
        if child.is_dir() and looks_like_imagefolder_root(child):
            candidates.append(child)

    if len(candidates) == 1:
        print(f"[DATASET] Auto-resolved ImageFolder root: {candidates[0]}")
        return candidates[0]

    second_level = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            grandchildren = [p for p in child.iterdir() if p.is_dir()]
        except OSError:
            continue
        for grandchild in grandchildren:
            if looks_like_imagefolder_root(grandchild):
                second_level.append(grandchild)

    all_candidates = candidates + second_level
    unique_candidates = []
    seen = set()

    for p in all_candidates:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            unique_candidates.append(p.resolve())

    if len(unique_candidates) == 1:
        print(f"[DATASET] Auto-resolved ImageFolder root: {unique_candidates[0]}")
        return unique_candidates[0]

    msg = [
        f"Could not uniquely resolve an ImageFolder root from: {root}",
        "Expected a directory whose immediate subdirectories are target classes.",
    ]

    if unique_candidates:
        msg.append("Possible candidates:")
        msg.extend(f"  - {p}" for p in unique_candidates[:20])

    raise RuntimeError("\n".join(msg))


class PathImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path


def build_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_lc_model(model_name: str, num_classes: int = 5) -> nn.Module:
    if model_name == "densenet121":
        model = densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
        )

    elif model_name == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )

    elif model_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif model_name == "mobilenetv2":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )

    elif model_name == "vit_b16":
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            num_classes,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "model_state",
            "net",
            "weights",
        ):
            value = obj.get(key)

            if isinstance(value, dict) and value:
                return value

        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj

    raise RuntimeError(
        "Could not find a model state_dict in checkpoint. "
        "Expected a raw state_dict or a dict containing "
        "model_state_dict/state_dict/model."
    )


def normalize_state_dict_keys(state):
    normalized = {}

    for key, value in state.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        normalized[new_key] = value

    return normalized


def replace_head_with_identity(
    model_name: str,
    model: nn.Module,
) -> None:

    if model_name == "densenet121":
        model.classifier = nn.Identity()

    elif model_name == "efficientnet_b3":
        model.classifier = nn.Identity()

    elif model_name == "resnet50":
        model.fc = nn.Identity()

    elif model_name == "mobilenetv2":
        model.classifier = nn.Identity()

    elif model_name == "vit_b16":
        model.heads = nn.Identity()

    else:
        raise ValueError(model_name)


def load_lc_feature_extractor(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint_path = checkpoint_path.expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing final checkpoint: {checkpoint_path}"
        )

    model = build_lc_model(
        model_name,
        num_classes=5,
    )

    checkpoint_obj = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state = normalize_state_dict_keys(
        unwrap_state_dict(checkpoint_obj)
    )

    try:
        model.load_state_dict(
            state,
            strict=True,
        )

    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint does not strictly match torchvision "
            f"{model_name} with a 5-class head:\n"
            f"{checkpoint_path}\n\n"
            f"Original error:\n{e}"
        ) from e

    replace_head_with_identity(
        model_name,
        model,
    )

    for p in model.parameters():
        p.requires_grad_(False)

    model.eval()
    model.to(device)

    return model, sha256_file(checkpoint_path)


def build_imagenet_densenet_control(
    device: torch.device,
):
    model = densenet121(
        weights=DenseNet121_Weights.IMAGENET1K_V1
    )

    model.classifier = nn.Identity()

    for p in model.parameters():
        p.requires_grad_(False)

    model.eval()
    model.to(device)

    return model


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_dim: int,
):
    feature_chunks = []
    label_chunks = []
    paths = []

    with torch.inference_mode():

        for batch_idx, (
            xb,
            yb,
            batch_paths,
        ) in enumerate(
            loader,
            start=1,
        ):

            xb = xb.to(
                device,
                non_blocking=True,
            )

            feats = model(xb)

            if feats.ndim != 2:
                feats = torch.flatten(
                    feats,
                    1,
                )

            if feats.shape[1] != expected_dim:
                raise RuntimeError(
                    f"Feature dimension mismatch: "
                    f"expected {expected_dim}, "
                    f"got {feats.shape[1]}."
                )

            feature_chunks.append(
                feats
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )

            label_chunks.append(
                yb
                .numpy()
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            paths.extend(
                str(p)
                for p in batch_paths
            )

            if (
                batch_idx % 100 == 0
                or batch_idx == len(loader)
            ):
                done = min(
                    batch_idx * loader.batch_size,
                    len(loader.dataset),
                )

                print(
                    f"[FEATURES] "
                    f"{done}/{len(loader.dataset)} "
                    f"({100.0 * done / len(loader.dataset):.1f}%)"
                )

    X = np.concatenate(
        feature_chunks,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    y = np.concatenate(
        label_chunks,
        axis=0,
    ).astype(
        np.int64,
        copy=False,
    )

    paths = np.asarray(
        paths,
        dtype=object,
    )

    if len(X) != len(loader.dataset):
        raise RuntimeError(
            f"Feature extraction count mismatch: "
            f"{len(X)} vs {len(loader.dataset)}"
        )

    return X, y, paths


def macro_fnr(
    y_true,
    y_pred,
    labels,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    actual = cm.sum(axis=1)
    tp = np.diag(cm)
    fn = actual - tp

    fnr = np.divide(
        fn,
        actual,
        out=np.zeros_like(
            fn,
            dtype=np.float64,
        ),
        where=actual != 0,
    )

    return float(
        np.mean(fnr)
    )


def sanitize_column_name(
    name: str,
) -> str:

    name = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        name.strip(),
    )

    return (
        name.strip("_")
        or "class"
    )


def evaluate_logistic_probe(
    X,
    y,
    paths,
    class_names,
    source_name: str,
    out_dir: Path,
):
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_dir = (
        out_dir
        / "predictions"
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = np.arange(
        len(class_names),
        dtype=np.int64,
    )

    per_seed = []

    for seed in SEEDS:

        indices = np.arange(
            len(y)
        )

        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y,
        )

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        clf = LogisticRegression(
            max_iter=LR_MAX_ITER,
            n_jobs=-1,
            random_state=seed,
        )

        with warnings.catch_warnings(
            record=True,
        ) as caught:

            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            clf.fit(
                X_train,
                y_train,
            )

        convergence_warnings = [
            str(w.message)
            for w in caught
            if issubclass(
                w.category,
                ConvergenceWarning,
            )
        ]

        y_pred = clf.predict(
            X_test
        )

        probs = clf.predict_proba(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        (
            precision,
            recall,
            f1,
            _,
        ) = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        auc = roc_auc_score(
            y_test,
            probs,
            labels=labels,
            multi_class="ovr",
            average="macro",
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred,
        )

        fnr = macro_fnr(
            y_test,
            y_pred,
            labels,
        )

        metric_row = {
            "source": source_name,
            "seed": seed,
            "n_train": int(
                len(train_idx)
            ),
            "n_test": int(
                len(test_idx)
            ),
            "accuracy": float(
                accuracy
            ),
            "macro_precision": float(
                precision
            ),
            "macro_recall": float(
                recall
            ),
            "macro_f1": float(
                f1
            ),
            "macro_ovr_auc": float(
                auc
            ),
            "mcc": float(
                mcc
            ),
            "macro_fnr": float(
                fnr
            ),
            "max_n_iter": int(
                np.max(
                    clf.n_iter_
                )
            ),
            "convergence_warning": bool(
                convergence_warnings
            ),
        }

        per_seed.append(
            metric_row
        )

        pred_data = {
            "dataset_index": test_idx,
            "path": paths[test_idx],
            "y_true": y_test,
            "y_true_class": [
                class_names[i]
                for i in y_test
            ],
            "y_pred": y_pred,
            "y_pred_class": [
                class_names[i]
                for i in y_pred
            ],
        }

        for (
            j,
            class_name,
        ) in enumerate(
            class_names
        ):

            safe = sanitize_column_name(
                class_name
            )

            pred_data[
                f"prob_{j}_{safe}"
            ] = probs[:, j]

        pred_df = pd.DataFrame(
            pred_data
        )

        pred_path = (
            pred_dir
            / f"seed_{seed}_predictions.csv"
        )

        pred_df.to_csv(
            pred_path,
            index=False,
        )

        print(
            f"[{source_name}]"
            f"[seed {seed}] "
            f"acc={accuracy * 100:.3f}% | "
            f"macro-F1={f1 * 100:.3f}% | "
            f"AUC={auc * 100:.3f}% | "
            f"MCC={mcc:.4f} | "
            f"macro-FNR={fnr * 100:.3f}%"
        )

        if convergence_warnings:
            print(
                f"[WARNING] Logistic Regression emitted "
                f"{len(convergence_warnings)} "
                f"convergence warning(s) "
                f"for seed {seed}."
            )

    metrics_df = pd.DataFrame(
        per_seed
    )

    metrics_df.to_csv(
        out_dir
        / "per_seed_metrics.csv",
        index=False,
    )

    summary_metrics = {}

    for metric in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_ovr_auc",
        "mcc",
        "macro_fnr",
    ]:

        values = metrics_df[
            metric
        ].to_numpy(
            dtype=float
        )

        summary_metrics[
            metric
        ] = {
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
        }

    return (
        per_seed,
        summary_metrics,
    )


def parse_sources(
    value: str,
):
    value = (
        value
        .strip()
        .lower()
    )

    if value == "all":
        return list(
            SOURCES
        )

    requested = [
        v.strip()
        for v
        in value.split(",")
        if v.strip()
    ]

    unknown = [
        v
        for v
        in requested
        if v not in SOURCES
    ]

    if unknown:
        raise ValueError(
            f"Unknown source(s): "
            f"{unknown}. "
            f"Valid values: "
            f"{SOURCES} or 'all'."
        )

    return requested


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Standardized frozen-feature "
            "target-dataset probes."
        )
    )

    ap.add_argument(
        "--dataset_name",
        required=True,
        help=(
            "Short output label, "
            "e.g. nct or crc."
        ),
    )

    ap.add_argument(
        "--dataset_root",
        required=True,
        help=(
            "Target dataset ImageFolder root, "
            "or a parent containing "
            "one obvious root."
        ),
    )

    ap.add_argument(
        "--lc_runs_root",
        default=(
            "runs/"
            "standardized_lc25000"
        ),
        help=(
            "Root containing the five "
            "standardized LC25000 "
            "model folders."
        ),
    )

    ap.add_argument(
        "--out_root",
        default=(
            "runs/"
            "standardized_target_probes"
        ),
        help=(
            "Root for target-probe "
            "outputs."
        ),
    )

    ap.add_argument(
        "--sources",
        default="all",
        help=(
            "all or comma-separated subset of: "
            + ",".join(SOURCES)
        ),
    )

    ap.add_argument(
        "--include_imagenet_densenet_control",
        action="store_true",
        help=(
            "Also evaluate untouched torchvision "
            "ImageNet-1K V1 DenseNet121."
        ),
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    args = ap.parse_args()

    set_determinism(42)

    dataset_name = (
        args.dataset_name
        .strip()
        .lower()
    )

    dataset_root = (
        resolve_imagefolder_root(
            Path(
                args.dataset_root
            )
        )
    )

    lc_runs_root = (
        Path(
            args.lc_runs_root
        )
        .expanduser()
        .resolve()
    )

    out_root = (
        Path(
            args.out_root
        )
        .expanduser()
        .resolve()
        / dataset_name
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "===== STANDARDIZED TARGET "
        "FEATURE-PROBE PROTOCOL ====="
    )

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Dataset root: "
        f"{dataset_root}"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        "Split: stratified "
        "80/20 per seed"
    )

    print(
        "Image size: 224"
    )

    print(
        "Normalization: "
        "ImageNet mean/std"
    )

    print(
        f"LogisticRegression "
        f"max_iter: "
        f"{LR_MAX_ITER}"
    )

    transform = build_transform()

    dataset = PathImageFolder(
        dataset_root,
        transform=transform,
    )

    if len(dataset.classes) < 2:
        raise RuntimeError(
            "Target dataset must contain "
            "at least two classes."
        )

    print(
        f"Samples: {len(dataset)}"
    )

    print(
        f"Classes "
        f"({len(dataset.classes)}): "
        f"{dataset.classes}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    dataset_manifest = pd.DataFrame({
        "dataset_index": np.arange(
            len(dataset),
            dtype=np.int64,
        ),
        "path": [
            p
            for p, _
            in dataset.samples
        ],
        "label": [
            int(y)
            for _, y
            in dataset.samples
        ],
        "class_name": [
            dataset.classes[int(y)]
            for _, y
            in dataset.samples
        ],
    })

    dataset_manifest.to_csv(
        out_root
        / "dataset_manifest.csv",
        index=False,
    )

    sources = parse_sources(
        args.sources
    )

    run_sources = list(
        sources
    )

    if (
        args
        .include_imagenet_densenet_control
    ):
        run_sources.append(
            "imagenet_densenet121"
        )

    all_summaries = {}

    for source in run_sources:

        print()

        print(
            "=" * 72
        )

        print(
            f"SOURCE: {source}"
        )

        print(
            "=" * 72
        )

        source_out = (
            out_root
            / source
        )

        source_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = None
        checkpoint_sha256 = None

        if (
            source
            == "imagenet_densenet121"
        ):
            model = (
                build_imagenet_densenet_control(
                    device
                )
            )

            checkpoint_descriptor = (
                "torchvision DenseNet121 "
                "IMAGENET1K_V1"
            )

        else:
            checkpoint_path = (
                lc_runs_root
                / source
                / "final"
                / "final_model.pt"
            )

            (
                model,
                checkpoint_sha256,
            ) = load_lc_feature_extractor(
                source,
                checkpoint_path,
                device,
            )

            checkpoint_descriptor = str(
                checkpoint_path
            )

        expected_dim = (
            EXPECTED_FEATURE_DIMS[
                source
            ]
        )

        print(
            f"Feature dimension: "
            f"{expected_dim}"
        )

        print(
            f"Checkpoint/source: "
            f"{checkpoint_descriptor}"
        )

        X, y, paths = extract_features(
            model,
            loader,
            device,
            expected_dim=expected_dim,
        )

        print(
            f"[FEATURES] "
            f"shape={X.shape} | "
            f"dtype={X.dtype} | "
            f"labels={y.shape}"
        )

        (
            per_seed,
            summary_metrics,
        ) = evaluate_logistic_probe(
            X=X,
            y=y,
            paths=paths,
            class_names=dataset.classes,
            source_name=source,
            out_dir=source_out,
        )

        source_summary = {
            "dataset_name": (
                dataset_name
            ),
            "dataset_root": str(
                dataset_root
            ),
            "n_samples": int(
                len(dataset)
            ),
            "class_names": (
                dataset.classes
            ),
            "source": source,
            "feature_dim": int(
                expected_dim
            ),
            "checkpoint": (
                checkpoint_descriptor
            ),
            "checkpoint_sha256": (
                checkpoint_sha256
            ),
            "seeds": SEEDS,
            "test_size": TEST_SIZE,
            "logistic_regression": {
                "max_iter": (
                    LR_MAX_ITER
                ),
                "n_jobs": -1,
                "scaling": "none",
            },
            "transform": {
                "resize": [
                    224,
                    224,
                ],
                "to_tensor": True,
                "normalize_mean": (
                    IMAGENET_MEAN
                ),
                "normalize_std": (
                    IMAGENET_STD
                ),
            },
            "metrics": (
                summary_metrics
            ),
            "per_seed": (
                per_seed
            ),
            "environment": {
                "python": (
                    sys.version
                ),
                "torch": (
                    torch.__version__
                ),
                "torchvision": (
                    torchvision.__version__
                ),
                "sklearn": (
                    sklearn.__version__
                ),
                "numpy": (
                    np.__version__
                ),
                "device": str(
                    device
                ),
                "gpu": (
                    torch
                    .cuda
                    .get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "cublas_workspace_config": (
                    os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    )
                ),
            },
        }

        with (
            source_out
            / "summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                source_summary,
                f,
                indent=2,
            )

        all_summaries[
            source
        ] = source_summary

        acc = (
            summary_metrics[
                "accuracy"
            ]
        )

        f1 = (
            summary_metrics[
                "macro_f1"
            ]
        )

        print()

        print(
            f"===== "
            f"{source} "
            f"SUMMARY ====="
        )

        print(
            f"Accuracy: "
            f"{acc['mean'] * 100:.3f}% "
            f"± "
            f"{acc['std'] * 100:.3f}%"
        )

        print(
            f"Macro-F1: "
            f"{f1['mean'] * 100:.3f}% "
            f"± "
            f"{f1['std'] * 100:.3f}%"
        )

        del model, X, y, paths

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    compact_rows = []

    for (
        source,
        result,
    ) in all_summaries.items():

        row = {
            "source": (
                source
            ),
            "feature_dim": (
                result[
                    "feature_dim"
                ]
            ),
        }

        for (
            metric,
            stats,
        ) in result[
            "metrics"
        ].items():

            row[
                f"{metric}_mean"
            ] = (
                stats[
                    "mean"
                ]
            )

            row[
                f"{metric}_std"
            ] = (
                stats[
                    "std"
                ]
            )

        compact_rows.append(
            row
        )

    pd.DataFrame(
        compact_rows
    ).to_csv(
        out_root
        / "all_sources_summary.csv",
        index=False,
    )

    with (
        out_root
        / "all_sources_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_summaries,
            f,
            indent=2,
        )

    print()

    print(
        "===== TARGET PROBE "
        "COMPLETE ====="
    )

    print(
        f"Dataset: "
        f"{dataset_name}"
    )

    print(
        f"Outputs: "
        f"{out_root}"
    )


if __name__ == "__main__":
    main()