# evaluation/xai_case_miner_glas.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # thesis_code/
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import efficientnet_b3

import cv2


# -----------------------------
# Grad-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)

        def _backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_full_backward_hook(_backward_hook)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def generate(self, x, class_idx: int):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        loss = out[:, class_idx].sum()
        loss.backward()

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = grads.mean(dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cam


def overlay_cam(img_rgb_uint8, cam_224, alpha=0.45):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_224), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + img_rgb_uint8 * (1 - alpha)).astype(np.uint8)
    return overlay


# -----------------------------
# GlaS dataset
# -----------------------------
class GlasDataset(Dataset):
    def __init__(self, warwick_dir, grade_csv, transform, split="all"):
        self.warwick_dir = Path(warwick_dir)
        self.transform = transform

        df = pd.read_csv(grade_csv)
        df.columns = [c.strip() for c in df.columns]

        name_col = "name"
        label_col = "grade (GlaS)"

        if split == "train":
            df = df[df[name_col].str.startswith("train_")]
        elif split == "testA":
            df = df[df[name_col].str.startswith("testA_")]
        elif split == "testB":
            df = df[df[name_col].str.startswith("testB_")]
        elif split == "all":
            pass
        else:
            raise ValueError("split must be one of: all, train, testA, testB")

        def map_label(s):
            s = str(s).strip().lower()
            if "benign" in s:
                return 0
            if "malignant" in s:
                return 1
            raise ValueError(f"Unknown label: {s}")

        self.samples = []
        for _, row in df.iterrows():
            stem = row[name_col].strip()
            img_path = self.warwick_dir / f"{stem}.bmp"
            if not img_path.exists():
                continue
            y = map_label(row[label_col])
            self.samples.append((str(img_path), y))

        if len(self.samples) == 0:
            raise RuntimeError("No GlaS samples found. Check paths.")

        print(f"[GlaS] Samples: {len(self.samples)} | split={split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y, path


MALIGNANT_IDX = [0, 2, 4]
BENIGN_IDX = [1, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--warwick_dir", default="datasets/Glas/Warwick_QU_Dataset")
    ap.add_argument("--grade_csv", default="datasets/Glas/Warwick_QU_Dataset/Grade.csv")
    ap.add_argument("--split", default="all", choices=["all", "train", "testA", "testB"])
    ap.add_argument("--out_dir", default="xai_cases_glas")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--malignant_threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    correct_dir = out_dir / "correct"
    borderline_dir = out_dir / "borderline"
    fn_dir = out_dir / "false_negatives"
    wrong_dir = out_dir / "wrong"

    correct_dir.mkdir(exist_ok=True)
    borderline_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)
    wrong_dir.mkdir(exist_ok=True)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    ds = GlasDataset(args.warwick_dir, args.grade_csv, tfm, split=args.split)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if isinstance(state, dict) and "classifier.weight" in state and "classifier.1.weight" not in state:
        state["classifier.1.weight"] = state.pop("classifier.weight")
        state["classifier.1.bias"] = state.pop("classifier.bias")

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    cam_engine = GradCAM(model, model.features[-1])
    softmax = nn.Softmax(dim=1)

    records = []

    with torch.no_grad():
        for xb, yb, paths in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs5 = softmax(logits).cpu().numpy()

            p_mal = probs5[:, MALIGNANT_IDX].sum(axis=1)
            p_ben = probs5[:, BENIGN_IDX].sum(axis=1)

            probs2 = np.stack([p_ben, p_mal], axis=1)

            pred2 = (probs2[:, 1] >= args.malignant_threshold).astype(int)
            conf2 = np.where(pred2 == 1, probs2[:, 1], probs2[:, 0])

            for i in range(len(pred2)):
                records.append({
                    "path": paths[i],
                    "y_true": int(yb[i]),
                    "y_pred": int(pred2[i]),
                    "conf": float(conf2[i]),
                    "p_mal": float(probs2[i, 1]),
                    "p_ben": float(probs2[i, 0]),
                })

    correct = [r for r in records if r["y_true"] == r["y_pred"]]
    wrong = [r for r in records if r["y_true"] != r["y_pred"]]
    false_negs = [r for r in wrong if r["y_true"] == 1 and r["y_pred"] == 0]

    correct_top = sorted(correct, key=lambda r: r["conf"], reverse=True)[:args.top_k]
    borderline_top = sorted(correct, key=lambda r: r["conf"])[:args.top_k]
    fn_top = sorted(false_negs, key=lambda r: r["conf"], reverse=True)[:args.top_k]
    wrong_top = sorted(wrong, key=lambda r: r["conf"], reverse=True)[:args.top_k]

    print(f"Threshold (malignant): {args.malignant_threshold}")
    print(f"Total samples: {len(records)}")
    print(f"Correct: {len(correct)} | Wrong: {len(wrong)} | False Negatives: {len(false_negs)}")
    print(f"Saving: {len(correct_top)} correct, {len(borderline_top)} borderline, {len(fn_top)} false negatives, {len(wrong_top)} wrong")

    def pick_5class_target_from_binary(pred_bin: int, out5: np.ndarray) -> int:
        if pred_bin == 1:
            return MALIGNANT_IDX[int(np.argmax(out5[MALIGNANT_IDX]))]
        return BENIGN_IDX[int(np.argmax(out5[BENIGN_IDX]))]

    def save_overlay(sample, out_folder: Path, tag: str):
        img_path = sample["path"]
        y_true = sample["y_true"]
        y_pred = sample["y_pred"]
        conf = sample["conf"]
        p_mal = sample["p_mal"]

        pil = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(pil).astype(np.uint8)

        x = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            out5 = model(x).detach().cpu().numpy()[0]

        cls_idx = pick_5class_target_from_binary(y_pred, out5)

        torch.set_grad_enabled(True)
        cam = cam_engine.generate(x, int(cls_idx))
        torch.set_grad_enabled(False)

        ov = overlay_cam(img_np, cam)

        name = (
            f"{tag}_true{y_true}_pred{y_pred}_"
            f"conf{conf:.2f}_pmal{p_mal:.2f}_{Path(img_path).stem}.png"
        )

        cv2.imwrite(str(out_folder / name), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    for i, s in enumerate(correct_top):
        save_overlay(s, correct_dir, f"correct{i:02d}")

    for i, s in enumerate(borderline_top):
        save_overlay(s, borderline_dir, f"borderline{i:02d}")

    for i, s in enumerate(fn_top):
        save_overlay(s, fn_dir, f"fn{i:02d}")

    for i, s in enumerate(wrong_top):
        save_overlay(s, wrong_dir, f"wrong{i:02d}")

    print(f"✅ Saved GlaS XAI cases to: {args.out_dir}")


if __name__ == "__main__":
    main()