import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import efficientnet_b3

import cv2


# -----------------------------
# LC25000 Folder Dataset (Test Set)
# -----------------------------
class LC25000FolderDataset(Dataset):
    """
    Expects:
    datasets/LC25000/lung_colon_image_set/Test Set/
        colon_aca/
        colon_n/
        lung_aca/
        lung_n/
        lung_scc/
    """
    CLASS_TO_IDX = {
        "colon_aca": 0,
        "colon_n": 1,
        "lung_aca": 2,
        "lung_n": 3,
        "lung_scc": 4,
    }

    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for cls_name, cls_idx in self.CLASS_TO_IDX.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                for p in cls_dir.glob(ext):
                    self.samples.append((str(p), cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y, path


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

        # backward hook (older PyTorch-safe)
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

        grads = self.gradients[0]      # (C,H,W)
        acts = self.activations[0]     # (C,H,W)

        weights = grads.mean(dim=(1, 2))  # (C,)

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cam


def overlay_cam_on_image(img_rgb_uint8, cam_224, alpha=0.45):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_224), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + img_rgb_uint8 * (1 - alpha)).astype(np.uint8)
    return overlay


# -----------------------------
# LC25000 malignant/benign mapping (for false negatives)
# malignant classes: colon_aca (0), lung_aca (2), lung_scc (4)
# benign classes: colon_n (1), lung_n (3)
# -----------------------------
MALIGNANT = {0, 2, 4}
BENIGN = {1, 3}

def to_binary(label_5):
    return 1 if label_5 in MALIGNANT else 0


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="baseline_efficientnet_b3_lc25000.pth")
    parser.add_argument("--test_root", default="datasets/LC25000/lung_colon_image_set/Test Set")
    parser.add_argument("--out_dir", default="xai_cases")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=10)  # how many samples per category
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # output folders
    correct_dir = Path(args.out_dir) / "correct"
    borderline_dir = Path(args.out_dir) / "borderline"
    fn_dir = Path(args.out_dir) / "false_negatives"
    correct_dir.mkdir(parents=True, exist_ok=True)
    borderline_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    ds = LC25000FolderDataset(args.test_root, tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Load model
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    state = torch.load(args.checkpoint, map_location=device)

    # handle your saved checkpoint format (sometimes classifier.weight vs classifier.1.weight)
    if "classifier.weight" in state:
        state["classifier.1.weight"] = state.pop("classifier.weight")
        state["classifier.1.bias"] = state.pop("classifier.bias")

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # GradCAM target layer
    target_layer = model.features[-1]
    cam_engine = GradCAM(model, target_layer)
    softmax = nn.Softmax(dim=1)

    # Collect per-sample info
    records = []

    with torch.no_grad():
        for xb, yb, paths in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = softmax(logits).cpu().numpy()   # (B,5)
            pred = np.argmax(probs, axis=1)
            conf = probs[np.arange(len(pred)), pred]

            for i in range(len(pred)):
                records.append({
                    "path": paths[i],
                    "y_true": int(yb[i].item()),
                    "y_pred": int(pred[i]),
                    "conf": float(conf[i]),
                })

    # Split into groups
    correct = [r for r in records if r["y_pred"] == r["y_true"]]
    wrong = [r for r in records if r["y_pred"] != r["y_true"]]

    # Correct high-confidence: sort desc
    correct_sorted = sorted(correct, key=lambda r: r["conf"], reverse=True)
    correct_top = correct_sorted[:args.top_k]

    # Borderline correct: correct but lowest confidence
    borderline_sorted = sorted(correct, key=lambda r: r["conf"])
    borderline_top = borderline_sorted[:args.top_k]

    # False negatives (binary): true malignant but predicted benign
    false_negs = []
    for r in wrong:
        true_bin = to_binary(r["y_true"])
        pred_bin = to_binary(r["y_pred"])
        if true_bin == 1 and pred_bin == 0:
            false_negs.append(r)

    # sort false negatives by confidence (highest confidence wrong is interesting)
    false_negs = sorted(false_negs, key=lambda r: r["conf"], reverse=True)[:args.top_k]

    print(f"Total samples: {len(records)}")
    print(f"Correct: {len(correct)} | Wrong: {len(wrong)}")
    print(f"Saving: {len(correct_top)} correct, {len(borderline_top)} borderline, {len(false_negs)} false negatives")

    # Helper: save cam overlay
    def save_cam(sample, out_folder: Path, tag: str):
        img_path = sample["path"]
        y_true = sample["y_true"]
        y_pred = sample["y_pred"]

        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(img).astype(np.uint8)

        # Need gradients -> do a forward/backward with grad enabled
        x = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)

        out = model(x)
        pred_class = int(torch.argmax(out, dim=1).item())

        cam = cam_engine.generate(x, pred_class)
        overlay = overlay_cam_on_image(img_np, cam)

        out_name = f"{tag}_true{y_true}_pred{y_pred}_{Path(img_path).stem}.png"
        cv2.imwrite(str(out_folder / out_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Enable grads for GradCAM saving
    torch.set_grad_enabled(True)

    for i, s in enumerate(correct_top):
        save_cam(s, correct_dir, f"correct{i:02d}")

    for i, s in enumerate(borderline_top):
        save_cam(s, borderline_dir, f"borderline{i:02d}")

    for i, s in enumerate(false_negs):
        save_cam(s, fn_dir, f"fn{i:02d}")

    torch.set_grad_enabled(False)

    print(f"✅ Saved XAI cases to: {args.out_dir}")


if __name__ == "__main__":
    main()