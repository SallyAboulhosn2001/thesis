import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.models import vit_b_16


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--output", default="vit_attention.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load image -----
    img = Image.open(args.image).convert("RGB")
    img_224 = img.resize((224, 224))

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    x = tfm(img).unsqueeze(0).to(device)

    # ----- Build ViT -----
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, args.num_classes)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # ----- Capture attention weights from last block -----
    attn_store = []

    last_block = model.encoder.layers[-1]
    sa = last_block.self_attention
    orig_forward = sa.forward

    def new_forward(query, key, value, **kwargs):
        # Force need_weights=True and average_attn_weights=False when supported
        kwargs["need_weights"] = True
        if "average_attn_weights" in kwargs:
            kwargs["average_attn_weights"] = False

        out = orig_forward(query, key, value, **kwargs)

        # out can be:
        # (attn_out, attn_weights) OR sometimes just attn_out
        if isinstance(out, tuple) and len(out) == 2:
            attn_out, attn_w = out
            attn_store.append(attn_w.detach().cpu())
            return attn_out, attn_w
        else:
            return out

    sa.forward = new_forward

    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

    # Restore
    sa.forward = orig_forward

    if len(attn_store) == 0:
        print("❌ Could not capture attention weights. Torchvision internals may differ.")
        return

    attn = attn_store[0]  # possible shapes:
    # (B, heads, T, T) OR (B, T, T) OR (heads, T, T) OR (T, T)

    # ----- Make it shape-robust -----
    if attn.dim() == 4:
        # (B, H, T, T)
        attn = attn[0].mean(0)  # -> (T, T)
    elif attn.dim() == 3:
        # could be (B, T, T) or (H, T, T)
        if attn.shape[0] == 1:
            attn = attn[0]  # (T, T)
        else:
            attn = attn.mean(0)  # (T, T)
    elif attn.dim() == 2:
        # (T, T)
        pass
    else:
        print(f"❌ Unexpected attention shape: {tuple(attn.shape)}")
        return

    T_tokens = attn.shape[0]  # includes CLS
    if T_tokens < 2:
        print("❌ Not enough tokens in attention map.")
        return

    # CLS token attention to patches
    cls_to_patches = attn[0, 1:]  # (num_patches,)
    num_patches = cls_to_patches.numel()

    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        print(f"❌ Patch count not a square: {num_patches}")
        return

    heat = cls_to_patches.reshape(grid_size, grid_size).numpy()
    heat = _normalize01(heat)

    # Upsample heat to 224×224 for overlay
    heat_up = np.kron(heat, np.ones((224 // grid_size, 224 // grid_size), dtype=np.float32))
    heat_up = heat_up[:224, :224]

    # ----- Plot -----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].imshow(img_224)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(img_224)
    ax[1].imshow(heat_up, cmap="jet", alpha=0.5)
    ax[1].set_title(f"ViT Attention (pred={pred})")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    main()