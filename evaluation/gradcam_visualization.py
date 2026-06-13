import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import argparse

import torchvision.transforms as T
from torchvision.models import efficientnet_b3


# -----------------------------
# Grad-CAM
# -----------------------------
class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx):

        self.model.zero_grad()

        output = self.model(x)
        loss = output[:, class_idx]
        loss.backward()

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = torch.mean(grads, dim=(1, 2))

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        return cam


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--output", type=str, default="gradcam.png")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, args.num_classes)

    state = torch.load(args.checkpoint, map_location=device)

    if "classifier.weight" in state:
        state["classifier.1.weight"] = state.pop("classifier.weight")
        state["classifier.1.bias"] = state.pop("classifier.bias")

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Target layer (IMPORTANT)
    target_layer = model.features[-1]

    gradcam = GradCAM(model, target_layer)

    output = model(x)
    pred_class = torch.argmax(output, dim=1).item()

    cam = gradcam.generate(x, pred_class)

    img_np = np.array(img.resize((224, 224)))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * 0.4 + img_np * 0.6
    overlay = overlay.astype(np.uint8)

    cv2.imwrite(args.output, overlay)

    print(f"\nSaved Grad-CAM: {args.output}")
    print(f"Predicted class: {pred_class}")


if __name__ == "__main__":
    main()