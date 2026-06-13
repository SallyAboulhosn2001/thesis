import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.models import efficientnet_b3
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load model
# -------------------------
model = efficientnet_b3(weights=None)
model.classifier = torch.nn.Linear(model.classifier[1].in_features, 5)

state = torch.load("baseline_efficientnet_b3_lc25000.pth", map_location=device)

# Handle possible wrapper
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

model.load_state_dict(state, strict=True)


model = model.to(device)
model.eval()


# -------------------------
# Transform (must match training)
# -------------------------
transform = T.Compose([
    T.Resize((300,300)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# -------------------------
# GradCAM class
# -------------------------
class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, class_idx=None):

        output = self.model(input_image)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[:, class_idx]

        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1,2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (300,300))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


# -------------------------
# Select target layer
# -------------------------
target_layer = model.features[-1]

gradcam = GradCAM(model, target_layer)


# -------------------------
# Load image
# -------------------------
img_path = "C:/Users/sally/Desktop/thesis_code/datasets/LC25000/lung_colon_image_set/Test Set/colon_aca/colonca4273.jpeg"
img = Image.open(img_path).convert("RGB")

input_tensor = transform(img).unsqueeze(0).to(device)


# -------------------------
# Generate heatmap
# -------------------------
cam = gradcam.generate(input_tensor)

img_np = np.array(img.resize((300,300)))

heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
overlay = heatmap * 0.4 + img_np


# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("GradCAM")
plt.imshow(cam, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay.astype(np.uint8))
plt.axis("off")

plt.show()
