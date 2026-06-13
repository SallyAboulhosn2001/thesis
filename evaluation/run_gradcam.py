import sys
from pathlib import Path

# ---- PROJECT ROOT ----
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from datasets_code.dataset_lc25000 import LC25000Dataset
from models.student import get_efficientnet_b0_student
from models.densenet import get_densenet121
from models.backbone import get_efficientnet_b3
from xai.gradcam_simple import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------------------------------
# 1️⃣ Load one image automatically from test set
# --------------------------------------------------

dataset = LC25000Dataset(split="test")
image_tensor, label = dataset[0]

# Add batch dimension
input_tensor = image_tensor.unsqueeze(0).to(device)

# Convert tensor back to PIL for visualization
image_np = image_tensor.permute(1, 2, 0).numpy()
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

# --------------------------------------------------
# 2️⃣ Load Models
# --------------------------------------------------

b3 = get_efficientnet_b3(num_classes=5)
b3.load_state_dict(torch.load("baseline_efficientnet_b3_lc25000.pth", map_location=device))
b3 = b3.to(device)
b3.eval()

b0 = get_efficientnet_b0_student(num_classes=5)
b0.load_state_dict(torch.load("student_efficientnet_b0_kd_lc25000.pth", map_location=device))
b0 = b0.to(device)
b0.eval()

dn = get_densenet121(num_classes=5)
dn.load_state_dict(torch.load("densenet121_lc25000.pth", map_location=device))
dn = dn.to(device)
dn.eval()

# --------------------------------------------------
# 3️⃣ Target layers (last conv blocks)
# --------------------------------------------------

target_b3 = b3.features[-1]
target_b0 = b0.features[-1]
target_dn = dn.features[-1]

# --------------------------------------------------
# 4️⃣ GradCAM
# --------------------------------------------------

cam_b3 = GradCAM(b3, target_b3)
cam_b0 = GradCAM(b0, target_b0)
cam_dn = GradCAM(dn, target_dn)

heatmap_b3 = cam_b3.generate(input_tensor)
heatmap_b0 = cam_b0.generate(input_tensor)
heatmap_dn = cam_dn.generate(input_tensor)

# --------------------------------------------------
# 5️⃣ Plot
# --------------------------------------------------

fig, ax = plt.subplots(1, 4, figsize=(16, 5))

ax[0].imshow(image_np)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(image_np)
ax[1].imshow(heatmap_b3, cmap='jet', alpha=0.5)
ax[1].set_title("EfficientNet-B3")
ax[1].axis("off")

ax[2].imshow(image_np)
ax[2].imshow(heatmap_b0, cmap='jet', alpha=0.5)
ax[2].set_title("B0 + KD")
ax[2].axis("off")

ax[3].imshow(image_np)
ax[3].imshow(heatmap_dn, cmap='jet', alpha=0.5)
ax[3].set_title("DenseNet121")
ax[3].axis("off")

plt.tight_layout()
plt.show()
