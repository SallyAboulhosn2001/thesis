import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torchvision.models import densenet121
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Dataset (LC25000)
# -----------------------------
class LC25000Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        class_to_idx = {
            'colon_aca': 0,
            'colon_n': 1,
            'lung_aca': 2,
            'lung_n': 3,
            'lung_scc': 4
        }

        for cls, idx in class_to_idx.items():
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, idx))

        if len(self.samples) == 0:
            raise RuntimeError("No LC25000 images found!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# -----------------------------
# MAIN
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚠️ IMPORTANT: update this path if needed
    dataset_path = "datasets/LC25000/lung_colon_image_set/Train and Validation Set"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = LC25000Dataset(dataset_path, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Samples:", len(dataset))

    # -----------------------------
    # Load DenseNet (feature extractor)
    # -----------------------------
    model = densenet121(weights=None)
    model.classifier = nn.Identity()  # remove classifier
    model.to(device)
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for xb, yb in tqdm(loader):
            xb = xb.to(device)
            feats = model(xb)

            features.append(feats.cpu().numpy())
            labels.append(yb.numpy())

    X = np.concatenate(features)
    y = np.concatenate(labels)

    torch.save({"X": X, "y": y}, "lc25000_features_densenet121.pt")

    print("\n✅ LC25000 features saved")
    print("Shape:", X.shape)


if __name__ == "__main__":
    main()