import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

import torchvision.transforms as T
from torchvision.models import efficientnet_b3

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Dataset
# -----------------------------
class FolderDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}

        classes = sorted(os.listdir(root))

        for i, cls in enumerate(classes):
            self.class_to_idx[cls] = i
            cls_dir = os.path.join(root, cls)

            for f in os.listdir(cls_dir):
                if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                    self.samples.append((os.path.join(cls_dir,f), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])

    dataset = FolderDataset(args.dataset, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, args.num_classes)

    state = torch.load(args.checkpoint, map_location=device)

    if "classifier.weight" in state:
        state["classifier.1.weight"] = state.pop("classifier.weight")
        state["classifier.1.bias"] = state.pop("classifier.bias")

    model.load_state_dict(state)

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x,y in loader:

            x = x.to(device)

            logits = model(x)
            preds = torch.argmax(logits,1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    cm = confusion_matrix(y_true,y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)

    print("\nSaved: confusion_matrix.png")


if __name__ == "__main__":
    main()