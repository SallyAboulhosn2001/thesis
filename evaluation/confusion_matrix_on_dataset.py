import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

import torchvision.transforms as T
from torchvision.models import densenet121

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.samples = []
        self.transform = transform

        self.classes = sorted(
            [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
        )

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        for class_name in self.classes:
            class_index = self.class_to_idx[class_name]
            class_dir = os.path.join(root, class_name)

            for filename in os.listdir(class_dir):
                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                ):
                    image_path = os.path.join(class_dir, filename)
                    self.samples.append((image_path, class_index))

        print("Classes:", self.class_to_idx)
        print("Samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--out",
        type=str,
        default="confusion_matrix_densenet121_lc25000.png"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = FolderDataset(
        args.dataset,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = densenet121(weights=None)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        len(dataset.classes)
    )

    state = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(
        state,
        strict=True
    )

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            predictions = torch.argmax(
                logits,
                dim=1
            ).cpu().numpy()

            y_pred.extend(predictions)
            y_true.extend(labels.numpy())

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(dataset.classes)))
    )

    print("\nConfusion Matrix:")
    print(cm)

    display_labels = [
        class_name.replace("_", " ")
        for class_name in dataset.classes
    ]

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    plt.title(
        "DenseNet121 Confusion Matrix on LC25000 Test Set"
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        args.out,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()