import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class GLASDataset(Dataset):
    def __init__(self, root_dir="datasets/Glas/Warwick_QU_Dataset"):
        self.root = Path(root_dir)

        grade_csv = self.root / "Grade.csv"
        df = pd.read_csv(grade_csv)

        self.samples = []
        self.labels = []

        for _, row in df.iterrows():
            name = row["name"]
            label = row[" grade (GlaS)"].strip()

            img_path = self.root / f"{name}.bmp"

            if not img_path.exists():
                continue

            # benign = 0, malignant = 1
            y = 0 if label == "benign" else 1

            self.samples.append(img_path)
            self.labels.append(y)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"[GlaS] Samples: {len(self.samples)}")
        print(f"[GlaS] Classes: {{'benign': 0, 'malignant': 1}}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label