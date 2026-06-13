from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CRCVALDataset(Dataset):
    def __init__(self, root_dir="datasets/CRC-VAL-HE-7K/CRC-VAL-HE-7K"):
        self.root_dir = Path(root_dir)

        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )

        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        self.samples = []

        for cls in self.classes:
            cls_folder = self.root_dir / cls
            for img_path in cls_folder.glob("*.*"):
                self.samples.append((img_path, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError("No CRC-VAL images found.")

        print(f"[CRC-VAL] Samples: {len(self.samples)}")
        print(f"[CRC-VAL] Classes: {self.class_to_idx}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label