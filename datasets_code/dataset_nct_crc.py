from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from preprocessing.transforms import get_basic_transforms


class NCTCRCDataset(Dataset):
    def __init__(self, transform=None):
        # NOTE: dataset is nested one level deeper
        self.root_dir = Path(
            "datasets/NCT-CRC-HE-100K-NONORM/NCT-CRC-HE-100K-NONORM"
        )

        if not self.root_dir.exists():
            raise RuntimeError(f"NCT-CRC path not found: {self.root_dir}")

        self.transform = transform or get_basic_transforms(train=False)

        # ---- CLASSES ----
        self.class_names = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        # ---- STORAGE ----
        self.image_paths = []
        self.labels = []

        for cls in self.class_names:
            cls_dir = self.root_dir / cls
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                for img_path in cls_dir.glob(ext):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

        print(f"[NCT-CRC] Samples: {len(self.image_paths)}")
        print(f"[NCT-CRC] Classes: {self.class_to_idx}")

        if len(self.image_paths) == 0:
            raise RuntimeError("No NCT-CRC images found.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
