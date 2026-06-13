from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset

from preprocessing.transforms import get_train_transforms, get_eval_transforms


class LC25000Dataset(Dataset):
    """
    Expects:
    datasets/LC25000/lung_colon_image_set/
        Train and Validation Set/<class folders>/
        Test Set/<class folders>/
    """

    def __init__(self, split="train", root_dir=None, transform=None, use_stain_norm=False):
        self.split = split.lower()

        if root_dir is None:
            base_dir = Path("datasets/LC25000/lung_colon_image_set")
        else:
            base_dir = Path(root_dir)

        if self.split == "train":
            self.data_dir = base_dir / "Train and Validation Set"
            default_transform = get_train_transforms(use_stain_norm=use_stain_norm)
        elif self.split == "test":
            self.data_dir = base_dir / "Test Set"
            default_transform = get_eval_transforms(use_stain_norm=use_stain_norm)
        else:
            raise ValueError("split must be 'train' or 'test'")

        if not self.data_dir.exists():
            raise RuntimeError(f"Dataset folder not found: {self.data_dir}")

        self.transform = transform if transform is not None else default_transform

        self.image_paths = []
        self.labels = []

        class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        if len(class_names) == 0:
            raise RuntimeError(f"No class folders found in: {self.data_dir}")

        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        for cls in class_names:
            cls_dir = self.data_dir / cls
            for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                for img_path in cls_dir.glob(ext):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.data_dir}")

        print(f"[LC25000] Split: {self.split}")
        print(f"[LC25000] Samples: {len(self.image_paths)}")
        print(f"[LC25000] Classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label