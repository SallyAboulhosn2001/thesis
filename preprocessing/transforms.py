import numpy as np
from PIL import Image
import torchvision.transforms as T

from preprocessing.stain_normalization import reinhard_normalize_rgb


class ReinhardTransform:
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("RGB")).astype(np.uint8)
        arr = reinhard_normalize_rgb(arr)
        return Image.fromarray(arr)


def get_basic_transforms(train=True, use_stain_norm=False):
    ops = []

    if use_stain_norm:
        ops.append(ReinhardTransform())

    if train:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
        ]

    ops += [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ]
    return T.Compose(ops)


def get_train_transforms(use_stain_norm=False):
    # Alias used by dataset_lc25000.py
    return get_basic_transforms(train=True, use_stain_norm=use_stain_norm)


def get_eval_transforms(image_size=(224, 224), use_stain_norm=False):
    ops = []

    if use_stain_norm:
        ops.append(ReinhardTransform())

    ops += [
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ]
    return T.Compose(ops)