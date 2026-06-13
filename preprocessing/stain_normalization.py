import numpy as np
import cv2

def reinhard_normalize_rgb(
    img_rgb: np.ndarray,
    target_means=(50.0, 0.0, 0.0),
    target_stds=(10.0, 1.0, 1.0),
) -> np.ndarray:
    img = img_rgb.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    means = lab.reshape(-1, 3).mean(axis=0)
    stds = lab.reshape(-1, 3).std(axis=0) + 1e-8

    target_means = np.array(target_means, dtype=np.float32)
    target_stds = np.array(target_stds, dtype=np.float32)

    lab_norm = (lab - means) / stds
    lab_norm = lab_norm * target_stds + target_means

    lab_norm = np.clip(lab_norm, 0, 255).astype(np.uint8)
    rgb_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)
    return rgb_norm