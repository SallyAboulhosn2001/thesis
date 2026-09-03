import numpy as np
import cv2


def reinhard_normalize_rgb(
    img_rgb: np.ndarray,
    target_means=(50.0, 0.0, 0.0),
    target_stds=(10.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Apply Reinhard-style color normalization in standard CIELAB coordinates.

    OpenCV stores 8-bit LAB differently from conventional CIELAB:
      - L is encoded on [0, 255] instead of approximately [0, 100]
      - a and b are shifted by +128

    Therefore, OpenCV's uint8 LAB output is explicitly converted to standard
    CIELAB coordinates before computing channel statistics and applying the
    target means/stds. The normalized CIELAB image is then converted back to
    OpenCV's 8-bit LAB representation for RGB reconstruction.

    The default target statistics are preserved from the archived thesis code
    for backward compatibility. They are implementation constants, not claimed
    to be universal or optimized Reinhard target statistics.
    """
    img = np.asarray(img_rgb, dtype=np.uint8)

    # RGB -> OpenCV 8-bit LAB
    lab_cv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    # OpenCV 8-bit LAB -> conventional CIELAB
    lab = np.empty_like(lab_cv, dtype=np.float32)
    lab[..., 0] = lab_cv[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab_cv[..., 1] - 128.0
    lab[..., 2] = lab_cv[..., 2] - 128.0

    pixels = lab.reshape(-1, 3)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)

    # Avoid division by zero
    stds = np.maximum(stds, 1e-8)

    target_means = np.asarray(target_means, dtype=np.float32)
    target_stds = np.asarray(target_stds, dtype=np.float32)

    if target_means.shape != (3,) or target_stds.shape != (3,):
        raise ValueError(
            "target_means and target_stds must each contain exactly 3 values."
        )

    if np.any(target_stds <= 0):
        raise ValueError("target_stds must contain strictly positive values.")

    # Reinhard-style normalization in conventional CIELAB coordinates
    lab_norm = (lab - means) / stds
    lab_norm = lab_norm * target_stds + target_means

    # Clip to representable CIELAB ranges
    lab_norm[..., 0] = np.clip(lab_norm[..., 0], 0.0, 100.0)
    lab_norm[..., 1] = np.clip(lab_norm[..., 1], -128.0, 127.0)
    lab_norm[..., 2] = np.clip(lab_norm[..., 2], -128.0, 127.0)

    # Conventional CIELAB -> OpenCV 8-bit LAB
    lab_cv_norm = np.empty_like(lab_norm, dtype=np.float32)
    lab_cv_norm[..., 0] = lab_norm[..., 0] * (255.0 / 100.0)
    lab_cv_norm[..., 1] = lab_norm[..., 1] + 128.0
    lab_cv_norm[..., 2] = lab_norm[..., 2] + 128.0

    lab_cv_norm = np.clip(lab_cv_norm, 0.0, 255.0).astype(np.uint8)

    # LAB -> RGB
    rgb_norm = cv2.cvtColor(lab_cv_norm, cv2.COLOR_LAB2RGB)

    return rgb_norm
