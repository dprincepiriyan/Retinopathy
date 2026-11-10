"""Retinal-safe data augmentations and 8-view Dihedral Test-Time Augmentation (TTA).

Replaces the paper's latent-space SMOTE (Bottleneck 6) with image-space spatial
augmentations that train the CNN and ViT feature extractors on rare lesions:
1. Lesion Copy-Paste Synthesis (copying IDRiD MA/HE/EX/CWS masks onto Grade 0/1 retinas).
2. Retinal MixUp with ordinal target interpolation.
3. 8-View Dihedral TTA (4 rotations x 2 reflections) preserving circular retinal symmetry.
"""

from typing import List, Tuple
import cv2
import numpy as np


def retinal_mixup(
    img1: np.ndarray,
    label1: float,
    img2: np.ndarray,
    label2: float,
    alpha: float = 0.4,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float]:
    """Perform MixUp in image/tensor space with continuous ordinal severity label blending."""
    if rng is None:
        rng = np.random.default_rng()
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    mixed_img = (lam * img1.astype(np.float32) + (1.0 - lam) * img2.astype(np.float32)).astype(img1.dtype)
    mixed_label = float(lam * label1 + (1.0 - lam) * label2)
    return mixed_img, mixed_label


def lesion_copy_paste_synthesis(
    base_fundus_rgb: np.ndarray,
    donor_fundus_rgb: np.ndarray,
    lesion_binary_mask: np.ndarray,
    fov_mask: np.ndarray,
    feather_sigma: float = 2.0,
) -> np.ndarray:
    """Synthesize rare retinal pathology by blending donor lesions into a recipient scan.

    Uses Gaussian-feathered alpha blending strictly inside the recipient's valid FOV
    so CNN/ViT spatial backbones learn rare microaneurysm/hemorrhage/exudate textures.
    """
    if base_fundus_rgb.shape != donor_fundus_rgb.shape:
        donor_fundus_rgb = cv2.resize(
            donor_fundus_rgb, (base_fundus_rgb.shape[1], base_fundus_rgb.shape[0])
        )
    if lesion_binary_mask.shape[:2] != base_fundus_rgb.shape[:2]:
        lesion_binary_mask = cv2.resize(
            lesion_binary_mask.astype(np.uint8),
            (base_fundus_rgb.shape[1], base_fundus_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    valid_lesion = ((lesion_binary_mask > 0) & (fov_mask > 0)).astype(np.float32)
    alpha = cv2.GaussianBlur(valid_lesion, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]

    blended = (
        (1.0 - alpha) * base_fundus_rgb.astype(np.float32)
        + alpha * donor_fundus_rgb.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_dihedral_tta(tensor_chw: np.ndarray) -> List[np.ndarray]:
    """Generate all 8 views of the Dihedral group D4 for Test-Time Augmentation.

    Args:
        tensor_chw: Input image tensor of shape (C, H, W).

    Returns:
        List of 8 tensors of shape (C, H, W) corresponding to rotations (0, 90, 180, 270)
        and their horizontal reflections.
    """
    if tensor_chw.ndim != 3:
        raise ValueError(f"Expected 3D CHW tensor, got shape {tensor_chw.shape}")

    views: List[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(tensor_chw, k=k, axes=(1, 2)).copy()
        views.append(rotated)
        flipped = np.flip(rotated, axis=2).copy()
        views.append(flipped)
    return views
