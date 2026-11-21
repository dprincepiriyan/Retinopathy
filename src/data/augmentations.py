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


class RetinalTrainAugment:
    """Training-time augmentation for preprocessed fundus images (uint8 RGB, H x W x 3).

    Camera / site variation (white balance, exposure, focus, framing) is the main
    reason fundus models fail on new data, so on top of the lossless D4 dihedral
    group this adds mild photometric and geometric jitter. Magnitudes are kept small
    so sub-10px microaneurysms survive. Pixels outside the circular FOV stay black.

    Uses the global `np.random` state so DataLoader workers seeded through
    `src.training.common.seed_worker` are reproducible.
    """

    def __init__(
        self,
        strength: str = "medium",
        p_affine: float = 0.5,
        max_rotation_deg: float = 15.0,
        scale_range: Tuple[float, float] = (0.95, 1.08),
        p_photometric: float = 0.8,
        brightness: float = 0.12,
        contrast: float = 0.15,
        gamma_range: Tuple[float, float] = (0.85, 1.15),
        channel_gain: float = 0.06,
        p_blur: float = 0.15,
        blur_sigma_range: Tuple[float, float] = (0.3, 0.9),
    ) -> None:
        if strength not in {"d4_only", "light", "medium", "strong"}:
            raise ValueError("strength must be one of d4_only | light | medium | strong")
        mult = {"d4_only": 0.0, "light": 0.5, "medium": 1.0, "strong": 1.5}[strength]
        self.strength = strength
        self.p_affine = p_affine if mult > 0 else 0.0
        self.max_rotation_deg = max_rotation_deg * mult
        self.scale_range = (1.0 - (1.0 - scale_range[0]) * mult, 1.0 + (scale_range[1] - 1.0) * mult)
        self.p_photometric = p_photometric if mult > 0 else 0.0
        self.brightness = brightness * mult
        self.contrast = contrast * mult
        self.gamma_range = (1.0 - (1.0 - gamma_range[0]) * mult, 1.0 + (gamma_range[1] - 1.0) * mult)
        self.channel_gain = channel_gain * mult
        self.p_blur = p_blur if mult > 0 else 0.0
        self.blur_sigma_range = blur_sigma_range

    @staticmethod
    def _d4(img: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
        if np.random.rand() < 0.5:
            img = np.flipud(img)
        k_rot = int(np.random.randint(0, 4))
        if k_rot:
            img = np.rot90(img, k=k_rot)
        return np.ascontiguousarray(img)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = self._d4(img)
        fov = img.max(axis=2) > 8  # preprocessed images are zeroed outside the FOV

        if np.random.rand() < self.p_affine:
            h, w = img.shape[:2]
            angle = float(np.random.uniform(-self.max_rotation_deg, self.max_rotation_deg))
            scale = float(np.random.uniform(*self.scale_range))
            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            img = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            fov = cv2.warpAffine(fov.astype(np.uint8), mat, (w, h), flags=cv2.INTER_NEAREST) > 0

        if np.random.rand() < self.p_photometric:
            x = img.astype(np.float32) / 255.0
            gains = 1.0 + np.random.uniform(-self.channel_gain, self.channel_gain, size=(1, 1, 3))
            x = x * gains.astype(np.float32)
            c = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            b = np.random.uniform(-self.brightness, self.brightness)
            mean = float(x[fov].mean()) if np.any(fov) else 0.5
            x = (x - mean) * c + mean + b
            x = np.clip(x, 0.0, 1.0)
            gamma = np.random.uniform(*self.gamma_range)
            x = np.power(x, gamma)
            img = (x * 255.0 + 0.5).astype(np.uint8)

        if np.random.rand() < self.p_blur:
            sigma = float(np.random.uniform(*self.blur_sigma_range))
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

        img[~fov] = 0
        return np.ascontiguousarray(img)
