"""Retinal-safe training augmentations (D4 symmetry, affine, photometric, blur, FOV-preserving)."""

from typing import Tuple
import cv2
import numpy as np


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
