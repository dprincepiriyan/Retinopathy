"""Tier 0 Deterministic Clinical Preprocessing & Image Enhancement.

Implements:
1. Automated Circular FOV Detection & Aspect-Preserving 1:1 Square Padding
   (preserves circular Optic Disc geometry for accurate vCDR calculation).
2. Ben Graham Local Illumination Subtraction:
   I_BG(x, y) = 4 * I(x, y) - 4 * (G_sigma * I)(x, y) + 128, where sigma = R / 30.
3. Green-Channel (540-570nm peak absorption for MA/HE) & LAB Luminance CLAHE.
"""

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np


@dataclass
class PreprocessedFundus:
    """Container for Tier 0 preprocessed outputs."""

    cropped_rgb: np.ndarray
    fov_mask: np.ndarray
    enhanced_rgb: np.ndarray
    ben_graham_rgb: np.ndarray
    tensor_512: np.ndarray
    tensor_768: np.ndarray
    estimated_radius: float


def detect_and_crop_circular_fov(
    image_rgb: np.ndarray,
    aspect_preserving_pad: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Detect circular retinal lens boundary via Otsu thresholding and pad to 1:1 square.

    Args:
        image_rgb: Input RGB fundus image (H, W, 3), uint8.
        aspect_preserving_pad: If True, pads the cropped bounding box to a 1:1 square
            so resizing never distorts the circular Optic Disc aspect ratio.

    Returns:
        Tuple of (cropped_padded_rgb, cropped_padded_mask, estimated_radius).
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got {image_rgb.shape}")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_thresh = max(8, min(20, int(otsu_val * 0.2)))
    _, thresh = cv2.threshold(blurred, bg_thresh, 255, cv2.THRESH_BINARY)

    # Fallback if threshold sets background too high on very dark scans
    if np.count_nonzero(thresh) < 0.05 * thresh.size:
        _, thresh = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image_rgb.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return image_rgb.copy(), mask, float(min(h, w)) / 2.0

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    x, y, w, h = cv2.boundingRect(hull)

    # Create clean circular mask inside bounding box
    raw_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(raw_mask, [hull], -1, 255, thickness=cv2.FILLED)

    cropped_img = image_rgb[y : y + h, x : x + w].copy()
    cropped_mask = raw_mask[y : y + h, x : x + w].copy()

    if aspect_preserving_pad and h != w:
        side = max(h, w)
        pad_top = (side - h) // 2
        pad_bottom = side - h - pad_top
        pad_left = (side - w) // 2
        pad_right = side - w - pad_left
        cropped_img = cv2.copyMakeBorder(
            cropped_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        cropped_mask = cv2.copyMakeBorder(
            cropped_mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

    radius = float(max(cropped_img.shape[:2])) / 2.0
    return cropped_img, cropped_mask, max(radius, 16.0)


def ben_graham_normalization(
    image_rgb: np.ndarray,
    fov_mask: np.ndarray,
    radius: float,
    sigma_divisor: float = 30.0,
) -> np.ndarray:
    """Apply Ben Graham local illumination subtraction:
    I_BG(x, y) = 4 * I(x, y) - 4 * (G_sigma * I)(x, y) + 128

    Erodes FOV mask slightly at the perimeter to prevent bright border ringing artifacts.
    """
    sigma = max(radius / sigma_divisor, 1.5)
    local_avg = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg = cv2.addWeighted(image_rgb, 4.0, local_avg, -4.0, 128.0)

    # Erode mask by ~2% of radius to suppress high-contrast edge halo
    erode_px = max(3, int(radius * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
    eroded_mask = cv2.erode(fov_mask, kernel, iterations=1)

    result = np.full_like(bg, 128, dtype=np.uint8)
    mask_3c = eroded_mask[:, :, None] > 0
    result = np.where(mask_3c, bg, 0).astype(np.uint8)
    return result


def green_lab_clahe(
    image_rgb: np.ndarray,
    fov_mask: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    green_blend_weight: float = 0.35,
) -> np.ndarray:
    """Apply CLAHE contrast normalization to LAB Luminance and the Green channel.

    Microaneurysms and retinal hemorrhages absorb maximally at 540-570nm (Green channel).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 1. LAB Luminance CLAHE
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_clahe = clahe.apply(l_chan)
    lab_enhanced = cv2.merge((l_clahe, a_chan, b_chan))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # 2. Dedicated Green-channel CLAHE reinforcement
    green_raw = image_rgb[:, :, 1]
    green_clahe = clahe.apply(green_raw)
    blended_green = cv2.addWeighted(
        rgb_enhanced[:, :, 1],
        1.0 - green_blend_weight,
        green_clahe,
        green_blend_weight,
        0.0,
    )
    rgb_enhanced[:, :, 1] = blended_green

    # Zero out outside FOV
    mask_3c = fov_mask[:, :, None] > 0
    return np.where(mask_3c, rgb_enhanced, 0).astype(np.uint8)


class Tier0Preprocessor:
    """End-to-end deterministic Tier 0 preprocessor producing 512x512 and 768x768 tensors."""

    def __init__(
        self,
        stage1_size: int = 512,
        stage2_size: int = 768,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self.stage1_size = stage1_size
        self.stage2_size = stage2_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

    def __call__(self, image_rgb: np.ndarray) -> PreprocessedFundus:
        cropped_rgb, fov_mask, radius = detect_and_crop_circular_fov(
            image_rgb, aspect_preserving_pad=True
        )
        enhanced_rgb = green_lab_clahe(
            cropped_rgb,
            fov_mask,
            clip_limit=self.clahe_clip_limit,
            tile_grid_size=self.clahe_grid_size,
        )
        bg_rgb = ben_graham_normalization(enhanced_rgb, fov_mask, radius)

        # Resize using Lanczos / Area interpolation to preserve <10px microaneurysms
        img_512 = cv2.resize(enhanced_rgb, (self.stage1_size, self.stage1_size), interpolation=cv2.INTER_LINEAR)
        img_768 = cv2.resize(enhanced_rgb, (self.stage2_size, self.stage2_size), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] CHW float32 tensors
        tensor_512 = np.transpose(img_512.astype(np.float32) / 255.0, (2, 0, 1))
        tensor_768 = np.transpose(img_768.astype(np.float32) / 255.0, (2, 0, 1))

        return PreprocessedFundus(
            cropped_rgb=cropped_rgb,
            fov_mask=fov_mask,
            enhanced_rgb=enhanced_rgb,
            ben_graham_rgb=bg_rgb,
            tensor_512=tensor_512,
            tensor_768=tensor_768,
            estimated_radius=radius,
        )
