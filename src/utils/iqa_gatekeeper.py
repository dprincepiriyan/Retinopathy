"""Tier 0 Pre-Inference Image Quality Assessment (IQA) Gatekeeper.

Evaluates fundus scans for severe motion/defocus blur, flash glare overexposure,
media opacity / extreme underexposure, and valid circular FOV geometry before
allowing automated clinical grading.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np


@dataclass
class IQAResult:
    """Structured output of the Tier 0 Image Quality Assessment gatekeeper."""

    is_gradable: bool
    quality_score: float
    blur_laplacian_var: float
    glare_fraction: float
    mean_intensity: float
    circularity: float
    rejection_reasons: List[str] = field(default_factory=list)


class IQAGatekeeper:
    """Deterministic ophthalmic image quality gatekeeper."""

    def __init__(
        self,
        min_laplacian_variance: float = 18.0,
        max_overexposed_fraction: float = 0.18,
        min_underexposed_mean: float = 20.0,
        min_circularity_score: float = 0.65,
    ) -> None:
        self.min_laplacian_variance = min_laplacian_variance
        self.max_overexposed_fraction = max_overexposed_fraction
        self.min_underexposed_mean = min_underexposed_mean
        self.min_circularity_score = min_circularity_score

    def evaluate(
        self,
        image_rgb: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
    ) -> IQAResult:
        """Assess whether a fundus image meets clinical gradability criteria.

        Args:
            image_rgb: uint8 RGB image array of shape (H, W, 3).
            fov_mask: Optional binary mask (H, W) with non-zero inside the fundus FOV.

        Returns:
            IQAResult with gradability boolean, sub-metrics, and rejection reasons.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got {image_rgb.shape}")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        green = image_rgb[:, :, 1]

        if fov_mask is None:
            _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        mask_bool = fov_mask > 0
        total_fov_pixels = int(np.count_nonzero(mask_bool))

        if total_fov_pixels < 100:
            return IQAResult(
                is_gradable=False,
                quality_score=0.0,
                blur_laplacian_var=0.0,
                glare_fraction=0.0,
                mean_intensity=0.0,
                circularity=0.0,
                rejection_reasons=["No valid retinal field-of-view detected."],
            )

        # 1. Sharpness / Focus assessment on the green channel within interior FOV
        # Normalize to 256x256 equivalent scale and erode FOV border by 5px so the
        # sharp black-border step edge does not artificially inflate Laplacian variance.
        green_resized = cv2.resize(green, (256, 256), interpolation=cv2.INTER_AREA)
        mask_u8_256 = cv2.resize(
            fov_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST
        )
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        inner_mask_256 = cv2.erode(mask_u8_256, erode_kernel, iterations=1) > 0
        if not np.any(inner_mask_256):
            inner_mask_256 = mask_u8_256 > 0
        laplacian = cv2.Laplacian(green_resized, cv2.CV_64F)
        blur_var = float(np.var(laplacian[inner_mask_256])) if np.any(inner_mask_256) else 0.0

        # 2. Flash glare / overexposure fraction inside FOV
        fov_gray_pixels = gray[mask_bool]
        glare_fraction = float(np.mean(fov_gray_pixels > 245))

        # 3. Media opacity / severe underexposure check
        mean_intensity = float(np.mean(fov_gray_pixels))

        # 4. Circularity of the largest contour
        contours, _ = cv2.findContours(
            fov_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        circularity = 0.0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = float(min(1.0, (4.0 * np.pi * area) / (perimeter ** 2)))

        reasons: List[str] = []
        if blur_var < self.min_laplacian_variance:
            reasons.append(
                f"Severe blur or media opacity detected (sharpness={blur_var:.1f} < {self.min_laplacian_variance})."
            )
        if glare_fraction > self.max_overexposed_fraction:
            reasons.append(
                f"Excessive flash glare/overexposure ({glare_fraction * 100:.1f}% > {self.max_overexposed_fraction * 100:.1f}%)."
            )
        if mean_intensity < self.min_underexposed_mean:
            reasons.append(
                f"Severe underexposure or lens occlusion (mean intensity={mean_intensity:.1f} < {self.min_underexposed_mean})."
            )
        if circularity < self.min_circularity_score:
            reasons.append(
                f"Incomplete or severely clipped fundus FOV (circularity={circularity:.2f} < {self.min_circularity_score})."
            )

        # Compute composite normalized quality score in [0, 1]
        sharpness_norm = min(1.0, blur_var / max(self.min_laplacian_variance * 3.0, 1e-6))
        glare_norm = max(0.0, 1.0 - (glare_fraction / max(self.max_overexposed_fraction * 2.0, 1e-6)))
        exposure_norm = min(1.0, max(0.0, (mean_intensity - 10.0) / 80.0))
        quality_score = float(
            np.clip(0.45 * sharpness_norm + 0.25 * glare_norm + 0.15 * exposure_norm + 0.15 * circularity, 0.0, 1.0)
        )

        is_gradable = len(reasons) == 0
        return IQAResult(
            is_gradable=is_gradable,
            quality_score=round(quality_score, 4),
            blur_laplacian_var=round(blur_var, 4),
            glare_fraction=round(glare_fraction, 4),
            mean_intensity=round(mean_intensity, 4),
            circularity=round(circularity, 4),
            rejection_reasons=reasons,
        )
