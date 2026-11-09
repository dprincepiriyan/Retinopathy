"""Tests for Tier 0 & Tier 1 Micro-Lesion (<10px) Retention & Domain Harmonization."""

import cv2
import numpy as np
import torch

from app.assets.synthetic_fundus import generate_synthetic_fundus_image
from src.data.preprocessing import detect_and_crop_circular_fov, green_lab_clahe
from src.tier1_purifiers.fda import fourier_domain_adaptation
from src.tier1_purifiers.residual_cdae import ResidualAttentionUNetCDAE


def test_fov_crop_preserves_square_aspect_ratio() -> None:
    # Create rectangular fundus scan with black side borders (e.g., 360 x 520)
    base_sq = generate_synthetic_fundus_image(size=360, laterality="OD", seed=7)
    rect_img = cv2.copyMakeBorder(base_sq, 0, 0, 80, 80, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    cropped_rgb, cropped_mask, radius = detect_and_crop_circular_fov(rect_img, aspect_preserving_pad=True)
    h, w = cropped_rgb.shape[:2]
    assert h == w, f"Expected 1:1 aspect-preserving square crop, got {h}x{w}"
    assert radius > 100.0
    assert cropped_mask.shape == (h, w)


def test_fda_retains_sub_10px_synthetic_microaneurysms() -> None:
    # Create clean fundus scan and inject a known 5px diameter microaneurysm at (140, 140)
    src_img = generate_synthetic_fundus_image(size=256, laterality="OD", num_microaneurysms=0, seed=12)
    ref_img = generate_synthetic_fundus_image(size=256, laterality="OS", num_microaneurysms=0, seed=99)

    lesion_y, lesion_x = 140, 140
    src_with_ma = src_img.copy()
    cv2.circle(src_with_ma, (lesion_x, lesion_y), 2, (95, 5, 5), thickness=-1)  # 5px diameter (<10px)

    # Apply Tier-1 Fourier Domain Adaptation
    harmonized = fourier_domain_adaptation(
        src_with_ma, ref_img, beta=0.015, blend_alpha=0.65
    )

    # Verify the high-frequency microaneurysm contrast in the Green channel is preserved
    patch_bg = float(np.mean(harmonized[lesion_y - 12 : lesion_y - 6, lesion_x - 12 : lesion_x - 6, 1]))
    lesion_center_g = float(harmonized[lesion_y, lesion_x, 1])
    contrast_drop = patch_bg - lesion_center_g

    assert contrast_drop > 25.0, (
        f"Expected <10px microaneurysm Green-channel dip > 25 intensity units after FDA, got {contrast_drop:.2f}"
    )

    # Verify Green-LAB CLAHE further preserves/sharpens the micro-lesion
    mask = (cv2.cvtColor(src_with_ma, cv2.COLOR_RGB2GRAY) > 5).astype(np.uint8) * 255
    clahe_img = green_lab_clahe(harmonized, mask)
    clahe_bg = float(np.mean(clahe_img[lesion_y - 12 : lesion_y - 6, lesion_x - 12 : lesion_x - 6, 1]))
    clahe_center = float(clahe_img[lesion_y, lesion_x, 1])
    assert (clahe_bg - clahe_center) >= contrast_drop * 0.90


def test_residual_cdae_preserves_spatial_structure() -> None:
    cdae = ResidualAttentionUNetCDAE(in_channels=3, base_channels=16, residual_scale_alpha=0.25).eval()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        x_clean, delta_x = cdae(x)
    assert x_clean.shape == x.shape
    assert delta_x.shape == x.shape
    assert torch.all(x_clean >= 0.0) and torch.all(x_clean <= 1.0)
    # Zero-initialized residual head ensures untrained CDAE does not destroy micro-lesions
    assert torch.allclose(x_clean, x, atol=1e-5)
