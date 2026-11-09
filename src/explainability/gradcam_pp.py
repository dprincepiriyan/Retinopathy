"""High-Resolution Grad-CAM++ on Stride-4 FPN Feature Maps for Micro-Lesion Localization.

Computes second- and third-order spatial gradient weights (Chattopadhay et al., WACV 2018)
so multiple tiny focal microaneurysms (<10px) and flame hemorrhages across the retina
receive equal visual attribution without fading out.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
import torch


@dataclass
class MicroLesionCandidate:
    """Localized micro-lesion candidate extracted from high-res Grad-CAM++ & lesion maps."""

    x: int
    y: int
    radius_px: int
    activation_score: float
    lesion_type: str


class GradCAMPlusPlus:
    """Grad-CAM++ heatmap generator and micro-lesion coordinate extractor."""

    def __init__(self, heatmap_alpha: float = 0.42, top_k_candidates: int = 5) -> None:
        self.heatmap_alpha = heatmap_alpha
        self.top_k_candidates = top_k_candidates

    def compute_heatmap_from_activations(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor | None = None,
        output_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """Compute normalized [0, 1] Grad-CAM++ map of shape `output_size` (H, W).

        If `gradients` is None (e.g., fast no-grad inference mode), uses GeM-weighted
        spatial activation energy across the Stride-4 FPN channels.
        """
        act = activations.detach()
        if act.ndim == 4:
            act = act[0]  # (C, H, W)

        if gradients is not None:
            grad = gradients.detach()
            if grad.ndim == 4:
                grad = grad[0]
            grad_2 = grad.pow(2)
            grad_3 = grad.pow(3)
            denom = 2.0 * grad_2 + (act * grad_3).sum(dim=(1, 2), keepdim=True) + 1e-7
            alpha = grad_2 / denom
            weights = (alpha * torch.relu(grad)).sum(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=0))
        else:
            # Fast activation-based saliency weighted by channel L3 norm (matches GeM p=3)
            act_pos = torch.relu(act)
            channel_weights = act_pos.pow(3.0).mean(dim=(1, 2), keepdim=True).pow(1.0 / 3.0)
            cam = (channel_weights * act_pos).sum(dim=0)

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = float(np.min(cam_np)), float(np.max(cam_np))
        if cam_max - cam_min > 1e-6:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        cam_resized = cv2.resize(cam_np, (output_size[1], output_size[0]), interpolation=cv2.INTER_CUBIC)
        return np.clip(cam_resized, 0.0, 1.0).astype(np.float32)

    def overlay_heatmap_on_rgb(
        self,
        image_rgb: np.ndarray,
        heatmap_01: np.ndarray,
        fov_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create a clinical JET/TURBO heatmap overlay blended onto the fundus RGB image."""
        h, w = image_rgb.shape[:2]
        if heatmap_01.shape[:2] != (h, w):
            heatmap_01 = cv2.resize(heatmap_01, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap_u8 = np.uint8(255 * np.clip(heatmap_01, 0.0, 1.0))
        colored_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        # Only blend strongly where activation > 0.25 so normal retinal tissue remains clear
        dynamic_alpha = np.where(heatmap_01 > 0.25, self.heatmap_alpha * heatmap_01, 0.08 * heatmap_01)[:, :, None]
        overlaid = (
            (1.0 - dynamic_alpha) * image_rgb.astype(np.float32)
            + dynamic_alpha * colored_rgb.astype(np.float32)
        )
        if fov_mask is not None:
            mask_3c = fov_mask[:, :, None] > 0
            overlaid = np.where(mask_3c, overlaid, 0.0)
        return np.clip(overlaid, 0, 255).astype(np.uint8)

    def extract_microlesion_coordinates(
        self,
        heatmap_01: np.ndarray,
        lesion_prob_maps: np.ndarray | None = None,
        fov_mask: np.ndarray | None = None,
    ) -> List[MicroLesionCandidate]:
        """Extract top-K focal micro-lesion coordinates from high-res activation maps."""
        h, w = heatmap_01.shape[:2]
        work_map = heatmap_01.copy()
        if fov_mask is not None:
            # Erode slightly to ignore FOV perimeter edge
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            inner = cv2.erode(fov_mask.astype(np.uint8), kernel, iterations=1)
            work_map = np.where(inner > 0, work_map, 0.0)

        thresh_val = max(0.55, float(np.quantile(work_map, 0.96)))
        binary_peaks = (work_map >= thresh_val).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lesion_names = ["Microaneurysm (MA)", "Hemorrhage (HE)", "Hard Exudate (EX)", "Cotton Wool Spot (CWS)"]
        candidates: List[MicroLesionCandidate] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            ix, iy = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
            score = float(work_map[iy, ix])

            l_type = "Microaneurysm (MA)" if radius <= 8 else "Hemorrhage / Exudate Focus"
            if lesion_prob_maps is not None and lesion_prob_maps.ndim == 3 and lesion_prob_maps.shape[0] == 4:
                lh, lw = lesion_prob_maps.shape[1:]
                ly = int(np.clip(iy * lh / h, 0, lh - 1))
                lx = int(np.clip(ix * lw / w, 0, lw - 1))
                best_ch = int(np.argmax(lesion_prob_maps[:, ly, lx]))
                l_type = lesion_names[best_ch]

            candidates.append(
                MicroLesionCandidate(
                    x=ix,
                    y=iy,
                    radius_px=max(4, min(28, int(round(radius)))),
                    activation_score=round(score, 4),
                    lesion_type=l_type,
                )
            )

        candidates.sort(key=lambda c: c.activation_score, reverse=True)
        return candidates[: self.top_k_candidates]
