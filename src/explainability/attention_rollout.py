"""Cross-Attention Rollout & 4-Quadrant Clinical Involvement Analyzer.

Maps Vision Transformer & Bidirectional Cross-Attention weights onto the 4 anatomical
retinal quadrants (Superior-Temporal, Superior-Nasal, Inferior-Temporal, Inferior-Nasal)
to quantify compliance with the clinical ICDR 4-2-1 severe NPDR rule.
"""

from dataclasses import dataclass
from typing import Dict
import cv2
import numpy as np
import torch


@dataclass
class QuadrantInvolvement:
    """Spatial attention distribution across the 4 anatomical retinal quadrants."""

    superior_temporal: float
    superior_nasal: float
    inferior_temporal: float
    inferior_nasal: float
    active_quadrant_count: int
    dominant_quadrant: str
    rollout_map: np.ndarray


class CrossAttentionRollout:
    """Computes spatial attention rollout and 4-quadrant retinal pathology distribution."""

    def __init__(self, active_quadrant_threshold: float = 0.22) -> None:
        self.active_quadrant_threshold = active_quadrant_threshold

    def compute_quadrant_involvement(
        self,
        cross_attention_weights: torch.Tensor | np.ndarray,
        laterality: str = "OD",
        output_size: tuple[int, int] = (512, 512),
    ) -> QuadrantInvolvement:
        """Compute normalized 4-quadrant attention scores and 2D spatial rollout map.

        Args:
            cross_attention_weights: Attention matrix of shape (B, N_q, N_k) or (N_q, N_k).
            laterality: "OD" (Right Eye) or "OS" (Left Eye) to orient Temporal vs Nasal halves.
            output_size: Spatial size (H, W) of the returned rollout map.
        """
        if isinstance(cross_attention_weights, torch.Tensor):
            attn = cross_attention_weights.detach().cpu().numpy()
        else:
            attn = np.asarray(cross_attention_weights, dtype=np.float32)

        if attn.ndim == 3:
            attn = attn[0]  # (N_q, N_k)

        # Average attention received by each spatial key token
        token_importance = np.mean(attn, axis=0)
        grid_side = int(np.sqrt(len(token_importance)))
        if grid_side * grid_side != len(token_importance):
            grid_side = max(1, grid_side)
            token_importance = token_importance[: grid_side * grid_side]

        grid_2d = token_importance.reshape(grid_side, grid_side)
        g_min, g_max = float(np.min(grid_2d)), float(np.max(grid_2d))
        if g_max - g_min > 1e-7:
            grid_norm = (grid_2d - g_min) / (g_max - g_min)
        else:
            grid_norm = np.ones_like(grid_2d) * 0.25

        mid = max(1, grid_side // 2)
        top_left = float(np.mean(grid_norm[:mid, :mid]))
        top_right = float(np.mean(grid_norm[:mid, mid:]))
        bot_left = float(np.mean(grid_norm[mid:, :mid]))
        bot_right = float(np.mean(grid_norm[mid:, mid:]))

        total = max(top_left + top_right + bot_left + bot_right, 1e-6)
        top_left /= total
        top_right /= total
        bot_left /= total
        bot_right /= total

        # Anatomical mapping:
        # For Right Eye (OD): Left side of image is Temporal, Right side is Nasal.
        # For Left Eye (OS): Left side of image is Nasal, Right side is Temporal.
        if laterality.upper() == "OD":
            sup_temp, sup_nasal = top_left, top_right
            inf_temp, inf_nasal = bot_left, bot_right
        else:
            sup_nasal, sup_temp = top_left, top_right
            inf_nasal, inf_temp = bot_left, bot_right

        quad_dict: Dict[str, float] = {
            "Superior-Temporal": round(sup_temp, 4),
            "Superior-Nasal": round(sup_nasal, 4),
            "Inferior-Temporal": round(inf_temp, 4),
            "Inferior-Nasal": round(inf_nasal, 4),
        }
        active_count = sum(1 for v in quad_dict.values() if v >= self.active_quadrant_threshold)
        dominant = max(quad_dict.items(), key=lambda kv: kv[1])[0]

        rollout_map = cv2.resize(
            grid_norm.astype(np.float32),
            (output_size[1], output_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        return QuadrantInvolvement(
            superior_temporal=quad_dict["Superior-Temporal"],
            superior_nasal=quad_dict["Superior-Nasal"],
            inferior_temporal=quad_dict["Inferior-Temporal"],
            inferior_nasal=quad_dict["Inferior-Nasal"],
            active_quadrant_count=active_count,
            dominant_quadrant=dominant,
            rollout_map=np.clip(rollout_map, 0.0, 1.0),
        )
