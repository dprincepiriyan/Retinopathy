"""Tier 4 Dual Clinical Explainability Engine: Grad-CAM++ & Cross-Attention Rollout."""

from src.explainability.attention_rollout import (
    CrossAttentionRollout,
    QuadrantInvolvement,
)
from src.explainability.gradcam_pp import GradCAMPlusPlus, MicroLesionCandidate

__all__ = [
    "MicroLesionCandidate",
    "GradCAMPlusPlus",
    "QuadrantInvolvement",
    "CrossAttentionRollout",
]
