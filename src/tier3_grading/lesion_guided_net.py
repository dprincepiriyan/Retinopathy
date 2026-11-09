"""Model 3B: Lesion-Guided Multi-Task Spatial Attention Branch.

Produces explicit 4-channel spatial probability maps corresponding to IDRiD lesion types:
  - Channel 0: Microaneurysms (MA) — earliest indicator of Mild NPDR (Grade 1)
  - Channel 1: Retinal Hemorrhages (HE)
  - Channel 2: Hard Exudates (EX)
  - Channel 3: Cotton Wool Spots (CWS)
Modulates high-resolution Stride-4 FPN features so tiny micro-lesions receive boosted weight.
"""

from typing import Tuple
import torch
import torch.nn as nn


LESION_CHANNEL_NAMES = ("Microaneurysms (MA)", "Hemorrhages (HE)", "Hard Exudates (EX)", "Cotton Wool Spots (CWS)")


class LesionGuidedAttentionBranch(nn.Module):
    """Predicts 4-channel lesion probability maps and re-weights spatial feature maps."""

    def __init__(self, in_channels: int, num_lesion_types: int = 4) -> None:
        super().__init__()
        mid_ch = max(16, in_channels // 2)
        self.lesion_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, num_lesion_types, kernel_size=1, bias=True),
        )
        self.channel_gate = nn.Sequential(
            nn.Conv2d(num_lesion_types, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, fpn_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute lesion probability maps and lesion-boosted spatial features.

        Args:
            fpn_features: High-resolution feature map (B, C, H, W).

        Returns:
            Tuple of (gated_features (B, C, H, W), lesion_prob_maps (B, 4, H, W)).
        """
        lesion_logits = self.lesion_conv(fpn_features)
        lesion_probs = torch.sigmoid(lesion_logits)

        # Spatial union of any detected lesion boosts feature activation (residual attention 1 + gate)
        spatial_boost = self.channel_gate(lesion_probs)
        gated_features = fpn_features * (1.0 + spatial_boost)
        return gated_features, lesion_probs
