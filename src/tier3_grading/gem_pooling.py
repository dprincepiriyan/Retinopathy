"""Generalized Mean (GeM) Pooling for Focal Micro-Lesion Preservation.

Replaces Global Average Pooling (GAP), which washes out tiny 1-10px microaneurysm
activations across a 768x768 feature map.
    GeM(f) = ( 1/|X| * sum_{x in X} max(x, eps)^p )^(1/p)
As p -> infinity, GeM approaches Max Pooling; with learnable p ~ 3.0, it concentrates
gradients on high-activation focal pathology while preserving smooth optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling2d(nn.Module):
    """2D Generalized Mean Pooling with learnable norm parameter p."""

    def __init__(
        self,
        initial_p: float = 3.0,
        learnable: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        p_tensor = torch.tensor([ float(initial_p) ], dtype=torch.float32)
        if learnable:
            self.p = nn.Parameter(p_tensor)
        else:
            self.register_buffer("p", p_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool spatial dimensions (B, C, H, W) -> (B, C)."""
        p_clamped = torch.clamp(self.p, min=1.0, max=10.0)
        # Apply ReLU/clamp to ensure non-negative base before fractional exponentiation
        x_pos = F.relu(x) + self.eps
        powered = torch.pow(x_pos, p_clamped.view(1, 1, 1, 1))
        avg_pooled = F.adaptive_avg_pool2d(powered, (1, 1)).flatten(1)
        return torch.pow(avg_pooled, 1.0 / p_clamped)
