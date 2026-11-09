"""Asymmetric Focal Loss (ASL) for Stage-1 Multi-Label Pathology Triage.

Formulates Stage 1 as independent binary Sigmoid heads for [DR, Glaucoma, Cataract]
rather than mutually exclusive Softmax, allowing co-existing conditions
(e.g., Cataract + DR) to be detected simultaneously without masking one another.
"""

import torch
import torch.nn as nn


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss for multi-label classification with hard-negative thresholding."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        label_smoothing: float = 0.02,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-label Asymmetric Focal Loss.

        Args:
            logits: Raw unnormalized logits of shape (B, C).
            targets: Binary multi-label targets in {0, 1} of shape (B, C).
        """
        targets_float = targets.to(dtype=logits.dtype)
        if self.label_smoothing > 0:
            targets_float = targets_float * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets_float * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets_float) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets_float
            pt1 = xs_neg * (1.0 - targets_float)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets_float + self.gamma_neg * (1.0 - targets_float)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()
