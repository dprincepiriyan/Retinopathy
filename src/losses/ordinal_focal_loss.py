"""Consistent Rank Logits (CORAL) Ordinal Focal Loss for ICDR Severity Grading.

Addresses Bottleneck 5 (Nominal Cross-Entropy vs. Ordinal Disease Progression):
Decomposes 5-class DR grading (0, 1, 2, 3, 4) into K-1 = 4 ordered binary tasks:
    Task k: Is severity > k?  (for k = 0, 1, 2, 3)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_ordinal_targets(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Convert integer DR grades (B,) in {0..K-1} into (B, K-1) binary cumulative targets.

    Example for num_classes=5:
      Grade 0 -> [0, 0, 0, 0]
      Grade 1 -> [1, 0, 0, 0]
      Grade 2 -> [1, 1, 0, 0]
      Grade 3 -> [1, 1, 1, 0]
      Grade 4 -> [1, 1, 1, 1]
    """
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > thresholds).to(dtype=torch.float32)


def coral_logits_to_probs(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """Convert (B, K-1) cumulative ordinal logits into (B, K) normalized class probabilities."""
    cum_probs = torch.sigmoid(ordinal_logits)  # P(y > 0), P(y > 1), P(y > 2), P(y > 3)
    # Enforce monotonic non-increasing cumulative probabilities via cumulative minimum
    # so P(y > 0) >= P(y > 1) >= P(y > 2) >= P(y > 3)
    p0 = 1.0 - cum_probs[:, 0:1]
    p_mid = cum_probs[:, :-1] - cum_probs[:, 1:]
    p_last = cum_probs[:, -1:]
    raw_probs = torch.cat([p0, p_mid, p_last], dim=1)
    raw_probs = torch.clamp(raw_probs, min=1e-6)
    return raw_probs / raw_probs.sum(dim=1, keepdim=True)


def coral_logits_to_expected_grade(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """Compute expected continuous severity score E[y] = sum_{k=0}^{K-2} P(y > k) in [0, K-1]."""
    return torch.sigmoid(ordinal_logits).sum(dim=1)


class CORALOrdinalFocalLoss(nn.Module):
    """Ordinal Focal Loss over K-1 cumulative thresholds with optional threshold importance weights."""

    def __init__(
        self,
        num_classes: int = 5,
        gamma: float = 2.0,
        alpha: float = 0.55,
        threshold_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        if threshold_weights is None:
            # Boost weight on threshold 0 (Normal vs Mild DR) and threshold 1 (Referable Grade >= 2)
            threshold_weights = torch.tensor([1.35, 1.25, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer("threshold_weights", threshold_weights)

    def forward(self, ordinal_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal binary cross-entropy across the K-1 cumulative ordinal thresholds.

        Args:
            ordinal_logits: Tensor of shape (B, K-1).
            labels: Integer tensor of shape (B,) in {0..K-1}.
        """
        targets = encode_ordinal_targets(labels, num_classes=self.num_classes)
        probs = torch.sigmoid(ordinal_logits)
        bce = F.binary_cross_entropy_with_logits(ordinal_logits, targets, reduction="none")

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = torch.pow(1.0 - p_t, self.gamma)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        weights = self.threshold_weights.to(device=ordinal_logits.device, dtype=ordinal_logits.dtype)
        weighted_loss = focal_factor * alpha_factor * bce * weights.unsqueeze(0)
        return weighted_loss.mean()
