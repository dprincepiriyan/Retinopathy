"""Differentiable Quadratic Weighted Kappa (Soft-QWK) Loss.

Directly optimizes Cohen's Quadratic Weighted Kappa during gradient descent:
    W_{i,j} = (i - j)^2 / (K - 1)^2
    L_{QWK} = 1.0 - Kappa_{soft}(P, Y)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftQWKLoss(nn.Module):
    """Differentiable Quadratic Weighted Kappa loss for ordinal 5-class DR grading."""

    def __init__(self, num_classes: int = 5, eps: float = 1e-7) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

        grid_i, grid_j = torch.meshgrid(
            torch.arange(num_classes, dtype=torch.float32),
            torch.arange(num_classes, dtype=torch.float32),
            indexing="ij",
        )
        penalty_matrix = torch.pow(grid_i - grid_j, 2.0) / float((num_classes - 1) ** 2)
        self.register_buffer("penalty_matrix", penalty_matrix)

    def forward(self, probs_or_logits: torch.Tensor, targets: torch.Tensor, is_probs: bool = True) -> torch.Tensor:
        """Compute 1 - SoftQWK in [0.0, 2.0].

        Args:
            probs_or_logits: Tensor of shape (B, K) containing probabilities (if is_probs=True)
                or raw class logits (if is_probs=False).
            targets: Integer class labels of shape (B,) in {0..K-1}.
            is_probs: Whether `probs_or_logits` is already normalized probabilities.
        """
        probs = probs_or_logits if is_probs else F.softmax(probs_or_logits, dim=-1)
        probs = probs.clamp(min=self.eps, max=1.0)
        batch_size = float(probs.shape[0])

        y_onehot = F.one_hot(targets.to(torch.long), num_classes=self.num_classes).to(dtype=probs.dtype)
        w = self.penalty_matrix.to(device=probs.device, dtype=probs.dtype)

        # Observed confusion matrix O = Y^T @ P
        observed = torch.matmul(y_onehot.t(), probs) / max(batch_size, 1.0)

        # Expected outer product E = hist(Y) (x) hist(P)
        hist_true = y_onehot.sum(dim=0) / max(batch_size, 1.0)
        hist_pred = probs.sum(dim=0) / max(batch_size, 1.0)
        expected = torch.outer(hist_true, hist_pred)

        numerator = torch.sum(w * observed)
        denominator = torch.sum(w * expected) + self.eps
        qwk = 1.0 - (numerator / denominator)
        return 1.0 - qwk
