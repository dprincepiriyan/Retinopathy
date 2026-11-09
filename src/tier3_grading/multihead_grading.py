"""Tri-Head Diagnostic Output Layer for Stage-2 DR Severity Grading.

Combines three complementary clinical prediction heads:
1. CORAL Ordinal Head (Grades 0-4): Shared weight projection + ordered bias thresholds
   guaranteeing rank-consistent cumulative probabilities P(y > 0) >= P(y > 1) >= P(y > 2) >= P(y > 3).
2. Dedicated Binary Referable-DR Head (Grade >= 2): Directly optimizes sensitivity
   on sight-threatening / referable retinopathy.
3. Continuous Severity Regressor (s in [0, 4.5]): Used by Nelder-Mead simplex threshold
   search on Out-of-Fold validation predictions to directly maximize Cohen's QWK.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.ordinal_focal_loss import coral_logits_to_probs


@dataclass
class TriHeadOutput:
    """Structured output of the Tri-Head Diagnostic Layer."""

    ordinal_logits: torch.Tensor       # (B, 4) cumulative logits for y > 0, 1, 2, 3
    class_probs: torch.Tensor          # (B, 5) calibrated 5-class probabilities
    referable_logit: torch.Tensor      # (B, 1) binary logit for Grade >= 2
    referable_prob: torch.Tensor       # (B, 1) binary probability P(Grade >= 2)
    continuous_score: torch.Tensor     # (B, 1) continuous severity scalar in [0, 4.5]


class TriHeadDiagnosticLayer(nn.Module):
    """Rank-consistent CORAL ordinal head + Referable DR binary head + Continuous regressor."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(p=dropout)

        # 1. Rank-consistent CORAL head: single shared weight vector + monotonically ordered biases
        self.coral_weight = nn.Linear(in_features, 1, bias=False)
        # Parameterize bias increments via softplus so b_0 > b_1 > b_2 > b_3 is mathematically guaranteed
        self.base_bias = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.bias_deltas = nn.Parameter(torch.zeros(self.num_thresholds - 1, dtype=torch.float32))

        # 2. Dedicated Referable DR (Grade >= 2) binary safety head
        self.referable_head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.SiLU(inplace=True),
            nn.Linear(in_features // 2, 1),
        )

        # 3. Continuous severity score regressor s in [0, 4.5]
        self.continuous_head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.SiLU(inplace=True),
            nn.Linear(in_features // 2, 1),
        )

    def get_ordered_coral_biases(self) -> torch.Tensor:
        """Compute monotonically decreasing bias vector [b_0, b_1, b_2, b_3]."""
        positive_gaps = F.softplus(self.bias_deltas) + 0.15
        cum_drops = torch.cumsum(positive_gaps, dim=0)
        biases = torch.cat(
            [self.base_bias.unsqueeze(0), self.base_bias.unsqueeze(0) - cum_drops], dim=0
        )
        return biases

    def forward(self, features: torch.Tensor) -> TriHeadOutput:
        h = self.drop(self.norm(features))

        # Head 1: CORAL ordinal logits
        shared_scalar = self.coral_weight(h)  # (B, 1)
        ordered_biases = self.get_ordered_coral_biases().unsqueeze(0)  # (1, 4)
        ordinal_logits = shared_scalar + ordered_biases  # (B, 4)
        class_probs = coral_logits_to_probs(ordinal_logits)  # (B, 5)

        # Head 2: Referable DR (Grade >= 2)
        ref_logit = self.referable_head(h)
        # Fuse explicit binary head with cumulative ordinal threshold P(y > 1)
        ref_prob = 0.5 * torch.sigmoid(ref_logit) + 0.5 * torch.sigmoid(ordinal_logits[:, 1:2])

        # Head 3: Continuous severity score bounded smoothly to [0.0, 4.5]
        raw_cont = torch.sigmoid(self.continuous_head(h)) * 4.5
        expected_coral = torch.sigmoid(ordinal_logits).sum(dim=1, keepdim=True)
        cont_score = 0.6 * raw_cont + 0.4 * expected_coral

        return TriHeadOutput(
            ordinal_logits=ordinal_logits,
            class_probs=class_probs,
            referable_logit=ref_logit,
            referable_prob=ref_prob,
            continuous_score=cont_score,
        )
