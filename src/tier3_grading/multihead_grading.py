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

# Parameter names of the CORAL thresholds (their lr multiplier is set in
# `src.models.pretrained.parameter_groups`).
CORAL_BIAS_PARAM_SUFFIXES = ("tri_head.base_bias", "tri_head.bias_deltas")
MIN_CORAL_GAP = 0.15
DEFAULT_CORAL_GAP = 2.0


def _softplus_inverse(x) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32).clamp_min(1e-4)
    return x + torch.log(-torch.expm1(-x))


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
        # Parameterize bias increments via softplus so b_0 > b_1 > b_2 > b_3 is mathematically guaranteed.
        # The gaps MUST start wide: with gaps < ~1.1 logits a middle grade can never be the argmax
        # class (P(y=k) = sigmoid(s+b_{k-1}) - sigmoid(s+b_k) is too small), and Adam moves these
        # scalars only ~lr per step, so the initial gaps stay close to where they start.
        self.base_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bias_deltas = nn.Parameter(
            torch.full((self.num_thresholds - 1,), _softplus_inverse(DEFAULT_CORAL_GAP - MIN_CORAL_GAP))
        )

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

    @torch.no_grad()
    def init_ordinal_biases(self, class_prior, min_gap: float = DEFAULT_CORAL_GAP) -> torch.Tensor:
        """Initialise the CORAL thresholds from the (training / sampling) class distribution.

        b_k starts at logit(P(y > k)) so that a zero score reproduces the class prior, but every
        gap is widened to at least `min_gap` so that all grades are reachable by argmax.
        Returns the initial bias vector.
        """
        prior = torch.as_tensor(class_prior, dtype=torch.float64).flatten()
        if prior.numel() != self.num_classes or float(prior.sum()) <= 0:
            raise ValueError(f"class_prior must have {self.num_classes} non-negative entries")
        prior = prior / prior.sum()
        exceed = (1.0 - torch.cumsum(prior, 0)[:-1]).clamp(1e-3, 1 - 1e-3)  # P(y > k), k = 0..K-2
        target = torch.log(exceed / (1.0 - exceed))
        biases = [float(target[0])]
        for k in range(1, self.num_thresholds):
            biases.append(min(float(target[k]), biases[-1] - min_gap))
        gaps = torch.tensor([biases[k - 1] - biases[k] for k in range(1, self.num_thresholds)])
        self.base_bias.copy_(torch.tensor(biases[0], dtype=self.base_bias.dtype))
        self.bias_deltas.copy_(_softplus_inverse(gaps - MIN_CORAL_GAP).to(self.bias_deltas.dtype))
        return self.get_ordered_coral_biases().detach()

    def get_ordered_coral_biases(self) -> torch.Tensor:
        """Compute monotonically decreasing bias vector [b_0, b_1, b_2, b_3]."""
        positive_gaps = F.softplus(self.bias_deltas) + MIN_CORAL_GAP
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
