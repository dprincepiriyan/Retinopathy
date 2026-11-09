"""Custom clinical loss functions for multi-label triage and ordinal DR severity grading."""

from src.losses.asymmetric_focal_loss import AsymmetricFocalLoss
from src.losses.ordinal_focal_loss import (
    CORALOrdinalFocalLoss,
    coral_logits_to_expected_grade,
    coral_logits_to_probs,
    encode_ordinal_targets,
)
from src.losses.soft_qwk_loss import SoftQWKLoss

__all__ = [
    "AsymmetricFocalLoss",
    "CORALOrdinalFocalLoss",
    "encode_ordinal_targets",
    "coral_logits_to_probs",
    "coral_logits_to_expected_grade",
    "SoftQWKLoss",
]
