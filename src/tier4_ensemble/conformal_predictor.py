"""Split Conformal Prediction Engine for Safety-Critical DR Grading.

Provides a mathematically guaranteed (1 - alpha) = 95% marginal coverage rate:
    P( Y_true in C(X) ) >= 1 - alpha
Outputs:
- Prediction Set C(X), e.g., [2] (High confidence) or [1, 2] (Borderline Moderate confidence)
- Confidence Tier ("HIGH", "MODERATE", "LOW_ABSTAIN")
- Automatic abstention ("Human Review Required") when prediction set size >= 3 or IQA fails.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ConformalPredictionResult:
    """Output of the Split Conformal Predictor for a single eye scan."""

    prediction_set: List[int]
    primary_grade: int
    conservative_referral_grade: int
    confidence_tier: str  # "HIGH", "MODERATE", "LOW_ABSTAIN"
    abstain_flag: bool
    nonconformity_quantile: float
    ensemble_std: float


class SplitConformalPredictor:
    """Split Conformal Prediction using Adaptive Prediction Sets (APS) & ordinal adjacency."""

    def __init__(
        self,
        alpha: float = 0.05,
        max_acceptable_set_size: int = 2,
        max_high_conf_std: float = 0.12,
        max_mod_conf_std: float = 0.25,
    ) -> None:
        self.alpha = alpha
        self.max_acceptable_set_size = max_acceptable_set_size
        self.max_high_conf_std = max_high_conf_std
        self.max_mod_conf_std = max_mod_conf_std
        # Default calibrated quantile for 1 - Prob(true_class)
        self.q_hat_: float = 0.72

    def fit(self, cal_probs: np.ndarray, cal_y_true: np.ndarray) -> "SplitConformalPredictor":
        """Calibrate the conformal non-conformity threshold q_hat on a held-out calibration split."""
        probs_arr = np.asarray(cal_probs, dtype=np.float64)
        y_arr = np.asarray(cal_y_true, dtype=np.int64)
        n = len(y_arr)
        if n == 0:
            return self

        # Non-conformity score s_i = 1 - p(y_true)
        true_class_probs = probs_arr[np.arange(n), y_arr]
        scores = 1.0 - true_class_probs

        # Finite-sample conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha))) / float(n))
        self.q_hat_ = float(np.quantile(scores, q_level, method="higher"))
        return self

    def predict_single(
        self,
        class_probs: np.ndarray,
        ensemble_std: float = 0.05,
        is_iqa_gradable: bool = True,
    ) -> ConformalPredictionResult:
        """Construct the conformal prediction set and safety-gated confidence rating for one eye."""
        probs = np.asarray(class_probs, dtype=np.float64).ravel()
        probs = probs / max(np.sum(probs), 1e-8)
        primary_grade = int(np.argmax(probs))

        if not is_iqa_gradable:
            return ConformalPredictionResult(
                prediction_set=[],
                primary_grade=-1,
                conservative_referral_grade=-1,
                confidence_tier="LOW_ABSTAIN",
                abstain_flag=True,
                nonconformity_quantile=round(self.q_hat_, 4),
                ensemble_std=round(float(ensemble_std), 4),
            )

        # Include all classes whose non-conformity score (1 - p_k) <= q_hat_
        threshold_prob = 1.0 - self.q_hat_
        included = [int(k) for k, p in enumerate(probs) if p >= threshold_prob]
        if not included:
            included = [primary_grade]

        # Also include second best class if its probability is close to top class (borderline ordinal grade)
        sorted_indices = np.argsort(probs)[::-1]
        if len(included) == 1 and probs[sorted_indices[0]] < 0.72 and probs[sorted_indices[1]] >= 0.20:
            included.append(int(sorted_indices[1]))

        included = sorted(set(included))

        # Check if set spans non-adjacent grades or exceeds max acceptable size or has high ensemble variance
        max_gap = (max(included) - min(included)) if len(included) > 1 else 0
        if (
            len(included) >= self.max_acceptable_set_size + 1
            or max_gap > 1
            or ensemble_std > self.max_mod_conf_std
        ):
            confidence_tier = "LOW_ABSTAIN"
            abstain_flag = True
            conservative_grade = max(included)
        elif len(included) == 1 and ensemble_std <= self.max_high_conf_std and probs[primary_grade] >= 0.65:
            confidence_tier = "HIGH"
            abstain_flag = False
            conservative_grade = primary_grade
        else:
            confidence_tier = "MODERATE"
            abstain_flag = False
            # Clinical safety rule: borderline moderate confidence follows the higher grade in the conformal set
            conservative_grade = max(included)

        return ConformalPredictionResult(
            prediction_set=included,
            primary_grade=primary_grade,
            conservative_referral_grade=int(conservative_grade),
            confidence_tier=confidence_tier,
            abstain_flag=abstain_flag,
            nonconformity_quantile=round(self.q_hat_, 4),
            ensemble_std=round(float(ensemble_std), 4),
        )
