"""Out-of-Fold (OOF) Stacking Meta-Learner with 8x Dihedral Test-Time Augmentation (TTA).

Fuses features from:
- Stage-1 Multi-Label Triage probabilities: [P(DR), P(Glaucoma), P(Cataract), vCDR]
- Stage-2 Tri-Head outputs: [5 class probs, Referable DR prob, Continuous severity score]
- 8-View Dihedral TTA uncertainty (standard deviation across views)
"""

from typing import Dict
import numpy as np
from sklearn.linear_model import Ridge


class OOFStackingMetaLearner:
    """Stacking meta-learner combining Stage-1 and Stage-2 OOF predictions."""

    def __init__(self, ridge_alpha: float = 1.0) -> None:
        self.ridge_alpha = ridge_alpha
        self.model = Ridge(alpha=ridge_alpha)
        self.is_fitted: bool = False

    @staticmethod
    def build_meta_features(
        stage1_probs: np.ndarray,
        stage2_class_probs: np.ndarray,
        stage2_ref_prob: np.ndarray,
        stage2_cont_score: np.ndarray,
        tta_std: np.ndarray,
    ) -> np.ndarray:
        """Concatenate Stage-1 and Stage-2 outputs into a 1D or 2D meta-feature matrix."""
        s1 = np.atleast_2d(stage1_probs)
        s2_c = np.atleast_2d(stage2_class_probs)
        s2_r = np.atleast_2d(stage2_ref_prob).reshape(len(s1), -1)
        s2_s = np.atleast_2d(stage2_cont_score).reshape(len(s1), -1)
        std = np.atleast_2d(tta_std).reshape(len(s1), -1)
        return np.hstack([s1, s2_c, s2_r, s2_s, std]).astype(np.float64)

    def fit(self, meta_features: np.ndarray, y_true_grades: np.ndarray) -> "OOFStackingMetaLearner":
        """Fit the stacking regressor on Out-of-Fold validation features."""
        self.model.fit(meta_features, y_true_grades.astype(np.float64))
        self.is_fitted = True
        return self

    def predict_continuous_severity(
        self,
        meta_features: np.ndarray,
        fallback_cont_score: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Predict meta-blended continuous DR severity score in [0.0, 4.5]."""
        if not self.is_fitted:
            return np.clip(np.atleast_1d(fallback_cont_score), 0.0, 4.5)
        preds = self.model.predict(np.atleast_2d(meta_features))
        return np.clip(preds, 0.0, 4.5)
