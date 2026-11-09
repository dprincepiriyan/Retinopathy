"""Nelder-Mead Simplex Threshold Optimizer for Quadratic Weighted Kappa (QWK).

Converts continuous severity predictions s in [0, 4.5] into discrete ICDR grades {0, 1, 2, 3, 4}
using 4 ordered decision boundaries [tau_1, tau_2, tau_3, tau_4] optimized on
Out-of-Fold (OOF) validation predictions to directly maximize Cohen's QWK.
"""

from typing import Sequence
import numpy as np
from scipy.optimize import minimize

from src.utils.metrics import quadratic_weighted_kappa


class NelderMeadQWKOptimizer:
    """Post-hoc threshold optimizer maximizing Quadratic Weighted Kappa via Nelder-Mead simplex."""

    def __init__(
        self,
        initial_thresholds: Sequence[float] = (0.5, 1.5, 2.5, 3.5),
        max_iter: int = 1000,
    ) -> None:
        self.initial_thresholds = np.array(initial_thresholds, dtype=np.float64)
        self.thresholds_ = self.initial_thresholds.copy()
        self.max_iter = max_iter
        self.best_qwk_: float = 0.0

    @staticmethod
    def discretize(continuous_scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
        """Map continuous scores to integer grades {0, 1, 2, 3, 4} using sorted thresholds."""
        sorted_t = np.sort(np.asarray(thresholds, dtype=np.float64))
        return np.digitize(continuous_scores, bins=sorted_t).astype(np.int64)

    def fit(self, continuous_scores: np.ndarray, y_true: np.ndarray) -> "NelderMeadQWKOptimizer":
        """Optimize [tau_1, tau_2, tau_3, tau_4] on OOF validation scores to maximize QWK."""
        scores_arr = np.asarray(continuous_scores, dtype=np.float64).ravel()
        y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()

        def objective(thresh: np.ndarray) -> float:
            # Penalize unordered or collapsed thresholds
            diffs = np.diff(thresh)
            if np.any(diffs < 0.08):
                return 2.0 + float(np.sum(np.maximum(0.0, 0.08 - diffs)) * 10.0)
            preds = self.discretize(scores_arr, thresh)
            qwk = quadratic_weighted_kappa(y_true_arr, preds, num_classes=5)
            return -qwk

        res = minimize(
            objective,
            self.initial_thresholds,
            method="Nelder-Mead",
            options={"maxiter": self.max_iter, "xatol": 1e-4, "fatol": 1e-5},
        )
        candidate_thresholds = np.sort(res.x)
        init_qwk = quadratic_weighted_kappa(
            y_true_arr, self.discretize(scores_arr, self.initial_thresholds), num_classes=5
        )
        opt_qwk = quadratic_weighted_kappa(
            y_true_arr, self.discretize(scores_arr, candidate_thresholds), num_classes=5
        )

        if opt_qwk >= init_qwk:
            self.thresholds_ = candidate_thresholds
            self.best_qwk_ = float(opt_qwk)
        else:
            self.thresholds_ = self.initial_thresholds.copy()
            self.best_qwk_ = float(init_qwk)

        return self

    def predict(self, continuous_scores: np.ndarray) -> np.ndarray:
        """Convert continuous scores into discrete ICDR grades {0..4} using fitted thresholds."""
        return self.discretize(continuous_scores, self.thresholds_)
