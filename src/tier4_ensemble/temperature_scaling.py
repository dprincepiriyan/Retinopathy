"""Temperature Scaling for Probability Calibration (ECE < 0.04).

Scales uncalibrated logits by a learned scalar temperature T > 0 on a held-out
calibration split so predicted probabilities reflect true empirical accuracy.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax


class TemperatureScaler:
    """Post-hoc temperature scaling for multi-class Softmax or binary/multi-label Sigmoid logits."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        self.temperature: float = float(initial_temperature)

    def fit_multiclass(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Find optimal T > 0 minimizing Negative Log-Likelihood (NLL) on validation logits (N, K)."""
        logits_arr = np.asarray(logits, dtype=np.float64)
        y_true_arr = np.asarray(y_true, dtype=np.int64)

        def nll(temp: float) -> float:
            scaled = logits_arr / max(temp, 1e-3)
            probs = softmax(scaled, axis=1)
            chosen = np.clip(probs[np.arange(len(y_true_arr)), y_true_arr], 1e-12, 1.0)
            return float(-np.mean(np.log(chosen)))

        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def transform_multiclass(self, logits: np.ndarray) -> np.ndarray:
        """Return temperature-calibrated class probabilities of shape (N, K)."""
        logits_arr = np.asarray(logits, dtype=np.float64)
        return softmax(logits_arr / max(self.temperature, 1e-3), axis=-1)

    def transform_binary(self, logits: np.ndarray) -> np.ndarray:
        """Return temperature-calibrated binary/multi-label Sigmoid probabilities."""
        logits_arr = np.asarray(logits, dtype=np.float64)
        return expit(logits_arr / max(self.temperature, 1e-3))
