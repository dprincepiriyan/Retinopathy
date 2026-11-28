"""Class-balanced effective-number sampler.

Addresses extreme class imbalance in EyePACS/Messidor-2 (~73% Grade 0 vs ~1.9% Grade 4)
without duplicating test samples.
"""

import numpy as np


class EffectiveNumberSampler:
    """Computes sample weights using Effective Number of Samples:
    E_n = (1 - beta^n) / (1 - beta), weight_c = 1 / E_{n_c}.
    """

    def __init__(self, beta: float = 0.9999) -> None:
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1).")
        self.beta = beta

    def compute_class_weights(self, labels: np.ndarray, num_classes: int = 5) -> np.ndarray:
        """Return normalized per-class weights of shape (num_classes,)."""
        labels_arr = np.asarray(labels, dtype=np.int64)
        counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)

        effective_num = 1.0 - np.power(self.beta, counts)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / np.sum(weights) * num_classes
        return weights.astype(np.float32)

    def compute_sample_probabilities(self, labels: np.ndarray, num_classes: int = 5) -> np.ndarray:
        """Return per-sample sampling probabilities summing to 1.0."""
        labels_arr = np.asarray(labels, dtype=np.int64)
        class_weights = self.compute_class_weights(labels_arr, num_classes=num_classes)
        sample_weights = class_weights[labels_arr]
        return (sample_weights / np.sum(sample_weights)).astype(np.float64)
