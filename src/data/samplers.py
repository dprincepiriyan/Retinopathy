"""Class-balanced effective number sampler and Two-Phase curriculum scheduler.

Addresses extreme class imbalance in EyePACS/Messidor-2 (~73% Grade 0 vs ~1.9% Grade 4)
without duplicating test samples.
"""

from dataclasses import dataclass
from typing import Dict, List
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


@dataclass
class CurriculumStepConfig:
    """Configuration for a single epoch in the Two-Phase Stage-2 training curriculum."""

    epoch: int
    phase: str
    resolution: int
    learning_rate: float
    soft_qwk_weight: float
    enable_gem_learning: bool


class CurriculumResolutionScheduler:
    """Two-Phase progressive training schedule for Stage-2 ResViT-Pro:
    - Phase A (Epochs 1..phase_a_epochs): 512x512 resolution, stable CORAL-Focal representation learning.
    - Phase B (Epochs phase_a_epochs+1..total_epochs): 768x768 resolution, GeM pooling unlocked,
      Stride-4 FPN fine-tuning, and full Soft-QWK loss weight.
    """

    def __init__(
        self,
        phase_a_epochs: int = 15,
        phase_b_epochs: int = 20,
        phase_a_res: int = 512,
        phase_b_res: int = 768,
        phase_a_lr: float = 2.0e-4,
        phase_b_lr: float = 5.0e-5,
    ) -> None:
        self.phase_a_epochs = phase_a_epochs
        self.phase_b_epochs = phase_b_epochs
        self.total_epochs = phase_a_epochs + phase_b_epochs
        self.phase_a_res = phase_a_res
        self.phase_b_res = phase_b_res
        self.phase_a_lr = phase_a_lr
        self.phase_b_lr = phase_b_lr

    def get_epoch_config(self, epoch: int) -> CurriculumStepConfig:
        """Get resolution, learning rate, and loss weights for 1-indexed `epoch`."""
        if epoch <= self.phase_a_epochs:
            progress = epoch / max(1, self.phase_a_epochs)
            qwk_w = 0.1 + 0.2 * progress
            return CurriculumStepConfig(
                epoch=epoch,
                phase="Phase_A_512px",
                resolution=self.phase_a_res,
                learning_rate=self.phase_a_lr,
                soft_qwk_weight=round(qwk_w, 4),
                enable_gem_learning=False,
            )
        else:
            b_epoch = epoch - self.phase_a_epochs
            progress = min(1.0, b_epoch / max(1, self.phase_b_epochs))
            # Cosine decay in Phase B
            lr = self.phase_b_lr * 0.5 * (1.0 + float(np.cos(np.pi * progress)))
            lr = max(lr, 1.0e-6)
            return CurriculumStepConfig(
                epoch=epoch,
                phase="Phase_B_768px_MicroLesion",
                resolution=self.phase_b_res,
                learning_rate=round(lr, 8),
                soft_qwk_weight=0.75,
                enable_gem_learning=True,
            )
