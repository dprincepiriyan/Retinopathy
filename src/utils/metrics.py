"""Clinical evaluation metrics for RetinaNet-Pro.

Includes:
- Cohen's Quadratic Weighted Kappa (QWK) for ordinal DR severity grading (0-4)
- Expected Calibration Error (ECE) for reliability assessment
- Sensitivity at fixed Specificity (e.g., Referable DR Sensitivity @ 95% Specificity)
- Macro-F1 and ROC-AUC helpers
"""

from typing import Dict
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, roc_curve


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> float:
    """Compute Cohen's Quadratic Weighted Kappa (QWK) for ordinal grades.

    Args:
        y_true: 1D integer array of ground-truth grades in [0, num_classes - 1].
        y_pred: 1D integer array of predicted grades in [0, num_classes - 1].
        num_classes: Number of ordinal classes (default 5 for ICDR 0-4).

    Returns:
        Quadratic weighted kappa in [-1.0, 1.0].
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    if y_true_arr.size == 0:
        return 0.0
    if np.all(y_true_arr == y_pred_arr):
        return 1.0
    labels = list(range(num_classes))
    score = cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic", labels=labels)
    if np.isnan(score):
        return 0.0
    return float(score)


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class or binary probabilities.

    Args:
        probs: Array of shape (N, C) for multi-class or (N,) for binary probabilities.
        y_true: Array of shape (N,) with integer labels.
        n_bins: Number of uniform confidence bins in [0, 1].

    Returns:
        ECE scalar in [0.0, 1.0].
    """
    probs_arr = np.asarray(probs, dtype=np.float64)
    y_true_arr = np.asarray(y_true, dtype=np.int64)

    if probs_arr.ndim == 1:
        confidences = np.where(probs_arr >= 0.5, probs_arr, 1.0 - probs_arr)
        predictions = (probs_arr >= 0.5).astype(np.int64)
    else:
        confidences = np.max(probs_arr, axis=1)
        predictions = np.argmax(probs_arr, axis=1)

    accuracies = (predictions == y_true_arr).astype(np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true_arr)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        prop_in_bin = np.sum(in_bin) / max(n_samples, 1)
        if prop_in_bin > 0:
            avg_confidence = float(np.mean(confidences[in_bin]))
            avg_accuracy = float(np.mean(accuracies[in_bin]))
            ece += abs(avg_accuracy - avg_confidence) * prop_in_bin

    return float(ece)


def sensitivity_at_specificity(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.95,
) -> Dict[str, float]:
    """Compute Sensitivity (Recall) and optimal threshold at a target Specificity.

    Args:
        y_true_binary: Binary ground truth (0 or 1), e.g., Referable DR (Grade >= 2).
        y_score: Continuous predicted probability for the positive class.
        target_specificity: Minimum required specificity (1 - FPR), default 0.95.

    Returns:
        Dictionary with sensitivity, achieved_specificity, and threshold.
    """
    y_true_arr = np.asarray(y_true_binary, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    if len(np.unique(y_true_arr)) < 2:
        return {"sensitivity": 0.0, "achieved_specificity": 0.0, "threshold": 0.5}

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
    specificity = 1.0 - fpr
    valid_mask = specificity >= target_specificity
    if not np.any(valid_mask):
        best_idx = int(np.argmax(specificity))
    else:
        valid_indices = np.where(valid_mask)[0]
        best_idx = int(valid_indices[np.argmax(tpr[valid_indices])])

    return {
        "sensitivity": float(tpr[best_idx]),
        "achieved_specificity": float(specificity[best_idx]),
        "threshold": float(thresholds[best_idx]),
    }


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, float]:
    """Compute full suite of Stage-2 clinical metrics (QWK, Macro-F1, Accuracy, ECE, Referable Sens@95Spec)."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    y_probs_arr = np.asarray(y_probs, dtype=np.float64)

    qwk = quadratic_weighted_kappa(y_true_arr, y_pred_arr, num_classes=5)
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    ece = expected_calibration_error(y_probs_arr, y_true_arr)

    # Mild DR (Class 1) Recall — key bottleneck metric from the paper audit
    mild_mask = y_true_arr == 1
    mild_recall = float(np.mean(y_pred_arr[mild_mask] == 1)) if np.any(mild_mask) else 0.0

    # Referable DR (Grade >= 2) binary evaluation
    y_true_ref = (y_true_arr >= 2).astype(np.int64)
    y_score_ref = np.sum(y_probs_arr[:, 2:], axis=1) if y_probs_arr.ndim == 2 and y_probs_arr.shape[1] >= 5 else y_pred_arr >= 2
    ref_stats = sensitivity_at_specificity(y_true_ref, y_score_ref, target_specificity=0.85)

    try:
        ref_auc = float(roc_auc_score(y_true_ref, y_score_ref)) if len(np.unique(y_true_ref)) > 1 else 0.0
    except ValueError:
        ref_auc = 0.0

    return {
        "qwk": round(qwk, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "mild_dr_recall": round(mild_recall, 4),
        "ece": round(ece, 4),
        "referable_auc": round(ref_auc, 4),
        "referable_sensitivity_at_85spec": round(ref_stats["sensitivity"], 4),
    }
