"""Post-hoc calibration of Stage-2 from true out-of-fold (OOF) predictions.

Data use (no double-dipping between temperature and conformal calibration):

- QWK decision thresholds      <- all OOF scans (continuous severity score)
- Temperature T                <- OOF half A
- Conformal q_hat (alpha)      <- OOF half B, on temperature-scaled probabilities
- Reported coverage / QWK / ECE <- the locked TEST split (fold-ensemble predictions)

The pipeline applies these quantities at inference via the same `apply_temperature`
helper, and the primary grade comes from thresholding the continuous score.

Note on coverage: `q_hat` is fitted on single-fold OOF predictions, while inference
averages all fold models (and optionally TTA views). Averaged probabilities are
flatter, so formal split-conformal coverage does not automatically carry over;
trust `test_report["conformal_coverage"]`, which measures empirical coverage on
the fold-ensemble test predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.special import softmax

from src.tier4_ensemble.conformal_predictor import SplitConformalPredictor
from src.tier4_ensemble.qwk_threshold_optimizer import NelderMeadQWKOptimizer
from src.tier4_ensemble.temperature_scaling import TemperatureScaler
from src.utils.metrics import bootstrap_ci, expected_calibration_error, quadratic_weighted_kappa


def probs_to_pseudo_logits(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale CORAL class probabilities via their log-probabilities."""
    return softmax(probs_to_pseudo_logits(probs) / max(float(temperature), 1e-3), axis=-1)


def _stratified_halves(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a_idx, b_idx = [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        a_idx.extend(idx[:half])
        b_idx.extend(idx[half:])
    return np.sort(np.array(a_idx, dtype=np.int64)), np.sort(np.array(b_idx, dtype=np.int64))


def evaluate_calibrated(
    pred: Dict[str, np.ndarray],
    thresholds: np.ndarray,
    temperature: float,
    q_hat: float,
    alpha: float,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    y = pred["targets"].astype(np.int64)
    raw = pred["probs"]
    cal = apply_temperature(raw, temperature)
    grade_thr = NelderMeadQWKOptimizer.discretize(pred["continuous_scores"], thresholds)
    cp = SplitConformalPredictor(alpha=alpha)
    cp.q_hat_ = q_hat
    std = pred.get("member_referable_std", np.zeros(len(y)))
    # Same call as the deployed pipeline, so coverage / set size describe the sets clinicians see.
    results = [cp.predict_single(p, ensemble_std=float(sd), primary_grade=int(g)) for p, sd, g in zip(cal, std, grade_thr)]
    sets = [r.prediction_set for r in results]
    covered = np.array([yt in s for yt, s in zip(y, sets)])
    abstain = np.array([r.abstain_flag for r in results])
    qwk_thr, lo, hi = bootstrap_ci(lambda a, b: quadratic_weighted_kappa(a, b), y, grade_thr, n_boot=n_boot)
    ref_true = (y >= 2).astype(int)
    ref_pred = (grade_thr >= 2).astype(int)
    sens = float((ref_pred[ref_true == 1] == 1).mean()) if ref_true.any() else float("nan")
    spec = float((ref_pred[ref_true == 0] == 0).mean()) if (ref_true == 0).any() else float("nan")
    return {
        "n": int(len(y)),
        "qwk_argmax": round(quadratic_weighted_kappa(y, raw.argmax(1)), 4),
        "qwk_thresholded": round(qwk_thr, 4),
        "qwk_thresholded_95ci": [round(lo, 4), round(hi, 4)],
        "referable_sensitivity_thresholded": round(sens, 4),
        "referable_specificity_thresholded": round(spec, 4),
        "ece_before": round(expected_calibration_error(raw, y), 4),
        "ece_after_temperature": round(expected_calibration_error(cal, y), 4),
        "conformal_coverage": round(float(covered.mean()), 4),
        "conformal_target_coverage": round(1.0 - alpha, 4),
        "conformal_mean_set_size": round(float(np.mean([len(s) for s in sets])), 3),
        "abstention_rate": round(float(abstain.mean()), 4),
    }


def calibrate_stage2(
    oof: Dict[str, np.ndarray],
    test: Optional[Dict[str, np.ndarray]] = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    y = oof["targets"].astype(np.int64)
    thr_opt = NelderMeadQWKOptimizer().fit(oof["continuous_scores"], y)

    a_idx, b_idx = _stratified_halves(y, seed)
    temp = TemperatureScaler().fit_multiclass(probs_to_pseudo_logits(oof["probs"][a_idx]), y[a_idx])
    cal_b = apply_temperature(oof["probs"][b_idx], temp.temperature)
    conformal = SplitConformalPredictor(alpha=alpha).fit(cal_b, y[b_idx])

    payload: Dict[str, Any] = {
        "qwk_thresholds": [round(float(t), 6) for t in thr_opt.thresholds_],
        "best_oof_qwk": round(float(thr_opt.best_qwk_), 4),
        "temperature": round(float(temp.temperature), 6),
        "conformal_alpha": alpha,
        "conformal_score": "lac (1 - p_true) on temperature-scaled probabilities",
        "conformal_q_hat": round(float(conformal.q_hat_), 6),
        "n_oof": int(len(y)),
        "n_temperature_fit": int(len(a_idx)),
        "n_conformal_fit": int(len(b_idx)),
        "oof_half_b_report": evaluate_calibrated(
            {k: v[b_idx] for k, v in oof.items() if isinstance(v, np.ndarray) and len(v) == len(y)},
            thr_opt.thresholds_, temp.temperature, conformal.q_hat_, alpha, n_boot=200,
        ),
    }
    if test is not None:
        payload["test_report"] = evaluate_calibrated(test, thr_opt.thresholds_, temp.temperature, conformal.q_hat_, alpha)
    return payload


def calibrate_from_files(
    oof_npz: str | Path,
    out_json: str | Path,
    test_npz: Optional[str | Path] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    import json

    def load(p):
        with np.load(p, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}

    payload = calibrate_stage2(load(oof_npz), load(test_npz) if test_npz and Path(test_npz).exists() else None, alpha)
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
