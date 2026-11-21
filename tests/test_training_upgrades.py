"""Regression tests for the training / inference correctness fixes.

Covers: strict self-describing checkpoints, masked Stage-1 loss, the source-shortcut
manifest, augmentation, AMP stepping, LR schedule, calibration + its use at
inference, conformal primary-grade override, and honest ONH / lesion reporting.
"""

import csv
import json

import numpy as np
import pytest
import torch

from app.assets.synthetic_fundus import generate_synthetic_fundus_image
from src.data.augmentations import RetinalTrainAugment
from src.data.stage1_manifest import build_stage1_combined_manifest
from src.losses import AsymmetricFocalLoss
from src.models.factory import build_model, load_model_checkpoint, save_model_checkpoint
from src.tier2_triage.multi_label_triage import fuse_stage1_outputs
from src.tier4_ensemble.conformal_predictor import SplitConformalPredictor
from src.training.calibration import apply_temperature, calibrate_stage2
from src.training.common import warmup_cosine_factor
from src.utils.metrics import bootstrap_ci, multilabel_triage_metrics, quadratic_weighted_kappa
from src.utils.tpu_accelerator import ColabTPUAccelerator

timm = pytest.importorskip("timm")

TINY_S2 = {"task": "stage2", "arch": "timm", "backbone_name": "efficientnet_b0", "pretrained": False, "input_size": 64}
TINY_S1 = {"task": "stage1", "arch": "timm", "backbone_name": "efficientnet_b0", "pretrained": False, "input_size": 64}


def test_checkpoint_roundtrip_is_strict_and_exact(tmp_path) -> None:
    torch.manual_seed(0)
    model = build_model(TINY_S2).eval()
    path = save_model_checkpoint(tmp_path / "m.pth", model, TINY_S2, {"epoch": 3})
    loaded, spec, meta = load_model_checkpoint(path)
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        a = model(x)["tri_head"].class_probs
        b = loaded(x)["tri_head"].class_probs
    assert torch.allclose(a, b, atol=1e-6)
    assert spec["input_size"] == 64 and meta["epoch"] == 3


def test_mismatched_checkpoint_fails_loudly(tmp_path) -> None:
    model = build_model(TINY_S2)
    bad_spec = dict(TINY_S2, backbone_name="resnet18")
    save_model_checkpoint(tmp_path / "bad.pth", model, bad_spec)
    with pytest.raises(RuntimeError):
        load_model_checkpoint(tmp_path / "bad.pth")


def test_legacy_raw_state_dict_still_loads(tmp_path) -> None:
    legacy = build_model({"task": "stage2", "arch": "resvit_pro", "fpn_dim": 32, "num_classes": 5, "num_heads": 4})
    torch.save(legacy.state_dict(), tmp_path / "legacy.pth")
    loaded, spec, meta = load_model_checkpoint(tmp_path / "legacy.pth")
    assert spec["arch"] == "resvit_pro" and meta["legacy_format"] is True


def test_masked_asymmetric_loss_ignores_unknown_labels() -> None:
    loss_fn = AsymmetricFocalLoss(label_smoothing=0.0)
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    flipped = torch.tensor([[1.0, 1.0, 1.0]])  # unknown labels changed -> must not matter
    assert torch.allclose(loss_fn(logits, targets, mask), loss_fn(logits, flipped, mask))
    assert not torch.allclose(loss_fn(logits, targets), loss_fn(logits, flipped))


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _row(image_id, pid, src, dr, gl, cat, fold, split):
    return {"image_id": image_id, "patient_id": pid, "source_dataset": src, "dr_grade": dr, "glaucoma": gl,
            "cataract": cat, "fold": fold, "split": split, "path_512": f"/x/{image_id}.jpg", "path_768": ""}


def test_combined_stage1_manifest_breaks_source_shortcut(tmp_path) -> None:
    s1 = [_row("a", "odir_1", "odir_s1", 0, 1, 0, 1, "train_val"), _row("b", "eye_1", "eyepacs_s1", 2, 0, 0, 0, "test")]
    s2 = [_row("c", "ap_1", "aptos2019_highres", 0, 0, 0, 2, "train_val"),
          _row("d", "ap_2", "aptos2019_highres", 3, 0, 0, 0, "test")]
    _write_csv(tmp_path / "s1.csv", s1)
    _write_csv(tmp_path / "s2.csv", s2)
    summary = build_stage1_combined_manifest(tmp_path / "s1.csv", tmp_path / "s2.csv", tmp_path / "out.csv")
    rows = {r["image_id"]: r for r in csv.DictReader(open(tmp_path / "out.csv"))}
    assert summary["n_rows"] == 4
    # APTOS contributes both DR classes from one source
    assert {int(rows["c"]["dr_grade"]) > 0, int(rows["d"]["dr_grade"]) > 0} == {False, True}
    # Unknown glaucoma/cataract labels are masked for EyePACS / APTOS, known for ODIR
    assert (rows["b"]["mask_glaucoma"], rows["c"]["mask_cataract"], rows["a"]["mask_glaucoma"]) == ("0", "0", "1")
    assert rows["d"]["split"] == "test"  # fold assignment preserved


def test_combined_manifest_rejects_patient_leakage(tmp_path) -> None:
    s1 = [_row("a", "p1", "odir_s1", 0, 0, 0, 1, "train_val"), _row("b", "p1", "odir_s1", 0, 0, 0, 0, "test")]
    _write_csv(tmp_path / "s1.csv", s1)
    with pytest.raises(RuntimeError):
        build_stage1_combined_manifest(tmp_path / "s1.csv", tmp_path / "none.csv", tmp_path / "out.csv")


def test_train_augment_keeps_shape_and_black_background() -> None:
    np.random.seed(0)
    img = generate_synthetic_fundus_image(size=128, laterality="OD", seed=4)
    aug = RetinalTrainAugment("strong")
    for _ in range(5):
        out = aug(img.copy())
        assert out.shape == img.shape and out.dtype == np.uint8
        assert out[0, 0].sum() == 0 and out[-1, -1].sum() == 0  # corners outside FOV stay black


def test_cpu_accelerator_step_and_lr_schedule() -> None:
    accel = ColabTPUAccelerator(prefer_tpu=False)
    assert accel.grad_scaler is None or accel.uses_grad_scaler
    lin = torch.nn.Linear(3, 1)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    before = lin.weight.detach().clone()
    accel.backward(lin(torch.ones(4, 3)).sum())
    assert accel.clip_and_step(opt, lin.parameters(), max_norm=1.0)
    assert not torch.allclose(before, lin.weight)

    factors = [warmup_cosine_factor(s, 100, 10, 0.05) for s in range(100)]
    assert factors[0] < factors[9] <= 1.0 and abs(factors[-1] - 0.05) < 0.01
    assert all(a >= b - 1e-9 for a, b in zip(factors[10:], factors[11:]))


def _synthetic_oof(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(5, size=n, p=[0.5, 0.1, 0.25, 0.06, 0.09])
    cont = np.clip(y + rng.normal(0, 0.45, n), 0, 4.5)
    logits = -np.abs(cont[:, None] - np.arange(5)[None]) * 4.0  # over-confident on purpose
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return {"probs": probs, "continuous_scores": cont, "referable_probs": (cont >= 1.5).astype(float), "targets": y}


def test_calibration_reports_honest_test_coverage() -> None:
    payload = calibrate_stage2(_synthetic_oof(seed=1), _synthetic_oof(seed=2), alpha=0.1)
    assert payload["n_temperature_fit"] + payload["n_conformal_fit"] == payload["n_oof"]
    rep = payload["test_report"]
    assert rep["conformal_coverage"] >= 0.85  # target 0.90, finite-sample slack
    assert rep["ece_after_temperature"] <= rep["ece_before"] + 1e-6
    lo, hi = rep["qwk_thresholded_95ci"]
    assert lo <= rep["qwk_thresholded"] <= hi


def test_temperature_and_conformal_override() -> None:
    p = np.array([[0.7, 0.2, 0.05, 0.03, 0.02]])
    assert apply_temperature(p, 2.0)[0, 0] < 0.7 < apply_temperature(p, 0.5)[0, 0]
    cp = SplitConformalPredictor(alpha=0.05)
    cp.q_hat_ = 0.2
    res = cp.predict_single(np.array([0.85, 0.1, 0.03, 0.01, 0.01]), ensemble_std=0.0, primary_grade=1)
    assert res.primary_grade == 1 and 1 in res.prediction_set


def test_within_source_dr_auc_and_bootstrap() -> None:
    probs = np.array([[0.9, 0, 0], [0.8, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.95, 0, 0], [0.05, 0, 0]])
    targets = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
    sources = ["eyepacs", "aptos", "aptos", "odir", "aptos", "odir"]
    m = multilabel_triage_metrics(probs, targets, sources=sources)
    assert m["auc_dr_within_aptos"] == 1.0 and "auc_dr_within_odir" not in m
    y = np.array([0, 1, 2, 3, 4] * 20)
    point, lo, hi = bootstrap_ci(quadratic_weighted_kappa, y, y, n_boot=50)
    assert point == lo == hi == 1.0


def test_stage1_fusion_without_onh_keeps_neural_probability() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    out = {"probs": torch.sigmoid(logits), "member_logits": [logits], "member_names": ["m"]}
    fused = fuse_stage1_outputs(out, onh_biomarkers=None)
    assert fused.onh_biomarkers is None
    assert abs(fused.prob_glaucoma - float(torch.sigmoid(torch.tensor(1.0)))) < 1e-3


def test_pipeline_uses_trained_checkpoints_and_calibration(tmp_path) -> None:
    from src.pipeline import RetinaNetProPipeline
    from src.tier5_reporting.pdf_generator import ClinicalPDFGenerator

    torch.manual_seed(0)
    for fold in (1, 2):
        save_model_checkpoint(tmp_path / f"stage2_grader_fold{fold}.pth", build_model(TINY_S2), TINY_S2)
    save_model_checkpoint(tmp_path / "stage1_multi_label_best.pth", build_model(TINY_S1), TINY_S1)
    (tmp_path / "calibration_params.json").write_text(
        json.dumps({"qwk_thresholds": [0.5, 1.5, 2.5, 3.5], "temperature": 1.7, "conformal_q_hat": 0.8,
                    "conformal_alpha": 0.05})
    )
    pipe = RetinaNetProPipeline(device="cpu", checkpoints_dir=str(tmp_path), tta_views=2)
    assert pipe.stage2_trained and len(pipe.stage2_members) == 2
    assert pipe.cdae is None and pipe.onh_unet is None  # untrained modules are never used
    assert pipe.has_calibration and abs(pipe.temperature - 1.7) < 1e-9

    img = generate_synthetic_fundus_image(size=256, laterality="OD", num_microaneurysms=6, seed=7)
    findings, vis = pipe.evaluate_eye(img, "OD")
    assert findings.is_gradable and findings.onh_biomarkers is None
    assert abs(sum(findings.class_probabilities) - 1.0) < 1e-3
    assert all("Microaneurysm" not in c.lesion_type for c in findings.lesion_candidates)
    assert vis.gradcam_heatmap.max() <= 1.0

    report, _ = pipe.evaluate_bilateral_patient("P-1", od_image_rgb=img)
    assert "unavailable" in report.od_findings.other_pathologies_note
    assert ClinicalPDFGenerator().generate_pdf_bytes(report)[:4] == b"%PDF"
