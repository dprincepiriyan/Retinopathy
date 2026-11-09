"""Tests for Tier 5 Clinical Report Safety, Abstention Rules, Bilateral Aggregation & PDF Export."""

import numpy as np

from app.assets.synthetic_fundus import generate_synthetic_fundus_image
from src.pipeline import RetinaNetProPipeline
from src.tier2_triage.glaucoma_onh_branch import compute_vcdr_and_isnt_metrics
from src.tier4_ensemble.conformal_predictor import SplitConformalPredictor
from src.tier5_reporting.bilateral_aggregator import BilateralClinicalAggregator
from src.tier5_reporting.findings_schema import EyeClinicalFindings, ONHBiomarkerSummary
from src.tier5_reporting.pdf_generator import ClinicalPDFGenerator
from src.tier5_reporting.template_engine import DeterministicClinicalTemplateEngine
from src.utils.iqa_gatekeeper import IQAGatekeeper


def test_iqa_gatekeeper_rejects_blur_and_glare() -> None:
    gate = IQAGatekeeper()
    good_img = generate_synthetic_fundus_image(size=256, laterality="OD", seed=1)
    blur_img = generate_synthetic_fundus_image(size=256, laterality="OD", simulate_severe_blur=True, seed=2)
    glare_img = generate_synthetic_fundus_image(size=256, laterality="OS", simulate_flash_glare=True, seed=3)

    res_good = gate.evaluate(good_img)
    res_blur = gate.evaluate(blur_img)
    res_glare = gate.evaluate(glare_img)

    assert res_good.is_gradable is True
    assert res_blur.is_gradable is False
    assert any("blur" in r.lower() for r in res_blur.rejection_reasons)
    assert res_glare.is_gradable is False
    assert any("glare" in r.lower() for r in res_glare.rejection_reasons)


def test_conformal_predictor_coverage_and_abstention() -> None:
    cp = SplitConformalPredictor(alpha=0.05, max_acceptable_set_size=2)
    # High confidence Grade 2
    res_high = cp.predict_single(np.array([0.02, 0.05, 0.85, 0.05, 0.03]), ensemble_std=0.04)
    assert res_high.confidence_tier == "HIGH"
    assert res_high.abstain_flag is False
    assert res_high.prediction_set == [2]

    # Borderline Grade 1 vs Grade 2 -> Moderate confidence, conservative grade = 2
    res_mod = cp.predict_single(np.array([0.05, 0.42, 0.45, 0.05, 0.03]), ensemble_std=0.10)
    assert res_mod.confidence_tier == "MODERATE"
    assert res_mod.abstain_flag is False
    assert res_mod.conservative_referral_grade == 2

    # Flat/ambiguous distribution -> Low confidence abstention
    res_low = cp.predict_single(np.array([0.22, 0.21, 0.20, 0.19, 0.18]), ensemble_std=0.30)
    assert res_low.confidence_tier == "LOW_ABSTAIN"
    assert res_low.abstain_flag is True


def test_ungradable_scan_suppresses_automated_grade_and_flags_human_review() -> None:
    pipeline = RetinaNetProPipeline(device="cpu")
    blur_od = generate_synthetic_fundus_image(size=256, laterality="OD", simulate_severe_blur=True, seed=55)
    findings, _ = pipeline.evaluate_eye(blur_od, laterality="OD")

    assert findings.is_gradable is False
    assert findings.abstain_flag is True
    assert findings.displayed_dr_grade is None
    assert "Human Review Required" in findings.displayed_grade_label


def test_bilateral_worse_eye_priority_and_zero_hallucination_pdf() -> None:
    aggregator = BilateralClinicalAggregator()
    template_engine = DeterministicClinicalTemplateEngine()
    pdf_gen = ClinicalPDFGenerator(template_engine)

    od_mild = EyeClinicalFindings(
        laterality="OD",
        is_gradable=True,
        iqa_quality_score=0.91,
        prob_dr=0.62,
        prob_glaucoma=0.12,
        prob_cataract=0.05,
        onh_biomarkers=ONHBiomarkerSummary(
            vcdr=0.34, hcdr=0.33, area_ratio=0.11, isnt_rule_violated=False, glaucoma_biometric_risk_score=0.09
        ),
        raw_primary_dr_grade=1,
        conservative_dr_grade=1,
        continuous_severity_score=1.10,
        prob_referable_dr=0.18,
        class_probabilities=[0.15, 0.70, 0.10, 0.03, 0.02],
        conformal_prediction_set=[1],
        confidence_tier="HIGH",
        abstain_flag=False,
        ensemble_std=0.04,
    )

    os_severe_and_glaucoma = EyeClinicalFindings(
        laterality="OS",
        is_gradable=True,
        iqa_quality_score=0.88,
        prob_dr=0.94,
        prob_glaucoma=0.78,
        prob_cataract=0.10,
        onh_biomarkers=ONHBiomarkerSummary(
            vcdr=0.68, hcdr=0.62, area_ratio=0.42, isnt_rule_violated=True, glaucoma_biometric_risk_score=0.84
        ),
        raw_primary_dr_grade=3,
        conservative_dr_grade=3,
        continuous_severity_score=3.15,
        prob_referable_dr=0.92,
        class_probabilities=[0.01, 0.03, 0.12, 0.76, 0.08],
        conformal_prediction_set=[3],
        confidence_tier="HIGH",
        abstain_flag=False,
        ensemble_std=0.05,
    )

    report = aggregator.aggregate(
        patient_id="PT-TEST-900",
        od_findings=od_mild,
        os_findings=os_severe_and_glaucoma,
        patient_name="Test Patient",
        patient_age=63,
    )

    # Patient-level grade must follow the worse eye (OS Grade 3 Severe NPDR)
    assert report.patient_overall_dr_grade == 3
    assert "OS" in report.patient_worse_eye
    assert report.is_referable_dr is True
    assert report.referral_timeline == "2–4 Weeks"
    assert "vCDR: 0.34" in report.bilateral_other_pathologies_summary
    assert "vCDR: 0.68" in report.bilateral_other_pathologies_summary

    html = template_engine.render_html_report(report)
    assert "PT-TEST-900" in html
    assert "Grade 3" in html
    assert "2–4 Weeks" in html

    pdf_bytes = pdf_gen.generate_pdf_bytes(report)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 400


def test_onh_vcdr_and_isnt_computation() -> None:
    disc = np.zeros((128, 128), dtype=np.uint8)
    cup = np.zeros((128, 128), dtype=np.uint8)
    # Draw Optic Disc of radius 40 and large vertical Cup of radius 28 (vCDR = 0.70)
    import cv2

    cv2.circle(disc, (64, 64), 40, 1, thickness=-1)
    cv2.ellipse(cup, (64, 64), (20, 28), 0, 0, 360, 1, thickness=-1)

    bio = compute_vcdr_and_isnt_metrics(disc, cup, laterality="OD")
    assert 0.65 <= bio.vcdr <= 0.75
    assert bio.isnt_rule_violated is True
    assert bio.glaucoma_biometric_risk_score > 0.70
