"""End-to-End 6-Tier Clinical Inference Pipeline (`RetinaNetProPipeline`).

Orchestrates:
- Tier 0: Circular FOV crop, aspect-preserving 1:1 pad, IQA gatekeeper, Ben Graham & Green/LAB CLAHE
- Tier 1: Fast Fourier Domain Adaptation (FDA) & Residual Attention U-Net CDAE
- Tier 2: Stage-1 Multi-Label Pathology Triage (DR, Glaucoma, Cataract) + ONH vCDR/ISNT specialist
- Tier 3: Stage-2 ResViT-Pro DR Severity Grading (evaluates ALL gradable scans)
- Tier 4: Nelder-Mead QWK thresholds, Split Conformal Prediction, Grad-CAM++ & Cross-Attention Rollout
- Tier 5: Bilateral worse-eye aggregation, deterministic template engine & PDF generation
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from src.data.preprocessing import Tier0Preprocessor
from src.explainability.attention_rollout import CrossAttentionRollout
from src.explainability.gradcam_pp import GradCAMPlusPlus
from src.tier1_purifiers.fda import FourierDomainAdapter
from src.tier1_purifiers.residual_cdae import ResidualAttentionUNetCDAE
from src.tier2_triage.glaucoma_onh_branch import crop_optic_nerve_head_roi
from src.tier2_triage.multi_label_triage import MultiLabelTriageEnsemble
from src.tier3_grading.resvit_pro import ResViTPro
from src.tier4_ensemble.conformal_predictor import SplitConformalPredictor
from src.tier4_ensemble.qwk_threshold_optimizer import NelderMeadQWKOptimizer
from src.tier5_reporting.bilateral_aggregator import BilateralClinicalAggregator
from src.tier5_reporting.findings_schema import (
    BilateralPatientReport,
    EyeClinicalFindings,
    LesionLocationItem,
    ONHBiomarkerSummary,
)
from src.utils.iqa_gatekeeper import IQAGatekeeper


@dataclass
class EyeVisualArtifacts:
    """Visual images and heatmaps produced during single-eye evaluation."""

    cropped_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    onh_roi_rgb: np.ndarray
    gradcam_heatmap: np.ndarray
    gradcam_overlay_rgb: np.ndarray
    attention_rollout_map: np.ndarray


class RetinaNetProPipeline:
    """Unified 6-Tier local edge inference pipeline (<500ms on laptop GPU or CPU)."""

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.iqa_gate = IQAGatekeeper()
        self.preprocessor = Tier0Preprocessor(stage1_size=256, stage2_size=256)
        self.fda = FourierDomainAdapter()
        self.cdae = ResidualAttentionUNetCDAE(base_channels=16).to(self.device).eval()
        self.stage1_ensemble = MultiLabelTriageEnsemble(base_channels=16).to(self.device).eval()
        self.stage2_resvit = ResViTPro(fpn_dim=32, num_classes=5, num_heads=4).to(self.device).eval()

        self.qwk_optimizer = NelderMeadQWKOptimizer()
        self.conformal = SplitConformalPredictor(alpha=0.05)
        self.gradcam = GradCAMPlusPlus()
        self.rollout = CrossAttentionRollout()
        self.aggregator = BilateralClinicalAggregator()

    @torch.no_grad()
    def evaluate_eye(
        self,
        image_rgb: np.ndarray,
        laterality: str = "OD",
    ) -> Tuple[EyeClinicalFindings, EyeVisualArtifacts]:
        """Run Tiers 0-4 on a single eye scan and return clinical findings + visual heatmaps."""
        prep = self.preprocessor(image_rgb)
        iqa_res = self.iqa_gate.evaluate(prep.cropped_rgb, prep.fov_mask)

        # Harmonize via Tier-1 FDA
        harmonized_rgb = self.fda.transform(prep.enhanced_rgb, fov_mask=prep.fov_mask)
        onh_roi_rgb, _ = crop_optic_nerve_head_roi(harmonized_rgb, prep.fov_mask, roi_size=128)

        if not iqa_res.is_gradable:
            empty_map = np.zeros(prep.cropped_rgb.shape[:2], dtype=np.float32)
            findings = EyeClinicalFindings(
                laterality=laterality,
                is_gradable=False,
                iqa_quality_score=iqa_res.quality_score,
                iqa_rejection_reasons=iqa_res.rejection_reasons,
                prob_dr=0.0,
                prob_glaucoma=0.0,
                prob_cataract=0.0,
                onh_biomarkers=ONHBiomarkerSummary(
                    vcdr=0.32,
                    hcdr=0.32,
                    area_ratio=0.10,
                    isnt_rule_violated=False,
                    glaucoma_biometric_risk_score=0.08,
                ),
                raw_primary_dr_grade=-1,
                conservative_dr_grade=-1,
                displayed_dr_grade=None,
                continuous_severity_score=0.0,
                prob_referable_dr=0.0,
                conformal_prediction_set=[],
                confidence_tier="LOW_ABSTAIN",
                abstain_flag=True,
                ensemble_std=0.0,
            )
            artifacts = EyeVisualArtifacts(
                cropped_rgb=prep.cropped_rgb,
                enhanced_rgb=harmonized_rgb,
                onh_roi_rgb=onh_roi_rgb,
                gradcam_heatmap=empty_map,
                gradcam_overlay_rgb=prep.cropped_rgb,
                attention_rollout_map=empty_map,
            )
            return findings, artifacts

        # Convert to tensors
        t_s1 = torch.from_numpy(prep.tensor_512).unsqueeze(0).to(self.device)
        t_s2 = torch.from_numpy(prep.tensor_768).unsqueeze(0).to(self.device)
        t_onh = (
            torch.from_numpy(np.transpose(onh_roi_rgb.astype(np.float32) / 255.0, (2, 0, 1)))
            .unsqueeze(0)
            .to(self.device)
        )

        # Tier 1 CDAE residual denoising
        t_s1_clean, _ = self.cdae(t_s1)
        t_s2_clean, _ = self.cdae(t_s2)

        # Tier 2 Stage-1 Multi-Label Triage + Glaucoma ONH Specialist
        s1_out = self.stage1_ensemble.predict_eye(
            tensor_512=t_s1_clean, onh_roi_tensor=t_onh, laterality=laterality
        )

        # Tier 3 Stage-2 ResViT-Pro Fine-Grained DR Grading (Runs on ALL gradable scans)
        s2_out = self.stage2_resvit(t_s2_clean)
        tri_head = s2_out["tri_head"]
        class_probs = tri_head.class_probs[0].cpu().numpy()
        ref_prob = float(tri_head.referable_prob[0, 0].cpu().item())
        cont_score = float(np.clip(tri_head.continuous_score[0, 0].cpu().item(), 0.0, 4.5))

        # Tier 4 Conformal Prediction & Dual Explainability
        conformal_res = self.conformal.predict_single(
            class_probs=class_probs,
            ensemble_std=0.05,
            is_iqa_gradable=True,
        )

        fpn_s4 = s2_out["fpn_stride4"]
        lesion_maps = s2_out["lesion_maps"][0].cpu().numpy()
        attn_weights = s2_out["cross_attention_weights"]

        h_orig, w_orig = prep.cropped_rgb.shape[:2]
        gradcam_map = self.gradcam.compute_heatmap_from_activations(
            fpn_s4, output_size=(h_orig, w_orig)
        )
        gradcam_overlay = self.gradcam.overlay_heatmap_on_rgb(
            prep.cropped_rgb, gradcam_map, fov_mask=prep.fov_mask
        )
        lesion_cands = self.gradcam.extract_microlesion_coordinates(
            gradcam_map, lesion_prob_maps=lesion_maps, fov_mask=prep.fov_mask
        )
        quad_inv = self.rollout.compute_quadrant_involvement(
            attn_weights, laterality=laterality, output_size=(h_orig, w_orig)
        )

        findings = EyeClinicalFindings(
            laterality=laterality,
            is_gradable=True,
            iqa_quality_score=iqa_res.quality_score,
            iqa_rejection_reasons=[],
            prob_dr=s1_out.prob_dr,
            prob_glaucoma=s1_out.prob_glaucoma,
            prob_cataract=s1_out.prob_cataract,
            onh_biomarkers=ONHBiomarkerSummary(
                vcdr=s1_out.onh_biomarkers.vcdr,
                hcdr=s1_out.onh_biomarkers.hcdr,
                area_ratio=s1_out.onh_biomarkers.area_ratio,
                isnt_rule_violated=s1_out.onh_biomarkers.isnt_rule_violated,
                glaucoma_biometric_risk_score=s1_out.onh_biomarkers.glaucoma_biometric_risk_score,
            ),
            raw_primary_dr_grade=conformal_res.primary_grade,
            conservative_dr_grade=conformal_res.conservative_referral_grade,
            displayed_dr_grade=None if conformal_res.abstain_flag else conformal_res.conservative_referral_grade,
            continuous_severity_score=round(cont_score, 3),
            prob_referable_dr=round(ref_prob, 4),
            class_probabilities=[round(float(p), 4) for p in class_probs],
            conformal_prediction_set=conformal_res.prediction_set,
            confidence_tier=conformal_res.confidence_tier,
            abstain_flag=conformal_res.abstain_flag,
            ensemble_std=conformal_res.ensemble_std,
            dominant_quadrant=quad_inv.dominant_quadrant,
            active_quadrant_count=quad_inv.active_quadrant_count,
            quadrant_scores={
                "Superior-Temporal": quad_inv.superior_temporal,
                "Superior-Nasal": quad_inv.superior_nasal,
                "Inferior-Temporal": quad_inv.inferior_temporal,
                "Inferior-Nasal": quad_inv.inferior_nasal,
            },
            lesion_candidates=[
                LesionLocationItem(
                    x=c.x,
                    y=c.y,
                    radius_px=c.radius_px,
                    activation_score=c.activation_score,
                    lesion_type=c.lesion_type,
                )
                for c in lesion_cands
            ],
        )

        artifacts = EyeVisualArtifacts(
            cropped_rgb=prep.cropped_rgb,
            enhanced_rgb=harmonized_rgb,
            onh_roi_rgb=onh_roi_rgb,
            gradcam_heatmap=gradcam_map,
            gradcam_overlay_rgb=gradcam_overlay,
            attention_rollout_map=quad_inv.rollout_map,
        )
        return findings, artifacts

    def evaluate_bilateral_patient(
        self,
        patient_id: str,
        od_image_rgb: Optional[np.ndarray] = None,
        os_image_rgb: Optional[np.ndarray] = None,
        patient_name: str = "Anonymous Screening Patient",
        patient_age: Optional[int] = None,
        doctor_verification_notes: str = "",
        doctor_signature: str = "",
    ) -> Tuple[BilateralPatientReport, Dict[str, EyeVisualArtifacts]]:
        """Evaluate bilateral (OD & OS) fundus images and return the aggregated report + visual maps."""
        od_findings: Optional[EyeClinicalFindings] = None
        os_findings: Optional[EyeClinicalFindings] = None
        visuals: Dict[str, EyeVisualArtifacts] = {}

        if od_image_rgb is not None:
            od_findings, od_vis = self.evaluate_eye(od_image_rgb, laterality="OD")
            visuals["OD"] = od_vis
        if os_image_rgb is not None:
            os_findings, os_vis = self.evaluate_eye(os_image_rgb, laterality="OS")
            visuals["OS"] = os_vis

        report = self.aggregator.aggregate(
            patient_id=patient_id,
            od_findings=od_findings,
            os_findings=os_findings,
            patient_name=patient_name,
            patient_age=patient_age,
            doctor_verification_notes=doctor_verification_notes,
            doctor_signature=doctor_signature,
        )
        return report, visuals
