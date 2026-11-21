"""End-to-End 6-Tier Clinical Inference Pipeline (`RetinaNetProPipeline`).

Orchestrates:
- Tier 0: Circular FOV crop, aspect-preserving 1:1 pad, IQA gatekeeper, Green/LAB CLAHE
         (identical to the training-data writer, incl. INTER_AREA resize + JPEG q96 round-trip)
- Tier 1: FDA harmonisation (ONH ROI only); Residual CDAE only if a trained checkpoint exists
- Tier 2: Stage-1 multi-label triage (DR, glaucoma, cataract) + ONH vCDR/ISNT (trained ONH model only)
- Tier 3: Stage-2 DR grading - ensemble of all CV-fold checkpoints (optional D4 TTA)
- Tier 4: Temperature scaling, QWK thresholds, split-conformal sets, real Grad-CAM++
- Tier 5: Bilateral worse-eye aggregation, deterministic template engine & PDF generation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

from src.data.preprocessing import Tier0Preprocessor
from src.explainability.attention_rollout import CrossAttentionRollout
from src.explainability.gradcam_pp import GradCAMPlusPlus
from src.tier1_purifiers.fda import FourierDomainAdapter
from src.tier2_triage.glaucoma_onh_branch import compute_vcdr_and_isnt_metrics, crop_optic_nerve_head_roi
from src.tier2_triage.multi_label_triage import MultiLabelTriageEnsemble, fuse_stage1_outputs
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
    """Unified local inference pipeline.

    Safety rules enforced here:
    - Checkpoints are self-describing and loaded with `strict=True`.
    - Optional modules (Tier-1 CDAE denoiser, ONH disc/cup U-Net) are only used when a
      *trained* checkpoint exists; untrained networks never touch patient images.
    - Probabilities are temperature-scaled before conformal sets are built, exactly as
      during calibration, and the primary grade uses the QWK-optimised thresholds.
    """

    STAGE1_CKPT = "stage1_multi_label_best.pth"
    STAGE2_FOLD_GLOB = "stage2_grader_fold*.pth"
    STAGE2_LEGACY_CKPT = "stage2_resvit_pro_768_best.pth"
    ONH_CKPT = "stage1_onh_unet_best.pth"
    CDAE_CKPT = "tier1_residual_cdae_best.pth"
    CALIBRATION_JSON = "calibration_params.json"

    def __init__(
        self,
        device: Optional[str] = None,
        checkpoints_dir: Optional[str] = None,
        tta_views: int = 1,
        jpeg_roundtrip_quality: Optional[int] = 96,
    ) -> None:
        import json
        from pathlib import Path

        from src.models.factory import load_model_checkpoint

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        project_root = Path(__file__).resolve().parents[1]
        ckpt_dir = Path(checkpoints_dir) if checkpoints_dir else (project_root / "checkpoints")
        self.tta_views = int(np.clip(tta_views, 1, 8))
        # Training images were stored as JPEG (quality 96); mimic that at inference for parity.
        self.jpeg_roundtrip_quality = jpeg_roundtrip_quality

        self.iqa_gate = IQAGatekeeper()
        self.fda = FourierDomainAdapter()
        fda_path = project_root / "data" / "processed" / "fda_canonical_amplitude.npy"
        if fda_path.exists():
            try:
                self.fda.reference_amplitude = np.load(str(fda_path))
            except Exception:
                pass

        # Tier-1 CDAE: only if a trained checkpoint exists (random weights would corrupt inputs).
        self.cdae = None
        if (ckpt_dir / self.CDAE_CKPT).exists():
            self.cdae, _, _ = load_model_checkpoint(ckpt_dir / self.CDAE_CKPT, self.device)

        # ONH disc/cup U-Net: only if trained; otherwise biomarkers are reported as unavailable.
        self.onh_unet = None
        if (ckpt_dir / self.ONH_CKPT).exists():
            self.onh_unet, _, _ = load_model_checkpoint(ckpt_dir / self.ONH_CKPT, self.device)

        # Stage-1 triage
        s1_path = ckpt_dir / self.STAGE1_CKPT
        if s1_path.exists():
            self.stage1_model, s1_spec, _ = load_model_checkpoint(s1_path, self.device, task_hint="stage1")
            self.stage1_size = int(s1_spec.get("input_size", 512))
            self.stage1_trained = True
        else:
            self.stage1_model = MultiLabelTriageEnsemble(base_channels=16).to(self.device).eval()
            self.stage1_size = 256  # untrained demo / unit-test mode
            self.stage1_trained = False

        # Stage-2 grading: ensemble of all CV fold models (fallback: legacy single checkpoint).
        self.stage2_members: List[Tuple[torch.nn.Module, int]] = []
        fold_paths = sorted(ckpt_dir.glob(self.STAGE2_FOLD_GLOB))
        if not fold_paths and (ckpt_dir / self.STAGE2_LEGACY_CKPT).exists():
            fold_paths = [ckpt_dir / self.STAGE2_LEGACY_CKPT]
        for path in fold_paths:
            model, spec, _ = load_model_checkpoint(path, self.device, task_hint="stage2")
            self.stage2_members.append((model, int(spec.get("input_size", 768))))
        self.stage2_trained = bool(self.stage2_members)
        if not self.stage2_trained:
            self.stage2_members.append((ResViTPro(fpn_dim=32, num_classes=5, num_heads=4).to(self.device).eval(), 256))
        # Backward-compatible attribute names
        self.stage1_ensemble = self.stage1_model
        self.stage2_resvit = self.stage2_members[0][0]

        self.preprocessor = Tier0Preprocessor(stage1_size=self.stage1_size, stage2_size=self.stage2_members[0][1])

        # Calibration (thresholds, temperature, conformal q_hat) from notebook 03 / calibration CLI
        self.qwk_optimizer = NelderMeadQWKOptimizer()
        self.conformal = SplitConformalPredictor(alpha=0.05)
        self.temperature = 1.0
        self.has_calibration = False
        cal_path = ckpt_dir / self.CALIBRATION_JSON
        if cal_path.exists():
            with open(cal_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
            if "qwk_thresholds" in cal_data:
                self.qwk_optimizer.thresholds_ = np.array(cal_data["qwk_thresholds"], dtype=np.float64)
            if "conformal_q_hat" in cal_data:
                self.conformal.q_hat_ = float(cal_data["conformal_q_hat"])
            if "conformal_alpha" in cal_data:
                self.conformal.alpha = float(cal_data["conformal_alpha"])
            self.temperature = float(cal_data.get("temperature", 1.0))
            self.has_calibration = True

        self.gradcam = GradCAMPlusPlus()
        self.rollout = CrossAttentionRollout()
        self.aggregator = BilateralClinicalAggregator()

    # ------------------------------------------------------------------ helpers
    def _to_tensor(self, enhanced_rgb: np.ndarray, size: int) -> torch.Tensor:
        """Resize exactly like the training-data writer (INTER_AREA + JPEG q96) -> (1, 3, H, W)."""
        img = cv2.resize(enhanced_rgb, (size, size), interpolation=cv2.INTER_AREA)
        if self.jpeg_roundtrip_quality:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_roundtrip_quality)])
            if ok:
                img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0)
        t = t.to(self.device)
        if self.cdae is not None:
            t, _ = self.cdae(t)
        return t

    @staticmethod
    def _tta(x: torch.Tensor, view: int) -> torch.Tensor:
        """D4 view `view` in [0, 8): rotation k = view // 2, horizontal flip if view is odd."""
        x = torch.rot90(x, k=view // 2, dims=(2, 3))
        return torch.flip(x, dims=(3,)) if view % 2 else x

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        if abs(self.temperature - 1.0) < 1e-9:
            return probs
        logits = np.log(np.clip(probs, 1e-8, 1.0)) / max(self.temperature, 1e-3)
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def _grade_stage2(self, enhanced_rgb: np.ndarray) -> Dict[str, object]:
        """Fold-ensemble x TTA Stage-2 prediction for one eye."""
        class_probs, ref_probs, cont_scores = [], [], []
        first_out = None
        for model, size in self.stage2_members:
            x = self._to_tensor(enhanced_rgb, size)
            for v in range(self.tta_views):
                out = model(self._tta(x, v))
                tri = out["tri_head"]
                class_probs.append(tri.class_probs[0].float().cpu().numpy())
                ref_probs.append(float(tri.referable_prob[0, 0].float().cpu().item()))
                cont_scores.append(float(tri.continuous_score[0, 0].float().cpu().item()))
                if first_out is None:
                    first_out = out
        mean_probs = np.mean(class_probs, axis=0)
        return {
            "class_probs": self._apply_temperature(mean_probs / mean_probs.sum()),
            "referable_prob": float(np.mean(ref_probs)),
            "continuous_score": float(np.clip(np.mean(cont_scores), 0.0, 4.5)),
            "ensemble_std": float(np.std(ref_probs)) if len(ref_probs) > 1 else 0.0,
            "first_out": first_out,
        }

    def _gradcam_map(self, enhanced_rgb: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """Real Grad-CAM++ w.r.t. the predicted continuous severity of the first ensemble member."""
        model, size = self.stage2_members[0]
        x = self._to_tensor(enhanced_rgb, size).detach()
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            out = model(x)
            fmap = out["feature_map"]
            fmap.retain_grad()
            out["tri_head"].continuous_score.sum().backward()
            grads = fmap.grad
        model.zero_grad(set_to_none=True)
        return self.gradcam.compute_heatmap_from_activations(fmap.detach(), grads, output_size=output_size)

    @torch.no_grad()
    def evaluate_eye(
        self,
        image_rgb: np.ndarray,
        laterality: str = "OD",
    ) -> Tuple[EyeClinicalFindings, EyeVisualArtifacts]:
        """Run Tiers 0-4 on a single eye scan and return clinical findings + visual heatmaps."""
        prep = self.preprocessor(image_rgb)
        iqa_res = self.iqa_gate.evaluate(prep.cropped_rgb, prep.fov_mask)

        # FDA harmonisation is used for the ONH ROI / display only; the classifiers see exactly
        # the preprocessing they were trained on (FOV crop + green/LAB CLAHE).
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
                onh_biomarkers=None,
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

        # Tier 2: Stage-1 triage (+ ONH biomarkers only from a trained ONH model)
        onh_bio = None
        if self.onh_unet is not None:
            t_onh = (
                torch.from_numpy(np.transpose(onh_roi_rgb.astype(np.float32) / 255.0, (2, 0, 1)))
                .unsqueeze(0)
                .to(self.device)
            )
            seg = self.onh_unet(t_onh)[0].cpu().numpy()
            onh_bio = compute_vcdr_and_isnt_metrics(disc_mask=seg[0], cup_mask=seg[1], laterality=laterality)
        s1_out = fuse_stage1_outputs(
            self.stage1_model(self._to_tensor(prep.enhanced_rgb, self.stage1_size)), onh_bio
        )

        # Tier 3: Stage-2 grading on ALL gradable scans (fold ensemble x TTA)
        s2 = self._grade_stage2(prep.enhanced_rgb)
        class_probs = s2["class_probs"]
        cont_score = s2["continuous_score"]
        primary = int(self.qwk_optimizer.predict(np.array([cont_score]))[0]) if self.has_calibration else None

        # Tier 4: conformal set on temperature-scaled probabilities
        conformal_res = self.conformal.predict_single(
            class_probs=class_probs,
            ensemble_std=s2["ensemble_std"],
            is_iqa_gradable=True,
            primary_grade=primary,
        )

        # Explainability: real Grad-CAM++ (gradients of the continuous severity score)
        h_orig, w_orig = prep.cropped_rgb.shape[:2]
        gradcam_map = self._gradcam_map(prep.enhanced_rgb, (h_orig, w_orig))
        gradcam_overlay = self.gradcam.overlay_heatmap_on_rgb(prep.cropped_rgb, gradcam_map, fov_mask=prep.fov_mask)
        first_out = s2["first_out"]
        stage2_model = self.stage2_members[0][0]
        lesion_maps = first_out.get("lesion_maps")
        lesion_cands = self.gradcam.extract_microlesion_coordinates(
            gradcam_map,
            lesion_prob_maps=lesion_maps[0].cpu().numpy() if lesion_maps is not None else None,
            fov_mask=prep.fov_mask,
            lesion_maps_supervised=bool(getattr(stage2_model, "lesion_maps_supervised", False)),
        )
        attn_weights = first_out.get("cross_attention_weights")
        if attn_weights is not None:
            quad_inv = self.rollout.compute_quadrant_involvement(
                attn_weights, laterality=laterality, output_size=(h_orig, w_orig)
            )
        else:
            small = cv2.resize(gradcam_map, (16, 16), interpolation=cv2.INTER_AREA)
            quad_inv = self.rollout.compute_quadrant_involvement_from_map(
                small, laterality=laterality, output_size=(h_orig, w_orig)
            )

        findings = EyeClinicalFindings(
            laterality=laterality,
            is_gradable=True,
            iqa_quality_score=iqa_res.quality_score,
            iqa_rejection_reasons=[],
            prob_dr=s1_out.prob_dr,
            prob_glaucoma=s1_out.prob_glaucoma,
            prob_cataract=s1_out.prob_cataract,
            onh_biomarkers=(
                ONHBiomarkerSummary(
                    vcdr=onh_bio.vcdr,
                    hcdr=onh_bio.hcdr,
                    area_ratio=onh_bio.area_ratio,
                    isnt_rule_violated=onh_bio.isnt_rule_violated,
                    glaucoma_biometric_risk_score=onh_bio.glaucoma_biometric_risk_score,
                )
                if onh_bio is not None
                else None
            ),
            raw_primary_dr_grade=conformal_res.primary_grade,
            conservative_dr_grade=conformal_res.conservative_referral_grade,
            displayed_dr_grade=None if conformal_res.abstain_flag else conformal_res.conservative_referral_grade,
            continuous_severity_score=round(cont_score, 3),
            prob_referable_dr=round(float(np.clip(s2["referable_prob"], 0.0, 1.0)), 4),
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
