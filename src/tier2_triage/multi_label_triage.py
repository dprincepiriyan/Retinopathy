"""Stage-1 Multi-Label Pathology Triage Coordinator.

Combines StrongCNN (2A), ConvNeXt-V2-Tiny + Coordinate Attention (2B),
and the Optic Nerve Head (ONH) Glaucoma Specialist (2D) to produce independent,
calibrated Sigmoid probabilities for:
  - P(Diabetic Retinopathy)
  - P(Glaucoma)
  - P(Cataract)
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from src.tier2_triage.convnext_triage import ConvNeXtV2CoordinateTriage
from src.tier2_triage.glaucoma_onh_branch import (
    ONHBiomarkers,
    OpticNerveHeadUNet,
)
from src.tier2_triage.strong_cnn import StrongCNN


@dataclass
class Stage1TriageOutput:
    """Output of Tier 2 (Stage-1 Multi-Label Triage & Glaucoma ONH Specialist).

    `onh_biomarkers` is None when no *trained* ONH segmentation checkpoint is
    available: biomarkers computed from an untrained U-Net would be noise.
    """

    prob_dr: float
    prob_glaucoma: float
    prob_cataract: float
    onh_biomarkers: Optional[ONHBiomarkers]
    coexisting_conditions: list[str]
    model_probabilities: Dict[str, Dict[str, float]]


DEFAULT_STAGE1_DECISION_THRESHOLDS = {"dr": 0.40, "glaucoma": 0.45, "cataract": 0.50}


def fuse_stage1_outputs(
    model_out: Dict[str, object],
    onh_biomarkers: Optional[ONHBiomarkers] = None,
    decision_thresholds: Optional[Dict[str, float]] = None,
    onh_fusion_weight: float = 0.45,
) -> Stage1TriageOutput:
    """Turn a Stage-1 model output dict (batch of 1) into clinical triage probabilities.

    Works with any Stage-1 model that returns `probs` and `member_logits`.
    The ONH biometric risk is fused into P(glaucoma) only when real biomarkers exist.
    """
    thr = dict(DEFAULT_STAGE1_DECISION_THRESHOLDS)
    if decision_thresholds:
        thr.update(decision_thresholds)

    p_ens = model_out["probs"][0].detach().float().cpu().numpy()  # type: ignore[index]
    member_logits = model_out.get("member_logits", [])  # type: ignore[union-attr]
    member_names = model_out.get("member_names") or [f"member_{i}" for i in range(len(member_logits))]  # type: ignore[union-attr]

    prob_dr = float(np.clip(p_ens[0], 0.0, 1.0))
    prob_glaucoma = float(np.clip(p_ens[1], 0.0, 1.0))
    if onh_biomarkers is not None:
        prob_glaucoma = float(
            np.clip(
                (1.0 - onh_fusion_weight) * prob_glaucoma
                + onh_fusion_weight * onh_biomarkers.glaucoma_biometric_risk_score,
                0.0,
                1.0,
            )
        )
    prob_cataract = float(np.clip(p_ens[2], 0.0, 1.0))

    detected: list[str] = []
    if prob_dr >= thr["dr"]:
        detected.append("Diabetic Retinopathy")
    if prob_glaucoma >= thr["glaucoma"]:
        detected.append("Glaucoma Suspect")
    if prob_cataract >= thr["cataract"]:
        detected.append("Cataract / Media Opacity")

    model_probs: Dict[str, Dict[str, float]] = {}
    for name, logits in zip(member_names, member_logits):
        p = torch.sigmoid(logits[0].detach().float()).cpu().numpy()
        model_probs[str(name)] = {
            "dr": round(float(p[0]), 4),
            "glaucoma": round(float(p[1]), 4),
            "cataract": round(float(p[2]), 4),
        }

    return Stage1TriageOutput(
        prob_dr=round(prob_dr, 4),
        prob_glaucoma=round(prob_glaucoma, 4),
        prob_cataract=round(prob_cataract, 4),
        onh_biomarkers=onh_biomarkers,
        coexisting_conditions=detected,
        model_probabilities=model_probs,
    )


class MultiLabelTriageEnsemble(nn.Module):
    """Legacy from-scratch ensemble (StrongCNN + ConvNeXt-V2-CA) kept for ablations.

    Prefer `src.models.PretrainedTriageNet` (ImageNet-pretrained) for real training.
    The embedded `onh_unet` is never trained by the Stage-1 loop; the pipeline only
    uses ONH biomarkers from a separately trained ONH checkpoint.
    """

    lesion_maps_supervised = False
    member_weights = (0.45, 0.55)

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        self.strong_cnn = StrongCNN(num_classes=3, base_channels=base_channels)
        self.convnext_ca = ConvNeXtV2CoordinateTriage(
            num_classes=3, dims=(base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        )
        self.onh_unet = OpticNerveHeadUNet(base_ch=16)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        """Forward pass returning member logits and the weighted ensemble probabilities."""
        logits_strong = self.strong_cnn(x)
        logits_convnext = self.convnext_ca(x)
        probs_strong = torch.sigmoid(logits_strong)
        probs_convnext = torch.sigmoid(logits_convnext)
        w_s, w_c = self.member_weights
        probs_ens = w_s * probs_strong + w_c * probs_convnext
        return {
            "logits_strong": logits_strong,
            "logits_convnext": logits_convnext,
            "probs_strong": probs_strong,
            "probs_convnext": probs_convnext,
            # Uniform Stage-1 contract shared with PretrainedTriageNet
            "logits": torch.logit(probs_ens.clamp(1e-6, 1 - 1e-6)),
            "member_logits": [logits_strong, logits_convnext],
            "member_weights": [w_s, w_c],
            "member_names": ["StrongCNN", "ConvNeXtV2_CA"],
            "probs": probs_ens,
        }

