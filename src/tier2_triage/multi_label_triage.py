"""Stage-1 Multi-Label Pathology Triage Coordinator.

Combines StrongCNN (2A), ConvNeXt-V2-Tiny + Coordinate Attention (2B),
and the Optic Nerve Head (ONH) Glaucoma Specialist (2D) to produce independent,
calibrated Sigmoid probabilities for:
  - P(Diabetic Retinopathy)
  - P(Glaucoma)
  - P(Cataract)
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
import torch.nn as nn

from src.tier2_triage.convnext_triage import ConvNeXtV2CoordinateTriage
from src.tier2_triage.glaucoma_onh_branch import (
    ONHBiomarkers,
    OpticNerveHeadUNet,
    compute_vcdr_and_isnt_metrics,
)
from src.tier2_triage.strong_cnn import StrongCNN


@dataclass
class Stage1TriageOutput:
    """Output of Tier 2 (Stage-1 Multi-Label Triage & Glaucoma ONH Specialist)."""

    prob_dr: float
    prob_glaucoma: float
    prob_cataract: float
    onh_biomarkers: ONHBiomarkers
    coexisting_conditions: list[str]
    model_probabilities: Dict[str, Dict[str, float]]


class MultiLabelTriageEnsemble(nn.Module):
    """Ensembles multi-label neural triage models with biometric vCDR/ISNT Glaucoma fusion."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        self.strong_cnn = StrongCNN(num_classes=3, base_channels=base_channels)
        self.convnext_ca = ConvNeXtV2CoordinateTriage(
            num_classes=3, dims=(base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        )
        self.onh_unet = OpticNerveHeadUNet(base_ch=16)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning logits and averaged multi-label probabilities."""
        logits_strong = self.strong_cnn(x)
        logits_convnext = self.convnext_ca(x)
        probs_strong = torch.sigmoid(logits_strong)
        probs_convnext = torch.sigmoid(logits_convnext)
        probs_ens = 0.45 * probs_strong + 0.55 * probs_convnext
        return {
            "logits_strong": logits_strong,
            "logits_convnext": logits_convnext,
            "probs_strong": probs_strong,
            "probs_convnext": probs_convnext,
            "probs_ensemble": probs_ens,
        }

    @torch.no_grad()
    def predict_eye(
        self,
        tensor_512: torch.Tensor,
        onh_roi_tensor: torch.Tensor,
        laterality: str = "OD",
        biomarkers_override: ONHBiomarkers | None = None,
    ) -> Stage1TriageOutput:
        """Run complete Stage-1 inference for a single eye scan."""
        self.eval()
        if tensor_512.ndim == 3:
            tensor_512 = tensor_512.unsqueeze(0)
        if onh_roi_tensor.ndim == 3:
            onh_roi_tensor = onh_roi_tensor.unsqueeze(0)

        out = self.forward(tensor_512)
        p_strong = out["probs_strong"][0].cpu().numpy()
        p_conv = out["probs_convnext"][0].cpu().numpy()
        p_ens = out["probs_ensemble"][0].cpu().numpy()

        if biomarkers_override is not None:
            onh_bio = biomarkers_override
        else:
            seg_masks = self.onh_unet(onh_roi_tensor)[0].cpu().numpy()
            onh_bio = compute_vcdr_and_isnt_metrics(
                disc_mask=seg_masks[0], cup_mask=seg_masks[1], laterality=laterality
            )

        # Fuse neural glaucoma probability with explicit ONH biometric risk score
        prob_dr = float(np.clip(p_ens[0], 0.0, 1.0))
        prob_glaucoma = float(
            np.clip(0.55 * p_ens[1] + 0.45 * onh_bio.glaucoma_biometric_risk_score, 0.0, 1.0)
        )
        prob_cataract = float(np.clip(p_ens[2], 0.0, 1.0))

        detected: list[str] = []
        if prob_dr >= 0.40:
            detected.append("Diabetic Retinopathy")
        if prob_glaucoma >= 0.45:
            detected.append("Glaucoma Suspect")
        if prob_cataract >= 0.50:
            detected.append("Cataract / Media Opacity")

        return Stage1TriageOutput(
            prob_dr=round(prob_dr, 4),
            prob_glaucoma=round(prob_glaucoma, 4),
            prob_cataract=round(prob_cataract, 4),
            onh_biomarkers=onh_bio,
            coexisting_conditions=detected,
            model_probabilities={
                "StrongCNN": {
                    "dr": round(float(p_strong[0]), 4),
                    "glaucoma": round(float(p_strong[1]), 4),
                    "cataract": round(float(p_strong[2]), 4),
                },
                "ConvNeXtV2_CA": {
                    "dr": round(float(p_conv[0]), 4),
                    "glaucoma": round(float(p_conv[1]), 4),
                    "cataract": round(float(p_conv[2]), 4),
                },
            },
        )
