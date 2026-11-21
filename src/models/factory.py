"""Model factory + self-describing checkpoints.

Checkpoints are saved as::

    {"format_version": 2, "model_spec": {...}, "state_dict": {...}, "metadata": {...}}

so the inference pipeline can rebuild the *exact* architecture and load it with
`strict=True`. A mismatched checkpoint now fails loudly instead of silently
running with partially random weights.

Legacy raw `state_dict` files (format 1, produced by the original notebooks) are
still loadable: the architecture is inferred from their keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

CHECKPOINT_FORMAT_VERSION = 2

STAGE1_ARCHS = ("timm", "legacy_ensemble")
STAGE2_ARCHS = ("timm", "resvit_pro")


def build_model(spec: Dict[str, Any], pretrained_override: Optional[bool] = None) -> nn.Module:
    """Build a model from a spec dict: {"task": "stage1"|"stage2", "arch": ..., **kwargs}."""
    spec = dict(spec)
    task = spec.pop("task")
    arch = spec.pop("arch")
    spec.pop("input_size", None)
    if pretrained_override is not None and "pretrained" in spec:
        spec["pretrained"] = pretrained_override

    if task == "stage1":
        if arch == "timm":
            from src.models.pretrained import PretrainedTriageNet

            return PretrainedTriageNet(**spec)
        if arch == "legacy_ensemble":
            from src.tier2_triage.multi_label_triage import MultiLabelTriageEnsemble

            return MultiLabelTriageEnsemble(**spec)
        raise ValueError(f"Unknown stage1 arch '{arch}'. Choose from {STAGE1_ARCHS}.")

    if task == "stage2":
        if arch == "timm":
            from src.models.pretrained import PretrainedOrdinalGrader

            return PretrainedOrdinalGrader(**spec)
        if arch == "resvit_pro":
            from src.tier3_grading.resvit_pro import ResViTPro

            return ResViTPro(**spec)
        raise ValueError(f"Unknown stage2 arch '{arch}'. Choose from {STAGE2_ARCHS}.")

    if task == "onh_unet":
        from src.tier2_triage.glaucoma_onh_branch import OpticNerveHeadUNet

        return OpticNerveHeadUNet(**spec)

    if task == "cdae":
        from src.tier1_purifiers.residual_cdae import ResidualAttentionUNetCDAE

        return ResidualAttentionUNetCDAE(**spec)

    raise ValueError(f"Unknown task '{task}'.")


def save_model_checkpoint(
    path: str | Path,
    model: nn.Module,
    spec: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a self-describing CPU checkpoint loadable on any machine."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_spec": dict(spec),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": dict(metadata or {}),
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(out)  # atomic: a Colab disconnect mid-write never corrupts the previous file
    return out


def _infer_legacy_spec(state: Dict[str, torch.Tensor], task_hint: Optional[str]) -> Dict[str, Any]:
    if "strong_cnn.stem.0.weight" in state:
        return {
            "task": "stage1",
            "arch": "legacy_ensemble",
            "base_channels": int(state["strong_cnn.stem.0.weight"].shape[0]),
        }
    if "stage_s4.conv.0.weight" in state:
        return {
            "task": "stage2",
            "arch": "resvit_pro",
            "fpn_dim": int(state["stage_s4.conv.0.weight"].shape[0]),
            "num_classes": 5,
            "num_heads": 4,
        }
    raise ValueError(
        f"Cannot infer architecture of legacy checkpoint (task hint: {task_hint}). "
        "Re-save it with src.models.factory.save_model_checkpoint."
    )


def load_model_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    task_hint: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load a checkpoint, rebuild its architecture and load weights with strict=True.

    Returns:
        (model in eval mode on `device`, model_spec, metadata)
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and payload.get("format_version") == CHECKPOINT_FORMAT_VERSION:
        spec = dict(payload["model_spec"])
        state = payload["state_dict"]
        metadata = dict(payload.get("metadata", {}))
    else:
        state = payload
        spec = _infer_legacy_spec(state, task_hint)
        metadata = {"legacy_format": True}

    # Never download ImageNet weights at load time: the checkpoint already holds trained weights.
    model = build_model(spec, pretrained_override=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, spec, metadata
