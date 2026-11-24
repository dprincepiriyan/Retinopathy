"""ImageNet-pretrained `timm` backbones for Stage-1 triage and Stage-2 DR grading.

With only ~2.7k labelled training scans per stage, training from scratch is the
single largest performance limiter. These wrappers put the project's task heads
(multi-label sigmoid head; CORAL tri-head) on top of any `timm` backbone and own
their input normalisation, so datasets and the inference pipeline can keep
feeding plain [0, 1] RGB tensors.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.tier3_grading.gem_pooling import GeMPooling2d
from src.tier3_grading.multihead_grading import CORAL_BIAS_PARAM_SUFFIXES, TriHeadDiagnosticLayer


def _require_timm():
    try:
        import timm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "Pretrained backbones need `timm` (pip install 'timm>=0.9.8')."
        ) from exc
    import timm

    return timm


class InputNormalizer(nn.Module):
    """Maps [0, 1] RGB tensors to the backbone's expected mean/std normalisation."""

    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)


def _build_backbone(
    backbone_name: str,
    pretrained: bool,
    drop_path_rate: float,
) -> Tuple[nn.Module, int, InputNormalizer]:
    timm = _require_timm()
    kwargs = {"pretrained": pretrained, "num_classes": 0, "global_pool": ""}
    if drop_path_rate > 0:
        kwargs["drop_path_rate"] = drop_path_rate
    try:
        backbone = timm.create_model(backbone_name, **kwargs)
    except TypeError:
        # A few architectures do not accept drop_path_rate.
        kwargs.pop("drop_path_rate", None)
        backbone = timm.create_model(backbone_name, **kwargs)

    cfg = getattr(backbone, "pretrained_cfg", None) or getattr(backbone, "default_cfg", {}) or {}
    mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    return backbone, int(backbone.num_features), InputNormalizer(mean, std)


def _to_feature_map(features: torch.Tensor) -> torch.Tensor:
    """Ensure backbone output is (B, C, H, W) (some timm models emit NHWC)."""
    if features.ndim != 4:
        raise ValueError(
            f"Backbone must return a spatial feature map, got shape {tuple(features.shape)}. "
            "Use a CNN / hierarchical backbone (EfficientNet, ConvNeXt, ResNet, ...)."
        )
    return features


class PretrainedTriageNet(nn.Module):
    """Stage-1 multi-label triage: pretrained backbone + GeM + independent sigmoid logits."""

    lesion_maps_supervised = False

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        num_classes: int = 3,
        dropout: float = 0.3,
        drop_path_rate: float = 0.1,
        gem_p: float = 3.0,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim, self.normalize = _build_backbone(backbone_name, pretrained, drop_path_rate)
        self.pool = GeMPooling2d(initial_p=gem_p, learnable=True)
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        fmap = _to_feature_map(self.backbone.forward_features(self.normalize(x)))
        logits = self.classifier(self.drop(self.pool(fmap)))
        return {
            "logits": logits,
            "member_logits": [logits],
            "member_weights": [1.0],
            "member_names": ["PretrainedBackbone"],
            "probs": torch.sigmoid(logits),
            "feature_map": fmap,
        }


class PretrainedOrdinalGrader(nn.Module):
    """Stage-2 DR grader: pretrained backbone + GeM + CORAL tri-head.

    Output dict mirrors `ResViTPro` so training, calibration and the pipeline can
    treat both architectures identically. `lesion_maps` / `cross_attention_weights`
    are None because this model has no supervised lesion or cross-attention branch.
    """

    lesion_maps_supervised = False

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        num_classes: int = 5,
        dropout: float = 0.3,
        drop_path_rate: float = 0.2,
        gem_p: float = 3.0,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim, self.normalize = _build_backbone(backbone_name, pretrained, drop_path_rate)
        self.gem_pool = GeMPooling2d(initial_p=gem_p, learnable=True)
        self.tri_head = TriHeadDiagnosticLayer(in_features=feat_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        fmap = _to_feature_map(self.backbone.forward_features(self.normalize(x)))
        emb = self.gem_pool(fmap)
        tri_out = self.tri_head(emb)
        return {
            "tri_head": tri_out,
            "feature_map": fmap,
            "fpn_stride4": fmap,  # backward-compatible alias used by older callers
            "lesion_maps": None,
            "cross_attention_weights": None,
            "fused_embedding": emb,
        }


def parameter_groups(
    model: nn.Module,
    backbone_lr_mult: float,
    weight_decay: float,
    coral_bias_lr_mult: float = 1.0,
) -> List[dict]:
    """AdamW parameter groups.

    - Pretrained backbone parameters (names starting with `backbone.`) get `lr_mult`
      = `backbone_lr_mult` (typically 0.1-0.3) so ImageNet features are not destroyed.
    - The CORAL threshold scalars (`tri_head.base_bias`, `tri_head.bias_deltas`) get
      `lr_mult` = `coral_bias_lr_mult` (default 1.0). Their *initialisation* is what keeps every
      grade predictable; on real APTOS features a boosted lr let the loss shrink the gaps
      again until grade 3 was never predicted, so the default leaves it at the base lr.
    - Norm weights, biases and the GeM `p` get no weight decay.
    The schedule sets `group["lr"] = base_lr * factor * group["lr_mult"]`.
    """
    groups: Dict[Tuple[float, bool], List[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if name.endswith(CORAL_BIAS_PARAM_SUFFIXES):
            mult = coral_bias_lr_mult
        elif name.startswith("backbone."):
            mult = backbone_lr_mult
        else:
            mult = 1.0
        no_decay = param.ndim <= 1 or name.endswith(".bias") or name.endswith(".p")
        groups.setdefault((float(mult), no_decay), []).append(param)

    return [
        {"params": params, "lr_mult": mult, "weight_decay": 0.0 if no_decay else weight_decay}
        for (mult, no_decay), params in groups.items()
    ]
