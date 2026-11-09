"""Model 2A: StrongCNN (EfficientNet-B0 style MBConv + Squeeze-and-Excitation).

Upgrades the reference paper's 4-class Softmax StrongCNN into a 3-head
Multi-Label Binary Classifier (DR, Glaucoma, Cataract) with independent Sigmoids.
"""

import torch
import torch.nn as nn


class SqueezeExcitationBlock(nn.Module):
    """Channel-wise Squeeze-and-Excitation attention block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(4, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConvSEBlock(nn.Module):
    """Inverted Residual Mobile Bottleneck block with Squeeze-and-Excitation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: int = 4,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=mid_ch, bias=False
            ),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            SqueezeExcitationBlock(mid_ch, reduction=se_reduction),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return x + out if self.use_residual else out


class StrongCNN(nn.Module):
    """Multi-label StrongCNN for Stage-1 Pathology Triage (DR, Glaucoma, Cataract)."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.stages = nn.Sequential(
            MBConvSEBlock(base_channels, base_channels, stride=1, expand_ratio=2),
            MBConvSEBlock(base_channels, base_channels * 2, stride=2, expand_ratio=4),
            MBConvSEBlock(base_channels * 2, base_channels * 4, stride=2, expand_ratio=4),
            MBConvSEBlock(base_channels * 4, base_channels * 8, stride=2, expand_ratio=4),
        )
        feat_dim = base_channels * 8
        self.global_se = SqueezeExcitationBlock(feat_dim, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.stages(feat)
        feat = self.global_se(feat)
        return self.pool(feat).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw multi-label logits of shape (B, 3) for [DR, Glaucoma, Cataract]."""
        emb = self.forward_features(x)
        return self.classifier[1:](emb)
