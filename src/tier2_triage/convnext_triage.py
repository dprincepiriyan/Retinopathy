"""Model 2B: ConvNeXt-V2-Tiny with Coordinate Attention (CA) and Global Response Normalization.

Preserves exact 2D horizontal and vertical positional coordinates when detecting
regional lens opacities (Cataract), optic nerve excavation (Glaucoma), and vascular lesions (DR).
"""

import torch
import torch.nn as nn


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization (GRN) from ConvNeXt-V2."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class CoordinateAttention(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021) factorizing 2D pooling into X and Y directions."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Pool along width -> (B, C, H, 1) and height -> (B, C, 1, W)
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H + W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # (B, mid, 1, W)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 block with 7x7 depthwise conv, GRN, and Coordinate Attention."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(dim * 4)
        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.coord_attn = CoordinateAttention(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.coord_attn(x)
        return residual + x


class ConvNeXtV2CoordinateTriage(nn.Module):
    """ConvNeXt-V2-Tiny with Coordinate Attention for Multi-Label Stage-1 Triage."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        dims: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )
        stages = []
        for i in range(len(dims)):
            stages.append(ConvNeXtV2Block(dims[i]))
            if i < len(dims) - 1:
                stages.append(
                    nn.Sequential(
                        nn.GroupNorm(1, dims[i]),
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    )
                )
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(dims[-1]),
            nn.Dropout(p=dropout),
            nn.Linear(dims[-1], num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return multi-label logits of shape (B, 3)."""
        feat = self.forward_features(x)
        return self.head[1:](feat)
