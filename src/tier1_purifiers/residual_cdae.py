"""Algorithm 1C: Residual Attention U-Net CDAE (Convolutional Denoising Autoencoder).

Addresses Bottleneck 1 of the reference paper:
Instead of compressing the entire fundus image through a destructive bottleneck
that erases 1-10px microaneurysms, this network uses dense skip connections with
Attention Gates and predicts ONLY the bounded sensor noise residual:
    x_clean = clamp(x + alpha * Decoder(Encoder(x)), 0.0, 1.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double 3x3 Convolution + GroupNorm + SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialAttentionGate(nn.Module):
    """Attention Gate for U-Net skip connections to preserve focal micro-lesions."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.w_gate = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=True)
        self.w_skip = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=True)
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        attn = self.psi(self.w_gate(gate) + self.w_skip(skip))
        return skip * attn


class ResidualAttentionUNetCDAE(nn.Module):
    """Residual Attention U-Net predicting bounded noise residual Delta_x."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        residual_scale_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.residual_scale_alpha = residual_scale_alpha

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att2 = SpatialAttentionGate(gate_ch=c2, skip_ch=c2, inter_ch=c1)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att1 = SpatialAttentionGate(gate_ch=c1, skip_ch=c1, inter_ch=max(8, c1 // 2))
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.residual_head = nn.Sequential(
            nn.Conv2d(c1, in_channels, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        # Initialize residual head close to zero so untrained network acts as near-identity
        nn.init.zeros_(self.residual_head[0].weight)
        nn.init.zeros_(self.residual_head[0].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (x_clean, delta_x)."""
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        b = self.bottleneck(self.pool2(s2))

        d2 = self.up2(b)
        if d2.shape[2:] != s2.shape[2:]:
            d2 = F.interpolate(d2, size=s2.shape[2:], mode="bilinear", align_corners=False)
        s2_att = self.att2(d2, s2)
        d2 = self.dec2(torch.cat([d2, s2_att], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != s1.shape[2:]:
            d1 = F.interpolate(d1, size=s1.shape[2:], mode="bilinear", align_corners=False)
        s1_att = self.att1(d1, s1)
        d1 = self.dec1(torch.cat([d1, s1_att], dim=1))

        delta_x = self.residual_head(d1)
        x_clean = torch.clamp(x + self.residual_scale_alpha * delta_x, 0.0, 1.0)
        return x_clean, delta_x
