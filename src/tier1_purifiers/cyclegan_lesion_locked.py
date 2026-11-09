"""Algorithm 1A: Lesion-Locked CUT / CycleGAN for Colab Training Augmentation.

Used strictly during training-time domain augmentation on Colab GPUs.
Enforces a High-Pass Laplacian Green-channel identity lock so that <10px
microaneurysms and capillary borders cannot be erased or hallucinated by the generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    """Residual block with ReflectionPad and InstanceNorm for style translation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LesionLockedGenerator(nn.Module):
    """Lightweight ResNet generator with high-frequency green-channel structural bypass."""

    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_dim, kernel_size=7, bias=False),
            nn.InstanceNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResnetBlock(base_dim * 2) for _ in range(num_blocks)])
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_dim * 2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.InstanceNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.out_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_dim, in_channels, kernel_size=7),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.initial(x)
        h = self.down1(h)
        h = self.res_blocks(h)
        h = self.up1(h)
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.out_head(h)


class HighPassLesionIdentityLoss(nn.Module):
    """Penalizes any discrepancy in the high-pass Laplacian of the Green channel
    between the original fundus image x and the style-translated image G(x).
    """

    def __init__(self) -> None:
        super().__init__()
        laplacian_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", laplacian_kernel)

    def forward(self, real_rgb: torch.Tensor, translated_rgb: torch.Tensor) -> torch.Tensor:
        # Extract Green channel (index 1) where microaneurysms have highest contrast
        green_real = real_rgb[:, 1:2, :, :]
        green_trans = translated_rgb[:, 1:2, :, :]
        kernel = self.kernel.to(dtype=green_real.dtype, device=green_real.device)
        hp_real = F.conv2d(green_real, kernel, padding=1)
        hp_trans = F.conv2d(green_trans, kernel, padding=1)
        return F.l1_loss(hp_trans, hp_real)
