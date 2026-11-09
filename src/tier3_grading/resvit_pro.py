"""Model 3A: ResViT-Pro — High-Resolution Hybrid CNN + Vision Transformer with Bidirectional Cross-Attention.

Overcomes Bottleneck 1 (52% missed Mild DR in the reference paper):
1. Multi-Scale FPN Convolutional Stream preserving Stride-4 feature maps (sensitive to <10px microaneurysms).
2. Global Vision Transformer Stream capturing 4-quadrant distribution (clinical 4-2-1 rule).
3. Bidirectional Cross-Attention:
   - Local CNN features query Global ViT quadrant tokens.
   - Global ViT tokens query Local CNN micro-lesion coordinates.
4. Lesion-Guided Attention Branch (MA, HE, EX, CWS spatial maps) + Learnable GeM Pooling.
5. Tri-Head Diagnostic Output Layer (CORAL ordinal + Referable DR binary + Continuous severity).
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tier3_grading.gem_pooling import GeMPooling2d
from src.tier3_grading.lesion_guided_net import LesionGuidedAttentionBranch
from src.tier3_grading.multihead_grading import TriHeadDiagnosticLayer, TriHeadOutput


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional Cross-Attention bridging local CNN tokens and global ViT tokens."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_cnn = nn.LayerNorm(dim)
        self.norm_vit = nn.LayerNorm(dim)

        self.cnn_queries_vit = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.vit_queries_cnn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_cnn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.ffn_vit = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self, cnn_tokens: torch.Tensor, vit_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exchange context between local CNN tokens (B, N_c, D) and global ViT tokens (B, N_v, D).

        Returns:
            Tuple of (updated_cnn_tokens, updated_vit_tokens, cross_attention_weights).
        """
        c_norm = self.norm_cnn(cnn_tokens)
        v_norm = self.norm_vit(vit_tokens)

        c_attn, attn_c2v = self.cnn_queries_vit(query=c_norm, key=v_norm, value=v_norm)
        v_attn, attn_v2c = self.vit_queries_cnn(query=v_norm, key=c_norm, value=c_norm)

        c_out = cnn_tokens + c_attn
        c_out = c_out + self.ffn_cnn(c_out)

        v_out = vit_tokens + v_attn
        v_out = v_out + self.ffn_vit(v_out)

        return c_out, v_out, attn_v2c


class ResBlockFPN(nn.Module):
    """Residual block used in the multi-scale Stride-4 FPN CNN stream."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class ResViTPro(nn.Module):
    """Upgraded ResViT-Pro architecture for fine-grained 5-class DR severity grading."""

    def __init__(
        self,
        in_channels: int = 3,
        fpn_dim: int = 64,
        num_classes: int = 5,
        num_heads: int = 4,
        gem_p: float = 3.0,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.fpn_dim = fpn_dim

        # 1. Multi-scale CNN stream retaining Stride-4 high-resolution feature maps
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, fpn_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim // 2),
            nn.SiLU(inplace=True),
        )
        # Stride 4 stage (critical for <10px microaneurysms)
        self.stage_s4 = ResBlockFPN(fpn_dim // 2, fpn_dim, stride=2)
        # Stride 8 stage
        self.stage_s8 = ResBlockFPN(fpn_dim, fpn_dim, stride=2)
        # Stride 16 stage
        self.stage_s16 = ResBlockFPN(fpn_dim, fpn_dim, stride=2)

        # Top-down FPN lateral fusion back into Stride-4 map
        self.fpn_smooth_s4 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )

        # 2. Lesion-Guided Spatial Attention on the Stride-4 FPN map
        self.lesion_branch = LesionGuidedAttentionBranch(in_channels=fpn_dim, num_lesion_types=4)

        # 3. Global Vision Transformer stream on 8x8 patch grid
        self.vit_grid_size = 8
        self.vit_pos_embed = nn.Parameter(
            torch.randn(1, self.vit_grid_size * self.vit_grid_size, fpn_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fpn_dim,
            nhead=num_heads,
            dim_feedforward=fpn_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Bidirectional Cross-Attention between CNN local grid and ViT global tokens
        self.bi_cross_attn = BidirectionalCrossAttention(dim=fpn_dim, num_heads=num_heads)

        # 5. Generalized Mean (GeM) Pooling on high-resolution Stride-4 lesion-gated map
        self.gem_pool = GeMPooling2d(initial_p=gem_p, learnable=True)

        # 6. Tri-Head Diagnostic Output Layer (fused CNN GeM + Cross-Attended Local + Global ViT)
        self.tri_head = TriHeadDiagnosticLayer(
            in_features=fpn_dim * 3,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | TriHeadOutput]:
        """Forward pass returning TriHeadOutput, Stride-4 FPN features, lesion maps, and attention weights."""
        s2 = self.stem(x)
        s4 = self.stage_s4(s2)
        s8 = self.stage_s8(s4)
        s16 = self.stage_s16(s8)

        # Top-down FPN fusion into Stride-4 resolution
        up8 = F.interpolate(s16, size=s8.shape[2:], mode="bilinear", align_corners=False) + s8
        up4 = F.interpolate(up8, size=s4.shape[2:], mode="bilinear", align_corners=False) + s4
        fpn_s4 = self.fpn_smooth_s4(up4)

        # Lesion-guided spatial attention on Stride-4 feature map
        lesion_gated_s4, lesion_prob_maps = self.lesion_branch(fpn_s4)

        # Local CNN tokens pooled to 8x8 spatial tokens + Global ViT tokens
        local_grid = F.adaptive_avg_pool2d(lesion_gated_s4, (self.vit_grid_size, self.vit_grid_size))
        cnn_tokens = local_grid.flatten(2).transpose(1, 2)  # (B, 64, D)

        vit_in = F.adaptive_avg_pool2d(s16, (self.vit_grid_size, self.vit_grid_size)).flatten(2).transpose(1, 2)
        vit_tokens = self.vit_encoder(vit_in + self.vit_pos_embed)  # (B, 64, D)

        # Bidirectional Cross-Attention
        c_attended, v_attended, attn_weights = self.bi_cross_attn(cnn_tokens, vit_tokens)

        # Pool high-res Stride-4 map with GeM Pooling so focal <10px microaneurysms stand out
        gem_feat = self.gem_pool(lesion_gated_s4)  # (B, D)
        c_feat = c_attended.mean(dim=1)            # (B, D)
        v_feat = v_attended.mean(dim=1)            # (B, D)

        fused_embedding = torch.cat([gem_feat, c_feat, v_feat], dim=1)  # (B, 3*D)
        tri_out = self.tri_head(fused_embedding)

        return {
            "tri_head": tri_out,
            "fpn_stride4": lesion_gated_s4,
            "lesion_maps": lesion_prob_maps,
            "cross_attention_weights": attn_weights,
            "fused_embedding": fused_embedding,
        }
