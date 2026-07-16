import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stylegan2.model import PixelNorm


# v1 modification:
# These blocks implement the new delta-aware blending architecture:
# 1) HairAdditionBranchV1 preserves complex local hair texture.
# 2) ContextRecoveryBranchV1 restores newly exposed neck/background areas.
# 3) BoundaryFusionHeadV1 predicts a spatial gate for shadow/boundary smoothing.


class ConvBlockV1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchMemoryEncoderV1(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlockV1(in_channels, 64, stride=2),
            ConvBlockV1(64, 128, stride=2),
            ConvBlockV1(128, 256, stride=2),
            ConvBlockV1(256, hidden_dim, stride=2),
            ConvBlockV1(hidden_dim, hidden_dim, stride=1),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image * mask, mask], dim=1)
        return self.net(x)


class CrossAttentionMemoryV1(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = math.sqrt(self.head_dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, queries: torch.Tensor, memory_map: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = queries.shape
        memory = memory_map.flatten(2).transpose(1, 2)

        q = self.to_q(queries).view(bsz, n_tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(memory).view(bsz, memory.size(1), self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(memory).view(bsz, memory.size(1), self.heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, n_tokens, dim)
        return self.out(out)


class MaskStatsProjectorV1(nn.Module):
    def __init__(self, num_masks: int, n_styles: int = 12, hidden_dim: int = 512):
        super().__init__()
        self.n_styles = n_styles
        self.project = nn.Sequential(
            nn.Linear(num_masks * 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, masks: list[torch.Tensor]) -> torch.Tensor:
        pooled = [F.adaptive_avg_pool2d(mask.float(), (4, 4)).flatten(1) for mask in masks]
        stats = torch.cat(pooled, dim=1)
        cond = self.project(stats)
        return cond.unsqueeze(1).expand(-1, self.n_styles, -1)


class HairAdditionBranchV1(nn.Module):
    def __init__(self, modulation_cls, n_styles: int = 12):
        super().__init__()
        self.n_styles = n_styles
        self.pixelnorm = PixelNorm()
        self.source_encoder = PatchMemoryEncoderV1()
        self.color_encoder = PatchMemoryEncoderV1()
        self.source_attn = CrossAttentionMemoryV1()
        self.color_attn = CrossAttentionMemoryV1()
        self.mask_stats = MaskStatsProjectorV1(num_masks=4, n_styles=n_styles)
        self.style_gate = nn.Sequential(
            nn.Linear(4 * 16, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, n_styles),
            nn.Sigmoid(),
        )
        self.modulations = nn.ModuleList(
            [modulation_cls(n_styles, i == 4, inp=512 * 4, middle=1024) for i in range(5)]
        )

    def _mask_stats_flat(
        self,
        keep_mask: torch.Tensor,
        add_mask: torch.Tensor,
        color_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = [
            F.adaptive_avg_pool2d(mask.float(), (4, 4)).flatten(1)
            for mask in (keep_mask, add_mask, color_mask, boundary_mask)
        ]
        return torch.cat(pooled, dim=1)

    def forward(
        self,
        latent_face_tail: torch.Tensor,
        latent_color_tail: torch.Tensor,
        source_image: torch.Tensor,
        color_image: torch.Tensor,
        keep_mask: torch.Tensor,
        add_mask: torch.Tensor,
        color_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        # v1 refinement:
        # Use only the keep region as source memory. Feeding add-region source
        # pixels here encourages the branch to preserve source hair appearance
        # exactly where we want to create new target hair.
        source_memory = self.source_encoder(source_image, keep_mask)
        color_memory = self.color_encoder(color_image, color_mask)

        preserve_ctx = self.source_attn(latent_face_tail, source_memory)
        color_ctx = self.color_attn(latent_face_tail, color_memory)
        delta_ctx = self.mask_stats([keep_mask, add_mask, color_mask, boundary_mask])
        style_gate = self.style_gate(
            self._mask_stats_flat(keep_mask, add_mask, color_mask, boundary_mask)
        ).unsqueeze(-1)

        # v1 refinement:
        # Build the style tail from an explicit source/reference interpolation so
        # the reference latent can actually dominate when the edit adds or
        # replaces a large hair region.
        latent_base = (1.0 - style_gate) * latent_face_tail + style_gate * latent_color_tail

        condition = torch.cat([latent_color_tail, 1.25 * color_ctx, 0.5 * preserve_ctx, delta_ctx], dim=-1)

        dt_latent = self.pixelnorm(latent_base)
        for modulation in self.modulations:
            dt_latent = modulation(dt_latent, condition)
        return latent_base + 0.1 * dt_latent


class ContextRecoveryBranchV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlockV1(7, 64, stride=2),
            ConvBlockV1(64, 128, stride=2),
            ConvBlockV1(128, 256, stride=2),
            ConvBlockV1(256, 512, stride=2),
            ConvBlockV1(512, 512, stride=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )
        # v1 stabilization:
        # Start the disocclusion branch from a near-identity state so it does not
        # inject a large random residual into F-space at the beginning of training.
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        source_image: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        remove_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([source_image, src_mask, tgt_mask, remove_mask, boundary_mask], dim=1)
        return self.net(x)


class BoundaryFusionHeadV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlockV1(7, 32, stride=1),
            ConvBlockV1(32, 32, stride=1),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # v1 stabilization:
        # Bias the initial gate towards "closed" so random context features do
        # not create a gray translucent overlay before the branch learns.
        nn.init.zeros_(self.net[-2].weight)
        if self.net[-2].bias is not None:
            nn.init.constant_(self.net[-2].bias, -4.0)

    def forward(
        self,
        source_image: torch.Tensor,
        add_mask: torch.Tensor,
        remove_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
        delta_f: torch.Tensor,
    ) -> torch.Tensor:
        source_small = F.interpolate(source_image, size=(32, 32), mode="bilinear", align_corners=False)
        add_small = F.interpolate(add_mask, size=(32, 32), mode="bilinear", align_corners=False)
        remove_small = F.interpolate(remove_mask, size=(32, 32), mode="bilinear", align_corners=False)
        boundary_small = F.interpolate(boundary_mask, size=(32, 32), mode="bilinear", align_corners=False)
        delta_energy = delta_f.pow(2).mean(dim=1, keepdim=True)
        x = torch.cat([source_small, add_small, remove_small, boundary_small, delta_energy], dim=1)
        return self.net(x)
