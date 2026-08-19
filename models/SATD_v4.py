import torch
import torch.nn.functional as F
from torch import nn


class ConvGNAct_v4(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaskEncoder_v4(nn.Module):
    def __init__(self, in_ch: int = 5, mid_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct_v4(in_ch, 32),
            ConvGNAct_v4(32, 32),
            ConvGNAct_v4(32, mid_ch),
            ConvGNAct_v4(mid_ch, mid_ch),
        )

    def forward(self, masks_32: torch.Tensor) -> torch.Tensor:
        return self.net(masks_32)


class FeatureProjector_v4(nn.Module):
    def __init__(self, in_ch: int = 512, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShadowEstimator_v4(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct_v4(in_ch, 128),
            ConvGNAct_v4(128, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class BlendNet_v4(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct_v4(in_ch, 256),
            ConvGNAct_v4(256, 128),
            nn.Conv2d(128, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeltaHead_v4(nn.Module):
    def __init__(self, in_ch: int, feat_ch: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct_v4(in_ch, 256),
            ConvGNAct_v4(256, 128),
        )
        self.gate = nn.Conv2d(128, 1, kernel_size=1)
        self.delta = nn.Conv2d(128, feat_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.body(x)
        gate = torch.sigmoid(self.gate(feat))
        delta = self.delta(feat)
        return gate, delta


class SATD_v4(nn.Module):
    """
    Mask channel order:
    [M_add, M_remove, M_keep, M_boundary, M_ref_overlap]
    """

    def __init__(self, feat_ch: int = 512, proj_ch: int = 128, mask_ch: int = 5):
        super().__init__()
        self.mask_encoder = MaskEncoder_v4(mask_ch, 64)
        self.projector = FeatureProjector_v4(feat_ch, proj_ch)
        self.shadow_estimator = ShadowEstimator_v4(proj_ch * 3 + 64 + 3)
        self.blend_net = BlendNet_v4(proj_ch * 5 + 64 + 1)
        self.transfer_head = DeltaHead_v4(proj_ch * 4 + 64 + 1, feat_ch=feat_ch)
        self.cleanup_head = DeltaHead_v4(proj_ch * 3 + 64 + 1, feat_ch=feat_ch)

    @staticmethod
    def _down_rgb(image_256: torch.Tensor) -> torch.Tensor:
        return F.interpolate(image_256, size=(32, 32), mode="bicubic", align_corners=False)

    @staticmethod
    def _prior_logits(masks_32: torch.Tensor, shadow: torch.Tensor) -> torch.Tensor:
        add, remove, keep, boundary, ref_overlap = torch.chunk(masks_32, 5, dim=1)
        eq8_prior = 1.10 * keep + 0.20 * boundary * (1 - shadow) * (1 - add) + 0.12 * ref_overlap
        src_prior = 0.55 * keep + 0.08 * boundary * (1 - shadow) * (1 - add)
        src_inpaint_prior = remove + 0.28 * boundary * shadow * (1 - add) * (1 - ref_overlap)
        shape_prior = add + 0.25 * keep + 0.55 * boundary * (1 - shadow) + 0.20 * ref_overlap
        ref_prior = 1.10 * ref_overlap + 1.10 * add + 0.18 * boundary
        return 2.0 * torch.cat([eq8_prior, src_prior, src_inpaint_prior, shape_prior, ref_prior], dim=1)

    def forward(
        self,
        F_eq8: torch.Tensor,
        F_src: torch.Tensor,
        F_src_inpaint: torch.Tensor,
        F_shape_inpaint: torch.Tensor,
        F_ref: torch.Tensor,
        masks_256: torch.Tensor,
        source_rgb_256: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks_32 = F.interpolate(masks_256.float(), size=(32, 32), mode="nearest")
        mask_feat = self.mask_encoder(masks_32)

        P_eq8 = self.projector(F_eq8)
        P_src = self.projector(F_src)
        P_src_inpaint = self.projector(F_src_inpaint)
        P_shape = self.projector(F_shape_inpaint)
        P_ref = self.projector(F_ref)

        src_rgb_32 = self._down_rgb(source_rgb_256)
        shadow = self.shadow_estimator(torch.cat([P_eq8, P_src, P_src_inpaint, mask_feat, src_rgb_32], dim=1))

        blend_logits = self.blend_net(torch.cat([P_eq8, P_src, P_src_inpaint, P_shape, P_ref, mask_feat, shadow], dim=1))
        blend_logits = blend_logits + self._prior_logits(masks_32, shadow)
        weights = torch.softmax(blend_logits, dim=1)

        fused = (
            weights[:, 0:1] * F_eq8
            + weights[:, 1:2] * F_src
            + weights[:, 2:3] * F_src_inpaint
            + weights[:, 3:4] * F_shape_inpaint
            + weights[:, 4:5] * F_ref
        )
        ref_shape_mass = (weights[:, 3:4] + weights[:, 4:5]).clamp_min(1e-4)
        ref_shape_fused = (
            weights[:, 3:4] * F_shape_inpaint
            + weights[:, 4:5] * F_ref
        ) / ref_shape_mass

        add, remove, keep, boundary, ref_overlap = torch.chunk(masks_32, 5, dim=1)
        transfer_region = (add + 0.60 * keep + 0.90 * ref_overlap + 0.50 * boundary).clamp(0, 1)
        cleanup_region = (remove + 0.08 * boundary * (1 - add) * (1 - ref_overlap)).clamp(0, 1)

        transfer_gate, transfer_residual = self.transfer_head(
            torch.cat([P_eq8, P_src, P_shape, P_ref, mask_feat, shadow], dim=1)
        )
        cleanup_gate, cleanup_residual = self.cleanup_head(
            torch.cat([P_eq8, P_src, P_src_inpaint, mask_feat, shadow], dim=1)
        )

        texture_target = 0.70 * ref_shape_fused + 0.30 * F_ref
        transfer_prior = transfer_region * (texture_target - F_eq8)
        cleanup_prior = cleanup_region * (0.25 + 0.75 * shadow) * (F_src_inpaint - F_eq8)

        transfer_delta = transfer_prior + 0.22 * transfer_gate * transfer_region * transfer_residual
        cleanup_delta = cleanup_prior + 0.14 * cleanup_gate * cleanup_region * cleanup_residual
        output = F_eq8 + transfer_delta + cleanup_delta

        return output, {
            "weights": weights,
            "shadow": shadow,
            "masks_32": masks_32,
            "fused": fused,
            "transfer_region": transfer_region,
            "background_region": cleanup_region,
            "cleanup_region": cleanup_region,
            "transfer_gate": transfer_gate,
            "cleanup_gate": cleanup_gate,
            "transfer_prior": transfer_prior,
            "cleanup_prior": cleanup_prior,
            "transfer_delta": transfer_delta,
            "cleanup_delta": cleanup_delta,
        }
