import torch
import torch.nn.functional as F
from torch import nn


class ConvGNAct(nn.Module):
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


class MaskEncoder_v3(nn.Module):
    def __init__(self, in_ch: int = 5, mid_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, 32),
            ConvGNAct(32, 32),
            ConvGNAct(32, mid_ch),
            ConvGNAct(mid_ch, mid_ch),
        )

    def forward(self, masks_32: torch.Tensor) -> torch.Tensor:
        return self.net(masks_32)


class FeatureProjector_v3(nn.Module):
    def __init__(self, in_ch: int = 512, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShadowEstimator_v3(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, 128),
            ConvGNAct(128, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class BlendNet_v3(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, 256),
            ConvGNAct(256, 128),
            nn.Conv2d(128, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextureGate_v3(nn.Module):
    def __init__(self, in_ch: int, feat_ch: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(in_ch, 256),
            ConvGNAct(256, 128),
        )
        self.gate = nn.Conv2d(128, 1, kernel_size=1)
        self.residual = nn.Conv2d(128, feat_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.body(x)
        gate = torch.sigmoid(self.gate(feat))
        residual = self.residual(feat)
        return gate, residual


class SATD_v3(nn.Module):
    """
    Channel order for masks_256 / masks_32:
    [M_add, M_remove, M_keep, M_boundary, M_ref_overlap]
    """

    def __init__(self, feat_ch: int = 512, proj_ch: int = 128, mask_ch: int = 5):
        super().__init__()
        self.mask_encoder = MaskEncoder_v3(mask_ch, 64)
        self.projector = FeatureProjector_v3(feat_ch, proj_ch)
        self.shadow_estimator = ShadowEstimator_v3(proj_ch * 2 + 64 + 3)
        self.blend_net = BlendNet_v3(proj_ch * 4 + 64 + 1)
        self.texture_gate = TextureGate_v3(proj_ch * 2 + 64 + 1, feat_ch=feat_ch)

    @staticmethod
    def _down_rgb(image_256: torch.Tensor) -> torch.Tensor:
        return F.interpolate(image_256, size=(32, 32), mode="bicubic", align_corners=False)

    @staticmethod
    def _prior_logits(masks_32: torch.Tensor, shadow: torch.Tensor) -> torch.Tensor:
        add, remove, keep, boundary, ref_overlap = torch.chunk(masks_32, 5, dim=1)
        src_prior = keep + 0.35 * boundary * (1 - shadow)
        src_inpaint_prior = remove + 0.65 * boundary * shadow
        shape_prior = add + 0.40 * boundary
        ref_prior = ref_overlap + 0.55 * add
        return 2.0 * torch.cat([src_prior, src_inpaint_prior, shape_prior, ref_prior], dim=1)

    def forward(
        self,
        F_src: torch.Tensor,
        F_src_inpaint: torch.Tensor,
        F_shape_inpaint: torch.Tensor,
        F_ref: torch.Tensor,
        masks_256: torch.Tensor,
        source_rgb_256: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks_32 = F.interpolate(masks_256.float(), size=(32, 32), mode="nearest")
        mask_feat = self.mask_encoder(masks_32)

        P_src = self.projector(F_src)
        P_src_inpaint = self.projector(F_src_inpaint)
        P_shape = self.projector(F_shape_inpaint)
        P_ref = self.projector(F_ref)

        src_rgb_32 = self._down_rgb(source_rgb_256)
        shadow = self.shadow_estimator(torch.cat([P_src, P_src_inpaint, mask_feat, src_rgb_32], dim=1))

        blend_logits = self.blend_net(torch.cat([P_src, P_src_inpaint, P_shape, P_ref, mask_feat, shadow], dim=1))
        blend_logits = blend_logits + self._prior_logits(masks_32, shadow)
        weights = torch.softmax(blend_logits, dim=1)

        fused = (
            weights[:, 0:1] * F_src
            + weights[:, 1:2] * F_src_inpaint
            + weights[:, 2:3] * F_shape_inpaint
            + weights[:, 3:4] * F_ref
        )

        add, remove, _, boundary, _ = torch.chunk(masks_32, 5, dim=1)
        gate, texture_residual = self.texture_gate(torch.cat([P_shape, P_ref, mask_feat, shadow], dim=1))
        texture_region = (add + 0.5 * boundary).clamp(0, 1)
        background_region = remove

        background_delta = background_region * shadow * (F_src_inpaint - F_src)
        output = fused + 0.10 * gate * texture_region * texture_residual + 0.20 * background_delta

        return output, {
            "weights": weights,
            "shadow": shadow,
            "masks_32": masks_32,
            "texture_region": texture_region,
            "background_region": background_region,
        }
