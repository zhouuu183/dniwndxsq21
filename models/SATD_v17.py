import torch
import torch.nn.functional as F
from torch import nn


class ConvGNAct_v17(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaskEncoder_v17(nn.Module):
    def __init__(self, in_ch: int = 7, mid_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct_v17(in_ch, 32),
            ConvGNAct_v17(32, 32),
            ConvGNAct_v17(32, mid_ch),
            ConvGNAct_v17(mid_ch, mid_ch),
        )

    def forward(self, masks_32: torch.Tensor) -> torch.Tensor:
        return self.net(masks_32)


class FeatureProjector_v17(nn.Module):
    def __init__(self, in_ch: int = 512, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GateResidualHead_v17(nn.Module):
    def __init__(self, in_ch: int, feat_ch: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct_v17(in_ch, 256),
            ConvGNAct_v17(256, 128),
        )
        self.gate = nn.Conv2d(128, 1, kernel_size=1)
        self.delta = nn.Conv2d(128, feat_ch, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -6.0)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.body(x)
        gate = torch.sigmoid(self.gate(feat))
        delta = self.delta(feat)
        return gate, delta


class GateHead_v17(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct_v17(in_ch, 128),
            ConvGNAct_v17(128, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class ShadowContextHead_v17(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct_v17(in_ch, 128),
            ConvGNAct_v17(128, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class SATD_v17(nn.Module):
    """
    SATD cleanup branch used by the v17 alignment stage.

    Input mask order:
    [M_boundary, M_remove, M_remove_halo, M_remove_tail,
     M_remove_face, M_remove_neck, M_body_preserve]

    Outputs:
    - shadow-aware cleanup residual
    - halo-specific residual
    - tail-specific residual
    - neck-specific residual
    - source-copy residual
    - protect gate
    """

    def __init__(self, feat_ch: int = 512, proj_ch: int = 128, mask_ch: int = 7):
        super().__init__()
        self.mask_encoder = MaskEncoder_v17(mask_ch, 64)
        self.projector = FeatureProjector_v17(feat_ch, proj_ch)
        self.shadow_context = ShadowContextHead_v17(proj_ch * 3 + 64 + 3)
        self.halo_head = GateResidualHead_v17(proj_ch * 2 + 64 + 1, feat_ch=feat_ch)
        self.tail_head = GateResidualHead_v17(proj_ch * 3 + 64 + 1, feat_ch=feat_ch)
        self.neck_head = GateResidualHead_v17(proj_ch * 2 + 64 + 1, feat_ch=feat_ch)
        self.copy_head = GateResidualHead_v17(proj_ch * 2 + 64 + 1, feat_ch=feat_ch)
        self.protect_head = GateHead_v17(proj_ch * 2 + 64)

    @staticmethod
    def _down_rgb(image_256: torch.Tensor) -> torch.Tensor:
        return F.interpolate(image_256, size=(32, 32), mode="bicubic", align_corners=False)

    @staticmethod
    def _detail_map(image_256: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * image_256[:, 0:1] + 0.587 * image_256[:, 1:2] + 0.114 * image_256[:, 2:3]
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image_256.device, dtype=image_256.dtype).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image_256.device, dtype=image_256.dtype).view(1, 1, 3, 3)
        grad_x = F.conv2d(gray, kernel_x, padding=1)
        grad_y = F.conv2d(gray, kernel_y, padding=1)
        detail = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        detail = F.interpolate(detail, size=(32, 32), mode="bicubic", align_corners=False)
        detail = detail / detail.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return detail.clamp(0, 1)

    def forward(
        self,
        F_base: torch.Tensor,
        F_src: torch.Tensor,
        F_src_inpaint: torch.Tensor,
        cleanup_masks_256: torch.Tensor,
        source_rgb_256: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks_32 = F.interpolate(cleanup_masks_256.float(), size=(32, 32), mode="nearest")
        (
            boundary,
            remove,
            remove_halo,
            remove_tail,
            remove_face,
            remove_neck,
            body_preserve,
        ) = torch.chunk(masks_32, 7, dim=1)

        mask_feat = self.mask_encoder(masks_32)
        P_base = self.projector(F_base)
        P_src = self.projector(F_src)
        P_src_inpaint = self.projector(F_src_inpaint)
        src_rgb_32 = self._down_rgb(source_rgb_256)
        detail_map = self._detail_map(source_rgb_256)

        shadow = self.shadow_context(
            torch.cat([P_base, P_src, P_src_inpaint, mask_feat, src_rgb_32], dim=1)
        )

        shadow_region = (
            0.22 * boundary
            + 0.16 * remove
            + 0.90 * remove_halo
            + 0.48 * remove_face
            + 0.46 * remove_neck
        ).clamp(0, 1)
        halo_region = (remove_halo + 0.18 * remove + 0.28 * remove_face + 0.18 * remove_neck).clamp(0, 1)
        tail_region = (remove_tail + 0.46 * remove_halo + 0.36 * remove_neck + 0.14 * remove).clamp(0, 1)
        neck_region = (remove_neck + 0.82 * remove_tail + 0.26 * remove_halo).clamp(0, 1)
        copy_region = (
            (0.18 * boundary + 0.20 * remove_halo + 0.18 * remove_tail + 0.10 * remove_neck)
            * (1.0 - 0.94 * remove_face)
            * (1.0 - 0.55 * detail_map)
        ).clamp(0, 1)
        protect_region = (
            body_preserve
            + 0.30 * remove_face * detail_map
            + 0.06 * remove_neck * detail_map
        ).clamp(0, 1)

        halo_gate, halo_residual = self.halo_head(
            torch.cat([P_base, P_src_inpaint, mask_feat, shadow], dim=1)
        )
        tail_gate, tail_residual = self.tail_head(
            torch.cat([P_base, P_src, P_src_inpaint, mask_feat, shadow], dim=1)
        )
        neck_gate, neck_residual = self.neck_head(
            torch.cat([P_base, P_src_inpaint, mask_feat, shadow], dim=1)
        )
        copy_gate, copy_residual = self.copy_head(
            torch.cat([P_base, P_src, mask_feat, shadow], dim=1)
        )
        protect_gate = self.protect_head(torch.cat([P_base, P_src, mask_feat], dim=1))

        shadow_prior = 0.22 * shadow_region * shadow * (F_src_inpaint - F_base)
        halo_prior = 0.14 * halo_region * (F_src_inpaint - F_base)
        tail_target = 0.70 * F_src_inpaint + 0.30 * F_src
        tail_prior = 0.30 * tail_region * (tail_target - F_base)
        neck_prior = 0.30 * neck_region * (F_src_inpaint - F_base)
        copy_prior = 0.04 * copy_region * (F_src - F_base)

        cleanup_delta = shadow_prior
        cleanup_delta = cleanup_delta + 0.11 * halo_gate * halo_region * (halo_prior + halo_residual)
        cleanup_delta = cleanup_delta + 0.19 * tail_gate * tail_region * (tail_prior + tail_residual)
        cleanup_delta = cleanup_delta + 0.20 * neck_gate * neck_region * (neck_prior + neck_residual)
        cleanup_delta = cleanup_delta + 0.03 * copy_gate * copy_region * (copy_prior + copy_residual)

        protect = (protect_gate * protect_region).clamp(0, 1)
        cleanup_delta = cleanup_delta * (1.0 - 0.85 * protect)
        output = F_base + cleanup_delta

        return output, {
            "masks_32": masks_32,
            "shadow": shadow,
            "shadow_region": shadow_region,
            "halo_region": halo_region,
            "tail_region": tail_region,
            "neck_region": neck_region,
            "copy_region": copy_region,
            "protect": protect,
            "detail_map": detail_map,
            "halo_gate": halo_gate,
            "tail_gate": tail_gate,
            "neck_gate": neck_gate,
            "copy_gate": copy_gate,
            "shadow_prior": shadow_prior,
            "halo_prior": halo_prior,
            "tail_prior": tail_prior,
            "neck_prior": neck_prior,
            "copy_prior": copy_prior,
            "cleanup_delta": cleanup_delta,
        }
