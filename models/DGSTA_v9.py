import torch
import torch.nn.functional as F
from torch import nn


class ConvGNAct_v9(nn.Module):
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


class ResidualBranch_v9(nn.Module):
    def __init__(
        self,
        in_ch: int,
        feat_ch: int = 512,
        gate_bias: float = -2.2,
        delta_init_std: float = 1e-3,
    ):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct_v9(in_ch, 256),
            ConvGNAct_v9(256, 128),
            ConvGNAct_v9(128, 128),
        )
        self.gate = nn.Conv2d(128, 1, kernel_size=1)
        self.delta = nn.Conv2d(128, feat_ch, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)
        nn.init.normal_(self.delta.weight, mean=0.0, std=delta_init_std)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.body(x)
        return torch.sigmoid(self.gate(feat)), self.delta(feat)


class DGSTA_v9(nn.Module):
    """
    Diff-Guided Semantic Topology Alignment, lite version.

    It keeps the HairFast Eq.(8) alignment as F_base and only learns three
    residual corrections:
    - clean: remove old source hair / shadow leakage.
    - topology: inject target hair shape/topology on add+keep regions.
    - bang: local residual for forehead/bangs detail.
    """

    def __init__(self, feat_ch: int = 512, proj_ch: int = 128, mask_ch: int = 16):
        super().__init__()
        self.mask_encoder = nn.Sequential(
            ConvGNAct_v9(mask_ch, 32),
            ConvGNAct_v9(32, 64),
            ConvGNAct_v9(64, 64),
        )
        self.rgb_encoder = nn.Sequential(
            ConvGNAct_v9(6, 32),
            ConvGNAct_v9(32, 32),
        )
        self.projector = nn.Sequential(
            nn.Conv2d(feat_ch, proj_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, proj_ch), proj_ch),
            nn.SiLU(inplace=True),
        )

        common_ch = proj_ch * 5 + 64 + 32
        self.clean_branch = ResidualBranch_v9(common_ch, feat_ch=feat_ch, gate_bias=-2.0, delta_init_std=2e-3)
        self.topology_branch = ResidualBranch_v9(common_ch, feat_ch=feat_ch, gate_bias=-2.3, delta_init_std=1.5e-3)
        self.bang_branch = ResidualBranch_v9(common_ch, feat_ch=feat_ch, gate_bias=-2.2, delta_init_std=1.5e-3)

        self.clean_beta = nn.Parameter(torch.tensor(0.85))
        self.topology_beta = nn.Parameter(torch.tensor(0.65))
        self.bang_beta = nn.Parameter(torch.tensor(0.70))

    @staticmethod
    def _down_rgb(image_256: torch.Tensor) -> torch.Tensor:
        return F.interpolate(image_256, size=(32, 32), mode="bicubic", align_corners=False)

    def forward(
        self,
        F_base: torch.Tensor,
        F_src: torch.Tensor,
        F_ref: torch.Tensor,
        F_src_inpaint: torch.Tensor,
        F_shape_inpaint: torch.Tensor,
        dgsta_masks_256: torch.Tensor,
        source_rgb_256: torch.Tensor,
        shape_rgb_256: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks_32 = F.interpolate(dgsta_masks_256.float(), size=(32, 32), mode="nearest")
        (
            remove,
            add,
            keep,
            boundary,
            remove_halo,
            remove_face,
            remove_neck,
            remove_tail,
            body_preserve,
            visible_body_anchor,
            body_reveal_only,
            context_reveal_only,
            reveal_overlap,
            detail_protect,
            bang,
            bundle,
        ) = torch.chunk(masks_32, 16, dim=1)

        clean_region = (
            remove
            + 0.90 * remove_halo
            + 0.95 * remove_face
            + 0.90 * remove_neck
            + 0.85 * remove_tail
            + 0.65 * body_reveal_only
            + 0.65 * context_reveal_only
            + 0.55 * reveal_overlap
        ).clamp(0, 1)
        topology_region = (add + 0.45 * keep + 0.40 * boundary + 0.75 * bundle).clamp(0, 1)
        bang_region = (bang * (add + keep + 0.35 * boundary).clamp(0, 1)).clamp(0, 1)

        protect_region = (
            0.78 * body_preserve
            + 0.70 * detail_protect
            + 0.70 * visible_body_anchor * (1.0 - add)
        ).clamp(0, 1)

        P_base = self.projector(F_base)
        P_src = self.projector(F_src)
        P_ref = self.projector(F_ref)
        P_src_inpaint = self.projector(F_src_inpaint)
        P_shape_inpaint = self.projector(F_shape_inpaint)
        mask_feat = self.mask_encoder(masks_32)
        rgb_feat = self.rgb_encoder(torch.cat([self._down_rgb(source_rgb_256), self._down_rgb(shape_rgb_256)], dim=1))
        feat = torch.cat([P_base, P_src, P_ref, P_src_inpaint, P_shape_inpaint, mask_feat, rgb_feat], dim=1)

        clean_gate, clean_delta = self.clean_branch(feat)
        topology_gate, topology_delta = self.topology_branch(feat)
        bang_gate, bang_delta = self.bang_branch(feat)

        clean_beta = self.clean_beta.clamp(0.0, 1.5)
        topology_beta = self.topology_beta.clamp(0.0, 1.5)
        bang_beta = self.bang_beta.clamp(0.0, 1.5)

        delta = clean_beta * clean_region * clean_gate * clean_delta
        delta = delta + topology_beta * topology_region * topology_gate * topology_delta
        delta = delta + bang_beta * bang_region * bang_gate * bang_delta
        delta = delta * (1.0 - 0.72 * protect_region)

        return F_base + delta, {
            "masks_32": masks_32,
            "clean_region": clean_region,
            "topology_region": topology_region,
            "bang_region": bang_region,
            "protect_region": protect_region,
            "clean_gate": clean_gate,
            "topology_gate": topology_gate,
            "bang_gate": bang_gate,
            "clean_delta": clean_delta,
            "topology_delta": topology_delta,
            "bang_delta": bang_delta,
            "delta": delta,
        }
