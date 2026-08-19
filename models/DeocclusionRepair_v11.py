from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from utils.deocclusion_masks_v11 import (
    DGRR_INPUT_MASK_KEYS_V11,
    resize_mask,
    stack_deocclusion_masks_v11,
)


def _ensure_image_4d(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    squeezed = image.dim() == 3
    if squeezed:
        image = image.unsqueeze(0)
    return image.float(), squeezed


class ConvGNAct_v11(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
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


class UpBlock_v11(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvGNAct_v11(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class RegionDecoder_v11(nn.Module):
    def __init__(self, ch4: int, ch3: int, ch2: int, ch1: int):
        super().__init__()
        self.up3 = UpBlock_v11(ch4, ch3, ch3)
        self.up2 = UpBlock_v11(ch3, ch2, ch2)
        self.up1 = UpBlock_v11(ch2, ch1, ch1)
        self.head = nn.Conv2d(ch1, 4, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[3] = -1.0

    def forward(self, bottleneck: torch.Tensor, e3: torch.Tensor, e2: torch.Tensor, e1: torch.Tensor) -> torch.Tensor:
        d3 = self.up3(bottleneck, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


class DeocclusionRepair_v11(nn.Module):
    """
    DGRR-HairFast repair module.

    HairFast produces the hairstyle transfer result. This module only repairs
    reveal regions after old source hair has been removed. It uses three
    gated heads:
    - skin: forehead, cheek/ear boundary, and neck skin
    - struct: shoulder, collar, neck/body boundary
    - bg: revealed background behind removed long hair
    - fill: deep removed-region holes and non-background completion
    - clean: target-hair boundary residue cleanup
    """

    def __init__(
        self,
        *,
        mask_keys: tuple[str, ...] = DGRR_INPUT_MASK_KEYS_V11,
        base_channels: int = 48,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        self.mask_keys = mask_keys
        self.delta_scale = float(delta_scale)

        in_ch = 6 + len(mask_keys)
        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 8

        self.enc1 = ConvGNAct_v11(in_ch, ch1)
        self.enc2 = ConvGNAct_v11(ch1, ch2)
        self.enc3 = ConvGNAct_v11(ch2, ch3)
        self.bottleneck = ConvGNAct_v11(ch3, ch4)

        self.skin_decoder = RegionDecoder_v11(ch4, ch3, ch2, ch1)
        self.struct_decoder = RegionDecoder_v11(ch4, ch3, ch2, ch1)
        self.bg_decoder = RegionDecoder_v11(ch4 * 2, ch3, ch2, ch1)
        self.fill_decoder = RegionDecoder_v11(ch4 * 2, ch3, ch2, ch1)
        self.clean_decoder = RegionDecoder_v11(ch4, ch3, ch2, ch1)
        self.fill_ctx_proj = nn.Sequential(
            nn.Conv2d(ch4, ch4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch4, ch4, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.bg_ctx_proj = nn.Sequential(
            nn.Conv2d(ch4, ch4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch4, ch4, kernel_size=1),
            nn.SiLU(inplace=True),
        )

    def _build_input(
        self,
        source_image: torch.Tensor,
        base_image: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        source_image, _ = _ensure_image_4d(source_image)
        base_image, _ = _ensure_image_4d(base_image)
        size = base_image.shape[-2:]
        if source_image.shape[-2:] != size:
            source_image = F.interpolate(source_image, size=size, mode="bilinear", align_corners=False)
        mask_stack = stack_deocclusion_masks_v11(masks, keys=self.mask_keys, size=size)
        return torch.cat([source_image, base_image, mask_stack], dim=1)

    @staticmethod
    def _masked_context(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = resize_mask(mask, feat.shape[-2:], mode="bilinear")
        raw_sum = mask.sum(dim=(2, 3), keepdim=True)
        denom = raw_sum.clamp(min=1.0)
        pooled = (feat * mask).sum(dim=(2, 3), keepdim=True) / denom
        fallback = feat.mean(dim=(2, 3), keepdim=True)
        use_fallback = (raw_sum <= 1e-3).float()
        return pooled * (1.0 - use_fallback) + fallback * use_fallback

    @staticmethod
    def _masked_color(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = resize_mask(mask, image.shape[-2:], mode="bilinear")
        raw_sum = mask.sum(dim=(2, 3), keepdim=True)
        denom = raw_sum.clamp(min=1.0)
        pooled = (image * mask).sum(dim=(2, 3), keepdim=True) / denom
        fallback = image.mean(dim=(2, 3), keepdim=True)
        use_fallback = (raw_sum <= 1e-3).float()
        return pooled * (1.0 - use_fallback) + fallback * use_fallback

    def _compose_region(
        self,
        base_image: torch.Tensor,
        raw: torch.Tensor,
        gate_mask: torch.Tensor,
        alpha_scale: float,
        delta_limit: float,
        anchor_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if anchor_image is None:
            anchor_image = base_image
        gate_mask = resize_mask(gate_mask, base_image.shape[-2:], mode="bilinear")
        delta = self.delta_scale * float(delta_limit) * torch.tanh(raw[:, :3])
        alpha = (torch.sigmoid(raw[:, 3:4]) * gate_mask * float(alpha_scale)).clamp(0, 1)
        image = (anchor_image + delta).clamp(0, 1)
        return image, alpha, delta

    def forward(
        self,
        source_image: torch.Tensor,
        base_image: torch.Tensor,
        masks: dict[str, torch.Tensor],
        source_inpaint: torch.Tensor | None = None,
        *,
        alpha_scale: float = 1.0,
        enable_skin: bool = True,
        enable_struct: bool = True,
        enable_bg: bool = True,
        enable_fill: bool = True,
        enable_clean: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del source_inpaint
        base_image, squeezed = _ensure_image_4d(base_image)
        x = self._build_input(source_image, base_image, masks)

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.avg_pool2d(e2, kernel_size=2))
        b = self.bottleneck(F.avg_pool2d(e3, kernel_size=2))

        raw_skin = self.skin_decoder(b, e3, e2, e1)
        raw_struct = self.struct_decoder(b, e3, e2, e1)
        bg_ctx = self.bg_ctx_proj(self._masked_context(b, masks.get("R_bg_ctx", masks.get("M_bg_soft", masks.get("M_bg")))))
        raw_bg = self.bg_decoder(torch.cat([b, bg_ctx.expand_as(b)], dim=1), e3, e2, e1)
        fill_ctx = self.fill_ctx_proj(self._masked_context(b, masks.get("R_ctx", masks.get("M_fill_soft", masks.get("M_fill")))))
        raw_fill = self.fill_decoder(torch.cat([b, fill_ctx.expand_as(b)], dim=1), e3, e2, e1)
        raw_clean = self.clean_decoder(b, e3, e2, e1)
        bg_context = masks.get("R_bg_ctx", masks.get("M_bg_soft", masks.get("M_bg")))
        bg_anchor = self._masked_color(base_image, bg_context).expand_as(base_image)
        fill_context = masks.get("R_ctx", masks.get("M_bg_soft", masks.get("M_fill_soft", masks.get("M_fill"))))
        fill_anchor = self._masked_color(base_image, fill_context).expand_as(base_image)

        skin_rgb, alpha_skin, delta_skin = self._compose_region(
            base_image, raw_skin, masks.get("M_skin_soft", masks.get("M_skin")), alpha_scale, delta_limit=0.30
        )
        struct_rgb, alpha_struct, delta_struct = self._compose_region(
            base_image, raw_struct, masks.get("M_struct_soft", masks.get("M_struct")), alpha_scale, delta_limit=0.24
        )
        bg_rgb, alpha_bg, delta_bg = self._compose_region(
            base_image,
            raw_bg,
            masks.get("M_bg_soft", masks.get("M_bg")),
            alpha_scale,
            delta_limit=0.85,
            anchor_image=bg_anchor,
        )
        fill_rgb, alpha_fill, delta_fill = self._compose_region(
            base_image,
            raw_fill,
            masks.get("M_fill_soft", masks.get("M_fill")),
            alpha_scale,
            delta_limit=0.45,
            anchor_image=fill_anchor,
        )
        clean_rgb, alpha_clean, delta_clean = self._compose_region(
            base_image, raw_clean, masks.get("M_clean_soft", masks.get("M_clean")), alpha_scale, delta_limit=0.05
        )

        if not enable_skin:
            alpha_skin = torch.zeros_like(alpha_skin)
        if not enable_struct:
            alpha_struct = torch.zeros_like(alpha_struct)
        if not enable_bg:
            alpha_bg = torch.zeros_like(alpha_bg)
        if not enable_fill:
            alpha_fill = torch.zeros_like(alpha_fill)
        if not enable_clean:
            alpha_clean = torch.zeros_like(alpha_clean)

        face_preserve = masks.get("M_face_preserve")
        if face_preserve is not None:
            protect = resize_mask(face_preserve, base_image.shape[-2:], mode="bilinear").clamp(0, 1)
            allow = (1.0 - protect).clamp(0, 1)
            alpha_skin = alpha_skin * allow
            alpha_struct = alpha_struct * allow
            alpha_bg = alpha_bg * allow
            alpha_fill = alpha_fill * allow
            alpha_clean = alpha_clean * allow

        a_skin = alpha_skin
        a_struct = alpha_struct * (1.0 - a_skin)
        a_bg = alpha_bg * (1.0 - torch.maximum(a_skin, a_struct))
        a_fill = alpha_fill * (1.0 - torch.maximum(torch.maximum(a_skin, a_struct), a_bg))
        a_clean = 0.6 * alpha_clean * (1.0 - torch.maximum(torch.maximum(torch.maximum(a_skin, a_struct), a_bg), a_fill))

        out_skin = (base_image * (1.0 - a_skin) + skin_rgb * a_skin).clamp(0, 1)
        out_struct = (out_skin * (1.0 - a_struct) + struct_rgb * a_struct).clamp(0, 1)
        out_bg = (out_struct * (1.0 - a_bg) + bg_rgb * a_bg).clamp(0, 1)
        out_fill = (out_bg * (1.0 - a_fill) + fill_rgb * a_fill).clamp(0, 1)
        out = (out_fill * (1.0 - a_clean) + clean_rgb * a_clean).clamp(0, 1)
        alpha = (a_skin + a_struct + a_bg + a_fill + a_clean).clamp(0, 1)

        aux = {
            "alpha": alpha,
            "alpha_skin": a_skin,
            "alpha_struct": a_struct,
            "alpha_bg": a_bg,
            "alpha_fill": a_fill,
            "alpha_clean": a_clean,
            "skin_rgb": skin_rgb,
            "struct_rgb": struct_rgb,
            "bg_rgb": bg_rgb,
            "bg_anchor": bg_anchor,
            "fill_rgb": fill_rgb,
            "fill_anchor": fill_anchor,
            "clean_rgb": clean_rgb,
            "delta_skin": delta_skin,
            "delta_struct": delta_struct,
            "delta_bg": delta_bg,
            "delta_fill": delta_fill,
            "delta_clean": delta_clean,
            "raw_skin": raw_skin,
            "raw_struct": raw_struct,
            "raw_bg": raw_bg,
            "raw_fill": raw_fill,
            "raw_clean": raw_clean,
        }
        if squeezed:
            return out[0], {key: value[0] for key, value in aux.items()}
        return out, aux


def load_deocclusion_repair_v11(
    model: DeocclusionRepair_v11,
    checkpoint_path: str | dict,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    checkpoint = checkpoint_path if isinstance(checkpoint_path, dict) else torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("repair_v11_state_dict", checkpoint.get("model_state_dict", checkpoint))
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = {
        key
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape != value.shape
    }
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)
    return sorted(compatible), sorted(skipped)


__all__ = [
    "DeocclusionRepair_v11",
    "load_deocclusion_repair_v11",
]
