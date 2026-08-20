"""High-resolution trimap construction for V2.31 hair matting."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    return F.max_pool2d(mask, 2 * width + 1, stride=1, padding=width)


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - mask, width)


def _fill_small_enclosed_holes(mask: torch.Tensor, max_area: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill only enclosed non-hair components for trimap topology cleanup."""
    if max_area <= 0:
        return mask, torch.zeros_like(mask)
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("V2.31 trimap hole cleanup requires scipy") from exc
    filled = mask.detach().cpu().clone()
    accepted = torch.zeros_like(filled)
    for index in range(mask.size(0)):
        nonhair = (filled[index, 0].numpy() < 0.5)
        labels, count = ndimage.label(nonhair)
        for component in range(1, count + 1):
            region = labels == component
            area = int(region.sum())
            touches_border = bool(
                region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
            )
            if not touches_border and area <= max_area:
                region_tensor = torch.from_numpy(region)
                filled[index, 0][region_tensor] = 1
                accepted[index, 0][region_tensor] = 1
    return filled.to(mask), accepted.to(mask)


class TrimapBuilderV831(nn.Module):
    """Use the 256 mask only as a coarse prior, then refine at output size."""

    def __init__(
        self,
        inner_width: int = 8,
        outer_width: int = 8,
        face_contact_extra_inner: int = 4,
        max_trimap_hole_area: int = 16,
    ):
        super().__init__()
        self.inner_width = int(inner_width)
        self.outer_width = int(outer_width)
        self.face_contact_extra_inner = int(face_contact_extra_inner)
        self.max_trimap_hole_area = int(max_trimap_hole_area)
        if min(self.inner_width, self.outer_width) < 1:
            raise ValueError("V2.31 trimap widths must be positive")

    def config_dict(self) -> dict[str, int | str]:
        return {
            "construction_resolution": "OUTPUT_RESOLUTION",
            "inner_width": self.inner_width,
            "outer_width": self.outer_width,
            "face_contact_extra_inner": self.face_contact_extra_inner,
            "max_trimap_hole_area": self.max_trimap_hole_area,
            "coarse_mask_role": "TRIMAP_PRIOR_ONLY",
        }

    def forward(
        self,
        target_hair_mask: torch.Tensor,
        source_face_mask: torch.Tensor,
        *,
        output_size: tuple[int, int],
        return_aux: bool = False,
    ):
        coarse = F.interpolate(
            target_hair_mask.float(), size=output_size, mode="bilinear", align_corners=False
        ).clamp(0, 1)
        coarse_hard_raw = (coarse >= 0.5).to(coarse.dtype)
        coarse_hard, filled_holes = _fill_small_enclosed_holes(
            coarse_hard_raw, self.max_trimap_hole_area
        )
        face = F.interpolate(
            source_face_mask.float(), size=output_size, mode="nearest"
        ).clamp(0, 1)

        sure_fg_default = _erode(coarse_hard, self.inner_width)
        sure_bg = 1.0 - _dilate(coarse_hard, self.outer_width)
        coarse_contour = (coarse_hard - _erode(coarse_hard, 1)).clamp(0, 1)
        contact = _dilate(face, 1) * _dilate(coarse_contour, 1)
        contact_zone = _dilate(contact, self.face_contact_extra_inner)
        sure_fg = sure_fg_default * (1.0 - contact_zone)
        unknown = (1.0 - sure_fg - sure_bg).clamp(0, 1)

        trimap = torch.full_like(coarse, 0.5)
        trimap = torch.where(sure_bg > 0.5, torch.zeros_like(trimap), trimap)
        trimap = torch.where(sure_fg > 0.5, torch.ones_like(trimap), trimap)
        if not torch.isfinite(trimap).all():
            raise ValueError("V2.31 trimap produced NaN or Inf")
        if not return_aux:
            return trimap
        return trimap, {
            "coarse_hair": coarse,
            "coarse_hard": coarse_hard,
            "coarse_hard_raw": coarse_hard_raw,
            "filled_trimap_holes": filled_holes,
            "source_face_hr": face,
            "coarse_contour": coarse_contour,
            "face_contact_zone": contact_zone,
            "sure_fg": sure_fg,
            "sure_bg": sure_bg,
            "unknown": unknown,
            "trimap": trimap,
        }


__all__ = ["TrimapBuilderV831"]
