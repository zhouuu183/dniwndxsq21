"""Shared runtime tensor contract for V2.29 diagnostic and real inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.v828_runtime_inputs import build_v828_runtime_inputs


def _mask(value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(reference)
    value = value.float()
    if value.dim() == 3:
        value = value.unsqueeze(1)
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value, size=reference.shape[-2:], mode="nearest")
    if value.size(0) == 1 and reference.size(0) != 1:
        value = value.expand(reference.size(0), -1, -1, -1)
    return value.to(device=reference.device, dtype=reference.dtype).clamp(0, 1)


def build_v829_runtime_inputs(
    *,
    base_rgb: torch.Tensor,
    anchor_rgb: torch.Tensor,
    v226_rgb: torch.Tensor,
    target_hair_mask: torch.Tensor,
    target_hair_eroded: torch.Tensor | None = None,
    target_hair_dilated: torch.Tensor | None = None,
    source_subject_mask: torch.Tensor | None = None,
    source_skin_mask: torch.Tensor | None = None,
    hard_protect_mask: torch.Tensor | None = None,
    matte_width: int = 4,
) -> dict[str, torch.Tensor]:
    shared = build_v828_runtime_inputs(
        base_rgb=base_rgb,
        anchor_rgb=anchor_rgb,
        target_hair_mask=target_hair_mask,
        target_hair_eroded=target_hair_eroded,
        target_hair_dilated=target_hair_dilated,
        hard_protect_mask=hard_protect_mask,
        matte_width=matte_width,
    )
    v226 = v226_rgb.to(device=shared["base_rgb"].device, dtype=shared["base_rgb"].dtype).clamp(0, 1)
    if v226.shape != shared["base_rgb"].shape:
        raise ValueError("V2.29 v226_rgb must match base_rgb")
    reference = shared["target_hair_mask"]
    subject = _mask(source_subject_mask, reference)
    skin = _mask(source_skin_mask, reference)
    return {
        "base_rgb": shared["base_rgb"],
        "anchor_rgb": shared["anchor_rgb"],
        "v226_rgb": v226,
        "target_hair_mask": reference,
        "target_hair_eroded": shared["target_hair_eroded"],
        "target_hair_dilated": shared["target_hair_dilated"],
        "source_subject_mask": subject,
        "source_skin_mask": skin,
        "hard_protect_mask": shared["hard_protect_mask"],
    }


__all__ = ["build_v829_runtime_inputs"]
