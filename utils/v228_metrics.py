"""Appearance, protection, and PP-lock metrics for Blending V8.28."""

from __future__ import annotations

import math
from statistics import median

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import gaussian_blur2d, rgb_to_lab


def masked_mean_per_sample(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    numerator = (value * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1)
    return numerator / denominator.clamp_min(1e-6)


def _masked_channel_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (value * mask).flatten(2).sum(dim=2)
    denominator = mask.flatten(2).sum(dim=2)
    return numerator / denominator.clamp_min(1e-6)


def highpass(image: torch.Tensor, radius: int = 3) -> torch.Tensor:
    return image - gaussian_blur2d(image, radius)


def _masked_cosine(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.expand_as(a)
    av = (a * m).flatten(1)
    bv = (b * m).flatten(1)
    return F.cosine_similarity(av, bv, dim=1, eps=1e-8)


def _gradient(image: torch.Tensor) -> torch.Tensor:
    dx = F.pad(image[..., :, 1:] - image[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def appearance_metric_tensors(
    *,
    base_rgb: torch.Tensor,
    anchor_rgb: torch.Tensor,
    v226_rgb: torch.Tensor,
    output_rgb: torch.Tensor,
    core: torch.Tensor,
    edge: torch.Tensor,
    outer_ring: torch.Tensor,
    face_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    anchor_hf = highpass(anchor_rgb)
    output_hf = highpass(output_rgb)
    v226_hf = highpass(v226_rgb)
    anchor_lab = rgb_to_lab(anchor_rgb)
    output_lab = rgb_to_lab(output_rgb)
    v226_lab = rgb_to_lab(v226_rgb)
    base_lab = rgb_to_lab(base_rgb)

    anchor_core_mean = _masked_channel_mean(anchor_lab, core).unsqueeze(-1).unsqueeze(-1)
    output_edge_delta = output_lab - anchor_core_mean
    v226_edge_delta = v226_lab - anchor_core_mean
    output_edge_de = torch.linalg.vector_norm(output_edge_delta, dim=1, keepdim=True)
    v226_edge_de = torch.linalg.vector_norm(v226_edge_delta, dim=1, keepdim=True)
    output_edge_ab = torch.linalg.vector_norm(output_edge_delta[:, 1:], dim=1, keepdim=True)
    v226_edge_ab = torch.linalg.vector_norm(v226_edge_delta[:, 1:], dim=1, keepdim=True)

    anchor_texture = _gradient(anchor_rgb).mean(dim=1, keepdim=True)
    output_texture = _gradient(output_rgb).mean(dim=1, keepdim=True)
    flattening = ((output_texture < 0.60 * anchor_texture) & (output_edge_de > 8.0)).float()
    anchor_outer_shift = (anchor_rgb - base_rgb).abs().mean(dim=1, keepdim=True)
    output_outer_shift = (output_rgb - base_rgb).abs().mean(dim=1, keepdim=True)
    bg_anchor = masked_mean_per_sample(anchor_outer_shift, outer_ring)
    bg_output = masked_mean_per_sample(output_outer_shift, outer_ring)

    return {
        "anchor_hf_l1": masked_mean_per_sample((output_hf - anchor_hf).abs(), core),
        "v226_anchor_hf_l1": masked_mean_per_sample((v226_hf - anchor_hf).abs(), core),
        "anchor_hf_cosine": _masked_cosine(output_hf, anchor_hf, core),
        "gradient_l1_to_anchor": masked_mean_per_sample((_gradient(output_rgb) - _gradient(anchor_rgb)).abs(), core),
        "edge_core_delta_e": masked_mean_per_sample(output_edge_de, edge),
        "edge_core_delta_l": masked_mean_per_sample(output_edge_delta[:, :1].abs(), edge),
        "edge_core_delta_ab": masked_mean_per_sample(output_edge_ab, edge),
        "v226_edge_core_delta_e": masked_mean_per_sample(v226_edge_de, edge),
        "v226_edge_core_delta_l": masked_mean_per_sample(v226_edge_delta[:, :1].abs(), edge),
        "v226_edge_core_delta_ab": masked_mean_per_sample(v226_edge_ab, edge),
        "appearance_flattening_fraction": masked_mean_per_sample(flattening, edge),
        "texture_energy_anchor_edge": masked_mean_per_sample(anchor_texture, edge),
        "texture_energy_output_edge": masked_mean_per_sample(output_texture, edge),
        "background_contamination_retention": bg_output / bg_anchor.clamp_min(1e-8),
        "face_contamination_rgb": masked_mean_per_sample((output_rgb - base_rgb).abs(), face_mask),
    }


def pp_lock_metric_tensors(
    prepp_rgb: torch.Tensor,
    final_rgb: torch.Tensor,
    core: torch.Tensor,
    edge: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pre_lab = rgb_to_lab(prepp_rgb)
    final_lab = rgb_to_lab(final_rgb)
    delta = final_lab - pre_lab
    full = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    ab = torch.linalg.vector_norm(delta[:, 1:], dim=1, keepdim=True)
    hf_delta = (highpass(final_rgb) - highpass(prepp_rgb)).abs()
    return {
        "pp_core_ab_shift": masked_mean_per_sample(ab, core),
        "pp_core_l_shift": masked_mean_per_sample(delta[:, :1].abs(), core),
        "pp_core_full_de": masked_mean_per_sample(full, core),
        "pp_core_hf_shift": masked_mean_per_sample(hf_delta, core),
        "pp_edge_ab_shift": masked_mean_per_sample(ab, edge),
        "pp_edge_l_shift": masked_mean_per_sample(delta[:, :1].abs(), edge),
        "pp_edge_full_de": masked_mean_per_sample(full, edge),
    }


def aggregate_v228_records(records: list[dict[str, float | str | bool]]) -> dict[str, float | int]:
    if not records:
        return {"count": 0}
    keys = sorted({key for record in records for key, value in record.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    result: dict[str, float | int] = {"count": len(records)}
    for key in keys:
        values = [float(record[key]) for record in records if key in record and math.isfinite(float(record[key]))]
        if values:
            result[f"median_{key}"] = float(median(values))
            result[f"mean_{key}"] = float(sum(values) / len(values))
            result[f"max_{key}"] = float(max(values))
            result[f"p90_{key}"] = float(sorted(values)[min(len(values) - 1, int(0.9 * len(values)))])
    return result


def classify_v228(summary: dict[str, float | int], *, parity_max_diff: float, anchor_reliable_fraction: float) -> str:
    if parity_max_diff > 1e-5:
        return "V228_RUNTIME_PARITY_FAIL"
    if anchor_reliable_fraction < 0.70:
        return "V228_ANCHOR_CARRIER_NOT_GENERAL"
    if max(float(summary.get("max_outside_max_delta", 0.0)), float(summary.get("max_hard_protect_max_delta", 0.0))) > 5e-4:
        return "V228_MATTE_COMPOSITE_FAIL"
    if float(summary.get("median_background_contamination_retention", 1.0)) > 0.05:
        return "V228_BACKGROUND_DECONTAMINATION_FAIL"
    edge_de = float(summary.get("median_edge_core_delta_e", float("inf")))
    baseline_de = float(summary.get("median_v226_edge_core_delta_e", 0.0))
    edge_l = float(summary.get("median_edge_core_delta_l", float("inf")))
    baseline_l = float(summary.get("median_v226_edge_core_delta_l", 0.0))
    hf = float(summary.get("median_anchor_hf_l1", float("inf")))
    baseline_hf = float(summary.get("median_v226_anchor_hf_l1", 0.0))
    if edge_de > 0.75 * baseline_de or edge_l > 0.80 * baseline_l or hf >= baseline_hf:
        return "V228_APPEARANCE_PRESERVATION_FAIL"
    if float(summary.get("median_anchor_ref_full", float("inf"))) > 1.10 * float(summary.get("median_base_ref_full", 0.0)):
        return "V228_COLOR_REGRESSION"
    if max(float(summary.get("median_pp_core_ab_shift", 0.0)), float(summary.get("median_pp_core_l_shift", 0.0))) > 0.5:
        return "V228_PP_HAIR_LOCK_FAIL"
    return "V228_READY_FOR_VISUAL_REVIEW"


__all__ = [
    "aggregate_v228_records",
    "appearance_metric_tensors",
    "classify_v228",
    "highpass",
    "masked_mean_per_sample",
    "pp_lock_metric_tensors",
]
