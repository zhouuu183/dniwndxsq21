"""Phase-A metrics and decision gates for Blending V8.29."""

from __future__ import annotations

import math
from statistics import median

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829, normalized_blur_v829


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    return (value * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1e-6)


def highpass(image: torch.Tensor, radius: int = 3) -> torch.Tensor:
    return image - gaussian_blur_v829(image, radius)


def masked_std(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(value, mask)
    centered = value - mean[:, None, None, None]
    return masked_mean(centered.square(), mask).clamp_min(0).sqrt()


def masked_quantile(value: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    samples = []
    for index in range(value.size(0)):
        selected = value[index][mask[index].expand_as(value[index]) > 0.5]
        samples.append(
            torch.quantile(selected.float(), quantile).to(value.dtype)
            if selected.numel() else value.new_zeros(())
        )
    return torch.stack(samples)


def build_near_core_band(core: torch.Tensor, inner_edge: torch.Tensor, width: int = 3) -> torch.Tensor:
    near_edge = F.max_pool2d(inner_edge, 2 * width + 1, stride=1, padding=width)
    return core * near_edge


def paired_edge_nearcore_metrics(
    image_rgb: torch.Tensor,
    core: torch.Tensor,
    inner_edge: torch.Tensor,
    radius: int = 5,
) -> dict[str, torch.Tensor]:
    lab = rgb_to_lab(image_rgb)
    local_core_lab, support = normalized_blur_v829(lab, core, radius)
    valid_edge = inner_edge * (support >= 1e-3).to(inner_edge.dtype)
    delta = lab - local_core_lab
    return {
        "edge_to_nearcore_delta_e": masked_mean(torch.linalg.vector_norm(delta, dim=1, keepdim=True), valid_edge),
        "edge_to_nearcore_delta_l": masked_mean(delta[:, :1].abs(), valid_edge),
        "edge_to_nearcore_delta_ab": masked_mean(torch.linalg.vector_norm(delta[:, 1:], dim=1, keepdim=True), valid_edge),
        "nearcore_valid_fraction": masked_mean((support >= 1e-3).to(inner_edge.dtype), inner_edge),
    }


def base_color_rim_metrics(
    base_rgb: torch.Tensor,
    output_rgb: torch.Tensor,
    core: torch.Tensor,
    inner_edge: torch.Tensor,
    radius: int = 5,
) -> dict[str, torch.Tensor]:
    base_lab = rgb_to_lab(base_rgb)
    output_lab = rgb_to_lab(output_rgb)
    local_target, support = normalized_blur_v829(output_lab, core, radius)
    direction = local_target - base_lab
    progress = ((output_lab - base_lab) * direction).sum(dim=1, keepdim=True) / direction.square().sum(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    meaningful = (torch.linalg.vector_norm(direction, dim=1, keepdim=True) >= 3.0).to(inner_edge.dtype)
    valid = inner_edge * meaningful * (support >= 1e-3).to(inner_edge.dtype)
    rim = (progress < 0.35).to(inner_edge.dtype)
    return {
        "base_color_rim_fraction": masked_mean(rim, valid),
        "base_color_edge_progress": masked_mean(progress.clamp(-1, 2), valid),
        "base_rim_valid_fraction": masked_mean(valid, inner_edge),
    }


def v229_metric_tensors(
    *,
    base_rgb: torch.Tensor,
    anchor_rgb: torch.Tensor,
    v226_rgb: torch.Tensor,
    v228_rgb: torch.Tensor,
    output_rgb: torch.Tensor,
    target_lab: torch.Tensor,
    core: torch.Tensor,
    inner_edge: torch.Tensor,
    ownership: torch.Tensor,
    nonhair: torch.Tensor,
    source_skin: torch.Tensor,
    face_context: torch.Tensor,
    background_context: torch.Tensor,
) -> dict[str, torch.Tensor]:
    output_lab = rgb_to_lab(output_rgb)
    v226_lab = rgb_to_lab(v226_rgb)
    base_lab = rgb_to_lab(base_rgb)
    anchor_lab = rgb_to_lab(anchor_rgb)
    output_gradient = _gradient(output_rgb)
    anchor_gradient = _gradient(anchor_rgb)
    v228_gradient = _gradient(v228_rgb)
    result = paired_edge_nearcore_metrics(output_rgb, core, inner_edge)
    v228_local = paired_edge_nearcore_metrics(v228_rgb, core, inner_edge)
    result.update({f"v228_{key}": value for key, value in v228_local.items()})
    result.update(base_color_rim_metrics(base_rgb, output_rgb, core, inner_edge))
    v228_rim = base_color_rim_metrics(base_rgb, v228_rgb, core, inner_edge)
    result.update({f"v228_{key}": value for key, value in v228_rim.items()})
    result.update({
        "core_ref_full": masked_mean(torch.linalg.vector_norm(output_lab - target_lab, dim=1, keepdim=True), core),
        "core_ref_ab": masked_mean(torch.linalg.vector_norm(output_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True), core),
        "core_ref_l": masked_mean((output_lab[:, :1] - target_lab[:, :1]).abs(), core),
        "v226_core_ref_full": masked_mean(torch.linalg.vector_norm(v226_lab - target_lab, dim=1, keepdim=True), core),
        "v226_core_ref_ab": masked_mean(torch.linalg.vector_norm(v226_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True), core),
        "v226_core_ref_l": masked_mean((v226_lab[:, :1] - target_lab[:, :1]).abs(), core),
        "base_core_ref_full": masked_mean(torch.linalg.vector_norm(base_lab - target_lab, dim=1, keepdim=True), core),
        "anchor_core_ref_full": masked_mean(torch.linalg.vector_norm(anchor_lab - target_lab, dim=1, keepdim=True), core),
        "target_core_chroma": masked_mean(torch.linalg.vector_norm(target_lab[:, 1:], dim=1, keepdim=True), core),
        "target_core_warm_score": masked_mean((target_lab[:, 1:2] + target_lab[:, 2:3]).clamp_min(0), core),
        "anchor_hf_l1": masked_mean((highpass(output_rgb) - highpass(anchor_rgb)).abs(), core),
        "v226_anchor_hf_l1": masked_mean((highpass(v226_rgb) - highpass(anchor_rgb)).abs(), core),
        "gradient_l1_to_anchor": masked_mean((output_gradient - anchor_gradient).abs(), core),
        "edge_gradient_l1_to_anchor": masked_mean((output_gradient - anchor_gradient).abs(), inner_edge),
        "v228_edge_gradient_l1_to_anchor": masked_mean((v228_gradient - anchor_gradient).abs(), inner_edge),
        "edge_texture_l1_to_anchor": masked_mean((highpass(output_rgb) - highpass(anchor_rgb)).abs(), inner_edge),
        "v228_edge_texture_l1_to_anchor": masked_mean((highpass(v228_rgb) - highpass(anchor_rgb)).abs(), inner_edge),
        "core_l_std": masked_std(output_lab[:, :1], core),
        "anchor_core_l_std": masked_std(anchor_lab[:, :1], core),
        "v226_core_l_std": masked_std(v226_lab[:, :1], core),
        "core_l_highlight_p90": masked_quantile(output_lab[:, :1], core, 0.90),
        "anchor_core_l_highlight_p90": masked_quantile(anchor_lab[:, :1], core, 0.90),
        "v226_core_l_highlight_p90": masked_quantile(v226_lab[:, :1], core, 0.90),
        "outside_contamination_rgb": masked_mean((output_rgb - base_rgb).abs(), nonhair),
        "face_side_color_leak_rgb": masked_mean((output_rgb - base_rgb).abs(), source_skin * nonhair),
        "face_boundary_contamination": masked_mean((output_rgb - base_rgb).abs(), face_context),
        "background_boundary_contamination": masked_mean((output_rgb - base_rgb).abs(), background_context),
        "hair_owner_fraction": ownership.mean(dim=(1, 2, 3)),
    })
    return result


def _gradient(image: torch.Tensor) -> torch.Tensor:
    dx = F.pad(image[..., :, 1:] - image[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def pp_ownership_metrics(
    prepp_rgb: torch.Tensor,
    pp_rgb: torch.Tensor,
    final_rgb: torch.Tensor,
    ownership: torch.Tensor,
    inner_edge: torch.Tensor,
) -> dict[str, torch.Tensor]:
    owner_error = (final_rgb - prepp_rgb).abs() * ownership
    nonhair = 1.0 - ownership
    nonhair_error = (final_rgb - pp_rgb).abs() * nonhair
    pre_lab = rgb_to_lab(prepp_rgb)
    final_lab = rgb_to_lab(final_rgb)
    delta = final_lab - pre_lab
    return {
        "pp_owner_max_delta": owner_error.flatten(1).amax(dim=1),
        "nonhair_final_to_pp_max_delta": nonhair_error.flatten(1).amax(dim=1),
        "pp_owner_ab_shift": masked_mean(torch.linalg.vector_norm(delta[:, 1:], dim=1, keepdim=True), ownership),
        "pp_owner_l_shift": masked_mean(delta[:, :1].abs(), ownership),
        "pp_inner_edge_ab_shift": masked_mean(torch.linalg.vector_norm(delta[:, 1:], dim=1, keepdim=True), inner_edge),
        "pp_inner_edge_l_shift": masked_mean(delta[:, :1].abs(), inner_edge),
    }


def aggregate_v229(records: list[dict[str, object]]) -> dict[str, float | int]:
    if not records:
        return {"count": 0}
    keys = sorted({key for record in records for key, value in record.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    summary: dict[str, float | int] = {"count": len(records)}
    for key in keys:
        values = [float(record[key]) for record in records if key in record and math.isfinite(float(record[key]))]
        if not values:
            continue
        ordered = sorted(values)
        summary[f"median_{key}"] = float(median(values))
        summary[f"mean_{key}"] = float(sum(values) / len(values))
        summary[f"min_{key}"] = float(min(values))
        summary[f"max_{key}"] = float(max(values))
        summary[f"p90_{key}"] = float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))])
    return summary


def classify_v229(
    summary: dict[str, float | int],
    *,
    parity_max_diff: float,
    phase_b: bool,
    phase_a_summary: dict[str, float | int] | None = None,
) -> str:
    if parity_max_diff > 1e-5:
        return "V229_RUNTIME_PARITY_FAIL"
    if float(summary.get("median_core_ref_full", float("inf"))) > 1.10 * float(summary.get("median_v226_core_ref_full", 0.0)):
        return "V229_HYBRID_CORE_COLOR_FAIL"
    if float(summary.get("median_anchor_hf_l1", float("inf"))) > 0.35 * float(summary.get("median_v226_anchor_hf_l1", 0.0)):
        return "V229_HYBRID_APPEARANCE_FAIL"
    if float(summary.get("max_outside_max_delta", 1.0)) > 5e-4 or float(summary.get("median_background_contamination_retention", 1.0)) > 0.05:
        return "V229_BACKGROUND_RECOMPOSITION_FAIL"
    if float(summary.get("max_face_side_outer_hair_fraction", 1.0)) > 1e-6 or float(summary.get("median_face_side_color_leak_rgb", 1.0)) > 5e-4:
        return "V229_FACE_OWNERSHIP_FAIL"
    rim = float(summary.get("median_base_color_rim_fraction", float("inf")))
    v228_rim = float(summary.get("median_v228_base_color_rim_fraction", 0.0))
    edge_de = float(summary.get("median_edge_to_nearcore_delta_e", float("inf")))
    v228_edge_de = float(summary.get("median_v228_edge_to_nearcore_delta_e", 0.0))
    if rim > 0.60 * v228_rim or edge_de > 0.80 * v228_edge_de:
        return "V229_BASE_RIM_FAIL"
    if max(float(summary.get("max_pp_owner_max_delta", 1.0)), float(summary.get("max_nonhair_final_to_pp_max_delta", 1.0))) > 1e-5:
        return "V229_PP_OWNERSHIP_LOCK_FAIL"
    if float(summary.get("mean_clip_magnitude", 1.0)) > 0.01:
        return "V229_GAMUT_FAIL"
    if phase_b:
        if phase_a_summary is None:
            return "V229_FACE_OWNERSHIP_FAIL"
        if (
            float(summary.get("max_outer_strand_accepted_fraction", 0.0)) > 0.0
            and float(summary.get("min_outer_strand_min_residual_alignment", 0.0)) < 0.80
        ):
            return "V229_FACE_OWNERSHIP_FAIL"
        if float(summary.get("median_background_contamination_retention", 1.0)) > float(
            phase_a_summary.get("median_background_contamination_retention", 0.0)
        ) + 1e-6:
            return "V229_BACKGROUND_RECOMPOSITION_FAIL"
        if float(summary.get("median_face_side_color_leak_rgb", 1.0)) > float(
            phase_a_summary.get("median_face_side_color_leak_rgb", 0.0)
        ) + 1e-6:
            return "V229_FACE_OWNERSHIP_FAIL"
        if float(summary.get("mean_outer_strand_small_component_fraction", 1.0)) > float(
            phase_a_summary.get("mean_outer_strand_small_component_fraction", 0.0)
        ) + 1e-6:
            return "V229_FACE_OWNERSHIP_FAIL"
    return "V229_PHASE_B_READY_FOR_VISUAL_REVIEW" if phase_b else "V229_PHASE_A_READY_FOR_VISUAL_REVIEW"


__all__ = [
    "aggregate_v229",
    "base_color_rim_metrics",
    "build_near_core_band",
    "classify_v229",
    "paired_edge_nearcore_metrics",
    "pp_ownership_metrics",
    "v229_metric_tensors",
]
