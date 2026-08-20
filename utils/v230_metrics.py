"""Metrics and ordered hard gates for deterministic Blending V2.30."""

from __future__ import annotations

import math
from statistics import median

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from utils.v229_metrics import base_color_rim_metrics, paired_edge_nearcore_metrics


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    return (value * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1e-6)


def luminance(image: torch.Tensor) -> torch.Tensor:
    return 0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]


def highpass(image: torch.Tensor, radius: int = 2) -> torch.Tensor:
    return image - gaussian_blur_v829(image, radius)


def gradient(image: torch.Tensor) -> torch.Tensor:
    value = luminance(image)
    dx = F.pad(value[..., :, 1:] - value[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def seam_hf_energy(
    final_rgb: torch.Tensor,
    pp_rgb: torch.Tensor,
    contour_band: torch.Tensor,
) -> torch.Tensor:
    residual = final_rgb - pp_rgb
    return masked_mean(highpass(residual, 2).abs(), contour_band)


def v230_metric_tensors(
    *,
    base_rgb: torch.Tensor,
    anchor_rgb: torch.Tensor,
    v226_rgb: torch.Tensor,
    v229_final_rgb: torch.Tensor,
    pp_rgb: torch.Tensor,
    final_rgb: torch.Tensor,
    target_lab: torch.Tensor,
    core: torch.Tensor,
    inner_edge: torch.Tensor,
    face_contact: torch.Tensor,
    background_contact: torch.Tensor,
    contour_band: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target_size = base_rgb.shape[-2:]
    def match(value: torch.Tensor) -> torch.Tensor:
        return value if value.shape[-2:] == target_size else F.interpolate(
            value, size=target_size, mode="bilinear", align_corners=False
        )
    v229_final_rgb = match(v229_final_rgb)
    pp_rgb = match(pp_rgb)
    final_rgb = match(final_rgb)
    contour_band = match(contour_band)
    face_contact = match(face_contact)
    background_contact = match(background_contact)
    final_lab = rgb_to_lab(final_rgb)
    base_lab = rgb_to_lab(base_rgb)
    v226_lab = rgb_to_lab(v226_rgb)
    target_direction = target_lab - base_lab
    final_direction = final_lab - base_lab
    projection = (final_direction * target_direction).sum(dim=1, keepdim=True) / target_direction.square().sum(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    result = paired_edge_nearcore_metrics(final_rgb, core, inner_edge)
    result.update(base_color_rim_metrics(base_rgb, final_rgb, core, inner_edge))
    result.update({
        "core_ref_full": masked_mean(torch.linalg.vector_norm(final_lab - target_lab, dim=1, keepdim=True), core),
        "core_ref_ab": masked_mean(torch.linalg.vector_norm(final_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True), core),
        "core_ref_l": masked_mean((final_lab[:, :1] - target_lab[:, :1]).abs(), core),
        "v226_core_ref_full": masked_mean(torch.linalg.vector_norm(v226_lab - target_lab, dim=1, keepdim=True), core),
        "v226_core_ref_ab": masked_mean(torch.linalg.vector_norm(v226_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True), core),
        "core_luma_hf_to_anchor": masked_mean((highpass(luminance(final_rgb)) - highpass(luminance(anchor_rgb))).abs(), core),
        "v226_luma_hf_to_anchor": masked_mean((highpass(luminance(v226_rgb)) - highpass(luminance(anchor_rgb))).abs(), core),
        "gradient_to_anchor": masked_mean((gradient(final_rgb) - gradient(anchor_rgb)).abs(), core),
        "face_boundary_contamination": masked_mean((final_rgb - base_rgb).abs(), face_contact),
        "face_contact_color_leak_rgb": masked_mean((final_rgb - base_rgb).abs(), face_contact),
        "face_contact_color_leak_de": masked_mean(torch.linalg.vector_norm(final_lab - base_lab, dim=1, keepdim=True), face_contact),
        "face_contact_target_projection": masked_mean(projection.clamp_min(0), face_contact),
        "v229_face_contact_color_leak_rgb": masked_mean((v229_final_rgb - base_rgb).abs(), face_contact),
        "reference_background_color_retention": masked_mean((final_rgb - base_rgb).abs(), background_contact) / masked_mean((anchor_rgb - base_rgb).abs(), background_contact).clamp_min(1e-6),
        "pp_reference_background_color_retention": masked_mean((pp_rgb - base_rgb).abs(), background_contact) / masked_mean((anchor_rgb - base_rgb).abs(), background_contact).clamp_min(1e-6),
        "seam_hf_energy": seam_hf_energy(final_rgb, pp_rgb, contour_band),
        "v229_seam_hf_energy": seam_hf_energy(v229_final_rgb, pp_rgb, contour_band),
        "outer_gradient_to_pp": masked_mean((gradient(final_rgb) - gradient(pp_rgb)).abs(), contour_band),
        "outer_texture_to_pp": masked_mean((highpass(final_rgb) - highpass(pp_rgb)).abs(), contour_band),
    })
    return result


def aggregate_v230(records: list[dict[str, object]]) -> dict[str, float | int]:
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
        summary[f"max_{key}"] = float(max(values))
        summary[f"p90_{key}"] = float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))])
    return summary


def classify_v230(
    summary: dict[str, float | int],
    *,
    parser_audit_passed: bool,
    parity_max_diff: float,
) -> str:
    if not parser_audit_passed:
        return "V230_PARSER_REGION_AUDIT_FAIL"
    if parity_max_diff > 1e-5:
        return "V230_RUNTIME_PARITY_FAIL"
    if (
        float(summary.get("median_core_ref_full", float("inf"))) > 1.03 * float(summary.get("median_v226_core_ref_full", 0.0))
        or float(summary.get("median_core_ref_ab", float("inf"))) > 1.03 * float(summary.get("median_v226_core_ref_ab", 0.0))
    ):
        return "V230_CORE_COLOR_FAIL"
    if float(summary.get("median_core_luma_hf_to_anchor", float("inf"))) > 0.45 * float(
        summary.get("median_v226_luma_hf_to_anchor", 0.0)
    ):
        return "V230_APPEARANCE_FAIL"
    if (
        float(summary.get("median_base_color_rim_fraction", 1.0)) > 0.12
        or float(summary.get("p90_base_color_rim_fraction", 1.0)) > 0.25
        or float(summary.get("median_base_color_edge_progress", 0.0)) < 0.78
    ):
        return "V230_BASE_RIM_FAIL"
    if (
        float(summary.get("median_face_boundary_contamination", 1.0)) > 0.07
        or float(summary.get("p90_face_boundary_contamination", 1.0)) > 0.12
        or float(summary.get("median_face_contact_color_leak_rgb", 1.0)) > 0.70 * float(summary.get("median_v229_face_contact_color_leak_rgb", 0.0))
    ):
        return "V230_FACE_CONTACT_FAIL"
    if (
        float(summary.get("median_reference_background_color_retention", 1.0)) > 0.30
        or float(summary.get("median_reference_background_color_retention", 1.0)) > 1.10 * float(summary.get("median_pp_reference_background_color_retention", 0.0))
    ):
        return "V230_BACKGROUND_RETENTION_FAIL"
    if (
        float(summary.get("median_seam_hf_energy", 1.0)) > 0.60 * float(summary.get("median_v229_seam_hf_energy", 0.0))
        or float(summary.get("max_visible_contour_core_weight", 1.0)) > 1e-6
    ):
        return "V230_JAGGED_EDGE_FAIL"
    if (
        float(summary.get("max_accepted_face_hole_fraction", 1.0)) > 1e-8
        or float(summary.get("max_outside_exact_pp_max_delta", 1.0)) > 1e-6
    ):
        return "V230_TOPOLOGY_REPAIR_FAIL"
    return "V230_READY_FOR_VISUAL_REVIEW"


__all__ = ["aggregate_v230", "classify_v230", "seam_hf_energy", "v230_metric_tensors"]
