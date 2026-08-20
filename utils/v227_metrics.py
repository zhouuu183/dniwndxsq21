"""V2.27 paired boundary metrics and independent artifact screening."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from models.boundary_target_v826 import build_desired_edge_ab_target

V227_MEANINGFUL_EDGE_AB = 1.5
V227_MIN_MEANINGFUL_AB_FRACTION = 0.05


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3:
        value = value.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1.0)


def _quantile(values: Iterable[float], fraction: float) -> float:
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return 0.0
    pos = (len(values) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def meaningful_ab_metrics(
    *,
    base_lab: torch.Tensor,
    output_lab: torch.Tensor,
    target_ref_ab: torch.Tensor,
    hair_edge: torch.Tensor,
    edge_chroma_strength: float = 0.70,
) -> dict[str, torch.Tensor]:
    """Evaluate progress, direction and deficit on one shared meaningful mask."""
    base_ab, output_ab = base_lab[:, 1:], output_lab[:, 1:]
    desired_ab = build_desired_edge_ab_target(
        base_ab, target_ref_ab, edge_chroma_strength
    )
    desired_delta = desired_ab - base_ab
    output_delta = output_ab - base_ab
    desired_norm = torch.linalg.vector_norm(desired_delta, dim=1, keepdim=True)
    meaningful = hair_edge.float() * (desired_norm >= V227_MEANINGFUL_EDGE_AB).float()
    edge_count = hair_edge.flatten(1).sum(1)
    meaningful_count = meaningful.flatten(1).sum(1)
    fraction = meaningful_count / edge_count.clamp_min(1.0)
    valid = (fraction >= V227_MIN_MEANINGFUL_AB_FRACTION) & (meaningful_count > 0)

    error = _masked_mean(
        torch.linalg.vector_norm(output_ab - desired_ab, dim=1, keepdim=True),
        meaningful,
    )
    base_error = _masked_mean(desired_norm, meaningful).clamp_min(1e-6)
    progress = 1.0 - error / base_error
    direction_ok = ((output_delta * desired_delta).sum(1, keepdim=True) > 0).float()
    direction = _masked_mean(direction_ok, meaningful)
    desired_unit = desired_delta / desired_norm.clamp_min(1e-6)
    parallel_map = torch.relu((output_delta * desired_unit).sum(1, keepdim=True))
    desired_mag = _masked_mean(desired_norm, meaningful)
    current_parallel = _masked_mean(parallel_map, meaningful)
    deficit_map = torch.relu(desired_norm - parallel_map)
    deficit = _masked_mean(deficit_map, meaningful)
    deficit_fraction = deficit / desired_mag.clamp_min(1e-6)

    legacy_error = _masked_mean(
        torch.linalg.vector_norm(output_ab - desired_ab, dim=1, keepdim=True), hair_edge
    )
    legacy_base = _masked_mean(desired_norm, hair_edge).clamp_min(1e-6)
    return {
        "desired_edge_ab": desired_ab,
        "desired_ab_norm_map": desired_norm,
        "meaningful_ab_edge": meaningful,
        "meaningful_ab_pixel_count": meaningful_count,
        "meaningful_ab_pixel_fraction": fraction,
        "meaningful_ab_valid": valid,
        "edge_desired_ab_progress_meaningful": progress,
        "edge_desired_ab_direction_meaningful": direction,
        "edge_desired_ab_magnitude_ratio_meaningful": current_parallel / desired_mag.clamp_min(1e-6),
        "edge_remaining_deficit_fraction_meaningful": deficit_fraction,
        "edge_remaining_deficit_map": deficit_map * meaningful,
        "legacy_edge_desired_ab_progress_all_edge": 1.0 - legacy_error / legacy_base,
    }


def _local_core_reference(value: torch.Tensor, core: torch.Tensor, radius: int = 4) -> torch.Tensor:
    kernel = 2 * radius + 1
    numerator = F.avg_pool2d(value * core, kernel, stride=1, padding=radius)
    denominator = F.avg_pool2d(core, kernel, stride=1, padding=radius).clamp_min(1e-6)
    local = numerator / denominator
    global_mean = _masked_mean(value, core).view(-1, 1, 1, 1)
    return torch.where(denominator > 1e-5, local, global_mean)


def independent_edge_artifact_metrics(
    *, base_lab: torch.Tensor, output_lab: torch.Tensor,
    hair_core: torch.Tensor, hair_edge: torch.Tensor,
    l_margin: float = 5.0, chroma_margin: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Screen edge artifacts without using the projector's L guard threshold."""
    base_l, out_l = base_lab[:, :1], output_lab[:, :1]
    base_c = torch.linalg.vector_norm(base_lab[:, 1:], dim=1, keepdim=True)
    out_c = torch.linalg.vector_norm(output_lab[:, 1:], dim=1, keepdim=True)
    core_l = _local_core_reference(out_l, hair_core)
    core_ab = _local_core_reference(output_lab[:, 1:], hair_core)
    core_c = torch.linalg.vector_norm(core_ab, dim=1, keepdim=True)

    edge_core_l = (out_l - core_l).abs()
    edge_core_ab = torch.linalg.vector_norm(output_lab[:, 1:] - core_ab, dim=1, keepdim=True)
    edge_core_de = torch.sqrt(edge_core_l.square() + edge_core_ab.square())
    core_delta = torch.linalg.vector_norm(output_lab - base_lab, dim=1, keepdim=True)
    edge_delta = _masked_mean(core_delta, hair_edge)
    core_delta_mean = _masked_mean(core_delta, hair_core).clamp_min(1e-6)

    white_map = (
        (out_l > torch.maximum(base_l, core_l) + l_margin)
        & (out_c < torch.minimum(base_c, core_c) - chroma_margin)
    ).float() * hair_edge
    gray_map = (
        (out_l > torch.maximum(base_l, core_l) + l_margin)
        & (out_c < torch.minimum(base_c, core_c))
    ).float() * hair_edge
    oversaturation_map = (
        out_c > torch.maximum(base_c, core_c) + 5.0
    ).float() * hair_edge
    dark_map = (
        out_l < torch.minimum(base_l, core_l) - 5.0
    ).float() * hair_edge
    fringe_map = torch.maximum(
        torch.maximum(white_map, gray_map), torch.maximum(oversaturation_map, dark_map)
    )
    return {
        "edge_core_delta_e": _masked_mean(edge_core_de, hair_edge),
        "edge_core_delta_l": _masked_mean(edge_core_l, hair_edge),
        "edge_core_delta_ab": _masked_mean(edge_core_ab, hair_edge),
        "edge_transfer_continuity_ratio": edge_delta / core_delta_mean,
        "white_fringe_fraction": _masked_mean(white_map, hair_edge),
        "gray_fringe_fraction": _masked_mean(gray_map, hair_edge),
        "oversaturation_fringe_fraction": _masked_mean(oversaturation_map, hair_edge),
        "dark_fringe_fraction": _masked_mean(dark_map, hair_edge),
        "independent_fringe_map": fringe_map,
    }


def aggregate_v227_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"count": 0, "has_samples": False}
    valid = [r for r in records if bool(r.get("meaningful_ab_valid", False))]
    meaningful_l = [r for r in records if bool(r.get("meaningful_l_sample", False))]
    valid_halo = [r for r in records if bool(r.get("reference_halo_valid", False))]
    def values(key, source=records):
        return [float(r[key]) for r in source if key in r and math.isfinite(float(r[key]))]
    result = {
        "count": len(records), "has_samples": True,
        "meaningful_ab_sample_count": len(valid),
        "median_meaningful_ab_pixel_fraction": _quantile(values("meaningful_ab_pixel_fraction"), .5),
        "median_edge_desired_ab_progress_meaningful": _quantile(values("edge_desired_ab_progress_meaningful", valid), .5),
        "p25_edge_desired_ab_progress_meaningful": _quantile(values("edge_desired_ab_progress_meaningful", valid), .25),
        "mean_edge_desired_ab_direction_meaningful": sum(values("edge_desired_ab_direction_meaningful", valid)) / max(len(values("edge_desired_ab_direction_meaningful", valid)), 1),
        "median_edge_desired_ab_magnitude_ratio_meaningful": _quantile(values("edge_desired_ab_magnitude_ratio_meaningful", valid), .5),
        "median_remaining_deficit_fraction": _quantile(values("edge_remaining_deficit_fraction_meaningful", valid), .5),
        "median_legacy_all_edge_progress": _quantile(values("legacy_edge_desired_ab_progress_all_edge"), .5),
        "median_edge_transfer_ab_vs_anchor": _quantile(values("edge_transfer_ab"), .5),
        "meaningful_l_sample_count": len(meaningful_l),
        "median_meaningful_l_progress": _quantile(values("edge_signed_l_progress", meaningful_l), .5),
        "mean_meaningful_l_direction": sum(values("edge_signed_l_direction_agreement", meaningful_l)) / max(len(values("edge_signed_l_direction_agreement", meaningful_l)), 1),
        "guard_consistency_halo_ratio": _quantile((
            float(r["reference_halo_excess"]) / max(float(r["raw_reference_halo_excess"]), 1e-6)
            for r in valid_halo
        ), .5) if valid_halo else None,
        "core_full_error": _quantile(values("selective_full_stat_error"), .5),
        "base_core_full_error": _quantile(values("base_full_stat_error"), .5),
        "ab_retention": _quantile(values("ab_retention"), .5),
        "l_progress": _quantile(values("l_progress"), .5),
        "outside_hair_max_abs_delta": max(values("outside_hair_max_abs_delta") or [0.0]),
        "hard_protect_max_abs_delta": max(values("hard_protect_max_abs_delta") or [0.0]),
    }
    for key in ("edge_core_delta_e", "edge_core_delta_l", "edge_core_delta_ab", "edge_transfer_continuity_ratio", "white_fringe_fraction", "gray_fringe_fraction", "oversaturation_fringe_fraction", "dark_fringe_fraction"):
        result[f"median_{key}"] = _quantile(values(key), .5)
        result[f"p90_{key}"] = _quantile(values(key), .9)
    result["independent_edge_artifact_score"] = max(
        float(result["median_white_fringe_fraction"]),
        float(result["median_gray_fringe_fraction"]),
        float(result["median_oversaturation_fringe_fraction"]),
        float(result["median_dark_fringe_fraction"]),
    )
    return result


def classify_v227(
    baseline: dict[str, object], selected: dict[str, object],
    *, code_correctness: dict[str, bool], artifact_tolerance: float = 0.01,
) -> str:
    if not all(code_correctness.values()):
        return "V227_CODE_OR_PARITY_FAIL"
    if (float(selected.get("core_full_error", 0.0)) > max(float(baseline.get("core_full_error", 0.0)) * 1.05, float(baseline.get("core_full_error", 0.0)) + .20)
            or float(selected.get("ab_retention", 0.0)) < .85
            or float(selected.get("l_progress", 0.0)) < .75):
        return "V227_CORE_REGRESSION"
    if (float(selected.get("median_edge_desired_ab_progress_meaningful", 0.0)) < .60
            or float(selected.get("p25_edge_desired_ab_progress_meaningful", 0.0)) < .40
            or float(selected.get("mean_edge_desired_ab_direction_meaningful", 0.0)) < .88
            or float(selected.get("median_edge_transfer_ab_vs_anchor", 0.0)) < .45
            or float(selected.get("median_remaining_deficit_fraction", 1.0)) > .38):
        return "V227_NUMERIC_BOUNDARY_FAIL"
    artifact_keys = ("white_fringe_fraction", "oversaturation_fringe_fraction", "dark_fringe_fraction")
    for key in artifact_keys:
        if float(selected.get(f"median_{key}", 0.0)) > .03:
            return "V227_ARTIFACT_SCREEN_FAIL"
        if float(selected.get(f"median_{key}", 0.0)) > float(baseline.get(f"median_{key}", 0.0)) + artifact_tolerance:
            return "V227_ARTIFACT_SCREEN_FAIL"
    if float(selected.get("p90_white_fringe_fraction", 0.0)) > .08:
        return "V227_ARTIFACT_SCREEN_FAIL"
    return "V227_READY_FOR_VISUAL_REVIEW"
