"""Corrected V2.26 boundary metrics and deterministic acceptance helpers."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from models.boundary_target_v826 import (
    build_desired_edge_ab_target,
    build_desired_edge_l_target,
    compute_reference_relative_halo_limit,
)

V226_MEANINGFUL_EDGE_AB = 1.5
V226_MEANINGFUL_GLOBAL_DELTA_L = 3.0
V226_STRONG_GLOBAL_DELTA_L = 5.0
V226_RAW_HALO_EPS = 1e-6


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3:
        value = value.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (value * mask).flatten(1).sum(dim=1) / denominator


def _quantile(values: Iterable[float], fraction: float) -> float:
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return 0.0
    pos = (len(values) - 1) * float(fraction)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def boundary_metric_tensors(
    *,
    base_lab: torch.Tensor,
    anchor_lab: torch.Tensor,
    selective_lab: torch.Tensor,
    target_ref_ab: torch.Tensor,
    reference_delta_l: torch.Tensor,
    hair_edge: torch.Tensor,
    edge_chroma_strength: float = 0.70,
    edge_luma_strength: float = 0.65,
    edge_target_l_margin: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Compute per-sample corrected boundary metrics in Lab space."""
    if base_lab.shape != anchor_lab.shape or base_lab.shape != selective_lab.shape:
        raise ValueError("Lab inputs must have the same shape")
    base_ab, anchor_ab, selective_ab = base_lab[:, 1:], anchor_lab[:, 1:], selective_lab[:, 1:]
    desired_ab = build_desired_edge_ab_target(base_ab, target_ref_ab, edge_chroma_strength)
    desired_delta_ab = desired_ab - base_ab
    output_delta_ab = selective_ab - base_ab
    desired_norm = torch.linalg.vector_norm(desired_delta_ab, dim=1, keepdim=True)
    desired_error = _masked_mean(torch.linalg.vector_norm(selective_ab - desired_ab, dim=1, keepdim=True), hair_edge)
    base_error = _masked_mean(torch.linalg.vector_norm(base_ab - desired_ab, dim=1, keepdim=True), hair_edge).clamp_min(1e-6)
    desired_progress = 1.0 - desired_error / base_error
    valid_ab = (desired_norm >= V226_MEANINGFUL_EDGE_AB).to(base_lab.dtype)
    direction_ok = ((output_delta_ab * desired_delta_ab).sum(dim=1, keepdim=True) > 0).to(base_lab.dtype)
    direction = _masked_mean(direction_ok * valid_ab, hair_edge)
    current_parallel = torch.relu((output_delta_ab * desired_delta_ab / desired_norm.clamp_min(1e-6)).sum(dim=1, keepdim=True))
    desired_mag = _masked_mean(desired_norm, hair_edge)
    parallel_mag = _masked_mean(current_parallel, hair_edge)
    magnitude_ratio = parallel_mag / desired_mag.clamp_min(1e-6)

    delta_l = torch.as_tensor(reference_delta_l, device=base_lab.device, dtype=base_lab.dtype).reshape(-1)
    if delta_l.numel() != base_lab.size(0):
        raise ValueError("reference_delta_l must contain one value per sample")
    output_shift_l = _masked_mean(selective_lab[:, :1] - base_lab[:, :1], hair_edge)
    desired_shift_l = float(edge_luma_strength) * delta_l
    meaningful_l = delta_l.abs() >= V226_MEANINGFUL_GLOBAL_DELTA_L
    strong_l = delta_l.abs() >= V226_STRONG_GLOBAL_DELTA_L
    l_progress = 1.0 - (output_shift_l - desired_shift_l).abs() / desired_shift_l.abs().clamp_min(1e-6)
    l_direction = (output_shift_l * desired_shift_l > 0).to(base_lab.dtype)

    allowed_l = compute_reference_relative_halo_limit(
        base_lab[:, :1], delta_l, edge_luma_strength, edge_target_l_margin
    )
    selective_excess = _masked_mean(torch.relu(selective_lab[:, :1] - allowed_l), hair_edge)
    raw_excess = _masked_mean(torch.relu(anchor_lab[:, :1] - allowed_l), hair_edge)
    raw_valid = raw_excess > V226_RAW_HALO_EPS
    return {
        "desired_edge_ab": desired_ab,
        "desired_ab_error": desired_error,
        "base_to_desired_ab_error": base_error,
        "desired_ab_progress": desired_progress,
        "desired_ab_direction_agreement": direction,
        "desired_ab_magnitude_ratio": magnitude_ratio,
        "desired_ab_norm": desired_mag,
        "current_ab_parallel": parallel_mag,
        "output_edge_signed_l_shift": output_shift_l,
        "desired_edge_signed_l_shift": desired_shift_l,
        "signed_l_progress": l_progress,
        "signed_l_direction_agreement": l_direction,
        "meaningful_l_sample": meaningful_l,
        "strong_l_sample": strong_l,
        "allowed_edge_l": allowed_l,
        "reference_halo_excess": selective_excess,
        "raw_reference_halo_excess": raw_excess,
        "reference_halo_valid": raw_valid,
    }


def add_anchor_deficit_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    desired_mag = metrics["desired_ab_norm"]
    current_parallel = metrics["current_ab_parallel"]
    raw_deficit = torch.relu(desired_mag - current_parallel)
    bounded = torch.minimum(raw_deficit, torch.minimum(0.40 * desired_mag, raw_deficit.new_tensor(6.0)))
    metrics = dict(metrics)
    metrics.update({
        "raw_anchor_deficit_mag": raw_deficit,
        "bounded_anchor_deficit_mag": bounded,
        "anchor_deficit_fraction": raw_deficit / desired_mag.clamp_min(1e-6),
    })
    return metrics


def aggregate_v226_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"count": 0, "has_samples": False, "meaningful_l_count": 0, "strong_l_count": 0}
    def vals(key, subset=records):
        return [float(r[key]) for r in subset if key in r and math.isfinite(float(r[key]))]
    meaningful = [r for r in records if bool(r.get("meaningful_l_sample", False))]
    valid_halo = [r for r in records if bool(r.get("reference_halo_valid", False))]
    result = {
        "count": len(records), "has_samples": True,
        "meaningful_l_count": len(meaningful), "strong_l_count": sum(bool(r.get("strong_l_sample", False)) for r in records),
        "median_edge_desired_ab_progress": _quantile(vals("edge_desired_ab_progress"), .5),
        "p25_edge_desired_ab_progress": _quantile(vals("edge_desired_ab_progress"), .25),
        "mean_edge_desired_ab_direction_agreement": sum(vals("edge_desired_ab_direction_agreement")) / max(len(vals("edge_desired_ab_direction_agreement")), 1),
        "median_edge_desired_ab_magnitude_ratio": _quantile(vals("edge_desired_ab_magnitude_ratio"), .5),
        "mean_edge_desired_mag": sum(vals("desired_ab_norm")) / max(len(vals("desired_ab_norm")), 1),
        "median_edge_transfer_ab_vs_anchor": _quantile(vals("edge_transfer_ab"), .5),
        "median_edge_transfer_l_vs_anchor": _quantile(vals("edge_transfer_l"), .5),
        "median_edge_signed_l_progress": _quantile(vals("edge_signed_l_progress", meaningful), .5),
        "p25_edge_signed_l_progress": _quantile(vals("edge_signed_l_progress", meaningful), .25),
        "mean_edge_signed_l_direction_agreement": sum(vals("edge_signed_l_direction_agreement", meaningful)) / max(len(vals("edge_signed_l_direction_agreement", meaningful)), 1),
        "reference_halo_valid_count": len(valid_halo),
        "median_reference_halo_ratio": (_quantile((float(r["reference_halo_excess"]) / max(float(r["raw_reference_halo_excess"]), V226_RAW_HALO_EPS) for r in valid_halo), .5) if valid_halo else None),
        "median_edge_deficit_fraction": _quantile(vals("anchor_deficit_fraction"), .5),
        "p75_edge_deficit_fraction": _quantile(vals("anchor_deficit_fraction"), .75),
        "outside_hair_max_abs_delta": max(vals("outside_hair_max_abs_delta") or [0.0]),
        "hard_protect_max_abs_delta": max(vals("hard_protect_max_abs_delta") or [0.0]),
        "core_full_error": _quantile(vals("selective_full_stat_error"), .5),
        "core_base_full_error": _quantile(vals("base_full_stat_error"), .5),
        "full_retention": _quantile(vals("full_color_retention"), .5),
        "ab_retention": _quantile(vals("ab_retention"), .5),
        "l_progress": _quantile(vals("l_progress"), .5),
    }
    result["reference_halo_ratio_valid"] = bool(valid_halo)
    return result


def classify_v226_metric_alignment(summary: dict[str, object], *, parity_max_diff: float = 0.0) -> dict[str, object]:
    if parity_max_diff > 1e-6:
        return {"decision": "V226_GAMMA0_PARITY_BUG", "abort": True}
    if float(summary.get("outside_hair_max_abs_delta", 0.0)) > 5e-4 or float(summary.get("hard_protect_max_abs_delta", 0.0)) > 5e-4:
        return {"decision": "V226_CORE_OR_PROTECTION_REGRESSION", "abort": True}
    core = float(summary.get("core_full_error", 0.0))
    base = float(summary.get("core_base_full_error", 0.0))
    if core > max(base * 1.05, base + 0.20) or float(summary.get("full_retention", 0.0)) < .90 or float(summary.get("ab_retention", 0.0)) < .85 or float(summary.get("l_progress", 0.0)) < .75:
        return {"decision": "V226_CORE_OR_PROTECTION_REGRESSION", "abort": False}
    if int(summary.get("meaningful_l_count", 0)) == 0:
        return {"decision": "V226_INSUFFICIENT_MEANINGFUL_SAMPLES", "abort": False}
    if float(summary.get("median_edge_desired_ab_progress", 0.0)) >= .60 and float(summary.get("p25_edge_desired_ab_progress", 0.0)) >= .35 and float(summary.get("mean_edge_desired_ab_direction_agreement", 0.0)) >= .85 and float(summary.get("median_edge_transfer_ab_vs_anchor", 0.0)) >= .45 and float(summary.get("median_edge_transfer_l_vs_anchor", 0.0)) >= .45 and float(summary.get("median_edge_signed_l_progress", 0.0)) >= .55 and float(summary.get("mean_edge_signed_l_direction_agreement", 0.0)) >= .85 and (not summary.get("reference_halo_ratio_valid") or float(summary.get("median_reference_halo_ratio", 0.0)) <= .35):
        return {"decision": "V226_METRIC_ALIGNMENT_PASS_NO_COMPENSATION", "abort": False}
    if float(summary.get("median_edge_deficit_fraction", 0.0)) < .15:
        return {"decision": "V226_FAIL_NOT_ANCHOR_DEFICIT", "abort": False}
    return {"decision": "V226_NEEDS_BOUNDED_COMPENSATION", "abort": False}


def classify_v226_gamma_sweep(summary: dict[str, object], *, parity_max_diff: float = 0.0) -> dict[str, object]:
    if parity_max_diff > 1e-6:
        return {"decision": "V226_GAMMA0_PARITY_BUG", "abort": True}
    for gamma in (0.0, 0.25, 0.50, 0.75):
        candidate = summary.get(f"gamma_{gamma:.2f}")
        if candidate and classify_v226_metric_alignment(candidate)["decision"] == "V226_METRIC_ALIGNMENT_PASS_NO_COMPENSATION":
            return {"decision": "V226_BOUNDED_DEFICIT_PASS" if gamma else "V226_METRIC_ALIGNMENT_PASS_NO_COMPENSATION", "gamma_star": gamma, "abort": False}
    return {"decision": "V226_NO_SAFE_COMPENSATION", "abort": False}
