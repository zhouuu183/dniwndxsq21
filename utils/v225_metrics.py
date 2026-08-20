"""Reference-conditioned boundary metrics and acceptance for V2.25."""

from __future__ import annotations

from typing import Iterable

from utils.v223_metrics import _quantile, aggregate_v223_records, build_v223_normal_color_manifest


def build_v225_normal_color_manifest(records: list[dict[str, object]], min_safe_reference_fraction: float):
    manifest = build_v223_normal_color_manifest(records, min_safe_reference_fraction)
    manifest["version"] = "v2.25"
    for entry in manifest["entries"]:
        entry["extreme_for_v225"] = entry.pop("extreme_for_v223", False)
    return manifest


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def aggregate_v225_records(records: list[dict[str, object]], normal_ids: set[str] | None = None):
    selected = records if normal_ids is None else [r for r in records if r["sample_id"] in normal_ids]
    if not selected:
        return {"count": 0, "has_samples": False}
    summary = dict(aggregate_v223_records(selected))
    summary.update({
        "median_v224_selective_full_stat_error": _quantile((r["v224_selective_full_stat_error"] for r in selected), 0.5),
        "median_v224_edge_transfer_full": _quantile((r["v224_edge_transfer_full"] for r in selected), 0.5),
        "median_v224_edge_transfer_ab": _quantile((r["v224_edge_transfer_ab"] for r in selected), 0.5),
        "median_v224_edge_transfer_l": _quantile((r["v224_edge_transfer_l"] for r in selected), 0.5),
        "median_v225_edge_transfer_full": _quantile((r["edge_transfer_full"] for r in selected), 0.5),
        "median_v225_edge_transfer_ab": _quantile((r["edge_transfer_ab"] for r in selected), 0.5),
        "median_v225_edge_transfer_l": _quantile((r["edge_transfer_l"] for r in selected), 0.5),
        "median_edge_reference_ab_progress": _quantile((r["edge_reference_ab_progress"] for r in selected), 0.5),
        "median_edge_reference_l_progress": _quantile((r["edge_reference_l_progress"] for r in selected), 0.5),
        "mean_edge_ab_direction_agreement": _mean(r["edge_ab_direction_agreement_fraction"] for r in selected),
        "mean_edge_l_direction_agreement": _mean(r["edge_l_direction_agreement_fraction"] for r in selected),
        "mean_edge_parallel_cap_saturation": _mean(r["edge_parallel_cap_saturation_fraction"] for r in selected),
        "mean_edge_positive_overshoot_before": _mean(r["edge_positive_overshoot_before"] for r in selected),
        "mean_edge_positive_overshoot_after": _mean(r["edge_positive_overshoot_after"] for r in selected),
        "mean_edge_target_ab_norm": _mean(r["edge_reference_target_ab_norm"] for r in selected),
        "mean_edge_pseudo_ab_norm": _mean(r["edge_pseudo_ab_norm"] for r in selected),
        "mean_edge_reference_parallel_mag": _mean(r["edge_reference_parallel_mag"] for r in selected),
    })
    raw = float(summary.get("raw_anchor_edge_luma_excess_mean", 0.0))
    summary["edge_halo_ratio"] = float(summary.get("edge_luma_excess_mean", 0.0)) / raw if raw > 1e-6 else 0.0
    return summary


def classify_v225(summary: dict[str, object], *, parity_max_diff: float, outside_max_delta: float, hard_protect_max_delta: float):
    if parity_max_diff > 1e-6 or outside_max_delta > 5e-4 or hard_protect_max_delta > 5e-4:
        return {"decision": "V225_PROTECTION_OR_PARITY_BUG", "abort": True}
    old = float(summary.get("median_v224_selective_full_stat_error", summary.get("median_v223_old_selective_full_stat_error", 0.0)))
    new = float(summary.get("median_selective_full_stat_error", 0.0))
    if new > max(old * 1.05, old + 0.20) or float(summary.get("median_full_color_retention", 0.0)) < 0.90 or float(summary.get("median_ab_retention", 0.0)) < 0.85 or float(summary.get("median_l_progress", 0.0)) < 0.75:
        return {"decision": "V225_CORE_REGRESSION", "abort": False}
    if float(summary.get("mean_edge_reference_parallel_mag", 0.0)) < 1e-3:
        return {"decision": "V225_BOUNDARY_TARGET_MISSING", "abort": False}
    if float(summary.get("median_v225_edge_transfer_ab", 0.0)) < 0.25 and float(summary.get("median_edge_reference_ab_progress", 0.0)) < 0.45:
        return {"decision": "V225_BOUNDARY_AB_SUPPRESSED", "abort": False}
    if float(summary.get("median_v225_edge_transfer_l", 0.0)) < 0.35 and float(summary.get("median_edge_reference_l_progress", 0.0)) < 0.45:
        return {"decision": "V225_BOUNDARY_L_SUPPRESSED", "abort": False}
    if float(summary.get("edge_halo_ratio", 0.0)) > 0.50:
        return {"decision": "V225_HALO_REGRESSION", "abort": False}
    if (float(summary.get("median_v225_edge_transfer_full", 0.0)) < 0.35 or float(summary.get("median_v225_edge_transfer_ab", 0.0)) < 0.30 or float(summary.get("median_v225_edge_transfer_l", 0.0)) < 0.35 or float(summary.get("median_edge_reference_ab_progress", 0.0)) < 0.50 or float(summary.get("median_edge_reference_l_progress", 0.0)) < 0.50 or float(summary.get("mean_edge_ab_direction_agreement", 0.0)) < 0.85 or float(summary.get("mean_edge_l_direction_agreement", 0.0)) < 0.85):
        return {"decision": "V225_BOUNDARY_WEAK", "abort": False}
    return {"decision": "V225_REFERENCE_BOUNDARY_PASS", "abort": False}
