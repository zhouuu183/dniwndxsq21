"""Acceptance metrics for V2.24 boundary single-alpha diagnostics."""

from __future__ import annotations

from typing import Iterable

from utils.v223_metrics import (
    _quantile,
    aggregate_v223_records,
    build_v223_normal_color_manifest,
)


def build_v224_normal_color_manifest(
    records: list[dict[str, object]], min_safe_reference_fraction: float
) -> dict[str, object]:
    manifest = build_v223_normal_color_manifest(records, min_safe_reference_fraction)
    for entry in manifest["entries"]:
        entry["extreme_for_v224"] = entry.pop("extreme_for_v223", False)
    manifest["version"] = "v2.24"
    return manifest


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def aggregate_v224_records(
    records: list[dict[str, object]], normal_ids: set[str] | None = None
) -> dict[str, float | int | bool]:
    selected = records if normal_ids is None else [
        record for record in records if record["sample_id"] in normal_ids
    ]
    if not selected:
        return {"count": 0, "has_samples": False}
    summary = dict(aggregate_v223_records(selected))
    summary.update({
        "median_v224_edge_transfer_full": _quantile(
            (record["edge_transfer_full"] for record in selected), 0.50
        ),
        "p25_v224_edge_transfer_full": _quantile(
            (record["edge_transfer_full"] for record in selected), 0.25
        ),
        "median_v224_edge_transfer_ab": _quantile(
            (record["edge_transfer_ab"] for record in selected), 0.50
        ),
        "median_v224_edge_transfer_l": _quantile(
            (record["edge_transfer_l"] for record in selected if record["edge_l_transfer_valid"]), 0.50
        ),
        "mean_edge_l_direction_agreement_fraction": _mean(
            record["edge_l_direction_agreement_fraction"]
            for record in selected if record["edge_l_transfer_valid"]
        ),
        "median_v224_selective_full_stat_error": _quantile(
            (record["selective_full_stat_error"] for record in selected), 0.50
        ),
        "median_v223_old_selective_full_stat_error": _quantile(
            (record["v223_old_selective_full_stat_error"] for record in selected), 0.50
        ),
        "median_v223_old_full_color_retention": _quantile(
            (record["v223_old_full_color_retention"] for record in selected), 0.50
        ),
        "median_v223_old_ab_retention": _quantile(
            (record["v223_old_ab_retention"] for record in selected), 0.50
        ),
        "median_v223_old_l_progress": _quantile(
            (record["v223_old_l_progress"] for record in selected), 0.50
        ),
        "mean_edge_membership": _mean(record["mean_edge_membership"] for record in selected),
        "mean_edge_chroma_weight": _mean(record["mean_edge_chroma_weight"] for record in selected),
        "mean_edge_luma_weight": _mean(record["mean_edge_luma_weight"] for record in selected),
        "mean_core_chroma_weight": _mean(record["mean_core_chroma_weight"] for record in selected),
        "mean_core_luma_weight": _mean(record["mean_core_luma_weight"] for record in selected),
        "edge_luma_excess_after_guard": _mean(
            record["edge_luma_excess_after_guard"] for record in selected
        ),
    })
    raw_halo = float(summary.get("raw_anchor_edge_luma_excess_mean", 0.0))
    summary["edge_halo_ratio"] = (
        float(summary["edge_luma_excess_mean"]) / raw_halo if raw_halo > 1e-6 else 0.0
    )
    return summary


def classify_v224(
    summary: dict[str, object],
    *,
    parity_max_diff: float,
    outside_max_delta: float,
    hard_protect_max_delta: float,
) -> dict[str, object]:
    if (
        parity_max_diff > 1e-6
        or outside_max_delta > 5e-4
        or hard_protect_max_delta > 5e-4
    ):
        return {"decision": "V224_PROTECTION_OR_PARITY_BUG", "abort": True}
    old_full = float(summary.get("median_v223_old_selective_full_stat_error", 0.0))
    new_full = float(summary.get("median_v224_selective_full_stat_error", 0.0))
    if new_full > max(old_full * 1.05, old_full + 0.20):
        return {"decision": "V224_CORE_REGRESSION", "abort": False}
    if float(summary.get("median_full_color_retention", 0.0)) < 0.90:
        return {"decision": "V224_CORE_REGRESSION", "abort": False}
    if float(summary.get("median_ab_retention", 0.0)) < 0.85:
        return {"decision": "V224_CORE_REGRESSION", "abort": False}
    if float(summary.get("median_l_progress", 0.0)) < 0.75:
        return {"decision": "V224_CORE_REGRESSION", "abort": False}
    edge_full = float(summary.get("median_v224_edge_transfer_full", 0.0))
    halo_ratio = float(summary.get("edge_halo_ratio", 0.0))
    if edge_full < 0.20:
        if halo_ratio < 0.05:
            return {"decision": "V224_OVER_SUPPRESSED_BOUNDARY", "abort": False}
        return {"decision": "V224_BOUNDARY_STILL_COLLAPSED", "abort": False}
    if edge_full < 0.35:
        if halo_ratio < 0.05:
            return {"decision": "V224_OVER_SUPPRESSED_BOUNDARY", "abort": False}
        return {"decision": "V224_BOUNDARY_WEAK", "abort": False}
    if halo_ratio > 0.60:
        return {"decision": "V224_HALO_FAIL", "abort": False}
    if float(summary.get("median_v224_edge_transfer_ab", 0.0)) < 0.40:
        return {"decision": "V224_BOUNDARY_WEAK", "abort": False}
    return {"decision": "V224_BOUNDARY_PASS", "abort": False}
