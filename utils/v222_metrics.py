"""Aggregation and engineering gates for selective chroma v2.22."""

from __future__ import annotations

import math
from typing import Iterable


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def _quantile(values: Iterable[float], fraction: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def aggregate_v222_records(
    records: list[dict[str, object]],
    normal_ids: set[str] | None = None,
) -> dict[str, float | int | bool]:
    selected = records
    if normal_ids is not None:
        selected = [record for record in records if record["sample_id"] in normal_ids]
    retention = [
        float(record["color_retention_ratio"])
        for record in selected
        if bool(record["raw_anchor_positive_gain"])
    ]
    if not selected:
        return {"count": 0, "has_samples": False}
    return {
        "count": len(selected),
        "has_samples": True,
        "median_base_to_ref_ab": _quantile(
            (record["base_to_ref_ab"] for record in selected), 0.50
        ),
        "median_anchor_to_ref_ab": _quantile(
            (record["anchor_to_ref_ab"] for record in selected), 0.50
        ),
        "median_selective_to_ref_ab": _quantile(
            (record["selective_to_ref_ab"] for record in selected), 0.50
        ),
        "median_color_improvement": _quantile(
            (record["color_improvement_ratio"] for record in selected), 0.50
        ),
        "negative_improvement_fraction": _mean(
            float(record["color_improvement_ratio"]) < 0.0 for record in selected
        ),
        "median_color_retention_ratio": _quantile(retention, 0.50),
        "positive_anchor_gain_count": len(retention),
        "median_parallel_progress": _quantile(
            (record["parallel_progress"] for record in selected), 0.50
        ),
        "p25_parallel_progress": _quantile(
            (record["parallel_progress"] for record in selected), 0.25
        ),
        "median_orthogonal_error": _quantile(
            (record["orthogonal_error"] for record in selected), 0.50
        ),
        "p75_orthogonal_error": _quantile(
            (record["orthogonal_error"] for record in selected), 0.75
        ),
        "mean_reference_hue_error": _mean(
            record["selective_to_ref_hue"] for record in selected
        ),
        "mean_reference_chroma_error": _mean(
            record["selective_to_ref_chroma"] for record in selected
        ),
        "raw_anchor_edge_luma_excess_mean": _mean(
            record["anchor_edge_luma_excess_mean"] for record in selected
        ),
        "raw_anchor_face_keep_l1": _mean(
            record["anchor_face_keep_l1"] for record in selected
        ),
        "raw_anchor_outer_bg_keep_l1": _mean(
            record["anchor_outer_bg_keep_l1"] for record in selected
        ),
        "edge_luma_excess_mean": _mean(
            record["edge_luma_excess_mean"] for record in selected
        ),
        "edge_luma_excess_fraction": _mean(
            record["edge_luma_excess_fraction"] for record in selected
        ),
        "edge_hf_excess": _mean(record["edge_hf_excess"] for record in selected),
        "face_keep_l1": _mean(record["face_keep_l1"] for record in selected),
        "outer_bg_keep_l1": _mean(record["outer_bg_keep_l1"] for record in selected),
        "hard_protect_max_abs_delta": max(
            float(record["hard_protect_max_abs_delta"]) for record in selected
        ),
        "outside_hair_max_abs_delta": max(
            float(record["outside_hair_max_abs_delta"]) for record in selected
        ),
        "parallel_gain": _mean(record["parallel_gain"] for record in selected),
        "orth_keep": _mean(record["orth_keep"] for record in selected),
        "boundary_strength": _mean(record["boundary_strength"] for record in selected),
        "luma_strength": _mean(record["luma_strength"] for record in selected),
        "mean_A_core": _mean(record["mean_A_core"] for record in selected),
        "mean_A_edge": _mean(record["mean_A_edge"] for record in selected),
        "mean_halo_gate": _mean(record["mean_halo_gate"] for record in selected),
        "overshoot_fraction": _mean(record["overshoot_fraction"] for record in selected),
        "negative_parallel_fraction": _mean(
            record["negative_parallel_fraction"] for record in selected
        ),
    }


def classify_v222_pretrain(summary: dict[str, object]) -> dict[str, object]:
    if int(summary.get("count", 0)) == 0:
        return {"decision": "INSUFFICIENT_NORMAL_SAMPLES", "abort": True}
    if (
        float(summary["outside_hair_max_abs_delta"]) > 5e-4
        or float(summary["hard_protect_max_abs_delta"]) > 5e-4
    ):
        return {"decision": "MASK_PROTECTION_BUG", "abort": True}
    if float(summary["median_color_retention_ratio"]) < 0.70:
        return {"decision": "SELECTIVE_COLOR_LOSS_TOO_LARGE", "abort": False}
    raw_edge = float(summary["raw_anchor_edge_luma_excess_mean"])
    selective_edge = float(summary["edge_luma_excess_mean"])
    if raw_edge > 1e-6 and selective_edge > 0.50 * raw_edge:
        return {"decision": "SELECTIVE_BOUNDARY_FAIL", "abort": False}
    return {"decision": "SELECTIVE_PRETRAIN_PROMISING", "abort": False}


def v222_checkpoint_score(summary: dict[str, object]) -> float:
    final_ref_color_score = (
        float(summary["median_selective_to_ref_ab"])
        + 0.10 * float(summary["mean_reference_hue_error"])
        + 0.50 * float(summary["mean_reference_chroma_error"])
    )
    edge_halo_penalty = float(summary["edge_luma_excess_mean"])
    orthogonal_penalty = float(summary["median_orthogonal_error"])
    retention_penalty = max(
        0.0, 0.70 - float(summary["median_color_retention_ratio"])
    )
    return (
        final_ref_color_score
        + 1.5 * edge_halo_penalty
        + 0.5 * orthogonal_penalty
        + 2.0 * retention_penalty
    )
