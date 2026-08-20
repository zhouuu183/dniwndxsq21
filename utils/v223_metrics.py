"""Full-color diagnostics and acceptance gates for Blending V8 v2.23."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def ensure_scalar_map(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3:
        value = value.unsqueeze(1)
    expected = (reference.size(0), 1, reference.size(2), reference.size(3))
    if tuple(value.shape) != expected:
        raise ValueError(f"scalar map must have shape {expected}, got {tuple(value.shape)}")
    return value


def strict_masked_mean_per_sample(
    value: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    value = ensure_scalar_map(value, mask)
    if value.shape != mask.shape:
        raise ValueError(f"value and mask shapes must match, got {value.shape}, {mask.shape}")
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (value * mask).flatten(1).sum(dim=1) / denominator


def full_stat_error(mean_ab_error: torch.Tensor, median_l_error: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mean_ab_error.square() + median_l_error.square())


def build_v223_normal_color_manifest(
    records: list[dict[str, object]],
    min_safe_reference_fraction: float,
    min_normal_fraction: float = 0.30,
) -> dict[str, object]:
    """Select ordinary colors using both chroma and full-color tone distance."""
    reliable = []
    for record in records:
        pseudo_reliable = (
            float(record["pseudo_to_reference_ab_error"]) <= 4.0
            and float(record["pseudo_to_reference_hue_error"]) <= 12.0
            and float(record["safe_fraction"]) >= min_safe_reference_fraction
        )
        near_no_edit = (
            float(record["ref_base_ab_distance"]) < 3.0
            and float(record["ref_base_l_distance"]) < 3.0
        )
        if pseudo_reliable and not near_no_edit:
            reliable.append(record)
    full_cutoff = _quantile(
        (record["base_full_stat_error"] for record in reliable), 0.85
    )
    chroma_cutoff = _quantile(
        (record["reference_chroma_magnitude"] for record in reliable), 0.85
    )
    entries = []
    normal_count = 0
    for record in records:
        pseudo_reliable = (
            float(record["pseudo_to_reference_ab_error"]) <= 4.0
            and float(record["pseudo_to_reference_hue_error"]) <= 12.0
            and float(record["safe_fraction"]) >= min_safe_reference_fraction
        )
        near_no_edit = (
            float(record["ref_base_ab_distance"]) < 3.0
            and float(record["ref_base_l_distance"]) < 3.0
        )
        extreme = bool(
            reliable
            and pseudo_reliable
            and not near_no_edit
            and (
                float(record["base_full_stat_error"]) >= full_cutoff
                or float(record["reference_chroma_magnitude"]) >= chroma_cutoff
            )
        )
        normal = pseudo_reliable and not near_no_edit and not extreme
        reasons = []
        if not pseudo_reliable:
            reasons.append("pseudo_unreliable")
        if near_no_edit:
            reasons.append("near_no_edit_full_color")
        if extreme:
            reasons.append("extreme_top15_full_color")
        if normal:
            reasons.append("normal_color")
            normal_count += 1
        entries.append({
            "sample_id": record["sample_id"],
            "pseudo_reliable": pseudo_reliable,
            "near_no_edit": near_no_edit,
            "extreme_for_v223": extreme,
            "normal_color": normal,
            "color_weight": (
                0.0
                if (not pseudo_reliable or extreme)
                else 0.25
                if near_no_edit
                else 1.0
            ),
            "reasons": reasons,
            "ref_base_ab_distance": float(record["ref_base_ab_distance"]),
            "ref_base_l_distance": float(record["ref_base_l_distance"]),
            "base_full_stat_error": float(record["base_full_stat_error"]),
            "reference_chroma_magnitude": float(
                record["reference_chroma_magnitude"]
            ),
        })
    total = len(records)
    normal_fraction = normal_count / max(total, 1)
    return {
        "status": (
            "OK"
            if normal_fraction >= min_normal_fraction
            else "INSUFFICIENT_NORMAL_DIAGNOSTIC_SAMPLES"
        ),
        "total_count": total,
        "reliable_non_near_count": len(reliable),
        "normal_count": normal_count,
        "normal_fraction": normal_fraction,
        "min_normal_fraction": min_normal_fraction,
        "extreme_full_stat_cutoff": full_cutoff,
        "extreme_chroma_cutoff": chroma_cutoff,
        "entries": entries,
    }


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


def aggregate_v223_records(
    records: list[dict[str, object]],
    normal_ids: set[str] | None = None,
) -> dict[str, float | int | bool]:
    selected = records
    if normal_ids is not None:
        selected = [record for record in records if record["sample_id"] in normal_ids]
    if not selected:
        return {"count": 0, "has_samples": False}

    full_retention = [
        float(record["full_color_retention"])
        for record in selected
        if bool(record["raw_full_anchor_positive_gain"])
    ]
    ab_retention = [
        float(record["ab_retention"])
        for record in selected
        if bool(record["raw_anchor_positive_gain"])
    ]
    l_progress = [
        float(record["l_progress"])
        for record in selected
        if bool(record["l_progress_valid"])
    ]
    return {
        "count": len(selected),
        "has_samples": True,
        "median_base_ref_l_error": _quantile(
            (record["base_ref_l_error"] for record in selected), 0.50
        ),
        "median_anchor_ref_l_error": _quantile(
            (record["anchor_ref_l_error"] for record in selected), 0.50
        ),
        "median_selective_ref_l_error": _quantile(
            (record["selective_ref_l_error"] for record in selected), 0.50
        ),
        "median_base_ref_ab_error": _quantile(
            (record["base_ref_ab_error"] for record in selected), 0.50
        ),
        "median_anchor_ref_ab_error": _quantile(
            (record["anchor_ref_ab_error"] for record in selected), 0.50
        ),
        "median_base_full_stat_error": _quantile(
            (record["base_full_stat_error"] for record in selected), 0.50
        ),
        "median_anchor_full_stat_error": _quantile(
            (record["anchor_full_stat_error"] for record in selected), 0.50
        ),
        "median_selective_full_stat_error": _quantile(
            (record["selective_full_stat_error"] for record in selected), 0.50
        ),
        "median_pseudo_deltaE76": _quantile(
            (record["pseudo_deltaE76"] for record in selected), 0.50
        ),
        "median_color_retention": _quantile(full_retention, 0.50),
        "median_full_color_retention": _quantile(full_retention, 0.50),
        "median_ab_retention": _quantile(ab_retention, 0.50),
        "median_l_progress": _quantile(l_progress, 0.50),
        "p25_l_progress": _quantile(l_progress, 0.25),
        "positive_full_gain_count": len(full_retention),
        "positive_ab_gain_count": len(ab_retention),
        "meaningful_l_count": len(l_progress),
        "median_selective_ref_ab_error": _quantile(
            (record["selective_ref_ab_error"] for record in selected), 0.50
        ),
        "mean_reference_hue_error": _mean(
            record["selective_ref_hue_error"] for record in selected
        ),
        "mean_reference_chroma_error": _mean(
            record["selective_ref_chroma_error"] for record in selected
        ),
        "negative_full_color_improvement_fraction": _mean(
            float(record["full_color_improvement_ratio"]) < 0.0
            for record in selected
        ),
        "raw_anchor_edge_luma_excess_mean": _mean(
            record["raw_anchor_edge_luma_excess_mean"] for record in selected
        ),
        "edge_luma_excess_mean": _mean(
            record["edge_luma_excess_mean"] for record in selected
        ),
        "edge_hf_excess": _mean(record["edge_hf_excess"] for record in selected),
        "median_edge_transfer_ratio": _quantile(
            (record["edge_transfer_ratio"] for record in selected), 0.50
        ),
        "face_keep_l1": _mean(record["face_keep_l1"] for record in selected),
        "outer_bg_keep_l1": _mean(record["outer_bg_keep_l1"] for record in selected),
        "outside_hair_max_abs_delta": max(
            float(record["outside_hair_max_abs_delta"]) for record in selected
        ),
        "hard_protect_max_abs_delta": max(
            float(record["hard_protect_max_abs_delta"]) for record in selected
        ),
        "mean_luma_transfer_weight": _mean(
            record["mean_luma_transfer_weight"] for record in selected
        ),
        "mean_chroma_transfer_weight": _mean(
            record["mean_chroma_transfer_weight"] for record in selected
        ),
    }


def classify_v223(summary: dict[str, object]) -> dict[str, object]:
    if int(summary.get("count", 0)) == 0:
        return {"decision": "V223_FULL_COLOR_DIRECTION_FAIL", "abort": True}
    if (
        float(summary["outside_hair_max_abs_delta"]) > 5e-4
        or float(summary["hard_protect_max_abs_delta"]) > 5e-4
    ):
        return {"decision": "V223_MASK_PROTECTION_BUG", "abort": True}
    if int(summary["positive_full_gain_count"]) == 0:
        return {"decision": "V223_FULL_COLOR_DIRECTION_FAIL", "abort": True}
    if float(summary["median_selective_ref_l_error"]) > (
        0.65 * float(summary["median_base_ref_l_error"])
    ):
        return {"decision": "V223_LUMA_RETENTION_FAIL", "abort": False}
    if float(summary["median_full_color_retention"]) < 0.75:
        return {"decision": "V223_LUMA_RETENTION_FAIL", "abort": False}
    if float(summary["median_ab_retention"]) < 0.75:
        return {"decision": "V223_CHROMA_RETENTION_FAIL", "abort": False}
    if (
        float(summary["negative_full_color_improvement_fraction"]) > 0.15
    ):
        return {"decision": "V223_FULL_COLOR_DIRECTION_FAIL", "abort": False}
    raw_edge = float(summary["raw_anchor_edge_luma_excess_mean"])
    if raw_edge > 1e-6 and float(summary["edge_luma_excess_mean"]) > 0.60 * raw_edge:
        return {"decision": "V223_HALO_FAIL", "abort": False}
    if float(summary["median_edge_transfer_ratio"]) < 0.30:
        return {"decision": "V223_BOUNDARY_COLOR_COLLAPSE", "abort": False}
    if int(summary["meaningful_l_count"]) > 0 and (
        float(summary["median_l_progress"]) < 0.65
        or float(summary["p25_l_progress"]) < 0.35
    ):
        return {"decision": "V223_LUMA_RETENTION_FAIL", "abort": False}
    return {"decision": "V223_FULL_COLOR_PASS", "abort": False}


def v223_checkpoint_score(summary: dict[str, object]) -> float:
    return (
        1.0 * float(summary["median_selective_full_stat_error"])
        + 0.5 * float(summary["median_pseudo_deltaE76"])
        + 0.5 * float(summary["median_selective_ref_l_error"])
        + 0.10 * float(summary["mean_reference_hue_error"])
        + 0.25 * float(summary["mean_reference_chroma_error"])
        + 1.0 * float(summary["edge_luma_excess_mean"])
        + 0.5 * float(summary["edge_hf_excess"])
        + 2.0 * max(0.0, 0.75 - float(summary["median_full_color_retention"]))
        + 1.0 * max(0.0, 0.30 - float(summary["median_edge_transfer_ratio"]))
    )
