import math
from typing import Iterable


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def build_normal_color_manifest(
    records: list[dict[str, object]],
    min_safe_reference_fraction: float,
    min_normal_fraction: float = 0.30,
) -> dict[str, object]:
    candidates = []
    for record in records:
        pseudo_reliable = (
            float(record["pseudo_to_reference_ab_error"]) <= 4.0
            and float(record["pseudo_to_reference_hue_error"]) <= 12.0
            and float(record["safe_fraction"]) >= min_safe_reference_fraction
        )
        near_no_edit = float(record["ref_base_ab_distance"]) < 3.0
        if pseudo_reliable and not near_no_edit:
            candidates.append(record)

    distance_cutoff = _percentile(
        (float(record["ref_base_ab_distance"]) for record in candidates), 0.85
    )
    chroma_cutoff = _percentile(
        (float(record["reference_chroma_magnitude"]) for record in candidates), 0.85
    )
    entries = []
    normal_count = 0
    for record in records:
        pseudo_reliable = (
            float(record["pseudo_to_reference_ab_error"]) <= 4.0
            and float(record["pseudo_to_reference_hue_error"]) <= 12.0
            and float(record["safe_fraction"]) >= min_safe_reference_fraction
        )
        near_no_edit = float(record["ref_base_ab_distance"]) < 3.0
        extreme = bool(
            candidates
            and pseudo_reliable
            and not near_no_edit
            and (
                float(record["ref_base_ab_distance"]) >= distance_cutoff
                or float(record["reference_chroma_magnitude"]) >= chroma_cutoff
            )
        )
        normal = pseudo_reliable and not near_no_edit and not extreme
        reasons = []
        if not pseudo_reliable:
            reasons.append("pseudo_unreliable")
        if near_no_edit:
            reasons.append("near_no_edit")
        if extreme:
            reasons.append("extreme_top15")
        if normal:
            reasons.append("normal_color")
            normal_count += 1
        color_weight = 1.0
        if not pseudo_reliable:
            color_weight = 0.0
        elif extreme:
            color_weight = 0.20
        elif near_no_edit:
            color_weight = 0.50
        entries.append({
            "sample_id": record["sample_id"],
            "pseudo_reliable": pseudo_reliable,
            "near_no_edit": near_no_edit,
            "extreme_for_v221": extreme,
            "normal_color": normal,
            "color_weight": color_weight,
            "reasons": reasons,
            "ref_base_ab_distance": float(record["ref_base_ab_distance"]),
            "reference_chroma_magnitude": float(record["reference_chroma_magnitude"]),
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
        "reliable_non_near_count": len(candidates),
        "normal_count": normal_count,
        "normal_fraction": normal_fraction,
        "min_normal_fraction": min_normal_fraction,
        "extreme_distance_cutoff": distance_cutoff,
        "extreme_chroma_cutoff": chroma_cutoff,
        "entries": entries,
    }


def _aggregate_records(records: list[dict[str, object]]) -> dict[str, float]:
    progress = [float(record["reference_progress"]) for record in records]
    improvement = [float(record["color_improvement_ratio"]) for record in records]
    final_ab = [float(record["final_to_reference_ab_error"]) for record in records]
    final_hue = [float(record["final_to_reference_hue_error"]) for record in records]
    final_chroma = [float(record["final_to_reference_chroma_error"]) for record in records]
    return {
        "count": len(records),
        "mean_final_to_reference_ab": _mean(final_ab),
        "median_final_to_reference_ab": _percentile(final_ab, 0.50),
        "mean_final_to_reference_hue": _mean(final_hue),
        "mean_final_to_reference_chroma": _mean(final_chroma),
        "mean_reference_progress": _mean(progress),
        "median_reference_progress": _percentile(progress, 0.50),
        "p25_reference_progress": _percentile(progress, 0.25),
        "p75_reference_progress": _percentile(progress, 0.75),
        "mean_color_improvement_ratio": _mean(improvement),
        "median_color_improvement_ratio": _percentile(improvement, 0.50),
        "p25_color_improvement_ratio": _percentile(improvement, 0.25),
        "negative_improvement_fraction": _mean(value < 0.0 for value in improvement),
        "edge_luma_excess_mean": _mean(
            float(record["edge_luma_excess_mean"]) for record in records
        ),
        "edge_luma_excess_fraction": _mean(
            float(record["edge_luma_excess_fraction"]) for record in records
        ),
        "outer_bg_keep_l1": _mean(float(record["outer_bg_keep_l1"]) for record in records),
        "face_keep_l1": _mean(float(record["face_keep_l1"]) for record in records),
        "global_frac_l_excess_gt12": _mean(
            float(record["global_frac_l_excess_gt12"]) for record in records
        ),
        "mean_pseudo_to_reference_ab": _mean(
            float(record["pseudo_to_reference_ab_error"]) for record in records
        ),
        "mean_pseudo_to_reference_hue": _mean(
            float(record["pseudo_to_reference_hue_error"]) for record in records
        ),
    }


def aggregate_fixed_alpha_metrics(
    records_by_alpha: dict[float, list[dict[str, object]]],
    manifest: dict[str, object],
) -> dict[str, object]:
    normal_ids = {
        entry["sample_id"] for entry in manifest["entries"] if entry["normal_color"]
    }
    table = {}
    for alpha, records in sorted(records_by_alpha.items()):
        normal_records = [record for record in records if record["sample_id"] in normal_ids]
        table[f"{alpha:.2f}"] = {
            "all_valid": _aggregate_records(records),
            "normal": _aggregate_records(normal_records),
        }

    alpha0 = table["0.00"]["normal"]
    margins = {
        "edge_luma_excess_mean": 0.50,
        "outer_bg_keep_l1": 0.02,
        "face_keep_l1": 0.02,
    }
    for key, metrics in table.items():
        normal = metrics["normal"]
        catastrophic_metrics = []
        for metric_name, small_margin in margins.items():
            threshold = alpha0[metric_name] + max(
                0.30 * abs(alpha0[metric_name]), small_margin
            )
            normal[f"{metric_name}_catastrophic_threshold"] = threshold
            if normal[metric_name] > threshold:
                catastrophic_metrics.append(metric_name)
            normal[f"delta_{metric_name}_vs_alpha0"] = (
                normal[metric_name] - alpha0[metric_name]
            )
        normal["delta_edge_luma_excess_vs_alpha0"] = normal[
            "delta_edge_luma_excess_mean_vs_alpha0"
        ]
        normal["catastrophic_artifact"] = bool(catastrophic_metrics)
        normal["catastrophic_metrics"] = catastrophic_metrics

    ordered_alpha = sorted(records_by_alpha)
    comparable_alpha = [alpha for alpha in ordered_alpha if alpha > 0.0]
    by_sample = {
        record["sample_id"]: {} for record in records_by_alpha[comparable_alpha[0]]
        if record["sample_id"] in normal_ids
    }
    for alpha in comparable_alpha:
        for record in records_by_alpha[alpha]:
            if record["sample_id"] in by_sample:
                by_sample[record["sample_id"]][alpha] = float(record["reference_progress"])
    monotonic_checks = []
    for values in by_sample.values():
        for lower, upper in zip(comparable_alpha, comparable_alpha[1:]):
            if lower in values and upper in values:
                monotonic_checks.append(values[upper] >= values[lower] - 1e-6)
    return {
        "alphas": table,
        "artifact_small_margins": margins,
        "monotonic_progress_fraction": _mean(monotonic_checks),
        "monotonic_warning": _mean(monotonic_checks) < 0.60,
    }


def choose_direct_anchor_feasibility(
    summary: dict[str, object], manifest: dict[str, object]
) -> dict[str, object]:
    alpha1 = summary["alphas"]["1.00"]["normal"]
    common = {
        "alpha1_progress": alpha1["median_reference_progress"],
        "alpha1_improvement_ratio": alpha1["median_color_improvement_ratio"],
        "pseudo_to_reference_ab": alpha1["mean_pseudo_to_reference_ab"],
        "pseudo_to_reference_hue": alpha1["mean_pseudo_to_reference_hue"],
        "monotonic_progress_fraction": summary["monotonic_progress_fraction"],
        "warnings": (
            ["DIRECT_ANCHOR_NON_MONOTONIC"] if summary["monotonic_warning"] else []
        ),
    }
    if manifest["status"] != "OK":
        return {
            "decision": manifest["status"],
            "alpha_star": None,
            "reason": "normal diagnostic subset is smaller than 30% of validation",
            **common,
        }

    metrics_by_alpha = summary["alphas"]
    diagnostic = [metrics_by_alpha[f"{alpha:.2f}"]["normal"] for alpha in (0.8, 0.9, 1.0)]
    strong = [
        metrics for metrics in diagnostic
        if metrics["median_reference_progress"] >= 0.70
        and metrics["p25_reference_progress"] >= 0.45
        and metrics["median_color_improvement_ratio"] >= 0.60
        and metrics["negative_improvement_fraction"] <= 0.10
        and not metrics["catastrophic_artifact"]
    ]
    weak = [
        metrics for metrics in diagnostic
        if metrics["median_reference_progress"] >= 0.55
        and metrics["median_color_improvement_ratio"] >= 0.45
        and metrics["negative_improvement_fraction"] <= 0.20
        and not metrics["catastrophic_artifact"]
    ]
    decision = "PASS_STRONG" if strong else "PASS_WEAK" if weak else "FAIL_DIRECTION"
    if decision == "FAIL_DIRECTION":
        return {
            "decision": decision,
            "alpha_star": None,
            "reason": "no alpha in [0.8, 0.9, 1.0] satisfies the normal-color feasibility gate",
            **common,
        }

    candidates = []
    for alpha in (0.7, 0.8, 0.9, 1.0):
        metrics = metrics_by_alpha[f"{alpha:.2f}"]["normal"]
        if not metrics["catastrophic_artifact"]:
            candidates.append((alpha, metrics))
    best_improvement = max(metrics["median_color_improvement_ratio"] for _, metrics in candidates)
    near_best = [
        (alpha, metrics) for alpha, metrics in candidates
        if best_improvement - metrics["median_color_improvement_ratio"] < 0.03
    ]
    alpha_star, baseline = min(near_best, key=lambda item: item[0])
    return {
        "decision": decision,
        "alpha_star": alpha_star,
        "reason": (
            f"alpha={alpha_star:.2f} is the smallest non-catastrophic candidate within "
            "0.03 of the best median color improvement"
        ),
        "phase0_alpha_star_metrics": baseline,
        **common,
    }


def assert_finite_json(value, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON value at {path}: {value}")


def should_start_phase1(decision: dict[str, object]) -> bool:
    return decision.get("decision") in ("PASS_STRONG", "PASS_WEAK")


def evaluate_v221_checkpoint_gate(
    median_reference_progress: float,
    median_color_improvement_ratio: float,
    effective_alpha_mean: float,
    alpha_star: float,
    phase0_alpha_star_metrics: dict[str, object],
) -> dict[str, float | bool]:
    phase0_progress = float(
        phase0_alpha_star_metrics.get("median_reference_progress", 0.0)
    )
    phase0_improvement = float(
        phase0_alpha_star_metrics.get("median_color_improvement_ratio", 0.0)
    )
    progress_gate = max(0.60, 0.90 * phase0_progress)
    improvement_gate = max(0.45, 0.90 * phase0_improvement)
    anchor_drift_warning = effective_alpha_mean < alpha_star - 0.04
    training_regressed = (
        median_color_improvement_ratio < 0.95 * phase0_improvement
    )
    return {
        "color_progress_gate": progress_gate,
        "color_improvement_gate": improvement_gate,
        "anchor_drift_warning": anchor_drift_warning,
        "training_regressed_from_fixed_anchor": training_regressed,
        "color_gate_pass": bool(
            median_reference_progress >= progress_gate
            and median_color_improvement_ratio >= improvement_gate
            and not anchor_drift_warning
            and not training_regressed
        ),
    }
