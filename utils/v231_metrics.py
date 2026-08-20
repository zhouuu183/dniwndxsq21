"""Matte, color, edge, and ordered multi-gate reporting for V2.31."""

from __future__ import annotations

import math
from statistics import median

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.color_condition_v8 import compute_reference_fidelity_metrics
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from utils.v230_metrics import masked_mean
from utils.v229_metrics import base_color_rim_metrics


def _highpass(value: torch.Tensor, radius: int = 2) -> torch.Tensor:
    return value - gaussian_blur_v829(value, radius)


def _match(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return value if value.shape[-2:] == size else F.interpolate(
        value, size=size, mode="bilinear", align_corners=False
    )


def matte_metric_tensors(
    *, alpha_hr: torch.Tensor, coarse_hair: torch.Tensor, unknown: torch.Tensor,
    sure_fg: torch.Tensor, sure_bg: torch.Tensor, source_face_hr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    fractional = ((alpha_hr > 0.01) & (alpha_hr < 0.99)).to(alpha_hr.dtype)
    alpha_dx = F.pad(alpha_hr[..., 1:] - alpha_hr[..., :-1], (0, 1, 0, 0)).abs()
    coarse_dx = F.pad(coarse_hair[..., 1:] - coarse_hair[..., :-1], (0, 1, 0, 0)).abs()
    return {
        "sure_fg_error": (alpha_hr.sub(1).abs() * sure_fg).flatten(1).amax(dim=1),
        "sure_bg_error": (alpha_hr.abs() * sure_bg).flatten(1).amax(dim=1),
        "unknown_fractional_fraction": masked_mean(fractional, unknown),
        "alpha_unknown_hf": masked_mean(_highpass(alpha_hr).abs(), unknown),
        "alpha_boundary_deviation": masked_mean((alpha_hr - coarse_hair).abs(), unknown),
        "alpha_gradient_energy": masked_mean(alpha_dx, unknown),
        "coarse_gradient_energy": masked_mean(coarse_dx, unknown),
        "face_false_positive_alpha": masked_mean(
            alpha_hr, source_face_hr * unknown * (1.0 - (coarse_hair >= 0.5).to(alpha_hr.dtype))
        ),
    }


def direct_reference_color_metrics_v831(
    image_rgb: torch.Tensor,
    core: torch.Tensor,
    ref_stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    values = compute_reference_fidelity_metrics(rgb_to_lab(image_rgb), core, ref_stats)
    return {
        "direct_ref_ab_stat_error": values["mean_ab_error"],
        "direct_ref_l_stat_error": values["median_l_error"],
        "direct_ref_hue_error": values["hue_error"],
        "direct_ref_chroma_error": values["chroma_error"],
    }


def v231_metric_tensors(
    *, base_rgb: torch.Tensor, v226_rgb: torch.Tensor, v230_rgb: torch.Tensor,
    pp_rgb: torch.Tensor, phase_b_rgb: torch.Tensor, phase_c_rgb: torch.Tensor,
    target_lab: torch.Tensor, alpha_hr: torch.Tensor, source_face_mask: torch.Tensor,
    source_subject_mask: torch.Tensor, ref_stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    size = base_rgb.shape[-2:]
    pp = _match(pp_rgb, size)
    v230 = _match(v230_rgb, size)
    phase_b = _match(phase_b_rgb, size)
    phase_c = _match(phase_c_rgb, size)
    alpha = _match(alpha_hr, size)
    face = _match(source_face_mask, size)
    subject = _match(source_subject_mask, size)
    core = (alpha >= 0.98).to(alpha.dtype)
    edge = ((alpha > 0.02) & (alpha < 0.98)).to(alpha.dtype)
    face_contact = edge * face
    scene_contact = edge * (1.0 - subject)
    phase_b_lab = rgb_to_lab(phase_b)
    phase_c_lab = rgb_to_lab(phase_c)
    target_lab = _match(target_lab, size)
    v226_lab = rgb_to_lab(v226_rgb)
    result = base_color_rim_metrics(base_rgb, phase_c, core, edge)
    result.update({
        f"phase_b_{key}": value
        for key, value in base_color_rim_metrics(base_rgb, phase_b, core, edge).items()
    })
    result.update({
        "phase_b_core_ref_full_pseudo": masked_mean(torch.linalg.vector_norm(phase_b_lab - target_lab, dim=1, keepdim=True), core),
        "phase_b_core_ref_ab_pseudo": masked_mean(torch.linalg.vector_norm(phase_b_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True), core),
        "core_ref_full_pseudo": masked_mean(torch.linalg.vector_norm(phase_c_lab - target_lab, dim=1, keepdim=True), core),
        "core_ref_ab_pseudo": masked_mean(torch.linalg.vector_norm(phase_c_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True), core),
        "v226_core_ref_full_pseudo": masked_mean(torch.linalg.vector_norm(v226_lab - target_lab, dim=1, keepdim=True), core),
        "v226_core_ref_ab_pseudo": masked_mean(torch.linalg.vector_norm(v226_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True), core),
        "phase_b_face_rgb": masked_mean((phase_b - base_rgb).abs(), face_contact),
        "phase_c_face_rgb": masked_mean((phase_c - base_rgb).abs(), face_contact),
        "phase_c_face_de": masked_mean(torch.linalg.vector_norm(phase_c_lab - rgb_to_lab(base_rgb), dim=1, keepdim=True), face_contact),
        "background_retention": masked_mean((phase_c - base_rgb).abs(), scene_contact) / masked_mean((v226_rgb - base_rgb).abs(), scene_contact).clamp_min(1e-6),
        "pp_background_retention": masked_mean((pp - base_rgb).abs(), scene_contact) / masked_mean((v226_rgb - base_rgb).abs(), scene_contact).clamp_min(1e-6),
        "contour_hf_ab": masked_mean(_highpass(phase_c_lab[:, 1:]).abs(), edge),
        "phase_b_contour_hf_ab": masked_mean(_highpass(phase_b_lab[:, 1:]).abs(), edge),
        "v230_contour_hf_ab": masked_mean(_highpass(rgb_to_lab(v230)[:, 1:]).abs(), edge),
        "core_pp_hf_drift": masked_mean((_highpass(phase_c) - _highpass(pp)).abs(), core),
        "edge_pp_hf_drift": masked_mean((_highpass(phase_c) - _highpass(pp)).abs(), edge),
        "phase_b_core_pp_hf_drift": masked_mean((_highpass(phase_b) - _highpass(pp)).abs(), core),
        "phase_b_edge_pp_hf_drift": masked_mean((_highpass(phase_b) - _highpass(pp)).abs(), edge),
    })
    result.update({f"phase_b_{key}": value for key, value in direct_reference_color_metrics_v831(phase_b, core, ref_stats).items()})
    result.update({f"phase_c_{key}": value for key, value in direct_reference_color_metrics_v831(phase_c, core, ref_stats).items()})
    result.update({f"v226_{key}": value for key, value in direct_reference_color_metrics_v831(v226_rgb, core, ref_stats).items()})
    return result


def aggregate_v231(records: list[dict[str, object]]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"count": len(records)}
    keys = sorted({key for record in records for key, value in record.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    for key in keys:
        values = [float(item[key]) for item in records if key in item and math.isfinite(float(item[key]))]
        if not values:
            continue
        ordered = sorted(values)
        summary[f"median_{key}"] = float(median(values))
        summary[f"mean_{key}"] = float(sum(values) / len(values))
        summary[f"p90_{key}"] = float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))])
        summary[f"max_{key}"] = float(max(values))
    return summary


def classify_matte_v231(summary: dict[str, float | int]) -> dict[str, object]:
    failed = []
    if float(summary.get("max_sure_fg_error", 1.0)) > 1e-6 or float(summary.get("max_sure_bg_error", 1.0)) > 1e-6:
        failed.append("V231_MATTING_BACKEND_FAIL")
    if float(summary.get("median_unknown_fractional_fraction", 0.0)) < 0.05 or float(summary.get("median_alpha_boundary_deviation", 0.0)) < 1e-3:
        failed.append("V231_MATTE_NOT_REFINING")
    if float(summary.get("p90_face_false_positive_alpha", 1.0)) > 0.35:
        failed.append("V231_MATTE_FACE_FP")
    decision = failed[0] if failed else "V231_MATTE_READY"
    return {"failed_gates": failed, "primary_failure": failed[0] if failed else None, "automatic_decision": decision}


def classify_v231(summary: dict[str, float | int], *, parity_rgb: float, parity_alpha: float) -> dict[str, object]:
    failed = []
    if parity_rgb > 1e-5 or parity_alpha > 1e-6:
        failed.append("V231_RUNTIME_PARITY_FAIL")
    if (float(summary.get("median_core_ref_full_pseudo", 1e9)) > 1.015 * float(summary.get("median_v226_core_ref_full_pseudo", 0.0)) or
            float(summary.get("median_core_ref_ab_pseudo", 1e9)) > 1.015 * float(summary.get("median_v226_core_ref_ab_pseudo", 0.0))):
        failed.append("V231_CORE_COLOR_FAIL")
    if float(summary.get("median_phase_c_direct_ref_ab_stat_error", 1e9)) > 1.03 * float(summary.get("median_v226_direct_ref_ab_stat_error", 0.0)):
        failed.append("V231_DIRECT_REFERENCE_COLOR_FAIL")
    if (float(summary.get("median_base_color_rim_fraction", 1.0)) > 0.035 or float(summary.get("p90_base_color_rim_fraction", 1.0)) > 0.075):
        failed.append("V231_BASE_RIM_FAIL")
    if float(summary.get("median_phase_c_face_rgb", 1.0)) > 0.05 or float(summary.get("p90_phase_c_face_rgb", 1.0)) > 0.085:
        failed.append("V231_FACE_CONTACT_FAIL")
    if float(summary.get("median_contour_hf_ab", 1.0)) > 0.60 * float(summary.get("median_v230_contour_hf_ab", 0.0)):
        failed.append("V231_MICRO_JAGGED_FAIL")
    if (float(summary.get("median_background_retention", 1.0)) > 0.30 or float(summary.get("median_background_retention", 1.0)) > 1.10 * float(summary.get("median_pp_background_retention", 0.0))):
        failed.append("V231_BACKGROUND_RETENTION_FAIL")
    decision = failed[0] if failed else "V231_READY_FOR_VISUAL_REVIEW"
    return {"failed_gates": failed, "primary_failure": failed[0] if failed else None, "automatic_decision": decision}


__all__ = ["aggregate_v231", "classify_matte_v231", "classify_v231", "direct_reference_color_metrics_v831", "matte_metric_tensors", "v231_metric_tensors"]
