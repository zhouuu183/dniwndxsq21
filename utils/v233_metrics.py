"""Independent ownership, color, artifact, and acceptance metrics for V2.33."""

from __future__ import annotations

import math
from collections import deque
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.color_condition_v8 import compute_reference_fidelity_metrics
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829
from utils.v229_metrics import base_color_rim_metrics


def _match(value: torch.Tensor, size: tuple[int, int], *, mask: bool = False) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    return F.interpolate(value.float(), size=size, mode="nearest" if mask else "bilinear",
                         align_corners=None if mask else False)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(value).to(value.dtype)
    return (value * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(1.0)


def masked_p90(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rows = []
    for sample, support in zip(value, mask):
        selected = sample.flatten()[support.expand_as(sample).flatten() > 0.5]
        rows.append(torch.quantile(selected, 0.9) if selected.numel() else sample.new_tensor(0.0))
    return torch.stack(rows)


def _component_stats(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts, maximums = [], []
    for sample in mask.detach().cpu().numpy()[:, 0] > 0.5:
        seen = np.zeros_like(sample, dtype=bool)
        areas = []
        for y, x in zip(*np.nonzero(sample)):
            if seen[y, x]:
                continue
            seen[y, x] = True
            queue = deque([(int(y), int(x))])
            area = 0
            while queue:
                cy, cx = queue.popleft(); area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < sample.shape[0] and 0 <= nx < sample.shape[1] and sample[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; queue.append((ny, nx))
            areas.append(area)
        counts.append(float(len(areas))); maximums.append(float(max(areas, default=0)))
    device = mask.device
    return torch.tensor(counts, device=device), torch.tensor(maximums, device=device)


def _progress(base_lab: torch.Tensor, candidate_lab: torch.Tensor,
              target_lab: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    before = torch.linalg.vector_norm(base_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True)
    after = torch.linalg.vector_norm(candidate_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True)
    return masked_mean(1.0 - after / before.clamp_min(1.0), mask)


def _correlation(alpha: torch.Tensor, progress: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rows = []
    for a, p, m in zip(alpha, progress, mask):
        selected = m.flatten() > 0.5
        av, pv = a.flatten()[selected], p.flatten()[selected]
        if av.numel() < 2:
            rows.append(a.new_tensor(0.0)); continue
        av, pv = av - av.mean(), pv - pv.mean()
        rows.append((av * pv).sum().abs() / (av.square().sum().sqrt() * pv.square().sum().sqrt()).clamp_min(1e-8))
    return torch.stack(rows)


def v233_metric_tensors(*, base_rgb: torch.Tensor, v226_rgb: torch.Tensor,
                        pp_rgb: torch.Tensor, output_rgb: torch.Tensor,
                        foreground_pp_rgb: torch.Tensor, foreground_target_rgb: torch.Tensor,
                        alpha: torch.Tensor, alpha_eff: torch.Tensor,
                        foreground_confidence: torch.Tensor, background_confidence: torch.Tensor,
                        sure_fg: torch.Tensor, sure_bg: torch.Tensor,
                        face_unknown_all: torch.Tensor, face_contact_actual: torch.Tensor,
                        false_positive_proxy: torch.Tensor, source_face_mask: torch.Tensor,
                        source_subject_mask: torch.Tensor, target_lab: torch.Tensor,
                        ref_stats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    size = base_rgb.shape[-2:]
    pp = _match(pp_rgb, size); output = _match(output_rgb, size)
    f_pp = _match(foreground_pp_rgb, size); f_target = _match(foreground_target_rgb, size)
    alpha = _match(alpha, size); alpha_eff = _match(alpha_eff, size)
    fg_conf = _match(foreground_confidence, size); bg_conf = _match(background_confidence, size)
    sure_fg = _match(sure_fg, size, mask=True); sure_bg = _match(sure_bg, size, mask=True)
    face_unknown = _match(face_unknown_all, size); contact = _match(face_contact_actual, size)
    fp_proxy = _match(false_positive_proxy, size); face = _match(source_face_mask, size, mask=True)
    subject = _match(source_subject_mask, size, mask=True)
    target_lab = _match(target_lab, size)
    base_lab, v226_lab, pp_lab = rgb_to_lab(base_rgb), rgb_to_lab(v226_rgb), rgb_to_lab(pp)
    out_lab, fpp_lab, ft_lab = rgb_to_lab(output), rgb_to_lab(f_pp), rgb_to_lab(f_target)
    core = (alpha_eff >= 0.90).to(alpha.dtype)
    edge = ((alpha_eff > 0.01) & (alpha_eff < 0.90)).to(alpha.dtype)
    scene_edge = edge * (1.0 - subject)

    result = base_color_rim_metrics(base_rgb, output, core, edge)
    result.update({
        "foreground_conf_mean": masked_mean(fg_conf, (alpha > 0.01).float()),
        "foreground_conf_low_alpha": masked_mean(fg_conf, ((alpha > 0.01) & (alpha <= 0.15)).float()),
        "foreground_conf_high_alpha": masked_mean(fg_conf, (alpha >= 0.70).float()),
        "background_conf_mean": masked_mean(bg_conf, (alpha < 0.99).float()),
        "foreground_target_progress": _progress(fpp_lab, ft_lab, v226_lab, (alpha > 0.01).float()),
        "edge_foreground_progress": _progress(fpp_lab, ft_lab, v226_lab, edge),
        "core_ref_full": masked_mean(torch.linalg.vector_norm(out_lab - target_lab, dim=1, keepdim=True), core),
        "core_ref_ab": masked_mean(torch.linalg.vector_norm(out_lab[:, 1:] - target_lab[:, 1:], dim=1, keepdim=True), core),
        "face_contact_alpha": masked_mean(alpha_eff, contact),
        "face_contact_alpha_original": masked_mean(alpha, contact),
        "false_positive_alpha": masked_mean(alpha_eff, fp_proxy),
        "false_positive_alpha_original": masked_mean(alpha, fp_proxy),
        "sure_fg_error": ((alpha_eff - 1).abs() * sure_fg).flatten(1).amax(1),
        "sure_bg_error": (alpha_eff.abs() * sure_bg).flatten(1).amax(1),
        "face_contact_rgb": masked_mean((output - base_rgb).abs(), contact),
        "face_contact_rgb_p90": masked_p90((output - base_rgb).abs().mean(1, keepdim=True), contact),
        "face_contact_de": masked_mean(torch.linalg.vector_norm(out_lab - base_lab, dim=1, keepdim=True), contact),
        "face_unknown_alpha": masked_mean(alpha_eff, face_unknown),
        "background_retention": masked_mean((output - base_rgb).abs(), scene_edge) / masked_mean((v226_rgb - base_rgb).abs(), scene_edge).clamp_min(1e-6),
        "background_pp_drift": masked_mean((output - pp).abs(), scene_edge),
        "contour_hf_l": masked_mean((out_lab[:, :1] - gaussian_blur_v829(out_lab[:, :1], 2)).abs(), edge),
        "contour_hf_ab": masked_mean((out_lab[:, 1:] - gaussian_blur_v829(out_lab[:, 1:], 2)).abs(), edge),
    })
    direct = compute_reference_fidelity_metrics(out_lab, core, ref_stats)
    result.update({
        "direct_ref_ab": direct["mean_ab_error"], "direct_ref_l": direct["median_l_error"],
        "direct_ref_hue": direct["hue_error"], "direct_ref_chroma": direct["chroma_error"],
    })
    before = torch.linalg.vector_norm(fpp_lab[:, 1:] - v226_lab[:, 1:], dim=1, keepdim=True)
    after = torch.linalg.vector_norm(ft_lab[:, 1:] - v226_lab[:, 1:], dim=1, keepdim=True)
    progress_map = 1.0 - after / before.clamp_min(1.0)
    bin_values = []
    for index, (low, high) in enumerate(((0.05, .15), (.15, .30), (.30, .50), (.50, .70), (.70, .90), (.90, 1.01))):
        support = ((alpha >= low) & (alpha < high)).to(alpha.dtype)
        value = masked_mean(progress_map, support)
        result[f"alpha_bin_{index}_foreground_progress"] = value
        bin_values.append(value)
    result["alpha_bin_progress_std"] = torch.stack(bin_values, dim=1).std(dim=1, unbiased=False)
    result["alpha_color_progress_correlation"] = _correlation(alpha, progress_map, (alpha > 0.01).float())

    meaningful = (torch.linalg.vector_norm(v226_lab[:, 1:] - base_lab[:, 1:], dim=1, keepdim=True) >= 3.0).float()
    base_distance = torch.linalg.vector_norm(out_lab[:, 1:] - base_lab[:, 1:], dim=1, keepdim=True)
    target_distance = torch.linalg.vector_norm(out_lab[:, 1:] - v226_lab[:, 1:], dim=1, keepdim=True)
    patch = (target_distance > base_distance).float() * (alpha_eff >= 0.2).float() * meaningful
    patch_count, patch_max = _component_stats(patch)
    result.update({
        "original_color_patch_fraction": masked_mean(patch, (alpha_eff >= 0.2).float() * meaningful),
        "original_color_patch_count": patch_count,
        "max_original_color_patch_area": patch_max,
    })
    strip_error = (output - base_rgb).abs().mean(1, keepdim=True) * contact
    strip_bad = (strip_error > 0.06).float() * contact
    strip_count, strip_max = _component_stats(strip_bad)
    result.update({"face_strip_component_count": strip_count, "face_strip_max_area": strip_max})
    l_hf = out_lab[:, :1] - gaussian_blur_v829(out_lab[:, :1], 2)
    local_chroma = torch.linalg.vector_norm(out_lab[:, 1:], dim=1, keepdim=True)
    nearby_chroma = gaussian_blur_v829(local_chroma, 4)
    white_net = (l_hf > 4.0).float() * (local_chroma + 3.0 < nearby_chroma).float() * (alpha_eff >= 0.5).float()
    white_count, white_max = _component_stats(white_net)
    result.update({
        "white_net_fraction": masked_mean(white_net, (alpha_eff >= 0.5).float()),
        "white_net_count": white_count, "white_net_max_area": white_max,
        "foreground_clip_fraction": ((f_target <= 1e-5) | (f_target >= 1 - 1e-5)).float().flatten(1).mean(1),
    })
    return result


def aggregate_v233(records: list[dict[str, object]]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"count": len(records)}
    keys = sorted({k for row in records for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for key in keys:
        values = [float(row[key]) for row in records if key in row and math.isfinite(float(row[key]))]
        if values:
            ordered = sorted(values)
            summary[f"median_{key}"] = float(median(values))
            summary[f"p90_{key}"] = float(ordered[min(len(ordered) - 1, int(.9 * len(ordered)))])
            summary[f"max_{key}"] = float(max(values))
    return summary


AUDIT_KEYS = {
    "fb_reconstruction_audit.json": ("reconstruction", "rgb_mae", "rgb_max"),
    "fb_confidence_audit.json": ("foreground_conf", "background_conf"),
    "foreground_color_audit.json": ("foreground_target", "alpha_bin"),
    "direct_reference_color_audit.json": ("direct_ref", "core_ref"),
    "edge_foreground_progress_audit.json": ("edge_foreground", "base_color_edge"),
    "alpha_color_correlation_audit.json": ("correlation", "alpha_bin_progress_std"),
    "face_alpha_calibration_audit.json": ("alpha", "sure_fg", "sure_bg"),
    "face_contact_audit.json": ("face_contact_rgb", "face_contact_de"),
    "face_strip_audit.json": ("face_strip",),
    "background_target_audit.json": ("background_retention", "background_pp_drift"),
    "base_rim_audit.json": ("base_color_rim", "base_rim"),
    "block_patch_audit.json": ("original_color_patch",),
    "micro_jaggedness_audit.json": ("contour_hf",),
    "white_net_audit.json": ("white_net",),
}


def split_v233_audits(summary: dict[str, float | int]) -> dict[str, dict[str, float | int]]:
    return {
        filename: {key: value for key, value in summary.items() if key == "count" or any(token in key for token in tokens)}
        for filename, tokens in AUDIT_KEYS.items()
    }


FAILURE_PRIORITY = (
    "V233_DIAGNOSTIC_IMPLEMENTATION_FAIL", "V233_FB_CONFIDENCE_FAIL",
    "V233_FOREGROUND_TARGET_FAIL", "V233_ALPHA_CALIBRATION_FAIL",
    "V233_FACE_STRIP_FAIL", "V233_BACKGROUND_TARGET_FAIL", "V233_BASE_RIM_FAIL",
    "V233_BLOCK_PATCH_FAIL", "V233_CORE_COLOR_FAIL", "V233_DIRECT_REFERENCE_FAIL",
    "V233_MICRO_JAGGED_FAIL",
)


def classify_v233(summary: dict[str, float | int], baseline: dict[str, float | int],
                  *, parity: dict[str, float]) -> dict[str, object]:
    failed: list[str] = []
    if any(parity.get(key, 1.0) > limit for key, limit in (("alpha", 1e-6), ("alpha_eff", 1e-6), ("foreground_target", 1e-5), ("background_target", 1e-5), ("final", 1e-5))):
        failed.append("V233_DIAGNOSTIC_IMPLEMENTATION_FAIL")
    if float(summary.get("median_foreground_conf_high_alpha", 0)) < 0.50:
        failed.append("V233_FB_CONFIDENCE_FAIL")
    if (float(summary.get("median_reconstruction_transition_rgb_mae", 1)) > 0.01 or
            float(summary.get("p90_reconstruction_transition_rgb_mae", 1)) > 0.02):
        failed.append("V233_FB_CONFIDENCE_FAIL")
    if (float(summary.get("median_edge_foreground_progress", 0)) < 1.25 * float(baseline.get("median_edge_foreground_progress", 0)) or
            float(summary.get("median_alpha_bin_progress_std", 1)) > .75 * float(baseline.get("median_alpha_bin_progress_std", 1))):
        failed.append("V233_FOREGROUND_TARGET_FAIL")
    if float(summary.get("median_base_color_rim_fraction", 1)) > .70 * float(baseline.get("median_base_color_rim_fraction", 1)):
        failed.append("V233_BASE_RIM_FAIL")
    if (float(summary.get("median_original_color_patch_fraction", 1)) > .70 * float(baseline.get("median_original_color_patch_fraction", 1)) or
            float(summary.get("median_max_original_color_patch_area", 1)) > .70 * float(baseline.get("median_max_original_color_patch_area", 1))):
        failed.append("V233_BLOCK_PATCH_FAIL")
    if float(summary.get("median_false_positive_alpha", 1)) > .65 * float(baseline.get("median_false_positive_alpha", 1)) or float(summary.get("max_sure_fg_error", 1)) > 1e-6:
        failed.append("V233_ALPHA_CALIBRATION_FAIL")
    if (float(summary.get("median_face_contact_rgb", 1)) > .75 * float(baseline.get("median_face_contact_rgb", 1)) or
            float(summary.get("median_face_strip_max_area", 1)) > .70 * float(baseline.get("median_face_strip_max_area", 1))):
        failed.append("V233_FACE_STRIP_FAIL")
    if float(summary.get("median_background_retention", 1)) > .75 * float(baseline.get("median_background_retention", 1)):
        failed.append("V233_BACKGROUND_TARGET_FAIL")
    if float(summary.get("median_core_ref_full", 1e9)) > 1.03 * float(baseline.get("median_core_ref_full", 0)):
        failed.append("V233_CORE_COLOR_FAIL")
    if float(summary.get("median_direct_ref_ab", 1e9)) > 1.03 * float(baseline.get("median_direct_ref_ab", 0)):
        failed.append("V233_DIRECT_REFERENCE_FAIL")
    if float(summary.get("median_contour_hf_ab", 1e9)) > 1.10 * float(baseline.get("median_contour_hf_ab", 0)):
        failed.append("V233_MICRO_JAGGED_FAIL")
    failed = list(dict.fromkeys(failed))
    ordered = [name for name in FAILURE_PRIORITY if name in failed]
    primary = ordered[0] if ordered else None
    return {"failed_gates": failed, "primary_failure": primary,
            "automatic_decision": primary or "V233_READY_FOR_VISUAL_REVIEW"}


__all__ = ["aggregate_v233", "classify_v233", "split_v233_audits", "v233_metric_tensors"]
