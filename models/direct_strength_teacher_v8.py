from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch


DIRECT_STRENGTH_TEACHER_ARCH_V8_4 = "direct_color_anchor_v8_4"
DEFAULT_TEACHER_ALPHA_CANDIDATES = (0.0, 0.25, 0.50, 0.70, 0.85, 1.0)


def triplet_cache_key(triplet: Sequence[str]) -> str:
    if len(triplet) != 3:
        raise ValueError(f"Expected a 3-column triplet, got {triplet!r}")
    return json.dumps([str(value) for value in triplet], ensure_ascii=True, separators=(",", ":"))


def alpha_score_field(alpha: float) -> str:
    scaled = int(round(float(alpha) * 100.0))
    suffix = "0" if scaled == 0 else f"{scaled:03d}"
    return f"alpha_{suffix}_score"


def compute_teacher_color_score(
    mean_ab_error: torch.Tensor,
    hue_error: torch.Tensor,
    chroma_error: torch.Tensor,
) -> torch.Tensor:
    return mean_ab_error + 0.10 * hue_error + 0.50 * chroma_error


def compute_teacher_score(
    mean_ab_error: torch.Tensor,
    hue_error: torch.Tensor,
    chroma_error: torch.Tensor,
    face_keep_error: torch.Tensor,
    luma_artifact_penalty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    color_score = compute_teacher_color_score(mean_ab_error, hue_error, chroma_error)
    total_score = color_score + 0.25 * face_keep_error + luma_artifact_penalty
    return total_score, color_score


def select_teacher_strength(
    alpha_candidates: Sequence[float],
    total_scores: torch.Tensor,
    *,
    margin_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Select the lowest-score alpha along the last tensor dimension."""
    if total_scores.size(-1) != len(alpha_candidates):
        raise ValueError(
            f"Got {total_scores.size(-1)} scores for {len(alpha_candidates)} alpha candidates"
        )
    if len(alpha_candidates) < 2:
        raise ValueError("Teacher selection requires at least two alpha candidates")
    candidates = torch.as_tensor(
        alpha_candidates, device=total_scores.device, dtype=total_scores.dtype
    )
    sorted_scores, sorted_indices = total_scores.sort(dim=-1)
    best_index = sorted_indices[..., 0]
    best_score = sorted_scores[..., 0]
    second_best_score = sorted_scores[..., 1]
    teacher_alpha = candidates[best_index]
    confidence = (
        (second_best_score - best_score) / max(float(margin_scale), 1e-6)
    ).clamp(0, 1)
    return {
        "teacher_alpha": teacher_alpha,
        "teacher_score": best_score,
        "teacher_confidence": confidence,
        "teacher_margin": second_best_score - best_score,
        "teacher_index": best_index,
    }


def summarize_teacher_records(records: Mapping[str, Mapping[str, float]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("Cannot summarize an empty teacher cache")
    alphas = torch.tensor(
        [float(record["teacher_alpha"]) for record in records.values()], dtype=torch.float32
    )
    high_color = torch.tensor(
        [bool(record.get("high_color", False)) for record in records.values()], dtype=torch.bool
    )
    light_hair = torch.tensor(
        [bool(record.get("white_or_light_hair", False)) for record in records.values()],
        dtype=torch.bool,
    )

    def subset_mean(mask: torch.Tensor) -> float | None:
        return float(alphas[mask].mean().item()) if bool(mask.any()) else None

    return {
        "sample_count": int(alphas.numel()),
        "mean_alpha": float(alphas.mean().item()),
        "median_alpha": float(alphas.median().item()),
        "std_alpha": float(alphas.std(unbiased=False).item()),
        "alpha_eq_0_fraction": float((alphas == 0).float().mean().item()),
        "alpha_le_025_fraction": float((alphas <= 0.25).float().mean().item()),
        "alpha_ge_085_fraction": float((alphas >= 0.85).float().mean().item()),
        "alpha_eq_1_fraction": float((alphas == 1).float().mean().item()),
        "alpha_eq_070_fraction": float(torch.isclose(alphas, torch.tensor(0.70)).float().mean().item()),
        "high_chroma_mean_alpha": subset_mean(high_color),
        "white_light_hair_mean_alpha": subset_mean(light_hair),
    }


def validate_teacher_distribution(
    records: Mapping[str, Mapping[str, float]],
    *,
    collapse_fraction: float = 0.90,
) -> dict[str, float | int]:
    summary = summarize_teacher_records(records)
    if float(summary["alpha_eq_070_fraction"]) >= float(collapse_fraction):
        raise RuntimeError(
            "Teacher sweep collapsed around alpha=0.70: "
            f"fraction={summary['alpha_eq_070_fraction']:.3f}. "
            "Fix the teacher score before starting V8.4 training."
        )
    return summary


def load_teacher_cache(
    path: str | Path,
    *,
    required_keys: Sequence[str] | None = None,
) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing V8.4 direct-strength teacher cache: {path}. "
            "Run scripts/build_v8_direct_strength_teacher_cache.py first."
        )
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("arch") != DIRECT_STRENGTH_TEACHER_ARCH_V8_4:
        arch = payload.get("arch") if isinstance(payload, dict) else None
        raise RuntimeError(
            f"Incompatible teacher cache {path}: arch={arch!r}, "
            f"required={DIRECT_STRENGTH_TEACHER_ARCH_V8_4!r}"
        )
    records = payload.get("records")
    if not isinstance(records, dict) or not records:
        raise RuntimeError(f"Teacher cache has no records: {path}")
    missing = [key for key in (required_keys or ()) if key not in records]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Teacher cache {path} is missing {len(missing)} dataset triplets: {preview}"
        )
    validate_teacher_distribution(records)
    return payload
