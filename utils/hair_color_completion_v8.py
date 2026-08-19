"""Reference-guided V8 hair colour completion.

The V8 blending classes and scripts share this module.  It keeps the shape
branch's hair texture, changes colour only inside the target hair alpha, and
restores that corrected hair after PP so PP cannot repaint it with face colour.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from utils.hair_color_match_v8 import gaussian_blur2d, lab_to_rgb, masked_mean_std, rgb_to_lab


def _as_image(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.size(1) != 3:
        raise ValueError(f"Expected [B,3,H,W] image tensor, got {tuple(image.shape)}.")
    normalized = bool(image.detach().amin() < -0.05)
    if normalized:
        image = (image + 1.0) * 0.5
    return image.float().clamp(0, 1), normalized


def _restore_range(image: torch.Tensor, normalized: bool) -> torch.Tensor:
    image = image.clamp(0, 1)
    return image * 2.0 - 1.0 if normalized else image


def _as_mask(mask: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.zeros_like(image[:, :1])
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected [B,1,H,W] mask tensor, got {tuple(mask.shape)}.")
    if mask.size(0) != image.size(0):
        raise ValueError("Hair colour mask batch size must match the image batch size.")
    mask = mask[:, :1].to(device=image.device, dtype=image.dtype)
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return mask.clamp(0, 1)


def _combine_protect_masks(protect_masks: dict[str, torch.Tensor] | None, image: torch.Tensor) -> torch.Tensor:
    protected = torch.zeros_like(image[:, :1])
    if not isinstance(protect_masks, dict):
        return protected
    for value in protect_masks.values():
        if torch.is_tensor(value):
            protected = torch.maximum(protected, _as_mask(value, image))
    return protected


def _masked_channel_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand(-1, value.size(1), -1, -1)
    denom = expanded.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    return (value * expanded).sum(dim=(-2, -1), keepdim=True) / denom


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().mean().cpu())


def reference_color_completion_v8(
    *,
    source: torch.Tensor,
    reference: torch.Tensor,
    blend_raw: torch.Tensor,
    source_hair: torch.Tensor | None,
    reference_hair: torch.Tensor | None,
    target_hair: torch.Tensor | None,
    protect_masks: dict[str, torch.Tensor] | None = None,
    generated_hair: torch.Tensor | None = None,
    target_progress: float = 0.88,
    chroma_gain: float = 0.75,
    luma_gain: float = 0.20,
    max_chroma_shift: float = 0.075,
    max_luma_shift: float = 0.035,
    vivid_boost: float = 1.0,
    highlight_chroma_gain: float = 0.85,
    highlight_luma_gain: float = 0.35,
    global_palette_lock: float = 0.95,
    texture_preserve: float = 0.35,
    edge_width: int = 3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Transfer reference hair colour while retaining generated geometry.

    All image inputs may use either ``[0, 1]`` or StyleGAN ``[-1, 1]``.  The
    output has the same range as ``blend_raw``.  Parser and delta masks are
    constraints only; none are treated as RGB source authority.
    """
    blend, output_normalized = _as_image(blend_raw)
    reference_01, _ = _as_image(reference)
    source_01, _ = _as_image(source)
    if reference_01.shape[-2:] != blend.shape[-2:]:
        reference_01 = F.interpolate(reference_01, size=blend.shape[-2:], mode="bilinear", align_corners=False)
    if source_01.shape[-2:] != blend.shape[-2:]:
        source_01 = F.interpolate(source_01, size=blend.shape[-2:], mode="bilinear", align_corners=False)

    target = _as_mask(target_hair, blend)
    generated = _as_mask(generated_hair, blend)
    reference_mask = _as_mask(reference_hair, reference_01)
    source_mask = _as_mask(source_hair, source_01)
    protected = _combine_protect_masks(protect_masks, blend)

    # The shape target is the primary alpha.  Generated parsing only fills
    # small parser holes, and never opens a large unrelated region.
    progress = max(0.0, min(1.0, float(target_progress)))
    generated_fill = generated * gaussian_blur2d(target, radius=max(1, int(edge_width))).clamp(0, 1)
    structural_hair = torch.maximum(target, generated_fill * (1.0 - progress))
    completion_alpha = structural_hair * (1.0 - protected).clamp(0, 1)
    if int(edge_width) > 0:
        completion_alpha = gaussian_blur2d(completion_alpha, radius=int(edge_width)).clamp(0, 1)
        # Feather only inside the permitted hair region.  A protected face,
        # ear or body mask remains an exact no-write constraint even at the
        # colour transition edge.
        completion_alpha = completion_alpha * structural_hair * (1.0 - protected).clamp(0, 1)

    if completion_alpha.sum().item() < 1.0 or reference_mask.sum().item() < 1.0:
        masks = {
            "target_hair": target,
            "generated_hair": generated,
            "reference_hair": reference_mask,
            "source_hair": source_mask,
            "protected": protected,
            "completion_alpha": completion_alpha,
            "pp_guard": completion_alpha,
        }
        stats = {"completion_area": _scalar(completion_alpha), "reference_area": _scalar(reference_mask)}
        return _restore_range(blend, output_normalized), masks, stats

    blend_lab = rgb_to_lab(blend)
    reference_lab = rgb_to_lab(reference_01)
    target_stats = (completion_alpha > 0.5).to(blend.dtype)
    if target_stats.sum().item() < 16:
        target_stats = completion_alpha
    ref_stats = (reference_mask > 0.5).to(blend.dtype)
    if ref_stats.sum().item() < 16:
        ref_stats = reference_mask

    blend_l, blend_ab = blend_lab[:, :1], blend_lab[:, 1:]
    ref_l, ref_ab = reference_lab[:, :1], reference_lab[:, 1:]
    blend_l_mean, blend_l_std = masked_mean_std(blend_l, target_stats)
    ref_l_mean, ref_l_std = masked_mean_std(ref_l, ref_stats)
    blend_ab_mean, blend_ab_std = masked_mean_std(blend_ab, target_stats)
    ref_ab_mean, ref_ab_std = masked_mean_std(ref_ab, ref_stats)

    lock = max(0.0, min(1.0, float(global_palette_lock)))
    chroma_strength = max(0.0, float(chroma_gain)) * lock
    luma_strength = max(0.0, float(luma_gain)) * lock
    # Work in Lab units; option values are normalized to the common [0,1]
    # RGB scale, hence the conversion bounds below.
    ab_limit = max(0.0, float(max_chroma_shift)) * 180.0
    l_limit = max(0.0, float(max_luma_shift)) * 100.0
    l_delta = (ref_l_mean - blend_l_mean).clamp(-l_limit, l_limit)
    ab_delta = (ref_ab_mean - blend_ab_mean).clamp(-ab_limit, ab_limit)

    matched_l = blend_l + luma_strength * l_delta
    matched_ab = blend_ab + chroma_strength * ab_delta
    if vivid_boost != 1.0:
        matched_ab = blend_ab_mean + float(vivid_boost) * (matched_ab - blend_ab_mean)

    # Keep highlights stable: only a bounded extra move is permitted in the
    # bright hair tail, avoiding clipped silver/blonde patches.
    highlight = (blend_l >= blend_l_mean + 0.5 * blend_l_std).to(blend.dtype) * completion_alpha
    matched_l = matched_l + highlight * max(0.0, float(highlight_luma_gain)) * 0.25 * l_delta
    matched_ab = matched_ab + highlight * max(0.0, float(highlight_chroma_gain)) * 0.25 * ab_delta
    matched_rgb = lab_to_rgb(torch.cat((matched_l, matched_ab), dim=1))

    # Preserve high-frequency generated texture rather than replacing hair
    # with a flat statistics-matched field.
    preserve = max(0.0, min(1.0, float(texture_preserve)))
    low_raw = gaussian_blur2d(blend, radius=max(1, int(edge_width)))
    low_matched = gaussian_blur2d(matched_rgb, radius=max(1, int(edge_width)))
    textured_rgb = (low_matched + preserve * (blend - low_raw)).clamp(0, 1)
    completed = blend * (1.0 - completion_alpha) + textured_rgb * completion_alpha

    masks = {
        "target_hair": target,
        "generated_hair": generated,
        "reference_hair": reference_mask,
        "source_hair": source_mask,
        "protected": protected,
        "completion_alpha": completion_alpha,
        # PP guard is deliberately the same object/target hair alpha.  It
        # prevents the post-process decoder from recolouring the completed
        # hair, without protecting face or background pixels.
        "pp_guard": completion_alpha,
    }
    stats = {
        "completion_area": _scalar(completion_alpha),
        "reference_area": _scalar(reference_mask),
        "source_area": _scalar(source_mask),
        "luma_delta": _scalar(l_delta),
        "chroma_delta": _scalar(torch.linalg.vector_norm(ab_delta, dim=1, keepdim=True)),
        "luma_std_ratio": _scalar(ref_l_std / blend_l_std.clamp_min(1e-4)),
        "chroma_std_ratio": _scalar(ref_ab_std / blend_ab_std.clamp_min(1e-4)),
    }
    return _restore_range(completed, output_normalized), masks, stats


def apply_pp_hair_guard_v8(
    pp_output: torch.Tensor,
    color_fixed: torch.Tensor,
    completion_masks: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Keep completed target hair above the PP decoder output."""
    pp, pp_normalized = _as_image(pp_output)
    fixed, _ = _as_image(color_fixed)
    if fixed.shape[-2:] != pp.shape[-2:]:
        fixed = F.interpolate(fixed, size=pp.shape[-2:], mode="bilinear", align_corners=False)
    guard = _as_mask(completion_masks.get("pp_guard"), pp)
    return _restore_range(pp * (1.0 - guard) + fixed * guard, pp_normalized)


def compute_color_diagnostics_v8(
    source: torch.Tensor,
    reference: torch.Tensor,
    stages: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Return compact scalar colour diagnostics for V8 debug runs."""
    reference_01, _ = _as_image(reference)
    reference_mask = _as_mask(masks.get("reference_hair"), reference_01)
    ref_lab = rgb_to_lab(reference_01)
    ref_ab = _masked_channel_mean(ref_lab[:, 1:], reference_mask)
    diagnostics: dict[str, float] = {}
    for name, image in stages.items():
        image_01, _ = _as_image(image)
        stage_mask = _as_mask(masks.get("completion_alpha"), image_01)
        if stage_mask.sum().item() < 1.0:
            continue
        lab = rgb_to_lab(image_01)
        ab = _masked_channel_mean(lab[:, 1:], stage_mask)
        diagnostics[f"{name}_reference_ab_distance"] = _scalar(
            torch.linalg.vector_norm(ab - ref_ab, dim=1, keepdim=True)
        )
        diagnostics[f"{name}_hair_luma"] = _scalar(_masked_channel_mean(lab[:, :1], stage_mask))
    return diagnostics


def _save_image(path: Path, image: torch.Tensor) -> None:
    image_01, _ = _as_image(image)
    array = (image_01[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array).save(path)


def _save_mask(path: Path, mask: torch.Tensor) -> None:
    mask = mask[0, 0].detach().float().cpu().clamp(0, 1)
    Image.fromarray((mask.numpy() * 255.0).round().astype("uint8"), mode="L").save(path)


def save_color_debug_v8(
    output_dir: str | Path,
    blend_raw: torch.Tensor,
    masks: dict[str, torch.Tensor],
    color_fixed: torch.Tensor,
    pp_raw: torch.Tensor,
    pp_guard: torch.Tensor,
    final: torch.Tensor,
    stats: dict[str, float],
) -> None:
    """Save V8 colour stages and masks without depending on torchvision."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in {
        "blend_raw.png": blend_raw,
        "color_fixed.png": color_fixed,
        "pp_raw.png": pp_raw,
        "final.png": final,
    }.items():
        _save_image(output_dir / name, image)
    for name, value in {**masks, "pp_guard": pp_guard}.items():
        if torch.is_tensor(value):
            _save_mask(output_dir / f"{name}.png", value)
    serializable = {
        key: float(value.detach().cpu()) if torch.is_tensor(value) and value.numel() == 1 else float(value)
        for key, value in stats.items()
    }
    (output_dir / "stats.json").write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
