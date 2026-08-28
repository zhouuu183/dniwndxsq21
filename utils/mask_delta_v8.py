import torch
import torch.nn.functional as F


PRIMARY_FACE_LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17)
PRIMARY_CONTEXT_LABELS = PRIMARY_FACE_LABELS + (13, 14, 18)
OCCLUDER_LABELS = (14, 18)
FACE_SURFACE_LABELS = (1, 2)
EAR_SURFACE_LABELS = (8, 9, 15)
DETAIL_PROTECT_LABELS = (3, 4, 5, 6, 7, 10, 11, 12, 14)
NECK_LABELS = (16, 17)
CLOTH_LABELS = (18,)


def _ensure_batch(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return mask


def _to_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = _ensure_batch(mask)
    return (mask > 0.5).float()


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=width)


def _label_mask(parsing_mask: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(parsing_mask, dtype=torch.bool)
    for label in labels:
        mask |= parsing_mask == label
    return mask


def drop_parsing_labels(parsing_mask: torch.Tensor, labels: tuple[int, ...] | None = None) -> torch.Tensor:
    parsing_mask = _ensure_batch(parsing_mask).clone()
    labels = OCCLUDER_LABELS if labels is None else labels
    for label in labels:
        parsing_mask = torch.where(parsing_mask == label, torch.zeros_like(parsing_mask), parsing_mask)
    return parsing_mask


def build_primary_subject_support(
    parsing_mask: torch.Tensor,
    central_ratio: float = 0.65,
    face_labels: tuple[int, ...] | None = None,
    context_labels: tuple[int, ...] | None = None,
) -> torch.Tensor:
    parsing_mask = _ensure_batch(parsing_mask)
    _, _, height, width = parsing_mask.shape

    face_labels = PRIMARY_FACE_LABELS if face_labels is None else face_labels
    context_labels = PRIMARY_CONTEXT_LABELS if context_labels is None else context_labels
    face_mask = _label_mask(parsing_mask, face_labels)
    context_mask = _label_mask(parsing_mask, context_labels)

    y_margin = int(round((1.0 - central_ratio) * 0.5 * height))
    x_margin = int(round((1.0 - central_ratio) * 0.5 * width))
    central_window = torch.zeros_like(face_mask)
    central_window[..., y_margin:height - y_margin, x_margin:width - x_margin] = True

    support = torch.zeros_like(parsing_mask, dtype=torch.float32)
    for idx in range(parsing_mask.shape[0]):
        anchor = face_mask[idx : idx + 1] & central_window[idx : idx + 1]
        if not anchor.any():
            anchor = face_mask[idx : idx + 1]
        if not anchor.any():
            anchor = context_mask[idx : idx + 1] & central_window[idx : idx + 1]
        if not anchor.any():
            anchor = context_mask[idx : idx + 1]
        if not anchor.any():
            support[idx].fill_(1.0)
            continue

        coords = torch.nonzero(anchor[0, 0], as_tuple=False)
        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item())
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item())

        span = max(y_max - y_min + 1, x_max - x_min + 1)
        pad_x = max(18, int(round(0.55 * span)))
        pad_top = max(24, int(round(0.90 * span)))
        pad_bottom = max(30, int(round(1.55 * span)))

        y0 = max(0, y_min - pad_top)
        y1 = min(height, y_max + pad_bottom + 1)
        x0 = max(0, x_min - pad_x)
        x1 = min(width, x_max + pad_x + 1)
        support[idx, :, y0:y1, x0:x1] = 1.0

    return support


def apply_subject_support(parsing_mask: torch.Tensor, subject_support: torch.Tensor) -> torch.Tensor:
    parsing_mask = _ensure_batch(parsing_mask)
    subject_support = _to_mask(subject_support)
    return torch.where(subject_support > 0.5, parsing_mask, torch.zeros_like(parsing_mask))


def filter_parsing_to_primary_subject(
    parsing_mask: torch.Tensor,
    subject_support: torch.Tensor | None = None,
    face_labels: tuple[int, ...] | None = None,
    context_labels: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    parsing_mask = _ensure_batch(parsing_mask)
    if subject_support is None:
        subject_support = build_primary_subject_support(
            parsing_mask,
            face_labels=face_labels,
            context_labels=context_labels,
        )
    filtered = apply_subject_support(parsing_mask, subject_support)
    return filtered, subject_support


def restrict_hair_mask_to_subject(hair_mask: torch.Tensor, subject_support: torch.Tensor) -> torch.Tensor:
    return (_to_mask(hair_mask) * _to_mask(subject_support)).clamp(0, 1)


def build_delta_masks(
    src_hair: torch.Tensor,
    tgt_hair: torch.Tensor,
    ref_hair: torch.Tensor | None = None,
    boundary_width: int = 5,
) -> dict[str, torch.Tensor]:
    src = _to_mask(src_hair)
    tgt = _to_mask(tgt_hair)
    ref = _to_mask(ref_hair) if ref_hair is not None else tgt.clone()

    add = (tgt * (1.0 - src)).clamp(0, 1)
    remove = (src * (1.0 - tgt)).clamp(0, 1)
    keep = (src * tgt).clamp(0, 1)

    delta = (add + remove).clamp(0, 1)
    boundary = (_dilate(delta, boundary_width) - _erode(delta, boundary_width)).clamp(0, 1)
    ref_overlap = (ref * tgt).clamp(0, 1)

    return {
        "M_src": src,
        "M_tgt": tgt,
        "M_add": add,
        "M_remove": remove,
        "M_keep": keep,
        "M_boundary": boundary,
        "M_ref_overlap": ref_overlap,
    }


def enrich_delta_masks_with_halo(
    src_parsing: torch.Tensor,
    delta_masks: dict[str, torch.Tensor],
    subject_support: torch.Tensor | None = None,
    halo_width: int = 9,
    face_labels: tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    parsing = _ensure_batch(src_parsing)
    if subject_support is None:
        subject_support = build_primary_subject_support(parsing, face_labels=face_labels)

    subject_support = _to_mask(subject_support)
    face_labels = PRIMARY_FACE_LABELS if face_labels is None else face_labels
    face_region = (_label_mask(parsing, face_labels).float() * subject_support).clamp(0, 1)
    face_surface = (_label_mask(parsing, FACE_SURFACE_LABELS).float() * subject_support).clamp(0, 1)
    ear_surface = (_label_mask(parsing, EAR_SURFACE_LABELS).float() * subject_support).clamp(0, 1)
    detail_protect = (_dilate(_label_mask(parsing, DETAIL_PROTECT_LABELS).float(), 1) * subject_support).clamp(0, 1)
    neck_region = (_label_mask(parsing, NECK_LABELS).float() * subject_support).clamp(0, 1)
    cloth_region = (_label_mask(parsing, CLOTH_LABELS).float() * subject_support).clamp(0, 1)
    body_region = (neck_region + cloth_region).clamp(0, 1)
    context_region = (subject_support * (1.0 - face_region - body_region).clamp(0, 1)).clamp(0, 1)
    face_detail_guard = _dilate(detail_protect, 1).clamp(0, 1)
    face_cleanup_surface = (
        _dilate((face_surface + 0.75 * ear_surface).clamp(0, 1), 3)
        * (1.0 - 0.78 * face_detail_guard)
    ).clamp(0, 1)

    remove = _to_mask(delta_masks["M_remove"])
    tgt = _to_mask(delta_masks["M_tgt"])
    remove_halo = (_dilate(remove, halo_width) - remove).clamp(0, 1)
    remove_halo = (remove_halo * (1.0 - tgt) * subject_support).clamp(0, 1)
    body_block = (_dilate(body_region, 11) * (1.0 - tgt)).clamp(0, 1)
    cloth_guard = (_dilate(cloth_region, 9) * (1.0 - tgt)).clamp(0, 1)
    body_block = (body_block + 0.75 * cloth_guard).clamp(0, 1)
    remove_halo = (remove_halo * (1.0 - 0.90 * body_block)).clamp(0, 1)
    face_strand_probe = (
        _dilate(remove, 3)
        * face_cleanup_surface
        * (1.0 - 0.70 * face_detail_guard)
    ).clamp(0, 1)
    face_residue_seed = (
        (remove + 0.85 * remove_halo + 0.40 * face_strand_probe)
        * face_cleanup_surface
    ).clamp(0, 1)
    remove_face = (
        _dilate(face_residue_seed, 1)
        * face_cleanup_surface
        * (1.0 - 0.54 * face_detail_guard)
    ).clamp(0, 1)
    remove_context = (remove_halo * context_region * (1.0 - 0.55 * _dilate(detail_protect, 2))).clamp(0, 1)
    neck_residue_seed = (
        (remove + 0.72 * remove_halo)
        * _dilate(neck_region, 6)
        * (1.0 - 0.58 * cloth_guard)
    ).clamp(0, 1)
    remove_neck = (_dilate(neck_residue_seed, 1) * (1.0 - 0.50 * cloth_guard)).clamp(0, 1)
    visible_body_anchor = (
        _dilate((neck_region + 0.65 * cloth_region).clamp(0, 1), 1)
        * (1.0 - 0.20 * tgt)
    ).clamp(0, 1)

    _, _, height, _ = parsing.shape
    y_coords = torch.linspace(0.0, 1.0, steps=height, device=parsing.device).view(1, 1, height, 1)
    lower_region = (y_coords > 0.52).float()
    face_tail_guard = _dilate((face_surface + 0.45 * ear_surface).clamp(0, 1), 1)
    remove_tail = (
        (remove + 1.00 * remove_halo + 0.35 * remove_neck)
        * lower_region
        * (1.0 - 0.72 * face_tail_guard)
        * (1.0 - 0.58 * body_block)
    ).clamp(0, 1)
    body_reveal = (
        (remove_neck + 0.55 * remove_tail + 0.18 * remove_halo)
        * (_dilate(neck_region, 4) + 0.55 * cloth_guard).clamp(0, 1)
        * (1.0 - 0.72 * visible_body_anchor)
    ).clamp(0, 1)
    context_reveal = (
        (remove_context + 0.45 * remove_tail * context_region + 0.30 * remove_halo * context_region)
        * (1.0 - 0.55 * visible_body_anchor)
        * (1.0 - 0.25 * face_region)
    ).clamp(0, 1)
    reveal_overlap = torch.minimum(body_reveal, context_reveal).clamp(0, 1)
    body_reveal_only = (body_reveal - reveal_overlap).clamp(0, 1)
    context_reveal_only = (context_reveal - reveal_overlap).clamp(0, 1)
    body_preserve = (
        body_block
        * (1.0 - 0.88 * body_reveal_only)
        * (1.0 - 0.62 * reveal_overlap)
        * (1.0 - 0.35 * remove_tail)
    ).clamp(0, 1)

    enriched = dict(delta_masks)
    enriched.update(
        {
            "M_remove_halo": remove_halo,
            "M_remove_face": remove_face,
            "M_remove_neck": remove_neck,
            "M_remove_context": remove_context,
            "M_remove_tail": remove_tail,
            "M_face_strand_probe": face_strand_probe,
            "M_subject_support": subject_support,
            "M_face_region": face_region,
            "M_face_surface": face_surface,
            "M_ear_surface": ear_surface,
            "M_detail_protect": detail_protect,
            "M_face_cleanup_surface": face_cleanup_surface,
            "M_neck_region": neck_region,
            "M_cloth_region": cloth_region,
            "M_body_region": body_region,
            "M_body_preserve": body_preserve,
            "M_visible_body_anchor": visible_body_anchor,
            "M_body_reveal": body_reveal,
            "M_context_reveal": context_reveal,
            "M_reveal_overlap": reveal_overlap,
            "M_body_reveal_only": body_reveal_only,
            "M_context_reveal_only": context_reveal_only,
            "M_context_region": context_region,
        }
    )
    return enriched


def stack_satd_masks(delta_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            delta_masks["M_add"],
            delta_masks["M_remove"],
            delta_masks["M_keep"],
            delta_masks["M_boundary"],
            delta_masks["M_ref_overlap"],
        ],
        dim=1,
    )


def stack_cleanup_masks_v8(delta_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    remove = delta_masks["M_remove"]
    zero = torch.zeros_like(remove)
    return torch.cat(
        [
            delta_masks["M_boundary"],
            remove,
            delta_masks.get("M_remove_halo", zero),
            delta_masks.get("M_remove_tail", zero),
            delta_masks.get("M_remove_face", zero),
            delta_masks.get("M_remove_neck", zero),
            delta_masks.get("M_body_preserve", zero),
        ],
        dim=1,
    )


def protect_cleanup_masks_v8(
    cleanup_masks: torch.Tensor,
    protect_mask: torch.Tensor,
) -> torch.Tensor:
    """Remove edit authority and add preserve authority on protected pixels.

    V8 cleanup tensors contain six edit channels followed by
    ``M_body_preserve``.  These channel groups have opposite polarity and must
    never be masked with the same multiplication.
    """
    cleanup_masks = _ensure_batch(cleanup_masks)
    protect_mask = _ensure_batch(protect_mask).to(
        device=cleanup_masks.device,
        dtype=cleanup_masks.dtype,
    )
    if cleanup_masks.size(1) != 7:
        raise ValueError(
            "cleanup_masks must contain six edit channels plus M_body_preserve; "
            f"got {cleanup_masks.size(1)} channels."
        )
    if protect_mask.size(1) != 1:
        protect_mask = protect_mask[:, :1]
    if protect_mask.shape[-2:] != cleanup_masks.shape[-2:]:
        protect_mask = F.interpolate(
            protect_mask,
            size=cleanup_masks.shape[-2:],
            mode="nearest",
        )
    protect_mask = protect_mask.clamp(0, 1)
    channels = list(torch.chunk(cleanup_masks, 7, dim=1))
    for channel_index in range(6):
        channels[channel_index] = (
            channels[channel_index] * (1.0 - protect_mask)
        ).clamp(0, 1)
    channels[6] = torch.maximum(channels[6], protect_mask).clamp(0, 1)
    return torch.cat(channels, dim=1)


__all__ = [
    "apply_subject_support",
    "build_delta_masks",
    "build_primary_subject_support",
    "drop_parsing_labels",
    "enrich_delta_masks_with_halo",
    "filter_parsing_to_primary_subject",
    "protect_cleanup_masks_v8",
    "restrict_hair_mask_to_subject",
    "stack_satd_masks",
    "stack_cleanup_masks_v8",
]
