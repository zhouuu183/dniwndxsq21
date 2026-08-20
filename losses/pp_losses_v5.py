from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.pp_losses import LossBuilderMulti
from models.ear_modules_v5 import (
    RAW_DETAIL_LABELS,
    RAW_FACE_SURFACE_LABELS,
    build_weak_earring_mask,
    dilate_mask,
    erode_mask,
    ensure_mask_4d,
    high_pass_filter,
    low_pass_filter,
    resize_mask,
)

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_gray(image)
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=image.device, dtype=image.dtype)
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    if value.ndim == 4 and mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = resize_mask(mask, pred.shape[-2:])
    return masked_mean((pred - target).abs(), mask)


def masked_ratio(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = resize_mask(mask, value.shape[-2:])
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def masked_non_darker(pred_gray: torch.Tensor, anchor_gray: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = resize_mask(mask, pred_gray.shape[-2:])
    return masked_mean(F.relu(anchor_gray - pred_gray), mask)


def parsing_label_mask(parsing: torch.Tensor | None, labels: tuple[int, ...]) -> torch.Tensor | None:
    if parsing is None:
        return None
    parsing = ensure_mask_4d(parsing).long()
    mask = torch.zeros_like(parsing, dtype=torch.bool)
    for label in labels:
        mask |= parsing == label
    return mask.float()


def combine_aux_masks(aux: dict, keys: tuple[str, ...]) -> torch.Tensor | None:
    masks = []
    for key in keys:
        value = aux.get(key)
        if value is not None:
            masks.append(ensure_mask_4d(value).float())
    if not masks:
        return None
    return torch.stack(masks, dim=0).amax(dim=0).clamp(0, 1)


def combine_resized_aux_masks(
    aux: dict,
    keys: tuple[str, ...],
    size: tuple[int, int],
    template: torch.Tensor,
) -> torch.Tensor:
    """Combine optional aux masks without granting missing masks broad access."""
    combined = torch.zeros(
        template.size(0),
        1,
        size[0],
        size[1],
        device=template.device,
        dtype=template.dtype,
    )
    for key in keys:
        value = aux.get(key)
        if value is not None:
            combined = torch.clamp(combined + resize_mask(value, size), 0, 1)
    return combined


def build_v58_earring_keep_mask(
    aux: dict,
    size: tuple[int, int],
    template: torch.Tensor,
) -> torch.Tensor:
    """Return the complete aligned foreground alpha used by V5 losses."""
    confident = aux.get("earring_confident_mask")
    if confident is None:
        return torch.zeros(
            template.size(0),
            1,
            size[0],
            size[1],
            device=template.device,
            dtype=template.dtype,
        )
    confident = resize_mask(confident, size)
    # V5 treats the aligned source instance as the complete object target.
    # Localization/visible-ear ROI is a search prior only; intersecting it
    # here taught long earrings to stop at the lobe and made the learned
    # branch disagree with the final source-alpha compositor.
    keep = resize_mask(confident, size)
    hole = aux.get("hoop_hole_mask")
    if hole is not None:
        keep = keep * (1.0 - resize_mask(hole, size)).clamp(0, 1)
    return keep.clamp(0, 1)


def build_safe_earring_edit_mask(
    aux: dict,
    size: tuple[int, int],
    template: torch.Tensor,
) -> torch.Tensor:
    """Keep forehead/target anchors from treating an empty channel as editable."""
    return build_v58_earring_keep_mask(aux, size, template)


def build_target_hair_preserve_mask(
    aux: dict,
    size: tuple[int, int],
    template: torch.Tensor,
) -> torch.Tensor | None:
    """Protect all target hair except v58's gated confident earring pixels."""
    target_hair = aux.get("target_hair_mask")
    if target_hair is None:
        return None
    preserve = resize_mask(target_hair, size)
    earring_override = build_v58_earring_keep_mask(aux, size, template)
    return preserve * (1.0 - earring_override).clamp(0, 1)


def build_target_preserve_mask(
    aux: dict,
    size: tuple[int, int],
    template: torch.Tensor,
    local_edit_mask: torch.Tensor,
    exclude_dilate: int,
) -> torch.Tensor:
    """Anchor target content outside face/cleanup and safe earring edits."""
    preserve = torch.ones(
        template.size(0),
        1,
        size[0],
        size[1],
        device=template.device,
        dtype=template.dtype,
    )
    if exclude_dilate > 0:
        local_edit_mask = dilate_mask(local_edit_mask, exclude_dilate)
    preserve = preserve * (1.0 - local_edit_mask).clamp(0, 1)

    target_face = aux.get("target_face_surface_mask")
    if target_face is None:
        target_face = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
    if target_face is not None:
        preserve = preserve * (1.0 - resize_mask(target_face, size)).clamp(0, 1)

    editable_face_cleanup = combine_resized_aux_masks(
        aux,
        CLEANUP_MASK_KEYS + ("revealed_skin_mask", "revealed_skin_blend_mask"),
        size,
        template,
    )
    return preserve * (1.0 - editable_face_cleanup).clamp(0, 1)


def build_face_source_dark_reject_mask(
    aux: dict,
    size: tuple[int, int],
    template: torch.Tensor,
    local_edit_mask: torch.Tensor,
    *,
    source_hair_dilate: int = 7,
    detail_exclude_dilate: int = 5,
    ear_exclude_dilate: int = 7,
) -> torch.Tensor | None:
    """Select source-bang risk on target skin while protecting ears/earrings."""
    target_face = aux.get("target_face_surface_mask")
    if target_face is None:
        target_face = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
    source_hair = aux.get("source_hair_mask")
    if target_face is None or source_hair is None:
        return None

    reject = resize_mask(target_face, size)
    target_hair = aux.get("target_hair_mask")
    if target_hair is not None:
        reject = reject * (1.0 - resize_mask(target_hair, size)).clamp(0, 1)

    source_hair_risk = resize_mask(source_hair, size)
    if source_hair_dilate > 0:
        source_hair_risk = dilate_mask(source_hair_risk, source_hair_dilate)
    reject = reject * source_hair_risk

    target_detail = parsing_label_mask(aux.get("target_parsing"), RAW_DETAIL_LABELS)
    if target_detail is not None:
        detail_exclude = resize_mask(target_detail, size)
        if detail_exclude_dilate > 0:
            detail_exclude = dilate_mask(detail_exclude, detail_exclude_dilate)
        reject = reject * (1.0 - detail_exclude).clamp(0, 1)

    # Protect the complete ear region and v58's gated confident earrings.
    ear_exclude = torch.clamp(
        local_edit_mask
        + combine_resized_aux_masks(
            aux,
            (
                "ear_roi",
                "visible_ear_roi",
                "earring_confident_mask",
                "source_earring_mask",
            ),
            size,
            template,
        ),
        0,
        1,
    )
    if ear_exclude_dilate > 0:
        ear_exclude = dilate_mask(ear_exclude, ear_exclude_dilate)
    return reject * (1.0 - ear_exclude).clamp(0, 1)


def source_direction_leak_loss(
    source: torch.Tensor,
    anchor: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = resize_mask(mask, pred.shape[-2:])
    if source.shape[-2:] != pred.shape[-2:]:
        source = F.interpolate(source, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    if anchor.shape[-2:] != pred.shape[-2:]:
        anchor = F.interpolate(anchor, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    source_delta = (source - anchor).detach()
    pred_delta = pred - anchor
    source_norm = source_delta.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-4)
    source_dir = source_delta / source_norm
    copied_toward_source = F.relu((pred_delta * source_dir).sum(dim=1, keepdim=True))
    return masked_mean(copied_toward_source * source_norm.clamp(max=1.0), mask)


def source_dark_high_reject_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 0.006,
    source_threshold: float = 0.018,
    kernel_size: int = 11,
) -> torch.Tensor:
    """Reject source-bang dark residuals without anchoring skin to SATD."""
    mask = resize_mask(mask, pred.shape[-2:])
    if source.shape[-2:] != pred.shape[-2:]:
        source = F.interpolate(source, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    source_gray = rgb_to_gray(source)
    target_gray = rgb_to_gray(target)
    pred_gray = rgb_to_gray(pred)
    source_dark = (
        low_pass_filter(source_gray, kernel_size=kernel_size) - source_gray
    ).clamp(0, 1).detach()
    target_dark = (
        low_pass_filter(target_gray, kernel_size=kernel_size) - target_gray
    ).clamp(0, 1).detach()
    pred_dark = (
        low_pass_filter(pred_gray, kernel_size=kernel_size) - pred_gray
    ).clamp(0, 1)
    source_gate = (source_dark > source_threshold).float()
    excess_dark = F.relu(pred_dark - target_dark - margin)
    return masked_mean(excess_dark * source_gate, mask)


def masked_scalar_stats(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = resize_mask(mask, value.shape[-2:])
    if value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    denom = mask.sum().clamp_min(1.0)
    mean = (value * mask).sum() / denom
    var = ((value - mean).pow(2) * mask).sum() / denom
    return mean, torch.sqrt(var + 1e-6)


def texture_stat_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    pred_mask: torch.Tensor,
    reference_mask: torch.Tensor,
) -> torch.Tensor:
    pred_mask = resize_mask(pred_mask, pred.shape[-2:])
    reference_mask = resize_mask(reference_mask, reference.shape[-2:])
    if pred_mask.detach().sum().item() < 1 or reference_mask.detach().sum().item() < 1:
        return pred.sum() * 0.0

    pred_high_energy = high_pass_filter(pred).abs().mean(dim=1, keepdim=True)
    ref_high_energy = high_pass_filter(reference).abs().mean(dim=1, keepdim=True)
    pred_edge = sobel_edges(pred)
    ref_edge = sobel_edges(reference)

    pred_high_mean, pred_high_std = masked_scalar_stats(pred_high_energy, pred_mask)
    ref_high_mean, ref_high_std = masked_scalar_stats(ref_high_energy, reference_mask)
    pred_edge_mean, pred_edge_std = masked_scalar_stats(pred_edge, pred_mask)
    ref_edge_mean, ref_edge_std = masked_scalar_stats(ref_edge, reference_mask)
    return (
        (pred_high_mean - ref_high_mean.detach()).abs()
        + (pred_high_std - ref_high_std.detach()).abs()
        + 0.5 * (pred_edge_mean - ref_edge_mean.detach()).abs()
        + 0.5 * (pred_edge_std - ref_edge_std.detach()).abs()
    )


def masked_channel_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image, per-channel masked mean (shape B x C x 1 x 1)."""
    mask = resize_mask(mask, value.shape[-2:])
    denom = mask.flatten(2).sum(dim=2, keepdim=True).clamp_min(1.0)
    mean = (value * mask).flatten(2).sum(dim=2, keepdim=True) / denom
    return mean.unsqueeze(-1)


def build_revealed_skin_reference_masks(
    aux: dict,
    size: tuple[int, int],
    template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None:
    """Masks for the revealed-forehead repair loss.

    Returns (revealed_mask, legacy_source_reference_mask, recovered_skin_mask):
    - revealed_mask: newly exposed forehead (source had bangs, reference does not).
    - legacy_source_reference_mask: retained only for cache compatibility; it is
      not used as a texture/tone target by V5 losses.
    - recovered_skin_mask: target face skin the network has already recovered
      correctly (adjacent, non-revealed), used as the tone-continuity anchor so
      no band/seam forms between revealed and recovered skin.
    """
    revealed = aux.get("revealed_skin_mask")
    if revealed is None:
        return None
    revealed = resize_mask(revealed, size)
    if revealed.detach().sum().item() < 1:
        return None

    source_ref = aux.get("source_visible_skin_reference_mask")
    if source_ref is None:
        source_ref = aux.get("source_skin_valid_mask")
    if source_ref is not None:
        source_ref = resize_mask(source_ref, size)

    target_face = aux.get("target_face_surface_mask")
    if target_face is None:
        target_face = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
    if target_face is None:
        return None
    recovered = resize_mask(target_face, size) * (1.0 - dilate_mask(revealed, 5)).clamp(0, 1)
    target_hair = aux.get("target_hair_mask")
    if target_hair is not None:
        recovered = recovered * (1.0 - resize_mask(target_hair, size)).clamp(0, 1)
    detail = parsing_label_mask(aux.get("target_parsing"), RAW_DETAIL_LABELS)
    if detail is not None:
        recovered = recovered * (1.0 - dilate_mask(resize_mask(detail, size), 3)).clamp(0, 1)
    for key in ("earring_confident_mask", "source_earring_mask", "ear_roi", "visible_ear_roi"):
        value = aux.get(key)
        if value is not None:
            recovered = recovered * (1.0 - resize_mask(value, size)).clamp(0, 1)
    return revealed.clamp(0, 1), source_ref, recovered.clamp(0, 1)


def revealed_skin_tone_continuity_loss(
    pred: torch.Tensor,
    revealed_mask: torch.Tensor,
    recovered_mask: torch.Tensor,
) -> torch.Tensor:
    """Pull the revealed forehead's low-frequency tone to the recovered skin tone.

    Matching the masked low-frequency channel mean removes the SATD whitish band
    and, because the recovered anchor is the network's own adjacent skin, erases
    the boundary between covered and uncovered forehead.
    """
    revealed_mask = resize_mask(revealed_mask, pred.shape[-2:])
    recovered_mask = resize_mask(recovered_mask, pred.shape[-2:])
    if revealed_mask.detach().sum().item() < 1 or recovered_mask.detach().sum().item() < 1:
        return pred.sum() * 0.0
    pred_low = low_pass_filter(pred)
    revealed_mean = masked_channel_mean(pred_low, revealed_mask)
    recovered_mean = masked_channel_mean(pred_low, recovered_mask).detach()
    # Tone alone can still leave a transparent-looking band.  Match local
    # gradient statistics across the narrow revealed boundary without treating
    # the source's bang-covered pixels as a texture target.
    seam = (dilate_mask(revealed_mask, 5) - revealed_mask).clamp(0, 1) * recovered_mask
    seam = torch.where(
        (seam.flatten(1).sum(dim=1, keepdim=True) > 1.0).view(-1, 1, 1, 1),
        seam,
        recovered_mask,
    )
    revealed_grad = masked_channel_mean(sobel_edges(pred_low), revealed_mask)
    seam_grad = masked_channel_mean(sobel_edges(pred_low), seam).detach()
    return (revealed_mean - recovered_mean).abs().mean() + 0.5 * (revealed_grad - seam_grad).abs().mean()


def _revealed_boundary_bands(
    face_mask: torch.Tensor,
    revealed_mask: torch.Tensor,
    size: tuple[int, int],
    *,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two skin bands adjacent to a revealed-skin boundary."""
    face = resize_mask(face_mask, size)
    revealed = resize_mask(revealed_mask, size) * face
    inner = (revealed - erode_mask(revealed, width)).clamp(0, 1) * face
    outer = (dilate_mask(revealed, width) - revealed).clamp(0, 1) * face
    return inner.clamp(0, 1), outer.clamp(0, 1)


def face_lowfreq_continuity_loss(
    pred: torch.Tensor,
    face_mask: torch.Tensor,
    revealed_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize a low-frequency skin step on the two sides of one seam."""
    inner, outer = _revealed_boundary_bands(
        face_mask,
        revealed_mask,
        pred.shape[-2:],
        width=5,
    )
    if inner.detach().sum().item() < 1 or outer.detach().sum().item() < 1:
        return pred.sum() * 0.0
    low = low_pass_filter(pred, kernel_size=25, sigma=6.0)
    inner_mean = masked_channel_mean(low, inner)
    outer_mean = masked_channel_mean(low, outer).detach()
    inner_std = torch.sqrt(masked_channel_mean((low - inner_mean).pow(2), inner) + 1e-6)
    outer_std = torch.sqrt(masked_channel_mean((low - outer_mean).pow(2), outer) + 1e-6).detach()
    return (inner_mean - outer_mean).abs().mean() + 0.25 * (inner_std - outer_std).abs().mean()


def revealed_boundary_seam_loss(
    pred: torch.Tensor,
    face_mask: torch.Tensor,
    revealed_mask: torch.Tensor,
) -> torch.Tensor:
    """Match colour and gradient statistics immediately across a mask edge."""
    inner, outer = _revealed_boundary_bands(
        face_mask,
        revealed_mask,
        pred.shape[-2:],
        width=3,
    )
    if inner.detach().sum().item() < 1 or outer.detach().sum().item() < 1:
        return pred.sum() * 0.0
    low = low_pass_filter(pred, kernel_size=15, sigma=3.5)
    colour = (
        masked_channel_mean(low, inner)
        - masked_channel_mean(low, outer).detach()
    ).abs().mean()
    gradient = (
        masked_channel_mean(sobel_edges(low), inner)
        - masked_channel_mean(sobel_edges(low), outer).detach()
    ).abs().mean()
    return colour + 0.5 * gradient


def revealed_texture_stat_loss(
    pred: torch.Tensor,
    face_mask: torch.Tensor,
    revealed_mask: torch.Tensor,
) -> torch.Tensor:
    """Match revealed-skin texture statistics to adjacent generated skin."""
    _inner, outer = _revealed_boundary_bands(
        face_mask,
        revealed_mask,
        pred.shape[-2:],
        width=7,
    )
    revealed = resize_mask(revealed_mask, pred.shape[-2:]) * resize_mask(face_mask, pred.shape[-2:])
    return texture_stat_loss(pred, pred.detach(), revealed, outer)


def face_lowfreq_smoothness_loss(pred: torch.Tensor, face_mask: torch.Tensor) -> torch.Tensor:
    """Suppress low-frequency step seams without using source hair as skin GT."""
    face = resize_mask(face_mask, pred.shape[-2:])
    if face.detach().sum().item() < 1:
        return pred.sum() * 0.0
    low = low_pass_filter(pred, kernel_size=31, sigma=7.0)
    laplace = (
        -4.0 * low
        + F.pad(low[:, :, 1:], (0, 0, 0, 1))
        + F.pad(low[:, :, :-1], (0, 0, 1, 0))
        + F.pad(low[:, :, :, 1:], (0, 1, 0, 0))
        + F.pad(low[:, :, :, :-1], (1, 0, 0, 0))
    ).abs()
    return masked_mean(laplace, face)


def face_hair_boundary_seam_loss(
    pred: torch.Tensor,
    face_mask: torch.Tensor,
    hair_mask: torch.Tensor,
) -> torch.Tensor:
    """Detect a second low-frequency ring at the sole face/hair boundary."""
    face = resize_mask(face_mask, pred.shape[-2:])
    hair = resize_mask(hair_mask, pred.shape[-2:])
    boundary = (
        dilate_mask(hair, 3) - erode_mask(hair, 2)
    ).clamp(0, 1) * dilate_mask(face, 4)
    if boundary.detach().sum().item() < 1:
        return pred.sum() * 0.0
    low = low_pass_filter(pred, kernel_size=21, sigma=5.0)
    gradient = sobel_edges(low)
    local_mean = F.avg_pool2d(gradient, kernel_size=9, stride=1, padding=4)
    return masked_mean((gradient - local_mean).abs(), boundary)


def build_weak_ear_pseudo_mask(
    source: torch.Tensor,
    query_mask: torch.Tensor,
    parser_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = build_weak_earring_mask(source, query_mask, parser_mask)
    return outputs["weak_earring_mask"], outputs["weak_high_energy"]


def build_weak_presence_target(pseudo_mask: torch.Tensor, parser_mask: torch.Tensor | None = None) -> torch.Tensor:
    pseudo_mask = ensure_mask_4d(pseudo_mask).float()
    # Side identity is supplied by EarAnchoredQueryBuilder's parser/anchor
    # targets.  A generic weak pseudo-mask has no reliable side geometry, so it
    # may only provide an any-earring fallback rather than split at image centre.
    any_presence = pseudo_mask.flatten(1).amax(dim=1)

    if parser_mask is not None:
        parser_mask = resize_mask(parser_mask, pseudo_mask.shape[-2:])
        any_presence = torch.maximum(any_presence, parser_mask.flatten(1).amax(dim=1))

    return torch.stack([any_presence, any_presence, any_presence], dim=1).float()


class EarAwareLossBuilder(LossBuilderMulti):
    def __init__(self, losses_dict, device: str = "cuda"):
        super().__init__(losses_dict, device=device)
        self.mask_bce = nn.BCEWithLogitsLoss(reduction="none")
        self.presence_bce = nn.BCEWithLogitsLoss()

    def __call__(self, source, target, target_mask, HT_E, gen_w, F_w, gen_F, F_gen, aux=None, **kwargs):
        cleanup_mask = combine_aux_masks(aux, CLEANUP_MASK_KEYS) if aux is not None else None
        base_target_mask = target_mask
        base_exclude = None
        if cleanup_mask is not None:
            cleanup_for_base = resize_mask(cleanup_mask, target_mask.shape[-2:])
            exclude_strength = max(0.0, min(1.0, float(self.losses_dict.get("base_cleanup_exclude", 1.0))))
            base_exclude = exclude_strength * cleanup_for_base

        if aux is not None:
            source_hair_for_base = aux.get("source_hair_mask")
            target_face_for_base = aux.get("target_face_surface_mask")
            if source_hair_for_base is not None and target_face_for_base is not None:
                base_size = target_mask.shape[-2:]
                source_hair_risk = resize_mask(source_hair_for_base, base_size)
                risk_dilate = int(self.losses_dict.get("base_source_hair_exclude_dilate", 9))
                if risk_dilate > 0:
                    source_hair_risk = dilate_mask(source_hair_risk, risk_dilate)
                source_hair_risk = source_hair_risk * resize_mask(target_face_for_base, base_size)

                max_y = max(
                    0.0,
                    min(1.0, float(self.losses_dict.get("base_source_hair_exclude_max_y", 0.55))),
                )
                if max_y < 1.0:
                    y_coords = torch.linspace(
                        0,
                        1,
                        base_size[0],
                        device=source_hair_risk.device,
                        dtype=source_hair_risk.dtype,
                    ).view(1, 1, base_size[0], 1)
                    source_hair_risk = source_hair_risk * (y_coords <= max_y).float()

                target_hair_for_base = aux.get("target_hair_mask")
                if target_hair_for_base is not None:
                    source_hair_risk = source_hair_risk * (
                        1.0 - resize_mask(target_hair_for_base, base_size)
                    ).clamp(0, 1)

                ear_exclude = torch.zeros_like(source_hair_risk)
                for key in (
                    "ear_roi",
                    "visible_ear_roi",
                    "earring_confident_mask",
                    "source_earring_mask",
                ):
                    value = aux.get(key)
                    if value is not None:
                        ear_exclude = torch.clamp(
                            ear_exclude + resize_mask(value, base_size),
                            0,
                            1,
                        )
                ear_dilate = int(self.losses_dict.get("base_source_hair_exclude_ear_dilate", 9))
                if ear_dilate > 0:
                    ear_exclude = dilate_mask(ear_exclude, ear_dilate)
                source_hair_risk = source_hair_risk * (1.0 - ear_exclude).clamp(0, 1)

                source_hair_strength = max(
                    0.0,
                    min(1.0, float(self.losses_dict.get("base_source_hair_exclude_strength", 1.0))),
                )
                source_hair_exclude = source_hair_strength * source_hair_risk
                base_exclude = (
                    source_hair_exclude
                    if base_exclude is None
                    else torch.clamp(base_exclude + source_hair_exclude, 0, 1)
                )
                aux["base_source_hair_exclude_mask"] = source_hair_exclude

        if base_exclude is not None:
            base_target_mask = ensure_mask_4d(target_mask).float() * (1.0 - base_exclude).clamp(0, 1)

        losses = super().__call__(source, target, base_target_mask, HT_E, gen_w, F_w, gen_F, F_gen, **kwargs)
        if aux is None:
            return losses

        gen_F_256 = self.downsample_256(gen_F)
        gen_F_256_01 = ((gen_F_256 + 1) / 2).clamp(0, 1)
        gen_w_256_01 = None
        if gen_w is not None:
            gen_w_256 = self.downsample_256(gen_w)
            gen_w_256_01 = ((gen_w_256 + 1) / 2).clamp(0, 1)

        uses_safe_query = aux.get("ear_detail_query_mask") is not None or aux.get("hair_safe_query_mask") is not None
        query_mask = aux.get("ear_detail_query_mask", aux.get("hair_safe_query_mask", aux.get("query_mask")))
        source_ear_mask = aux.get("source_earring_mask")
        earring_confident_mask = aux.get("earring_confident_mask", source_ear_mask)
        earring_highlight_mask = aux.get("earring_highlight_mask")
        earring_reference = aux.get("earring_reference", source)
        revealed_skin_mask = aux.get("revealed_skin_mask")
        revealed_skin_blend_mask = aux.get("revealed_skin_blend_mask", revealed_skin_mask)
        fine_mask = aux.get("fine_mask")
        fine_mask_logits = aux.get("fine_mask_logits")
        presence_logits = aux.get("presence_logits")
        presence_target = aux.get("presence_target")
        visible_ear_roi = aux.get("visible_ear_roi")
        earring_valid_roi = aux.get("earring_valid_roi", visible_ear_roi)
        dataset_instance_authority = aux.get("earring_instance_is_dataset")
        if dataset_instance_authority is not None:
            dataset_instance_authority = ensure_mask_4d(dataset_instance_authority).to(
                device=query_mask.device,
                dtype=query_mask.dtype,
            )
            if dataset_instance_authority.shape[0] != query_mask.shape[0]:
                dataset_instance_authority = None
        source_hair_block_mask = aux.get("source_hair_block_mask")
        target_earring_suppress_mask = aux.get("target_earring_suppress_mask")
        target_ear_hair_occlusion_mask = aux.get("target_ear_hair_occlusion_mask")
        target_clean_01 = aux.get("target_clean_01")
        gamma = aux.get("gamma")
        beta = aux.get("beta")

        if query_mask is None or source_ear_mask is None:
            return losses

        if earring_valid_roi is not None:
            earring_valid_roi = resize_mask(earring_valid_roi, query_mask.shape[-2:])

            # The query builder already computes a per-side target ear/lobe
            # visibility decision.  Re-estimating an "earlobe" from the lower
            # 40% of the entire tensor can disagree with that decision on
            # crops/profile faces and silently erase a valid long-earring
            # label.  Keep this loss aligned with the explicit side gate used
            # by dataset generation and final inference.

            query_mask = ensure_mask_4d(query_mask).float() * earring_valid_roi
            source_ear_mask = ensure_mask_4d(source_ear_mask).float()
            earring_confident_mask = ensure_mask_4d(earring_confident_mask).float()
            if dataset_instance_authority is None:
                source_ear_mask = source_ear_mask * earring_valid_roi
                earring_confident_mask = earring_confident_mask * earring_valid_roi
            else:
                authority = dataset_instance_authority.expand_as(earring_valid_roi).clamp(0, 1)
                # A saved ring can extend beyond the compact ear ROI.  Dataset
                # authority keeps the complete object while legacy batches
                # retain the historical visible-ear clipping.
                source_ear_mask = source_ear_mask * (
                    authority + earring_valid_roi * (1.0 - authority)
                ).clamp(0, 1)
                earring_confident_mask = earring_confident_mask * (
                    authority + earring_valid_roi * (1.0 - authority)
                ).clamp(0, 1)
            if earring_highlight_mask is not None:
                earring_highlight_mask = ensure_mask_4d(earring_highlight_mask).float() * earring_valid_roi

        if source_hair_block_mask is not None and not uses_safe_query:
            source_hair_block_mask = resize_mask(source_hair_block_mask, query_mask.shape[-2:])
            block_strength = max(0.0, min(1.0, float(self.losses_dict.get("ear_block_strength", 0.95))))
            query_mask = query_mask * (1 - block_strength * source_hair_block_mask).clamp(0, 1)

        cleanup_regular_mask = cleanup_mask
        revealed_skin_mask_256 = None
        if revealed_skin_mask is not None:
            revealed_skin_mask_256 = resize_mask(revealed_skin_mask, gen_F_256_01.shape[-2:])
            if cleanup_regular_mask is not None:
                cleanup_regular_mask = resize_mask(cleanup_regular_mask, gen_F_256_01.shape[-2:])
                cleanup_exclude = revealed_skin_mask_256
                if revealed_skin_blend_mask is not None:
                    cleanup_exclude = resize_mask(revealed_skin_blend_mask, gen_F_256_01.shape[-2:])
                cleanup_regular_mask = cleanup_regular_mask * (1.0 - cleanup_exclude).clamp(0, 1)

        if cleanup_mask is not None:
            cleanup_mask = resize_mask(cleanup_mask, gen_F_256_01.shape[-2:])
            if cleanup_regular_mask is None:
                cleanup_regular_mask = cleanup_mask
            cleanup_low_anchor_weight = self.losses_dict.get(
                "cleanup_low_anchor",
                self.losses_dict.get("cleanup_anchor", 0.0),
            )
            if cleanup_low_anchor_weight > 0:
                losses["cleanup_low_anchor"] = cleanup_low_anchor_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    cleanup_regular_mask,
                )

            cleanup_source_reject_weight = self.losses_dict.get("cleanup_source_reject", 0.0)
            if cleanup_source_reject_weight > 0:
                losses["cleanup_source_reject"] = cleanup_source_reject_weight * source_direction_leak_loss(
                    source,
                    target,
                    gen_F_256_01,
                    cleanup_regular_mask,
                )

            cleanup_non_dark_weight = self.losses_dict.get("cleanup_non_dark", 0.0)
            if cleanup_non_dark_weight > 0:
                losses["cleanup_non_dark"] = cleanup_non_dark_weight * masked_non_darker(
                    rgb_to_gray(gen_F_256_01),
                    rgb_to_gray(target),
                    cleanup_regular_mask,
                )

            face_surface = parsing_label_mask(aux.get("source_parsing"), RAW_FACE_SURFACE_LABELS)
            source_hair_for_texture = aux.get("source_hair_mask")
            target_hair_for_texture = aux.get("target_hair_mask")
            if face_surface is not None:
                visible_skin_ref = resize_mask(face_surface, gen_F_256_01.shape[-2:])
                if source_hair_for_texture is not None:
                    visible_skin_ref = visible_skin_ref * (
                        1.0 - resize_mask(source_hair_for_texture, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)
                if target_hair_for_texture is not None:
                    visible_skin_ref = visible_skin_ref * (
                        1.0 - resize_mask(target_hair_for_texture, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)
                visible_skin_ref = visible_skin_ref * (1.0 - cleanup_mask).clamp(0, 1)

                cleanup_texture_weight = self.losses_dict.get("cleanup_texture_stat", 0.0)
                if cleanup_texture_weight > 0:
                    losses["cleanup_texture_stat"] = cleanup_texture_weight * texture_stat_loss(
                        gen_F_256_01,
                        source,
                        cleanup_regular_mask,
                        visible_skin_ref,
                    )

                cleanup_high_weight = self.losses_dict.get("cleanup_high", 0.0)
                cleanup_source_skin = cleanup_regular_mask * resize_mask(face_surface, gen_F_256_01.shape[-2:])
                if source_hair_for_texture is not None:
                    cleanup_source_skin = cleanup_source_skin * (
                        1.0 - resize_mask(source_hair_for_texture, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)
                if cleanup_high_weight > 0:
                    losses["cleanup_high"] = cleanup_high_weight * masked_l1(
                        high_pass_filter(gen_F_256_01),
                        high_pass_filter(source),
                        cleanup_source_skin,
                    )

        loss_size = gen_F_256_01.shape[-2:]
        loss_mask_aux = dict(aux)
        loss_mask_aux["ear_detail_query_mask"] = query_mask
        loss_mask_aux["source_earring_mask"] = source_ear_mask
        loss_mask_aux["earring_confident_mask"] = earring_confident_mask
        if earring_highlight_mask is not None:
            loss_mask_aux["earring_highlight_mask"] = earring_highlight_mask
        if fine_mask is not None:
            loss_mask_aux["fine_mask"] = fine_mask

        local_edit_mask = build_safe_earring_edit_mask(
            loss_mask_aux,
            loss_size,
            gen_F_256_01,
        )
        aux["safe_local_edit_mask"] = local_edit_mask

        target_preserve_weight = self.losses_dict.get("target_preserve", 0.0)
        if target_preserve_weight > 0:
            target_preserve_mask = build_target_preserve_mask(
                loss_mask_aux,
                loss_size,
                gen_F_256_01,
                local_edit_mask,
                int(self.losses_dict.get("target_preserve_exclude_dilate", 9)),
            )
            aux["target_preserve_loss_mask"] = target_preserve_mask
            losses["target_preserve"] = target_preserve_weight * (
                masked_l1(gen_F_256_01, target, target_preserve_mask)
                + 0.75
                * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    target_preserve_mask,
                )
            )

        face_dark_reject_weight = self.losses_dict.get("face_source_dark_reject", 0.0)
        if face_dark_reject_weight > 0:
            face_dark_reject_mask = build_face_source_dark_reject_mask(
                loss_mask_aux,
                loss_size,
                gen_F_256_01,
                local_edit_mask,
                source_hair_dilate=int(
                    self.losses_dict.get("face_source_dark_source_hair_dilate", 7)
                ),
                detail_exclude_dilate=int(
                    self.losses_dict.get("face_source_dark_detail_exclude_dilate", 5)
                ),
                ear_exclude_dilate=int(
                    self.losses_dict.get("face_source_dark_ear_exclude_dilate", 7)
                ),
            )
            if face_dark_reject_mask is not None:
                aux["face_source_dark_reject_mask"] = face_dark_reject_mask
                losses["face_source_dark_reject"] = (
                    face_dark_reject_weight
                    * source_dark_high_reject_loss(
                        source,
                        target,
                        gen_F_256_01,
                        face_dark_reject_mask,
                        margin=float(
                            self.losses_dict.get("face_source_dark_reject_margin", 0.006)
                        ),
                        source_threshold=float(
                            self.losses_dict.get(
                                "face_source_dark_reject_source_threshold",
                                0.018,
                            )
                        ),
                        kernel_size=int(
                            self.losses_dict.get("face_source_dark_reject_kernel", 11)
                        ),
                    )
                )

        target_hair_preserve_weight = self.losses_dict.get("target_hair_preserve", 0.0)
        if target_hair_preserve_weight > 0:
            target_hair_preserve_mask = build_target_hair_preserve_mask(
                loss_mask_aux,
                loss_size,
                gen_F_256_01,
            )
            if target_hair_preserve_mask is not None:
                aux["target_hair_preserve_loss_mask"] = target_hair_preserve_mask
                losses["target_hair_preserve"] = target_hair_preserve_weight * (
                    masked_l1(gen_F_256_01, target, target_hair_preserve_mask)
                    + masked_l1(
                        low_pass_filter(gen_F_256_01),
                        low_pass_filter(target),
                        target_hair_preserve_mask,
                    )
                )

        revealed_skin_weight = self.losses_dict.get("revealed_skin_texture", 0.0)
        revealed_tone_weight = self.losses_dict.get("revealed_skin_tone", 0.0)
        if revealed_skin_weight > 0 or revealed_tone_weight > 0:
            revealed_refs = build_revealed_skin_reference_masks(
                loss_mask_aux,
                loss_size,
                gen_F_256_01,
            )
            if revealed_refs is not None:
                revealed_mask, _source_texture_ref, recovered_mask = revealed_refs
                aux["revealed_skin_texture_loss_mask"] = revealed_mask
                aux["revealed_skin_recovered_ref_mask"] = recovered_mask
                revealed_loss = gen_F_256_01.sum() * 0.0
                if revealed_skin_weight > 0:
                    # The source fringe-hidden area is not skin GT.  This term
                    # is intentionally a local target/PP boundary continuity
                    # constraint, not a source texture-copy objective.
                    revealed_loss = revealed_loss + revealed_skin_weight * revealed_skin_tone_continuity_loss(
                        gen_F_256_01,
                        revealed_mask,
                        recovered_mask,
                    )
                if revealed_tone_weight > 0:
                    revealed_loss = revealed_loss + revealed_tone_weight * (
                        revealed_skin_tone_continuity_loss(
                            gen_F_256_01,
                            revealed_mask,
                            recovered_mask,
                        )
                        + revealed_skin_tone_continuity_loss(
                            gen_w_256_01 if gen_w_256_01 is not None else gen_F_256_01,
                            revealed_mask,
                            recovered_mask,
                        )
                    )
                losses["revealed_skin_texture"] = revealed_loss

        normal_face_weight = self.losses_dict.get("normal_face_preserve", 0.0)
        if normal_face_weight > 0:
            normal_face = aux.get("target_face_surface_mask")
            if normal_face is None:
                normal_face = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
            if normal_face is not None:
                normal_face = resize_mask(normal_face, gen_F_256_01.shape[-2:])
                if revealed_skin_mask is not None:
                    normal_face = normal_face * (
                        1.0 - dilate_mask(resize_mask(revealed_skin_mask, gen_F_256_01.shape[-2:]), 5)
                    ).clamp(0, 1)
                target_hair = aux.get("target_hair_mask")
                if target_hair is not None:
                    normal_face = normal_face * (
                        1.0 - resize_mask(target_hair, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)
                losses["normal_face_preserve"] = normal_face_weight * (
                    masked_l1(gen_F_256_01, target, normal_face)
                    + 0.5 * masked_l1(
                        low_pass_filter(gen_F_256_01), low_pass_filter(target), normal_face
                    )
                )

        # V5 source-visible face supervision uses real source skin as an
        # identity/detail anchor.  It is a loss mask only; the final
        # compositor never hard-cuts this region back to source or target.
        source_valid_face = aux.get("source_visible_skin_reference_mask", aux.get("source_skin_valid_mask"))
        if source_valid_face is not None:
            source_valid_face = resize_mask(source_valid_face, gen_F_256_01.shape[-2:])
            earring_guard = resize_mask(earring_confident_mask, gen_F_256_01.shape[-2:])
            source_valid_face = source_valid_face * (1.0 - dilate_mask(earring_guard, 3)).clamp(0, 1)
            source_face_reference = aux.get("source_face_reference_01", source)
            source_face_reference = F.interpolate(
                source_face_reference,
                size=gen_F_256_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0, 1)
            source_valid_weight = float(self.losses_dict.get("source_valid_face_high", 0.35))
            source_color_weight = float(self.losses_dict.get("source_valid_face_color", 0.15))
            if source_valid_weight > 0:
                losses["source_valid_face_high"] = source_valid_weight * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(source_face_reference),
                    source_valid_face,
                )
            if source_color_weight > 0:
                losses["source_valid_face_color"] = source_color_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(source_face_reference),
                    source_valid_face,
                )

        # V5 trains the exact composed output against artificial revealed-skin
        # boundaries.  These are local continuity constraints, never a request
        # to copy source bangs into a newly exposed forehead.
        revealed_boundary_weight = float(self.losses_dict.get("revealed_boundary_seam", 0.0))
        revealed_texture_stat_weight = float(self.losses_dict.get("revealed_texture_stat", 0.0))
        face_lowfreq_weight = float(self.losses_dict.get("face_lowfreq_continuity", 0.0))
        revealed_for_v5 = aux.get("revealed_skin_mask")
        face_for_v5 = aux.get("target_face_surface_mask")
        if face_for_v5 is None:
            face_for_v5 = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
        if (
            revealed_for_v5 is not None
            and face_for_v5 is not None
            and (revealed_boundary_weight > 0 or revealed_texture_stat_weight > 0 or face_lowfreq_weight > 0)
        ):
            revealed_for_v5 = resize_mask(revealed_for_v5, gen_F_256_01.shape[-2:])
            face_for_v5 = resize_mask(face_for_v5, gen_F_256_01.shape[-2:])
            aux["v5_revealed_continuity_mask"] = revealed_for_v5
            aux["v5_face_continuity_mask"] = face_for_v5
            if face_lowfreq_weight > 0:
                losses["face_lowfreq_continuity"] = face_lowfreq_weight * face_lowfreq_continuity_loss(
                    gen_F_256_01,
                    face_for_v5,
                    revealed_for_v5,
                )
            if revealed_boundary_weight > 0:
                losses["revealed_boundary_seam"] = revealed_boundary_weight * revealed_boundary_seam_loss(
                    gen_F_256_01,
                    face_for_v5,
                    revealed_for_v5,
                )
            if revealed_texture_stat_weight > 0:
                losses["revealed_texture_stat"] = revealed_texture_stat_weight * revealed_texture_stat_loss(
                    gen_F_256_01,
                    face_for_v5,
                    revealed_for_v5,
                )

        # V5 evaluates the rendered final compositor at 512px or native
        # resolution. Revealed regions are compared only to adjacent final
        # skin statistics; source RGB is used exclusively in valid skin/detail.
        if bool(self.losses_dict.get("enable_v5_structural_compositor", False)):
            final_hr = ((gen_F + 1.0) * 0.5).clamp(0, 1)
            max_side = max(final_hr.shape[-2:])
            if max_side > 512:
                scale = 512.0 / float(max_side)
                hr_size = (
                    max(1, int(round(final_hr.shape[-2] * scale))),
                    max(1, int(round(final_hr.shape[-1] * scale))),
                )
                final_hr = F.interpolate(final_hr, size=hr_size, mode="bilinear", align_corners=False)
            else:
                hr_size = tuple(final_hr.shape[-2:])
            face_hr = aux.get("output_v5_face_face_surface", aux.get("target_face_surface_mask"))
            valid_hr = aux.get("output_v5_face_source_valid_skin", source_valid_face)
            revealed_hr = aux.get("output_v5_face_revealed_skin", revealed_for_v5)
            hair_hr = aux.get("output_v5_face_target_hair_soft_alpha", aux.get("target_hair_mask"))
            if face_hr is not None:
                face_hr = resize_mask(face_hr, hr_size)
                source_hr = aux.get("source_face_reference_01", aux.get("source_full_01", source))
                source_hr = F.interpolate(source_hr, size=hr_size, mode="bilinear", align_corners=False).clamp(0, 1)
                if valid_hr is not None:
                    valid_hr = resize_mask(valid_hr, hr_size)
                    high_weight = float(self.losses_dict.get("face_source_detail_hr", 0.0))
                    color_weight = float(self.losses_dict.get("face_lowfreq_anchor_hr", 0.0))
                    if high_weight > 0:
                        losses["face_source_detail_hr"] = high_weight * masked_l1(
                            high_pass_filter(final_hr), high_pass_filter(source_hr), valid_hr
                        )
                    if color_weight > 0:
                        losses["face_lowfreq_anchor_hr"] = color_weight * masked_l1(
                            low_pass_filter(final_hr), low_pass_filter(source_hr), valid_hr
                        )
                continuity_weight = float(self.losses_dict.get("face_lowfreq_continuity_hr", 0.0))
                if continuity_weight > 0:
                    continuity = face_lowfreq_smoothness_loss(final_hr, face_hr)
                    if revealed_hr is not None:
                        continuity = continuity + face_lowfreq_continuity_loss(
                            final_hr, face_hr, resize_mask(revealed_hr, hr_size)
                        )
                    losses["face_lowfreq_continuity_hr"] = continuity_weight * continuity
                revealed_weight_hr = float(self.losses_dict.get("revealed_boundary_seam_hr", 0.0))
                if revealed_weight_hr > 0 and revealed_hr is not None:
                    losses["revealed_boundary_seam_hr"] = revealed_weight_hr * revealed_boundary_seam_loss(
                        final_hr, face_hr, resize_mask(revealed_hr, hr_size)
                    )
                # The source fringe-covered pixels are not an RGB target, but
                # their recovered skin must have the same *amount* of pore/
                # fine-detail energy as trusted visible source skin.  This
                # compares only aggregate high-pass and edge statistics, never
                # moves a source eyebrow, bang or shadow into the forehead.
                source_texture_weight = float(
                    self.losses_dict.get("revealed_source_texture_stat_hr", 0.0)
                )
                if (
                    source_texture_weight > 0
                    and revealed_hr is not None
                    and valid_hr is not None
                ):
                    losses["revealed_source_texture_stat_hr"] = (
                        source_texture_weight
                        * texture_stat_loss(
                            final_hr,
                            source_hr,
                            resize_mask(revealed_hr, hr_size),
                            valid_hr,
                        )
                    )
                hair_weight = float(self.losses_dict.get("face_hair_boundary_seam_hr", 0.0))
                if hair_weight > 0 and hair_hr is not None:
                    losses["face_hair_boundary_seam_hr"] = hair_weight * face_hair_boundary_seam_loss(
                        final_hr, face_hr, resize_mask(hair_hr, hr_size)
                    )

        earring_confident_mask = resize_mask(earring_confident_mask, gen_F_256_01.shape[-2:])
        earring_reference = F.interpolate(earring_reference, size=gen_F_256_01.shape[-2:], mode="bilinear", align_corners=False)
        earring_write_mask = resize_mask(
            aux.get("earring_write_mask", earring_confident_mask),
            gen_F_256_01.shape[-2:],
        )
        hoop_hole_mask = aux.get("hoop_hole_mask")
        if hoop_hole_mask is not None:
            hoop_hole_mask = resize_mask(hoop_hole_mask, gen_F_256_01.shape[-2:])
            # An inner hoop region belongs to the already transferred target,
            # never to source-object reconstruction.
            earring_write_mask = earring_write_mask * (1.0 - hoop_hole_mask).clamp(0, 1)
            earring_confident_mask = earring_confident_mask * (1.0 - hoop_hole_mask).clamp(0, 1)
        no_earring_case = aux.get("no_earring_case_mask", aux.get("no_earring_case"))
        if no_earring_case is not None:
            no_earring_case = resize_mask(no_earring_case, gen_F_256_01.shape[-2:])
        target_ear_boundary = aux.get("target_ear_boundary_protect_mask")
        if target_ear_boundary is not None:
            target_ear_boundary = resize_mask(target_ear_boundary, gen_F_256_01.shape[-2:])
            ear_geometry_weight = self.losses_dict.get("target_ear_geometry", 0.0)
            if ear_geometry_weight > 0:
                losses["target_ear_geometry"] = ear_geometry_weight * (
                    masked_l1(gen_F_256_01, target, target_ear_boundary)
                    + 0.5 * masked_l1(
                        sobel_edges(gen_F_256_01),
                        sobel_edges(target),
                        target_ear_boundary,
                    )
                )

        ear_roi_for_noop = aux.get("ear_roi", aux.get("visible_ear_roi"))
        no_earring_weight = self.losses_dict.get("no_earring_noop", 0.0)
        if no_earring_weight > 0 and no_earring_case is not None and ear_roi_for_noop is not None:
            noop_mask = resize_mask(ear_roi_for_noop, gen_F_256_01.shape[-2:]) * no_earring_case
            losses["no_earring_noop"] = no_earring_weight * (
                masked_l1(gen_F_256_01, target, noop_mask)
                + 0.5 * masked_l1(
                    low_pass_filter(gen_F_256_01), low_pass_filter(target), noop_mask
                )
            )

        hoop_hole_weight = self.losses_dict.get("hoop_hole_preserve", 0.0)
        if hoop_hole_weight > 0 and hoop_hole_mask is not None:
            losses["hoop_hole_preserve"] = hoop_hole_weight * masked_l1(
                gen_F_256_01, target, hoop_hole_mask
            )

        earring_restore_weight = self.losses_dict.get("earring_object_restore", 0.0)
        if earring_restore_weight > 0:
            # Object restoration is strictly confined to the final write mask;
            # no bounding box, search mask or background patch is supervised.
            losses["earring_object_restore"] = earring_restore_weight * (
                masked_l1(gen_F_256_01, earring_reference, earring_write_mask)
                + 0.5 * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(earring_reference),
                    earring_write_mask,
                )
            )
        foreground_restore_weight = self.losses_dict.get(
            "earring_foreground_restore",
            0.0,
        )
        target_hair_for_earring = aux.get("target_hair_mask")
        if foreground_restore_weight > 0 and target_hair_for_earring is not None:
            # The source-native instance is explicitly in front of target hair
            # in these pixels.  This removes the old conflict where the broad
            # target-hair preservation term taught a long pendant to disappear
            # below the exposed lobe.  Hoop centres are already removed from
            # ``earring_write_mask`` above and remain target-owned.
            foreground_mask = earring_write_mask * resize_mask(
                target_hair_for_earring,
                gen_F_256_01.shape[-2:],
            )
            losses["earring_foreground_restore"] = foreground_restore_weight * (
                masked_l1(gen_F_256_01, earring_reference, foreground_mask)
                + 0.5 * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(earring_reference),
                    foreground_mask,
                )
            )
        if target_earring_suppress_mask is not None:
            target_earring_suppress_mask = resize_mask(target_earring_suppress_mask, gen_F_256_01.shape[-2:])
            if target_clean_01 is None:
                target_clean_01 = low_pass_filter(target)
            else:
                target_clean_01 = F.interpolate(
                    target_clean_01,
                    size=gen_F_256_01.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0, 1)

            suppress_low_weight = self.losses_dict.get("target_earring_suppress_low", 0.0)
            if suppress_low_weight > 0:
                losses["target_earring_suppress_low"] = suppress_low_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target_clean_01),
                    target_earring_suppress_mask,
                )

            suppress_high_weight = self.losses_dict.get("target_earring_suppress_high", 0.0)
            if suppress_high_weight > 0:
                losses["target_earring_suppress_high"] = suppress_high_weight * masked_mean(
                    high_pass_filter(gen_F_256_01).abs(),
                    target_earring_suppress_mask,
                )

        occluded_hair_weight = self.losses_dict.get("occluded_hair_anchor", 0.0)
        if occluded_hair_weight > 0 and target_ear_hair_occlusion_mask is not None:
            occluded_hair_mask = resize_mask(
                target_ear_hair_occlusion_mask,
                gen_F_256_01.shape[-2:],
            )
            earring_hair_override = build_v58_earring_keep_mask(
                aux,
                gen_F_256_01.shape[-2:],
                gen_F_256_01,
            )
            occluded_hair_mask = occluded_hair_mask * (1.0 - earring_hair_override).clamp(0, 1)
            losses["occluded_hair_anchor"] = occluded_hair_weight * (
                masked_l1(gen_F_256_01, target, occluded_hair_mask)
                + 0.5 * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    occluded_hair_mask,
                )
                + 0.25 * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(target),
                    occluded_hair_mask,
                )
            )

        detail_mask = parsing_label_mask(aux.get("source_parsing"), RAW_DETAIL_LABELS)
        if detail_mask is not None:
            detail_mask = resize_mask(detail_mask, gen_F_256_01.shape[-2:])
            target_hair_mask = aux.get("target_hair_mask")
            if target_hair_mask is not None:
                detail_mask = detail_mask * (1 - resize_mask(target_hair_mask, gen_F_256_01.shape[-2:])).clamp(0, 1)
            if source_hair_block_mask is not None:
                detail_mask = detail_mask * (1 - resize_mask(source_hair_block_mask, gen_F_256_01.shape[-2:])).clamp(0, 1)
            if cleanup_mask is not None:
                detail_mask = detail_mask * (1 - cleanup_mask).clamp(0, 1)
            if target_ear_boundary is not None:
                detail_mask = detail_mask * (1.0 - target_ear_boundary).clamp(0, 1)
            # Earring restoration has its own object-only supervision.  Do not
            # let the generic face-detail loss rewrite the same ear region.
            detail_mask = detail_mask * (1.0 - dilate_mask(earring_write_mask, 3)).clamp(0, 1)
            detail_high_weight = self.losses_dict.get("detail_high", 0.0)
            if detail_high_weight > 0:
                losses["detail_high"] = detail_high_weight * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(source),
                    detail_mask,
                )

            detail_low_anchor_weight = self.losses_dict.get("detail_low_anchor", 0.0)
            if detail_low_anchor_weight > 0:
                losses["detail_low_anchor"] = detail_low_anchor_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    detail_mask,
                )

        earring_supervision_dilate = int(self.losses_dict.get("earring_supervision_dilate", 5))
        earring_supervision = dilate_mask(earring_confident_mask, earring_supervision_dilate) * query_mask
        weak_pseudo_mask = torch.zeros_like(earring_confident_mask)
        if earring_confident_mask.detach().sum().item() < 1:
            weak_pseudo_mask, _ = build_weak_ear_pseudo_mask(
                earring_reference,
                query_mask,
                earring_confident_mask,
            )
        if no_earring_case is not None:
            weak_pseudo_mask = weak_pseudo_mask * (1.0 - no_earring_case).clamp(0, 1)
        mask_target = torch.clamp(
            weak_pseudo_mask + earring_supervision + earring_confident_mask,
            0,
            1,
        )
        if hoop_hole_mask is not None:
            mask_target = mask_target * (1.0 - hoop_hole_mask).clamp(0, 1)
        query_expand = float(self.losses_dict.get("ear_query_expand", 0.02))
        supervision_mask = torch.clamp(mask_target + query_expand * query_mask, 0, 1)
        if fine_mask is not None:
            supervision_mask = torch.clamp(supervision_mask + 0.5 * fine_mask.detach(), 0, 1)

        mask_weight = self.losses_dict.get("ear_mask", 0.0)
        if mask_weight > 0 and fine_mask_logits is not None:
            valid_mask = torch.clamp(query_mask + mask_target, 0, 1)
            bce_map = self.mask_bce(fine_mask_logits, mask_target.float())
            losses["ear_mask"] = mask_weight * masked_mean(bce_map, valid_mask)

        area_weight = self.losses_dict.get("ear_mask_area", 0.0)
        if area_weight > 0 and fine_mask is not None:
            overflow = F.relu(fine_mask - mask_target)
            losses["ear_mask_area"] = area_weight * masked_ratio(overflow, query_mask)

        high_weight = self.losses_dict.get("ear_high", 0.0)
        if high_weight > 0:
            source_high = high_pass_filter(earring_reference)
            gen_high = high_pass_filter(gen_F_256_01)
            losses["ear_high"] = high_weight * masked_l1(gen_high, source_high, supervision_mask)

        color_weight = self.losses_dict.get("ear_color", 0.0)
        if color_weight > 0:
            losses["ear_color"] = color_weight * (
                masked_l1(gen_F_256_01, earring_reference, earring_confident_mask)
                + 0.5 * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(earring_reference),
                    earring_confident_mask,
                )
            )

        highlight_weight = self.losses_dict.get("ear_highlight", 0.0)
        if highlight_weight > 0 and earring_highlight_mask is not None:
            earring_highlight_mask = resize_mask(earring_highlight_mask, gen_F_256_01.shape[-2:])
            losses["ear_highlight"] = highlight_weight * (
                masked_l1(gen_F_256_01, earring_reference, earring_highlight_mask)
                + masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(earring_reference),
                    earring_highlight_mask,
                )
                + masked_non_darker(
                    rgb_to_gray(gen_F_256_01),
                    rgb_to_gray(earring_reference),
                    earring_highlight_mask,
                )
            )

        lighting_weight = self.losses_dict.get("ear_lighting", 0.0)
        if lighting_weight > 0:
            target_low = low_pass_filter(target)
            gen_low = low_pass_filter(gen_F_256_01)
            lighting_mask = torch.clamp(query_mask + (fine_mask if fine_mask is not None else 0), 0, 1)
            lighting_mask = lighting_mask * (1.0 - earring_confident_mask).clamp(0, 1)
            losses["ear_lighting"] = lighting_weight * masked_l1(gen_low, target_low, lighting_mask)

        leak_weight = self.losses_dict.get("ear_hair_leak", 0.0)
        if leak_weight > 0 and source_hair_block_mask is not None:
            source_hair_block_mask = ensure_mask_4d(source_hair_block_mask).float()
            block_strength = max(0.0, min(1.0, float(self.losses_dict.get("ear_block_strength", 0.95))))
            leak_mask = source_hair_block_mask * block_strength * (1.0 - earring_confident_mask).clamp(0, 1)
            losses["ear_hair_leak"] = leak_weight * source_direction_leak_loss(
                source,
                target,
                gen_F_256_01,
                leak_mask,
            )

            anchor_weight = self.losses_dict.get("ear_hair_anchor", 0.0)
            if anchor_weight > 0:
                losses["ear_hair_anchor"] = anchor_weight * (
                    masked_l1(gen_F_256_01, target, leak_mask)
                    + 0.5 * masked_l1(low_pass_filter(gen_F_256_01), low_pass_filter(target), leak_mask)
                )

        edge_weight = self.losses_dict.get("ear_edge", 0.0)
        if edge_weight > 0:
            source_edge = sobel_edges(earring_reference)
            gen_edge = sobel_edges(gen_F_256_01)
            losses["ear_edge"] = edge_weight * masked_l1(gen_edge, source_edge, supervision_mask)

        presence_weight = self.losses_dict.get("ear_presence", 0.0)
        if presence_weight > 0 and presence_logits is not None:
            weak_presence_target = build_weak_presence_target(mask_target, earring_confident_mask * query_mask)
            if presence_target is not None:
                weak_presence_target = torch.maximum(weak_presence_target, presence_target.float())
            losses["ear_presence"] = presence_weight * self.presence_bce(presence_logits, weak_presence_target)

        brightness_weight = self.losses_dict.get("ear_brightness_reg", 0.0)
        if brightness_weight > 0 and gamma is not None and beta is not None:
            losses["ear_brightness_reg"] = brightness_weight * ((gamma - 1).abs().mean() + beta.abs().mean())

        return losses
