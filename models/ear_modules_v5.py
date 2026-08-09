from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the project runtime normally provides OpenCV.
    cv2 = None

from models.CtrlHair.external_code.face_parsing.my_parsing_util import FaceParsing_tensor
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample

RAW_LEFT_EAR = 7
RAW_RIGHT_EAR = 8
RAW_EARRING = 9
RAW_FACE_SURFACE_LABELS = (1, 10)
RAW_SKIN_SURFACE_LABELS = (1, 7, 8, 10)
RAW_EAR_SURFACE_LABELS = (7, 8)
RAW_HAIR = 17
RAW_HAT = 18
RAW_DETAIL_LABELS = (2, 3, 4, 5, 6, 11, 12, 13)


def ensure_mask_4d(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    return mask


def normalized_to_01(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        return images.float().div(255)
    if images.min().item() < -0.05:
        return ((images + 1) / 2).clamp(0, 1)
    return images.clamp(0, 1)


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    return F.interpolate(mask, size=size, mode="bilinear", align_corners=False).clamp(0, 1)


def dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return (mask > 0).float()
    return F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size // 2)


def erode_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    return 1 - dilate_mask(1 - mask, kernel_size)


def shift_mask(mask: torch.Tensor, down: int = 0, right: int = 0) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    shifted = torch.zeros_like(mask)

    h_slice_src = slice(0, mask.size(-2) - max(0, down))
    h_slice_dst = slice(max(0, down), mask.size(-2))
    if down < 0:
        h_slice_src = slice(-down, mask.size(-2))
        h_slice_dst = slice(0, mask.size(-2) + down)

    w_slice_src = slice(0, mask.size(-1) - max(0, right))
    w_slice_dst = slice(max(0, right), mask.size(-1))
    if right < 0:
        w_slice_src = slice(-right, mask.size(-1))
        w_slice_dst = slice(0, mask.size(-1) + right)

    shifted[..., h_slice_dst, w_slice_dst] = mask[..., h_slice_src, w_slice_src]
    return shifted


def gaussian_blur(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return image

    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / max(2 * sigma ** 2, 1e-6))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d[None, None, ...].expand(image.size(1), 1, -1, -1)
    return F.conv2d(image, kernel_2d, padding=kernel_size // 2, groups=image.size(1))


def low_pass_filter(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    return gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)


def high_pass_filter(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    return image - low_pass_filter(image, kernel_size=kernel_size, sigma=sigma)


def parsing_label_mask(parsing: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    parsing = ensure_mask_4d(parsing).long()
    mask = torch.zeros_like(parsing, dtype=torch.bool)
    for label in labels:
        mask |= parsing == label
    return mask.float()


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_gray(image) if image.size(1) != 1 else image
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=image.device, dtype=image.dtype)
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def _masked_threshold_candidate(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_ratio: float = 0.45,
    std_ratio: float = 1.0,
    floor: float = 0.0,
) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    value = value * mask
    flat_value = value.flatten(2)
    flat_mask = mask.flatten(2)
    denom = flat_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    mean = (flat_value.sum(dim=2, keepdim=True) / denom).view(-1, 1, 1, 1)
    centered = (flat_value - mean.flatten(1).unsqueeze(-1)).pow(2) * flat_mask
    std = torch.sqrt(centered.sum(dim=2, keepdim=True) / denom).view(-1, 1, 1, 1)
    peak = flat_value.amax(dim=2, keepdim=True).view(-1, 1, 1, 1)
    threshold = torch.maximum(max_ratio * peak, mean + std_ratio * std)
    if floor > 0:
        threshold = torch.maximum(threshold, torch.full_like(threshold, floor))
    return (value >= threshold).float() * mask


def build_weak_earring_mask(
    source_01: torch.Tensor,
    query_mask: torch.Tensor,
    parser_mask: torch.Tensor | None = None,
    source_hair_block_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    query_mask = ensure_mask_4d(query_mask).float()
    parser_mask = torch.zeros_like(query_mask) if parser_mask is None else resize_mask(parser_mask, query_mask.shape[-2:])
    source_hair_block_mask = (
        torch.zeros_like(query_mask)
        if source_hair_block_mask is None
        else resize_mask(source_hair_block_mask, query_mask.shape[-2:])
    )

    high_energy = high_pass_filter(source_01).abs().mean(dim=1, keepdim=True) * query_mask
    edge_energy = sobel_magnitude(source_01) * query_mask
    gray = rgb_to_gray(source_01)
    local_gray = low_pass_filter(gray, kernel_size=9, sigma=2.0)
    local_contrast = (gray - local_gray).abs() * query_mask
    bright_spike = F.relu(gray - local_gray) * query_mask
    dark_edge = edge_energy * F.relu(local_gray - gray) * query_mask
    color_spread = (source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)) * query_mask

    high_candidate = _masked_threshold_candidate(high_energy, query_mask, max_ratio=0.40, std_ratio=0.75)
    edge_candidate = _masked_threshold_candidate(edge_energy, query_mask, max_ratio=0.42, std_ratio=0.85)
    contrast_candidate = _masked_threshold_candidate(local_contrast, query_mask, max_ratio=0.40, std_ratio=0.90)
    bright_candidate = _masked_threshold_candidate(bright_spike, query_mask, max_ratio=0.45, std_ratio=1.10)
    dark_candidate = _masked_threshold_candidate(dark_edge, query_mask, max_ratio=0.38, std_ratio=0.75)
    color_candidate = _masked_threshold_candidate(color_spread, query_mask, max_ratio=0.42, std_ratio=0.90)

    weak_mask = (
        high_candidate * torch.clamp(edge_candidate + contrast_candidate, 0, 1)
        + bright_candidate
        + dark_candidate
        + color_candidate * torch.clamp(edge_candidate + contrast_candidate, 0, 1)
        + parser_mask * query_mask
    ).clamp(0, 1)
    weak_mask = weak_mask * (1.0 - 0.20 * source_hair_block_mask).clamp(0, 1)
    weak_mask = dilate_mask(weak_mask, 3) * query_mask

    return {
        "weak_earring_mask": weak_mask.clamp(0, 1),
        "weak_high_energy": high_energy,
        "weak_edge_energy": edge_energy,
        "weak_local_contrast": local_contrast,
        "weak_bright_spike": bright_spike,
        "weak_dark_edge": dark_edge,
        "weak_color_spread": color_spread,
    }


def build_weak_earring_masks(
    source_01: torch.Tensor,
    visible_ear_roi: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    parser_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    source_hair_block_mask: torch.Tensor | None = None,
    source_parsing: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    source_01 = normalized_to_01(source_01)
    image_size = tuple(source_01.shape[-2:])
    if visible_ear_roi is None:
        if query_mask is None:
            raise ValueError("visible_ear_roi or query_mask is required to build weak earring masks.")
        visible_ear_roi = query_mask
    visible_ear_roi = resize_mask(visible_ear_roi, image_size)
    query_mask = visible_ear_roi if query_mask is None else resize_mask(query_mask, image_size)
    parser_mask = torch.zeros_like(query_mask) if parser_mask is None else resize_mask(parser_mask, image_size)

    if source_parsing is not None:
        parsing = ensure_mask_4d(source_parsing).float()
        if tuple(parsing.shape[-2:]) != image_size:
            parsing = F.interpolate(parsing, size=image_size, mode="nearest")
        parser_mask = torch.clamp(parser_mask + (parsing.long() == RAW_EARRING).float(), 0, 1)
        source_ear_mask = parsing_label_mask(parsing, RAW_EAR_SURFACE_LABELS)
        # This is a detection/search corridor, not a write mask.  The previous
        # 18px lower extent was shorter than the radius of a normal hoop at
        # 256px, so parser-missed hoops were reduced to the few pixels beside
        # the lobe before the strong-object check even ran.  The final write
        # mask remains constrained to connected visual evidence below.  The
        # lower two shifts cover large but still face-local hoops; they do not
        # grant write access by themselves.
        source_lobe_roi = torch.clamp(
            dilate_mask(source_ear_mask, 13)
            + dilate_mask(shift_mask(source_ear_mask, down=8), 15)
            + dilate_mask(shift_mask(source_ear_mask, down=20), 13)
            + dilate_mask(shift_mask(source_ear_mask, down=38), 11)
            + dilate_mask(shift_mask(source_ear_mask, down=56), 9)
            + dilate_mask(shift_mask(source_ear_mask, down=80), 9)
            + dilate_mask(shift_mask(source_ear_mask, down=108), 7)
            + dilate_mask(parser_mask, 11),
            0,
            1,
        )
    else:
        source_lobe_roi = visible_ear_roi

    source_hair_mask = torch.zeros_like(query_mask) if source_hair_mask is None else resize_mask(source_hair_mask, image_size)
    source_hair_block_mask = (
        torch.zeros_like(query_mask)
        if source_hair_block_mask is None
        else resize_mask(source_hair_block_mask, image_size)
    )

    source_earring_clean_mask = parser_mask * source_lobe_roi
    search_mask = torch.clamp(
        query_mask
        + 0.35 * visible_ear_roi
        + 0.90 * source_lobe_roi
        + dilate_mask(source_earring_clean_mask, 7),
        0,
        1,
    )
    hair_attenuation = (1.0 - 0.10 * source_hair_mask - 0.25 * source_hair_block_mask).clamp(0, 1)
    search_mask = search_mask * hair_attenuation

    gray = rgb_to_gray(source_01)
    local_gray = low_pass_filter(gray, kernel_size=9, sigma=2.0)
    high_energy = high_pass_filter(source_01).abs().mean(dim=1, keepdim=True) * search_mask
    edge_energy = sobel_magnitude(source_01) * search_mask
    local_contrast = (gray - local_gray).abs() * search_mask
    bright_spike = F.relu(gray - local_gray) * search_mask
    dark_spike = F.relu(local_gray - gray) * search_mask
    dark_edge = edge_energy * dark_spike
    color_spread = (source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)) * search_mask

    high_candidate = _masked_threshold_candidate(high_energy, search_mask, max_ratio=0.38, std_ratio=0.80, floor=0.006)
    edge_candidate = _masked_threshold_candidate(edge_energy, search_mask, max_ratio=0.38, std_ratio=0.75, floor=0.006)
    contrast_candidate = _masked_threshold_candidate(local_contrast, search_mask, max_ratio=0.38, std_ratio=0.80, floor=0.006)
    bright_candidate = _masked_threshold_candidate(bright_spike, search_mask, max_ratio=0.40, std_ratio=0.85, floor=0.006)
    dark_candidate = _masked_threshold_candidate(dark_edge, search_mask, max_ratio=0.36, std_ratio=0.70, floor=0.002)
    color_candidate = _masked_threshold_candidate(color_spread, search_mask, max_ratio=0.42, std_ratio=0.80, floor=0.030)

    structure_gate = torch.clamp(edge_candidate + contrast_candidate, 0, 1)
    texture_core = high_candidate * structure_gate
    highlight_core = bright_candidate * torch.clamp(high_candidate + edge_candidate + color_candidate, 0, 1)
    dark_core = dark_candidate * structure_gate
    chroma_core = color_candidate * torch.clamp(edge_candidate + contrast_candidate + bright_candidate, 0, 1)
    parser_core = dilate_mask(source_earring_clean_mask, 3) * search_mask

    earring_candidate_mask = torch.clamp(
        texture_core + highlight_core + dark_core + chroma_core + parser_core,
        0,
        1,
    )
    earring_confident_mask = dilate_mask(earring_candidate_mask, 5) * search_mask
    earring_highlight_mask = torch.clamp(highlight_core + color_candidate * high_candidate, 0, 1)
    earring_highlight_mask = dilate_mask(earring_highlight_mask, 3) * earring_confident_mask

    return {
        "weak_earring_mask": earring_confident_mask.clamp(0, 1),
        "source_earring_clean_mask": source_earring_clean_mask.clamp(0, 1),
        "earring_confident_mask": earring_confident_mask.clamp(0, 1),
        "earring_highlight_mask": earring_highlight_mask.clamp(0, 1),
        "earring_candidate_mask": earring_candidate_mask.clamp(0, 1),
        "earring_search_mask": search_mask.clamp(0, 1),
        "source_lobe_search_mask": source_lobe_roi.clamp(0, 1),
        "weak_high_energy": high_energy,
        "weak_edge_energy": edge_energy,
        "weak_local_contrast": local_contrast,
        "weak_bright_spike": bright_spike,
        "weak_dark_edge": dark_edge,
        "weak_color_spread": color_spread,
    }


def enhance_query_with_earring_recall(
    query_mask: torch.Tensor,
    source_earring_mask: torch.Tensor | None,
    source_hair_block_mask: torch.Tensor | None,
    weak_masks: dict[str, torch.Tensor],
    *,
    visibility_mask: torch.Tensor | None = None,
    recall_dilate: int = 7,
    downward_shift: int = 10,
    lower_lobe_weight: float = 0.20,
    candidate_boost: float = 0.90,
    block_protect: float = 0.85,
) -> dict[str, torch.Tensor]:
    query_mask = ensure_mask_4d(query_mask).float()
    source_earring_mask = (
        torch.zeros_like(query_mask)
        if source_earring_mask is None
        else resize_mask(source_earring_mask, query_mask.shape[-2:])
    )
    source_hair_block_mask = (
        torch.zeros_like(query_mask)
        if source_hair_block_mask is None
        else resize_mask(source_hair_block_mask, query_mask.shape[-2:])
    )
    visibility_mask = (
        torch.ones_like(query_mask)
        if visibility_mask is None
        else resize_mask(visibility_mask, query_mask.shape[-2:])
    )
    query_mask = query_mask * visibility_mask

    def weak_mask(name: str) -> torch.Tensor:
        value = weak_masks.get(name)
        if value is None:
            return torch.zeros_like(query_mask)
        return resize_mask(value, query_mask.shape[-2:])

    candidate_mask = weak_mask("earring_candidate_mask")
    clean_parser_mask = weak_mask("source_earring_clean_mask")
    search_mask = weak_mask("earring_search_mask")
    lobe_search_mask = weak_mask("source_lobe_search_mask")

    recall_seed = torch.clamp(source_earring_mask + clean_parser_mask + candidate_mask, 0, 1)
    object_detection_mask = dilate_mask(recall_seed, max(3, recall_dilate))
    object_recall_mask = object_detection_mask * visibility_mask
    query_recall_mask = object_detection_mask
    if downward_shift > 0:
        query_recall_mask = torch.clamp(query_recall_mask + shift_mask(query_recall_mask, down=downward_shift), 0, 1)
    query_recall_mask = query_recall_mask * visibility_mask

    has_recall = (recall_seed.flatten(1).amax(dim=1) > 0).float().view(-1, 1, 1, 1)
    lobe_support = lobe_search_mask * torch.clamp(search_mask + query_recall_mask, 0, 1) * has_recall * visibility_mask

    enhanced_query_mask = torch.clamp(
        query_mask
        + max(0.0, float(candidate_boost)) * query_recall_mask
        + max(0.0, float(lower_lobe_weight)) * lobe_support,
        0,
        1,
    )
    enhanced_query_mask = enhanced_query_mask * visibility_mask

    protect_mask = object_recall_mask
    if downward_shift > 1:
        protect_mask = torch.clamp(protect_mask + shift_mask(protect_mask, down=max(1, downward_shift // 2)), 0, 1)
    protected_block_mask = source_hair_block_mask * (1 - max(0.0, min(1.0, float(block_protect))) * protect_mask).clamp(0, 1)

    return {
        "query_mask_before_recall": query_mask,
        "query_mask": enhanced_query_mask,
        "earring_visibility_mask": visibility_mask,
        "source_hair_block_mask": protected_block_mask,
        "source_earring_mask": torch.clamp(source_earring_mask + object_recall_mask, 0, 1) * visibility_mask,
        "earring_query_recall_mask": query_recall_mask,
        "earring_object_recall_mask": object_recall_mask,
        "earring_object_detection_mask": object_detection_mask,
        "online_earring_candidate_mask": candidate_mask * visibility_mask,
        "online_earring_search_mask": search_mask * visibility_mask,
        "source_lobe_search_mask": lobe_search_mask * visibility_mask,
        "earring_recall_lobe_support": lobe_support,
        "earring_recall_block_protect_mask": protect_mask,
    }


def build_earring_highlight_mask(
    reference_01: torch.Tensor,
    earring_mask: torch.Tensor,
    query_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    earring_mask = ensure_mask_4d(earring_mask).float()
    query_mask = earring_mask if query_mask is None else resize_mask(query_mask, earring_mask.shape[-2:])
    highlight_support = (earring_mask * query_mask).clamp(0, 1)
    gray = rgb_to_gray(reference_01)
    local_gray = low_pass_filter(gray, kernel_size=9, sigma=2.0)
    bright_spike = F.relu(gray - local_gray) * highlight_support
    high_energy = high_pass_filter(reference_01).abs().mean(dim=1, keepdim=True) * highlight_support
    color_spread = (
        reference_01.amax(dim=1, keepdim=True) - reference_01.amin(dim=1, keepdim=True)
    ) * highlight_support

    bright_candidate = _masked_threshold_candidate(bright_spike, highlight_support, max_ratio=0.48, std_ratio=1.15)
    high_candidate = _masked_threshold_candidate(high_energy, highlight_support, max_ratio=0.45, std_ratio=0.90)
    color_candidate = _masked_threshold_candidate(color_spread, highlight_support, max_ratio=0.50, std_ratio=1.00)
    highlight = bright_candidate * torch.clamp(high_candidate + color_candidate + earring_mask, 0, 1)
    return (dilate_mask(highlight, 3) * highlight_support).clamp(0, 1)


def _batch_gate(
    value: bool | float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return a ``(B, 1, 1, 1)`` gate for a bool/scalar/batch mask.

    Earring presence is decided independently for the two sides.  A few old
    call sites pass a scalar while newer debug/training code passes a spatial
    mask, so keeping this conversion in one place prevents accidental
    broadcasting across samples or sides.
    """

    reference = ensure_mask_4d(reference).float()
    batch = reference.size(0)
    if not torch.is_tensor(value):
        return torch.full(
            (batch, 1, 1, 1),
            float(value),
            device=reference.device,
            dtype=reference.dtype,
        )

    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value.reshape(1, 1, 1, 1).expand(batch, 1, 1, 1)
    if value.ndim == 1:
        if value.numel() == 1:
            return value.reshape(1, 1, 1, 1).expand(batch, 1, 1, 1)
        if value.numel() == batch:
            return value.reshape(batch, 1, 1, 1)
    if value.ndim == 2 and value.size(0) == batch:
        return value.flatten(1).amax(dim=1).view(batch, 1, 1, 1)
    value = ensure_mask_4d(value)
    if value.size(0) != batch:
        if value.size(0) == 1:
            value = value.expand(batch, -1, -1, -1)
        else:
            raise ValueError("batch gate and reference batch sizes do not match")
    return value.flatten(1).amax(dim=1).view(batch, 1, 1, 1)


def _resize_like_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    """Resize an optional mask to ``reference`` while preserving its device."""

    reference = ensure_mask_4d(reference).float()
    if mask is None:
        return torch.zeros_like(reference)
    return resize_mask(mask.to(device=reference.device), reference.shape[-2:])


def build_earlobe_anchor(
    ear_mask: torch.Tensor | None,
    *,
    fallback_skin_mask: torch.Tensor | None = None,
    ear_roi: torch.Tensor | None = None,
    lower_ratio: float = 0.62,
    dilate: int = 3,
) -> torch.Tensor:
    """Return the lower 30-40 percent of an ear, never the complete ear.

    The parser can omit an exposed lobe.  In that case a lower-half skin patch
    inside the side-specific ear ROI is a conservative semantic fallback.  The
    function deliberately has no image-centre assumption: all geometry comes
    from the supplied side mask/ROI.
    """

    reference = ear_mask if ear_mask is not None else fallback_skin_mask
    if reference is None:
        raise ValueError("ear_mask or fallback_skin_mask is required for an earlobe anchor")
    reference = ensure_mask_4d(reference).float()
    ear = _resize_like_mask(ear_mask, reference)
    fallback = _resize_like_mask(fallback_skin_mask, reference)
    roi = torch.ones_like(reference) if ear_roi is None else _resize_like_mask(ear_roi, reference)

    def lower_band(mask: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = mask.shape
        rows = (mask > 0.5).amax(dim=-1)
        row_ids = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height)
        first = torch.where(rows, row_ids, torch.full_like(row_ids, float(height))).amin(dim=-1, keepdim=True)
        last = torch.where(rows, row_ids, torch.full_like(row_ids, -1.0)).amax(dim=-1, keepdim=True)
        start = first + (last - first).clamp_min(0.0) * float(lower_ratio)
        y = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
        valid = (last >= first).view(batch, 1, 1, 1)
        return (mask * (y >= start.view(batch, 1, 1, 1)).to(mask.dtype) * valid).clamp(0, 1)

    primary = lower_band(ear) * roi
    fallback_lower = lower_band(fallback * roi)
    primary_present = (primary.flatten(1).sum(dim=1) >= 2.0).view(-1, 1, 1, 1)
    anchor = torch.where(primary_present, primary, fallback_lower)
    if int(dilate) > 1:
        anchor = dilate_mask(anchor, int(dilate)) * roi
    return anchor.clamp(0, 1)


def assign_components_to_ear_sides(
    candidate_mask: torch.Tensor,
    left_roi: torch.Tensor,
    right_roi: torch.Tensor,
    left_lobe_anchor: torch.Tensor | None = None,
    right_lobe_anchor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign connected object candidates to side-specific ear geometry.

    Facial parser labels are in image coordinates, but neither label is
    guaranteed to lie on its conventional half after crop, profile pose or a
    mirrored image.  Components are therefore assigned from ROI overlap and
    distance to the actual lobe anchors.  This is intentionally a small CPU
    connected-component pass: mask construction is non-differentiable and the
    256px ear ROI contains very few foreground pixels.
    """

    candidate = ensure_mask_4d(candidate_mask).float()
    left_roi = _resize_like_mask(left_roi, candidate)
    right_roi = _resize_like_mask(right_roi, candidate)
    left_anchor = _resize_like_mask(left_lobe_anchor, candidate)
    right_anchor = _resize_like_mask(right_lobe_anchor, candidate)

    candidate_np = candidate.detach().cpu().numpy()
    binary = candidate_np > 0.5
    left_np = (left_roi.detach().cpu().numpy() > 0.05)
    right_np = (right_roi.detach().cpu().numpy() > 0.05)
    left_anchor_np = (left_anchor.detach().cpu().numpy() > 0.05)
    right_anchor_np = (right_anchor.detach().cpu().numpy() > 0.05)
    left_out = np.zeros_like(binary, dtype=np.float32)
    right_out = np.zeros_like(binary, dtype=np.float32)

    for batch_index in range(binary.shape[0]):
        foreground = binary[batch_index, 0]
        visited = np.zeros_like(foreground, dtype=bool)
        left_anchor_points = np.argwhere(left_anchor_np[batch_index, 0])
        right_anchor_points = np.argwhere(right_anchor_np[batch_index, 0])
        height, width = foreground.shape
        for start_y, start_x in np.argwhere(foreground):
            if visited[start_y, start_x]:
                continue
            stack = [(int(start_y), int(start_x))]
            visited[start_y, start_x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for yy in range(max(0, y - 1), min(height, y + 2)):
                    for xx in range(max(0, x - 1), min(width, x + 2)):
                        if foreground[yy, xx] and not visited[yy, xx]:
                            visited[yy, xx] = True
                            stack.append((yy, xx))

            coords = np.asarray(pixels, dtype=np.float32)
            ys = coords[:, 0].astype(np.int64)
            xs = coords[:, 1].astype(np.int64)
            left_overlap = float(left_np[batch_index, 0, ys, xs].mean())
            right_overlap = float(right_np[batch_index, 0, ys, xs].mean())
            centre = coords.mean(axis=0)

            def anchor_score(points: np.ndarray) -> float:
                if points.size == 0:
                    return 0.0
                # The nearest real lobe pixel is robust to elongated earrings.
                distance = np.sqrt(((points.astype(np.float32) - centre) ** 2).sum(axis=1)).min()
                return float(1.0 / (1.0 + distance / 24.0))

            left_score = left_overlap + anchor_score(left_anchor_points)
            right_score = right_overlap + anchor_score(right_anchor_points)
            if left_score <= 0.0 and right_score <= 0.0:
                continue
            destination = left_out if left_score >= right_score else right_out
            destination[batch_index, 0, ys, xs] = candidate_np[batch_index, 0, ys, xs]

    left = torch.from_numpy(left_out).to(device=candidate.device, dtype=candidate.dtype)
    right = torch.from_numpy(right_out).to(device=candidate.device, dtype=candidate.dtype)
    return left.clamp(0, 1), right.clamp(0, 1)


def compute_earring_hole_mask(component_mask: torch.Tensor) -> torch.Tensor:
    """Find enclosed holes in an earring component with border flood fill.

    A hole is background unreachable from an image boundary.  Keeping it zero
    makes a hoop's centre retain the target image even after later feature-mask
    dilation or refinement.
    """

    component = (ensure_mask_4d(component_mask).float() > 0.5).float()
    background = 1.0 - component
    border = torch.zeros_like(background)
    border[..., 0, :] = 1
    border[..., -1, :] = 1
    border[..., :, 0] = 1
    border[..., :, -1] = 1
    reachable = background * border
    # 8-connected propagation preserves diagonal thin hoop wires.
    for _ in range(max(component.shape[-2:])):
        next_reachable = dilate_mask(reachable, 3) * background
        if torch.equal(next_reachable > 0.5, reachable > 0.5):
            break
        reachable = next_reachable
    filled = (1.0 - reachable).clamp(0, 1)
    return (filled - component).clamp(0, 1)


def build_elliptical_hoop_candidates(
    source_01: torch.Tensor,
    search_mask: torch.Tensor,
    left_lobe_anchor: torch.Tensor,
    right_lobe_anchor: torch.Tensor,
    *,
    source_hair_mask: torch.Tensor | None = None,
    source_ear_mask: torch.Tensor | None = None,
    min_axis: float = 12.0,
    max_axis: float = 168.0,
    min_coverage: float = 0.22,
) -> dict[str, torch.Tensor]:
    """Find a complete source hoop before allowing visual-only recovery.

    A parser-missed hoop cannot be reconstructed from a handful of independent
    Sobel pixels: copying those pixels produces the broken metal squiggle seen
    beside the ear.  This helper accepts an image-only fallback only when its
    Canny contour supports a fitted ellipse around an actual lobe.  It returns
    the edge-supported trace and fitted interior separately: the trace grants
    source-RGB permission while the interior remains target-owned.
    """

    reference = ensure_mask_4d(search_mask).float()
    zeros = torch.zeros_like(reference)
    if cv2 is None:
        return {
            "left_elliptical_hoop": zeros,
            "right_elliptical_hoop": zeros,
            "left_elliptical_hoop_hole": zeros,
            "right_elliptical_hoop_hole": zeros,
        }

    source_01 = normalized_to_01(source_01)
    size = reference.shape[-2:]
    source_01 = F.interpolate(source_01, size=size, mode="bilinear", align_corners=False)
    search = _resize_like_mask(search_mask, reference)
    left_anchor = _resize_like_mask(left_lobe_anchor, reference)
    right_anchor = _resize_like_mask(right_lobe_anchor, reference)
    hair = _resize_like_mask(source_hair_mask, reference)
    ear = _resize_like_mask(source_ear_mask, reference)

    source_np = source_01.detach().cpu().permute(0, 2, 3, 1).numpy()
    search_np = (search.detach().cpu().numpy() > 0.5)
    hair_np = (hair.detach().cpu().numpy() > 0.5)
    ear_np = (ear.detach().cpu().numpy() > 0.5)
    left_anchor_np = (left_anchor.detach().cpu().numpy() > 0.05)
    right_anchor_np = (right_anchor.detach().cpu().numpy() > 0.05)
    left_out = np.zeros_like(search_np, dtype=np.float32)
    right_out = np.zeros_like(search_np, dtype=np.float32)
    left_hole_out = np.zeros_like(search_np, dtype=np.float32)
    right_hole_out = np.zeros_like(search_np, dtype=np.float32)
    height, width = size
    coordinate_scale = max(float(height), float(width)) / 256.0

    def scaled_kernel(base: float) -> int:
        kernel = max(3, int(round(float(base) * coordinate_scale)))
        return kernel if kernel % 2 == 1 else kernel + 1

    def trace_for_anchor(
        image: np.ndarray,
        base_search: np.ndarray,
        hair_mask: np.ndarray,
        ear_mask: np.ndarray,
        anchor_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.argwhere(anchor_mask)
        if points.size == 0:
            zero = np.zeros_like(base_search, dtype=np.float32)
            return zero, zero
        anchor_y, anchor_x = points.mean(axis=0)
        y_grid, x_grid = np.ogrid[:height, :width]
        # A hoop hangs from a lobe.  This is a *detection* corridor: it needs
        # to reach the lower half of a large hoop even when the parser misses
        # every earring pixel.  Acceptance below still requires lobe contact
        # and multi-arc image support, so this wider window cannot create a
        # generic lower-ear write region.
        local_window = (
            (y_grid >= anchor_y - 24.0 * coordinate_scale)
            & (y_grid <= anchor_y + 128.0 * coordinate_scale)
            & (x_grid >= anchor_x - 88.0 * coordinate_scale)
            & (x_grid <= anchor_x + 88.0 * coordinate_scale)
        )
        valid = base_search & local_window & ~hair_mask
        if int(valid.sum()) < 24:
            zero = np.zeros_like(base_search, dtype=np.float32)
            return zero, zero

        gray = np.clip(np.round(image.mean(axis=2) * 255.0), 0, 255).astype(np.uint8)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(grad_x, grad_y)
        masked_gradient = gradient[valid]
        low = max(8, int(np.percentile(masked_gradient, 55) * 0.50))
        high = max(low + 12, int(np.percentile(masked_gradient, 82)))
        edges = cv2.Canny(gray, low, min(255, high))
        edges[~valid] = 0
        # Join antialiased wire fragments for contour fitting only.  The final
        # mask below is a thin ellipse and therefore never copies this closure.
        fit_edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _ = cv2.findContours(fit_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        fitted_ellipses: list[tuple[float, float, float, float, float]] = []
        for contour in contours:
            if len(contour) < 12:
                continue
            (center_x, center_y), (axis_x, axis_y), angle = cv2.fitEllipse(contour)
            fitted_ellipses.append(
                (float(center_x), float(center_y), float(axis_x), float(axis_y), float(angle))
            )

        # Canny often breaks a reflective hoop into separate short arcs.  A
        # circle proposal joins those arcs geometrically, but it is accepted
        # only through the same edge-sector and lobe checks as a contour fit.
        # This recovers a real large hoop without falling back to arbitrary
        # dark/bright fragments beside an ear.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(12.0, float(min_axis) * 0.75),
            param1=max(24.0, float(high)),
            param2=10.0,
            minRadius=max(6, int(round(float(min_axis) / 2.0))),
            maxRadius=max(7, int(round(float(max_axis) / 2.0))),
        )
        if circles is not None:
            for center_x, center_y, radius in np.round(circles[0]).astype(np.float32):
                center_y_int = int(round(center_y))
                center_x_int = int(round(center_x))
                if not (0 <= center_y_int < height and 0 <= center_x_int < width):
                    continue
                if not local_window[center_y_int, center_x_int]:
                    continue
                diameter = float(radius) * 2.0
                fitted_ellipses.append((float(center_x), float(center_y), diameter, diameter, 0.0))

        best_mask = None
        best_hole = None
        best_score = -1.0
        support = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8)) > 0
        trace_support = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8)) > 0
        for center_x, center_y, axis_x, axis_y, angle in fitted_ellipses:
            major = max(axis_x, axis_y)
            minor = min(axis_x, axis_y)
            if not (float(min_axis) <= minor <= major <= float(max_axis)):
                continue
            if minor / max(major, 1e-6) < 0.32:
                continue

            ellipse = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                ellipse,
                (int(round(center_x)), int(round(center_y))),
                (max(1, int(round(axis_x / 2.0))), max(1, int(round(axis_y / 2.0)))),
                float(angle),
                0,
                360,
                255,
                2,
                lineType=cv2.LINE_8,
            )
            ellipse_bool = ellipse > 0
            coverage = float((ellipse_bool & support).sum()) / max(float(ellipse_bool.sum()), 1.0)
            if coverage < float(min_coverage):
                continue
            # Coverage alone can be satisfied by one curved hair/background
            # edge.  A real hoop has support over multiple directions around
            # its fitted perimeter.  Four of eight sectors retains partially
            # reflective metal while rejecting the old fragmented squiggle.
            support_y, support_x = np.where(ellipse_bool & support)
            if support_y.size == 0:
                continue
            sector_angle = np.arctan2(support_y - center_y, support_x - center_x)
            sectors = np.unique(np.floor((sector_angle + np.pi) * (8.0 / (2.0 * np.pi))).astype(np.int32) % 8)
            if sectors.size < 4:
                continue
            # The lobe must touch the fitted perimeter.  A background ellipse
            # farther from the ear is not a valid accessory candidate.
            lobe_touch = bool(
                (cv2.dilate(
                    ellipse,
                    np.ones((scaled_kernel(25), scaled_kernel(25)), dtype=np.uint8),
                ) > 0)[anchor_mask].any()
            )
            if not lobe_touch:
                continue
            # A wire can attach on the ear boundary.  Exclude only the semantic
            # ear interior, rather than a broad dilation that erases the
            # source connector from lobe to hoop.
            ear_interior = cv2.erode(
                ear_mask.astype(np.uint8),
                np.ones((scaled_kernel(7), scaled_kernel(7)), dtype=np.uint8),
            ).astype(bool)
            trace = ellipse_bool & trace_support & valid & ~ear_interior
            # The suspension wire between lobe and hoop is not part of an
            # ellipse perimeter, so fitting the ring alone necessarily drops
            # it.  Recover only observed edges in the narrow straight corridor
            # from the lobe to the nearest accepted ring pixel.  This cannot
            # paint a broad ear/background patch because the corridor is used
            # solely as an edge selector and only after hoop validation.
            trace_points = np.argwhere(trace)
            if trace_points.size > 0:
                nearest_index = np.argmin(
                    ((trace_points.astype(np.float32) - np.asarray([anchor_y, anchor_x])) ** 2).sum(axis=1)
                )
                nearest_y, nearest_x = trace_points[nearest_index]
                connector_distance = float(np.hypot(nearest_y - anchor_y, nearest_x - anchor_x))
                if connector_distance <= 72.0 * coordinate_scale:
                    connector_corridor = np.zeros((height, width), dtype=np.uint8)
                    cv2.line(
                        connector_corridor,
                        (int(round(anchor_x)), int(round(anchor_y))),
                        (int(nearest_x), int(nearest_y)),
                        255,
                        thickness=scaled_kernel(5),
                        lineType=cv2.LINE_8,
                    )
                    connector = (
                        (connector_corridor > 0)
                        & trace_support
                        & valid
                        & ~ear_interior
                    )
                    if int(connector.sum()) >= max(3, int(round(2.0 * coordinate_scale))):
                        trace = trace | connector
            if int(trace.sum()) < max(12, int(minor * 0.35)):
                continue
            filled_ellipse = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                filled_ellipse,
                (int(round(center_x)), int(round(center_y))),
                (max(1, int(round(axis_x / 2.0))), max(1, int(round(axis_y / 2.0)))),
                float(angle),
                0,
                360,
                255,
                -1,
                lineType=cv2.LINE_8,
            )
            # The geometry hole remains target-owned even when an antialiased
            # source wire is not pixel-wise closed at the detection resolution.
            hole = (filled_ellipse > 0) & ~cv2.dilate(
                ellipse,
                np.ones((scaled_kernel(5), scaled_kernel(5)), dtype=np.uint8),
            ).astype(bool)
            score = coverage * float(sectors.size) * float(trace.sum())
            if score > best_score:
                best_score = score
                best_mask = trace
                best_hole = hole
        if best_mask is None:
            zero = np.zeros_like(base_search, dtype=np.float32)
            return zero, zero
        return best_mask.astype(np.float32), best_hole.astype(np.float32)

    for index in range(reference.size(0)):
        left_out[index, 0], left_hole_out[index, 0] = trace_for_anchor(
            source_np[index],
            search_np[index, 0],
            hair_np[index, 0],
            ear_np[index, 0],
            left_anchor_np[index, 0],
        )
        right_out[index, 0], right_hole_out[index, 0] = trace_for_anchor(
            source_np[index],
            search_np[index, 0],
            hair_np[index, 0],
            ear_np[index, 0],
            right_anchor_np[index, 0],
        )

    return {
        "left_elliptical_hoop": torch.from_numpy(left_out).to(reference.device, reference.dtype),
        "right_elliptical_hoop": torch.from_numpy(right_out).to(reference.device, reference.dtype),
        "left_elliptical_hoop_hole": torch.from_numpy(left_hole_out).to(reference.device, reference.dtype),
        "right_elliptical_hoop_hole": torch.from_numpy(right_hole_out).to(reference.device, reference.dtype),
    }


def refine_earring_hoops_highres(
    source_01: torch.Tensor,
    search_mask: torch.Tensor,
    left_lobe_anchor: torch.Tensor,
    right_lobe_anchor: torch.Tensor,
    *,
    source_hair_mask: torch.Tensor | None = None,
    source_ear_mask: torch.Tensor | None = None,
    detection_size: int = 512,
) -> dict[str, torch.Tensor]:
    """Trace a validated hoop at 512px for the final RGB composite.

    Learned PP masks live at 256px.  Directly enlarging one of those masks
    makes a circular wire visibly thick and flattened.  This function retains
    the same ear-local search policy but reruns the edge-supported geometry on
    the high-resolution source reference at a bounded work size.
    """

    source_01 = normalized_to_01(source_01)
    output_size = tuple(source_01.shape[-2:])
    largest_side = max(output_size)
    resize_scale = min(1.0, float(max(64, int(detection_size))) / float(largest_side))
    detect_size = (
        max(64, int(round(output_size[0] * resize_scale))),
        max(64, int(round(output_size[1] * resize_scale))),
    )
    detector_scale = max(float(detect_size[0]), float(detect_size[1])) / 256.0

    def resize_for_detection(value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            return torch.zeros(
                source_01.shape[0],
                1,
                *detect_size,
                device=source_01.device,
                dtype=source_01.dtype,
            )
        value = ensure_mask_4d(value).to(device=source_01.device, dtype=source_01.dtype)
        return F.interpolate(value, size=detect_size, mode="nearest")

    candidates = build_elliptical_hoop_candidates(
        F.interpolate(source_01, size=detect_size, mode="bilinear", align_corners=False),
        resize_for_detection(search_mask),
        resize_for_detection(left_lobe_anchor),
        resize_for_detection(right_lobe_anchor),
        source_hair_mask=resize_for_detection(source_hair_mask),
        source_ear_mask=resize_for_detection(source_ear_mask),
        min_axis=12.0 * detector_scale,
        max_axis=168.0 * detector_scale,
    )
    return {
        key: F.interpolate(value, size=output_size, mode="nearest")
        for key, value in candidates.items()
    }


def build_strong_earring_candidate(
    source_01: torch.Tensor,
    candidate_mask: torch.Tensor,
    ear_roi: torch.Tensor,
    left_roi: torch.Tensor,
    right_roi: torch.Tensor,
    left_lobe_anchor: torch.Tensor,
    right_lobe_anchor: torch.Tensor,
    *,
    source_background_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    source_ear_mask: torch.Tensor | None = None,
    parser_earring_mask: torch.Tensor | None = None,
    min_area: float = 5.0,
    max_roi_density: float = 0.28,
) -> dict[str, torch.Tensor]:
    """Promote only object-like, lobe-connected visual evidence to strong.

    Weak recall remains search-only.  A strong candidate must live around a
    real ear, show local edge/chroma evidence, avoid source hair/background,
    and have a plausible component area/density.  It is the sole parser-miss
    fallback allowed to activate earring presence.
    """

    source_01 = normalized_to_01(source_01)
    reference = ensure_mask_4d(candidate_mask).float()
    size = reference.shape[-2:]
    candidate = resize_mask(candidate_mask, size)
    ear_roi = _resize_like_mask(ear_roi, reference)
    left_roi = _resize_like_mask(left_roi, reference)
    right_roi = _resize_like_mask(right_roi, reference)
    left_anchor = _resize_like_mask(left_lobe_anchor, reference)
    right_anchor = _resize_like_mask(right_lobe_anchor, reference)
    background = _resize_like_mask(source_background_mask, reference)
    hair = _resize_like_mask(source_hair_mask, reference)
    source_ear = _resize_like_mask(source_ear_mask, reference)
    parser_earring = _resize_like_mask(parser_earring_mask, reference)
    source_01 = F.interpolate(source_01, size=size, mode="bilinear", align_corners=False)

    edge = sobel_magnitude(source_01)
    chroma = source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)
    local_contrast = (rgb_to_gray(source_01) - low_pass_filter(rgb_to_gray(source_01), 9, 2.0)).abs()
    edge_support = _masked_threshold_candidate(edge, ear_roi, max_ratio=0.35, std_ratio=0.55, floor=0.008)
    chroma_support = _masked_threshold_candidate(chroma + local_contrast, ear_roi, max_ratio=0.35, std_ratio=0.60, floor=0.018)
    contrast_support = _masked_threshold_candidate(
        local_contrast,
        ear_roi,
        max_ratio=0.28,
        std_ratio=0.40,
        floor=0.006,
    )

    # Large metal hoops are often parser-missed and their thin wire may be
    # absent from the weak candidate after thresholding.  Promote only the
    # high-contrast wire outside the semantic ear *interior*; this admits the
    # lobe-to-hoop connector while rejecting the ear's own interior texture as
    # a fake accessory.
    # It remains subject to side, lobe and density checks below.
    ear_exterior = (1.0 - erode_mask(source_ear, 7)).clamp(0, 1)
    visual_wire = edge_support * contrast_support * ear_exterior * ear_roi
    candidate = torch.maximum(candidate, visual_wire)
    elliptical_hoops = build_elliptical_hoop_candidates(
        source_01,
        ear_roi,
        left_anchor,
        right_anchor,
        source_hair_mask=hair,
        source_ear_mask=source_ear,
    )
    elliptical_candidate = torch.clamp(
        elliptical_hoops["left_elliptical_hoop"] + elliptical_hoops["right_elliptical_hoop"],
        0,
        1,
    )
    elliptical_hole = torch.clamp(
        elliptical_hoops["left_elliptical_hoop_hole"]
        + elliptical_hoops["right_elliptical_hoop_hole"],
        0,
        1,
    )
    object_evidence = candidate * torch.clamp(edge_support + chroma_support + contrast_support, 0, 1)
    # A parser-missed metal wire is often labelled background.  Do not erase it
    # pixel-wise here; reject components by *coverage ratio* below instead.
    # Hair still receives a strong attenuation because hair edges are the most
    # common visual false positive around an ear.
    object_evidence = object_evidence * (1.0 - 0.75 * hair).clamp(0, 1) * ear_roi
    object_evidence = dilate_mask(object_evidence, 3) * candidate * ear_roi
    # A semantic label may grow a little into nearby visual evidence.  In the
    # parser-missed case, however, partial generic edges are not an earring:
    # only a closed, lobe-anchored elliptical hoop is accepted.  This removes
    # the broken-metal artefact while retaining label-9 studs and pendants.
    parser_neighbourhood = dilate_mask(parser_earring, 13)
    object_evidence = object_evidence * parser_neighbourhood
    # Once a lobe-connected hoop has passed its multi-arc geometry check, use
    # neighbouring *observed* source wire to follow the real object rather
    # than rendering the ideal fitted ellipse.  This keeps an imperfectly
    # circular source hoop's own thickness and shape, while the edge/contrast
    # gate prevents the surrounding source hair or background from joining it.
    observed_hoop_wire = visual_wire * dilate_mask(elliptical_candidate, 9) * ear_roi
    object_evidence = torch.maximum(
        object_evidence,
        torch.maximum(elliptical_candidate, observed_hoop_wire),
    )

    # A hoop wire frequently has one-pixel gaps after 256px downsampling.  Use
    # a small proxy solely to assign all arcs to the same ear side, then return
    # the original thin evidence so no background is added to the write mask.
    component_proxy = dilate_mask(object_evidence, 3) * ear_roi
    left_proxy, right_proxy = assign_components_to_ear_sides(
        component_proxy,
        left_roi,
        right_roi,
        left_anchor,
        right_anchor,
    )
    left = object_evidence * (left_proxy > 0.5).to(object_evidence.dtype)
    right = object_evidence * (right_proxy > 0.5).to(object_evidence.dtype)
    area_scale = float(size[0] * size[1]) / float(256 * 256)

    def validate(side: torch.Tensor, roi: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        area = side.flatten(1).sum(dim=1, keepdim=True)
        density = area / roi.flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
        background_ratio = (side * background).flatten(1).sum(dim=1, keepdim=True) / area.clamp_min(1.0)
        hair_ratio = (side * hair).flatten(1).sum(dim=1, keepdim=True) / area.clamp_min(1.0)
        near_lobe = (side * dilate_mask(anchor, 25)).flatten(1).sum(dim=1, keepdim=True) >= 1.0
        plausible = (
            (area >= float(min_area) * area_scale)
            & (density <= float(max_roi_density))
            # A fully parser-missed wire is commonly labelled background at
            # every wire pixel.  Do not reject that exact case here: the small
            # area/density, structured visual evidence, hair rejection and
            # lobe-attachment checks above are what distinguish it from a broad
            # lower-ear background patch.
            & (background_ratio <= 1.0)
            & (hair_ratio <= 0.40)
            & near_lobe
        ).to(side.dtype).view(-1, 1, 1, 1)
        return side * plausible

    left = validate(left, left_roi, left_anchor)
    right = validate(right, right_roi, right_anchor)
    return {
        "strong_candidate_mask": torch.clamp(left + right, 0, 1),
        "left_strong_candidate": left,
        "right_strong_candidate": right,
        "left_elliptical_hoop": elliptical_hoops["left_elliptical_hoop"],
        "right_elliptical_hoop": elliptical_hoops["right_elliptical_hoop"],
        "elliptical_hoop_hole": elliptical_hole,
        "left_elliptical_hoop_hole": elliptical_hoops["left_elliptical_hoop_hole"],
        "right_elliptical_hoop_hole": elliptical_hoops["right_elliptical_hoop_hole"],
    }


def extract_earring_object_core(
    trusted_object_mask: torch.Tensor,
    visible_ear_roi: torch.Tensor,
    earlobe_anchor_mask: torch.Tensor,
    *,
    connectivity_iters: int = 32,
    connectivity_kernel: int = 5,
    bridge_dilate: int = 17,
) -> torch.Tensor:
    """Keep trusted/strong object pixels connected to an exposed lobe."""

    trusted = ensure_mask_4d(trusted_object_mask).float()
    visible = _resize_like_mask(visible_ear_roi, trusted)
    anchor = _resize_like_mask(earlobe_anchor_mask, trusted)
    return extract_visible_earring_segment(
        trusted,
        trusted * visible,
        anchor,
        connectivity_iters=connectivity_iters,
        connectivity_kernel=connectivity_kernel,
        bridge_dilate=bridge_dilate,
    )


def extract_visible_earring_segment(
    source_earring_mask: torch.Tensor,
    write_mask: torch.Tensor,
    connectivity_anchor: torch.Tensor,
    *,
    connectivity_iters: int = 32,
    connectivity_kernel: int = 5,
    bridge_dilate: int = 17,
    threshold: float = 1e-4,
) -> torch.Tensor:
    """Keep only the earring pixels connected to a visible ear-lobe anchor.

    ``source_earring_mask`` is the source/object evidence and ``write_mask`` is
    the (already narrow) target write gate.  The old implementation grew a
    solid channel below the lobe; a hoop's hollow centre and the background in
    that channel were consequently copied back as if they were an earring.
    Here growth is performed *on the candidate pixels only*.  A small bridge
    dilation handles parser gaps between the lobe and a thin wire, while
    disconnected background islands are never reached.
    """

    source_earring_mask = ensure_mask_4d(source_earring_mask).float()
    write_mask = _resize_like_mask(write_mask, source_earring_mask)
    connectivity_anchor = _resize_like_mask(connectivity_anchor, source_earring_mask)

    candidate = (source_earring_mask * write_mask).clamp(0, 1)
    candidate_binary = (candidate > float(threshold)).float()
    if candidate_binary.numel() == 0:
        return candidate

    # Bridge only a small parser gap.  The guide remains the candidate itself,
    # so dilation cannot flood the hoop interior or source background.
    guide = dilate_mask(candidate_binary, max(1, int(bridge_dilate)))
    seed = dilate_mask(connectivity_anchor, max(1, int(bridge_dilate))) * guide
    reachable = seed.clamp(0, 1)
    for _ in range(max(0, int(connectivity_iters))):
        nxt = (dilate_mask(reachable, max(1, int(connectivity_kernel))) * guide).clamp(0, 1)
        reachable = torch.maximum(reachable, nxt)
    # Keep the actual candidate values (rather than a binary mask) for smooth
    # blending, but require a path to the visible lobe.
    return (candidate * reachable).clamp(0, 1)


def build_earring_search_mask(
    parser_earring_mask: torch.Tensor | None,
    visible_ear_roi: torch.Tensor,
    online_candidate_mask: torch.Tensor | None = None,
    weak_recall_mask: torch.Tensor | None = None,
    lobe_search_mask: torch.Tensor | None = None,
    *,
    downward_shift: int = 10,
    search_dilate: int = 7,
    no_earring: bool | float | torch.Tensor = False,
) -> torch.Tensor:
    """Build the broad *search* region used to find an earring.

    Search is intentionally permissive (parser evidence, online/weak
    candidates, the lobe neighbourhood and a downward offset).  It is not a
    write permission.  The final gate is :func:`build_earring_write_mask`.
    This separation lets long earrings be detected without copying the
    background around them.
    """

    visible_ear_roi = ensure_mask_4d(visible_ear_roi).float()
    parser_earring_mask = _resize_like_mask(parser_earring_mask, visible_ear_roi)
    online_candidate_mask = _resize_like_mask(online_candidate_mask, visible_ear_roi)
    weak_recall_mask = _resize_like_mask(weak_recall_mask, visible_ear_roi)
    lobe_search_mask = _resize_like_mask(lobe_search_mask, visible_ear_roi)

    object_seed = torch.clamp(
        parser_earring_mask + online_candidate_mask + weak_recall_mask,
        0,
        1,
    )
    search = dilate_mask(object_seed, max(1, int(search_dilate)))
    if int(downward_shift) != 0:
        search = torch.clamp(search + shift_mask(search, down=int(downward_shift)), 0, 1)
    # Lobe support is useful for tiny studs and for parser gaps, but remains a
    # search hint only.  It cannot become a write mask by itself.
    search = torch.clamp(search + lobe_search_mask, 0, 1) * visible_ear_roi
    active = (1.0 - _batch_gate(no_earring, visible_ear_roi)).clamp(0, 1)
    return (search * active).clamp(0, 1)


def _build_earring_write_mask_legacy(
    confident_earring_mask: torch.Tensor,
    visible_ear_roi: torch.Tensor,
    target_hair_occlusion_mask: torch.Tensor,
    earlobe_anchor_mask: torch.Tensor,
    no_earring: bool | float | torch.Tensor = False,
    *,
    source_background_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    source_hair_block_mask: torch.Tensor | None = None,
    source_semantic_block_mask: torch.Tensor | None = None,
    parser_earring_mask: torch.Tensor | None = None,
    trusted_earring_mask: torch.Tensor | None = None,
    max_target_hair_overlap: float = 0.30,
    source_block_dilate: int = 3,
    write_dilate: int = 3,
    connectivity_iters: int = 32,
    connectivity_kernel: int = 5,
    bridge_dilate: int = 17,
) -> torch.Tensor:
    """Build the narrow, final earring write-back mask.

    The mask is the intersection of confident object evidence, a visible ear
    ROI and a connected lobe-to-earring path.  Source background/hair are hard
    blockers (parser-confirmed or otherwise trusted object pixels are retained),
    and target hair can only be crossed by a small, area-capped fraction of the
    object.  A
    side with no reliable earring is a hard zero, never a weakly attenuated
    ear-lobe box.
    """

    confident_earring_mask = ensure_mask_4d(confident_earring_mask).float()
    visible_ear_roi = _resize_like_mask(visible_ear_roi, confident_earring_mask)
    target_hair_occlusion_mask = _resize_like_mask(target_hair_occlusion_mask, confident_earring_mask)
    earlobe_anchor_mask = _resize_like_mask(earlobe_anchor_mask, confident_earring_mask)
    source_background_mask = _resize_like_mask(source_background_mask, confident_earring_mask)
    source_hair_mask = _resize_like_mask(source_hair_mask, confident_earring_mask)
    source_hair_block_mask = _resize_like_mask(source_hair_block_mask, confident_earring_mask)
    source_semantic_block_mask = _resize_like_mask(
        source_semantic_block_mask,
        confident_earring_mask,
    )
    parser_earring_mask = (
        confident_earring_mask
        if parser_earring_mask is None
        else _resize_like_mask(parser_earring_mask, confident_earring_mask)
    )
    trusted_earring_mask = _resize_like_mask(trusted_earring_mask, confident_earring_mask)

    # One-pixel-ish margin catches a parser's antialiased earring edge without
    # recreating the former solid channel.  ``write_dilate=1`` is also valid.
    if int(write_dilate) > 1:
        object_evidence = dilate_mask(confident_earring_mask, int(write_dilate))
    else:
        object_evidence = confident_earring_mask.clamp(0, 1)
    candidate = object_evidence * visible_ear_roi

    # Source hair/background are never valid source pixels.  A parser-confirmed
    # label-9 earring wins over hair under-segmentation, but only at those exact
    # pixels; surrounding source content remains blocked.
    source_block = torch.clamp(
        source_background_mask
        + source_hair_mask
        + source_hair_block_mask
        + source_semantic_block_mask,
        0,
        1,
    )
    if int(source_block_dilate) > 1:
        source_block = dilate_mask(source_block, int(source_block_dilate))
    trusted_keep = (
        (parser_earring_mask > 0.5).float()
        + (trusted_earring_mask > 0.5).float()
    ).clamp(0, 1)
    source_gate = (1.0 - source_block * (1.0 - trusted_keep)).clamp(0, 1)
    candidate = candidate * source_gate

    target_hair_occlusion_mask = target_hair_occlusion_mask.clamp(0, 1)
    non_hair = candidate * (1.0 - target_hair_occlusion_mask)
    hair_part = candidate * target_hair_occlusion_mask
    # Hair-overlap pixels must touch a visible object/lobe; this prevents a
    # broad target hair component from admitting an entire hidden earring.
    overlap_support = dilate_mask(
        torch.clamp(non_hair + earlobe_anchor_mask, 0, 1),
        max(1, int(bridge_dilate)),
    )
    hair_part = hair_part * overlap_support

    # Cap the target-hair fraction by area.  Unlike a scalar attenuation this
    # preserves all visible pixels and only trims the hidden tail.
    max_overlap = max(0.0, min(0.95, float(max_target_hair_overlap)))
    non_hair_area = non_hair.flatten(1).sum(dim=1, keepdim=True)
    hair_area = hair_part.flatten(1).sum(dim=1, keepdim=True)
    allowed_hair = (max_overlap / max(1.0 - max_overlap, 1e-6)) * non_hair_area
    hair_scale = torch.where(
        hair_area > 1e-6,
        torch.minimum(torch.ones_like(hair_area), allowed_hair / hair_area.clamp_min(1e-6)),
        torch.zeros_like(hair_area),
    ).view(-1, 1, 1, 1)
    candidate = (non_hair + hair_part * hair_scale).clamp(0, 1)

    connected = extract_visible_earring_segment(
        confident_earring_mask,
        candidate,
        earlobe_anchor_mask,
        connectivity_iters=connectivity_iters,
        connectivity_kernel=connectivity_kernel,
        bridge_dilate=bridge_dilate,
    )

    # Connectivity can remove some visible pixels; re-cap once more so the
    # final mask always satisfies the advertised target-hair overlap bound.
    connected_hair = connected * target_hair_occlusion_mask
    connected_non_hair = connected * (1.0 - target_hair_occlusion_mask)
    nh_area = connected_non_hair.flatten(1).sum(dim=1, keepdim=True)
    h_area = connected_hair.flatten(1).sum(dim=1, keepdim=True)
    allowed = (max_overlap / max(1.0 - max_overlap, 1e-6)) * nh_area
    scale = torch.where(
        h_area > 1e-6,
        torch.minimum(torch.ones_like(h_area), allowed / h_area.clamp_min(1e-6)),
        torch.zeros_like(h_area),
    ).view(-1, 1, 1, 1)
    connected = (connected_non_hair + connected_hair * scale).clamp(0, 1)
    active = (1.0 - _batch_gate(no_earring, connected)).clamp(0, 1)
    return (connected * active).clamp(0, 1)


def build_earring_write_masks(
    trusted_object_mask: torch.Tensor,
    completion_candidate_mask: torch.Tensor | None,
    visible_ear_roi: torch.Tensor,
    target_hair_occlusion_mask: torch.Tensor,
    earlobe_anchor_mask: torch.Tensor,
    no_earring: bool | float | torch.Tensor = False,
    *,
    source_background_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    source_hair_block_mask: torch.Tensor | None = None,
    source_semantic_block_mask: torch.Tensor | None = None,
    max_target_hair_overlap: float = 0.30,
    source_block_dilate: int = 3,
    write_dilate: int = 3,
    connectivity_iters: int = 32,
    connectivity_kernel: int = 5,
    bridge_dilate: int = 17,
) -> dict[str, torch.Tensor]:
    """Build independent core/completion/write masks for earring recovery.

    Tier A is a verified parser, explicit object, or strong visual core.  It
    may cross the transferred hair when connected to a visible lobe.  Tier B is
    only a completion shell, always source-blocked and capped inside target
    hair.  This separation prevents a broad recall ROI from becoming a source
    patch while keeping genuine long earrings recoverable.
    """

    trusted = ensure_mask_4d(trusted_object_mask).float()
    completion = _resize_like_mask(completion_candidate_mask, trusted)
    visible = _resize_like_mask(visible_ear_roi, trusted)
    target_hair = _resize_like_mask(target_hair_occlusion_mask, trusted)
    anchor = _resize_like_mask(earlobe_anchor_mask, trusted)
    source_background = _resize_like_mask(source_background_mask, trusted)
    source_hair = _resize_like_mask(source_hair_mask, trusted)
    source_hair_block = _resize_like_mask(source_hair_block_mask, trusted)
    semantic_block = _resize_like_mask(source_semantic_block_mask, trusted)
    active = (1.0 - _batch_gate(no_earring, trusted)).clamp(0, 1)

    core = extract_earring_object_core(
        trusted * active,
        visible * active,
        anchor * active,
        connectivity_iters=connectivity_iters,
        connectivity_kernel=connectivity_kernel,
        bridge_dilate=bridge_dilate,
    )

    source_block = torch.clamp(
        source_background + source_hair + source_hair_block + semantic_block,
        0,
        1,
    )
    if int(source_block_dilate) > 1:
        source_block = dilate_mask(source_block, int(source_block_dilate))
    # Completion cannot bypass a source blocker.  In contrast, the separately
    # constructed core is permitted to survive parser hair/background mistakes.
    completion = completion * visible * (1.0 - source_block).clamp(0, 1) * active
    completion = extract_visible_earring_segment(
        completion,
        completion,
        anchor * active,
        connectivity_iters=connectivity_iters,
        connectivity_kernel=connectivity_kernel,
        bridge_dilate=bridge_dilate,
    )

    completion_non_hair = completion * (1.0 - target_hair).clamp(0, 1)
    completion_hair = completion * target_hair
    max_overlap = max(0.0, min(0.95, float(max_target_hair_overlap)))
    non_hair_area = completion_non_hair.flatten(1).sum(dim=1, keepdim=True)
    hair_area = completion_hair.flatten(1).sum(dim=1, keepdim=True)
    allowed_hair = max_overlap / max(1.0 - max_overlap, 1e-6) * non_hair_area
    hair_scale = torch.where(
        hair_area > 1e-6,
        torch.minimum(torch.ones_like(hair_area), allowed_hair / hair_area.clamp_min(1e-6)),
        torch.zeros_like(hair_area),
    ).view(-1, 1, 1, 1)
    completion = (completion_non_hair + completion_hair * hair_scale).clamp(0, 1)

    object_mask = torch.clamp(core + completion, 0, 1)
    hoop_hole = compute_earring_hole_mask(object_mask) * active
    filled = torch.clamp(object_mask + hoop_hole, 0, 1)
    # A hoop must not grow beyond its detected wire; for all objects the write
    # gate remains object-supported rather than a broad geometric dilation.
    has_hoop = (hoop_hole.flatten(1).sum(dim=1) > 0).view(-1, 1, 1, 1)
    applied_dilate = torch.where(
        has_hoop,
        torch.ones_like(has_hoop, dtype=torch.int64),
        torch.full_like(has_hoop, max(1, int(write_dilate)), dtype=torch.int64),
    )
    # Completion/object masks are already pixel-supported.  Keeping the final
    # dilation at one avoids admitting source background at a wire edge.  The
    # explicit value remains in debug output to prove hoop handling.
    write = object_mask * (1.0 - hoop_hole).clamp(0, 1) * active
    return {
        "core_mask": core.clamp(0, 1),
        "completion_mask": completion.clamp(0, 1),
        "write_mask": write.clamp(0, 1),
        "earring_object_mask": object_mask.clamp(0, 1),
        "earring_filled_mask": filled.clamp(0, 1),
        "hoop_hole_mask": hoop_hole.clamp(0, 1),
        "write_dilate": applied_dilate,
    }


def build_earring_write_mask(
    confident_earring_mask: torch.Tensor,
    visible_ear_roi: torch.Tensor,
    target_hair_occlusion_mask: torch.Tensor,
    earlobe_anchor_mask: torch.Tensor,
    no_earring: bool | float | torch.Tensor = False,
    **kwargs,
) -> torch.Tensor:
    """Compatibility wrapper returning the final object-supported write mask."""

    trusted = kwargs.pop("trusted_earring_mask", None)
    if trusted is None:
        trusted = kwargs.pop("parser_earring_mask", confident_earring_mask)
    masks = build_earring_write_masks(
        trusted,
        confident_earring_mask,
        visible_ear_roi,
        target_hair_occlusion_mask,
        earlobe_anchor_mask,
        no_earring=no_earring,
        **kwargs,
    )
    return masks["write_mask"]


def select_reference_earring_mask(
    clean_source_earring_mask: torch.Tensor,
    candidate_source_earring_mask: torch.Tensor,
    left_roi: torch.Tensor,
    right_roi: torch.Tensor,
    *,
    support_dilate: int = 13,
    min_clean_area: float = 8.0,
    min_candidate_area: float = 12.0,
    max_candidate_density: float = 0.40,
    candidate_balance_ratio: float = 0.70,
    missing_side_clean_ratio: float = 0.85,
) -> torch.Tensor:
    clean_source_earring_mask = ensure_mask_4d(clean_source_earring_mask).float()
    candidate_source_earring_mask = ensure_mask_4d(candidate_source_earring_mask).float()
    left_roi = ensure_mask_4d(left_roi).float()
    right_roi = ensure_mask_4d(right_roi).float()

    height, width = clean_source_earring_mask.shape[-2:]
    area_scale = float(height * width) / float(256 * 256)
    min_clean_area = float(min_clean_area) * area_scale
    min_candidate_area = float(min_candidate_area) * area_scale

    # Assign real connected components to the parser-defined ears.  Do not use
    # an image-centre split: crop/mirror/profile samples routinely put a left
    # ear component on the right half of the tensor (and vice versa).
    clean_left, clean_right = assign_components_to_ear_sides(
        clean_source_earring_mask,
        left_roi,
        right_roi,
        left_roi,
        right_roi,
    )
    candidate_left, candidate_right = assign_components_to_ear_sides(
        candidate_source_earring_mask,
        left_roi,
        right_roi,
        left_roi,
        right_roi,
    )

    def side_parts(
        side_roi: torch.Tensor,
        clean_side: torch.Tensor,
        candidate_side: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        clean_support = dilate_mask(clean_side, support_dilate) * side_roi
        parser_guided = torch.clamp(clean_side + candidate_side * clean_support, 0, 1)
        fallback = torch.clamp(clean_side + candidate_side, 0, 1)
        clean_area = clean_side.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
        candidate_area = candidate_side.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
        roi_area = side_roi.flatten(1).sum(dim=1).view(-1, 1, 1, 1).clamp_min(1.0)
        candidate_density = candidate_area / roi_area
        return {
            "parser_guided": parser_guided,
            "fallback": fallback,
            "clean_area": clean_area,
            "candidate_area": candidate_area,
            "candidate_density": candidate_density,
        }

    left = side_parts(left_roi, clean_left, candidate_left)
    right = side_parts(right_roi, clean_right, candidate_right)
    left_clean_present = left["clean_area"] >= min_clean_area
    right_clean_present = right["clean_area"] >= min_clean_area
    any_clean_present = left_clean_present | right_clean_present

    left_candidate_valid = (
        (left["candidate_area"] >= min_candidate_area)
        & (left["candidate_density"] <= max_candidate_density)
    )
    right_candidate_valid = (
        (right["candidate_area"] >= min_candidate_area)
        & (right["candidate_density"] <= max_candidate_density)
    )

    left_valid_area = torch.where(left_candidate_valid, left["candidate_area"], torch.zeros_like(left["candidate_area"]))
    right_valid_area = torch.where(right_candidate_valid, right["candidate_area"], torch.zeros_like(right["candidate_area"]))
    max_candidate_area = torch.maximum(left_valid_area, right_valid_area).clamp_min(1.0)
    left_candidate_balanced = left["candidate_area"] >= candidate_balance_ratio * max_candidate_area
    right_candidate_balanced = right["candidate_area"] >= candidate_balance_ratio * max_candidate_area
    max_clean_area = torch.maximum(left["clean_area"], right["clean_area"]).clamp_min(1.0)
    left_candidate_matches_clean = left["candidate_area"] >= missing_side_clean_ratio * max_clean_area
    right_candidate_matches_clean = right["candidate_area"] >= missing_side_clean_ratio * max_clean_area

    left_keep_candidate = left_candidate_valid & torch.where(
        any_clean_present,
        left_candidate_matches_clean,
        left_candidate_balanced,
    )
    right_keep_candidate = right_candidate_valid & torch.where(
        any_clean_present,
        right_candidate_matches_clean,
        right_candidate_balanced,
    )
    left_selected = torch.where(
        left_clean_present,
        left["parser_guided"],
        torch.where(left_keep_candidate, left["fallback"], torch.zeros_like(left["fallback"])),
    )
    right_selected = torch.where(
        right_clean_present,
        right["parser_guided"],
        torch.where(right_keep_candidate, right["fallback"], torch.zeros_like(right["fallback"])),
    )

    # Source earring presence check: prevent no-hole violations when the parser
    # mislabels source skin/neck as earring.  Only recover earrings that are
    # actually present in the source (non-trivial area).  This blocks the
    # "source has no earring but target gets a hole punched below the ear" symptom
    # the user reported.  A real earring (even a small stud) will have area well
    # above min_candidate_area after candidate completion; mislabeled flat skin has
    # near-zero candidate area (edge/brightness filters reject it).
    min_presence_area = min_candidate_area * 0.8  # slightly below candidate threshold to allow small studs
    left_has_earring = (left_selected.flatten(1).sum(dim=1, keepdim=True) >= min_presence_area).float().view(-1, 1, 1, 1)
    right_has_earring = (right_selected.flatten(1).sum(dim=1, keepdim=True) >= min_presence_area).float().view(-1, 1, 1, 1)

    left_selected = left_selected * left_has_earring
    right_selected = right_selected * right_has_earring

    return torch.clamp(left_selected + right_selected, 0, 1)


def expand_valid_roi_by_completion(
    earring_valid_roi: torch.Tensor,
    completed_earring_mask: torch.Tensor,
    *,
    grow_iters: int = 30,
    seed_dilate: int = 3,
) -> torch.Tensor:
    """Grow the placement ROI to include the full completed earring object.

    The `earring_valid_roi` is a thin geometric shell built from the raw parser
    earring label.  When the parser under-segments a large hoop (missing its
    outer arc), the shell clips that arc, and multiplying the completed earring
    mask by the shell discards the recovered outer pixels — the classic
    "hoop is only 3/4 complete" symptom.

    This function fixes that WITHOUT re-admitting background: it starts from the
    part of the completed earring already inside the shell, then geodesically
    grows along the completed earring mask.  Only completed-earring pixels that
    are spatially CONNECTED to the shell are added; disconnected background
    islands are never reached.  The result is a ROI that hugs the full earring
    object while remaining zero on fully-covered sides (where the shell, and
    thus the seed, is empty).
    """
    valid = ensure_mask_4d(earring_valid_roi).float()
    completed = ensure_mask_4d(completed_earring_mask).float()
    if completed.shape[-2:] != valid.shape[-2:]:
        completed = F.interpolate(completed, size=valid.shape[-2:], mode="nearest")
    completed_bin = (completed > 0.5).float()

    # Thin hoop wires routinely have one- or two-pixel antialiasing gaps at
    # 256px.  Build a temporary connectivity guide for traversal only, then
    # return the original visual pixels.  Returning the guide itself would
    # reintroduce the background-filled hoop and ear-hole failure.
    bridge = max(1, int(seed_dilate))
    guide = dilate_mask(completed_bin, bridge)
    # Seed: completed-earring pixels already inside (or touching) the shell.
    seed = (dilate_mask(valid, bridge) * guide).clamp(0, 1)
    if seed.sum() == 0:
        return valid

    grown = seed
    for _ in range(int(grow_iters)):
        nxt = (dilate_mask(grown, 3) * guide).clamp(0, 1)
        if nxt.sum() == grown.sum():
            break
        grown = nxt

    reached_visual = completed_bin * dilate_mask(grown, bridge)
    return torch.clamp(valid + reached_visual, 0, 1)


def _weighted_centroid(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = ensure_mask_4d(mask).float()
    batch, _, height, width = mask.shape
    y_coords = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    x_coords = torch.arange(width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
    denom = mask.sum(dim=(-2, -1), keepdim=True)
    valid = denom.view(batch) > 1.0
    safe_denom = denom.clamp_min(1.0)
    y_center = (mask * y_coords).sum(dim=(-2, -1), keepdim=True) / safe_denom
    x_center = (mask * x_coords).sum(dim=(-2, -1), keepdim=True) / safe_denom
    return y_center.view(batch), x_center.view(batch), valid


def shift_tensor_per_batch(
    tensor: torch.Tensor,
    shift_y: torch.Tensor,
    shift_x: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_x = torch.zeros_like(shift_y) if shift_x is None else shift_x
    shifted = torch.zeros_like(tensor)
    height, width = tensor.shape[-2:]
    for idx in range(tensor.size(0)):
        sy = int(shift_y[idx].item())
        sx = int(shift_x[idx].item())

        src_y0 = max(0, -sy)
        src_y1 = min(height, height - sy)
        dst_y0 = max(0, sy)
        dst_y1 = min(height, height + sy)
        src_x0 = max(0, -sx)
        src_x1 = min(width, width - sx)
        dst_x0 = max(0, sx)
        dst_x1 = min(width, width + sx)
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            continue
        shifted[idx, :, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[idx, :, src_y0:src_y1, src_x0:src_x1]
    return shifted


def align_earring_reference_to_target(
    source_01: torch.Tensor,
    source_earring_mask: torch.Tensor,
    source_left_ear_mask: torch.Tensor,
    source_right_ear_mask: torch.Tensor,
    target_left_ear_mask: torch.Tensor,
    target_right_ear_mask: torch.Tensor,
    left_roi: torch.Tensor,
    right_roi: torch.Tensor,
    *,
    max_vertical_shift: int = 12,
    max_horizontal_shift: int = 6,
    reference_base: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    source_01 = normalized_to_01(source_01)
    source_earring_mask = ensure_mask_4d(source_earring_mask).float()
    left_roi = ensure_mask_4d(left_roi).float()
    right_roi = ensure_mask_4d(right_roi).float()
    source_left_ear_mask = ensure_mask_4d(source_left_ear_mask).float()
    source_right_ear_mask = ensure_mask_4d(source_right_ear_mask).float()
    target_left_ear_mask = ensure_mask_4d(target_left_ear_mask).float()
    target_right_ear_mask = ensure_mask_4d(target_right_ear_mask).float()

    left_ring = source_earring_mask * left_roi
    right_ring = source_earring_mask * right_roi
    left_source_anchor = (source_left_ear_mask + left_ring).clamp(0, 1) * left_roi
    right_source_anchor = (source_right_ear_mask + right_ring).clamp(0, 1) * right_roi
    left_target_anchor = target_left_ear_mask * left_roi
    right_target_anchor = target_right_ear_mask * right_roi

    left_src_y, left_src_x, left_src_valid = _weighted_centroid(left_source_anchor)
    left_tgt_y, left_tgt_x, left_tgt_valid = _weighted_centroid(left_target_anchor)
    right_src_y, right_src_x, right_src_valid = _weighted_centroid(right_source_anchor)
    right_tgt_y, right_tgt_x, right_tgt_valid = _weighted_centroid(right_target_anchor)

    left_valid = left_src_valid & left_tgt_valid & (left_ring.flatten(1).sum(dim=1) > 1.0)
    right_valid = right_src_valid & right_tgt_valid & (right_ring.flatten(1).sum(dim=1) > 1.0)
    left_shift_y = torch.where(left_valid, torch.round(left_tgt_y - left_src_y), torch.zeros_like(left_src_y))
    left_shift_x = torch.where(left_valid, torch.round(left_tgt_x - left_src_x), torch.zeros_like(left_src_x))
    right_shift_y = torch.where(right_valid, torch.round(right_tgt_y - right_src_y), torch.zeros_like(right_src_y))
    right_shift_x = torch.where(right_valid, torch.round(right_tgt_x - right_src_x), torch.zeros_like(right_src_x))

    left_shift_y = left_shift_y.clamp(-max_vertical_shift, max_vertical_shift)
    right_shift_y = right_shift_y.clamp(-max_vertical_shift, max_vertical_shift)
    left_shift_x = left_shift_x.clamp(-max_horizontal_shift, max_horizontal_shift)
    right_shift_x = right_shift_x.clamp(-max_horizontal_shift, max_horizontal_shift)

    shifted_left_mask = shift_tensor_per_batch(left_ring, left_shift_y, left_shift_x)
    shifted_right_mask = shift_tensor_per_batch(right_ring, right_shift_y, right_shift_x)
    shifted_left_rgb = shift_tensor_per_batch(source_01 * left_ring, left_shift_y, left_shift_x)
    shifted_right_rgb = shift_tensor_per_batch(source_01 * right_ring, right_shift_y, right_shift_x)

    original_ring = (left_ring + right_ring).clamp(0, 1)
    aligned_mask = (shifted_left_mask + shifted_right_mask).clamp(0, 1)
    earring_rgb = shifted_left_rgb + shifted_right_rgb
    aligned_support = dilate_mask(aligned_mask, 3)
    if reference_base is None:
        source_cleanup_mask = dilate_mask(original_ring, 3) * (1.0 - aligned_support).clamp(0, 1)
        source_cleanup_fill = low_pass_filter(source_01, kernel_size=9, sigma=2.0)
        clean_reference_base = (
            source_01 * (1.0 - source_cleanup_mask)
            + source_cleanup_fill * source_cleanup_mask
        ).clamp(0, 1)
    else:
        clean_reference_base = normalized_to_01(reference_base)
        if tuple(clean_reference_base.shape[-2:]) != tuple(source_01.shape[-2:]):
            clean_reference_base = F.interpolate(
                clean_reference_base,
                size=source_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0, 1)
        source_cleanup_mask = torch.zeros_like(original_ring)
    aligned_source = (clean_reference_base * (1.0 - aligned_mask) + earring_rgb).clamp(0, 1)

    return {
        "earring_reference": aligned_source,
        "earring_confident_mask": aligned_mask,
        "source_earring_original_mask": original_ring,
        "source_earring_cleanup_mask": source_cleanup_mask,
        "source_left_earring_aligned_mask": shifted_left_mask,
        "source_right_earring_aligned_mask": shifted_right_mask,
        "left_earring_shift_y": left_shift_y.view(-1, 1),
        "left_earring_shift_x": left_shift_x.view(-1, 1),
        "right_earring_shift_y": right_shift_y.view(-1, 1),
        "right_earring_shift_x": right_shift_x.view(-1, 1),
    }


def combine_cleanup_masks(cleanup_masks: dict[str, torch.Tensor]) -> torch.Tensor | None:
    masks = [ensure_mask_4d(mask).float() for mask in cleanup_masks.values() if torch.is_tensor(mask)]
    if not masks:
        return None
    return torch.stack(masks, dim=0).amax(dim=0).clamp(0, 1)


def build_revealed_skin_mask(
    cleanup_masks: dict[str, torch.Tensor],
    target_parsing: torch.Tensor,
    source_parsing: torch.Tensor | None = None,
    target_hair_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    earring_mask: torch.Tensor | None = None,
    source_image: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cleanup_mask = combine_cleanup_masks(cleanup_masks)
    if cleanup_mask is None:
        base = parsing_label_mask(target_parsing, RAW_FACE_SURFACE_LABELS)
        cleanup_mask = torch.zeros_like(base)

    face_surface = parsing_label_mask(target_parsing, RAW_FACE_SURFACE_LABELS)
    target_detail = parsing_label_mask(target_parsing, RAW_DETAIL_LABELS)
    remove_face = cleanup_masks.get("M_remove_face", torch.zeros_like(cleanup_mask))
    remove_halo = cleanup_masks.get("M_remove_halo", torch.zeros_like(cleanup_mask))
    remove_face = resize_mask(remove_face, cleanup_mask.shape[-2:])
    remove_halo = resize_mask(remove_halo, cleanup_mask.shape[-2:])
    cleanup_inner_edge = (cleanup_mask - erode_mask(cleanup_mask, 5)).clamp(0, 1)

    target_hair_mask = torch.zeros_like(cleanup_mask) if target_hair_mask is None else resize_mask(target_hair_mask, cleanup_mask.shape[-2:])
    source_hair_mask = torch.zeros_like(cleanup_mask) if source_hair_mask is None else resize_mask(source_hair_mask, cleanup_mask.shape[-2:])
    earring_mask = torch.zeros_like(cleanup_mask) if earring_mask is None else resize_mask(earring_mask, cleanup_mask.shape[-2:])

    revealed = (
        remove_face
        + 0.45 * cleanup_inner_edge * dilate_mask(remove_face + remove_halo, 5)
        + 0.22 * remove_halo * dilate_mask(remove_face, 5)
    ).clamp(0, 1)
    revealed = (
        revealed
        * face_surface
        * (1.0 - target_hair_mask).clamp(0, 1)
        * (1.0 - dilate_mask(target_detail, 3)).clamp(0, 1)
        * (1.0 - dilate_mask(earring_mask, 3)).clamp(0, 1)
    ).clamp(0, 1)

    # The transition ring is part of the learnable forehead repair.  Keeping it
    # separate from the core prevents the old PP/cleanup objectives from
    # fighting the repair exactly at the visible skin boundary.
    revealed_binary = (revealed > 0.05).float()
    safe_target_skin = (
        face_surface
        * (1.0 - target_hair_mask).clamp(0, 1)
        * (1.0 - dilate_mask(target_detail, 3)).clamp(0, 1)
        * (1.0 - dilate_mask(earring_mask, 3)).clamp(0, 1)
    ).clamp(0, 1)
    revealed_feather = gaussian_blur(
        dilate_mask(revealed_binary, 9),
        kernel_size=15,
        sigma=3.0,
    ).clamp(0, 1) * safe_target_skin
    revealed_seam = (
        revealed_feather - erode_mask(revealed_binary, 5)
    ).clamp(0, 1) * safe_target_skin
    revealed_blend = torch.maximum(revealed, revealed_feather).clamp(0, 1)

    # Source pixels hidden by bangs are never valid skin references.  The
    # reference mask also rejects parsing boundaries, facial details and
    # earrings, where copying high-frequency content would create black hairs
    # or accessory fragments in the forehead.
    if source_parsing is not None:
        source_parsing = ensure_mask_4d(source_parsing).long()
        source_face_skin = (source_parsing == 1).float()
        source_parser_hair = (source_parsing == RAW_HAIR).float()
        source_detail = parsing_label_mask(source_parsing, RAW_DETAIL_LABELS)
        source_earring = (source_parsing == RAW_EARRING).float()
    else:
        source_face_skin = torch.zeros_like(revealed)
        source_parser_hair = torch.zeros_like(revealed)
        source_detail = torch.zeros_like(revealed)
        source_earring = torch.zeros_like(revealed)

    source_hair_guard = dilate_mask(torch.clamp(source_hair_mask + source_parser_hair, 0, 1), 5)
    source_detail_guard = dilate_mask(source_detail, 3)
    source_earring_guard = dilate_mask(torch.clamp(source_earring + earring_mask, 0, 1), 5)
    source_visible_skin = (
        source_face_skin
        * (1.0 - source_hair_guard).clamp(0, 1)
        * (1.0 - source_detail_guard).clamp(0, 1)
        * (1.0 - source_earring_guard).clamp(0, 1)
        * (1.0 - dilate_mask(revealed_binary, 3)).clamp(0, 1)
    ).clamp(0, 1)

    if source_image is not None:
        source_rgb = normalized_to_01(source_image)
        if source_rgb.shape[-2:] != source_visible_skin.shape[-2:]:
            source_rgb = F.interpolate(
                source_rgb,
                size=source_visible_skin.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        source_gray = (
            0.299 * source_rgb[:, 0:1]
            + 0.587 * source_rgb[:, 1:2]
            + 0.114 * source_rgb[:, 2:3]
        )
        source_gray_low = low_pass_filter(source_gray, kernel_size=9, sigma=2.0)
        source_edge_energy = (source_gray - source_gray_low).abs()
        mask_denom = source_visible_skin.flatten(2).sum(dim=2).clamp_min(1.0)
        gray_mean = (source_gray_low * source_visible_skin).flatten(2).sum(dim=2) / mask_denom
        gray_var = (
            (source_gray_low - gray_mean.unsqueeze(-1).unsqueeze(-1)).pow(2)
            * source_visible_skin
        ).flatten(2).sum(dim=2) / mask_denom
        edge_mean = (source_edge_energy * source_visible_skin).flatten(2).sum(dim=2) / mask_denom
        edge_var = (
            (source_edge_energy - edge_mean.unsqueeze(-1).unsqueeze(-1)).pow(2)
            * source_visible_skin
        ).flatten(2).sum(dim=2) / mask_denom
        dark_limit = (gray_mean - 2.5 * torch.sqrt(gray_var + 1e-6)).clamp(0.04, 0.55)
        edge_limit = (
            edge_mean + 3.0 * torch.sqrt(edge_var + 1e-6) + 0.01
        ).clamp_min(0.02)
        photometric_safe = (
            (source_gray_low >= dark_limit.unsqueeze(-1).unsqueeze(-1))
            & (source_edge_energy <= edge_limit.unsqueeze(-1).unsqueeze(-1))
        ).float()
        source_visible_skin = source_visible_skin * photometric_safe
    source_visible_skin = erode_mask(source_visible_skin, 3).clamp(0, 1)

    # Prefer visible skin near the newly exposed forehead.  If bangs leave too
    # little local evidence, fall back per sample to the full visible face
    # (mostly cheeks/lower face), never to SATD target pixels.
    local_reference = source_visible_skin * dilate_mask(revealed_binary, 65)
    area_scale = float(revealed.shape[-2] * revealed.shape[-1]) / float(256 * 256)
    has_local_reference = (
        local_reference.flatten(1).sum(dim=1) >= 64.0 * area_scale
    ).view(-1, 1, 1, 1)
    source_visible_skin_reference = torch.where(
        has_local_reference,
        local_reference,
        source_visible_skin,
    ).clamp(0, 1)

    return {
        "cleanup_mask": cleanup_mask,
        "cleanup_inner_edge": cleanup_inner_edge,
        "revealed_skin_mask": revealed,
        "revealed_skin_seam_mask": revealed_seam,
        "revealed_skin_blend_mask": revealed_blend,
        "source_visible_skin_reference_mask": source_visible_skin_reference,
        # Backward-compatible name.  It now has the same strict no-hair,
        # no-detail semantics as the explicit source reference mask.
        "source_skin_valid_mask": source_visible_skin_reference,
        "target_face_surface_mask": face_surface,
    }


def build_raw_face_masks(parsing: torch.Tensor) -> dict[str, torch.Tensor]:
    parsing = ensure_mask_4d(parsing).long()
    return {
        "left_ear": (parsing == RAW_LEFT_EAR).float(),
        "right_ear": (parsing == RAW_RIGHT_EAR).float(),
        "earring": (parsing == RAW_EARRING).float(),
        "face_surface": parsing_label_mask(parsing, RAW_FACE_SURFACE_LABELS),
        "skin_surface": parsing_label_mask(parsing, RAW_SKIN_SURFACE_LABELS),
        "ear_surface": parsing_label_mask(parsing, RAW_EAR_SURFACE_LABELS),
        "detail": torch.stack([(parsing == label).float() for label in RAW_DETAIL_LABELS], dim=0).amax(dim=0),
        "hair": (parsing == RAW_HAIR).float(),
        "hat": (parsing == RAW_HAT).float(),
    }


class FaceParsingHelperV5:
    def __init__(self, parse_size: int = 512):
        self.parse_size = parse_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def parse(self, images: torch.Tensor, out_size: tuple[int, int] | None = None) -> torch.Tensor:
        images = normalized_to_01(images)
        out_size = out_size or tuple(images.shape[-2:])
        parse_images = F.interpolate(images, size=(self.parse_size, self.parse_size), mode="bilinear",
                                     align_corners=False)

        device = parse_images.device
        norm_mean = self.mean.to(device=device, dtype=parse_images.dtype)
        norm_std = self.std.to(device=device, dtype=parse_images.dtype)
        parse_images = (parse_images - norm_mean) / norm_std

        if device.type != "cuda":
            parse_images = parse_images.to("cuda")

        parsings = []
        for image in parse_images:
            parsing, _ = FaceParsing_tensor.parsing_img(image.unsqueeze(0))
            parsings.append(parsing)

        parsing = torch.stack(parsings, dim=0).unsqueeze(1).float()
        parsing = F.interpolate(parsing, size=out_size, mode="nearest")
        return parsing.long().to(images.device)


class HairMaskExtractorV5(nn.Module):
    def __init__(self, device: str = "cuda", dilate_erosion: int = 5):
        super().__init__()
        from utils.image_utils import DilateErosion

        self.device = device
        self.seg = BiSeNet(n_classes=16).to(device)
        self.seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth", map_location=device))
        self.seg.eval()
        for param in self.seg.parameters():
            param.requires_grad = False

        self.downsample_512 = BicubicDownSample(factor=2)
        self.dilate_erosion = DilateErosion(dilate_erosion=dilate_erosion, device=device)

    @torch.no_grad()
    def generate_mask(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = normalized_to_01(images).to(self.device)
        seg_in = (self.downsample_512(images) - seg_mean) / seg_std
        seg_logits, _, _ = self.seg(seg_in)
        current_mask = torch.argmax(seg_logits, dim=1).float()
        hair_mask = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        hair_mask = F.interpolate(hair_mask.unsqueeze(1), size=(256, 256), mode="nearest")
        return self.dilate_erosion.mask(hair_mask)


class EarAnchoredQueryBuilder(nn.Module):
    def __init__(
        self,
        ear_dilate: int = 21,
        hair_change_dilate: int = 25,
        earring_expand: int = 15,
        downward_shift: int = 10,
        target_hair_dilate: int = 11,
        earring_occlusion_dilate: int = 3,
        source_hair_block_dilate: int = 8,
        source_hair_block_strength: float = 0.95,
        target_visibility_expand: int = 5,
        max_target_hair_overlap: float = 0.30,
        min_target_visible_overlap: float = 0.10,
        min_target_ear_area: float = 8.0,
        earring_channel_down: int = 32,
        earring_align_max_shift: int = 12,
        earring_grow_iters: int = 15,
        earring_grow_kernel: int = 7,
        earring_bridge_dilate: int = 9,
        earring_shell_dilate: int = 9,
    ):
        super().__init__()
        self.ear_dilate = ear_dilate
        self.hair_change_dilate = hair_change_dilate
        self.earring_expand = earring_expand
        self.downward_shift = downward_shift
        self.target_hair_dilate = target_hair_dilate
        self.earring_occlusion_dilate = earring_occlusion_dilate
        self.source_hair_block_dilate = source_hair_block_dilate
        self.source_hair_block_strength = max(0.0, min(1.0, float(source_hair_block_strength)))
        self.target_visibility_expand = target_visibility_expand
        self.max_target_hair_overlap = max_target_hair_overlap
        self.min_target_visible_overlap = min_target_visible_overlap
        self.min_target_ear_area = min_target_ear_area
        self.earring_channel_down = max(8, int(earring_channel_down))
        self.earring_align_max_shift = earring_align_max_shift
        # Earring-guided geodesic growth: iters * (kernel//2) bounds the maximum
        # reach below the lobe (e.g. 6 * 3 = 18px per step chain), but growth is
        # confined to the source earring mask, so it only travels where an earring
        # actually exists — long earrings get followed, absent earrings do not.
        self.earring_grow_iters = max(0, int(earring_grow_iters))
        self.earring_grow_kernel = max(3, int(earring_grow_kernel))
        self.earring_bridge_dilate = max(1, int(earring_bridge_dilate))
        # Shell thickness around the earring pixels below the lobe.  Small enough
        # to hug the earring (excluding a hoop's hollow centre and the exterior
        # background), large enough to catch thin wires the parser under-segments.
        self.earring_shell_dilate = max(1, int(earring_shell_dilate))

    def _split_by_side(
        self,
        mask: torch.Tensor,
        left_hint: torch.Tensor,
        right_hint: torch.Tensor,
        left_lobe_anchor: torch.Tensor | None = None,
        right_lobe_anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign candidates by ear overlap/anchor distance, never x-midline."""

        return assign_components_to_ear_sides(
            mask,
            left_hint,
            right_hint,
            left_lobe_anchor,
            right_lobe_anchor,
        )

    @staticmethod
    def _roi_overlap(mask: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        mask = ensure_mask_4d(mask).float()
        roi = ensure_mask_4d(roi).float()
        numer = (mask * roi).flatten(1).sum(dim=1)
        denom = roi.flatten(1).sum(dim=1).clamp_min(1.0)
        return numer / denom

    def forward(
        self,
        source_parsing: torch.Tensor,
        target_parsing: torch.Tensor,
        source_hair_mask: torch.Tensor | None = None,
        target_hair_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source_masks = build_raw_face_masks(source_parsing)
        target_masks = build_raw_face_masks(target_parsing)

        source_hair_mask = source_masks["hair"] if source_hair_mask is None else ensure_mask_4d(source_hair_mask).float()
        target_hair_mask = target_masks["hair"] if target_hair_mask is None else ensure_mask_4d(target_hair_mask).float()

        left_hint = dilate_mask(source_masks["left_ear"] + target_masks["left_ear"], max(3, self.ear_dilate // 2))
        right_hint = dilate_mask(source_masks["right_ear"] + target_masks["right_ear"], max(3, self.ear_dilate // 2))
        source_left_lobe = build_earlobe_anchor(source_masks["left_ear"], ear_roi=left_hint)
        source_right_lobe = build_earlobe_anchor(source_masks["right_ear"], ear_roi=right_hint)
        target_left_lobe = build_earlobe_anchor(target_masks["left_ear"], ear_roi=left_hint)
        target_right_lobe = build_earlobe_anchor(target_masks["right_ear"], ear_roi=right_hint)
        source_left_ring, source_right_ring = self._split_by_side(
            source_masks["earring"],
            left_hint,
            right_hint,
            source_left_lobe + target_left_lobe,
            source_right_lobe + target_right_lobe,
        )

        # Expand the raw parser earring first to cover under-segmented regions
        # (e.g., the outer arc of a large hoop that the parser missed), so the
        # ROI boundary can reach it.  Without this, left_roi built from the raw
        # parser label clips the outer edge, and no downstream completion can
        # recover pixels outside the ROI.
        ring_hint = dilate_mask(source_masks["earring"], self.earring_expand)
        source_left_ring_expanded, source_right_ring_expanded = self._split_by_side(
            ring_hint,
            left_hint,
            right_hint,
            source_left_lobe + target_left_lobe,
            source_right_lobe + target_right_lobe,
        )

        left_roi = dilate_mask(
            source_masks["left_ear"] + target_masks["left_ear"] + source_left_ring + source_left_ring_expanded,
            self.ear_dilate,
        )
        right_roi = dilate_mask(
            source_masks["right_ear"] + target_masks["right_ear"] + source_right_ring + source_right_ring_expanded,
            self.ear_dilate,
        )
        left_roi = torch.clamp(left_roi + shift_mask(left_roi, down=self.downward_shift), 0, 1)
        right_roi = torch.clamp(right_roi + shift_mask(right_roi, down=self.downward_shift), 0, 1)
        ear_roi = torch.clamp(left_roi + right_roi, 0, 1)

        hair_change = dilate_mask((source_hair_mask - target_hair_mask).abs(), self.hair_change_dilate)
        ring_hint = dilate_mask(source_masks["earring"], self.earring_expand)
        hat_mask = torch.clamp(source_masks["hat"] + target_masks["hat"], 0, 1)
        target_hair_context = dilate_mask(target_hair_mask, self.target_hair_dilate)
        target_ear_hair_context = dilate_mask(target_hair_mask, self.earring_occlusion_dilate)
        source_hair_context = dilate_mask(source_hair_mask, self.source_hair_block_dilate)
        source_earring_keep = dilate_mask(source_masks["earring"], max(3, self.earring_expand // 2))

        target_open_mask = (1 - target_ear_hair_context).clamp(0, 1) * (1 - hat_mask)
        left_visible_roi = left_roi * target_open_mask
        right_visible_roi = right_roi * target_open_mask
        visible_ear_roi = torch.clamp(left_visible_roi + right_visible_roi, 0, 1)

        # Ordinary detail recovery stays restricted to genuinely open pixels.
        # An earring is different: when the lobe is visible, its hanging part
        # may be in front of target hair.  Open only a narrow lobe-to-downward
        # channel for the earring branch; never open the complete ear ROI.
        def build_earring_channel(
            side_roi: torch.Tensor,
            visible_side_roi: torch.Tensor,
            target_ear_mask: torch.Tensor,
            source_side_ring: torch.Tensor,
            source_lobe_anchor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            side_area = side_roi.flatten(1).sum(dim=1).clamp_min(1.0)
            visible_ratio = (visible_side_roi * side_roi).flatten(1).sum(dim=1) / side_area
            hair_ratio = (target_ear_hair_context * side_roi).flatten(1).sum(dim=1) / side_area
            ear_area = (target_ear_mask * side_roi).flatten(1).sum(dim=1)
            area_scale = float(side_roi.shape[-1] * side_roi.shape[-2]) / float(256 * 256)

            parser_visible = (
                (ear_area >= float(self.min_target_ear_area) * area_scale)
                & (visible_ratio >= max(float(self.min_target_visible_overlap), 0.05))
            )
            semantic_fallback = (
                side_roi
                * target_masks["skin_surface"]
                * (1.0 - target_ear_hair_context).clamp(0, 1)
                * (1.0 - hat_mask).clamp(0, 1)
            ).clamp(0, 1)
            fallback_area = semantic_fallback.flatten(1).sum(dim=1)
            fallback_visible = (
                (visible_ratio >= max(float(self.min_target_visible_overlap), 0.08))
                & (hair_ratio <= min(float(self.max_target_hair_overlap), 0.35))
                & (fallback_area >= float(self.min_target_ear_area) * area_scale)
            )
            # A source label-9 earring is already trusted object evidence.  The
            # transferred target may render the earlobe as generic skin or omit
            # its ear label completely, so do not discard that side solely for
            # a parser disagreement.  It must still be target-open and cannot
            # pass through a hair-covered/hat-covered ear.
            source_ring_area = (source_side_ring * side_roi).flatten(1).sum(dim=1)
            source_ring_visible = (
                (source_ring_area >= 2.0 * area_scale)
                & (visible_ratio >= 0.03)
                & (hair_ratio <= 0.55)
            )
            side_visible = (
                parser_visible | fallback_visible | source_ring_visible
            ).float().view(-1, 1, 1, 1)

            # Parser lower lobe first; semantic lower-half skin is only a
            # fallback when parser ear pixels are sparse.  The source lobe is
            # aligned to the target frame and is a safe last resort for a
            # parser-missed target ear when an explicit source earring exists.
            target_lobe_anchor = build_earlobe_anchor(
                target_ear_mask,
                fallback_skin_mask=semantic_fallback,
                ear_roi=side_roi,
                lower_ratio=0.62,
                dilate=3,
            ) * (1.0 - target_ear_hair_context).clamp(0, 1)
            source_lobe_anchor = (
                ensure_mask_4d(source_lobe_anchor).float()
                * side_roi
                * (1.0 - target_ear_hair_context).clamp(0, 1)
                * (1.0 - hat_mask).clamp(0, 1)
            )
            target_lobe_present = (
                target_lobe_anchor.flatten(1).sum(dim=1) >= 2.0 * area_scale
            ).view(-1, 1, 1, 1)
            lobe_anchor = torch.where(target_lobe_present, target_lobe_anchor, source_lobe_anchor)

            # Earring-guided downward channel.  A blind geometric box below the
            # lobe opens background/neck when there is no earring (holes) and is
            # too short for long earrings.  Instead, grow the channel ALONG the
            # source earring pixels via geodesic dilation seeded at the lobe, so
            # the channel reaches exactly as far as the earring actually hangs:
            #   - no source earring  -> channel stays at the lobe   (no hole)
            #   - short earring      -> channel covers just the stud (perfect)
            #   - long earring       -> channel follows it all the way down
            ring_guide = (ensure_mask_4d(source_side_ring).float() * side_roi).clamp(0, 1)
            ring_present = ring_guide.flatten(1).amax(dim=1).view(-1, 1, 1, 1) > 0.01

            # If there is NO earring at all on this side, the channel is just the
            # lobe (no downward growth) to prevent holes.  Only grow when an
            # earring is actually present.
            channel_no_ring = lobe_anchor
            if ring_guide.sum().item() > 0:
                # Bridge small parser gaps between the lobe and the earring so the
                # geodesic growth can travel from the ear into the earring.
                growth_guide = torch.clamp(
                    dilate_mask(ring_guide, self.earring_bridge_dilate) + lobe_anchor, 0, 1
                )
                grown = lobe_anchor.clamp(0, 1)
                for _ in range(int(self.earring_grow_iters)):
                    grown = (dilate_mask(grown, self.earring_grow_kernel) * growth_guide).clamp(0, 1)

                # The grown region is a SOLID convex-ish blob: for a large hoop
                # earring it also covers the hoop's central hole and the exterior
                # background, whose "source content" is background/hair, not the
                # earring.  Recovering those creates the hole you see.  So the
                # valid channel is NOT the solid blob — it is a THIN SHELL that
                # hugs the actual earring pixels: dilate the earring mask by a
                # small margin (to catch thin wires the parser misses) and
                # intersect with the reachable grown region.  The hoop's hollow
                # centre and the surrounding background are excluded, because
                # they are not earring pixels.
                earring_shell = dilate_mask(ring_guide, self.earring_shell_dilate)
                # Lobe stays fully valid (the stud/short-earring case behind the
                # lobe is genuine ear, safe to recover); below the lobe only the
                # earring shell is valid.
                below_lobe = (grown - lobe_anchor).clamp(0, 1)
                channel_with_ring = torch.clamp(
                    lobe_anchor + below_lobe * earring_shell, 0, 1
                ) * side_roi
            else:
                channel_with_ring = channel_no_ring

            # Per-sample selection: use the grown channel only for samples where
            # an earring is present; otherwise keep it at the lobe to prevent holes.
            channel = torch.where(ring_present, channel_with_ring, channel_no_ring) * side_visible
            return (
                side_visible,
                channel.clamp(0, 1),
                visible_ratio,
                hair_ratio,
                parser_visible.float().view(-1, 1, 1, 1),
                fallback_visible.float().view(-1, 1, 1, 1),
                lobe_anchor.clamp(0, 1),
            )

        (
            left_side_visible,
            left_earring_channel,
            left_visible_ratio,
            left_hair_ratio,
            left_parser_visible,
            left_fallback_visible,
            left_lobe_anchor,
        ) = build_earring_channel(
            left_roi,
            left_visible_roi,
            target_masks["left_ear"],
            source_left_ring,
            source_left_lobe,
        )
        (
            right_side_visible,
            right_earring_channel,
            right_visible_ratio,
            right_hair_ratio,
            right_parser_visible,
            right_fallback_visible,
            right_lobe_anchor,
        ) = build_earring_channel(
            right_roi,
            right_visible_roi,
            target_masks["right_ear"],
            source_right_ring,
            source_right_lobe,
        )
        left_earring_valid_roi = torch.clamp(
            left_visible_roi * left_side_visible + left_earring_channel,
            0,
            1,
        )
        right_earring_valid_roi = torch.clamp(
            right_visible_roi * right_side_visible + right_earring_channel,
            0,
            1,
        )
        earring_valid_roi = torch.clamp(
            left_earring_valid_roi + right_earring_valid_roi,
            0,
            1,
        ) * (1 - hat_mask)
        target_ear_hair_occlusion_mask = (ear_roi * target_hair_mask * (1 - hat_mask)).clamp(0, 1)
        source_earring_detection_mask = (source_masks["earring"] * ear_roi * (1 - hat_mask)).clamp(0, 1)

        source_hair_block_mask = (
            visible_ear_roi
            * source_hair_context
            * (1 - target_hair_context)
            * (1 - source_earring_keep)
            * (1 - hat_mask)
        ).clamp(0, 1)
        block_attenuation = (1 - self.source_hair_block_strength * source_hair_block_mask).clamp(0, 1)
        query_seed = torch.clamp(hair_change + 0.25 * ring_hint, 0, 1)
        query_mask = visible_ear_roi * query_seed * block_attenuation
        query_mask = query_mask * (1 - hat_mask)

        fallback_query = visible_ear_roi * (1 - hat_mask) * block_attenuation
        empty_query = query_mask.flatten(1).amax(dim=1).view(-1, 1, 1, 1) == 0
        query_mask = torch.where(empty_query, fallback_query, query_mask)

        source_left_parser_earring = source_left_ring
        source_right_parser_earring = source_right_ring
        source_left_ring = source_left_ring * left_earring_valid_roi
        source_right_ring = source_right_ring * right_earring_valid_roi
        left_presence = source_left_ring.flatten(1).amax(dim=1)
        right_presence = source_right_ring.flatten(1).amax(dim=1)
        presence_target = torch.stack(
            [left_presence, right_presence, torch.maximum(left_presence, right_presence)],
            dim=1,
        ).float()
        visibility_target = torch.stack(
            [
                left_side_visible.flatten(1).amax(dim=1),
                right_side_visible.flatten(1).amax(dim=1),
                torch.maximum(
                    left_side_visible.flatten(1).amax(dim=1),
                    right_side_visible.flatten(1).amax(dim=1),
                ),
            ],
            dim=1,
        ).float()

        source_earring_mask = source_masks["earring"] * earring_valid_roi
        target_ear_mask = torch.clamp(target_masks["left_ear"] + target_masks["right_ear"], 0, 1)
        target_ear_boundary = (
            dilate_mask(target_ear_mask, 5) - erode_mask(target_ear_mask, 5)
        ).clamp(0, 1)
        target_ear_interior = erode_mask(target_ear_mask, 5).clamp(0, 1)

        return {
            "left_ear_roi": left_roi,
            "right_ear_roi": right_roi,
            "ear_roi": ear_roi,
            "visible_ear_roi": visible_ear_roi,
            "earring_valid_roi": earring_valid_roi,
            "left_earring_valid_roi": left_earring_valid_roi,
            "right_earring_valid_roi": right_earring_valid_roi,
            "earring_visibility_mask": earring_valid_roi,
            "target_ear_hair_occlusion_mask": target_ear_hair_occlusion_mask,
            "query_mask": query_mask,
            "source_earring_detection_mask": source_earring_detection_mask,
            "source_hair_block_mask": source_hair_block_mask,
            "source_hair_mask": source_hair_mask,
            "target_hair_mask": target_hair_mask,
            "source_earring_mask": source_earring_mask,
            "target_earring_mask": target_masks["earring"],
            "source_left_earring_mask": source_left_ring,
            "source_right_earring_mask": source_right_ring,
            "source_left_parser_earring": source_left_parser_earring,
            "source_right_parser_earring": source_right_parser_earring,
            "source_left_ear_mask": source_masks["left_ear"],
            "source_right_ear_mask": source_masks["right_ear"],
            "target_left_ear_mask": target_masks["left_ear"],
            "target_right_ear_mask": target_masks["right_ear"],
            "target_left_ear": target_masks["left_ear"],
            "target_right_ear": target_masks["right_ear"],
            "left_lobe_anchor": left_lobe_anchor,
            "right_lobe_anchor": right_lobe_anchor,
            "left_parser_visible": left_parser_visible,
            "right_parser_visible": right_parser_visible,
            "left_fallback_visible": left_fallback_visible,
            "right_fallback_visible": right_fallback_visible,
            "left_side_active": left_side_visible,
            "right_side_active": right_side_visible,
            "target_ear_boundary_protect_mask": target_ear_boundary,
            "target_ear_interior_mask": target_ear_interior,
            "source_face_surface_mask": source_masks["face_surface"],
            "target_face_surface_mask": target_masks["face_surface"],
            "source_skin_surface_mask": source_masks["skin_surface"],
            "target_skin_surface_mask": target_masks["skin_surface"],
            "presence_target": presence_target,
            "visibility_target": visibility_target,
            "left_target_ear_visible_ratio": left_visible_ratio.view(-1, 1),
            "right_target_ear_visible_ratio": right_visible_ratio.view(-1, 1),
            "left_target_ear_hair_ratio": left_hair_ratio.view(-1, 1),
            "right_target_ear_hair_ratio": right_hair_ratio.view(-1, 1),
        }


class ShadowSuppressedHFExtractor(nn.Module):
    def __init__(
        self,
        low_alpha: float = 0.1,
        feature_channels: int = 128,
        blur_kernel: int = 11,
        blur_sigma: float = 3.0,
    ):
        super().__init__()
        self.low_alpha = low_alpha
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

        self.low_gate = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.prior_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(
        self,
        source_01: torch.Tensor,
        query_mask: torch.Tensor,
        prior_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_mask = ensure_mask_4d(query_mask).float()
        prior_mask = query_mask if prior_mask is None else ensure_mask_4d(prior_mask).float()
        prior_mask = prior_mask * query_mask
        low = low_pass_filter(source_01, kernel_size=self.blur_kernel, sigma=self.blur_sigma)
        high = source_01 - low
        low_gate = self.low_gate(torch.cat([low, prior_mask], dim=1))
        prior = (high + self.low_alpha * low_gate * low) * prior_mask
        high_energy = high.abs().mean(dim=1, keepdim=True) * prior_mask
        prior_feature = self.prior_encoder(torch.cat([prior, prior_mask], dim=1))

        return {
            "low": low,
            "high": high,
            "prior": prior,
            "prior_feature": prior_feature,
            "low_gate": low_gate,
            "high_energy": high_energy,
            "prior_mask": prior_mask,
        }


class DynamicFineMaskRefresher(nn.Module):
    def __init__(self, hidden_channels: int = 32, init_bias: float = -4.0, fine_support_dilate: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(12, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.mask_head = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels // 2, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 3),
        )
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, float(init_bias))
        nn.init.zeros_(self.presence_head[-1].weight)
        nn.init.zeros_(self.presence_head[-1].bias)

    def forward(
        self,
        source_01: torch.Tensor,
        target_01: torch.Tensor,
        high_energy: torch.Tensor,
        query_mask: torch.Tensor,
        source_ear_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_mask = ensure_mask_4d(query_mask).float()
        source_ear_mask = ensure_mask_4d(source_ear_mask).float()
        residual = (source_01 - target_01).abs()
        features = self.backbone(torch.cat([source_01, target_01, residual, high_energy, query_mask, source_ear_mask], dim=1))
        fine_mask_logits = self.mask_head(features)
        fine_mask = torch.sigmoid(fine_mask_logits) * query_mask

        pooled_features = features * (query_mask + source_ear_mask).clamp(0, 1)
        presence_logits = self.presence_head(pooled_features)
        return {
            "fine_mask_logits": fine_mask_logits,
            "fine_mask": fine_mask,
            "presence_logits": presence_logits,
        }


class BrightnessReEstimator(nn.Module):
    def __init__(self, feature_channels: int = 128):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_gamma = nn.Sequential(
            nn.Linear(32 + feature_channels, feature_channels),
            nn.Tanh(),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(32 + feature_channels, feature_channels),
            nn.Tanh(),
        )
        nn.init.zeros_(self.to_gamma[0].weight)
        nn.init.zeros_(self.to_gamma[0].bias)
        nn.init.zeros_(self.to_beta[0].weight)
        nn.init.zeros_(self.to_beta[0].bias)

    def forward(
        self,
        target_01: torch.Tensor,
        prior_feature: torch.Tensor,
        fine_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fine_mask = resize_mask(fine_mask, target_01.shape[-2:])
        query_mask = resize_mask(query_mask, target_01.shape[-2:])
        context_input = torch.cat([target_01, fine_mask, query_mask, target_01 * query_mask], dim=1)
        context_embedding = self.context_encoder(context_input).flatten(1)
        prior_embedding = F.adaptive_avg_pool2d(prior_feature, 1).flatten(1)
        embedding = torch.cat([context_embedding, prior_embedding], dim=1)

        gamma = 1 + 0.1 * self.to_gamma(embedding).unsqueeze(-1).unsqueeze(-1)
        beta = 0.1 * self.to_beta(embedding).unsqueeze(-1).unsqueeze(-1)
        adjusted = gamma * prior_feature + beta
        return adjusted, {"gamma": gamma, "beta": beta}


class HFDAGatedInjectionUnit(nn.Module):
    def __init__(self, base_channels: int = 512, prior_channels: int = 128):
        super().__init__()
        self.prior_proj = nn.Sequential(
            nn.Conv2d(prior_channels, base_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
        )
        nn.init.zeros_(self.prior_proj[0].weight)
        if self.prior_proj[0].bias is not None:
            nn.init.zeros_(self.prior_proj[0].bias)
        nn.init.zeros_(self.fuse[2].weight)
        if self.fuse[2].bias is not None:
            nn.init.zeros_(self.fuse[2].bias)

    def forward(
        self,
        base_feature: torch.Tensor,
        prior_feature: torch.Tensor,
        fine_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fine_mask = resize_mask(fine_mask, base_feature.shape[-2:])
        projected_prior = self.prior_proj(prior_feature)
        delta = self.fuse(torch.cat([base_feature, projected_prior], dim=1))
        fused = base_feature + delta * fine_mask
        return fused, fine_mask
