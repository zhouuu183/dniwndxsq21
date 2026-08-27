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
RAW_NECK_SURFACE_LABELS = (14, 15)
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
        coordinate_scale = max(float(height), float(width)) / 256.0

        def nearest_component_distance(points: np.ndarray, component: np.ndarray) -> float:
            """Return the closest source-lobe/component-pixel distance.

            The component centroid is a poor side cue for a hanging pendant or
            a large hoop: its lower arc can be much closer to the opposite
            lobe than the centroid suggests.  Sampling caps the temporary
            pairwise array for high-resolution inference while retaining the
            nearest-pixel test.
            """

            if points.size == 0 or component.size == 0:
                return float("inf")
            if points.shape[0] > 512:
                points = points[np.linspace(0, points.shape[0] - 1, 512).astype(np.int64)]
            if component.shape[0] > 2048:
                component = component[
                    np.linspace(0, component.shape[0] - 1, 2048).astype(np.int64)
                ]
            delta = points.astype(np.float32)[:, None, :] - component.astype(np.float32)[None, :, :]
            return float(np.sqrt((delta * delta).sum(axis=2)).min())

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

            def anchor_score(points: np.ndarray) -> tuple[float, float]:
                if points.size == 0:
                    return 0.0, float("inf")
                # Use the nearest real lobe pixel, not the component centroid.
                distance = nearest_component_distance(points, coords)
                score = max(0.0, 1.0 - distance / max(48.0 * coordinate_scale, 1.0))
                return score, distance

            left_anchor_score, left_distance = anchor_score(left_anchor_points)
            right_anchor_score, right_distance = anchor_score(right_anchor_points)
            # A candidate must have real side-context overlap or be close to a
            # real lobe.  Without this gate, every unrelated foreground object
            # gets a tiny nonzero inverse-distance score and can be assigned.
            max_anchor_distance = 72.0 * coordinate_scale
            left_associated = left_overlap > 0.0 or left_distance <= max_anchor_distance
            right_associated = right_overlap > 0.0 or right_distance <= max_anchor_distance
            if not left_associated and not right_associated:
                continue
            left_score = left_overlap + left_anchor_score
            right_score = right_overlap + right_anchor_score
            if left_associated and not right_associated:
                destination = left_out
            elif right_associated and not left_associated:
                destination = right_out
            elif left_score > right_score:
                destination = left_out
            elif right_score > left_score:
                destination = right_out
            elif left_distance < right_distance:
                destination = left_out
            elif right_distance < left_distance:
                destination = right_out
            else:
                # Exact ties have no source-side evidence; do not let tensor
                # position decide which ear receives the object.
                continue
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
    # This topology mask never carries a gradient.  On a 1024px output the
    # former tensor flood-fill could require 1024 sequential max-pool passes,
    # which made validation and dataset generation disproportionately slow.
    # Connected components gives the same 8-connected border-reachability
    # result in one native pass per sample.
    if cv2 is not None:
        component_np = component.detach().cpu().numpy()[:, 0].astype(bool)
        holes_np = np.zeros_like(component_np, dtype=np.float32)
        for batch_index, object_pixels in enumerate(component_np):
            background_pixels = (~object_pixels).astype(np.uint8)
            component_count, labels = cv2.connectedComponents(
                background_pixels,
                connectivity=8,
            )
            if component_count <= 1:
                continue
            border_labels = np.unique(
                np.concatenate(
                    (
                        labels[0, :],
                        labels[-1, :],
                        labels[:, 0],
                        labels[:, -1],
                    )
                )
            )
            reachable = np.isin(labels, border_labels)
            holes_np[batch_index] = background_pixels.astype(bool) & ~reachable
        return torch.from_numpy(holes_np).unsqueeze(1).to(
            device=component.device,
            dtype=component.dtype,
        )

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


def _instance_parsing_mask(
    parsing: torch.Tensor | None,
    labels: tuple[int, ...],
    size: tuple[int, int],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Read raw parser labels at an RGB resolution without soft label edges."""

    if parsing is None:
        return torch.zeros_like(reference)
    labels_map = ensure_mask_4d(parsing).to(device=reference.device)
    if labels_map.shape[-2:] != size:
        labels_map = F.interpolate(labels_map.float(), size=size, mode="nearest")
    labels_map = labels_map.long()
    result = torch.zeros_like(labels_map, dtype=torch.bool)
    for label in labels:
        result |= labels_map == int(label)
    return result.to(dtype=reference.dtype)


def _instance_adaptive_threshold(
    value: torch.Tensor,
    support: torch.Tensor,
    *,
    std_scale: float,
    minimum_delta: float,
    minimum_area: float,
) -> torch.Tensor:
    """Per-image threshold used by the source-object locator.

    The earring detector must work for silver, black, gold and coloured
    earrings, so an absolute RGB threshold is brittle.  The threshold is
    evaluated only in the source ear/lobe corridor.
    """

    support = (ensure_mask_4d(support).float() > 0.5).to(dtype=value.dtype)
    if value.size(1) != 1:
        value = value.mean(dim=1, keepdim=True)
    flat_support = support.flatten(2)
    area = flat_support.sum(dim=2, keepdim=True).clamp_min(1.0)
    mean = (value * support).flatten(2).sum(dim=2, keepdim=True) / area
    mean = mean.view(-1, 1, 1, 1)
    variance = ((value - mean).pow(2) * support).flatten(2).sum(dim=2, keepdim=True) / area
    std = variance.sqrt().view(-1, 1, 1, 1)
    enough = (flat_support.sum(dim=2, keepdim=True) >= float(minimum_area)).view(-1, 1, 1, 1)
    return ((value > mean + float(std_scale) * std + float(minimum_delta)) * support * enough).to(value.dtype)


def _instance_filter_components(
    mask: torch.Tensor,
    support: torch.Tensor,
    *,
    minimum_area: int,
    maximum_area: int,
    keep_per_side: int = 3,
    left_context: torch.Tensor | None = None,
    right_context: torch.Tensor | None = None,
    left_lobe_anchor: torch.Tensor | None = None,
    right_lobe_anchor: torch.Tensor | None = None,
    require_side_context: bool = False,
) -> torch.Tensor:
    """Keep compact components after real source-side association.

    The old implementation sorted components by image x-coordinate before the
    source ears were consulted.  Crops, mirrored inputs and profile views make
    that split unreliable, and a good component on one side could suppress a
    valid one on the other.  When side context is supplied, each component is
    assigned from source-ear overlap / lobe proximity first, then ranked only
    against components for that same side.
    """

    binary = ((ensure_mask_4d(mask).float() > 0.5) * (ensure_mask_4d(support).float() > 0.05)).detach()
    output = torch.zeros_like(binary)
    if binary.flatten(1).amax().item() <= 0:
        return output

    binary_np = binary.cpu().numpy().astype(np.uint8)
    reference = binary
    left_context_np = (_resize_like_mask(left_context, reference).detach().cpu().numpy() > 0.05)
    right_context_np = (_resize_like_mask(right_context, reference).detach().cpu().numpy() > 0.05)
    left_anchor_np = (_resize_like_mask(left_lobe_anchor, reference).detach().cpu().numpy() > 0.05)
    right_anchor_np = (_resize_like_mask(right_lobe_anchor, reference).detach().cpu().numpy() > 0.05)
    output_np = np.zeros_like(binary_np, dtype=np.uint8)
    for batch_idx in range(binary_np.shape[0]):
        # Context is a property of one source sample, never of the entire
        # batch.  A sample without a parsed ear/lobe must not inherit another
        # sample's side-aware policy (or vice versa).
        use_side_context = bool(
            left_context_np[batch_idx, 0].any()
            or right_context_np[batch_idx, 0].any()
            or left_anchor_np[batch_idx, 0].any()
            or right_anchor_np[batch_idx, 0].any()
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_np[batch_idx, 0],
            connectivity=8,
        ) if cv2 is not None else (0, None, None, None)
        if component_count <= 1:
            # OpenCV is optional in import-only environments.  Parser labels
            # are still retained by the caller, while visual recall becomes a
            # no-op instead of producing an unchecked broad mask.
            continue
        left_best: list[tuple[float, int, int]] = []
        right_best: list[tuple[float, int, int]] = []
        global_best: list[tuple[int, int]] = []
        height, width = binary_np.shape[-2:]
        coordinate_scale = max(float(height), float(width)) / 256.0

        def distance_map(mask_np: np.ndarray) -> np.ndarray | None:
            if not mask_np.any():
                return None
            # ``distanceTransform`` measures distance to zeros, so invert the
            # foreground side cue before querying component pixels.
            return cv2.distanceTransform((~mask_np).astype(np.uint8), cv2.DIST_L2, 3)

        left_context_distance = distance_map(left_context_np[batch_idx, 0])
        right_context_distance = distance_map(right_context_np[batch_idx, 0])
        left_anchor_distance = distance_map(left_anchor_np[batch_idx, 0])
        right_anchor_distance = distance_map(right_anchor_np[batch_idx, 0])

        def side_score(
            component: np.ndarray,
            context_np: np.ndarray,
            context_distance: np.ndarray | None,
            anchor_distance: np.ndarray | None,
        ) -> tuple[float, bool]:
            area = max(1, int(component.sum()))
            overlap = float((component & context_np).sum()) / float(area)
            context_near = (
                float(context_distance[component].min())
                if context_distance is not None
                else float("inf")
            )
            anchor_near = (
                float(anchor_distance[component].min())
                if anchor_distance is not None
                else float("inf")
            )
            # Direct source-ear corridor overlap is authoritative.  A very
            # small parser/seed component may instead start immediately beside
            # the lobe, but a distant textured component cannot enter merely
            # because it is on the same half of the image.
            associated = (
                overlap > 0.0
                or context_near <= 13.0 * coordinate_scale
                or anchor_near <= 72.0 * coordinate_scale
            )
            if not associated:
                return 0.0, False
            score = (
                2.50 * overlap
                + max(0.0, 1.0 - context_near / max(32.0 * coordinate_scale, 1.0))
                + max(0.0, 1.25 - anchor_near / max(64.0 * coordinate_scale, 1.0))
            )
            return score, True

        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < int(minimum_area) or area > int(maximum_area):
                continue
            component = labels == component_id
            if not use_side_context:
                if not require_side_context:
                    global_best.append((area, component_id))
                continue
            left_score, left_associated = side_score(
                component,
                left_context_np[batch_idx, 0],
                left_context_distance,
                left_anchor_distance,
            )
            right_score, right_associated = side_score(
                component,
                right_context_np[batch_idx, 0],
                right_context_distance,
                right_anchor_distance,
            )
            if left_associated and not right_associated:
                left_best.append((left_score, area, component_id))
            elif right_associated and not left_associated:
                right_best.append((right_score, area, component_id))
            elif left_associated and right_associated:
                if left_score > right_score:
                    left_best.append((left_score, area, component_id))
                elif right_score > left_score:
                    right_best.append((right_score, area, component_id))
                # A perfect tie is source-side ambiguous.  Dropping it is
                # safer than assigning with an image-centre fallback.
        if use_side_context:
            for collection in (left_best, right_best):
                collection.sort(reverse=True)
                for _, _, component_id in collection[:max(1, int(keep_per_side))]:
                    output_np[batch_idx, 0][labels == component_id] = 1
        else:
            global_best.sort(reverse=True)
            # Preserve the legacy total-capacity budget for callers that have
            # no side geometry, without inventing an image-midline split.
            for _, component_id in global_best[:max(1, 2 * int(keep_per_side))]:
                output_np[batch_idx, 0][labels == component_id] = 1
    return torch.from_numpy(output_np).to(device=mask.device, dtype=mask.dtype)


def _retain_raw_components_touching_proxy(
    raw_mask: torch.Tensor,
    accepted_proxy: torch.Tensor,
    *,
    minimum_area: int,
    maximum_area: int,
    left_context: torch.Tensor | None = None,
    right_context: torch.Tensor | None = None,
    left_lobe_anchor: torch.Tensor | None = None,
    right_lobe_anchor: torch.Tensor | None = None,
    require_side_context: bool = True,
) -> torch.Tensor:
    """Keep full raw components only after a real ear-side proxy accepted them.

    The proxy is intentionally compact: it answers whether a parser component
    belongs to a real source ear/lobe corridor.  It must not also define the
    object extent.  Applying the proxy directly to label-9 was cutting long
    pendants, hook wires and hoop arcs to the small lobe neighbourhood before
    the high-resolution stage ever saw them.  Conversely, restoring *all* raw
    label-9 pixels reintroduced parser mistakes as black source patches.

    This helper therefore keeps a complete raw connected component only when
    that component touches an already accepted proxy, is itself associated with
    one real source ear/lobe in the same sample, and its original area is
    plausible.  The source-side split still happens afterwards, so no image
    midpoint is used to decide the receiving ear.
    """

    raw = (ensure_mask_4d(raw_mask).float() > 0.5).detach()
    proxy = (ensure_mask_4d(accepted_proxy).float() > 0.5).detach()
    if proxy.shape[-2:] != raw.shape[-2:]:
        proxy = F.interpolate(proxy.float(), size=raw.shape[-2:], mode="nearest") > 0.5
    left_context = _resize_like_mask(left_context, raw)
    right_context = _resize_like_mask(right_context, raw)
    left_anchor = _resize_like_mask(left_lobe_anchor, raw)
    right_anchor = _resize_like_mask(right_lobe_anchor, raw)
    output = torch.zeros_like(raw)
    if cv2 is None or not bool(raw.flatten(1).amax().item() > 0):
        # The production V5 runtime provides OpenCV for the high-resolution
        # locator.  In an import-only environment, fail closed rather than
        # turning every raw parser pixel into a direct RGB write mask.
        return output

    raw_np = raw.cpu().numpy().astype(np.uint8)
    proxy_np = proxy.cpu().numpy().astype(np.uint8)
    left_context_np = left_context.detach().cpu().numpy() > 0.05
    right_context_np = right_context.detach().cpu().numpy() > 0.05
    left_anchor_np = left_anchor.detach().cpu().numpy() > 0.05
    right_anchor_np = right_anchor.detach().cpu().numpy() > 0.05
    output_np = np.zeros_like(raw_np, dtype=np.uint8)
    for batch_idx in range(raw_np.shape[0]):
        # Context must belong to this source sample.  Without it, a raw label
        # near an unrelated crop/background object has no side or lobe proof
        # and must never gain RGB authority from another batch item.
        left_anchor_points = np.argwhere(left_anchor_np[batch_idx, 0])
        right_anchor_points = np.argwhere(right_anchor_np[batch_idx, 0])
        has_side_context = bool(
            left_context_np[batch_idx, 0].any()
            or right_context_np[batch_idx, 0].any()
            or left_anchor_points.size > 0
            or right_anchor_points.size > 0
        )
        if require_side_context and not has_side_context:
            continue

        height, width = raw_np.shape[-2:]
        coordinate_scale = max(float(height), float(width)) / 256.0

        def nearest_anchor_distance(points: np.ndarray, component_points: np.ndarray) -> float:
            if points.size == 0 or component_points.size == 0:
                return float("inf")
            if points.shape[0] > 512:
                points = points[np.linspace(0, points.shape[0] - 1, 512).astype(np.int64)]
            if component_points.shape[0] > 2048:
                component_points = component_points[
                    np.linspace(0, component_points.shape[0] - 1, 2048).astype(np.int64)
                ]
            delta = (
                points.astype(np.float32)[:, None, :]
                - component_points.astype(np.float32)[None, :, :]
            )
            return float(np.sqrt((delta * delta).sum(axis=2)).min())

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw_np[batch_idx, 0],
            connectivity=8,
        )
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < int(minimum_area) or area > int(maximum_area):
                continue
            component = labels == component_id
            if not np.any(component & (proxy_np[batch_idx, 0] > 0)):
                continue
            component_points = np.argwhere(component)

            def side_is_associated(
                context: np.ndarray,
                anchor_points: np.ndarray,
            ) -> tuple[float, bool]:
                context_overlap = bool(np.any(component & context))
                anchor_distance = nearest_anchor_distance(anchor_points, component_points)
                # The lobe is the primary proof.  Direct source-ear corridor
                # overlap is retained only for parser-degenerate tiny ears
                # where no lower-lobe anchor can be formed at all.
                associated = (
                    anchor_distance <= 72.0 * coordinate_scale
                    if anchor_points.size > 0
                    else context_overlap
                )
                if not associated:
                    return 0.0, False
                score = (
                    (1.0 if context_overlap else 0.0)
                    + max(0.0, 1.0 - anchor_distance / max(72.0 * coordinate_scale, 1.0))
                )
                return score, True

            left_score, left_associated = side_is_associated(
                left_context_np[batch_idx, 0], left_anchor_points
            )
            right_score, right_associated = side_is_associated(
                right_context_np[batch_idx, 0], right_anchor_points
            )
            if not left_associated and not right_associated:
                continue
            if left_associated and right_associated and left_score == right_score:
                # An unresolved two-ear tie is not a safe object association.
                continue
            output_np[batch_idx, 0][component] = 1
    return torch.from_numpy(output_np).to(device=raw_mask.device, dtype=raw_mask.dtype)


def _build_parser_miss_visual_instances_v5(
    source_01: torch.Tensor,
    visual_seed: torch.Tensor,
    visual_support: torch.Tensor,
    parser_earring: torch.Tensor,
    associated_seed: torch.Tensor,
    source_background: torch.Tensor,
    background_permission: torch.Tensor,
    source_hair: torch.Tensor,
    source_ear: torch.Tensor,
    source_face_surface: torch.Tensor,
    left_context: torch.Tensor,
    right_context: torch.Tensor,
    left_recall_context: torch.Tensor,
    right_recall_context: torch.Tensor,
    left_lobe_anchor: torch.Tensor,
    right_lobe_anchor: torch.Tensor,
    left_parser_instance: torch.Tensor,
    right_parser_instance: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Find strict native-resolution ordinary earrings missed by parser label 9.

    This is an ordinary-instance path.  It never fits an ellipse, fills a
    contour hole, or returns a geometric footprint.  A separately returned
    seed may ask the dedicated paired-contour verifier to inspect a native
    closed hoop, but that seed has no RGB authority by itself.  Every accepted
    component must:

    * be in one sample's real source ear/lobe corridor and within lobe distance;
    * have independent high-frequency/chroma evidence;
    * avoid source background, hair, and eroded ear/face interiors; and
    * touch a narrow exterior lobe attachment band while staying compact.

    The returned masks contain only observed visual evidence pixels.  Temporary
    morphology is used for component grouping and attachment tests, never as a
    direct RGB write region.  A parser label or associated coarse seed on a side
    suppresses this fallback for that side, leaving the parser-first long-tail
    path authoritative there.
    """

    reference = ensure_mask_4d(source_01).float()
    zeros = torch.zeros_like(reference[:, :1])
    if cv2 is None:
        return {
            "visual_recall_instance_mask": zeros,
            "left_visual_recall_instance_mask": zeros,
            "right_visual_recall_instance_mask": zeros,
            "visual_recall_seed_mask": zeros,
            "left_visual_recall_seed_mask": zeros,
            "right_visual_recall_seed_mask": zeros,
            "visual_recall_attachment_mask": zeros,
            "left_visual_recall_attachment_mask": zeros,
            "right_visual_recall_attachment_mask": zeros,
            "visual_recall_candidate_mask": zeros,
            "left_visual_recall_candidate_mask": zeros,
            "right_visual_recall_candidate_mask": zeros,
            "visual_recall_hoop_seed_mask": zeros,
            "left_visual_recall_hoop_seed_mask": zeros,
            "right_visual_recall_hoop_seed_mask": zeros,
            "visual_recall_present": zeros[:, :, :1, :1],
            "left_visual_recall_present": zeros[:, :, :1, :1],
            "right_visual_recall_present": zeros[:, :, :1, :1],
        }

    image_size = tuple(reference.shape[-2:])

    def mask_np(value: torch.Tensor | None) -> np.ndarray:
        value = _resize_like_mask(value, reference)
        return value.detach().cpu().numpy()[:, 0] > 0.5

    source_np = np.clip(
        reference.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)
    seed_np = mask_np(visual_seed)
    support_np = mask_np(visual_support)
    parser_np = mask_np(parser_earring)
    associated_np = mask_np(associated_seed)
    background_np = mask_np(source_background)
    background_permission_np = mask_np(background_permission)
    hair_np = mask_np(source_hair)
    ear_np = mask_np(source_ear)
    face_np = mask_np(source_face_surface)
    left_context_np = mask_np(left_context)
    right_context_np = mask_np(right_context)
    left_recall_context_np = mask_np(left_recall_context)
    right_recall_context_np = mask_np(right_recall_context)
    left_anchor_np = mask_np(left_lobe_anchor)
    right_anchor_np = mask_np(right_lobe_anchor)
    left_parser_np = mask_np(left_parser_instance)
    right_parser_np = mask_np(right_parser_instance)

    batch, height, width = parser_np.shape
    scale = max(height, width) / 256.0

    def kernel(value: float, minimum: int = 3) -> int:
        size = max(int(minimum), int(round(float(value) * scale)))
        return size if size % 2 == 1 else size + 1

    def empty() -> np.ndarray:
        return np.zeros((height, width), dtype=bool)

    left_out = np.zeros((batch, 1, height, width), dtype=np.float32)
    right_out = np.zeros_like(left_out)
    left_seed_out = np.zeros_like(left_out)
    right_seed_out = np.zeros_like(left_out)
    left_attach_out = np.zeros_like(left_out)
    right_attach_out = np.zeros_like(left_out)
    left_candidate_out = np.zeros_like(left_out)
    right_candidate_out = np.zeros_like(left_out)
    left_hoop_seed_out = np.zeros_like(left_out)
    right_hoop_seed_out = np.zeros_like(left_out)

    def trace_side(
        image: np.ndarray,
        candidate_seed: np.ndarray,
        candidate_support: np.ndarray,
        parser: np.ndarray,
        associated: np.ndarray,
        background: np.ndarray,
        background_permission: np.ndarray,
        hair: np.ndarray,
        ear: np.ndarray,
        face: np.ndarray,
        context: np.ndarray,
        anchor: np.ndarray,
        parser_instance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        zero = empty()
        anchor_points = np.argwhere(anchor)
        if anchor_points.size == 0:
            return zero, zero, zero, zero, zero

        # A parser-miss fallback is side-local and compact.  A raw parser
        # instance remains authoritative, but an associated coarse seed is only
        # a search hint, not an observed source object.  Suppressing recall for
        # that seed made every background-labelled parser miss impossible to
        # recover, even after the low-resolution branch had identified the
        # correct lobe.
        anchor_y, anchor_x = anchor_points.mean(axis=0)
        y_grid, x_grid = np.ogrid[:height, :width]
        local_window = (
            (y_grid >= anchor_y - 36.0 * scale)
            & (y_grid <= anchor_y + 144.0 * scale)
            & (x_grid >= anchor_x - 84.0 * scale)
            & (x_grid <= anchor_x + 84.0 * scale)
        )
        # The lobe-relative box bounds the maximum ordinary accessory size;
        # the real source-side corridor proves that this is the matching ear.
        # A tiny dilation handles parsing boundaries but is never returned.
        local_window &= cv2.dilate(
            context.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(5), kernel(5))),
        ).astype(bool)
        # A short accepted parser arc must not disable native recall for this
        # entire side.  It is a connection proof for the *same* accessory, not
        # permission to add an unrelated second object.  The component loop
        # below therefore requires any extension to touch this narrow link.
        parser_link = cv2.dilate(
            (parser_instance & local_window).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(7), kernel(7))),
        ).astype(bool)
        parser_present = bool(parser_link.any())
        # ``associated`` is the accepted low-resolution source-side seed.  It
        # is deliberately not copied as RGB (upsampling it would recreate the
        # old blocky ear/background patch), but it must participate as a
        # *connection proof*.  Previously this argument was passed all the way
        # into this function and then never used, so a parser-missed earring
        # could be located at 256px yet had no way to start native-resolution
        # recall.  Keep the link local to this measured ear corridor; the
        # native edge/chroma, attachment and component checks below remain the
        # actual object authority.
        associated_link = cv2.dilate(
            (associated & local_window).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(7), kernel(7))),
        ).astype(bool)
        associated_present = bool(associated_link.any())
        source_link = parser_link | associated_link

        # Do not let broad morphology turn a background/hair edge into a write
        # pixel.  The exterior attachment belt is used only to prove contact;
        # pixels inside the semantic ear/face cores remain forbidden.
        hair_guard = cv2.dilate(
            hair.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(5), kernel(5))),
        ).astype(bool)
        background_guard = cv2.dilate(
            background.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(3), kernel(3))),
        ).astype(bool)
        # A parser-missed metal/gem pixel is frequently labelled background.
        # It can be considered only where the independently validated strong
        # seed has asked for a native-resolution inspection.  The permission is
        # still not RGB authority: the component below must pass edge/chroma,
        # lobe attachment and compactness checks before it is returned.
        if background_permission.any():
            background_permission = cv2.dilate(
                background_permission.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(3), kernel(3))),
            ).astype(bool)
            background_guard &= ~background_permission
        # A parser-recognised short arc may be surrounded by background-labelled
        # metal.  Open only the immediate native inspection band around that
        # already accepted arc; the pixels still require edge/chroma evidence
        # and a connected accepted component before they can be copied.
        if parser_present:
            parser_background_permission = (
                background
                & parser_link
                & local_window
            )
            background_guard &= ~parser_background_permission
        if associated_present:
            # A real metal/gem accessory is frequently label-0 around a
            # low-resolution accepted seed.  Open only image-evidence pixels
            # in this already side-associated inspection window.  The final
            # connected component must still touch ``source_link`` and the
            # exterior lobe attachment band, so this is not a broad
            # background write permission.
            associated_background_permission = (
                background
                & local_window
                & cv2.dilate(
                    candidate_seed.astype(np.uint8) | candidate_support.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(3), kernel(3))),
                ).astype(bool)
            )
            background_guard &= ~associated_background_permission
        ear_core = cv2.erode(
            ear.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(7), kernel(7))),
        ).astype(bool)
        face_core = cv2.erode(
            face.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(7), kernel(7))),
        ).astype(bool)
        attachment_band = (
            cv2.dilate(
                anchor.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(17), kernel(5))),
            ).astype(bool)
            & ~ear_core
            & (~face_core | associated_present)
            & ~hair_guard
            & ~background_guard
            & local_window
        )
        valid = (
            local_window
            & ~hair_guard
            & ~background_guard
            & ~ear_core
            # A reliable low-resolution source seed may be attached to an
            # accessory that the parser incorrectly labelled as face surface
            # (very common for dark discs and cheek-side pendants).  Keep the
            # face-core exclusion for unseeded visual recall, but let the
            # seed-linked branch inspect native structure there.  It still has
            # to pass the lobe-link/component checks below before any source
            # pixel can become an instance alpha.
            & (~face_core | associated_present)
        )
        if int(valid.sum()) < max(24, int(round(8.0 * scale * scale))):
            return zero, zero, zero, zero, zero

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        values = gray[valid]
        if values.size < max(24, int(round(8.0 * scale * scale))):
            return zero, zero, zero, zero, zero
        low = max(12, int(np.percentile(values, 35)))
        high = max(low + 18, int(np.percentile(values, 82)))
        edges = cv2.Canny(gray, low, min(255, high)) > 0
        blurred = cv2.GaussianBlur(image, (kernel(9), kernel(9)), 0)
        local_delta = np.abs(image.astype(np.int16) - blurred.astype(np.int16)).mean(axis=2)
        chroma = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
        delta_floor = max(5.0, float(np.percentile(local_delta[valid], 76)))
        chroma_floor = max(10.0, float(np.percentile(chroma[valid], 84)))
        edge_band = cv2.dilate(
            edges.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(3), kernel(3))),
        ).astype(bool)
        evidence = (
            edge_band
            & ((local_delta >= delta_floor) | (chroma >= chroma_floor))
            & valid
        )
        # A high-resolution edge is evidence, not an object detector by
        # itself.  Letting every lobe-local edge start a component recreated
        # earrings on grass, hair and ear folds.  ``candidate_seed`` is either
        # non-background source evidence or an explicitly associated strong
        # low-resolution proposal; both cases still need this native edge test
        # before any RGB pixel is accepted.
        seed = candidate_seed & evidence & valid
        support = candidate_support & evidence & valid
        candidate = seed | support
        if associated_present:
            # The associated seed is a locator only.  It contributes no RGB
            # pixels, but permits measured native edges in the same strict
            # lobe-local window to form a connected component.  Without this
            # path, a parser-missed solid earring labelled as face/background
            # has an empty ``candidate_seed`` and can never reach the later
            # high-resolution object verifier.
            candidate |= evidence
        if int(candidate.sum()) < max(3, int(round(0.75 * scale * scale))):
            return zero, zero, zero, zero, zero

        proxy = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(3), kernel(3))),
        ).astype(bool)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            proxy.astype(np.uint8),
            connectivity=8,
        )
        selected = zero.copy()
        selected_seed = zero.copy()
        selected_attachment = zero.copy()
        selected_candidate = zero.copy()
        selected_hoop_seed = zero.copy()
        # This is a *native visual candidate* budget, not an earring-stud
        # classifier.  The former 360px-at-256 limit rejects the complete
        # contour of a large disk/pendant before the high-resolution verifier
        # can decide whether its interior is safe.  Keep an image-relative cap
        # against background components while allowing a real ornament body.
        max_area = min(
            max(8, int(round(0.055 * height * width))),
            max(8, int(round(1400.0 * scale * scale))),
        )
        # Keep small native studs after resolution scaling.  They still need
        # the exterior/attachment/edge checks below; this is not a free area
        # relaxation for background components.
        min_area = max(3, int(round(0.75 * scale * scale)))
        max_width = max(12, int(round(84.0 * scale)))
        max_height = max(20, int(round(144.0 * scale)))
        attachment_proxy = cv2.dilate(
            attachment_band.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(5), kernel(5))),
        ).astype(bool)
        best_score = -1.0
        for component_id in range(1, component_count):
            component = labels == component_id
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            bbox_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            bbox_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            if area < min_area or area > max_area or bbox_width > max_width or bbox_height > max_height:
                continue
            points = np.argwhere(component)
            if points.size == 0:
                continue
            distance = np.sqrt(
                (points[:, 0] - anchor_y) ** 2 + (points[:, 1] - anchor_x) ** 2
            ).min()
            if distance > 72.0 * scale:
                continue
            if not np.any(component & attachment_proxy):
                continue
            if (parser_present or associated_present) and not np.any(component & source_link):
                # Neither a parsed fragment nor an accepted low-resolution
                # locator may license a second unrelated ear-side texture.
                # A native component has to meet one of those side-local links
                # before it can become a recovery candidate.
                continue
            object_pixels = candidate & component
            object_area = int(object_pixels.sum())
            if object_area < min_area:
                continue
            # A parser-missed fallback must contain a visible exterior part.
            # A high-contrast ear fold or lobe boundary can satisfy every
            # texture test while remaining wholly inside labels 7/8 (or the
            # face surface); allowing it would synthesize a small earring on
            # an accessory-free source.  Parser-confirmed pixels are handled
            # by the parser-first path and do not reach this fallback.
            # Background-labelled pixels are allowed only inside the narrow
            # strong-candidate inspection permission created by the caller.
            # Treat them as exterior for this check, but do not generalise that
            # exception to the rest of the parser background.
            exterior = object_pixels & (
                ~(ear | face | hair | background)
                | (background & background_permission)
                | (face & associated_present)
            )
            exterior_area = int(exterior.sum())
            if exterior_area < max(1, int(np.ceil(0.18 * object_area))):
                continue
            density = float(object_area) / float(max(1, bbox_width * bbox_height))
            if density < 0.015:
                continue
            attachment_pixels = object_pixels & attachment_band
            attachment_area = int(attachment_pixels.sum())
            if attachment_area < max(1, int(round(0.5 * scale))):
                continue
            score = (
                float(attachment_area) * 4.0
                + float(object_area)
                + max(0.0, 1.0 - distance / max(72.0 * scale, 1.0)) * 8.0
            )
            if score <= best_score:
                continue
            best_score = score
            selected[:] = False
            selected_seed[:] = False
            selected_attachment[:] = False
            selected_candidate[:] = False
            selected_hoop_seed[:] = False
            selected[object_pixels] = True
            selected_seed[candidate_seed & component] = True
            selected_attachment[attachment_pixels] = True
            selected_candidate[component] = True
            # A visual candidate may authorize the paired-contour verifier only
            # when its native support already encloses a compact interior.  The
            # verifier still has to confirm two independent boundaries and
            # lobe contact; this seed never grants RGB authority or hole pixels.
            component_crop = component.astype(np.uint8)
            component_closed = cv2.morphologyEx(
                component_crop,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(5), kernel(5))),
            )
            contour_info = cv2.findContours(
                component_closed,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_NONE,
            )
            contours = contour_info[-2]
            hierarchy = contour_info[-1]
            hoop_like = False
            if hierarchy is not None and contours:
                for contour_index, contour in enumerate(contours):
                    parent = int(hierarchy[0, contour_index, 3])
                    if parent < 0:
                        continue
                    outer_area = float(cv2.contourArea(contours[parent]))
                    inner_area = float(cv2.contourArea(contour))
                    if outer_area <= 1.0 or inner_area < max(3.0, 0.04 * outer_area):
                        continue
                    if inner_area > 0.70 * outer_area:
                        continue
                    outer_x, outer_y, outer_w, outer_h = cv2.boundingRect(contours[parent])
                    ratio = min(outer_w, outer_h) / max(float(max(outer_w, outer_h)), 1.0)
                    if ratio < 0.38:
                        continue
                    hoop_like = True
                    break
            if hoop_like:
                selected_hoop_seed[candidate_seed & component] = True
        return selected, selected_seed, selected_attachment, selected_candidate, selected_hoop_seed

    for batch_idx in range(batch):
        (
            left_out[batch_idx, 0],
            left_seed_out[batch_idx, 0],
            left_attach_out[batch_idx, 0],
            left_candidate_out[batch_idx, 0],
            left_hoop_seed_out[batch_idx, 0],
        ) = trace_side(
            source_np[batch_idx],
            seed_np[batch_idx],
            support_np[batch_idx],
            parser_np[batch_idx],
            associated_np[batch_idx],
            background_np[batch_idx],
            background_permission_np[batch_idx],
            hair_np[batch_idx],
            ear_np[batch_idx],
            face_np[batch_idx],
            # The visual fallback may follow a verified long pendant below the
            # compact parser/hoop corridor.  Parser instance membership and
            # hoop evidence remain compact-gated elsewhere; only this native
            # recall trace receives the extended rail.
            left_recall_context_np[batch_idx],
            left_anchor_np[batch_idx],
            left_parser_np[batch_idx],
        )
        (
            right_out[batch_idx, 0],
            right_seed_out[batch_idx, 0],
            right_attach_out[batch_idx, 0],
            right_candidate_out[batch_idx, 0],
            right_hoop_seed_out[batch_idx, 0],
        ) = trace_side(
            source_np[batch_idx],
            seed_np[batch_idx],
            support_np[batch_idx],
            parser_np[batch_idx],
            associated_np[batch_idx],
            background_np[batch_idx],
            background_permission_np[batch_idx],
            hair_np[batch_idx],
            ear_np[batch_idx],
            face_np[batch_idx],
            right_recall_context_np[batch_idx],
            right_anchor_np[batch_idx],
            right_parser_np[batch_idx],
        )

    left = torch.from_numpy(left_out).to(reference.device, reference.dtype)
    right = torch.from_numpy(right_out).to(reference.device, reference.dtype)
    left_seed = torch.from_numpy(left_seed_out).to(reference.device, reference.dtype)
    right_seed = torch.from_numpy(right_seed_out).to(reference.device, reference.dtype)
    left_attachment = torch.from_numpy(left_attach_out).to(reference.device, reference.dtype)
    right_attachment = torch.from_numpy(right_attach_out).to(reference.device, reference.dtype)
    left_candidate = torch.from_numpy(left_candidate_out).to(reference.device, reference.dtype)
    right_candidate = torch.from_numpy(right_candidate_out).to(reference.device, reference.dtype)
    left_hoop_seed = torch.from_numpy(left_hoop_seed_out).to(reference.device, reference.dtype)
    right_hoop_seed = torch.from_numpy(right_hoop_seed_out).to(reference.device, reference.dtype)
    combined = torch.clamp(left + right, 0, 1)
    combined_seed = torch.clamp(left_seed + right_seed, 0, 1)
    combined_attachment = torch.clamp(left_attachment + right_attachment, 0, 1)
    combined_candidate = torch.clamp(left_candidate + right_candidate, 0, 1)
    combined_hoop_seed = torch.clamp(left_hoop_seed + right_hoop_seed, 0, 1)
    left_present = (left.flatten(1).sum(dim=1, keepdim=True) >= 1.0).to(reference.dtype).view(batch, 1, 1, 1)
    right_present = (right.flatten(1).sum(dim=1, keepdim=True) >= 1.0).to(reference.dtype).view(batch, 1, 1, 1)
    combined_present = torch.clamp(left_present + right_present, 0, 1)
    return {
        "visual_recall_instance_mask": combined,
        "left_visual_recall_instance_mask": left,
        "right_visual_recall_instance_mask": right,
        "visual_recall_seed_mask": combined_seed,
        "left_visual_recall_seed_mask": left_seed,
        "right_visual_recall_seed_mask": right_seed,
        "visual_recall_attachment_mask": combined_attachment,
        "left_visual_recall_attachment_mask": left_attachment,
        "right_visual_recall_attachment_mask": right_attachment,
        "visual_recall_candidate_mask": combined_candidate,
        "left_visual_recall_candidate_mask": left_candidate,
        "right_visual_recall_candidate_mask": right_candidate,
        "visual_recall_hoop_seed_mask": combined_hoop_seed,
        "left_visual_recall_hoop_seed_mask": left_hoop_seed,
        "right_visual_recall_hoop_seed_mask": right_hoop_seed,
        "visual_recall_present": combined_present,
        "left_visual_recall_present": left_present,
        "right_visual_recall_present": right_present,
    }


def build_source_earring_instance_masks_v5(
    source_01: torch.Tensor,
    source_parsing: torch.Tensor | None,
    *,
    source_hair_mask: torch.Tensor | None = None,
    source_seed_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Locate actual source earring pixels for the final V5 RGB composite.

    This is a narrow port of the old PP earring locator that recovered normal
    solid earrings reliably.  Unlike the former V5 high-resolution path it
    does not fit an ellipse, run GrabCut over a crop, or treat the surrounding
    source background as an object.  Raw label 9 remains a seed, while
    parser-missed earrings are admitted only as compact, high-contrast objects
    in the source ear/lobe corridor.
    """

    source_01 = normalized_to_01(source_01)
    reference = source_01[:, :1]
    size = tuple(source_01.shape[-2:])
    zeros = torch.zeros_like(reference)
    if source_parsing is None:
        return {
            "instance_mask": zeros,
            "left_instance_mask": zeros,
            "right_instance_mask": zeros,
            "left_parser_instance_mask": zeros,
            "right_parser_instance_mask": zeros,
            "hoop_hole_mask": zeros,
            "left_hoop_hole_mask": zeros,
            "right_hoop_hole_mask": zeros,
            "locator_roi": zeros,
            "locator_seed": zeros,
            "locator_presence_seed": zeros,
            "locator_support": zeros,
            "locator_ring_support": zeros,
            "locator_parser_mask": zeros,
            "left_context": zeros,
            "right_context": zeros,
            "left_lobe_anchor": zeros,
            "right_lobe_anchor": zeros,
            "visual_recall_instance_mask": zeros,
            "left_visual_recall_instance_mask": zeros,
            "right_visual_recall_instance_mask": zeros,
            "visual_recall_seed_mask": zeros,
            "left_visual_recall_seed_mask": zeros,
            "right_visual_recall_seed_mask": zeros,
            "visual_recall_attachment_mask": zeros,
            "left_visual_recall_attachment_mask": zeros,
            "right_visual_recall_attachment_mask": zeros,
            "visual_recall_candidate_mask": zeros,
            "left_visual_recall_candidate_mask": zeros,
            "right_visual_recall_candidate_mask": zeros,
            "visual_recall_present": zeros[:, :, :1, :1],
            "left_visual_recall_present": zeros[:, :, :1, :1],
            "right_visual_recall_present": zeros[:, :, :1, :1],
        }

    parser_earring = _instance_parsing_mask(source_parsing, (RAW_EARRING,), size, reference)
    left_ear = _instance_parsing_mask(source_parsing, (RAW_LEFT_EAR,), size, reference)
    right_ear = _instance_parsing_mask(source_parsing, (RAW_RIGHT_EAR,), size, reference)
    source_ear = torch.clamp(left_ear + right_ear, 0, 1)
    face_surface = _instance_parsing_mask(source_parsing, RAW_FACE_SURFACE_LABELS, size, reference)
    source_background = _instance_parsing_mask(source_parsing, (0,), size, reference)
    source_hair = (
        _instance_parsing_mask(source_parsing, (RAW_HAIR,), size, reference)
        if source_hair_mask is None
        else (F.interpolate(
            ensure_mask_4d(source_hair_mask).float().to(device=source_01.device),
            size=size,
            mode="nearest",
        ) > 0.5).to(dtype=source_01.dtype)
    )
    # ``source_seed_mask`` is a low-resolution *presence/locator* hint.  It
    # must never become an object pixel by itself: an upsampled 256px seed is
    # exactly the blocky source-background/ear patch that caused black debris,
    # false hoops and duplicated lobes in the final composite.
    supplied_seed = zeros if source_seed_mask is None else (
        F.interpolate(
            ensure_mask_4d(source_seed_mask).float().to(device=source_01.device),
            size=size,
            mode="nearest",
        ) > 0.5
    ).to(dtype=source_01.dtype)

    # The search area follows real source ears into the lobe region.  It is a
    # detector corridor only: it never becomes a final write mask by itself.
    scale = max(size) / 256.0

    def scaled(value: float, minimum: int = 1) -> int:
        rounded = max(int(minimum), int(round(float(value) * scale)))
        return rounded if rounded % 2 == 1 else rounded + 1

    # Establish source-side geometry *before* any parser label or learned seed
    # can influence the locator.  A coarse mask is useful only after it has
    # been linked to an actual source ear/lobe corridor; it must not enlarge a
    # free-standing lower-face/background search area.
    left_context = torch.clamp(
        dilate_mask(left_ear, scaled(27, 3))
        + shift_mask(dilate_mask(left_ear, scaled(23, 3)), down=scaled(36, 1)),
        0,
        1,
    )
    right_context = torch.clamp(
        dilate_mask(right_ear, scaled(27, 3))
        + shift_mask(dilate_mask(right_ear, scaled(23, 3)), down=scaled(36, 1)),
        0,
        1,
    )
    left_lobe_anchor = build_earlobe_anchor(
        left_ear,
        ear_roi=left_context,
        lower_ratio=0.62,
        dilate=scaled(3, 1),
    )
    right_lobe_anchor = build_earlobe_anchor(
        right_ear,
        ear_roi=right_context,
        lower_ratio=0.62,
        dilate=scaled(3, 1),
    )
    # A genuine hanging object can start at the lower lobe and extend below
    # the raw ear label.  This is still source-ear-derived geometry rather than
    # an image-midline fallback.
    left_context = torch.clamp(
        left_context
        + shift_mask(dilate_mask(left_lobe_anchor, scaled(19, 3)), down=scaled(52, 1)),
        0,
        1,
    )
    right_context = torch.clamp(
        right_context
        + shift_mask(dilate_mask(right_lobe_anchor, scaled(19, 3)), down=scaled(52, 1)),
        0,
        1,
    )
    # The two expanded corridors can overlap near a frontal face.  Make that
    # overlap exclusive from the real source-lobe locations so a seed on one
    # side cannot activate both ears downstream.  Equal-distance pixels remain
    # unassigned rather than falling back to an image-centre convention.
    y_coords = torch.arange(size[0], device=reference.device, dtype=reference.dtype).view(1, 1, -1, 1)
    x_coords = torch.arange(size[1], device=reference.device, dtype=reference.dtype).view(1, 1, 1, -1)

    def anchor_centre(anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_area = anchor.sum(dim=(2, 3), keepdim=True)
        present = anchor_area >= 1.0
        centre_y = (anchor * y_coords).sum(dim=(2, 3), keepdim=True) / anchor_area.clamp_min(1.0)
        centre_x = (anchor * x_coords).sum(dim=(2, 3), keepdim=True) / anchor_area.clamp_min(1.0)
        return centre_y, centre_x, present

    left_anchor_y, left_anchor_x, left_anchor_present = anchor_centre(left_lobe_anchor)
    right_anchor_y, right_anchor_x, right_anchor_present = anchor_centre(right_lobe_anchor)
    both_anchors_present = left_anchor_present & right_anchor_present
    overlap = (left_context * right_context).clamp(0, 1)
    left_distance = (y_coords - left_anchor_y).pow(2) + (x_coords - left_anchor_x).pow(2)
    right_distance = (y_coords - right_anchor_y).pow(2) + (x_coords - right_anchor_x).pow(2)
    left_overlap = overlap * (left_distance < right_distance).to(dtype=reference.dtype)
    right_overlap = overlap * (right_distance < left_distance).to(dtype=reference.dtype)
    left_separated = torch.clamp(left_context * (1.0 - overlap) + left_overlap, 0, 1)
    right_separated = torch.clamp(right_context * (1.0 - overlap) + right_overlap, 0, 1)
    left_context = torch.where(both_anchors_present, left_separated, left_context)
    right_context = torch.where(both_anchors_present, right_separated, right_context)

    # Keep the association corridor compact, but give the already strict
    # native ordinary-instance verifier a separate, narrow downward rail.
    # A verified long pendant can extend much farther than the 52px lobe
    # search offset above.  Reusing that short association box as its final
    # crop was cutting valid tails even though the component had already been
    # linked to this exact source lobe.  This rail is passed only to the
    # visual-recall helper; parser/seed association and hoop authorization
    # continue to use the compact contexts above.
    left_recall_context = torch.clamp(
        left_context
        + shift_mask(dilate_mask(left_lobe_anchor, scaled(13, 3)), down=scaled(104, 1))
        + shift_mask(dilate_mask(left_lobe_anchor, scaled(11, 3)), down=scaled(140, 1)),
        0,
        1,
    )
    right_recall_context = torch.clamp(
        right_context
        + shift_mask(dilate_mask(right_lobe_anchor, scaled(13, 3)), down=scaled(104, 1))
        + shift_mask(dilate_mask(right_lobe_anchor, scaled(11, 3)), down=scaled(140, 1)),
        0,
        1,
    )
    recall_overlap = (left_recall_context * right_recall_context).clamp(0, 1)
    left_recall_context = torch.where(
        both_anchors_present,
        torch.clamp(
            left_recall_context * (1.0 - recall_overlap)
            + recall_overlap * (left_distance < right_distance).to(dtype=reference.dtype),
            0,
            1,
        ),
        left_recall_context,
    )
    right_recall_context = torch.where(
        both_anchors_present,
        torch.clamp(
            right_recall_context * (1.0 - recall_overlap)
            + recall_overlap * (right_distance < left_distance).to(dtype=reference.dtype),
            0,
            1,
        ),
        right_recall_context,
    )
    source_side_context = torch.clamp(left_context + right_context, 0, 1)
    association_support = dilate_mask(source_side_context, scaled(7, 1))

    # Parser label 9 remains the normal-earring authority, but a parser blob
    # on an unrelated cheek/background component cannot be re-pasted merely by
    # living on the conventional image half.  The same source-side rule applies
    # to a learned low-resolution seed.
    # This limit is applied to an ear-local association proxy, not to the
    # final raw parser component.  Large dangling accessories can occupy more
    # than 3.5% of a portrait while still being a single label-9 component
    # attached to one lobe.  Keep a larger budget only for this semantic,
    # side-associated path; visual-only candidates retain their much smaller
    # object budget below and cannot use this relaxation to copy background.
    # Large/long source earrings can legitimately exceed the old 5.5% cap;
    # the one-root side association and source semantic filters below keep
    # this larger allowance from admitting an entire background patch.
    association_maximum_area = max(1, int(round(0.080 * size[0] * size[1])))
    # A 3px association dilation increases the temporary proxy area even when
    # the original label-9 component is within the final object budget.  Give
    # that proxy margin, then enforce ``association_maximum_area`` again on the
    # complete raw component below; this preserves long verified tails without
    # allowing a larger RGB write object.
    association_proxy_maximum_area = max(
        association_maximum_area,
        int(round(0.075 * size[0] * size[1])),
    )
    parser_association_proxy = _instance_filter_components(
        dilate_mask(parser_earring, scaled(3, 1)) * association_support,
        association_support,
        minimum_area=max(1, int(round(0.50 * scale * scale))),
        maximum_area=association_proxy_maximum_area,
        # One physical accessory is expected per ear.  Keeping several parser
        # roots lets nearby source strands enter the same side as a second
        # earring; a complete long pendant remains one connected raw object.
        keep_per_side=1,
        left_context=left_context,
        right_context=right_context,
        left_lobe_anchor=left_lobe_anchor,
        right_lobe_anchor=right_lobe_anchor,
        require_side_context=True,
    )
    parser_earring = _retain_raw_components_touching_proxy(
        parser_earring,
        dilate_mask(parser_association_proxy, scaled(3, 1)),
        minimum_area=max(1, int(round(0.50 * scale * scale))),
        maximum_area=association_maximum_area,
        left_context=left_context,
        right_context=right_context,
        left_lobe_anchor=left_lobe_anchor,
        right_lobe_anchor=right_lobe_anchor,
        require_side_context=True,
    )
    seed_association_proxy = _instance_filter_components(
        dilate_mask(supplied_seed, scaled(3, 1)) * association_support,
        association_support,
        minimum_area=max(1, int(round(0.50 * scale * scale))),
        maximum_area=association_proxy_maximum_area,
        keep_per_side=1,
        left_context=left_context,
        right_context=right_context,
        left_lobe_anchor=left_lobe_anchor,
        right_lobe_anchor=right_lobe_anchor,
        require_side_context=True,
    )
    associated_seed = supplied_seed * dilate_mask(seed_association_proxy, scaled(3, 1))
    associated_seed = associated_seed * association_support

    ear_base = torch.clamp(source_ear + parser_earring, 0, 1)
    lobe_corridor = torch.clamp(
        source_side_context
        + dilate_mask(ear_base, scaled(19, 3))
        + dilate_mask(shift_mask(source_ear, down=scaled(12, 1)), scaled(17, 3))
        + 0.80 * dilate_mask(shift_mask(source_ear, down=scaled(30, 1)), scaled(15, 3))
        + 0.55 * dilate_mask(shift_mask(source_ear, down=scaled(52, 1)), scaled(13, 3))
        + dilate_mask(parser_earring, scaled(9, 3)),
        0,
        1,
    )
    # Never scan the facial centre for an accessory.  Parser label 9 is
    # exempt so a large hanging earring is not clipped by a weak ear label.
    face_inner = erode_mask(face_surface, scaled(9, 3))
    locator_roi = lobe_corridor * (1.0 - 0.82 * face_inner).clamp(0, 1)
    locator_roi = torch.clamp(locator_roi + parser_earring, 0, 1)

    gray = rgb_to_gray(source_01)
    local_small = F.avg_pool2d(gray, kernel_size=scaled(7, 3), stride=1, padding=scaled(7, 3) // 2)
    local_large = F.avg_pool2d(gray, kernel_size=scaled(21, 3), stride=1, padding=scaled(21, 3) // 2)
    high = (gray - local_small).abs()
    contrast = (gray - local_large).abs()
    chroma = source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)
    edge = sobel_magnitude(source_01).clamp(max=1.0)
    local_colour = low_pass_filter(source_01, kernel_size=scaled(21, 3), sigma=max(1.0, 6.0 * scale))
    colour_delta = (source_01 - local_colour).abs().mean(dim=1, keepdim=True)

    # Background-coloured pixels are exactly the source content that created
    # the old "hole" halo.  Measure objectness relative to local background,
    # skin and hair rather than using edge magnitude alone.
    def masked_mean(mask: torch.Tensor, minimum_area: float = 24.0) -> tuple[torch.Tensor, torch.Tensor]:
        area = mask.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
        valid = (area >= float(minimum_area)).to(dtype=source_01.dtype)
        mean = (source_01 * mask).sum(dim=(2, 3), keepdim=True) / area.clamp_min(1.0)
        return mean, valid

    skin_mean, skin_valid = masked_mean(erode_mask(face_surface, scaled(9, 3)))
    ear_mean, ear_valid = masked_mean(erode_mask(source_ear, scaled(7, 3)))
    hair_mean, hair_valid = masked_mean(erode_mask(source_hair, scaled(7, 3)))
    bg_mean, bg_valid = masked_mean(source_background * dilate_mask(locator_roi, scaled(41, 3)))
    dist_skin = (source_01 - skin_mean).pow(2).mean(dim=1, keepdim=True).sqrt()
    dist_ear = (source_01 - ear_mean).pow(2).mean(dim=1, keepdim=True).sqrt()
    dist_hair = (source_01 - hair_mean).pow(2).mean(dim=1, keepdim=True).sqrt()
    dist_bg = (source_01 - bg_mean).pow(2).mean(dim=1, keepdim=True).sqrt()
    dist_hair = torch.where(hair_valid > 0, dist_hair, torch.ones_like(dist_hair))
    dist_bg = torch.where(bg_valid > 0, dist_bg, torch.ones_like(dist_bg))
    dist_ear = torch.where(ear_valid > 0, dist_ear, torch.ones_like(dist_ear))
    colour_objectness = torch.clamp(
        0.95 * dist_skin + 0.55 * torch.minimum(dist_skin, torch.minimum(dist_hair, dist_bg)) + 0.45 * chroma,
        0,
        1,
    )
    texture_objectness = torch.clamp(high + contrast + 0.55 * edge + 0.45 * colour_delta, 0, 1)
    evidence = torch.clamp(
        1.15 * texture_objectness + 1.05 * colour_objectness + 0.45 * chroma,
        0,
        1,
    ) * locator_roi

    raw_visual_seed = _instance_adaptive_threshold(
        evidence,
        locator_roi,
        std_scale=0.20,
        minimum_delta=0.010,
        minimum_area=8.0 * scale * scale,
    )
    raw_visual_support = _instance_adaptive_threshold(
        0.75 * evidence + 0.25 * texture_objectness,
        locator_roi,
        std_scale=0.04,
        minimum_delta=0.003,
        minimum_area=8.0 * scale * scale,
    )
    # A generic visual object may not originate from source background or
    # source hair.  The earlier policy admitted textured background as long
    # as it looked unlike its local neighbours, which is exactly why grass was
    # pasted around the ear.  Hoops use a separate geometry-verified support
    # path below because their wire can be parser-labelled background.
    #
    # Ear skin is not excluded wholesale: a small stud sits on it.  It must,
    # however, be visually distinct from the local ear colour/texture; this
    # removes the duplicated round earlobe while retaining metal and gems.
    ear_detail = _instance_adaptive_threshold(
        dist_ear + 0.65 * chroma + 0.45 * colour_delta + 0.25 * edge,
        source_ear * locator_roi,
        std_scale=0.10,
        minimum_delta=0.004,
        minimum_area=8.0 * scale * scale,
    )
    visual_source_gate = (
        (1.0 - source_hair).clamp(0, 1)
        * (1.0 - source_background).clamp(0, 1)
        * torch.clamp((1.0 - source_ear) + ear_detail, 0, 1)
    )
    # Ring geometry gets a separate, more permissive support map.  It is a
    # verifier only and never creates a synthetic ellipse for RGB output.
    ring_support = raw_visual_support * (1.0 - source_hair).clamp(0, 1)
    ring_support = ring_support * (1.0 - erode_mask(source_ear, scaled(7, 3))).clamp(0, 1)
    visual_seed = raw_visual_seed * visual_source_gate
    visual_support = raw_visual_support * visual_source_gate

    # A real but parser-missed accessory is commonly assigned label 0.  Do not
    # reopen the whole background: only an already source-side-associated
    # low-resolution strong candidate may request a native-resolution check,
    # and the checked pixel must still be materially distinct from the local
    # background.  This produces a narrow *inspection permission* for the
    # ordinary visual-recall helper below, never a direct source-RGB mask.
    background_seed_neighbourhood = (
        dilate_mask(associated_seed, scaled(5, 1)) * locator_roi * source_background
    ).clamp(0, 1)
    # Some genuine metal is labelled background and has no low-resolution
    # strong seed at all.  Permit a native probe only on a narrow rail grown
    # from a measured source lobe.  This is deliberately side-local and much
    # smaller than ``locator_roi``; it gives the strict visual extractor a
    # chance to find an exposed stud/pendant without reopening the ear-side
    # grass or background crop.
    source_lobe_anchors = torch.clamp(left_lobe_anchor + right_lobe_anchor, 0, 1)
    native_lobe_rail = torch.clamp(
        dilate_mask(source_lobe_anchors, scaled(9, 3))
        + shift_mask(dilate_mask(source_lobe_anchors, scaled(7, 3)), down=scaled(15, 1))
        + shift_mask(dilate_mask(source_lobe_anchors, scaled(7, 3)), down=scaled(30, 1))
        + shift_mask(dilate_mask(source_lobe_anchors, scaled(7, 3)), down=scaled(48, 1))
        + shift_mask(dilate_mask(source_lobe_anchors, scaled(7, 3)), down=scaled(68, 1))
        + shift_mask(dilate_mask(source_lobe_anchors, scaled(5, 3)), down=scaled(88, 1))
        + shift_mask(dilate_mask(source_lobe_anchors, scaled(5, 3)), down=scaled(108, 1)),
        0,
        1,
    )
    native_background_probe = (
        source_background
        * native_lobe_rail
        * locator_roi
        * (1.0 - source_hair).clamp(0, 1)
    ).clamp(0, 1)
    background_seed_neighbourhood = torch.clamp(
        background_seed_neighbourhood + native_background_probe,
        0,
        1,
    )
    background_distinct = _instance_adaptive_threshold(
        0.85 * dist_bg + 0.60 * colour_delta + 0.35 * chroma + 0.30 * edge,
        background_seed_neighbourhood,
        std_scale=0.18,
        minimum_delta=0.006,
        minimum_area=3.0 * scale * scale,
    )
    background_recall_gate = (
        background_seed_neighbourhood * background_distinct
    ).clamp(0, 1)
    recall_source_gate = torch.clamp(
        visual_source_gate + background_recall_gate,
        0,
        1,
    )
    recall_visual_seed = raw_visual_seed * recall_source_gate
    recall_visual_support = raw_visual_support * recall_source_gate
    # The generic source-instance path is parser-first.  A coarse strong seed
    # is useful for the dedicated parser-miss visual inspector below, but it
    # must not enter this generic grower: an upsampled seed overlapping an ear
    # fold can otherwise turn a bright/dark lobe texture into source RGB.  This
    # was the direct route behind duplicated earlobes and black ear chunks.
    proposal_seed = parser_earring
    proposal_neighbourhood = dilate_mask(proposal_seed, scaled(9, 3)) * locator_roi
    seeded_visual = visual_seed * proposal_neighbourhood
    seed = torch.clamp(parser_earring + seeded_visual, 0, 1)
    support = torch.clamp(
        visual_support * proposal_neighbourhood
        + dilate_mask(parser_earring + seeded_visual, scaled(3, 1)),
        0,
        1,
    ) * locator_roi

    # Grow only through evidence pixels.  The proxy repairs one-pixel gaps in
    # a wire for connected-component selection; the returned RGB mask remains
    # source-object supported and is never the grown solid region.
    reachable = seed * locator_roi
    for _ in range(2):
        reachable = torch.clamp(reachable + dilate_mask(reachable, scaled(3, 1)) * support, 0, 1)
    # The compact locator is sufficient to find the attachment, but it is not
    # long enough for a pendant body.  Extend only measured visual evidence
    # through the source-lobe recall contexts; ``recall_source_gate`` already
    # applies the source hair/background/material checks.
    # Keep ordinary visual recall in the compact detector corridor.  The
    # expanded lobe rail is reserved for parser-labelled long pendants; using
    # it for generic visual candidates admitted source background strands and
    # duplicated earrings in recent V6 outputs.
    extended_visual = recall_visual_support * recall_source_gate * locator_roi
    object_pixels = torch.clamp(
        reachable * visual_support + parser_earring + extended_visual,
        0,
        1,
    ) * locator_roi
    proxy = dilate_mask(object_pixels, scaled(3, 1)) * locator_roi
    minimum_area = max(2, int(round(3.0 * scale * scale)))
    maximum_area = max(minimum_area, int(round(0.040 * size[0] * size[1])))
    kept_proxy = _instance_filter_components(
        proxy,
        locator_roi,
        minimum_area=minimum_area,
        maximum_area=maximum_area,
        # Final RGB recovery permits one connected ordinary accessory per
        # ear side.  A hoop follows its own complete-contour path below; it
        # must not be reconstructed as several nearby ordinary components.
        keep_per_side=1,
        # This filter is only for native visual recall.  Use the extended
        # lobe rail so a long pendant is not discarded before side assignment;
        # parser/strong-seed association and hoop authority stay compact.
        left_context=left_recall_context,
        right_context=right_recall_context,
        left_lobe_anchor=left_lobe_anchor,
        right_lobe_anchor=right_lobe_anchor,
    )
    left_kept_proxy, right_kept_proxy = assign_components_to_ear_sides(
        kept_proxy,
        left_recall_context,
        right_recall_context,
        left_lobe_anchor,
        right_lobe_anchor,
    )
    left_visual_instance = object_pixels * dilate_mask(left_kept_proxy, scaled(3, 1))
    right_visual_instance = object_pixels * dilate_mask(right_kept_proxy, scaled(3, 1))

    # ``parser_earring`` now contains only complete raw components that first
    # touched an accepted source-ear proxy.  Do not run it through a second
    # compact proxy filter here: that second filter was the remaining route
    # that cut a verified long pendant or full ring back to a lobe-sized dot.
    # The side assignment retains complete components and is mutually exclusive
    # for the two real source ears.
    # Parser label-9 components have already been associated with a real
    # source lobe by ``_retain_raw_components_touching_proxy`` above.  Use the
    # extended recall contexts for side assignment here; the compact context
    # ends near the lobe and silently cuts the lower body of long earrings
    # while leaving a misleading elongated mask rail.
    left_parser_instance, right_parser_instance = assign_components_to_ear_sides(
        parser_earring,
        left_recall_context,
        right_recall_context,
        left_lobe_anchor,
        right_lobe_anchor,
    )
    # Recover a parser-missed *ordinary* accessory only from independent
    # high-resolution structure in the exterior lobe belt.  This fallback is
    # intentionally separate from the contour-hoop verifier and never grants
    # permission to copy a background-labelled annulus or its hole.
    visual_recall = _build_parser_miss_visual_instances_v5(
        source_01,
        recall_visual_seed,
        recall_visual_support,
        parser_earring,
        associated_seed,
        source_background,
        background_recall_gate,
        source_hair,
        source_ear,
        face_surface,
        left_context,
        right_context,
        left_recall_context,
        right_recall_context,
        left_lobe_anchor,
        right_lobe_anchor,
        left_parser_instance,
        right_parser_instance,
    )
    left_visual_recall = visual_recall["left_visual_recall_instance_mask"]
    right_visual_recall = visual_recall["right_visual_recall_instance_mask"]

    # ``locator_roi`` is a detector corridor, not an object boundary.  The
    # parser component has already passed source-side association and the
    # complete raw component is deliberately retained above.  Multiplying the
    # union by this corridor would cut a verified pendant/hoop back to the
    # lobe-sized fragment that caused the original short-arc regression.
    # Visual-recall pixels are native-resolution evidence generated from
    # ``visual_seed``/``visual_support`` and independently linked to the lobe
    # by the helper.  Do not recrop them with ``locator_roi``: that low-res
    # detector corridor ends above a verified long pendant and was the last
    # geometric cause of missing lower tails.
    # ``locator_roi`` is a detector corridor, not an object boundary.  The
    # visual verifier has already produced source-lobe-associated native
    # pixels; applying the compact ROI here cuts long pendants back to a thin
    # rail and leaves a mask that is long but misses the actual body.
    left_visual_instance = left_visual_instance * locator_roi
    right_visual_instance = right_visual_instance * locator_roi
    # Ordinary accessories require an associated raw parser component for RGB
    # authority.  The visual detector is still retained as a closed-loop hoop
    # seed below, but a generic lobe-adjacent edge is not sufficient evidence
    # to paste a solid pendant, stud, ear fold or background texture.
    #
    # This deliberately favours a no-op over a fabricated accessory.  A true
    # parser-missed hollow hoop remains available to the paired-contour
    # verifier through ``*_visual_recall_hoop_seed_mask``.
    left_instance_base = left_parser_instance
    right_instance_base = right_parser_instance
    # The native visual fallback is a verified source object, not merely a
    # search proposal: it has passed lobe attachment, exterior-content,
    # contrast and connected-component checks.  Merge it with the parser
    # fragment so a parser-missed pendant tail is written back.  Keep its
    # topology solid here; hollow rings are authorised separately by the
    # paired-contour verifier below.
    # A parser-confirmed source earring is normally the complete semantic
    # object.  A parser fragment can still miss the lower half of a long
    # pendant, so permit visual recall only when its *single* native component
    # is actually adjacent to that parser fragment.  An unrelated visual
    # highlight/strand is rejected instead of being added as a second earring.
    left_parser_present = (
        left_instance_base.flatten(1).sum(dim=1, keepdim=True) >= 1.0
    ).to(left_instance_base.dtype).view(-1, 1, 1, 1)
    right_parser_present = (
        right_instance_base.flatten(1).sum(dim=1, keepdim=True) >= 1.0
    ).to(right_instance_base.dtype).view(-1, 1, 1, 1)
    left_visual_linked = (
        (left_visual_recall * dilate_mask(left_instance_base, 9))
        .flatten(1)
        .sum(dim=1, keepdim=True)
        >= 1.0
    ).to(left_instance_base.dtype).view(-1, 1, 1, 1)
    right_visual_linked = (
        (right_visual_recall * dilate_mask(right_instance_base, 9))
        .flatten(1)
        .sum(dim=1, keepdim=True)
        >= 1.0
    ).to(right_instance_base.dtype).view(-1, 1, 1, 1)
    left_visual_allowed = (1.0 - left_parser_present) + left_parser_present * left_visual_linked
    right_visual_allowed = (1.0 - right_parser_present) + right_parser_present * right_visual_linked
    left_instance = torch.clamp(
        left_instance_base + left_visual_recall * left_visual_allowed,
        0,
        1,
    )
    right_instance = torch.clamp(
        right_instance_base + right_visual_recall * right_visual_allowed,
        0,
        1,
    )

    # The hole is inferred only from a locally closed visual object.  It is
    # composed from the target output below, so it can never retain source
    # hair/background even for an unusually large hollow earring.
    # Do not infer a hollow centre from the parser-miss visual fallback.  It is
    # an ordinary-solid path; only the established parser/coarse instance may
    # contribute a low-resolution topology hint, while complete hoops are
    # handled by the paired-contour branch below.
    left_proxy = dilate_mask(left_instance_base, scaled(3, 1))
    right_proxy = dilate_mask(right_instance_base, scaled(3, 1))
    left_hole = compute_earring_hole_mask(left_proxy) * locator_roi
    right_hole = compute_earring_hole_mask(right_proxy) * locator_roi
    return {
        "instance_mask": torch.clamp(left_instance + right_instance, 0, 1),
        "left_instance_mask": left_instance.clamp(0, 1),
        "right_instance_mask": right_instance.clamp(0, 1),
        "left_parser_instance_mask": left_parser_instance.clamp(0, 1),
        "right_parser_instance_mask": right_parser_instance.clamp(0, 1),
        # Export the exact recall alpha that was merged into the instance, not
        # merely the helper's pre-merge diagnostic view.
        "visual_recall_instance_mask": torch.clamp(left_visual_recall + right_visual_recall, 0, 1),
        "left_visual_recall_instance_mask": left_visual_recall.clamp(0, 1),
        "right_visual_recall_instance_mask": right_visual_recall.clamp(0, 1),
        "visual_recall_seed_mask": visual_recall["visual_recall_seed_mask"].clamp(0, 1),
        "left_visual_recall_seed_mask": visual_recall["left_visual_recall_seed_mask"].clamp(0, 1),
        "right_visual_recall_seed_mask": visual_recall["right_visual_recall_seed_mask"].clamp(0, 1),
        "visual_recall_attachment_mask": visual_recall["visual_recall_attachment_mask"].clamp(0, 1),
        "left_visual_recall_attachment_mask": visual_recall["left_visual_recall_attachment_mask"].clamp(0, 1),
        "right_visual_recall_attachment_mask": visual_recall["right_visual_recall_attachment_mask"].clamp(0, 1),
        "visual_recall_candidate_mask": visual_recall["visual_recall_candidate_mask"].clamp(0, 1),
        "left_visual_recall_candidate_mask": visual_recall["left_visual_recall_candidate_mask"].clamp(0, 1),
        "right_visual_recall_candidate_mask": visual_recall["right_visual_recall_candidate_mask"].clamp(0, 1),
        # Only a native closed-loop seed may reach the paired-contour hoop
        # verifier.  Ordinary visual recall remains an ordinary-instance path
        # and can never authorize a geometric ring by itself.
        "visual_recall_hoop_seed_mask": visual_recall["visual_recall_hoop_seed_mask"].clamp(0, 1),
        "left_visual_recall_hoop_seed_mask": visual_recall["left_visual_recall_hoop_seed_mask"].clamp(0, 1),
        "right_visual_recall_hoop_seed_mask": visual_recall["right_visual_recall_hoop_seed_mask"].clamp(0, 1),
        "visual_recall_present": visual_recall["visual_recall_present"],
        "left_visual_recall_present": visual_recall["left_visual_recall_present"],
        "right_visual_recall_present": visual_recall["right_visual_recall_present"],
        "hoop_hole_mask": torch.clamp(left_hole + right_hole, 0, 1),
        "left_hoop_hole_mask": left_hole.clamp(0, 1),
        "right_hoop_hole_mask": right_hole.clamp(0, 1),
        "locator_roi": locator_roi.clamp(0, 1),
        "locator_seed": seed.clamp(0, 1),
        # Downstream high-resolution hoop validation may consume this as
        # presence evidence.  Export the source-side-associated seed, never
        # the raw low-resolution proposal, otherwise a broad noisy seed can
        # bypass the component checks above and authorize a false ring.
        "locator_presence_seed": associated_seed.clamp(0, 1),
        "locator_support": support.clamp(0, 1),
        "locator_ring_support": ring_support.clamp(0, 1),
        # Native parser-miss diagnostics.  These are inspection masks only and
        # are never consumed as direct compositor alpha.
        "locator_background_recall_permission": background_recall_gate.clamp(0, 1),
        "locator_recall_seed": recall_visual_seed.clamp(0, 1),
        "locator_recall_support": recall_visual_support.clamp(0, 1),
        "locator_parser_mask": parser_earring.clamp(0, 1),
        "left_context": left_context.clamp(0, 1),
        "right_context": right_context.clamp(0, 1),
        "left_lobe_anchor": left_lobe_anchor.clamp(0, 1),
        "right_lobe_anchor": right_lobe_anchor.clamp(0, 1),
    }


def extract_source_earring_foreground_v5(
    source_01: torch.Tensor,
    source_parsing: torch.Tensor | None,
    *,
    source_seed_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Extract source earring foreground alpha for the V19 V5 compositor.

    This is deliberately source-only.  Parser labels and visual cues locate
    the object, while the returned alpha is built from source pixels inside a
    side-local foreground segmentation.  No target ROI, hair mask, bbox or
    synthetic hoop geometry can grant write permission here.
    """
    source_01 = normalized_to_01(source_01)
    bsz, _, height, width = source_01.shape
    device, dtype = source_01.device, source_01.dtype
    zeros = torch.zeros(bsz, 1, height, width, device=device, dtype=dtype)
    presence = torch.zeros(bsz, 2, device=device, dtype=dtype)
    confidence = torch.zeros(bsz, 2, device=device, dtype=dtype)
    if source_parsing is None:
        return {
            "source_alpha": zeros,
            "left_source_alpha": zeros.clone(),
            "right_source_alpha": zeros.clone(),
            "source_rgb": source_01,
            "presence_state": presence,
            "instance_confidence": confidence,
            "localization_roi": zeros.clone(),
            "foreground_seed": zeros.clone(),
            "raw_foreground": zeros.clone(),
            "boundary_band": zeros.clone(),
            "hole_mask": zeros.clone(),
        }

    parsing = ensure_mask_4d(source_parsing).long().to(device=device)
    if parsing.shape[-2:] != (height, width):
        parsing = F.interpolate(parsing.float(), size=(height, width), mode="nearest").long()
    parser_earring = (parsing == RAW_EARRING).float()
    left_ear = (parsing == RAW_LEFT_EAR).float()
    right_ear = (parsing == RAW_RIGHT_EAR).float()
    hair = (
        parsing == RAW_HAIR
        if source_hair_mask is None
        else ensure_mask_4d(source_hair_mask).to(device=device) > 0.5
    ).float()
    # The 256px candidate mask is a localisation prior, never a native object
    # contour.  Upsampling it and adding it directly to the seed made its
    # square lobe/near-ear footprint eligible for final source RGB paste.
    supplied = zeros if source_seed_mask is None else resize_mask(source_seed_mask, (height, width))
    supplied = (supplied > 0.25).to(dtype)
    semantic_subject = (
        parsing_label_mask(parsing, RAW_SKIN_SURFACE_LABELS)
        + parsing_label_mask(parsing, RAW_DETAIL_LABELS)
        + (parsing == RAW_HAIR).float()
    ).clamp(0, 1)

    scale = max(height, width) / 256.0
    def k(value: float, minimum: int = 1) -> int:
        n = max(minimum, int(round(value * scale)))
        return n if n % 2 else n + 1

    # High-resolution visual evidence.  It is intentionally only probable
    # foreground; component association below is the authority boundary.
    gray = rgb_to_gray(source_01)
    local = low_pass_filter(gray, kernel_size=k(9, 3), sigma=max(1.0, 2.5 * scale))
    large = low_pass_filter(gray, kernel_size=k(31, 5), sigma=max(2.0, 8.0 * scale))
    local_rgb = low_pass_filter(source_01, kernel_size=k(9, 3), sigma=max(1.0, 2.5 * scale))
    colour_residual = (source_01 - local_rgb).abs().mean(dim=1, keepdim=True)
    edge = sobel_magnitude(source_01).clamp(0, 1)
    chroma = source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)
    evidence = (0.8 * (gray - local).abs() + 0.7 * (gray - large).abs() + 0.8 * edge + 0.4 * chroma).clamp(0, 1)

    y_grid = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    x_grid = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    left_out, right_out = [], []
    roi_out, seed_out, raw_out, band_out, hole_out = [], [], [], [], []

    def _side_context(ear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = build_earlobe_anchor(ear, ear_roi=dilate_mask(ear, k(35, 5)), lower_ratio=0.60, dilate=k(3, 1))
        rail = torch.zeros_like(ear)
        # A continuous rail follows a pendant from the lobe instead of one
        # displaced blob.  This keeps the localization ROI over a long drop
        # at both 64/256px tests and native 512/1024px inference.
        for offset in (0, 24, 48, 72, 96, 120, 144):
            rail = torch.maximum(
                rail,
                shift_mask(dilate_mask(anchor, k(20, 3)), down=k(offset, 0)),
            )
        context = torch.clamp(
            dilate_mask(ear, k(35, 5))
            + rail,
            0,
            1,
        )
        return context, anchor

    left_context, left_anchor = _side_context(left_ear)
    right_context, right_anchor = _side_context(right_ear)
    left_area = left_anchor.sum(dim=(2, 3), keepdim=True)
    right_area = right_anchor.sum(dim=(2, 3), keepdim=True)
    left_y = (left_anchor * y_grid).sum(dim=(2, 3), keepdim=True) / left_area.clamp_min(1.0)
    left_x = (left_anchor * x_grid).sum(dim=(2, 3), keepdim=True) / left_area.clamp_min(1.0)
    right_y = (right_anchor * y_grid).sum(dim=(2, 3), keepdim=True) / right_area.clamp_min(1.0)
    right_x = (right_anchor * x_grid).sum(dim=(2, 3), keepdim=True) / right_area.clamp_min(1.0)
    left_present = left_area >= 1.0
    right_present = right_area >= 1.0
    # If a parser omits the ear label, source-side assignment still remains
    # deterministic and local rather than using a global face mask.
    left_context = torch.where(left_present, left_context, (x_grid < width * 0.5).to(dtype) * dilate_mask(parser_earring, k(27, 3)))
    right_context = torch.where(right_present, right_context, (x_grid >= width * 0.5).to(dtype) * dilate_mask(parser_earring, k(27, 3)))

    def _run_one(rgb_np, roi_np, seed_np, probable_np, material_np, parser_np, anchor_np, subject_np):
        roi_np = roi_np.astype(np.uint8)
        seed_np = seed_np.astype(bool)
        probable_np = probable_np.astype(bool)
        material_np = material_np.astype(bool)
        parser_np = parser_np.astype(bool)
        anchor_np = anchor_np.astype(bool)
        gc = np.full(roi_np.shape, cv2.GC_BGD if cv2 is not None else 0, np.uint8)
        if cv2 is not None:
            gc[roi_np > 0] = cv2.GC_PR_BGD
            gc[probable_np & (roi_np > 0)] = cv2.GC_PR_FGD
            gc[parser_np | seed_np] = cv2.GC_FGD
            if int((gc == cv2.GC_FGD).sum()) >= 1:
                try:
                    cv2.grabCut((rgb_np * 255).astype(np.uint8), gc, None, None, None, 3, cv2.GC_INIT_WITH_MASK)
                    fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
                except cv2.error:
                    fg = parser_np | seed_np
            else:
                fg = parser_np | seed_np
        else:
            fg = parser_np | seed_np | probable_np
        # GrabCut can keep only the high-contrast rim of a flat pendant.  Its
        # probable-foreground evidence remains eligible for component graph
        # association, so the measured body is not reduced to the root dot.
        fg = (fg | probable_np | parser_np | seed_np) & (roi_np > 0)
        # Component graph association: retain seed components and components
        # within a small appearance-compatible gap.  This replaces the old
        # pixel-by-pixel lobe connectivity requirement.
        if cv2 is not None:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
            keep = np.zeros_like(fg, dtype=bool)
            selected = []
            for idx in range(1, n):
                comp = labels == idx
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area <= 0 or area > int(0.08 * height * width):
                    continue
                if bool((comp & (parser_np | seed_np)).any()):
                    selected.append(idx)
            # Parser-missed small studs are uncertain, not negative.  Let a
            # compact visual component immediately beside the lobe open an
            # inspection path, but still require real pixel evidence rather
            # than treating the lobe/search ROI as foreground.
            if not selected and anchor_np.any():
                uncertain_gap = max(3, int(round(8 * scale)))
                anchor_near = cv2.dilate(
                    anchor_np.astype(np.uint8),
                    np.ones((uncertain_gap * 2 + 1,) * 2, np.uint8),
                    1,
                ).astype(bool)
                nearby = []
                for idx in range(1, n):
                    comp = labels == idx
                    area = int(stats[idx, cv2.CC_STAT_AREA])
                    if 0 < area <= int(0.02 * height * width) and bool((comp & anchor_near).any()):
                        nearby.append(idx)
                selected.extend(nearby[:2])
            # Attach detached pieces by a 1-3px source gap, while requiring
            # visual support and preserving a bounded side-local extent.
            max_gap = max(2, int(round(4 * scale)))
            for idx in range(1, n):
                if idx in selected:
                    continue
                comp = labels == idx
                if not comp.any() or int(stats[idx, cv2.CC_STAT_AREA]) > int(0.08 * height * width):
                    continue
                near = cv2.dilate(comp.astype(np.uint8), np.ones((2 * max_gap + 1, 2 * max_gap + 1), np.uint8), 1).astype(bool)
                if any(bool((near & (labels == base)).any()) for base in selected):
                    selected.append(idx)
            for idx in selected:
                keep |= labels == idx
            fg = keep | (parser_np | seed_np)
        # GrabCut may colour a broad connected source background as probable
        # foreground.  Final alpha remains constrained to direct source
        # material/edge evidence (plus explicit parser/seed pixels), never to
        # the ROI or a filled connected component alone.
        # A semantic face/ear pixel is not earring material merely because it
        # resembles a small low-resolution candidate.  It needs a native
        # parser label or an actual native visual seed.  Background-labelled
        # pendant pixels remain eligible through ``material_np``.
        material_allowed = (~subject_np) | parser_np | seed_np
        return fg & ((material_np & material_allowed) | parser_np | seed_np)

    for batch_idx in range(bsz):
        side_results = []
        for side_idx, (context, anchor, cx, parser_side) in enumerate(
            ((left_context, left_anchor, left_x, parser_earring * left_context),
             (right_context, right_anchor, right_x, parser_earring * right_context))
        ):
            roi = torch.clamp(context[batch_idx:batch_idx + 1] + parser_side[batch_idx:batch_idx + 1], 0, 1)
            roi_evidence = evidence[batch_idx:batch_idx + 1] * roi
            mean = roi_evidence.flatten(1).sum(dim=1, keepdim=True) / roi.flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
            var = ((roi_evidence - mean.view(1, 1, 1, 1)) ** 2 * roi).flatten(1).sum(dim=1, keepdim=True) / roi.flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
            probable = (roi_evidence >= (mean + 0.35 * torch.sqrt(var + 1e-6)).view(1, 1, 1, 1)).float() * roi
            strong_visual = (roi_evidence >= (mean + 1.0 * torch.sqrt(var + 1e-6)).view(1, 1, 1, 1)).float() * roi
            # A coarse candidate may open inspection of a parser-missed stud,
            # but only pixels with native high-evidence can seed foreground.
            native_seed = (
                supplied[batch_idx:batch_idx + 1]
                * context[batch_idx:batch_idx + 1]
                * strong_visual
            )
            seed = torch.clamp(parser_side[batch_idx:batch_idx + 1] + native_seed, 0, 1)
            seed_area_tensor = seed.flatten(1).sum(dim=1, keepdim=True).view(1, 1, 1, 1)
            seed_colour = (source_01[batch_idx:batch_idx + 1] * seed).sum(dim=(2, 3), keepdim=True) / seed_area_tensor.clamp_min(1.0)
            colour_distance = (source_01[batch_idx:batch_idx + 1] - seed_colour).pow(2).mean(dim=1, keepdim=True).sqrt()
            # Flat metal/stone bodies often have low edge energy internally;
            # local colour agreement with the trusted seed supplies probable
            # foreground without turning the whole ROI into write alpha.
            material = (colour_distance <= (0.12 + 0.35 * torch.sqrt(var + 1e-6)).view(1, 1, 1, 1)).float()
            seedless_material = strong_visual * (colour_residual[batch_idx:batch_idx + 1] >= 0.10).to(dtype)
            material = torch.where(seed_area_tensor > 0, material, seedless_material)
            probable = torch.maximum(probable, material * roi * (seed_area_tensor > 0).to(dtype))
            # Preserve parser/seed evidence even when the object is a one-pixel
            # stud; area affects confidence, never presence hard-off.
            fg_np = _run_one(
                source_01[batch_idx].permute(1, 2, 0).detach().cpu().numpy(),
                roi[0, 0].detach().cpu().numpy(),
                seed[0, 0].detach().cpu().numpy(),
                probable[0, 0].detach().cpu().numpy(),
                material[0, 0].detach().cpu().numpy(),
                parser_side[batch_idx, 0].detach().cpu().numpy(),
                anchor[batch_idx, 0].detach().cpu().numpy(),
                semantic_subject[batch_idx, 0].detach().cpu().numpy() > 0.5,
            )
            alpha = torch.from_numpy(fg_np.astype(np.float32)).to(device=device, dtype=dtype).view(1, 1, height, width)
            alpha = alpha * roi
            seed_area = float(seed.sum().item())
            alpha_area = float(alpha.sum().item())
            if seed_area <= 0 and alpha_area <= 0:
                state = 0.0
            elif seed_area > 0 or alpha_area >= max(1.0, 2.0 * scale):
                state = 2.0
            else:
                state = 1.0
            conf = min(1.0, (0.45 if seed_area > 0 else 0.15) + min(0.4, alpha_area / max(1.0, 0.02 * height * width)))
            presence[batch_idx, side_idx] = state
            confidence[batch_idx, side_idx] = conf
            side_results.append(alpha)
        left_alpha, right_alpha = side_results
        left_out.append(left_alpha)
        right_out.append(right_alpha)
        roi_out.append(torch.clamp(left_context[batch_idx:batch_idx + 1] + right_context[batch_idx:batch_idx + 1], 0, 1))
        seed_out.append(torch.clamp(supplied[batch_idx:batch_idx + 1] + parser_earring[batch_idx:batch_idx + 1], 0, 1))
        raw_out.append(torch.clamp(left_alpha + right_alpha, 0, 1))
        band_out.append((dilate_mask(torch.clamp(left_alpha + right_alpha, 0, 1), 3) - erode_mask(torch.clamp(left_alpha + right_alpha, 0, 1), 3)).clamp(0, 1))
        combined_alpha = torch.clamp(left_alpha + right_alpha, 0, 1)
        # A hole is derived from the segmented source alpha itself.  It is
        # topology metadata, never a synthetic ring footprint or a separate
        # compositor branch.
        hole_out.append(compute_earring_hole_mask(combined_alpha))

    left_alpha = torch.cat(left_out, dim=0)
    right_alpha = torch.cat(right_out, dim=0)
    hole = torch.cat(hole_out, dim=0)
    left_alpha = left_alpha * (1.0 - hole).clamp(0, 1)
    right_alpha = right_alpha * (1.0 - hole).clamp(0, 1)
    alpha = torch.clamp(left_alpha + right_alpha, 0, 1) * (1.0 - hole).clamp(0, 1)
    return {
        "source_alpha": alpha,
        "left_source_alpha": left_alpha,
        "right_source_alpha": right_alpha,
        "source_rgb": source_01,
        "presence_state": presence,
        "instance_confidence": confidence,
        "localization_roi": torch.cat(roi_out, dim=0),
        "foreground_seed": torch.cat(seed_out, dim=0),
        "raw_foreground": torch.cat(raw_out, dim=0),
        "boundary_band": torch.cat(band_out, dim=0),
        "hole_mask": hole,
    }


def refine_earring_instances_highres(
    source_01: torch.Tensor,
    source_parsing: torch.Tensor | None,
    coarse_object_mask: torch.Tensor | None,
    search_mask: torch.Tensor | None,
    left_lobe_anchor: torch.Tensor | None,
    right_lobe_anchor: torch.Tensor | None,
    left_side_active: torch.Tensor | None,
    right_side_active: torch.Tensor | None,
    *,
    source_hair_mask: torch.Tensor | None = None,
    source_ear_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Extract source earring *instances* at the final RGB resolution.

    The PP network works at 256px, where a thin hoop frequently becomes a
    few disconnected pixels.  Fitting a replacement ellipse made that failure
    look like a different, flattened and over-thick earring.  This routine
    instead uses the low-resolution mask only as a GrabCut foreground seed and
    retains the object boundary from the source image itself.  It never writes
    the surrounding crop: source hair is background, components must touch a
    seed near the lobe, and enclosed holes are returned separately.
    """

    source_01 = normalized_to_01(source_01)
    output_size = tuple(source_01.shape[-2:])
    reference = source_01[:, :1]
    zeros = torch.zeros_like(reference)
    if cv2 is None:
        return {
            "instance_mask": zeros,
            "hoop_hole_mask": zeros,
            "left_instance_mask": zeros,
            "right_instance_mask": zeros,
            "left_hoop_hole_mask": zeros,
            "right_hoop_hole_mask": zeros,
            "left_connector_mask": zeros,
            "right_connector_mask": zeros,
            "visual_recall_instance_mask": zeros,
            "left_visual_recall_instance_mask": zeros,
            "right_visual_recall_instance_mask": zeros,
            "left_visual_recall_seed_mask": zeros,
            "right_visual_recall_seed_mask": zeros,
            "left_visual_recall_attachment_mask": zeros,
            "right_visual_recall_attachment_mask": zeros,
        }

    def nearest(value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            return torch.zeros_like(reference)
        value = ensure_mask_4d(value).to(device=source_01.device, dtype=source_01.dtype)
        if value.shape[-2:] != output_size:
            value = F.interpolate(value, size=output_size, mode="nearest")
        return (value[:, :1] > 0.5).to(dtype=source_01.dtype)

    coarse = nearest(coarse_object_mask)
    search = nearest(search_mask)
    left_anchor = nearest(left_lobe_anchor)
    right_anchor = nearest(right_lobe_anchor)
    left_active = nearest(left_side_active)
    right_active = nearest(right_side_active)
    hair = nearest(source_hair_mask)
    ear = nearest(source_ear_mask)
    if source_parsing is None:
        parser_earring = torch.zeros_like(reference)
        source_background = torch.zeros_like(reference)
    else:
        parsing = ensure_mask_4d(source_parsing).to(device=source_01.device)
        if parsing.shape[-2:] != output_size:
            parsing = F.interpolate(parsing.float(), size=output_size, mode="nearest")
        parser_earring = (parsing.long() == RAW_EARRING).to(dtype=source_01.dtype)
        source_background = (parsing.long() == 0).to(dtype=source_01.dtype)

    image_np = np.clip(
        source_01.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)
    coarse_np = coarse.detach().cpu().numpy() > 0.5
    search_np = search.detach().cpu().numpy() > 0.5
    parser_np = parser_earring.detach().cpu().numpy() > 0.5
    background_np = source_background.detach().cpu().numpy() > 0.5
    hair_np = hair.detach().cpu().numpy() > 0.5
    ear_np = ear.detach().cpu().numpy() > 0.5
    left_anchor_np = left_anchor.detach().cpu().numpy() > 0.5
    right_anchor_np = right_anchor.detach().cpu().numpy() > 0.5
    left_active_np = left_active.detach().cpu().numpy() > 0.5
    right_active_np = right_active.detach().cpu().numpy() > 0.5

    batch, _, height, width = reference.shape
    scale = max(height, width) / 256.0

    def odd(value: float, minimum: int = 3) -> int:
        kernel = max(minimum, int(round(value * scale)))
        return kernel if kernel % 2 == 1 else kernel + 1

    def component_mask_connected_to_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
        if not candidate.any() or not seed.any():
            return np.zeros_like(candidate, dtype=bool)
        proxy = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
        )
        count, labels = cv2.connectedComponents(proxy)
        seed_touch = cv2.dilate(
            seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(11), odd(11))),
        ).astype(bool)
        keep = np.zeros_like(candidate, dtype=bool)
        for label in range(1, count):
            component = labels == label
            if np.any(component & seed_touch):
                keep |= component
        return candidate & keep

    def enclosed_holes(instance: np.ndarray, local: np.ndarray) -> np.ndarray:
        if int(instance.sum()) < max(12, int(round(5.0 * scale))):
            return np.zeros_like(instance, dtype=bool)
        closed = cv2.morphologyEx(
            instance.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(5), odd(5))),
        ).astype(bool)
        background = (~closed & local).astype(np.uint8)
        # ``local`` is an ear-local window, not the full image.  Its boundary
        # is therefore the flood-fill border; otherwise every external pixel
        # in a window away from the image edge would be misclassified as a
        # hoop hole.
        local_boundary = local & ~cv2.erode(
            local.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
        ).astype(bool)
        border = background & local_boundary.astype(np.uint8)
        flood = border.copy()
        kernel = np.ones((3, 3), dtype=np.uint8)
        while True:
            grown = cv2.dilate(flood, kernel) & background
            if np.array_equal(grown, flood):
                break
            flood = grown
        hole = background.astype(bool) & ~flood.astype(bool)
        # Reject an accidental large enclosed background patch.  A true hoop
        # centre is local to the traced object and much smaller than the crop.
        count, labels = cv2.connectedComponents(hole.astype(np.uint8))
        accepted = np.zeros_like(hole, dtype=bool)
        object_area = max(1, int(instance.sum()))
        for label in range(1, count):
            component = labels == label
            area = int(component.sum())
            near_wire = cv2.dilate(
                instance.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(9), odd(9))),
            ).astype(bool)
            if max(8, int(round(2.0 * scale))) <= area <= max(64, object_area * 6) and np.any(component & near_wire):
                accepted |= component
        return accepted

    def strict_visual_seed(
        image: np.ndarray,
        local_window: np.ndarray,
        hair_mask: np.ndarray,
        ear_mask: np.ndarray,
        anchor_mask: np.ndarray,
    ) -> np.ndarray:
        """Find a compact, lobe-adjacent seed when label 9 is absent.

        This is deliberately a seed generator rather than a write mask.  It
        accepts only structured source pixels outside known hair/ear interiors;
        a later component and GrabCut pass must still confirm an object.  That
        lets an unlabelled stud or pendant start recovery without turning a
        generic exposed ear into an accessory.
        """

        anchor_points = np.argwhere(anchor_mask)
        if anchor_points.size == 0:
            return np.zeros_like(local_window, dtype=bool)
        hair_interior = cv2.erode(
            hair_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(7), odd(7))),
        ).astype(bool)
        ear_interior = cv2.erode(
            ear_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(7), odd(7))),
        ).astype(bool)
        anchor_guard = cv2.dilate(
            anchor_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(11), odd(11))),
        ).astype(bool)
        valid = (local_window & ~(hair_interior | ear_interior)) | anchor_guard
        if int(valid.sum()) < max(24, int(round(8.0 * scale))):
            return np.zeros_like(local_window, dtype=bool)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        local_values = gray[valid]
        low = max(12, int(np.percentile(local_values, 35)))
        high = max(low + 18, int(np.percentile(local_values, 82)))
        edges = cv2.Canny(gray, low, min(255, high)) > 0
        blurred = cv2.GaussianBlur(image, (odd(9), odd(9)), 0)
        local_delta = np.abs(image.astype(np.int16) - blurred.astype(np.int16)).mean(axis=2)
        chroma = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
        delta_floor = max(5.0, float(np.percentile(local_delta[valid], 78)))
        chroma_floor = max(10.0, float(np.percentile(chroma[valid], 86)))
        edge_band = cv2.dilate(
            edges.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
        ).astype(bool)
        candidate = (
            edge_band
            & ((local_delta >= delta_floor) | (chroma >= chroma_floor))
            & valid
        )
        if not candidate.any():
            return np.zeros_like(local_window, dtype=bool)

        proxy = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
        )
        count, labels = cv2.connectedComponents(proxy)
        allowed = np.zeros_like(candidate, dtype=bool)
        anchor_y, anchor_x = anchor_points.mean(axis=0)
        max_distance = 68.0 * scale
        for label in range(1, count):
            component = labels == label
            component_area = int(component.sum())
            if component_area < max(3, int(round(0.75 * scale))):
                continue
            points = np.argwhere(component)
            distance = np.sqrt(
                (points[:, 0] - anchor_y) ** 2 + (points[:, 1] - anchor_x) ** 2
            ).min()
            if distance <= max_distance:
                allowed |= component
        return candidate & allowed

    def trace_side(
        active: np.ndarray,
        anchor: np.ndarray,
        side_seed: np.ndarray,
        observed_seed: np.ndarray,
        side_search: np.ndarray,
        side_background: np.ndarray,
        side_hair: np.ndarray,
        side_ear: np.ndarray,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty = np.zeros((height, width), dtype=bool)
        if not active.any():
            return empty, empty, empty
        # ``side_seed`` may come from a 256px detector.  It is allowed to
        # initialise GrabCut and connectivity, but it is not a source object
        # pixel.  Only the raw parser label is observed directly; every other
        # recovered pixel has to survive the high-resolution structure test.
        seed = side_seed.copy()
        observed = observed_seed.copy()
        anchor_points = np.argwhere(anchor)
        if anchor_points.size == 0:
            # A coarse component cannot substitute for a real source lobe.
            # Otherwise a textured background seed can establish its own local
            # crop and later be copied back as a false earring.
            return empty, empty, empty
        anchor_y, anchor_x = anchor_points.mean(axis=0)

        y_grid, x_grid = np.ogrid[:height, :width]
        local_window = (
            (y_grid >= anchor_y - 72.0 * scale)
            & (y_grid <= anchor_y + 176.0 * scale)
            & (x_grid >= anchor_x - 120.0 * scale)
            & (x_grid <= anchor_x + 120.0 * scale)
        )
        # When the parser misses an ordinary earring, bootstrap from compact
        # image evidence near the lobe.  This is still a source-instance seed,
        # never permission to copy the lobe neighbourhood itself.
        seed &= local_window
        observed &= local_window
        # A native parser can retain only one or two pixels of a small stud at
        # the lobe.  That is enough to start *inspection*, not enough to write
        # RGB: every returned pixel still has to pass the connected-component,
        # material/edge and area checks below.  Scaling this locator threshold
        # with output resolution discarded valid studs before those checks and
        # also prevented a pendant root from reaching its visible body.
        if int(seed.sum()) < 1:
            return empty, empty, empty

        # ``side_search`` is built at the PP resolution and is intentionally
        # compact.  It is useful to initialise the low-resolution locator, but
        # it cannot be the high-resolution crop boundary: a parser fragment at
        # the lobe then limits a long pendant or a large round ornament to its
        # upper edge before GrabCut/contour validation can inspect the body.
        # The real-source lobe and ``seed`` already proved which side may be
        # inspected.  Keep that bounded ear-local window available here; every
        # eventual write pixel still has to pass the object, contour, hair and
        # area checks below.
        local = local_window
        ys, xs = np.where(local | seed | anchor)
        if ys.size == 0:
            return empty, empty, empty
        margin = odd(9)
        y0, y1 = max(0, ys.min() - margin), min(height, ys.max() + margin + 1)
        x0, x1 = max(0, xs.min() - margin), min(width, xs.max() + margin + 1)

        crop_image = image[y0:y1, x0:x1]
        crop_seed = seed[y0:y1, x0:x1]
        crop_observed = observed[y0:y1, x0:x1]
        crop_anchor = anchor[y0:y1, x0:x1]
        crop_local = local[y0:y1, x0:x1]
        crop_background = side_background[y0:y1, x0:x1]
        crop_hair = side_hair[y0:y1, x0:x1]
        crop_ear = side_ear[y0:y1, x0:x1]
        init = np.full(crop_seed.shape, cv2.GC_BGD, dtype=np.uint8)
        init[crop_local] = cv2.GC_PR_BGD
        near_seed = cv2.dilate(
            crop_seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(13), odd(13))),
        ).astype(bool) & crop_local
        init[near_seed] = cv2.GC_PR_FGD
        init[crop_seed] = cv2.GC_FGD
        # Known source hair is a background constraint, except exactly where
        # parser/visual evidence confirms an earring in front of hair.
        crop_hair_interior = cv2.erode(
            crop_hair.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(7), odd(7))),
        ).astype(bool)
        init[crop_hair_interior & ~near_seed] = cv2.GC_BGD
        ear_interior = cv2.erode(
            crop_ear.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(7), odd(7))),
        ).astype(bool)

        # A compact stud commonly sits entirely inside the semantic earlobe.
        # Treating that whole area as background forces the native extractor to
        # keep only the blocky 256px label-9 fragment.  Do not open the ear
        # generally: this exception requires a source-parser observed fragment
        # in a tight lower-lobe band, then limits native expansion to a small
        # neighbourhood of that measured fragment.  Ear folds without label-9
        # evidence therefore remain hard background as before.
        interior_stud_anchor_band = (
            cv2.dilate(
                crop_anchor.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (odd(23), odd(17)),
                ),
            ).astype(bool)
            & crop_local
        )
        interior_stud_observed = (
            crop_observed & ear_interior & interior_stud_anchor_band
        )
        interior_stud_min_seed = max(2, int(round(0.25 * scale * scale)))
        interior_stud_enabled = (
            int(interior_stud_observed.sum()) >= interior_stud_min_seed
        )
        interior_stud_zone = np.zeros_like(crop_local, dtype=bool)
        if interior_stud_enabled:
            interior_stud_zone = (
                ear_interior
                & interior_stud_anchor_band
                & cv2.dilate(
                    interior_stud_observed.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (odd(19), odd(19)),
                    ),
                ).astype(bool)
                & ~crop_hair_interior
            )
        init[ear_interior & ~near_seed & ~interior_stud_zone] = cv2.GC_BGD
        if interior_stud_enabled:
            init[interior_stud_zone] = cv2.GC_PR_FGD

        def is_compact_interior_stud(body: np.ndarray) -> bool:
            """Accept only a parser-anchored, bounded earlobe stud body."""

            if not interior_stud_enabled:
                return False
            body_area = int(body.sum())
            if body_area < max(8, int(round(2.0 * scale * scale))):
                return False
            inside_area = int((body & interior_stud_zone).sum())
            if inside_area < int(np.ceil(0.70 * body_area)):
                return False
            if int((body & interior_stud_observed).sum()) < interior_stud_min_seed:
                return False
            # A label-9 fragment is a locator, not direct RGB authority.  It
            # must touch a native source boundary before a skin-coloured ear
            # parser mistake can be admitted as an in-ear stud.
            if int((body & interior_stud_zone & visual_support).sum()) < 1:
                return False
            points = np.argwhere(body)
            if points.size == 0:
                return False
            bbox_height = int(points[:, 0].max() - points[:, 0].min() + 1)
            bbox_width = int(points[:, 1].max() - points[:, 1].min() + 1)
            max_stud_area = min(
                max(
                    int(round(320.0 * scale * scale)),
                    int(crop_observed.sum()) * 24,
                ),
                int(round(0.012 * height * width)),
            )
            return (
                body_area <= max_stud_area
                and bbox_width <= max(20, int(round(48.0 * scale)))
                and bbox_height <= max(20, int(round(48.0 * scale)))
            )

        candidate = crop_seed.copy()
        if int(crop_seed.sum()) >= 3 and crop_image.shape[0] >= 3 and crop_image.shape[1] >= 3:
            background_model = np.zeros((1, 65), dtype=np.float64)
            foreground_model = np.zeros((1, 65), dtype=np.float64)
            try:
                cv2.grabCut(
                    crop_image,
                    init,
                    None,
                    background_model,
                    foreground_model,
                    3,
                    cv2.GC_INIT_WITH_MASK,
                )
                candidate = ((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)) & crop_local
            except cv2.error:
                candidate = crop_seed.copy()
        # Keep GrabCut's complete foreground separate from the edge-only
        # candidate below.  Large ornate/solid earrings often have a textured
        # outline but a low-contrast gold or gem interior; reducing every
        # accepted foreground pixel to an edge leaves only a thin crescent.
        # The complete region is admitted later only after its own strict
        # lobe, contour and area checks pass.
        grabcut_foreground = candidate.copy()
        # GrabCut's probable foreground includes a local colour region, not
        # necessarily an object.  Admit only real source structure outside the
        # confirmed seed, so a flat source background cannot survive as a
        # rectangular earring patch.
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 24, 96) > 0
        edge_band = cv2.dilate(
            edges.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
        ).astype(bool)
        blurred = cv2.GaussianBlur(crop_image, (odd(9), odd(9)), 0)
        local_delta = np.abs(crop_image.astype(np.int16) - blurred.astype(np.int16)).mean(axis=2)
        delta_floor = max(5.0, float(np.percentile(local_delta[crop_local], 78)))
        visual_support = edge_band & (local_delta >= delta_floor)
        # Coarse pixels are proposals only.  They must contain measured source
        # structure before they can seed connectivity, while parser label-9
        # pixels remain observed source evidence.  This removes the path where
        # an upsampled seed plus high-texture background starts a crop-sized
        # GrabCut foreground and survives to RGB compositing.
        structured_seed = crop_seed & visual_support
        trusted_start = crop_observed | structured_seed
        if int(trusted_start.sum()) < 1:
            return empty, empty, empty
        candidate &= visual_support
        # Outside the ear, parser pixels remain observed object evidence.  In
        # the ear core they only seed the native verifier above; copying an
        # unmeasured label-9 island is how a source earlobe patch became a
        # dark split in the target lobe.
        candidate |= crop_observed & (~ear_interior | visual_support)
        # A parser commonly labels a thin pendant body as background even
        # though the source pixels have a continuous material edge from an
        # observed lobe-side root.  Keep only that structured candidate; the
        # connectivity pass immediately below still rejects detached source
        # background, and the complete-object path keeps flat regions out.
        candidate &= (~crop_background | crop_observed | visual_support)
        candidate &= (~crop_hair_interior | near_seed)
        # Normal ear-core pixels stay forbidden.  The only expansion exception
        # is the compact, parser-anchored earlobe-stud zone above; it still
        # needs native edge support and later compactness checks.
        candidate &= (~ear_interior | crop_observed | interior_stud_zone)
        # Raw parser pixels are observed structure, but only after the caller
        # has limited them to the source-side-associated coarse component.  An
        # isolated parser mistake on an ear fold therefore cannot bypass the
        # verified source-instance gate and become a direct RGB paste.
        candidate = component_mask_connected_to_seed(candidate, trusted_start)

        # A connected GrabCut foreground may supply the low-texture interior
        # of a solid pendant.  It cannot be accepted merely because it is near
        # an ear: require the component to be connected to independently
        # observed/structured source pixels, bounded to a normal accessory
        # area, and supported by a substantial image contour along its own
        # boundary.  This keeps flat source background, ear folds and source
        # hair from becoming a write mask while allowing an ornate disk to be
        # restored as one real object instead of a detached edge.
        complete_candidate = grabcut_foreground.copy()
        complete_candidate &= (~crop_hair_interior | near_seed)
        complete_candidate &= (~ear_interior | crop_observed | interior_stud_zone)
        complete_candidate = component_mask_connected_to_seed(
            complete_candidate,
            trusted_start,
        )
        complete_area = int(complete_candidate.sum())
        max_complete_area = min(
            max(
                int(round(512.0 * scale * scale)),
                int(round(crop_seed.sum() * 10.0)),
            ),
            int(round(0.055 * height * width)),
        )
        boundary = cv2.morphologyEx(
            complete_candidate.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
        ).astype(bool)
        boundary = cv2.dilate(
            boundary.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
        ).astype(bool)
        boundary_area = max(1, int(boundary.sum()))
        boundary_coverage = float((boundary & visual_support).sum()) / float(boundary_area)
        interior_support = float((complete_candidate & visual_support).sum()) / float(max(1, complete_area))
        expanded_ear_interior = complete_candidate & ear_interior & ~crop_observed
        completion_valid = (
            complete_area >= max(8, int(round(2.0 * scale * scale)))
            and complete_area <= max_complete_area
            and boundary_coverage >= 0.18
            and interior_support >= 0.025
            and (
                not expanded_ear_interior.any()
                or is_compact_interior_stud(complete_candidate)
            )
        )

        def closed_solid_body_from_source() -> np.ndarray:
            """Recover a source-supported solid ornament body from its contour.

            GrabCut is intentionally seeded from the small parser instance.  That
            is safe for studs and thin pendants, but can leave a large solid
            medallion as a narrow rim when its centre is low-texture or has a
            different colour.  This helper still requires that exact parser
            instance to touch a measured closed contour; it fills only that
            contour's source-supported interior, never the generic ear-side
            search window.  A nested region that looks like the surrounding
            source content is treated as a real hoop hole rather than a solid
            ornament body.
            """

            contour_input = (edges & crop_local).astype(np.uint8)
            contour_input = cv2.morphologyEx(
                contour_input,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
            )
            contours, hierarchy = cv2.findContours(
                contour_input,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                return np.zeros_like(crop_local, dtype=bool)

            best_body = np.zeros_like(crop_local, dtype=bool)
            best_score = -1.0
            seed_boundary = cv2.dilate(
                trusted_start.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(9), odd(9))),
            ).astype(bool)
            local_colour = crop_image.astype(np.float32)
            for contour_index, contour in enumerate(contours):
                if len(contour) < 12:
                    continue
                contour_fill = np.zeros_like(crop_local, dtype=np.uint8)
                cv2.drawContours(
                    contour_fill,
                    [contour],
                    -1,
                    1,
                    thickness=-1,
                    lineType=cv2.LINE_8,
                )
                contour_fill = contour_fill.astype(bool) & crop_local
                contour_area = int(contour_fill.sum())
                if (
                    contour_area < max(8, int(round(2.0 * scale * scale)))
                    or contour_area > max_complete_area
                ):
                    continue
                contour_boundary = np.zeros_like(crop_local, dtype=np.uint8)
                cv2.drawContours(
                    contour_boundary,
                    [contour],
                    -1,
                    1,
                    thickness=1,
                    lineType=cv2.LINE_8,
                )
                contour_boundary = cv2.dilate(
                    contour_boundary,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
                ).astype(bool)
                boundary_area = max(1, int(contour_boundary.sum()))
                boundary_support = float(
                    (contour_boundary & visual_support).sum()
                ) / float(boundary_area)
                boundary_seed_contact = int((contour_boundary & seed_boundary).sum())
                if boundary_support < 0.22 or boundary_seed_contact < max(2, int(round(scale))):
                    continue

                # A closed inner contour is a likely hoop only when its image
                # colour agrees with the immediately surrounding source area.
                # Decorative inlays remain part of a solid earring because
                # their colour is materially different from the source face or
                # background outside the disc.
                is_likely_hollow = False
                if hierarchy is not None:
                    outer_area = max(1, contour_area)
                    for inner_index, inner_contour in enumerate(contours):
                        if int(hierarchy[0, inner_index, 3]) != contour_index:
                            continue
                        inner_fill = np.zeros_like(crop_local, dtype=np.uint8)
                        cv2.drawContours(
                            inner_fill,
                            [inner_contour],
                            -1,
                            1,
                            thickness=-1,
                            lineType=cv2.LINE_8,
                        )
                        inner_fill = inner_fill.astype(bool) & contour_fill
                        inner_area = int(inner_fill.sum())
                        if not (0.08 * outer_area <= inner_area <= 0.82 * outer_area):
                            continue
                        exterior_band = cv2.dilate(
                            contour_fill.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(11), odd(11))),
                        ).astype(bool) & ~contour_fill & crop_local
                        if int(exterior_band.sum()) < max(12, int(round(3.0 * scale * scale))):
                            continue
                        inner_mean = local_colour[inner_fill].mean(axis=0)
                        exterior_mean = local_colour[exterior_band].mean(axis=0)
                        if float(np.linalg.norm(inner_mean - exterior_mean)) <= 46.0:
                            is_likely_hollow = True
                            break
                if is_likely_hollow:
                    continue

                body = contour_fill.copy()
                body &= (~crop_hair_interior | near_seed)
                body &= (~ear_interior | crop_observed | interior_stud_zone)
                body = component_mask_connected_to_seed(body, trusted_start)
                body_area = int(body.sum())
                if body_area < max(8, int(round(2.0 * scale * scale))):
                    continue
                exterior_ratio = float((body & ~ear_interior).sum()) / float(max(1, body_area))
                if exterior_ratio < 0.60 and not is_compact_interior_stud(body):
                    continue
                score = float(body_area) * boundary_support
                if score > best_score:
                    best_score = score
                    best_body = body
            return best_body

        def seeded_colour_body_from_source() -> np.ndarray:
            """Follow a lobe-linked low-contrast pendant without copying its ROI.

            A long solid pendant can have only one detectable rim at the lobe.
            GrabCut then treats the rest of a low-contrast body as background,
            and the historical edge-only fallback writes back precisely that
            rim.  Starting from the already verified source seed, trace only a
            connected colour region in a bounded vertical pendant corridor.

            This is intentionally not a rectangle fill.  Every returned pixel
            has to be colour-connected to the seed, stay outside source hair
            and the ear core, remain below the same lobe corridor, and exhibit
            either a measured image boundary or a material colour difference
            from its exterior.  A flat source background therefore either
            leaks beyond the strict area cap or fails the boundary test.
            """

            seed_points = np.argwhere(trusted_start)
            anchor_points = np.argwhere(crop_anchor)
            if seed_points.size == 0 or anchor_points.size == 0:
                return np.zeros_like(crop_local, dtype=bool)

            anchor_y, anchor_x = anchor_points.mean(axis=0)
            y_grid, x_grid = np.ogrid[:crop_local.shape[0], :crop_local.shape[1]]
            pendant_corridor = (
                (y_grid >= anchor_y - 28.0 * scale)
                & (y_grid <= anchor_y + 182.0 * scale)
                & (x_grid >= anchor_x - 76.0 * scale)
                & (x_grid <= anchor_x + 76.0 * scale)
            )
            allowed = (
                crop_local
                & pendant_corridor
                & ~crop_hair_interior
                # Permit the same compact parser-anchored in-ear stud zone as
                # the GrabCut path, never a generic ear-core colour flood.
                & (~ear_interior | crop_observed | interior_stud_zone)
            )
            if int(allowed.sum()) < max(16, int(round(4.0 * scale * scale))):
                return np.zeros_like(crop_local, dtype=bool)

            # Work in Lab space so warm metal, dark enamel and coloured gems
            # can be compared by material colour instead of RGB channel noise.
            lab = cv2.cvtColor(crop_image, cv2.COLOR_RGB2LAB).astype(np.float32)
            seed_values = lab[trusted_start]
            if seed_values.size == 0:
                return np.zeros_like(crop_local, dtype=bool)
            seed_centre = np.median(seed_values, axis=0)
            seed_distance = np.linalg.norm(seed_values - seed_centre, axis=1)
            # Parser pixels can contain a one-pixel antialiased edge.  Allow a
            # modest robust spread, but never a permissive colour flood across
            # a generic ear-side background.
            colour_limit = float(
                np.clip(
                    np.percentile(seed_distance, 90) + 7.0,
                    7.0,
                    32.0,
                )
            )
            colour_distance = np.linalg.norm(lab - seed_centre, axis=2)
            colour_candidate = allowed & (colour_distance <= colour_limit)
            colour_candidate |= trusted_start & allowed
            body = component_mask_connected_to_seed(colour_candidate, trusted_start)
            body_area = int(body.sum())
            if (
                body_area < max(8, int(round(2.0 * scale * scale)))
                or body_area > max_complete_area
            ):
                return np.zeros_like(crop_local, dtype=bool)

            exterior_ratio = float((body & ~ear_interior).sum()) / float(max(1, body_area))
            if exterior_ratio < 0.60 and not is_compact_interior_stud(body):
                return np.zeros_like(crop_local, dtype=bool)

            boundary = cv2.morphologyEx(
                body.astype(np.uint8),
                cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
            ).astype(bool)
            boundary = cv2.dilate(
                boundary.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(3), odd(3))),
            ).astype(bool)
            boundary_area = max(1, int(boundary.sum()))
            edge_coverage = float((boundary & visual_support).sum()) / float(boundary_area)
            outside_band = (
                cv2.dilate(
                    body.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(7), odd(7))),
                ).astype(bool)
                & ~body
                & crop_local
                & ~crop_hair_interior
                & (~ear_interior | interior_stud_zone)
            )
            if int(outside_band.sum()) < max(8, int(round(2.0 * scale * scale))):
                return np.zeros_like(crop_local, dtype=bool)
            interior_band = body & boundary
            if int(interior_band.sum()) < 1:
                interior_band = body
            material_delta = float(
                np.linalg.norm(
                    lab[interior_band].mean(axis=0) - lab[outside_band].mean(axis=0)
                )
            )
            # An image contour is the preferred proof.  For genuinely
            # low-contrast opaque pendants, a measurable Lab material edge is
            # sufficient, but a flat region with neither signal remains out.
            if edge_coverage < 0.07 and material_delta < 7.0:
                return np.zeros_like(crop_local, dtype=bool)

            source_evidence = int((body & (visual_support | crop_observed)).sum())
            if source_evidence < max(2, int(round(0.5 * scale * scale))):
                return np.zeros_like(crop_local, dtype=bool)
            return body

        # A parser can trace only the rim of a large solid earring.  Give that
        # measured contour one conservative chance to restore its body before
        # accepting the historical edge-only candidate.  This remains a
        # source-object alpha; neither the ear corridor nor its bounding box is
        # ever copied to the output.
        solid_body = closed_solid_body_from_source()
        colour_body = seeded_colour_body_from_source()
        solid_body_area = int(solid_body.sum())
        colour_body_area = int(colour_body.sum())
        if (
            solid_body_area >= max(8, int(round(2.0 * scale * scale)))
            and solid_body_area <= max_complete_area
        ):
            complete_candidate = solid_body
            completion_valid = True
        if (
            colour_body_area >= max(8, int(round(2.0 * scale * scale)))
            and colour_body_area <= max_complete_area
        ):
            # The contour-fill branch is strongest for a closed medallion;
            # the colour-connected branch is stronger for a long low-contrast
            # drop with only a short upper parser rim.  They may be combined
            # only when they already describe one measured object.
            overlap = int((solid_body & colour_body).sum())
            if solid_body_area > 0 and overlap >= max(1, int(round(0.10 * min(solid_body_area, colour_body_area)))):
                merged_body = solid_body | colour_body
                if int(merged_body.sum()) <= max_complete_area:
                    complete_candidate = merged_body
                else:
                    complete_candidate = colour_body
            elif colour_body_area >= solid_body_area:
                complete_candidate = colour_body
            completion_valid = True
        if completion_valid:
            candidate = complete_candidate

        # GraphCut can occasionally absorb a flat background region when a
        # source parser label is tiny.  Fall back to genuine source edges plus
        # the confirmed seed rather than pasting that region over target hair.
        seed_area = max(1, int(crop_seed.sum()))
        max_area = max(int(round(96.0 * scale * scale)), int(round(seed_area * 6.0)))
        if completion_valid:
            # The complete-object gate above has already verified the contour
            # and capped its area.  Do not run that accepted pendant through
            # the historical edge-only small-object fallback.
            max_area = max(max_area, max_complete_area)
        if int(candidate.sum()) > max_area:
            edge_band = cv2.dilate(
                edges.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(5), odd(5))),
            ).astype(bool)
            candidate = component_mask_connected_to_seed(
                (edge_band & crop_local & (~crop_background | visual_support))
                | crop_observed,
                trusted_start,
            )

        hole = enclosed_holes(candidate, crop_local)
        # Only this narrow, source-observed lobe attachment may enter a target
        # ear interior.  The main instance must stay outside so an ear fold,
        # shadow, or GrabCut lobe cannot be pasted back as a black block.
        attachment_band = cv2.dilate(
            crop_anchor.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(9), odd(3))),
        ).astype(bool)
        # ``connector`` is also the final source-native authority for an
        # accepted in-ear stud.  It remains exact object alpha: no ear ROI,
        # parser dilation, or target-side rectangle is returned here.
        compact_interior_stud = is_compact_interior_stud(candidate)
        # The compact-stud verifier has already checked parser anchoring, native
        # boundary evidence, lobe connectivity, area and bounding box.  Its
        # authority must cover the verified *object*, not merely the overlap
        # between that object and the source parser's ear core.  The latter is
        # in a different frame from the target ear core and was slicing a real
        # stud into a lobe-shaped fragment after alignment.
        interior_stud_authority = (
            candidate
            if compact_interior_stud
            else np.zeros_like(crop_local, dtype=bool)
        )
        connector = (
            candidate
            & (
                (ear_interior & attachment_band)
                | interior_stud_authority
            )
            & ~hole
        )
        instance = np.zeros((height, width), dtype=bool)
        instance[y0:y1, x0:x1] = candidate & ~hole
        holes = np.zeros((height, width), dtype=bool)
        holes[y0:y1, x0:x1] = hole
        connectors = np.zeros((height, width), dtype=bool)
        connectors[y0:y1, x0:x1] = connector
        return instance, holes, connectors

    left_out = np.zeros((batch, 1, height, width), dtype=np.float32)
    right_out = np.zeros_like(left_out)
    left_hole_out = np.zeros_like(left_out)
    right_hole_out = np.zeros_like(left_out)
    left_connector_out = np.zeros_like(left_out)
    right_connector_out = np.zeros_like(left_out)

    def split_by_lobe_side(
        mask: np.ndarray,
        left_anchor_mask: np.ndarray,
        right_anchor_mask: np.ndarray,
        *,
        require_component_association: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split only through real source-lobe corridors, never image halves."""

        empty = np.zeros((height, width), dtype=bool)
        left_points = np.argwhere(left_anchor_mask)
        right_points = np.argwhere(right_anchor_mask)
        if left_points.size == 0 and right_points.size == 0:
            return empty, empty

        y_grid, x_grid = np.ogrid[:height, :width]

        def lobe_window(points: np.ndarray) -> np.ndarray:
            if points.size == 0:
                return np.zeros((height, width), dtype=bool)
            anchor_y, anchor_x = points.mean(axis=0)
            return (
                (y_grid >= anchor_y - 72.0 * scale)
                & (y_grid <= anchor_y + 176.0 * scale)
                & (x_grid >= anchor_x - 120.0 * scale)
                & (x_grid <= anchor_x + 120.0 * scale)
            )

        left_window = lobe_window(left_points)
        right_window = lobe_window(right_points)
        if left_points.size > 0 and right_points.size > 0:
            left_y, left_x = left_points.mean(axis=0)
            right_y, right_x = right_points.mean(axis=0)
            left_region = ((y_grid - left_y) ** 2 + (x_grid - left_x) ** 2) <= (
                (y_grid - right_y) ** 2 + (x_grid - right_x) ** 2
            )
            right_region = ~left_region
        elif left_points.size > 0:
            left_region = np.ones((height, width), dtype=bool)
            right_region = empty
        else:
            left_region = empty
            right_region = np.ones((height, width), dtype=bool)

        if not require_component_association:
            return (
                mask & left_window & left_region,
                mask & right_window & right_region,
            )

        def nearest_distance(component_points: np.ndarray, anchor_points: np.ndarray) -> float:
            if component_points.size == 0 or anchor_points.size == 0:
                return float("inf")
            if component_points.shape[0] > 2048:
                component_points = component_points[
                    np.linspace(0, component_points.shape[0] - 1, 2048).astype(np.int64)
                ]
            if anchor_points.shape[0] > 512:
                anchor_points = anchor_points[
                    np.linspace(0, anchor_points.shape[0] - 1, 512).astype(np.int64)
                ]
            delta = component_points.astype(np.float32)[:, None, :] - anchor_points.astype(np.float32)[None, :, :]
            return float(np.sqrt((delta * delta).sum(axis=2)).min())

        component_count, labels = cv2.connectedComponents(mask.astype(np.uint8))
        left_out = np.zeros_like(mask, dtype=bool)
        right_out = np.zeros_like(mask, dtype=bool)
        for component_id in range(1, component_count):
            component = labels == component_id
            component_points = np.argwhere(component)
            area = max(1, int(component.sum()))

            def side_score(points: np.ndarray, window: np.ndarray) -> tuple[float, bool]:
                if points.size == 0:
                    return 0.0, False
                overlap = int((component & window).sum())
                distance = nearest_distance(component_points, points)
                associated = (
                    overlap >= max(1, int(round(0.005 * area)))
                    or distance <= 72.0 * scale
                )
                if not associated:
                    return 0.0, False
                return (
                    float(overlap) / float(area)
                    + max(0.0, 1.0 - distance / max(72.0 * scale, 1.0)),
                    True,
                )

            left_score, left_associated = side_score(left_points, left_window)
            right_score, right_associated = side_score(right_points, right_window)
            if left_associated and not right_associated:
                left_out |= component
            elif right_associated and not left_associated:
                right_out |= component
            elif left_associated and right_associated:
                if left_score > right_score:
                    left_out |= component
                elif right_score > left_score:
                    right_out |= component
                # An exact tie has no source-side proof and is intentionally
                # discarded instead of being resolved by tensor x-coordinate.
        return left_out, right_out

    def retain_parser_components_touching_coarse(
        raw_parser: np.ndarray,
        accepted_coarse: np.ndarray,
        left_anchor_mask: np.ndarray,
        right_anchor_mask: np.ndarray,
    ) -> np.ndarray:
        """Return source label-9 components tied to an accepted lobe seed.

        ``accepted_coarse`` is the per-sample, ear-associated locator output.
        Its role here is membership proof, not a final spatial crop.  In
        particular, do not dilate it and then intersect that neighbourhood with
        raw label 9: a nearby parser mistake would become observed RGB content
        even though the coarse locator rejected that component.

        A long pendant is often represented by two disconnected parser islands:
        a tiny hook at the earlobe and a lower body.  Requiring every raw parser
        component to overlap the upper island discards the body before the
        native image refiner can inspect it.  The lower island is accepted only
        when it lies on the same real-lobe, vertically bounded pendant rail as
        an already accepted side.  This remains label-9 evidence rather than a
        broad ROI copy, and components in the opposite ear or generic image
        halves are never admitted.
        """

        if not raw_parser.any() or not accepted_coarse.any():
            return np.zeros_like(raw_parser, dtype=bool)
        component_count, labels = cv2.connectedComponents(raw_parser.astype(np.uint8))
        retained = np.zeros_like(raw_parser, dtype=bool)
        anchors = (left_anchor_mask, right_anchor_mask)
        y_grid, x_grid = np.ogrid[:height, :width]

        def side_pendant_window(anchor_mask: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
            points = np.argwhere(anchor_mask)
            if points.size == 0:
                return np.zeros_like(raw_parser, dtype=bool), None
            anchor_y, anchor_x = points.mean(axis=0)
            window = (
                (y_grid >= anchor_y - 56.0 * scale)
                & (y_grid <= anchor_y + 188.0 * scale)
                & (x_grid >= anchor_x - 88.0 * scale)
                & (x_grid <= anchor_x + 88.0 * scale)
            )
            return window, (float(anchor_y), float(anchor_x))

        side_windows = tuple(side_pendant_window(anchor) for anchor in anchors)
        for component_id in range(1, component_count):
            component = labels == component_id
            if np.any(component & accepted_coarse):
                retained |= component
                continue

            # A separate parser island is not trusted merely because it is in
            # the image half of an ear.  It must fit one (and only one)
            # lobe-relative pendant rail and that side must already carry the
            # accepted locator seed.  This lets a thin unlabeled bridge be
            # crossed without allowing an unrelated background fragment to
            # become source RGB authority.
            accepted_sides: list[np.ndarray] = []
            component_area = max(1, int(component.sum()))
            for side_index, (window, anchor_center) in enumerate(side_windows):
                if anchor_center is None:
                    continue
                if not np.any(accepted_coarse & window):
                    continue
                in_window = component & window
                in_window_area = int(in_window.sum())
                if in_window_area < max(1, int(round(0.70 * component_area))):
                    continue
                points = np.argwhere(in_window)
                if points.size == 0:
                    continue
                anchor_y, anchor_x = anchor_center
                centre_y, centre_x = points.mean(axis=0)
                # A pendant hangs under the lobe and remains in its lateral
                # corridor.  Permit a small upward hook, but reject an
                # isolated eyebrow/temple label or a distant second object.
                if (
                    centre_y < anchor_y - 32.0 * scale
                    or abs(centre_x - anchor_x) > 58.0 * scale
                ):
                    continue
                accepted_sides.append(in_window)
            if len(accepted_sides) == 1:
                retained |= accepted_sides[0]
        return retained

    for index in range(batch):
        # The raw parser is only observed after direct component membership in
        # the accepted source-side locator.  Dilation remains available inside
        # connectivity/GrabCut as a temporary guide, never as direct RGB
        # authority for a label-9 neighbourhood.
        associated_parser = retain_parser_components_touching_coarse(
            parser_np[index, 0],
            coarse_np[index, 0],
            left_anchor_np[index, 0],
            right_anchor_np[index, 0],
        )
        left_seed, right_seed = split_by_lobe_side(
            coarse_np[index, 0] | associated_parser,
            left_anchor_np[index, 0], right_anchor_np[index, 0],
            require_component_association=True,
        )
        left_observed, right_observed = split_by_lobe_side(
            associated_parser,
            left_anchor_np[index, 0], right_anchor_np[index, 0],
            require_component_association=True,
        )
        left_search, right_search = split_by_lobe_side(
            search_np[index, 0],
            left_anchor_np[index, 0], right_anchor_np[index, 0],
            require_component_association=False,
        )
        (
            left_out[index, 0],
            left_hole_out[index, 0],
            left_connector_out[index, 0],
        ) = trace_side(
            left_active_np[index, 0],
            left_anchor_np[index, 0],
            left_seed,
            left_observed,
            left_search,
            background_np[index, 0],
            hair_np[index, 0],
            ear_np[index, 0],
            image_np[index],
        )
        (
            right_out[index, 0],
            right_hole_out[index, 0],
            right_connector_out[index, 0],
        ) = trace_side(
            right_active_np[index, 0],
            right_anchor_np[index, 0],
            right_seed,
            right_observed,
            right_search,
            background_np[index, 0],
            hair_np[index, 0],
            ear_np[index, 0],
            image_np[index],
        )

    left_instance = torch.from_numpy(left_out).to(device=source_01.device, dtype=source_01.dtype)
    right_instance = torch.from_numpy(right_out).to(device=source_01.device, dtype=source_01.dtype)
    left_hole = torch.from_numpy(left_hole_out).to(device=source_01.device, dtype=source_01.dtype)
    right_hole = torch.from_numpy(right_hole_out).to(device=source_01.device, dtype=source_01.dtype)
    left_connector = torch.from_numpy(left_connector_out).to(
        device=source_01.device,
        dtype=source_01.dtype,
    )
    right_connector = torch.from_numpy(right_connector_out).to(
        device=source_01.device,
        dtype=source_01.dtype,
    )
    return {
        "instance_mask": torch.clamp(left_instance + right_instance, 0, 1),
        "hoop_hole_mask": torch.clamp(left_hole + right_hole, 0, 1),
        "left_instance_mask": left_instance,
        "right_instance_mask": right_instance,
        "left_hoop_hole_mask": left_hole,
        "right_hoop_hole_mask": right_hole,
        "left_connector_mask": left_connector,
        "right_connector_mask": right_connector,
    }


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
            # its fitted perimeter.  Small hoops occupy too few pixels for
            # four sectors, but still need three independent arcs.
            support_y, support_x = np.where(ellipse_bool & support)
            if support_y.size == 0:
                continue
            sector_angle = np.arctan2(support_y - center_y, support_x - center_x)
            sectors = np.unique(np.floor((sector_angle + np.pi) * (8.0 / (2.0 * np.pi))).astype(np.int32) % 8)
            required_sectors = 3 if minor <= 18.0 * coordinate_scale else 4
            if sectors.size < required_sectors:
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


def build_contour_hoop_instances_v5(
    source_01: torch.Tensor,
    search_mask: torch.Tensor,
    left_lobe_anchor: torch.Tensor,
    right_lobe_anchor: torch.Tensor,
    *,
    left_source_evidence: torch.Tensor | None = None,
    right_source_evidence: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    source_ear_mask: torch.Tensor | None = None,
    min_axis: float = 6.0,
    max_axis: float = 168.0,
    min_coverage: float = 0.42,
) -> dict[str, torch.Tensor]:
    """Build hoop alpha from paired source contours, never from a fitted ellipse.

    The source image must provide two nested boundaries with coverage around
    the ring.  The region between those boundaries is the only source RGB
    write permission.  The enclosed region is exported separately, so the
    final compositor cannot copy source hair or background through the hoop.
    """

    source_01 = normalized_to_01(source_01)
    reference = ensure_mask_4d(search_mask).float()
    size = tuple(reference.shape[-2:])
    source_01 = F.interpolate(source_01, size=size, mode="bilinear", align_corners=False)
    zeros = torch.zeros_like(reference)
    empty = {
        "left_elliptical_hoop": zeros,
        "right_elliptical_hoop": zeros,
        "left_elliptical_hoop_hole": zeros,
        "right_elliptical_hoop_hole": zeros,
        "left_elliptical_hoop_footprint": zeros,
        "right_elliptical_hoop_footprint": zeros,
        "left_elliptical_hoop_connector": zeros,
        "right_elliptical_hoop_connector": zeros,
    }
    if cv2 is None:
        return empty

    search = _resize_like_mask(search_mask, reference)
    left_evidence = _resize_like_mask(left_source_evidence, reference)
    right_evidence = _resize_like_mask(right_source_evidence, reference)
    # A broad search corridor is not source-earring evidence.  Callers that
    # cannot provide a side-specific parser/strong seed therefore get no hoop
    # completion rather than allowing an unrelated background contour through.
    left_anchor = _resize_like_mask(left_lobe_anchor, reference)
    right_anchor = _resize_like_mask(right_lobe_anchor, reference)
    hair = _resize_like_mask(source_hair_mask, reference)
    ear = _resize_like_mask(source_ear_mask, reference)
    source_np = np.clip(
        source_01.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)
    search_np = search.detach().cpu().numpy() > 0.5
    left_evidence_np = left_evidence.detach().cpu().numpy() > 0.5
    right_evidence_np = right_evidence.detach().cpu().numpy() > 0.5
    hair_np = hair.detach().cpu().numpy() > 0.5
    ear_np = ear.detach().cpu().numpy() > 0.5
    left_anchor_np = left_anchor.detach().cpu().numpy() > 0.5
    right_anchor_np = right_anchor.detach().cpu().numpy() > 0.5

    batch, _, height, width = reference.shape
    coordinate_scale = max(float(height), float(width)) / 256.0

    def kernel(base: float, minimum: int = 3) -> int:
        value = max(int(minimum), int(round(base * coordinate_scale)))
        return value if value % 2 == 1 else value + 1

    def filled(contour: np.ndarray) -> np.ndarray:
        result = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(result, [contour], -1, 1, thickness=-1, lineType=cv2.LINE_8)
        return result.astype(bool)

    def outline(contour: np.ndarray) -> np.ndarray:
        result = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(result, [contour], -1, 1, thickness=1, lineType=cv2.LINE_8)
        return result.astype(bool)

    def ellipse_shape(contour: np.ndarray) -> tuple[float, float, float, float, float] | None:
        if len(contour) < 5:
            return None
        (center_x, center_y), (axis_x, axis_y), angle = cv2.fitEllipse(contour)
        major = max(float(axis_x), float(axis_y))
        minor = min(float(axis_x), float(axis_y))
        if major <= 1e-6 or minor <= 1e-6:
            return None
        return float(center_x), float(center_y), major, minor, float(angle)

    def supported_sectors(
        boundary: np.ndarray,
        support: np.ndarray,
        center_x: float,
        center_y: float,
    ) -> tuple[float, int]:
        points = np.argwhere(boundary)
        if points.size == 0:
            return 0.0, 0
        observed = support[points[:, 0], points[:, 1]]
        coverage = float(observed.mean())
        if not observed.any():
            return coverage, 0
        observed_points = points[observed]
        angles = np.arctan2(observed_points[:, 0] - center_y, observed_points[:, 1] - center_x)
        sectors = np.unique(
            np.floor((angles + np.pi) * (8.0 / (2.0 * np.pi))).astype(np.int32) % 8
        )
        return coverage, int(sectors.size)

    def connector_from_source(
        raw_support: np.ndarray,
        valid: np.ndarray,
        anchor_mask: np.ndarray,
        outer_boundary: np.ndarray,
    ) -> np.ndarray:
        anchor_points = np.argwhere(anchor_mask)
        boundary_points = np.argwhere(outer_boundary)
        if anchor_points.size == 0 or boundary_points.size == 0:
            return np.zeros_like(valid, dtype=bool)
        anchor_y, anchor_x = anchor_points.mean(axis=0)
        distances = (
            (boundary_points[:, 0].astype(np.float32) - anchor_y) ** 2
            + (boundary_points[:, 1].astype(np.float32) - anchor_x) ** 2
        )
        nearest_y, nearest_x = boundary_points[int(np.argmin(distances))]
        if float(np.sqrt(float(distances.min()))) > 78.0 * coordinate_scale:
            return np.zeros_like(valid, dtype=bool)

        corridor = np.zeros((height, width), dtype=np.uint8)
        cv2.line(
            corridor,
            (int(round(anchor_x)), int(round(anchor_y))),
            (int(nearest_x), int(nearest_y)),
            1,
            thickness=kernel(5, 3),
            lineType=cv2.LINE_8,
        )
        candidate = raw_support & (corridor > 0) & valid
        if int(candidate.sum()) < max(3, int(round(2.0 * coordinate_scale))):
            return np.zeros_like(valid, dtype=bool)
        proxy = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(3), kernel(3))),
        )
        component_count, labels = cv2.connectedComponents(proxy)
        start = cv2.dilate(
            anchor_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(9), kernel(3))),
        ).astype(bool)
        end = cv2.dilate(
            outer_boundary.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(7), kernel(3))),
        ).astype(bool)
        connected = np.zeros_like(candidate, dtype=bool)
        for label in range(1, component_count):
            component = labels == label
            if np.any(component & start) and np.any(component & end):
                connected |= component
        # The component is a connectivity proxy.  Return only observed source
        # structure so a straight corridor can never draw a synthetic wire.
        return candidate & connected

    def trace_side(
        image: np.ndarray,
        base_search: np.ndarray,
        side_evidence: np.ndarray,
        hair_mask: np.ndarray,
        ear_mask: np.ndarray,
        anchor_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        zero = np.zeros((height, width), dtype=np.float32)
        anchor_points = np.argwhere(anchor_mask)
        if anchor_points.size == 0:
            return zero, zero, zero, zero
        anchor_y, anchor_x = anchor_points.mean(axis=0)
        y_grid, x_grid = np.ogrid[:height, :width]
        local_window = (
            # A hoop can rise well above the lobe before its lower arc reaches
            # the dangling region.  The old -28px upper bound physically cut a
            # legitimate ring into a short lower arc before contour validation.
            (y_grid >= anchor_y - 96.0 * coordinate_scale)
            & (y_grid <= anchor_y + 180.0 * coordinate_scale)
            & (x_grid >= anchor_x - 120.0 * coordinate_scale)
            & (x_grid <= anchor_x + 120.0 * coordinate_scale)
        )
        hair_core = cv2.erode(
            hair_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(7), kernel(3))),
        ).astype(bool)
        ear_core = cv2.erode(
            ear_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(7), kernel(3))),
        ).astype(bool)
        # The saved search mask can be narrow for a parser-missed large hoop.
        # It is a useful preference but not a geometric clip.  The contour
        # verifier below is the authority that makes this wider lobe corridor
        # safe on a source image with no earring.
        # Source evidence authorizes the geometry pass for this side.  It is a
        # narrow parser/strong-instance seed, not the broad ear corridor.  A
        # hoop can cross pixels a parser called hair or ear, so evidence-near
        # pixels are retained for contour detection; unrelated hair/ear edges
        # remain excluded.
        evidence_near = cv2.dilate(
            side_evidence.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(17), kernel(3))),
        ).astype(bool)
        # Keep this tighter band separate from ``evidence_near``.  The wide
        # band only preserves valid contour pixels around a sparse parser arc;
        # it must not by itself prove that a full annulus belongs to that arc.
        evidence_boundary_band = cv2.dilate(
            side_evidence.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(5), kernel(3))),
        ).astype(bool)
        search_near = cv2.dilate(
            base_search.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(7), kernel(3))),
        ).astype(bool)
        minimum_evidence_area = max(3, int(round(1.5 * coordinate_scale * coordinate_scale)))
        if int((side_evidence & local_window).sum()) < minimum_evidence_area:
            return zero, zero, zero, zero
        anchor_evidence_guard = cv2.dilate(
            anchor_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(29), kernel(3))),
        ).astype(bool)
        evidence_component_count, evidence_labels = cv2.connectedComponents(
            side_evidence.astype(np.uint8)
        )
        lobe_associated_evidence = np.zeros_like(side_evidence, dtype=bool)
        for evidence_component_id in range(1, evidence_component_count):
            evidence_component = evidence_labels == evidence_component_id
            if np.any(evidence_component & anchor_evidence_guard):
                lobe_associated_evidence |= evidence_component
        lobe_evidence_boundary_band = cv2.dilate(
            lobe_associated_evidence.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(5), kernel(3))),
        ).astype(bool)
        valid = local_window & ((~hair_core & ~ear_core) | evidence_near | search_near)
        if int(valid.sum()) < max(80, int(round(24.0 * coordinate_scale * coordinate_scale))):
            return zero, zero, zero, zero

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gradient_x, gradient_y)
        values = gradient[valid]
        if values.size < 24:
            return zero, zero, zero, zero
        low = max(10, int(np.percentile(values, 48) * 0.55))
        high = max(low + 16, int(np.percentile(values, 84)))
        raw_edges = cv2.Canny(gray, low, min(255, high)) > 0
        raw_edges &= valid
        if int(raw_edges.sum()) < 20:
            return zero, zero, zero, zero
        raw_support = cv2.dilate(
            raw_edges.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(3), kernel(3))),
        ).astype(bool)
        contour_edges = cv2.morphologyEx(
            raw_edges.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(3), kernel(3))),
        )
        contours, hierarchy = cv2.findContours(contour_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) < 2 or hierarchy is None:
            return zero, zero, zero, zero

        contour_entries: list[
            tuple[int, int, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float, float], float]
        ] = []
        for contour_index, contour in enumerate(contours):
            if len(contour) < 12:
                continue
            shape = ellipse_shape(contour)
            if shape is None:
                continue
            center_x, center_y, major, minor, _ = shape
            if not (float(min_axis) <= minor <= major <= float(max_axis)):
                continue
            if minor / max(major, 1e-6) < 0.35:
                continue
            contour_fill = filled(contour)
            area = float(contour_fill.sum())
            if area < max(20.0, float(min_axis * min_axis) * 0.35):
                continue
            parent_index = int(hierarchy[0, contour_index, 3])
            contour_entries.append(
                (contour_index, parent_index, contour, contour_fill, outline(contour), shape, area)
            )
        if len(contour_entries) < 2:
            return zero, zero, zero, zero

        best_score = -1.0
        best_alpha = None
        best_hole = None
        best_footprint = None
        best_connector = None
        for outer_entry in contour_entries:
            (
                outer_contour_index,
                _,
                _,
                outer_fill,
                outer_boundary,
                outer_shape,
                outer_area,
            ) = outer_entry
            outer_x, outer_y, outer_major, outer_minor, _ = outer_shape
            for inner_entry in contour_entries:
                (
                    inner_contour_index,
                    inner_parent_index,
                    _,
                    inner_fill,
                    inner_boundary,
                    inner_shape,
                    inner_area,
                ) = inner_entry
                if inner_contour_index == outer_contour_index or inner_area >= outer_area * 0.82:
                    continue
                # An inner hole must be a direct child of its outer boundary.
                # Arbitrary nested background/grass contours are not a hoop.
                if inner_parent_index != outer_contour_index:
                    continue
                if not np.all(inner_fill <= outer_fill):
                    continue
                inner_x, inner_y, inner_major, inner_minor, _ = inner_shape
                center_distance = float(np.hypot(outer_x - inner_x, outer_y - inner_y))
                if center_distance > 0.18 * outer_major:
                    continue
                if abs(outer_major / max(inner_major, 1e-6) - outer_minor / max(inner_minor, 1e-6)) > 0.45:
                    continue
                annulus = outer_fill & ~inner_fill
                annulus_area = float(annulus.sum())
                if not (0.05 * outer_area <= annulus_area <= 0.78 * outer_area):
                    continue
                # A full nested contour is not enough on its own: the annulus
                # needs a meaningful, same-side source-evidence contact on its
                # measured boundary.  The old two-pixel test let a nearby grass
                # edge or an unrelated stud authorize a complete background
                # ring.  Keep the broad evidence band for candidate discovery,
                # but prove ownership with this much tighter boundary band.
                boundary = outer_boundary | inner_boundary
                boundary_evidence = boundary & evidence_boundary_band
                lobe_boundary_evidence = boundary & lobe_evidence_boundary_band
                boundary_area = max(1, int(boundary.sum()))
                boundary_contact = int(boundary_evidence.sum())
                minimum_boundary_contact = max(
                    int(round(6.0 * coordinate_scale)),
                    int(np.ceil(0.045 * boundary_area)),
                )
                if boundary_contact < minimum_boundary_contact:
                    continue
                evidence_points = np.argwhere(boundary_evidence)
                evidence_angles = np.arctan2(
                    evidence_points[:, 0] - outer_y,
                    evidence_points[:, 1] - outer_x,
                )
                evidence_sectors = np.unique(
                    np.floor((evidence_angles + np.pi) * (8.0 / (2.0 * np.pi))).astype(np.int32) % 8
                )
                if evidence_sectors.size < 2:
                    continue
                annulus_evidence = int((annulus & evidence_boundary_band).sum())
                if annulus_evidence < max(
                    int(round(2.0 * coordinate_scale * coordinate_scale)),
                    int(np.ceil(0.015 * annulus_area)),
                    4,
                ):
                    continue
                outer_coverage, outer_sectors = supported_sectors(
                    outer_boundary, raw_support, outer_x, outer_y
                )
                inner_coverage, inner_sectors = supported_sectors(
                    inner_boundary, raw_support, inner_x, inner_y
                )
                combined_coverage = 0.5 * (outer_coverage + inner_coverage)
                # Callers choose a lower coverage for reflective/small hoops.
                # Do not silently override it to 0.42 after they supplied
                # 0.36; the nested-contour, source-evidence and lobe-link
                # checks above remain the false-positive protection.
                required_coverage = max(0.34, float(min_coverage))
                required_sectors = 3 if min(outer_major, outer_minor) <= 28.0 * coordinate_scale else 4
                if (
                    outer_coverage < required_coverage
                    or inner_coverage < required_coverage
                    or outer_sectors < required_sectors
                    or inner_sectors < required_sectors
                ):
                    continue
                # Ear folds form nested contours too.  A real hoop is mostly
                # outside the ear surface, whereas a helix-shaped false match
                # remains inside it.
                ear_guard = cv2.dilate(
                    ear_mask.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(7), kernel(3))),
                ).astype(bool)
                exterior_ratio = float((annulus & ~ear_guard).sum()) / max(annulus_area, 1.0)
                if exterior_ratio < 0.45:
                    continue
                lobe_touch = bool(
                    (cv2.dilate(
                        outer_boundary.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel(25), kernel(3))),
                    ) > 0)[anchor_mask].any()
                )
                connector = connector_from_source(raw_support, valid, anchor_mask, outer_boundary)
                connector_requirement = max(5, int(round(3.0 * coordinate_scale)))
                # The accepted evidence must itself belong to this lobe, or a
                # measured source connector must link the lobe to the accepted
                # annulus.  Do not crop the annulus by hair/ear labels here:
                # once that proof succeeds, its complete source contour is the
                # correct RGB object, including arcs crossing rough parser masks.
                if (
                    (not lobe_touch and int(connector.sum()) < connector_requirement)
                    or (
                        int(lobe_boundary_evidence.sum()) < minimum_boundary_contact
                        and int(connector.sum()) < connector_requirement
                    )
                ):
                    continue

                # Trim only a one-pixel uncertain contour boundary.  This
                # avoids copying the dark background halo while preserving the
                # source ring's measured thickness and non-elliptic shape.
                trim_kernel = np.ones((3, 3), dtype=np.uint8)
                trimmed_outer = cv2.erode(outer_fill.astype(np.uint8), trim_kernel).astype(bool)
                trimmed_inner = cv2.dilate(inner_fill.astype(np.uint8), trim_kernel).astype(bool)
                # Once the source contains a valid nested contour pair, keep
                # the measured annulus rather than clipping it by rough hair/
                # ear parsing.  Clipping at this stage is what turned a real
                # hoop into one short arc.  The hierarchy, per-side seed and
                # edge coverage above are the safety checks.
                alpha = trimmed_outer & ~trimmed_inner
                if int(alpha.sum()) < max(8, int(round(3.0 * coordinate_scale))):
                    alpha = annulus
                alpha |= connector
                hole = inner_fill & outer_fill
                score = combined_coverage * float(outer_sectors + inner_sectors) * annulus_area
                # Prefer candidates close to the saved ear-local search, but
                # never require it: a complete hanging hoop can extend below
                # a 256px proposal while still being source-verified here.
                search_overlap = float((annulus & base_search).sum()) / max(annulus_area, 1.0)
                score *= 0.75 + 0.25 * search_overlap
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
                    best_hole = hole
                    best_footprint = outer_fill
                    best_connector = connector
        if best_alpha is None:
            return zero, zero, zero, zero
        return (
            best_alpha.astype(np.float32),
            best_hole.astype(np.float32),
            best_footprint.astype(np.float32),
            best_connector.astype(np.float32),
        )

    left_out = np.zeros((batch, 1, height, width), dtype=np.float32)
    right_out = np.zeros_like(left_out)
    left_hole_out = np.zeros_like(left_out)
    right_hole_out = np.zeros_like(left_out)
    left_connector_out = np.zeros_like(left_out)
    right_connector_out = np.zeros_like(left_out)
    left_footprint_out = np.zeros_like(left_out)
    right_footprint_out = np.zeros_like(left_out)
    for index in range(batch):
        (
            left_out[index, 0],
            left_hole_out[index, 0],
            left_footprint_out[index, 0],
            left_connector_out[index, 0],
        ) = trace_side(
            source_np[index],
            search_np[index, 0],
            left_evidence_np[index, 0],
            hair_np[index, 0],
            ear_np[index, 0],
            left_anchor_np[index, 0],
        )
        (
            right_out[index, 0],
            right_hole_out[index, 0],
            right_footprint_out[index, 0],
            right_connector_out[index, 0],
        ) = trace_side(
            source_np[index],
            search_np[index, 0],
            right_evidence_np[index, 0],
            hair_np[index, 0],
            ear_np[index, 0],
            right_anchor_np[index, 0],
        )

    return {
        "left_elliptical_hoop": torch.from_numpy(left_out).to(reference.device, reference.dtype),
        "right_elliptical_hoop": torch.from_numpy(right_out).to(reference.device, reference.dtype),
        "left_elliptical_hoop_hole": torch.from_numpy(left_hole_out).to(reference.device, reference.dtype),
        "right_elliptical_hoop_hole": torch.from_numpy(right_hole_out).to(reference.device, reference.dtype),
        "left_elliptical_hoop_footprint": torch.from_numpy(left_footprint_out).to(reference.device, reference.dtype),
        "right_elliptical_hoop_footprint": torch.from_numpy(right_footprint_out).to(reference.device, reference.dtype),
        "left_elliptical_hoop_connector": torch.from_numpy(left_connector_out).to(reference.device, reference.dtype),
        "right_elliptical_hoop_connector": torch.from_numpy(right_connector_out).to(reference.device, reference.dtype),
    }


def refine_earring_hoops_highres(
    source_01: torch.Tensor,
    search_mask: torch.Tensor,
    left_lobe_anchor: torch.Tensor,
    right_lobe_anchor: torch.Tensor,
    *,
    left_source_evidence: torch.Tensor | None = None,
    right_source_evidence: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    source_ear_mask: torch.Tensor | None = None,
    detection_size: int = 512,
    min_axis: float = 6.0,
    min_coverage: float = 0.28,
) -> dict[str, torch.Tensor]:
    """Extract a complete hoop as a source-native annular instance.

    The old path fitted one ellipse and returned only the Canny pixels that
    happened to lie on that perimeter.  A reflective hoop therefore became a
    short arc, while its hole came from a different synthetic ellipse.  The
    implementation below accepts a hoop only when the source contains a pair
    of nested, edge-supported contours.  Its RGB alpha is the real annulus
    between those contours; the inner contour is returned as a separate hole
    that must remain target-owned.
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

    candidates = build_contour_hoop_instances_v5(
        F.interpolate(source_01, size=detect_size, mode="bilinear", align_corners=False),
        resize_for_detection(search_mask),
        resize_for_detection(left_lobe_anchor),
        resize_for_detection(right_lobe_anchor),
        left_source_evidence=(
            None if left_source_evidence is None else resize_for_detection(left_source_evidence)
        ),
        right_source_evidence=(
            None if right_source_evidence is None else resize_for_detection(right_source_evidence)
        ),
        source_hair_mask=resize_for_detection(source_hair_mask),
        source_ear_mask=resize_for_detection(source_ear_mask),
        min_axis=max(4.0 * detector_scale, float(min_axis) * detector_scale),
        max_axis=168.0 * detector_scale,
        min_coverage=float(min_coverage),
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
    candidate_seed = candidate.clone()
    ear_exterior = (1.0 - erode_mask(source_ear, 7)).clamp(0, 1)
    # Visual structure may complete an existing weak/parser proposal, but it
    # may not start an accessory anywhere in the ear corridor.  The latter was
    # the route by which grass, hair edges and ear folds became fake earrings.
    visual_wire = (
        edge_support
        * contrast_support
        * ear_exterior
        * ear_roi
        * dilate_mask(candidate_seed + parser_earring, 7)
    )
    candidate = torch.maximum(candidate, visual_wire)
    object_evidence = candidate * torch.clamp(edge_support + chroma_support + contrast_support, 0, 1)
    # A parser-missed metal wire is often labelled background.  Do not erase it
    # pixel-wise here; reject components by *coverage ratio* below instead.
    # Hair still receives a strong attenuation because hair edges are the most
    # common visual false positive around an ear.
    object_evidence = object_evidence * (1.0 - 0.75 * hair).clamp(0, 1) * ear_roi
    object_evidence = dilate_mask(object_evidence, 3) * candidate * ear_roi
    # A semantic earring label may grow into immediately adjacent observed
    # pixels, but it must remain an image-derived object.  The previous policy
    # discarded every parser-missed ordinary earring unless an ellipse fitter
    # could redraw it, which is why solid earrings started disappearing.
    parser_neighbourhood = dilate_mask(parser_earring, 13)
    parser_present = (
        parser_earring.flatten(1).sum(dim=1, keepdim=True) >= 1.0
    ).view(-1, 1, 1, 1)
    parser_guided = torch.maximum(
        parser_earring,
        object_evidence * parser_neighbourhood,
    )
    # With no parser label, keep the compact lobe-connected visual evidence.
    # The component and density checks below, followed by the final direct
    # instance extractor, prevent this from turning an exposed ear into a
    # broad source-background paste.
    object_evidence = torch.where(parser_present, parser_guided, object_evidence)

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
        structure_ratio = (
            side * edge_support * contrast_support
        ).flatten(1).sum(dim=1, keepdim=True) / area.clamp_min(1.0)
        near_lobe = (side * dilate_mask(anchor, 17)).flatten(1).sum(dim=1, keepdim=True) >= 1.0
        # This low-resolution candidate is only a coarse presence hint.  It
        # cannot distinguish a metal edge from grass, foliage or a hard
        # background edge near the lobe, so background-heavy components must
        # not activate the generic earring branch here.  Parser-missed objects
        # that really are labelled background are handled separately by the
        # source-native verifier below, where their compact instance (or paired
        # hoop contours) is measured before any RGB write is allowed.
        background_ok = background_ratio <= 0.68
        plausible = (
            (area >= float(min_area) * area_scale)
            & (density <= float(max_roi_density))
            & background_ok
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
        # Kept as empty compatibility/debug entries for existing dataset
        # readers.  Ellipse fitting no longer controls either RGB recovery or
        # the hollow part of a hoop.
        "left_elliptical_hoop": torch.zeros_like(left),
        "right_elliptical_hoop": torch.zeros_like(right),
        "elliptical_hoop_hole": torch.zeros_like(left),
        "left_elliptical_hoop_hole": torch.zeros_like(left),
        "right_elliptical_hoop_hole": torch.zeros_like(right),
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

    Tier A is a verified parser, explicit object, or strong visual core.  Tier B
    is a source-blocked, lobe-connected continuation.  Both are allowed to sit
    in front of transferred target hair once that lobe is visible: target hair
    is a compositing layer behind a verified accessory, not evidence that the
    accessory does not exist.  Source-background/hair/semantic blockers and
    the later topology hole mask remain the safeguards against empty patches.
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

    # ``trusted`` is already an object-supported, side-gated mask.  Do not
    # intersect it with the compact target-ear ROI here: that ROI is a search
    # and completion corridor, while a confirmed pendant/hoop may extend well
    # beyond the ear shell.  Using ``visible`` for this tier silently reduced
    # complete source instances to lobe-sized fragments before the native
    # compositor could recover them.
    core = extract_earring_object_core(
        trusted * active,
        trusted * active,
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

    # Do not cap a source-safe continuation by its overlap with *target* hair.
    # The old ratio policy made ``allowed_hair`` zero whenever a long pendant
    # fell wholly below an exposed lobe onto transferred hair, so the lower
    # body disappeared while exactly the same source object was kept over a
    # bare neck.  At this point ``completion`` has already passed source
    # background/hair/semantic blocking and lobe connectivity; it is therefore
    # object alpha, not an ear-side crop.  The final compositor writes this
    # alpha after target hair, while ``hoop_hole`` below keeps hollow centres
    # target-owned.  Keep the argument for the legacy wrapper/API contract.
    completion = completion.clamp(0, 1)

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


def _ear_bottom_attachment(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure the visible lower-lobe point used to hang an earring."""
    mask = (ensure_mask_4d(mask).float() > 0.5).float()
    height = mask.shape[-2]
    y = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    y_min = torch.where(
        mask > 0,
        y.expand_as(mask),
        torch.full_like(mask, float(height)),
    ).flatten(1).amin(dim=1, keepdim=True)
    y_max = (mask * y).flatten(1).amax(dim=1, keepdim=True)
    band_height = torch.maximum(
        torch.full_like(y_max, 2.0),
        (y_max - y_min).clamp_min(1.0) * 0.14,
    )
    bottom = mask * (y >= (y_max - band_height).view(-1, 1, 1, 1)).float()
    bottom_area = bottom.flatten(1).sum(dim=1, keepdim=True)
    use_bottom = (bottom_area > 0.5).float()
    return _weighted_centroid(
        bottom * use_bottom.view(-1, 1, 1, 1)
        + mask * (1.0 - use_bottom.view(-1, 1, 1, 1))
    )


def _earring_top_attachment(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure the upper band of a confirmed source-native accessory."""
    mask = (ensure_mask_4d(mask).float() > 0.5).float()
    height = mask.shape[-2]
    y = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    y_min = torch.where(
        mask > 0,
        y.expand_as(mask),
        torch.full_like(mask, float(height)),
    ).flatten(1).amin(dim=1, keepdim=True)
    y_max = (mask * y).flatten(1).amax(dim=1, keepdim=True)
    band_height = torch.maximum(
        torch.full_like(y_min, 2.0),
        (y_max - y_min).clamp_min(1.0) * 0.10,
    )
    top = mask * (y <= (y_min + band_height).view(-1, 1, 1, 1)).float()
    top_area = top.flatten(1).sum(dim=1, keepdim=True)
    use_top = (top_area > 0.5).float()
    return _weighted_centroid(
        top * use_top.view(-1, 1, 1, 1)
        + mask * (1.0 - use_top.view(-1, 1, 1, 1))
    )


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
    target_left_roi: torch.Tensor | None = None,
    target_right_roi: torch.Tensor | None = None,
    max_vertical_shift: int = 0,
    max_horizontal_shift: int = 0,
    reference_base: torch.Tensor | None = None,
    source_left_instance_mask: torch.Tensor | None = None,
    source_right_instance_mask: torch.Tensor | None = None,
    source_left_hole_mask: torch.Tensor | None = None,
    source_right_hole_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    source_01 = normalized_to_01(source_01)
    source_earring_mask = ensure_mask_4d(source_earring_mask).float()
    left_roi = ensure_mask_4d(left_roi).float()
    right_roi = ensure_mask_4d(right_roi).float()
    source_left_ear_mask = ensure_mask_4d(source_left_ear_mask).float()
    source_right_ear_mask = ensure_mask_4d(source_right_ear_mask).float()
    target_left_ear_mask = ensure_mask_4d(target_left_ear_mask).float()
    target_right_ear_mask = ensure_mask_4d(target_right_ear_mask).float()
    target_left_roi = (
        torch.ones_like(target_left_ear_mask)
        if target_left_roi is None
        else ensure_mask_4d(target_left_roi).float().to(target_left_ear_mask.device)
    )
    target_right_roi = (
        torch.ones_like(target_right_ear_mask)
        if target_right_roi is None
        else ensure_mask_4d(target_right_roi).float().to(target_right_ear_mask.device)
    )
    if target_left_roi.shape[-2:] != target_left_ear_mask.shape[-2:]:
        target_left_roi = F.interpolate(
            target_left_roi,
            size=target_left_ear_mask.shape[-2:],
            mode="nearest",
        )
    if target_right_roi.shape[-2:] != target_right_ear_mask.shape[-2:]:
        target_right_roi = F.interpolate(
            target_right_roi,
            size=target_right_ear_mask.shape[-2:],
            mode="nearest",
        )

    def instance_or_roi(value: torch.Tensor | None, roi: torch.Tensor) -> torch.Tensor:
        if value is None:
            return source_earring_mask * roi
        value = ensure_mask_4d(value).float().to(source_earring_mask.device)
        if value.shape[-2:] != source_earring_mask.shape[-2:]:
            value = F.interpolate(value, size=source_earring_mask.shape[-2:], mode="nearest")
        return value.clamp(0, 1)

    def hole_or_zero(value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            return torch.zeros_like(source_earring_mask)
        value = ensure_mask_4d(value).float().to(source_earring_mask.device)
        if value.shape[-2:] != source_earring_mask.shape[-2:]:
            value = F.interpolate(value, size=source_earring_mask.shape[-2:], mode="nearest")
        return value.clamp(0, 1)

    # A large hoop regularly extends beyond the compact ear ROI.  Use the
    # explicit side instance when available; the ROI remains only a fallback
    # for legacy ordinary-earring callers.
    left_ring = instance_or_roi(source_left_instance_mask, left_roi)
    right_ring = instance_or_roi(source_right_instance_mask, right_roi)
    left_hole = hole_or_zero(source_left_hole_mask)
    right_hole = hole_or_zero(source_right_hole_mask)
    # Use the source accessory's upper attachment band and the target ear's
    # visible lower band.  A mixed ear/earring centroid is biased upward by a
    # long pendant and places it in the middle of the target lobe.  Target
    # ROIs are deliberately ignored for this measurement: they are source-
    # side search envelopes and can crop the real target lobe.
    left_src_y, left_src_x, left_src_valid = _earring_top_attachment(left_ring)
    right_src_y, right_src_x, right_src_valid = _earring_top_attachment(right_ring)
    left_tgt_y, left_tgt_x, left_tgt_valid = _ear_bottom_attachment(target_left_ear_mask)
    right_tgt_y, right_tgt_x, right_tgt_valid = _ear_bottom_attachment(target_right_ear_mask)

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
    shifted_left_hole = shift_tensor_per_batch(left_hole, left_shift_y, left_shift_x)
    shifted_right_hole = shift_tensor_per_batch(right_hole, right_shift_y, right_shift_x)
    shifted_left_rgb = shift_tensor_per_batch(source_01 * left_ring, left_shift_y, left_shift_x)
    shifted_right_rgb = shift_tensor_per_batch(source_01 * right_ring, right_shift_y, right_shift_x)

    original_ring = (left_ring + right_ring).clamp(0, 1)
    aligned_mask = (shifted_left_mask + shifted_right_mask).clamp(0, 1)
    aligned_hole = (shifted_left_hole + shifted_right_hole).clamp(0, 1)
    aligned_mask = aligned_mask * (1.0 - aligned_hole).clamp(0, 1)
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
        "hoop_hole_mask": aligned_hole,
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

        # Visibility answers only whether the target ear/lobe itself is
        # exposed.  A one-pixel target-hair boundary often lands on the lobe
        # after hairstyle transfer even when the hair is visibly *below* the
        # ear.  Treat only a compact hair interior as a lobe occluder here.
        # This keeps a real, exposed earlobe eligible for a foreground pendant
        # while a materially hair-covered ear still remains closed.
        target_hair_lobe_core = erode_mask(target_hair_mask, 3)
        target_open_mask = (1 - target_hair_lobe_core).clamp(0, 1) * (1 - hat_mask)
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
            # ``side_roi`` also contains the complete source earring corridor.
            # It may extend far below a visible lobe, where transferred long
            # hair is expected.  Measure target visibility only in a compact
            # ear/lobe probe, otherwise that background hair suppresses valid
            # long pendants and hoops.
            compact_ear = (target_ear_mask * side_roi).clamp(0, 1)
            visibility_probe = torch.clamp(
                dilate_mask(compact_ear, 9),
                0,
                1,
            ) * side_roi
            visible_ear = compact_ear * target_open_mask
            ear_area = compact_ear.flatten(1).sum(dim=1)
            visible_ear_area = visible_ear.flatten(1).sum(dim=1)
            probe_area = visibility_probe.flatten(1).sum(dim=1).clamp_min(1.0)
            visible_ratio = visible_ear_area / ear_area.clamp_min(1.0)
            hair_ratio = (
                target_hair_mask * visibility_probe
            ).flatten(1).sum(dim=1) / probe_area
            area_scale = float(side_roi.shape[-1] * side_roi.shape[-2]) / float(256 * 256)

            # A narrow exposed lobe is still sufficient to recover an
            # earring.  The old full ``min_target_ear_area`` threshold made
            # one- to a few-pixel lobes look covered and suppressed the whole
            # side during both dataset generation and inference.
            parser_visible = (
                visible_ear_area
                >= max(1.0, 0.10 * float(self.min_target_ear_area) * area_scale)
            )
            # Target-side openness is a hard safety contract.  Generic cheek
            # skin inside an expanded ear ROI is not proof that the target
            # ear/lobe is visible: that fallback was reopening fully
            # hair-covered ears and then forcing a source earring through the
            # transferred hairstyle.  Require an actual target ear label and
            # a lower-lobe anchor clear of *hair interior*.  A hair mask edge
            # or hair below the lobe must not disable the full hanging
            # accessory, which is composited as foreground later.
            semantic_fallback = torch.zeros_like(visibility_probe)
            fallback_visible = torch.zeros_like(parser_visible)
            target_lobe_anchor = build_earlobe_anchor(
                target_ear_mask,
                ear_roi=side_roi,
                lower_ratio=0.62,
                dilate=3,
            ) * target_open_mask
            target_lobe_present = (
                target_lobe_anchor.flatten(1).sum(dim=1) >= 2.0 * area_scale
            ).view(-1, 1, 1, 1)
            lobe_anchor = target_lobe_anchor
            # A visible upper ear is insufficient: earrings hang from the
            # lower lobe, and a lobe under transferred hair must stay closed.
            side_visible = (parser_visible.view(-1, 1, 1, 1) & target_lobe_present).float()

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
        # Keep the complete source-native instance.  ``*_earring_valid_roi``
        # is a target-side permission/gate, not a pixel crop: applying it here
        # truncates hanging earrings and hoops when the target lobe is only
        # partially visible.  Per-side visibility is applied by the V5
        # compositor after instance extraction.
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

        # As above, expose the full source object to downstream instance and
        # hole handling.  The target-side gate must not become an alpha crop.
        source_earring_mask = source_masks["earring"]
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
