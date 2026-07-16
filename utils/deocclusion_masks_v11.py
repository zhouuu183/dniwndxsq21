from __future__ import annotations

import torch
import torch.nn.functional as F


# CelebA-style labels after swap_parsing_label_to_celeba_mask().
BACKGROUND_LABEL_V11 = 0
FACE_SURFACE_LABELS_V11 = (1, 2)
DETAIL_PROTECT_LABELS_V11 = (3, 4, 5, 6, 7, 10, 11, 12, 14)
EAR_LABELS_V11 = (8, 9, 15)
HAIR_LABELS_V11 = (13,)
NECK_LABELS_V11 = (16, 17)
CLOTH_LABELS_V11 = (18,)

DGRR_INPUT_MASK_KEYS_V11 = (
    "M_source_hair",
    "M_target_hair",
    "M_removed",
    "M_halo",
    "M_skin",
    "M_struct",
    "M_bg",
    "M_fill",
    "M_clean",
    "R_ctx",
    "R_bg_ctx",
    "E_struct",
    "D_t_norm",
)

DEOCCLUSION_MASK_KEYS_V11 = (
    "M_source_hair",
    "M_target_hair",
    "M_removed",
    "M_removed_soft",
    "M_halo",
    "M_reveal",
    "M_reveal_soft",
    "M_skin",
    "M_skin_soft",
    "M_struct",
    "M_struct_soft",
    "M_fill",
    "M_fill_soft",
    "M_bg",
    "M_bg_soft",
    "M_clean",
    "M_clean_soft",
    "R_ctx",
    "R_bg_ctx",
    "M_safe",
    "M_face_preserve",
    "M_shadow",
    "E_struct",
    "D_t_norm",
    "M_clean_far",
    "M_detail_protect",
)


def ensure_mask_4d(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        return mask.unsqueeze(0)
    return mask


def resize_mask(mask: torch.Tensor | None, size: tuple[int, int], mode: str = "nearest") -> torch.Tensor | None:
    mask = ensure_mask_4d(mask)
    if mask is None:
        return None
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    return F.interpolate(mask.float(), size=size, mode=mode, align_corners=align_corners).clamp(0, 1)


def resize_label_map(label_map: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    label_map = ensure_mask_4d(label_map)
    return F.interpolate(label_map.float(), size=size, mode="nearest")


def labels_to_mask(parsing_mask: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    parsing_mask = ensure_mask_4d(parsing_mask)
    out = torch.zeros_like(parsing_mask, dtype=torch.float32)
    for label in labels:
        out = torch.where(parsing_mask == label, torch.ones_like(out), out)
    return out.float()


def hair_mask_from_parsing(parsing_mask: torch.Tensor) -> torch.Tensor:
    return labels_to_mask(parsing_mask, HAIR_LABELS_V11)


def _binary(mask: torch.Tensor) -> torch.Tensor:
    return (ensure_mask_4d(mask).float() > 0.5).float()


def dilate_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width).clamp(0, 1)


def erode_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return (1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=width)).clamp(0, 1)


def boundary_mask(mask: torch.Tensor, width: int = 2) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    return (dilate_mask(mask, width) - erode_mask(mask, width)).clamp(0, 1)


def distance_from_mask_v11(mask: torch.Tensor, max_distance: int = 64) -> torch.Tensor:
    mask = _binary(mask)
    known = mask.clone()
    distance = torch.zeros_like(mask)
    frontier = mask.clone()
    for step in range(1, max_distance + 1):
        expanded = dilate_mask(frontier, 1)
        new_ring = (expanded * (1.0 - known)).clamp(0, 1)
        if float(new_ring.sum().detach().cpu()) == 0.0:
            break
        distance = distance + new_ring * float(step)
        known = (known + new_ring).clamp(0, 1)
        frontier = new_ring
    distance = torch.where(known > 0, distance, torch.full_like(distance, float(max_distance)))
    return distance.clamp(0, float(max_distance))


def _gaussian_kernel_1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords.pow(2)) / (2 * sigma * sigma))
    return kernel / kernel.sum().clamp(min=1e-8)


def blur_mask(mask: torch.Tensor, kernel_size: int = 17, sigma: float | None = None) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    if kernel_size <= 1:
        return mask.clamp(0, 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if sigma is None or sigma <= 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    kernel_1d = _gaussian_kernel_1d(kernel_size, sigma, mask.device, mask.dtype)
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size).repeat(mask.shape[1], 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1).repeat(mask.shape[1], 1, 1, 1)
    blurred = F.conv2d(mask, kernel_x, padding=(0, kernel_size // 2), groups=mask.shape[1])
    blurred = F.conv2d(blurred, kernel_y, padding=(kernel_size // 2, 0), groups=mask.shape[1])
    return blurred.clamp(0, 1)


def soft_mask(mask: torch.Tensor, dilate_width: int = 3, blur_kernel: int = 17) -> torch.Tensor:
    return blur_mask(dilate_mask(mask, dilate_width), kernel_size=blur_kernel).clamp(0, 1)


def contained_soft_mask(mask: torch.Tensor, blur_kernel: int = 7, contain_width: int = 1) -> torch.Tensor:
    contain = dilate_mask(mask, contain_width)
    return (blur_mask(mask.float(), kernel_size=blur_kernel) * contain).clamp(0, 1)


def _vertical_band(mask_ref: torch.Tensor, y_min: float = 0.0, y_max: float = 1.0) -> torch.Tensor:
    _, _, height, _ = mask_ref.shape
    y_coords = torch.linspace(0.0, 1.0, steps=height, device=mask_ref.device, dtype=mask_ref.dtype).view(1, 1, height, 1)
    return ((y_coords >= y_min) & (y_coords <= y_max)).float()


def _merge_semantic(source_parsing: torch.Tensor, target_parsing: torch.Tensor | None, labels: tuple[int, ...]) -> torch.Tensor:
    merged = labels_to_mask(source_parsing, labels)
    if target_parsing is not None:
        if target_parsing.shape[-2:] != source_parsing.shape[-2:]:
            target_parsing = resize_label_map(target_parsing, source_parsing.shape[-2:])
        merged = torch.maximum(merged, labels_to_mask(target_parsing, labels))
    return merged.float()


def build_deocclusion_masks_v11(
    source_parsing: torch.Tensor,
    target_hair_mask: torch.Tensor,
    *,
    source_hair_mask: torch.Tensor | None = None,
    target_parsing: torch.Tensor | None = None,
    target_protect_width: int = 2,
    removed_dilate_width: int = 3,
    halo_width: int = 8,
    skin_expand_width: int = 8,
    struct_expand_width: int = 22,
    clean_boundary_width: int = 4,
    fill_distance: int = 9,
    context_width: int = 9,
    max_distance: int = 64,
    safe_width: int = 4,
    face_protect_width: int = 5,
    tail_y_min: float = 0.52,
    reveal_blur_kernel: int = 7,
) -> dict[str, torch.Tensor]:
    source_parsing = ensure_mask_4d(source_parsing)
    target_hair = _binary(target_hair_mask)
    if source_parsing.shape[-2:] != target_hair.shape[-2:]:
        source_parsing = resize_label_map(source_parsing, target_hair.shape[-2:])

    if source_hair_mask is None:
        source_hair = hair_mask_from_parsing(source_parsing)
    else:
        source_hair = _binary(source_hair_mask)
    if source_hair.shape[-2:] != target_hair.shape[-2:]:
        source_hair = resize_mask(source_hair, target_hair.shape[-2:], mode="nearest")

    target_semantic = ensure_mask_4d(target_parsing) if target_parsing is not None else None
    if target_semantic is not None and target_semantic.shape[-2:] != target_hair.shape[-2:]:
        target_semantic = resize_label_map(target_semantic, target_hair.shape[-2:])

    target_protect = dilate_mask(target_hair, target_protect_width)
    removed = (source_hair * (1.0 - target_protect)).clamp(0, 1)
    removed_soft = soft_mask(removed, removed_dilate_width, reveal_blur_kernel)
    halo = (dilate_mask(removed, halo_width) - removed).clamp(0, 1)
    halo = (halo * (1.0 - target_protect)).clamp(0, 1)
    reveal = (removed + halo).clamp(0, 1)
    reveal_soft = soft_mask(reveal, removed_dilate_width, reveal_blur_kernel)
    target_boundary = boundary_mask(target_hair, 2)
    d_t = distance_from_mask_v11(target_boundary, max_distance=max_distance)
    d_t_norm = (d_t / float(max_distance)).clamp(0, 1)

    face = _merge_semantic(source_parsing, target_semantic, FACE_SURFACE_LABELS_V11)
    ear = _merge_semantic(source_parsing, target_semantic, EAR_LABELS_V11)
    neck = _merge_semantic(source_parsing, target_semantic, NECK_LABELS_V11)
    cloth = _merge_semantic(source_parsing, target_semantic, CLOTH_LABELS_V11)
    background = _merge_semantic(source_parsing, target_semantic, (BACKGROUND_LABEL_V11,))
    detail_protect = dilate_mask(_merge_semantic(source_parsing, target_semantic, DETAIL_PROTECT_LABELS_V11), 2)
    face_core = erode_mask(face, max(1, face_protect_width))
    face_edge_band = (face - face_core).clamp(0, 1)
    face_preserve = ((face_core + detail_protect).clamp(0, 1) * (1.0 - dilate_mask(removed, 1))).clamp(0, 1)

    upper_band = _vertical_band(target_hair, 0.0, 0.76)
    lower_band = _vertical_band(target_hair, tail_y_min, 1.0)
    skin_route = dilate_mask((face + 0.7 * ear + 0.85 * neck).clamp(0, 1), skin_expand_width)
    skin_route = (skin_route * (1.0 - 0.65 * detail_protect)).clamp(0, 1)
    m_skin = (reveal_soft * skin_route * (upper_band + lower_band).clamp(0, 1) * (1.0 - face_preserve)).clamp(0, 1)

    body = (neck + cloth).clamp(0, 1)
    body_near = dilate_mask(body, struct_expand_width)
    neck_boundary = boundary_mask((neck + cloth).clamp(0, 1), 3)
    bg_nearbody = (background * body_near).clamp(0, 1)
    shoulder_band = _vertical_band(target_hair, 0.48, 1.0)
    struct_route = (cloth + 0.7 * neck_boundary + 0.55 * bg_nearbody + 0.45 * body_near).clamp(0, 1)
    m_struct = (reveal_soft * struct_route * shoulder_band).clamp(0, 1)

    target_boundary_band = dilate_mask(target_boundary, max(1, clean_boundary_width))
    near_removed = dilate_mask(removed, max(1, clean_boundary_width * 2))
    clean_semantic = (face_edge_band + ear + neck_boundary + bg_nearbody + body_near).clamp(0, 1)
    m_clean = (
        target_boundary_band
        * near_removed
        * (1.0 - target_hair)
        * clean_semantic
        * (1.0 - face_preserve)
    ).clamp(0, 1)
    m_clean = erode_mask(m_clean, 1)

    unknown = (1.0 - (face + ear + neck + cloth + target_hair).clamp(0, 1)).clamp(0, 1)
    bg_route = (background + 0.4 * unknown).clamp(0, 1)
    m_bg = (reveal_soft * bg_route * (1.0 - target_hair) * (1.0 - face_preserve)).clamp(0, 1)
    fill_semantic = (
        cloth
        + 0.85 * body_near
        + 0.85 * neck_boundary
        + 0.45 * bg_nearbody
        + 0.35 * unknown
    ).clamp(0, 1)
    fill_far = (d_t >= float(fill_distance)).float()
    m_fill = (removed * fill_far * (1.0 - m_skin) * (1.0 - m_bg) * (1.0 - m_clean) * fill_semantic).clamp(0, 1)
    m_fill = erode_mask(dilate_mask(m_fill, 1), 1)

    m_skin_soft = (contained_soft_mask(m_skin, reveal_blur_kernel, 1) * (1.0 - face_preserve)).clamp(0, 1)
    m_struct_soft = (contained_soft_mask(m_struct, reveal_blur_kernel, 1) * (1.0 - face_preserve)).clamp(0, 1)
    m_fill_soft = (contained_soft_mask(m_fill, reveal_blur_kernel, 1) * (1.0 - face_preserve)).clamp(0, 1)
    m_bg_soft = (contained_soft_mask(m_bg, reveal_blur_kernel, 1) * (1.0 - face_preserve)).clamp(0, 1)
    m_clean_soft = (contained_soft_mask(m_clean, max(5, reveal_blur_kernel), 0) * (1.0 - face_preserve)).clamp(0, 1)
    fill_context_semantic = (cloth + body_near + neck_boundary + 0.5 * bg_nearbody + 0.25 * unknown).clamp(0, 1)
    r_ctx = (dilate_mask(m_fill, context_width) - m_fill).clamp(0, 1)
    r_ctx = (r_ctx * fill_context_semantic * (1.0 - source_hair) * (1.0 - target_hair)).clamp(0, 1)
    bg_context_semantic = (background + 0.35 * unknown).clamp(0, 1)
    r_bg_ctx = (dilate_mask(m_bg, context_width) - m_bg).clamp(0, 1)
    r_bg_ctx = (r_bg_ctx * bg_context_semantic * (1.0 - source_hair) * (1.0 - target_hair)).clamp(0, 1)
    m_safe = (1.0 - dilate_mask(reveal, safe_width) - target_hair).clamp(0, 1)
    m_safe = torch.maximum(m_safe, face_preserve * (1.0 - target_hair))

    e_struct = (
        boundary_mask(face, 2)
        + boundary_mask(neck, 2)
        + boundary_mask(cloth, 2)
        + boundary_mask(target_hair, 2)
    ).clamp(0, 1)
    e_struct = blur_mask(e_struct, kernel_size=5)
    m_shadow = (
        (m_skin_soft + m_fill_soft + m_bg_soft + 0.25 * halo).clamp(0, 1)
        * (1.0 - target_hair)
        * (1.0 - face_preserve)
    ).clamp(0, 1)
    m_clean_far = (m_clean_soft * (d_t >= float(fill_distance)).float()).clamp(0, 1)

    return {
        "M_source_hair": source_hair,
        "M_target_hair": target_hair,
        "M_removed": removed,
        "M_removed_soft": removed_soft,
        "M_halo": halo,
        "M_reveal": reveal,
        "M_reveal_soft": reveal_soft,
        "M_skin": m_skin,
        "M_skin_soft": m_skin_soft,
        "M_struct": m_struct,
        "M_struct_soft": m_struct_soft,
        "M_fill": m_fill,
        "M_fill_soft": m_fill_soft,
        "M_bg": m_bg,
        "M_bg_soft": m_bg_soft,
        "M_clean": m_clean,
        "M_clean_soft": m_clean_soft,
        "R_ctx": r_ctx,
        "R_bg_ctx": r_bg_ctx,
        "M_safe": m_safe,
        "M_face_preserve": face_preserve,
        "M_shadow": m_shadow,
        "E_struct": e_struct,
        "D_t_norm": d_t_norm,
        "M_clean_far": m_clean_far,
        "M_detail_protect": detail_protect,
    }


def stack_deocclusion_masks_v11(
    masks: dict[str, torch.Tensor],
    *,
    keys: tuple[str, ...] = DGRR_INPUT_MASK_KEYS_V11,
    size: tuple[int, int] | None = None,
) -> torch.Tensor:
    ref = next(iter(masks.values()))
    ref = ensure_mask_4d(ref)
    zeros = torch.zeros_like(ref)
    stacked = []
    for key in keys:
        value = ensure_mask_4d(masks.get(key, zeros))
        if size is not None and value.shape[-2:] != size:
            mode = "bilinear" if (key.endswith("_soft") or key.startswith("E_") or key.startswith("D_")) else "nearest"
            value = resize_mask(value, size, mode=mode)
        stacked.append(value.float())
    return torch.cat(stacked, dim=1).clamp(0, 1)


__all__ = [
    "DGRR_INPUT_MASK_KEYS_V11",
    "DEOCCLUSION_MASK_KEYS_V11",
    "build_deocclusion_masks_v11",
    "blur_mask",
    "boundary_mask",
    "dilate_mask",
    "distance_from_mask_v11",
    "erode_mask",
    "ensure_mask_4d",
    "hair_mask_from_parsing",
    "labels_to_mask",
    "resize_mask",
    "soft_mask",
    "stack_deocclusion_masks_v11",
]
