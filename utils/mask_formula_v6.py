from __future__ import annotations

import torch
import torch.nn.functional as F


# CelebA-style parsing labels after swap_parsing_label_to_celeba_mask():
# 0 background, 1 skin_other, 2 nose, 3 eye_g, 4 l_eye, 5 r_eye, 6 l_brow,
# 7 r_brow, 8 l_ear, 9 r_ear, 10 mouth, 11 u_lip, 12 l_lip, 13 hair,
# 14 hat, 15 ear_r, 16 neck_l, 17 neck, 18 cloth
FACE_SURFACE_LABELS_V6 = (1, 2)
EAR_SURFACE_LABELS_V6 = (8, 9, 15)
BODY_SURFACE_LABELS_V6 = (16, 17, 18)
DETAIL_BLOCK_LABELS_V6 = (3, 4, 5, 6, 7, 10, 11, 12, 14)
SKIN_LABELS_V6 = FACE_SURFACE_LABELS_V6 + EAR_SURFACE_LABELS_V6 + BODY_SURFACE_LABELS_V6
SKIN_SURFACE_LABELS_V6 = FACE_SURFACE_LABELS_V6 + EAR_SURFACE_LABELS_V6
BACKGROUND_LABEL_V6 = 0
HAIR_LABEL_V6 = 13


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


def labels_to_mask(parsing_mask: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    parsing_mask = ensure_mask_4d(parsing_mask)
    label_mask = torch.zeros_like(parsing_mask, dtype=torch.float32)
    for label in labels:
        label_mask = torch.where(parsing_mask == label, torch.ones_like(label_mask), label_mask)
    return label_mask.float()


def hair_mask_from_parsing(parsing_mask: torch.Tensor, hair_label: int = HAIR_LABEL_V6) -> torch.Tensor:
    parsing_mask = ensure_mask_4d(parsing_mask)
    return torch.where(parsing_mask == hair_label, torch.ones_like(parsing_mask), torch.zeros_like(parsing_mask)).float()


def skin_surface_mask_from_parsing(
    parsing_mask: torch.Tensor,
    skin_surface_labels: tuple[int, ...] = SKIN_SURFACE_LABELS_V6,
) -> torch.Tensor:
    return labels_to_mask(parsing_mask, skin_surface_labels)


def face_surface_mask_from_parsing(
    parsing_mask: torch.Tensor,
    face_surface_labels: tuple[int, ...] = FACE_SURFACE_LABELS_V6,
) -> torch.Tensor:
    return labels_to_mask(parsing_mask, face_surface_labels)


def ear_surface_mask_from_parsing(
    parsing_mask: torch.Tensor,
    ear_surface_labels: tuple[int, ...] = EAR_SURFACE_LABELS_V6,
) -> torch.Tensor:
    return labels_to_mask(parsing_mask, ear_surface_labels)


def body_surface_mask_from_parsing(
    parsing_mask: torch.Tensor,
    body_surface_labels: tuple[int, ...] = BODY_SURFACE_LABELS_V6,
) -> torch.Tensor:
    return labels_to_mask(parsing_mask, body_surface_labels)


def blocked_detail_mask_from_parsing(
    parsing_mask: torch.Tensor,
    blocked_labels: tuple[int, ...] = DETAIL_BLOCK_LABELS_V6,
) -> torch.Tensor:
    return labels_to_mask(parsing_mask, blocked_labels)


def background_mask_from_parsing(
    parsing_mask: torch.Tensor,
    background_label: int = BACKGROUND_LABEL_V6,
) -> torch.Tensor:
    return labels_to_mask(parsing_mask, (background_label,))


def non_hair_non_background_mask_from_parsing(
    parsing_mask: torch.Tensor,
    background_label: int = BACKGROUND_LABEL_V6,
    hair_label: int = HAIR_LABEL_V6,
) -> torch.Tensor:
    parsing_mask = ensure_mask_4d(parsing_mask)
    return torch.where(
        (parsing_mask != background_label) & (parsing_mask != hair_label),
        torch.ones_like(parsing_mask, dtype=torch.float32),
        torch.zeros_like(parsing_mask, dtype=torch.float32),
    ).float()


def semantic_sample_mask_from_parsing(
    parsing_mask: torch.Tensor,
    skin_labels: tuple[int, ...] = SKIN_LABELS_V6,
    background_label: int = BACKGROUND_LABEL_V6,
) -> torch.Tensor:
    sample_mask = labels_to_mask(parsing_mask, skin_labels)
    sample_mask = torch.where(background_mask_from_parsing(parsing_mask, background_label) > 0.5, torch.ones_like(sample_mask), sample_mask)
    return sample_mask.float()


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _gaussian_kernel_1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords.pow(2)) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp(min=1e-8)
    return kernel


def gaussian_blur_mask(mask: torch.Tensor, kernel_size: int, sigma: float | None = None) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        raise ValueError("gaussian_blur_mask expects an odd kernel_size.")

    if sigma is None or sigma <= 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    kernel_1d = _gaussian_kernel_1d(kernel_size, sigma, device=mask.device, dtype=mask.dtype)
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1)
    channels = mask.shape[1]
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)

    blurred = F.conv2d(mask, kernel_x, padding=(0, kernel_size // 2), groups=channels)
    blurred = F.conv2d(blurred, kernel_y, padding=(kernel_size // 2, 0), groups=channels)
    return blurred.clamp(0, 1)


def build_soft_difference_mask(
    diff_mask: torch.Tensor,
    dilate_radius: int = 5,
    blur_kernel_size: int = 11,
    blur_sigma: float | None = None,
) -> torch.Tensor:
    diff_mask = ensure_mask_4d(diff_mask).float()
    soft = dilate_mask(diff_mask, dilate_radius)
    soft = gaussian_blur_mask(soft, blur_kernel_size, blur_sigma)
    return soft.clamp(0, 1)


def build_difference_masks(
    source_hair_mask: torch.Tensor,
    target_hair_mask: torch.Tensor,
    source_parsing: torch.Tensor | None = None,
    target_parsing: torch.Tensor | None = None,
    diff_dilate_radius: int = 5,
    diff_blur_kernel_size: int = 11,
    diff_blur_sigma: float | None = None,
) -> dict[str, torch.Tensor]:
    h_source = ensure_mask_4d(source_hair_mask).float().clamp(0, 1)
    h_align = ensure_mask_4d(target_hair_mask).float().clamp(0, 1)

    m_diff = (h_source - h_align).clamp(min=0.0, max=1.0)
    m_valid = ((1.0 - h_source) * (1.0 - h_align)).clamp(0, 1)
    if source_parsing is None:
        m_sample_face = torch.zeros_like(m_valid)
        m_sample_ear = torch.zeros_like(m_valid)
        m_sample_body = torch.zeros_like(m_valid)
        m_sample_skin = torch.zeros_like(m_valid)
        m_sample_background = m_valid.clone()
        m_sample = m_valid.clone()
    else:
        m_sample_face = face_surface_mask_from_parsing(source_parsing)
        m_sample_ear = ear_surface_mask_from_parsing(source_parsing)
        m_sample_body = body_surface_mask_from_parsing(source_parsing)
        m_sample_skin = (m_sample_face + m_sample_ear).clamp(0, 1)
        m_sample_background = background_mask_from_parsing(source_parsing)
        m_sample = (m_sample_face + m_sample_ear + m_sample_body + m_sample_background).clamp(0, 1)

    route_parsing = target_parsing if target_parsing is not None else source_parsing
    if route_parsing is None:
        m_route_face = torch.zeros_like(m_valid)
        m_route_ear = torch.zeros_like(m_valid)
        m_route_body = torch.zeros_like(m_valid)
        m_route_skin = torch.zeros_like(m_valid)
        m_route_background = torch.zeros_like(m_valid)
        m_route_other = m_valid.clone()
    else:
        m_route_face = face_surface_mask_from_parsing(route_parsing)
        m_route_ear = ear_surface_mask_from_parsing(route_parsing)
        m_route_body = body_surface_mask_from_parsing(route_parsing)
        m_route_skin = (m_route_face + m_route_ear).clamp(0, 1)
        m_route_background = background_mask_from_parsing(route_parsing)
        route_known = (m_route_face + m_route_ear + m_route_body + m_route_background).clamp(0, 1)
        m_route_other = (
            blocked_detail_mask_from_parsing(route_parsing)
            + non_hair_non_background_mask_from_parsing(route_parsing) * (1.0 - route_known)
        ).clamp(0, 1)

    m_diff_soft = build_soft_difference_mask(
        m_diff,
        dilate_radius=diff_dilate_radius,
        blur_kernel_size=diff_blur_kernel_size,
        blur_sigma=diff_blur_sigma,
    )

    return {
        "M_diff": m_diff,
        "M_valid": m_valid,
        "M_sample": m_sample,
        "M_sample_face": m_sample_face,
        "M_sample_ear": m_sample_ear,
        "M_sample_body": m_sample_body,
        "M_sample_skin": m_sample_skin,
        "M_sample_background": m_sample_background,
        "M_route_face": m_route_face,
        "M_route_ear": m_route_ear,
        "M_route_body": m_route_body,
        "M_route_skin": m_route_skin,
        "M_route_background": m_route_background,
        "M_route_other": m_route_other,
        "M_diff_soft": m_diff_soft,
        "H_source": h_source,
        "H_align": h_align,
    }
