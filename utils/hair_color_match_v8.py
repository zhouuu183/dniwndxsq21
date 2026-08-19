import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import gaussian_blur2d, lab_to_rgb, masked_mean_std, rgb_to_lab


def _as_bchw(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    squeezed = image.dim() == 3
    if squeezed:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.size(1) != 3:
        raise ValueError(f"Expected RGB tensor with shape [B, 3, H, W], got {tuple(image.shape)}")
    return image, squeezed


def _as_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Expected mask tensor with shape [B, 1, H, W], got {tuple(mask.shape)}")

    mask = mask.float().clamp(0, 1)
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    return mask.clamp(0, 1)


def _to_rgb01(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    was_norm = bool(image.detach().min() < -0.05)
    if was_norm:
        image = (image + 1.0) * 0.5
    return image.float().clamp(0, 1), was_norm


def _restore_range(image: torch.Tensor, was_norm: bool) -> torch.Tensor:
    image = image.clamp(0, 1)
    return image * 2.0 - 1.0 if was_norm else image


def _valid_stats_mask(mask: torch.Tensor, fallback: torch.Tensor, min_pixels: float = 16.0) -> torch.Tensor:
    mask = mask.float().clamp(0, 1)
    fallback = fallback.float().clamp(0, 1)
    valid = mask.sum(dim=(-2, -1), keepdim=True) >= min_pixels
    fallback_valid = fallback.sum(dim=(-2, -1), keepdim=True) >= min_pixels
    return torch.where(valid, mask, torch.where(fallback_valid, fallback, torch.ones_like(mask)))


def match_hair_color_lab_v8(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_hair_mask: torch.Tensor,
    reference_hair_mask: torch.Tensor,
    *,
    strength: float = 1.0,
    chroma_strength: float = 1.0,
    luma_strength: float = 1.0,
    std_strength: float = 1.0,
    alpha_blur_radius: int = 5,
) -> torch.Tensor:
    """
    Match final hair color to the reference in Lab space under the target hair mask.
    The local target hair texture is kept through z-score matching; only masked hair
    pixels are blended back into the image.
    """
    if strength <= 0:
        return image

    image, squeezed = _as_bchw(image)
    reference, _ = _as_bchw(reference)
    image_rgb, image_was_norm = _to_rgb01(image)
    reference_rgb, _ = _to_rgb01(reference)

    target_mask = _as_mask(image_hair_mask, image_rgb.shape[-2:])
    ref_mask = _as_mask(reference_hair_mask, reference_rgb.shape[-2:])
    if target_mask.sum().item() < 1.0 or ref_mask.sum().item() < 1.0:
        out = _restore_range(image_rgb, image_was_norm)
        return out[0] if squeezed else out
    target_stats_mask = _valid_stats_mask(target_mask, fallback=target_mask)
    ref_stats_mask = _valid_stats_mask(ref_mask, fallback=torch.ones_like(ref_mask))

    image_lab = rgb_to_lab(image_rgb)
    ref_lab = rgb_to_lab(reference_rgb)

    image_l = image_lab[:, 0:1]
    image_ab = image_lab[:, 1:3]
    ref_l = ref_lab[:, 0:1]
    ref_ab = ref_lab[:, 1:3]

    image_l_mean, image_l_std = masked_mean_std(image_l, target_stats_mask)
    ref_l_mean, ref_l_std = masked_mean_std(ref_l, ref_stats_mask)
    image_ab_mean, image_ab_std = masked_mean_std(image_ab, target_stats_mask)
    ref_ab_mean, ref_ab_std = masked_mean_std(ref_ab, ref_stats_mask)

    std_strength = float(std_strength)
    luma_strength = float(luma_strength)
    chroma_strength = float(chroma_strength)
    strength = float(strength)

    matched_l = (image_l - image_l_mean) * (ref_l_std / image_l_std.clamp(min=1e-4)) + ref_l_mean
    matched_ab = (image_ab - image_ab_mean) * (ref_ab_std / image_ab_std.clamp(min=1e-4)) + ref_ab_mean
    mean_only_ab = image_ab + (ref_ab_mean - image_ab_mean)
    matched_ab = mean_only_ab + std_strength * (matched_ab - mean_only_ab)

    out_l = image_l + luma_strength * (matched_l - image_l)
    out_ab = image_ab + chroma_strength * (matched_ab - image_ab)
    matched_rgb = lab_to_rgb(torch.cat([out_l, out_ab], dim=1))

    alpha = target_mask
    if alpha_blur_radius > 0:
        alpha = gaussian_blur2d(alpha, radius=int(alpha_blur_radius))
    alpha = (alpha * strength).clamp(0, 1)

    out_rgb = (matched_rgb * alpha + image_rgb * (1.0 - alpha)).clamp(0, 1)
    out = _restore_range(out_rgb, image_was_norm)
    return out[0] if squeezed else out
