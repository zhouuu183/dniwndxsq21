import torch
import torch.nn.functional as F


def _srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    return torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(min=0)
    return torch.where(image <= 0.0031308, 12.92 * image, 1.055 * image.pow(1.0 / 2.4) - 0.055)


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.min() < -0.05:
        image = (image + 1.0) * 0.5
    image = image.float().clamp(0, 1)
    rgb = _srgb_to_linear(image)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    x = x / 0.95047
    z = z / 1.08883
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, t.clamp(min=eps).pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return torch.cat([l, a, b_lab], dim=1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    if lab.dim() == 3:
        lab = lab.unsqueeze(0)
    l, a, b_lab = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b_lab / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def finv(t: torch.Tensor) -> torch.Tensor:
        t3 = t.pow(3)
        return torch.where(t3 > eps, t3, (116.0 * t - 16.0) / kappa)

    x = finv(fx) * 0.95047
    y = finv(fy)
    z = finv(fz) * 1.08883
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb = torch.cat([r, g, b], dim=1)
    return _linear_to_srgb(rgb).clamp(0, 1)


def gaussian_blur2d(image: torch.Tensor, radius: int = 5, sigma: float | None = None) -> torch.Tensor:
    if radius <= 0:
        return image
    sigma = max(float(radius) / 2.0, 1e-4) if sigma is None else float(sigma)
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(coords.pow(2)) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp(min=1e-8)
    channels = image.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    image = F.pad(image, (radius, radius, 0, 0), mode="reflect")
    image = F.conv2d(image, kernel_x, groups=channels)
    image = F.pad(image, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(image, kernel_y, groups=channels)


def masked_mean_std(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float().clamp(0, 1)
    if mask.shape[-2:] != features.shape[-2:]:
        mask = F.interpolate(mask, size=features.shape[-2:], mode="nearest")
    if mask.shape[1] == 1 and features.shape[1] != 1:
        mask = mask.expand(-1, features.shape[1], -1, -1)
    denom = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
    mean = (features * mask).sum(dim=(-2, -1), keepdim=True) / denom
    var = ((features - mean).pow(2) * mask).sum(dim=(-2, -1), keepdim=True) / denom
    return mean, torch.sqrt(var + 1e-6)


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
