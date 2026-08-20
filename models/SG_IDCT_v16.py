from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _as_4d_mask(mask: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1) if mask.shape[0] != 1 else mask.unsqueeze(0)
    elif mask.dim() != 4:
        raise ValueError(f"Expected a 2D/3D/4D mask tensor, got shape={tuple(mask.shape)}")

    mask = mask.float().clamp(0, 1)
    if size is not None and mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="nearest")
    return mask


def _as_rgb01(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.min() < -0.05:
        image = (image + 1.0) * 0.5
    return image.float().clamp(0, 1)


def _srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    return torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(min=0)
    # torch.where evaluates both branches. Fractional pow has an infinite
    # derivative at zero, which otherwise turns the unused branch into NaN
    # gradients for clipped RGB channels.
    nonlinear = 1.055 * image.clamp_min(1e-8).pow(1.0 / 2.4) - 0.055
    return torch.where(image <= 0.0031308, 12.92 * image, nonlinear)


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0, 1] to CIE Lab. Shape: [B, 3, H, W]."""
    image = _as_rgb01(image)
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
    """Convert CIE Lab to RGB in [0, 1]. Shape: [B, 3, H, W]."""
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


def _gaussian_kernel1d(radius: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords.pow(2)) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp(min=1e-8)


def gaussian_blur2d(image: torch.Tensor, radius: int = 5, sigma: float | None = None) -> torch.Tensor:
    if radius <= 0:
        return image
    sigma = max(float(radius) / 2.0, 1e-4) if sigma is None else float(sigma)
    kernel_1d = _gaussian_kernel1d(radius, sigma, image.device, image.dtype)
    channels = image.shape[1]

    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    image = F.pad(image, (radius, radius, 0, 0), mode="reflect")
    image = F.conv2d(image, kernel_x, groups=channels)
    image = F.pad(image, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(image, kernel_y, groups=channels)


def _erode(mask: torch.Tensor, width: int = 1) -> torch.Tensor:
    if width <= 0:
        return mask
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=2 * width + 1, stride=1, padding=width)


def _dilate(mask: torch.Tensor, width: int = 1) -> torch.Tensor:
    if width <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * width + 1, stride=1, padding=width)


def distance_transform_soft(mask: torch.Tensor, tau: float = 8.0, iterations: int = 8) -> torch.Tensor:
    """Approximate a soft inner-distance map with repeated erosions."""
    mask = _as_4d_mask(mask)
    current = mask
    accum = torch.zeros_like(mask)
    for _ in range(max(int(iterations), 1)):
        current = _erode(current, 1)
        accum = accum + current
    soft = (accum / max(float(iterations), 1.0)).clamp(0, 1)
    return torch.sigmoid((soft - 0.5) * max(float(tau), 1e-4)).clamp(0, 1)


def guided_filter(mask: torch.Tensor, radius: int = 5, eps: float = 0.01) -> torch.Tensor:
    """First-version guided alpha: Gaussian smoothing with mask-range safety."""
    del eps
    mask = _as_4d_mask(mask)
    return gaussian_blur2d(mask, radius=radius, sigma=max(radius / 2.0, 1e-4)).clamp(0, 1)


def _safe_stats_mask(mask: torch.Tensor, fallback: torch.Tensor | None = None, min_pixels: float = 16.0) -> torch.Tensor:
    mask = mask.float().clamp(0, 1)
    if fallback is None:
        fallback = torch.ones_like(mask)
    else:
        fallback = fallback.float().clamp(0, 1)

    valid = mask.sum(dim=(-2, -1), keepdim=True) >= min_pixels
    fallback_valid = fallback.sum(dim=(-2, -1), keepdim=True) >= min_pixels
    all_ones = torch.ones_like(mask)
    return torch.where(valid, mask, torch.where(fallback_valid, fallback, all_ones))


def masked_mean_std(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _as_4d_mask(mask, size=features.shape[-2:])
    if mask.shape[1] == 1 and features.shape[1] != 1:
        mask = mask.expand(-1, features.shape[1], -1, -1)

    denom = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
    mean = (features * mask).sum(dim=(-2, -1), keepdim=True) / denom
    var = ((features - mean).pow(2) * mask).sum(dim=(-2, -1), keepdim=True) / denom
    return mean, torch.sqrt(var + 1e-6)


def _safe_norm(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(features.pow(2).sum(dim=1, keepdim=True) + eps)


def _normalize_luma_by_mask(luma: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean, std = masked_mean_std(luma, mask)
    low = mean - 1.75 * std
    high = mean + 1.75 * std
    return ((luma - low) / (high - low).clamp(min=1e-4)).clamp(0, 1)


def _conditional_ab_by_luma(
    target_luma_norm: torch.Tensor,
    ref_luma_norm: torch.Tensor,
    ref_ab: torch.Tensor,
    ref_mask: torch.Tensor,
    bins: int = 9,
) -> torch.Tensor:
    bins = max(int(bins), 3)
    centers = torch.linspace(0.0, 1.0, steps=bins, device=ref_ab.device, dtype=ref_ab.dtype).view(1, bins, 1, 1)
    sigma = 0.55 / max(bins - 1, 1)

    ref_mask = _as_4d_mask(ref_mask, size=ref_ab.shape[-2:])
    ref_weights = torch.exp(-0.5 * ((ref_luma_norm - centers) / sigma).pow(2)) * ref_mask
    ref_weights_5d = ref_weights.unsqueeze(2)
    ref_denom = ref_weights_5d.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    bin_ab = (ref_ab.unsqueeze(1) * ref_weights_5d).sum(dim=(-2, -1), keepdim=True) / ref_denom

    global_ab = masked_mean_std(ref_ab, ref_mask)[0].unsqueeze(1)
    valid = ref_denom > 8.0
    bin_ab = torch.where(valid, bin_ab, global_ab)

    target_weights = torch.exp(-0.5 * ((target_luma_norm - centers) / sigma).pow(2)).unsqueeze(2)
    target_denom = target_weights.sum(dim=1).clamp(min=1e-6)
    return (bin_ab * target_weights).sum(dim=1) / target_denom


class SG_IDCT_v16(nn.Module):
    """
    SATD-Guided Illumination-Decoupled Color Transfer.

    The module transfers chroma only inside safe hair regions and hard-locks
    remove/face/neck/ear regions to the SATD-cleaned image.
    """

    def __init__(
        self,
        beta: float = 0.05,
        chroma_strength: float = 0.90,
        chroma_std_strength: float = 0.0,
        chroma_delta_limit: float = 24.0,
        texture_strength: float = 0.90,
        luma_bins: int = 9,
        alpha_strength: float = 1.05,
        alpha_dilate: int = 6,
        alpha_blur_radius: int = 7,
        learnable: bool = False,
        use_gamut_map: bool = False,
    ):
        super().__init__()
        if learnable:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
            self.chroma_strength = nn.Parameter(torch.tensor(float(chroma_strength)))
            self.chroma_std_strength = nn.Parameter(torch.tensor(float(chroma_std_strength)))
            self.chroma_delta_limit = nn.Parameter(torch.tensor(float(chroma_delta_limit)))
            self.texture_strength = nn.Parameter(torch.tensor(float(texture_strength)))
            self.alpha_strength = nn.Parameter(torch.tensor(float(alpha_strength)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))
            self.register_buffer("chroma_strength", torch.tensor(float(chroma_strength)))
            self.register_buffer("chroma_std_strength", torch.tensor(float(chroma_std_strength)))
            self.register_buffer("chroma_delta_limit", torch.tensor(float(chroma_delta_limit)))
            self.register_buffer("texture_strength", torch.tensor(float(texture_strength)))
            self.register_buffer("alpha_strength", torch.tensor(float(alpha_strength)))

        self.alpha_dilate = int(alpha_dilate)
        self.alpha_blur_radius = int(alpha_blur_radius)
        self.luma_bins = int(luma_bins)
        self.use_gamut_map = use_gamut_map
        self.gamut_map = nn.Conv2d(3, 3, kernel_size=1, bias=True)
        self._init_identity_gamut_map()

    def _init_identity_gamut_map(self):
        with torch.no_grad():
            self.gamut_map.weight.zero_()
            for idx in range(3):
                self.gamut_map.weight[idx, idx, 0, 0] = 1.0
            self.gamut_map.bias.zero_()

    @staticmethod
    def build_masks(
        H_align: torch.Tensor,
        M_remove: torch.Tensor,
        M_face: torch.Tensor,
        M_neck: torch.Tensor,
        M_ear: torch.Tensor,
        size: tuple[int, int],
        alpha_dilate: int = 3,
        alpha_blur_radius: int = 7,
    ) -> dict[str, torch.Tensor]:
        H_align = _as_4d_mask(H_align, size=size)
        M_remove = _as_4d_mask(M_remove, size=size)
        M_face = _as_4d_mask(M_face, size=size)
        M_neck = _as_4d_mask(M_neck, size=size)
        M_ear = _as_4d_mask(M_ear, size=size)

        # M_remove is old source hair to clean. Only lock it outside the final
        # target hair; otherwise it punches holes into the color-transfer area.
        M_remove_lock = (M_remove * (1.0 - H_align)).clamp(0, 1)
        M_hard = (M_remove_lock + M_face + M_neck + M_ear).clamp(0, 1)
        M_safe = (H_align * (1.0 - M_face) * (1.0 - M_neck) * (1.0 - M_ear)).clamp(0, 1)
        M_core = distance_transform_soft(M_safe)
        hair_support = _dilate(H_align, max(alpha_dilate, 1))
        alpha_seed = _dilate((0.72 * M_safe + 0.28 * M_core).clamp(0, 1), alpha_dilate)
        A_hair = guided_filter(alpha_seed, radius=alpha_blur_radius)
        A_hair = (A_hair * hair_support * (1.0 - M_hard)).clamp(0, 1)
        return {
            "H_align": H_align,
            "M_remove": M_remove,
            "M_remove_lock": M_remove_lock,
            "M_face": M_face,
            "M_neck": M_neck,
            "M_ear": M_ear,
            "M_hard": M_hard,
            "M_safe": M_safe,
            "M_core": M_core,
            "hair_support": hair_support,
            "A_hair": A_hair,
        }

    def forward(
        self,
        I_satd: torch.Tensor,
        I_color: torch.Tensor,
        H_align: torch.Tensor,
        M_remove: torch.Tensor,
        M_face: torch.Tensor,
        M_neck: torch.Tensor,
        M_ear: torch.Tensor,
        H_color: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        I_satd = _as_rgb01(I_satd)
        I_color = _as_rgb01(I_color)
        size = I_satd.shape[-2:]
        if I_color.shape[-2:] != size:
            I_color = F.interpolate(I_color, size=size, mode="bicubic", align_corners=False).clamp(0, 1)

        masks = self.build_masks(
            H_align,
            M_remove,
            M_face,
            M_neck,
            M_ear,
            size,
            alpha_dilate=self.alpha_dilate,
            alpha_blur_radius=self.alpha_blur_radius,
        )
        H_color = masks["H_align"] if H_color is None else _as_4d_mask(H_color, size=size)
        satd_stats_mask = _safe_stats_mask(masks["M_safe"], fallback=masks["H_align"])
        color_stats_mask = _safe_stats_mask(H_color, fallback=torch.ones_like(H_color))

        lab_satd = rgb_to_lab(I_satd)
        lab_color = rgb_to_lab(I_color)

        L_satd = lab_satd[:, 0:1]
        ab_satd = lab_satd[:, 1:3]
        L_color = lab_color[:, 0:1]
        ab_color = lab_color[:, 1:3]

        mu_L_satd, sigma_L_satd = masked_mean_std(L_satd, satd_stats_mask)
        mu_L_color, sigma_L_color = masked_mean_std(L_color, color_stats_mask)
        mu_ab_satd, sigma_ab_satd = masked_mean_std(ab_satd, satd_stats_mask)
        mu_ab_color, sigma_ab_color = masked_mean_std(ab_color, color_stats_mask)

        beta = self.beta.clamp(0.0, 1.0)
        chroma_strength = self.chroma_strength.clamp(0.0, 2.0)
        chroma_std_strength = self.chroma_std_strength.clamp(0.0, 1.0)
        chroma_delta_limit = self.chroma_delta_limit.clamp(2.0, 45.0)
        texture_strength = self.texture_strength.clamp(0.0, 1.0)
        alpha_strength = self.alpha_strength.clamp(0.0, 1.5)

        L_match = (L_satd - mu_L_satd) * (sigma_L_color / sigma_L_satd.clamp(min=1e-4)) + mu_L_color
        L_out = L_satd + beta * (L_match - L_satd)

        target_dir = mu_ab_color / _safe_norm(mu_ab_color)
        target_chroma_mag = _safe_norm(mu_ab_color)

        luma_z = (L_satd - mu_L_satd) / sigma_L_satd.clamp(min=1e-4)
        luma_texture = torch.sigmoid(0.85 * luma_z)
        luma_texture = (0.35 + 0.85 * luma_texture).clamp(0.25, 1.10)
        target_textured = target_dir * target_chroma_mag * luma_texture

        satd_luma_norm = _normalize_luma_by_mask(L_satd, satd_stats_mask)
        color_luma_norm = _normalize_luma_by_mask(L_color, color_stats_mask)
        target_binned = _conditional_ab_by_luma(
            satd_luma_norm,
            color_luma_norm,
            ab_color,
            color_stats_mask,
            bins=self.luma_bins,
        )

        ab_target = texture_strength * target_binned + (1.0 - texture_strength) * target_textured
        ab_match = (ab_satd - mu_ab_satd) * (sigma_ab_color / sigma_ab_satd.clamp(min=1e-4)) + mu_ab_color
        ab_stats = ab_target + chroma_std_strength * (ab_match - ab_target)
        ab_delta = chroma_strength * (ab_stats - ab_satd)
        ab_delta = ab_delta.clamp(-chroma_delta_limit, chroma_delta_limit)
        ab_out = ab_satd + ab_delta

        I_trans = lab_to_rgb(torch.cat([L_out, ab_out], dim=1)).clamp(0, 1)
        if self.use_gamut_map:
            I_trans = self.gamut_map(I_trans).clamp(0, 1)

        A_hair = (masks["A_hair"] * alpha_strength).clamp(0, 1)
        I_ct = (A_hair * I_trans + (1.0 - A_hair) * I_satd).clamp(0, 1)
        preserve = (1.0 - masks["hair_support"] + masks["M_hard"]).clamp(0, 1)
        I_ct = (I_ct * (1.0 - preserve) + I_satd * preserve).clamp(0, 1)

        if return_aux:
            aux = dict(masks)
            aux.update(
                {
                    "I_trans": I_trans,
                    "satd_stats_mask": satd_stats_mask,
                    "color_stats_mask": color_stats_mask,
                    "beta": beta.detach().view(1),
                    "chroma_strength": chroma_strength.detach().view(1),
                    "alpha_strength": alpha_strength.detach().view(1),
                    "chroma_delta_limit": chroma_delta_limit.detach().view(1),
                    "texture_strength": texture_strength.detach().view(1),
                    "luma_texture": luma_texture.detach(),
                }
            )
            return I_ct, aux
        return I_ct


SG_IDCT = SG_IDCT_v16
