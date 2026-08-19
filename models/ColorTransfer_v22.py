from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def as_4d_mask(mask: torch.Tensor, size: tuple[int, int] | None = None, mode: str = "nearest") -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Expected mask with shape [B, 1, H, W], got {tuple(mask.shape)}")
    mask = mask.float().clamp(0, 1)
    if size is not None and mask.shape[-2:] != size:
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        mask = F.interpolate(mask, size=size, mode=mode, **kwargs)
    return mask.clamp(0, 1)


def rgb01(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.detach().min() < -0.05:
        image = (image + 1.0) * 0.5
    return image.float().clamp(0, 1)


def norm_from_rgb01(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(0, 1) * 2.0 - 1.0


def gaussian_blur2d(image: torch.Tensor, radius: int = 5, sigma: float | None = None) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return image
    sigma = float(sigma) if sigma is not None else max(radius / 2.0, 1e-4)
    size = 2 * radius + 1
    coords = torch.arange(size, device=image.device, dtype=image.dtype) - radius
    kernel_1d = torch.exp(-0.5 * (coords / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
    kernel_x = kernel_1d.view(1, 1, 1, size).expand(image.size(1), 1, 1, size)
    kernel_y = kernel_1d.view(1, 1, size, 1).expand(image.size(1), 1, size, 1)
    blurred = F.conv2d(image, kernel_x, padding=(0, radius), groups=image.size(1))
    return F.conv2d(blurred, kernel_y, padding=(radius, 0), groups=image.size(1))


def erode_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask.float().clamp(0, 1)
    return 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size=2 * width + 1, stride=1, padding=width)


def dilate_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask.float().clamp(0, 1)
    return F.max_pool2d(mask.float(), kernel_size=2 * width + 1, stride=1, padding=width).clamp(0, 1)


def weighted_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = as_4d_mask(mask, value.shape[-2:])
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    return (value * mask).sum(dim=(-2, -1), keepdim=True) / mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)


def masked_mean_std(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = as_4d_mask(mask, features.shape[-2:])
    if mask.size(1) == 1 and features.size(1) != 1:
        mask = mask.expand(-1, features.size(1), -1, -1)
    denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (features * mask).sum(dim=(-2, -1), keepdim=True) / denom
    var = ((features - mean).square() * mask).sum(dim=(-2, -1), keepdim=True) / denom
    return mean, var.clamp_min(1e-8).sqrt()


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0, 1)
    return torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)


def _linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp_min(0)
    return torch.where(rgb > 0.0031308, 1.055 * rgb.pow(1.0 / 2.4) - 0.055, 12.92 * rgb)


def _lab_f(value: torch.Tensor) -> torch.Tensor:
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    return torch.where(value > epsilon, value.clamp_min(1e-8).pow(1.0 / 3.0), (kappa * value + 16.0) / 116.0)


def _lab_f_inv(value: torch.Tensor) -> torch.Tensor:
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    value3 = value.pow(3.0)
    return torch.where(value3 > epsilon, value3, (116.0 * value - 16.0) / kappa)


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    rgb = rgb01(image)
    linear = _srgb_to_linear(rgb)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    fx = _lab_f(x / 0.95047)
    fy = _lab_f(y)
    fz = _lab_f(z / 1.08883)
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.cat([l, a, bb], dim=1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    l, a, bb = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - bb / 200.0
    x = 0.95047 * _lab_f_inv(fx)
    y = _lab_f_inv(fy)
    z = 1.08883 * _lab_f_inv(fz)
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return _linear_to_srgb(torch.cat([r, g, b], dim=1)).clamp(0, 1)


def rgb_saturation(image: torch.Tensor) -> torch.Tensor:
    max_v = image.max(dim=1, keepdim=True).values
    min_v = image.min(dim=1, keepdim=True).values
    return ((max_v - min_v) / max_v.clamp_min(1e-6)).clamp(0, 1)


def vector_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value.square().sum(dim=1, keepdim=True).clamp_min(eps).sqrt()


def build_v22_lock_mask(align_shape: dict[str, object], size: tuple[int, int] | None = None) -> torch.Tensor:
    hair = as_4d_mask(align_shape["HM_X"], size)
    delta_masks = align_shape.get("delta_masks")
    if not isinstance(delta_masks, dict):
        return torch.zeros_like(hair)

    zero = torch.zeros_like(hair)

    def get(name: str) -> torch.Tensor:
        value = delta_masks.get(name)
        if not isinstance(value, torch.Tensor):
            return zero
        return as_4d_mask(value, hair.shape[-2:])

    lock = (
        1.00 * get("M_remove_face")
        + 1.00 * get("M_face_region")
        + 0.95 * get("M_face_surface")
        + 1.00 * get("M_remove_neck")
        + 1.00 * get("M_neck_region")
        + 0.95 * get("M_ear_surface")
        + 0.80 * get("M_face_strand_probe")
        + 0.45 * get("M_body_preserve")
    )
    return lock.clamp(0, 1)


def conditional_ab_by_luma(
    query_luma_norm: torch.Tensor,
    ref_luma_norm: torch.Tensor,
    ref_ab: torch.Tensor,
    ref_mask: torch.Tensor,
    bins: int,
) -> torch.Tensor:
    bins = max(int(bins), 3)
    centers = torch.linspace(0.0, 1.0, bins, device=ref_ab.device, dtype=ref_ab.dtype).view(1, bins, 1, 1)
    sigma = 0.55 / max(bins - 1, 1)
    ref_mask = as_4d_mask(ref_mask, ref_ab.shape[-2:])

    weights = torch.exp(-0.5 * ((ref_luma_norm - centers) / sigma).square()) * ref_mask
    weights_5d = weights.unsqueeze(2)
    denom = weights_5d.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    bin_ab = (ref_ab.unsqueeze(1) * weights_5d).sum(dim=(-2, -1), keepdim=True) / denom
    global_ab = weighted_mean(ref_ab, ref_mask).unsqueeze(1)
    bin_ab = torch.where(denom > 8.0, bin_ab, global_ab)

    query_weights = torch.exp(-0.5 * ((query_luma_norm - centers) / sigma).square()).unsqueeze(2)
    query_denom = query_weights.sum(dim=1).clamp_min(1e-6)
    return (bin_ab * query_weights).sum(dim=1) / query_denom


class V22ColorTransfer(nn.Module):
    """
    Strict-mask SATD-based hair recoloring.

    The module preserves I_satd outside the safe hair alpha exactly. Inside the
    safe region it changes Lab ab primarily, with a bounded reflectance-aware
    L mean shift for dark-to-light reference colors.
    """

    def __init__(
        self,
        *,
        ab_strength: float = 0.58,
        ab_std_strength: float = 0.10,
        luma_boost_scale: float = 0.20,
        max_luma_shift: float = 5.0,
        min_luma_shift: float = -3.0,
        saturation_compress: float = 0.92,
        extreme_ab_scale: float = 0.74,
        luma_gap_threshold: float = 0.12,
        extreme_luma_gap: float = 0.25,
        dark_luma_threshold: float = 28.0,
        light_luma_threshold: float = 62.0,
        shadow_threshold: float = 18.0,
        highlight_threshold: float = 82.0,
        highlight_shadow_ab_scale: float = 0.42,
        bins: int = 7,
        target_erode: int = 1,
        lock_dilate: int = 1,
        alpha_blur_radius: int = 3,
        ref_erode: int = 2,
        ref_l_min: float = 5.0,
        ref_l_max: float = 95.0,
        ref_min_sat: float = 0.02,
        learnable: bool = False,
    ):
        super().__init__()
        self.bins = int(bins)
        self.target_erode = int(target_erode)
        self.lock_dilate = int(lock_dilate)
        self.alpha_blur_radius = int(alpha_blur_radius)
        self.ref_erode = int(ref_erode)
        self.ref_l_min = float(ref_l_min)
        self.ref_l_max = float(ref_l_max)
        self.ref_min_sat = float(ref_min_sat)
        self.luma_gap_threshold = float(luma_gap_threshold)
        self.extreme_luma_gap = float(extreme_luma_gap)
        self.dark_luma_threshold = float(dark_luma_threshold)
        self.light_luma_threshold = float(light_luma_threshold)
        self.shadow_threshold = float(shadow_threshold)
        self.highlight_threshold = float(highlight_threshold)

        def register(name: str, value: float):
            tensor = torch.tensor(float(value))
            if learnable:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

        register("ab_strength", ab_strength)
        register("ab_std_strength", ab_std_strength)
        register("luma_boost_scale", luma_boost_scale)
        register("max_luma_shift", max_luma_shift)
        register("min_luma_shift", min_luma_shift)
        register("saturation_compress", saturation_compress)
        register("extreme_ab_scale", extreme_ab_scale)
        register("highlight_shadow_ab_scale", highlight_shadow_ab_scale)

    def build_masks(
        self,
        target_hair_mask: torch.Tensor,
        lock_mask: torch.Tensor,
        size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        hair = as_4d_mask(target_hair_mask, size)
        lock = as_4d_mask(lock_mask, size)
        lock_guard = dilate_mask(lock, self.lock_dilate)
        allowed = (hair * (1.0 - lock_guard)).clamp(0, 1)
        core = erode_mask(allowed, self.target_erode) * allowed
        alpha = gaussian_blur2d(allowed, radius=self.alpha_blur_radius).clamp(0, 1)
        alpha = (alpha * allowed * (1.0 - lock_guard)).clamp(0, 1)
        return {"hair": hair, "lock": lock, "lock_guard": lock_guard, "allowed": allowed, "core": core, "alpha": alpha}

    def build_reference_mask(self, reference_rgb: torch.Tensor, reference_hair_mask: torch.Tensor) -> torch.Tensor:
        ref_mask = as_4d_mask(reference_hair_mask, reference_rgb.shape[-2:])
        ref_core = erode_mask(ref_mask, self.ref_erode)
        ref_lab = rgb_to_lab(reference_rgb)
        ref_l = ref_lab[:, 0:1]
        ref_sat = rgb_saturation(reference_rgb)
        robust = (
            ref_core
            * (ref_l >= self.ref_l_min).float()
            * (ref_l <= self.ref_l_max).float()
            * (ref_sat >= self.ref_min_sat).float()
        )
        core_valid = ref_core.sum(dim=(-2, -1), keepdim=True) >= 16.0
        fallback = torch.where(core_valid, ref_core, ref_mask)
        valid = robust.sum(dim=(-2, -1), keepdim=True) >= 16.0
        return torch.where(valid, robust, fallback).clamp(0, 1)

    def forward(
        self,
        i_satd: torch.Tensor,
        i_color: torch.Tensor,
        target_hair_mask: torch.Tensor,
        reference_hair_mask: torch.Tensor,
        lock_mask: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        satd_rgb = rgb01(i_satd)
        color_rgb = rgb01(i_color)
        size = satd_rgb.shape[-2:]
        if color_rgb.shape[-2:] != size:
            color_rgb = F.interpolate(color_rgb, size=size, mode="bicubic", align_corners=False).clamp(0, 1)

        masks = self.build_masks(target_hair_mask, lock_mask, size)
        ref_mask = self.build_reference_mask(color_rgb, reference_hair_mask)
        valid_sample = (
            (masks["core"].sum(dim=(-2, -1), keepdim=True) >= 4.0)
            & (ref_mask.sum(dim=(-2, -1), keepdim=True) >= 4.0)
        ).float()
        if valid_sample.sum().item() < 1.0:
            aux = dict(masks)
            aux["ref_mask"] = ref_mask
            return (satd_rgb, aux) if return_aux else satd_rgb

        satd_lab = rgb_to_lab(satd_rgb)
        color_lab = rgb_to_lab(color_rgb)
        satd_l = satd_lab[:, 0:1]
        satd_ab = satd_lab[:, 1:3]
        ref_l = color_lab[:, 0:1]
        ref_ab = color_lab[:, 1:3]

        core = masks["core"]
        target_l_mean, target_l_std = masked_mean_std(satd_l, core)
        ref_l_mean, ref_l_std = masked_mean_std(ref_l, ref_mask)
        target_ab_mean, target_ab_std = masked_mean_std(satd_ab, core)
        ref_ab_mean, ref_ab_std = masked_mean_std(ref_ab, ref_mask)

        luma_gap = (ref_l_mean - target_l_mean) / 100.0
        extreme = ((luma_gap >= self.extreme_luma_gap) | ((target_l_mean < self.dark_luma_threshold) & (ref_l_mean > self.light_luma_threshold))).float()

        ab_strength = self.ab_strength.clamp(0.0, 0.85)
        ab_strength = ab_strength * (1.0 - extreme + extreme * self.extreme_ab_scale.clamp(0.65, 1.0))
        local_extreme = ((satd_l < self.shadow_threshold) | (satd_l > self.highlight_threshold)).float()
        local_ab_scale = 1.0 - local_extreme * (1.0 - self.highlight_shadow_ab_scale.clamp(0.35, 1.0))

        mean_shift_ab = ref_ab_mean - target_ab_mean
        target_centered_ab = satd_ab - target_ab_mean
        target_chroma = vector_norm(target_ab_mean)
        ref_chroma = vector_norm(ref_ab_mean)
        chroma_ratio = (ref_chroma / target_chroma).clamp(0.70, 1.22)
        std_ratio = (ref_ab_std.mean(dim=1, keepdim=True) / target_ab_std.mean(dim=1, keepdim=True).clamp_min(1e-4)).clamp(0.78, 1.18)
        texture_scale = (0.70 * chroma_ratio + 0.30 * std_ratio).clamp(0.72, 1.18)
        target_like_ab = ref_ab_mean + target_centered_ab * texture_scale
        target_like_ab = (
            0.82 * target_like_ab
            + 0.18 * (satd_ab + mean_shift_ab)
        )
        raw_delta_ab = target_like_ab - satd_ab
        max_delta = torch.where(extreme > 0.5, torch.full_like(ref_chroma, 14.0), torch.full_like(ref_chroma, 20.0))
        delta_norm = vector_norm(raw_delta_ab)
        raw_delta_ab = raw_delta_ab * (max_delta / delta_norm).clamp_max(1.0)
        out_ab = satd_ab + local_ab_scale * ab_strength * raw_delta_ab

        raw_positive_shift = (ref_l_mean - target_l_mean) * self.luma_boost_scale.clamp(0.0, 0.9)
        positive_shift = torch.minimum(
            raw_positive_shift.clamp_min(0.0),
            self.max_luma_shift.clamp(0.0, 24.0).to(raw_positive_shift.device, raw_positive_shift.dtype),
        )
        raw_negative_shift = (ref_l_mean - target_l_mean) * 0.25
        negative_shift = torch.maximum(
            raw_negative_shift.clamp_max(0.0),
            self.min_luma_shift.clamp(-16.0, 0.0).to(raw_negative_shift.device, raw_negative_shift.dtype),
        )
        use_boost = (luma_gap > self.luma_gap_threshold).float()
        luma_shift = use_boost * positive_shift + (1.0 - use_boost) * negative_shift
        midtone_weight = (1.0 - ((satd_l - 50.0).abs() / 58.0).clamp(0, 1).pow(1.6)).clamp(0.25, 1.0)
        out_l = (satd_l + luma_shift * midtone_weight).clamp(0, 100)

        out_ab = out_ab * torch.where(
            use_boost > 0.5,
            self.saturation_compress.clamp(0.85, 1.0),
            torch.ones_like(self.saturation_compress),
        )
        transferred = lab_to_rgb(torch.cat([out_l, out_ab], dim=1)).clamp(0, 1)
        alpha = masks["alpha"] * valid_sample
        out = (transferred * alpha + satd_rgb * (1.0 - alpha)).clamp(0, 1)
        out = (out * (1.0 - masks["lock_guard"]) + satd_rgb * masks["lock_guard"]).clamp(0, 1)

        if return_aux:
            aux = dict(masks)
            aux.update(
                {
                    "ref_mask": ref_mask,
                    "transferred": transferred,
                    "target_l_mean": target_l_mean.detach(),
                    "ref_l_mean": ref_l_mean.detach(),
                    "luma_gap": luma_gap.detach(),
                    "luma_shift": luma_shift.detach(),
                    "ab_strength_effective": ab_strength.detach(),
                    "extreme": extreme.detach(),
                }
            )
            return out, aux
        return out
