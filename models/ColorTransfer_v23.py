from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def as_4d_mask(mask: torch.Tensor | None, size: tuple[int, int] | None = None, mode: str = "nearest") -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Expected mask shape [B, 1, H, W], got {tuple(mask.shape)}")
    mask = mask.float().clamp(0, 1)
    if size is not None and mask.shape[-2:] != size:
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        mask = F.interpolate(mask, size=size, mode=mode, **kwargs)
    return mask.clamp(0, 1)


def rgb01(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.float()
    if image.detach().min() < -0.05:
        image = (image + 1.0) * 0.5
    return image.clamp(0, 1)


def norm_from_rgb01(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(0, 1) * 2.0 - 1.0


def erode_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask.float().clamp(0, 1)
    return 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size=2 * width + 1, stride=1, padding=width)


def dilate_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask.float().clamp(0, 1)
    return F.max_pool2d(mask.float(), kernel_size=2 * width + 1, stride=1, padding=width).clamp(0, 1)


def gaussian_blur2d(image: torch.Tensor, radius: int, sigma: float | None = None) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return image
    sigma = float(sigma) if sigma is not None else max(radius / 2.0, 1e-4)
    size = 2 * radius + 1
    coords = torch.arange(size, device=image.device, dtype=image.dtype) - radius
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    kernel_x = kernel.view(1, 1, 1, size).expand(image.size(1), 1, 1, size)
    kernel_y = kernel.view(1, 1, size, 1).expand(image.size(1), 1, size, 1)
    blurred = F.conv2d(image, kernel_x, padding=(0, radius), groups=image.size(1))
    return F.conv2d(blurred, kernel_y, padding=(radius, 0), groups=image.size(1))


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
    luma = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return torch.cat([luma, a, b_lab], dim=1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    luma, a, b_lab = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (luma + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b_lab / 200.0
    x = 0.95047 * _lab_f_inv(fx)
    y = _lab_f_inv(fy)
    z = 1.08883 * _lab_f_inv(fz)
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return _linear_to_srgb(torch.cat([r, g, b], dim=1)).clamp(0, 1)


def rgb_saturation(image: torch.Tensor) -> torch.Tensor:
    image = rgb01(image)
    max_v = image.max(dim=1, keepdim=True).values
    min_v = image.min(dim=1, keepdim=True).values
    return ((max_v - min_v) / max_v.clamp_min(1e-6)).clamp(0, 1)


def masked_mean_std(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = as_4d_mask(mask, value.shape[-2:])
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (value * mask).sum(dim=(-2, -1), keepdim=True) / denom
    var = ((value - mean).square() * mask).sum(dim=(-2, -1), keepdim=True) / denom
    return mean, var.clamp_min(1e-8).sqrt()


def weighted_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = as_4d_mask(mask, value.shape[-2:])
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    return (value * mask).sum(dim=(-2, -1), keepdim=True) / mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)


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


def _resize_delta_masks(delta_masks: dict[str, torch.Tensor], size: tuple[int, int]) -> dict[str, torch.Tensor]:
    resized = {}
    for key, value in delta_masks.items():
        if torch.is_tensor(value):
            resized[key] = as_4d_mask(value, size)
    return resized


def build_v23_protect_masks(align_shape: dict[str, object], size: tuple[int, int] | None = None) -> dict[str, torch.Tensor]:
    hair = as_4d_mask(align_shape["HM_X"], size)
    delta_masks = align_shape.get("delta_masks")
    if not isinstance(delta_masks, dict):
        zero = torch.zeros_like(hair)
        return {"hard_lock": zero, "soft_lock": zero}

    delta = _resize_delta_masks(delta_masks, hair.shape[-2:])
    zero = torch.zeros_like(hair)

    def get(name: str) -> torch.Tensor:
        return delta.get(name, zero)

    hard_lock = (
        1.00 * get("M_face_region")
        + 1.00 * get("M_face_surface")
        + 0.95 * get("M_ear_surface")
        + 1.00 * get("M_neck_region")
        + 1.00 * get("M_body_region")
        + 0.90 * get("M_body_preserve")
        + 1.00 * get("M_remove_face")
        + 1.00 * get("M_remove_neck")
        + 0.85 * get("M_remove_context")
        + 0.85 * get("M_remove_tail")
        + 0.70 * get("M_remove")
    ).clamp(0, 1)

    soft_lock = (
        0.85 * get("M_boundary")
        + 0.70 * get("M_remove_halo")
        + 0.55 * get("M_face_strand_probe")
        + 0.45 * get("M_context_region")
    ).clamp(0, 1)

    return {"hard_lock": hard_lock, "soft_lock": soft_lock}


class V23ColorTransfer(nn.Module):
    """
    SATD-preserving hair recolor.

    The module estimates chroma/ab transfer on a fixed working resolution,
    upsamples only the Lab delta and alpha, then applies them to the original
    SATD image. Non-alpha and hard-lock pixels are copied from SATD exactly.
    """

    def __init__(
        self,
        *,
        work_size: int = 256,
        bins: int = 7,
        ab_strength: float = 1.0,
        ab_std_strength: float = 0.12,
        main_color_strength: float = 0.18,
        chroma_floor_strength: float = 0.72,
        chroma_floor_min_ref: float = 8.0,
        luma_boost_scale: float = 0.18,
        darken_scale: float = 0.16,
        max_luma_shift: float = 8.0,
        min_luma_shift: float = -4.0,
        shadow_threshold: float = 18.0,
        highlight_threshold: float = 84.0,
        extreme_luma_chroma_scale: float = 0.68,
        extreme_luma_shift_scale: float = 0.20,
        target_erode: int = 1,
        target_band_dilate: int = 1,
        band_alpha: float = 0.45,
        soft_lock_strength: float = 0.65,
        alpha_blur_radius: int = 3,
        ref_erode: int = 2,
        ref_l_min: float = 5.0,
        ref_l_max: float = 95.0,
        ref_min_sat: float = 0.025,
        learnable: bool = False,
    ):
        super().__init__()
        self.work_size = int(work_size)
        self.bins = int(bins)
        self.chroma_floor_min_ref = float(chroma_floor_min_ref)
        self.shadow_threshold = float(shadow_threshold)
        self.highlight_threshold = float(highlight_threshold)
        self.target_erode = int(target_erode)
        self.target_band_dilate = int(target_band_dilate)
        self.alpha_blur_radius = int(alpha_blur_radius)
        self.ref_erode = int(ref_erode)
        self.ref_l_min = float(ref_l_min)
        self.ref_l_max = float(ref_l_max)
        self.ref_min_sat = float(ref_min_sat)

        def register_scalar(name: str, value: float) -> None:
            tensor = torch.tensor(float(value))
            if learnable:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

        register_scalar("ab_strength", ab_strength)
        register_scalar("ab_std_strength", ab_std_strength)
        register_scalar("main_color_strength", main_color_strength)
        register_scalar("chroma_floor_strength", chroma_floor_strength)
        register_scalar("luma_boost_scale", luma_boost_scale)
        register_scalar("darken_scale", darken_scale)
        register_scalar("max_luma_shift", max_luma_shift)
        register_scalar("min_luma_shift", min_luma_shift)
        register_scalar("extreme_luma_chroma_scale", extreme_luma_chroma_scale)
        register_scalar("extreme_luma_shift_scale", extreme_luma_shift_scale)
        register_scalar("band_alpha", band_alpha)
        register_scalar("soft_lock_strength", soft_lock_strength)

    def _work_size(self, image: torch.Tensor) -> tuple[int, int]:
        if self.work_size <= 0:
            return image.shape[-2:]
        return (self.work_size, self.work_size)

    def build_target_masks(
        self,
        target_hair_mask: torch.Tensor,
        hard_lock_mask: torch.Tensor | None,
        soft_lock_mask: torch.Tensor | None,
        size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        hair = as_4d_mask(target_hair_mask, size)
        hard_lock = as_4d_mask(hard_lock_mask, size) if hard_lock_mask is not None else torch.zeros_like(hair)
        soft_lock = as_4d_mask(soft_lock_mask, size) if soft_lock_mask is not None else torch.zeros_like(hair)
        allowed = (hair * (1.0 - hard_lock) * (1.0 - self.soft_lock_strength * soft_lock)).clamp(0, 1)
        core = erode_mask(allowed, self.target_erode) * allowed
        band = (dilate_mask(allowed, self.target_band_dilate) * hair - core).clamp(0, 1)
        alpha_seed = (core + self.band_alpha * band).clamp(0, 1)
        alpha = gaussian_blur2d(alpha_seed, self.alpha_blur_radius).clamp(0, 1)
        alpha = (alpha * hair * (1.0 - hard_lock) * (1.0 - self.soft_lock_strength * soft_lock)).clamp(0, 1)
        return {
            "hair": hair,
            "hard_lock": hard_lock,
            "soft_lock": soft_lock,
            "allowed": allowed,
            "core": core,
            "band": band,
            "alpha": alpha,
        }

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
        ).clamp(0, 1)
        core_valid = ref_core.sum(dim=(-2, -1), keepdim=True) >= 16.0
        robust_valid = robust.sum(dim=(-2, -1), keepdim=True) >= 16.0
        fallback = torch.where(core_valid, ref_core, ref_mask)
        return torch.where(robust_valid, robust, fallback).clamp(0, 1)

    def _compute_lowres_delta(
        self,
        satd_rgb: torch.Tensor,
        color_rgb: torch.Tensor,
        target_hair_mask: torch.Tensor,
        reference_hair_mask: torch.Tensor,
        hard_lock_mask: torch.Tensor | None,
        soft_lock_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        size = satd_rgb.shape[-2:]
        masks = self.build_target_masks(target_hair_mask, hard_lock_mask, soft_lock_mask, size)
        ref_mask = self.build_reference_mask(color_rgb, reference_hair_mask)
        valid = (
            (masks["core"].sum(dim=(-2, -1), keepdim=True) >= 4.0)
            & (ref_mask.sum(dim=(-2, -1), keepdim=True) >= 4.0)
        ).float()

        satd_lab = rgb_to_lab(satd_rgb)
        color_lab = rgb_to_lab(color_rgb)
        satd_l, satd_ab = satd_lab[:, 0:1], satd_lab[:, 1:3]
        ref_l, ref_ab = color_lab[:, 0:1], color_lab[:, 1:3]

        target_l_mean, target_l_std = masked_mean_std(satd_l, masks["core"])
        ref_l_mean, ref_l_std = masked_mean_std(ref_l, ref_mask)
        target_ab_mean, target_ab_std = masked_mean_std(satd_ab, masks["core"])
        ref_ab_mean, ref_ab_std = masked_mean_std(ref_ab, ref_mask)

        target_l_norm = ((satd_l - (target_l_mean - 1.75 * target_l_std)) / (3.5 * target_l_std).clamp_min(1e-4)).clamp(0, 1)
        ref_l_norm = ((ref_l - (ref_l_mean - 1.75 * ref_l_std)) / (3.5 * ref_l_std).clamp_min(1e-4)).clamp(0, 1)
        target_cond_ab = conditional_ab_by_luma(target_l_norm, target_l_norm, satd_ab, masks["core"], self.bins)
        ref_cond_ab = conditional_ab_by_luma(target_l_norm, ref_l_norm, ref_ab, ref_mask, self.bins)

        delta_ab = ref_cond_ab - target_cond_ab
        stats_ab = (satd_ab - target_ab_mean) * (ref_ab_std / target_ab_std.clamp_min(1e-4)) + ref_ab_mean
        stats_delta = stats_ab - satd_ab
        main_delta = ref_ab_mean - target_ab_mean

        local_extreme = ((satd_l < self.shadow_threshold) | (satd_l > self.highlight_threshold)).float()
        chroma_scale = 1.0 - local_extreme * (1.0 - self.extreme_luma_chroma_scale)
        out_ab = satd_ab + chroma_scale * (
            self.ab_strength * delta_ab
            + self.ab_std_strength * stats_delta
            + self.main_color_strength * main_delta
        )

        ref_main_ab = weighted_mean(ref_ab, ref_mask)
        ref_chroma = ref_main_ab.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        ref_dir = ref_main_ab / ref_chroma
        desired_projection = (ref_chroma * self.chroma_floor_strength).clamp_min(0.0)
        out_projection = (out_ab * ref_dir).sum(dim=1, keepdim=True)
        chroma_floor_active = (ref_chroma >= self.chroma_floor_min_ref).float()
        projection_deficit = (desired_projection - out_projection).clamp_min(0.0)
        out_ab = out_ab + chroma_floor_active * projection_deficit * ref_dir

        luma_gap = ref_l_mean - target_l_mean
        positive_shift = torch.relu(luma_gap * self.luma_boost_scale)
        positive_shift = torch.minimum(positive_shift, self.max_luma_shift.clamp_min(0.0))
        negative_shift = torch.minimum(luma_gap * self.darken_scale, torch.zeros_like(luma_gap))
        negative_shift = torch.maximum(negative_shift, self.min_luma_shift.clamp_max(0.0))
        luma_shift = torch.where(luma_gap > 0, positive_shift, negative_shift)
        midtone_weight = (1.0 - ((satd_l - 50.0).abs() / 58.0).clamp(0, 1).pow(1.6)).clamp(0.25, 1.0)
        luma_scale = 1.0 - local_extreme * (1.0 - self.extreme_luma_shift_scale)
        out_l = (satd_l + luma_shift * midtone_weight * luma_scale).clamp(0, 100)

        delta_lab = torch.cat([out_l - satd_l, out_ab - satd_ab], dim=1) * valid
        alpha = masks["alpha"] * valid
        aux = dict(masks)
        aux.update(
            {
                "ref_mask": ref_mask,
                "delta_lab": delta_lab,
                "target_l_mean": target_l_mean.detach(),
                "ref_l_mean": ref_l_mean.detach(),
                "ref_ab_mean": ref_ab_mean.detach(),
                "target_ab_mean": target_ab_mean.detach(),
                "alpha": alpha,
                "valid": valid,
            }
        )
        return delta_lab, alpha, aux

    def forward(
        self,
        i_satd: torch.Tensor,
        i_color: torch.Tensor,
        target_hair_mask: torch.Tensor,
        reference_hair_mask: torch.Tensor,
        hard_lock_mask: torch.Tensor | None = None,
        soft_lock_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        satd_rgb = rgb01(i_satd)
        color_rgb = rgb01(i_color)
        full_size = satd_rgb.shape[-2:]
        work_size = self._work_size(satd_rgb)

        satd_low = satd_rgb
        if satd_low.shape[-2:] != work_size:
            satd_low = F.interpolate(satd_low, size=work_size, mode="bicubic", align_corners=False).clamp(0, 1)
        if color_rgb.shape[-2:] != work_size:
            color_low = F.interpolate(color_rgb, size=work_size, mode="bicubic", align_corners=False).clamp(0, 1)
        else:
            color_low = color_rgb

        delta_lab_low, alpha_low, aux = self._compute_lowres_delta(
            satd_low,
            color_low,
            target_hair_mask,
            reference_hair_mask,
            hard_lock_mask,
            soft_lock_mask,
        )

        if full_size != work_size:
            delta_lab = F.interpolate(delta_lab_low, size=full_size, mode="bilinear", align_corners=False)
            alpha = F.interpolate(alpha_low, size=full_size, mode="bilinear", align_corners=False).clamp(0, 1)
        else:
            delta_lab = delta_lab_low
            alpha = alpha_low

        satd_lab = rgb_to_lab(satd_rgb)
        transferred = lab_to_rgb(satd_lab + delta_lab).clamp(0, 1)
        hard_lock_full = as_4d_mask(hard_lock_mask, full_size) if hard_lock_mask is not None else torch.zeros_like(alpha)
        out = (transferred * alpha + satd_rgb * (1.0 - alpha)).clamp(0, 1)
        out = (out * (1.0 - hard_lock_full) + satd_rgb * hard_lock_full).clamp(0, 1)

        if return_aux:
            aux["alpha_full"] = alpha
            aux["hard_lock_full"] = hard_lock_full
            aux["transferred_full"] = transferred
            return out, aux
        return out
