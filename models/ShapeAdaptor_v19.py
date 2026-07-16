from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.shape_metrics_v19 import laplacian_edges, sobel_edges

V19_PRIOR_KEYS = (
    "source_hair_mask",
    "rotated_hair_mask",
    "coarse_target_mask",
    "soft_alpha_matte",
    "boundary_edge_map",
    "hair_detail_map",
    "boundary_band",
    "distance_transform",
    "source_face_protect",
)

V19_MASK_KEYS = (
    "source_hair_mask",
    "rotated_hair_mask",
    "coarse_target_mask",
    "pseudo_target_mask",
    "boundary_band",
    "source_face_protect",
)


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
) -> nn.Sequential:
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


class ResidualBlock_v19(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            _conv_block(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


def ensure_bchw_mask_v19(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        if mask.shape[0] == 1:
            return mask.unsqueeze(0)
        return mask.unsqueeze(1)
    return mask


def blur_mask_v19(mask: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    return F.avg_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=padding)


def dilate_mask_v19(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    if radius <= 0:
        return mask.float()
    kernel = radius * 2 + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius)


def erode_mask_v19(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    return 1.0 - dilate_mask_v19(1.0 - mask.float(), radius=radius)


def hair_mask_from_parsing_v19(parsing_mask: torch.Tensor) -> torch.Tensor:
    parsing_mask = ensure_bchw_mask_v19(parsing_mask)
    if parsing_mask.max() <= 1.0:
        return parsing_mask.float().clamp(0, 1)
    return (parsing_mask == 13).float()


def face_protect_from_parsing_v19(parsing_mask: torch.Tensor) -> torch.Tensor:
    parsing_mask = ensure_bchw_mask_v19(parsing_mask)
    if parsing_mask.max() <= 1.0:
        return parsing_mask.float().clamp(0, 1)
    skin = (parsing_mask == 1).float()
    _, _, height, _ = skin.shape
    y = torch.linspace(0.0, 1.0, height, device=parsing_mask.device, dtype=torch.float32).view(1, 1, height, 1)
    skin_rows = (skin.amax(dim=-1, keepdim=True) > 0.5).float()
    skin_top = torch.where(skin_rows > 0, y, torch.ones_like(y)).amin(dim=-2, keepdim=True)
    skin_bottom = torch.where(skin_rows > 0, y, torch.zeros_like(y)).amax(dim=-2, keepdim=True)
    forehead_cut = skin_top + 0.38 * (skin_bottom - skin_top)
    lower_face_skin = skin * (y >= forehead_cut).float()

    protect_labels = {2, 3, 4, 5, 6, 10, 11, 12}
    protect = lower_face_skin
    for label in protect_labels:
        protect = torch.maximum(protect, (parsing_mask == label).float())
    return dilate_mask_v19(protect, radius=1).clamp(0, 1)


def compute_distance_transform_v19(mask: torch.Tensor) -> torch.Tensor:
    mask = hair_mask_from_parsing_v19(mask)
    batch = []
    for mask_item in mask.detach().cpu().numpy():
        binary = (mask_item[0] > 0.5).astype(np.uint8)
        inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
        signed = inside - outside
        scale = max(float(np.abs(signed).max()), 1.0)
        batch.append(torch.from_numpy(signed / scale).float())
    return torch.stack(batch, dim=0).unsqueeze(1).to(mask.device)


def topology_regions_v19(
    coarse_target_mask: torch.Tensor,
    rotated_hair_mask: torch.Tensor,
    radius: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coarse_target_mask = hair_mask_from_parsing_v19(coarse_target_mask)
    rotated_hair_mask = hair_mask_from_parsing_v19(rotated_hair_mask)
    coarse_dilated = dilate_mask_v19(coarse_target_mask, radius=radius)
    rotated_dilated = dilate_mask_v19(rotated_hair_mask, radius=radius)
    add_region = (rotated_hair_mask * (1.0 - coarse_dilated)).clamp(0, 1)
    remove_region = (coarse_target_mask * (1.0 - rotated_dilated)).clamp(0, 1)
    topology_support = dilate_mask_v19((add_region + remove_region).clamp(0, 1), radius=max(2, radius // 2))
    return add_region, remove_region, topology_support.clamp(0, 1)


def extract_boundary_priors_v19(
    image_256: torch.Tensor,
    hair_mask_256: torch.Tensor,
    source_hair_mask_256: torch.Tensor | None = None,
    coarse_target_mask_256: torch.Tensor | None = None,
    source_face_protect_256: torch.Tensor | None = None,
    boundary_radius: int = 3,
) -> dict[str, torch.Tensor]:
    image_256 = image_256.float()
    if image_256.min() < 0:
        image_01 = ((image_256 + 1.0) / 2.0).clamp(0, 1)
    else:
        image_01 = image_256.clamp(0, 1)

    hair_mask_256 = hair_mask_from_parsing_v19(hair_mask_256)
    source_hair_mask_256 = (
        hair_mask_from_parsing_v19(source_hair_mask_256)
        if source_hair_mask_256 is not None
        else torch.zeros_like(hair_mask_256)
    )
    coarse_target_mask_256 = (
        hair_mask_from_parsing_v19(coarse_target_mask_256)
        if coarse_target_mask_256 is not None
        else hair_mask_256
    )
    if source_face_protect_256 is None:
        source_face_protect_256 = torch.zeros_like(hair_mask_256)
    else:
        source_face_protect_256 = ensure_bchw_mask_v19(source_face_protect_256).float().clamp(0, 1)

    dilated = dilate_mask_v19(hair_mask_256, radius=boundary_radius)
    eroded = erode_mask_v19(hair_mask_256, radius=boundary_radius)
    boundary_band = (dilated - eroded).clamp(0, 1)
    soft_alpha = (0.5 * hair_mask_256 + 0.5 * blur_mask_v19(hair_mask_256, kernel_size=7)).clamp(0, 1)

    rgb_blur = F.avg_pool2d(image_01, kernel_size=5, stride=1, padding=2)
    hair_detail = (image_01 - rgb_blur).abs().mean(dim=1, keepdim=True)
    hair_detail = hair_detail * (0.4 * hair_mask_256 + 0.6 * boundary_band)

    boundary_edge = 0.6 * sobel_edges(image_01) + 0.4 * laplacian_edges(image_01)
    boundary_edge = boundary_edge * (0.25 * hair_mask_256 + 0.75 * boundary_band)
    boundary_edge = boundary_edge / boundary_edge.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

    priors = {
        "source_hair_mask": source_hair_mask_256.float(),
        "rotated_hair_mask": hair_mask_256.float(),
        "coarse_target_mask": coarse_target_mask_256.float(),
        "soft_alpha_matte": soft_alpha.float(),
        "boundary_edge_map": boundary_edge.float(),
        "hair_detail_map": hair_detail.float(),
        "boundary_band": boundary_band.float(),
        "distance_transform": compute_distance_transform_v19(hair_mask_256).float(),
        "source_face_protect": source_face_protect_256.float(),
    }
    return priors


def build_pseudo_target_mask_v19(
    coarse_target_mask: torch.Tensor,
    rotated_hair_mask: torch.Tensor,
    priors_256: dict[str, torch.Tensor],
    residual_gain: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse_target_mask = hair_mask_from_parsing_v19(coarse_target_mask)
    rotated_hair_mask = hair_mask_from_parsing_v19(rotated_hair_mask)
    boundary_band = priors_256["boundary_band"].float()
    soft_alpha = priors_256["soft_alpha_matte"].float()
    detail_map = priors_256["hair_detail_map"].float()
    add_region, remove_region, topology_support = topology_regions_v19(
        coarse_target_mask=coarse_target_mask,
        rotated_hair_mask=rotated_hair_mask,
        radius=3,
    )
    fine_add_support = (
        rotated_hair_mask
        * (1.0 - coarse_target_mask)
        * (0.35 + boundary_band + detail_map).clamp(0, 1.5)
    ).clamp(0, 1)
    support = torch.maximum(boundary_band, torch.maximum(topology_support, dilate_mask_v19(fine_add_support, radius=1)))
    signed_residual = (
        0.70 * (rotated_hair_mask - coarse_target_mask)
        + 0.30 * (soft_alpha - coarse_target_mask)
    )
    signed_residual = signed_residual * (0.65 + detail_map).clamp(0, 1.5) * support
    pseudo_target = coarse_target_mask + residual_gain * signed_residual
    pseudo_target = pseudo_target + 1.05 * add_region - 0.75 * remove_region
    pseudo_target = torch.maximum(
        pseudo_target,
        add_region * (0.85 * rotated_hair_mask + 0.15 * soft_alpha).clamp(0, 1),
    )
    pseudo_target = torch.maximum(pseudo_target, fine_add_support)
    pseudo_target = torch.minimum(
        pseudo_target,
        torch.where(remove_region > 0, 0.15 * coarse_target_mask, torch.ones_like(coarse_target_mask)),
    )
    pseudo_target = pseudo_target.clamp(0, 1)
    pseudo_target = torch.maximum(pseudo_target, coarse_target_mask * (1 - 0.20 * support))
    signed_residual = (pseudo_target - coarse_target_mask).clamp(-1, 1)
    return pseudo_target, signed_residual


def merge_hair_mask_into_parsing_v19(
    base_parsing_mask: torch.Tensor,
    source_parsing_mask: torch.Tensor,
    hair_mask: torch.Tensor,
) -> torch.Tensor:
    base_parsing_mask = ensure_bchw_mask_v19(base_parsing_mask).clone()
    source_parsing_mask = ensure_bchw_mask_v19(source_parsing_mask)
    hair_mask = hair_mask_from_parsing_v19(hair_mask)
    non_hair_fill = torch.where(base_parsing_mask == 13, source_parsing_mask, base_parsing_mask)
    return torch.where(hair_mask > 0.5, torch.full_like(non_hair_fill, 13), non_hair_fill)


def compose_boundary_detail_feature_v19(
    base_feature_64: torch.Tensor,
    shape_feature_64: torch.Tensor,
    boundary_band_256: torch.Tensor,
    detail_map_256: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    band_64 = F.interpolate(boundary_band_256.float(), size=base_feature_64.shape[-2:], mode="bilinear", align_corners=False)
    detail_64 = F.interpolate(detail_map_256.float(), size=base_feature_64.shape[-2:], mode="bilinear", align_corners=False)
    gate = (band_64 * (0.5 + detail_64)).clamp(0, 1)
    return base_feature_64 + strength * gate * (shape_feature_64 - base_feature_64)


def build_edit_composite_mask_v19(
    source_hair_mask: torch.Tensor,
    target_hair_mask: torch.Tensor,
    boundary_band: torch.Tensor,
    expand_radius: int = 2,
    blur_kernel_size: int = 9,
) -> torch.Tensor:
    source_hair_mask = hair_mask_from_parsing_v19(source_hair_mask)
    target_hair_mask = ensure_bchw_mask_v19(target_hair_mask).float().clamp(0, 1)
    boundary_band = ensure_bchw_mask_v19(boundary_band).float().clamp(0, 1)
    remove_region = (source_hair_mask * (1.0 - target_hair_mask)).clamp(0, 1)
    edit_mask = torch.maximum(torch.maximum(target_hair_mask, remove_region), boundary_band)
    edit_mask = dilate_mask_v19(edit_mask, radius=expand_radius)
    edit_mask = torch.maximum(edit_mask, blur_mask_v19(edit_mask, kernel_size=blur_kernel_size))
    return edit_mask.clamp(0, 1)


def build_training_masks_v19(
    source_hair_mask: torch.Tensor,
    target_hair_mask: torch.Tensor,
    boundary_band: torch.Tensor,
    source_face_protect: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    source_hair_mask = hair_mask_from_parsing_v19(source_hair_mask)
    target_hair_mask = hair_mask_from_parsing_v19(target_hair_mask)
    boundary_band = ensure_bchw_mask_v19(boundary_band).float().clamp(0, 1)
    if source_face_protect is None:
        source_face_protect = torch.zeros_like(target_hair_mask)
    else:
        source_face_protect = ensure_bchw_mask_v19(source_face_protect).float().clamp(0, 1)

    edit_mask = target_hair_mask.clamp(0, 1)
    remove_mask = (source_hair_mask * (1.0 - target_hair_mask)).clamp(0, 1)
    keep_mask = (source_hair_mask * target_hair_mask).clamp(0, 1)
    edit_mask = torch.maximum(edit_mask, remove_mask)
    preserve_mask = (source_face_protect + (1.0 - torch.maximum(source_hair_mask, target_hair_mask))).clamp(0, 1)
    return {
        "M_edit": edit_mask,
        "M_remove": remove_mask,
        "M_keep": keep_mask,
        "M_boundary": boundary_band,
        "M_face_protect": source_face_protect,
        "M_preserve": preserve_mask,
    }


class BoundaryLayerShapeAdaptor_v19(nn.Module):
    def __init__(self, hidden_channels: int = 256):
        super().__init__()
        prior_channels = len(V19_PRIOR_KEYS)
        self.prior_128 = nn.Sequential(
            _conv_block(prior_channels, 64, stride=2),
            ResidualBlock_v19(64),
        )
        self.prior_64 = nn.Sequential(
            _conv_block(64, 96, stride=2),
            ResidualBlock_v19(96),
        )
        self.prior_32 = nn.Sequential(
            _conv_block(96, 128, stride=2),
            ResidualBlock_v19(128),
        )

        self.mask_up_64 = nn.Sequential(
            _conv_block(128 + 96, 96),
            ResidualBlock_v19(96),
        )
        self.mask_up_128 = nn.Sequential(
            _conv_block(96 + 64, 64),
            ResidualBlock_v19(64),
        )
        self.mask_up_256 = nn.Sequential(
            _conv_block(64, 48),
            ResidualBlock_v19(48),
        )
        self.mask_head = nn.Conv2d(48, 2, kernel_size=1)

        f32_in_channels = 512 * 3 + 128 + 3
        self.f32_head = nn.Sequential(
            _conv_block(f32_in_channels, hidden_channels, kernel_size=1, padding=0),
            ResidualBlock_v19(hidden_channels),
            ResidualBlock_v19(hidden_channels),
            nn.Conv2d(hidden_channels, 512, kernel_size=1),
        )

        f64_in_channels = 512 * 3 + 96 + 3
        self.f64_head = nn.Sequential(
            _conv_block(f64_in_channels, hidden_channels, kernel_size=1, padding=0),
            ResidualBlock_v19(hidden_channels),
            ResidualBlock_v19(hidden_channels),
            nn.Conv2d(hidden_channels, 512, kernel_size=1),
        )

    @staticmethod
    def _stack_priors(priors_256: dict[str, torch.Tensor], keys: Iterable[str]) -> torch.Tensor:
        return torch.cat([ensure_bchw_mask_v19(priors_256[key]).float() for key in keys], dim=1)

    def forward(
        self,
        F_base: torch.Tensor,
        F_src: torch.Tensor,
        F_shape: torch.Tensor,
        F_base64: torch.Tensor,
        F_shape64: torch.Tensor,
        priors_256: dict[str, torch.Tensor],
        F_reference: torch.Tensor | None = None,
        F_reference64: torch.Tensor | None = None,
        strength: float = 1.0,
        shape_prior_strength: float = 0.0,
        detail_strength: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        reference32 = F_shape if F_reference is None else F_reference
        reference64 = F_shape64 if F_reference64 is None else F_reference64
        prior_input = self._stack_priors(priors_256, V19_PRIOR_KEYS)
        prior_128 = self.prior_128(prior_input)
        prior_64 = self.prior_64(prior_128)
        prior_32 = self.prior_32(prior_64)

        mask_64 = self.mask_up_64(torch.cat([F.interpolate(prior_32, scale_factor=2, mode="bilinear", align_corners=False), prior_64], dim=1))
        mask_128 = self.mask_up_128(torch.cat([F.interpolate(mask_64, scale_factor=2, mode="bilinear", align_corners=False), prior_128], dim=1))
        mask_256 = self.mask_up_256(F.interpolate(mask_128, scale_factor=2, mode="bilinear", align_corners=False))
        mask_logits = self.mask_head(mask_256)
        coarse_target_mask = torch.sigmoid(mask_logits[:, 0:1])
        boundary_residual = torch.tanh(mask_logits[:, 1:2])

        boundary_band_256 = priors_256["boundary_band"].float()
        source_hair_mask_256 = ensure_bchw_mask_v19(priors_256["source_hair_mask"]).float()
        detail_map_256 = ensure_bchw_mask_v19(priors_256["hair_detail_map"]).float()
        pseudo_target_mask = priors_256.get("pseudo_target_mask")
        if pseudo_target_mask is None:
            pseudo_target_mask, _ = build_pseudo_target_mask_v19(
                coarse_target_mask=priors_256["coarse_target_mask"],
                rotated_hair_mask=priors_256["rotated_hair_mask"],
                priors_256=priors_256,
            )
        pseudo_target_mask = ensure_bchw_mask_v19(pseudo_target_mask).float()
        remove_region_256 = (source_hair_mask_256 * (1.0 - pseudo_target_mask)).clamp(0, 1)

        target_hair_mask_soft = (
            0.55 * pseudo_target_mask
            + 0.45 * coarse_target_mask
            + detail_strength * boundary_residual * boundary_band_256
        ).clamp(0, 1)
        target_hair_mask_soft = torch.maximum(
            target_hair_mask_soft,
            pseudo_target_mask * (0.72 + 0.28 * (boundary_band_256 * detail_map_256).clamp(0, 1)).clamp(0, 1),
        ).clamp(0, 1)
        target_hair_mask = (target_hair_mask_soft > 0.5).float()

        coarse_32 = F.interpolate(coarse_target_mask, size=F_base.shape[-2:], mode="bilinear", align_corners=False)
        target_32 = F.interpolate(target_hair_mask_soft, size=F_base.shape[-2:], mode="bilinear", align_corners=False)
        boundary_32 = F.interpolate(boundary_band_256, size=F_base.shape[-2:], mode="bilinear", align_corners=False)
        remove_32 = F.interpolate(remove_region_256, size=F_base.shape[-2:], mode="bilinear", align_corners=False)
        topology_delta_32 = (target_32 - coarse_32).abs()
        feature_gap_32 = (reference32 - F_base).abs().mean(dim=1, keepdim=True)
        feature_gap_32 = feature_gap_32 / feature_gap_32.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        repair_support_32 = torch.maximum(
            torch.maximum(boundary_32, (0.85 * remove_32).clamp(0, 1)),
            torch.maximum((0.60 * topology_delta_32).clamp(0, 1), (0.55 * target_32 * feature_gap_32).clamp(0, 1)),
        )

        f32_input = torch.cat([F_base, F_src, reference32, prior_32, coarse_32, target_32, boundary_32], dim=1)
        learned_delta_F32 = self.f32_head(f32_input)
        reference_support_32 = repair_support_32
        learned_support_32 = repair_support_32
        reference_base_32 = F_base + reference_support_32 * (reference32 - F_base)
        latent_F_refined = reference_base_32 + strength * learned_support_32 * learned_delta_F32
        latent_F_refined = latent_F_refined + shape_prior_strength * repair_support_32 * (reference32 - reference_base_32)

        target_64 = F.interpolate(target_hair_mask_soft, size=F_base64.shape[-2:], mode="bilinear", align_corners=False)
        coarse_64 = F.interpolate(coarse_target_mask, size=F_base64.shape[-2:], mode="bilinear", align_corners=False)
        boundary_64 = F.interpolate(boundary_band_256, size=F_base64.shape[-2:], mode="bilinear", align_corners=False)
        detail_64 = F.interpolate(detail_map_256, size=F_base64.shape[-2:], mode="bilinear", align_corners=False)
        remove_64 = F.interpolate(remove_region_256, size=F_base64.shape[-2:], mode="bilinear", align_corners=False)
        f32_to_64 = F.interpolate(latent_F_refined, size=F_base64.shape[-2:], mode="bilinear", align_corners=False)
        f64_input = torch.cat([F_base64, reference64, f32_to_64, prior_64, target_64, boundary_64, detail_64], dim=1)
        learned_delta_F64 = self.f64_head(f64_input)

        base_detail_feature = compose_boundary_detail_feature_v19(
            base_feature_64=F_base64,
            shape_feature_64=reference64,
            boundary_band_256=boundary_band_256,
            detail_map_256=detail_map_256,
            strength=detail_strength,
        )
        topology_delta_64 = (target_64 - coarse_64).abs()
        feature_gap_64 = (reference64 - F_base64).abs().mean(dim=1, keepdim=True)
        feature_gap_64 = feature_gap_64 / feature_gap_64.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        detail_support_64 = torch.maximum(
            torch.maximum(boundary_64, (0.55 * remove_64).clamp(0, 1)),
            torch.maximum((0.60 * topology_delta_64).clamp(0, 1), torch.maximum((0.50 * target_64 * feature_gap_64).clamp(0, 1), (0.40 * target_64 * detail_64).clamp(0, 1))),
        )
        reference_base_64 = F_base64 + detail_support_64 * (reference64 - F_base64)
        latent_F64_detail = reference_base_64 + boundary_64 * (base_detail_feature - reference_base_64)
        latent_F64_detail = latent_F64_detail + detail_strength * detail_support_64 * learned_delta_F64

        return {
            "coarse_target_mask_logit": mask_logits[:, 0:1],
            "boundary_residual_logit": mask_logits[:, 1:2],
            "coarse_target_mask": coarse_target_mask,
            "boundary_residual": boundary_residual,
            "target_hair_mask_soft": target_hair_mask_soft,
            "target_hair_mask": target_hair_mask,
            "latent_F_refined": latent_F_refined,
            "latent_F64_detail": latent_F64_detail,
            "learned_delta_F32": learned_delta_F32,
            "learned_delta_F64": learned_delta_F64,
        }


def load_shape_adaptor_v19(model: nn.Module, checkpoint: dict) -> tuple[list[str], list[str]]:
    state = checkpoint
    for key in ("shape_adapter_v19_state_dict", "state_dict", "model_state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break

    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return sorted(compatible.keys()), sorted(skipped)
