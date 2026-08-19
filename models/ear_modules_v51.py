from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CtrlHair.external_code.face_parsing.my_parsing_util import FaceParsing_tensor
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion

RAW_LEFT_EAR = 7
RAW_RIGHT_EAR = 8
RAW_EARRING = 9
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


def build_raw_face_masks(parsing: torch.Tensor) -> dict[str, torch.Tensor]:
    parsing = ensure_mask_4d(parsing).long()
    return {
        "left_ear": (parsing == RAW_LEFT_EAR).float(),
        "right_ear": (parsing == RAW_RIGHT_EAR).float(),
        "earring": (parsing == RAW_EARRING).float(),
        "detail": torch.stack([(parsing == label).float() for label in RAW_DETAIL_LABELS], dim=0).amax(dim=0),
        "hair": (parsing == RAW_HAIR).float(),
        "hat": (parsing == RAW_HAT).float(),
    }


class FaceParsingHelperV51:
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


class HairMaskExtractorV51(nn.Module):
    def __init__(self, device: str = "cuda", dilate_erosion: int = 5):
        super().__init__()
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
        source_hair_block_dilate: int = 5,
        source_hair_block_strength: float = 0.95,
        target_visibility_expand: int = 5,
        max_target_hair_overlap: float = 0.55,
    ):
        super().__init__()
        self.ear_dilate = ear_dilate
        self.hair_change_dilate = hair_change_dilate
        self.earring_expand = earring_expand
        self.downward_shift = downward_shift
        self.target_hair_dilate = target_hair_dilate
        self.source_hair_block_dilate = source_hair_block_dilate
        self.source_hair_block_strength = max(0.0, min(1.0, float(source_hair_block_strength)))
        self.target_visibility_expand = target_visibility_expand
        self.max_target_hair_overlap = max_target_hair_overlap

    def _split_by_side(self, mask: torch.Tensor, left_hint: torch.Tensor, right_hint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        width = mask.size(-1)
        x_coords = torch.linspace(0, 1, width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
        left_half = (x_coords <= 0.5).float()
        right_half = 1 - left_half

        left_mask = mask * torch.clamp(left_hint + left_half, 0, 1)
        right_mask = mask * torch.clamp(right_hint + right_half, 0, 1)
        return left_mask, right_mask

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
        source_left_ring, source_right_ring = self._split_by_side(source_masks["earring"], left_hint, right_hint)
        target_left_ring, target_right_ring = self._split_by_side(target_masks["earring"], left_hint, right_hint)

        left_roi = dilate_mask(source_masks["left_ear"] + target_masks["left_ear"] + source_left_ring, self.ear_dilate)
        right_roi = dilate_mask(source_masks["right_ear"] + target_masks["right_ear"] + source_right_ring, self.ear_dilate)
        left_roi = torch.clamp(left_roi + shift_mask(left_roi, down=self.downward_shift), 0, 1)
        right_roi = torch.clamp(right_roi + shift_mask(right_roi, down=self.downward_shift), 0, 1)
        ear_roi = torch.clamp(left_roi + right_roi, 0, 1)

        hair_change = dilate_mask((source_hair_mask - target_hair_mask).abs(), self.hair_change_dilate)
        ring_hint = dilate_mask(source_masks["earring"], self.earring_expand)
        hat_mask = torch.clamp(source_masks["hat"] + target_masks["hat"], 0, 1)
        target_hair_context = dilate_mask(target_hair_mask, self.target_hair_dilate)
        source_hair_context = dilate_mask(source_hair_mask, self.source_hair_block_dilate)
        source_earring_keep = dilate_mask(source_masks["earring"], max(3, self.earring_expand // 2))

        left_target_visible = target_masks["left_ear"] + target_left_ring
        right_target_visible = target_masks["right_ear"] + target_right_ring
        left_target_visible = dilate_mask(left_target_visible * left_roi, self.target_visibility_expand) * left_roi
        right_target_visible = dilate_mask(right_target_visible * right_roi, self.target_visibility_expand) * right_roi

        left_hair_overlap = self._roi_overlap(target_hair_context, left_roi)
        right_hair_overlap = self._roi_overlap(target_hair_context, right_roi)
        left_has_visible_target = left_target_visible.flatten(1).amax(dim=1) > 0
        right_has_visible_target = right_target_visible.flatten(1).amax(dim=1) > 0

        left_side_visible = ((left_hair_overlap < self.max_target_hair_overlap) | left_has_visible_target).float()
        right_side_visible = ((right_hair_overlap < self.max_target_hair_overlap) | right_has_visible_target).float()
        left_side_visible_map = left_side_visible.view(-1, 1, 1, 1)
        right_side_visible_map = right_side_visible.view(-1, 1, 1, 1)

        left_open_region = left_roi * (1 - target_hair_context)
        right_open_region = right_roi * (1 - target_hair_context)
        left_visible_roi = torch.clamp(left_open_region + left_target_visible, 0, 1) * left_side_visible_map
        right_visible_roi = torch.clamp(right_open_region + right_target_visible, 0, 1) * right_side_visible_map
        visible_ear_roi = torch.clamp(left_visible_roi + right_visible_roi, 0, 1)

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

        presence_target = torch.stack(
            [
                source_left_ring.flatten(1).amax(dim=1),
                source_right_ring.flatten(1).amax(dim=1),
                source_masks["earring"].flatten(1).amax(dim=1),
            ],
            dim=1,
        ).float()
        visibility_target = torch.stack(
            [
                left_side_visible,
                right_side_visible,
                torch.maximum(left_side_visible, right_side_visible),
            ],
            dim=1,
        ).float()
        presence_target = presence_target * visibility_target

        source_left_ring = source_left_ring * left_visible_roi
        source_right_ring = source_right_ring * right_visible_roi
        source_earring_mask = source_masks["earring"] * visible_ear_roi

        return {
            "left_ear_roi": left_roi,
            "right_ear_roi": right_roi,
            "ear_roi": ear_roi,
            "visible_ear_roi": visible_ear_roi,
            "query_mask": query_mask,
            "source_hair_block_mask": source_hair_block_mask,
            "source_hair_mask": source_hair_mask,
            "target_hair_mask": target_hair_mask,
            "source_earring_mask": source_earring_mask,
            "target_earring_mask": target_masks["earring"],
            "source_left_earring_mask": source_left_ring,
            "source_right_earring_mask": source_right_ring,
            "presence_target": presence_target,
            "visibility_target": visibility_target,
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
    def __init__(self, hidden_channels: int = 32, init_bias: float = -4.0):
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
    def __init__(self, base_channels: int = 512, prior_channels: int = 128, residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
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
        fused = base_feature + self.residual_scale * delta * fine_mask
        return fused, fine_mask

