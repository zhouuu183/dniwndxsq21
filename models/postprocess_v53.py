from __future__ import annotations

import argparse
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoders import FeatureEncoderMult, FeatureiResnet, ModulationModule
from models.ear_modules_v53 import (
    BrightnessReEstimator,
    DynamicFineMaskRefresher,
    EarAnchoredQueryBuilder,
    FaceParsingHelperV53,
    HFDAGatedInjectionUnit,
    ShadowSuppressedHFExtractor,
    build_aligned_earring_reference,
    build_weak_earring_masks,
    dilate_mask,
    gaussian_blur,
    ensure_mask_4d,
    high_pass_filter,
    normalized_to_01,
    rgb_to_gray,
    resize_mask,
)
from models.stylegan2.model import PixelNorm

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")
FACE_SURFACE_LABELS = (1, 2)
FACE_OUTPUT_GUARD_LABELS = (1, 2, 3, 4, 5, 6, 11, 12, 13)


def build_query_builder_compat(**kwargs) -> EarAnchoredQueryBuilder:
    accepted = inspect.signature(EarAnchoredQueryBuilder.__init__).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return EarAnchoredQueryBuilder(**filtered_kwargs)


def build_strict_earring_core_mask(
    reference_image: torch.Tensor,
    candidate_mask: torch.Tensor,
    fallback_mask: torch.Tensor | None = None,
    min_high: float = 0.014,
    min_chroma: float = 0.038,
    min_contrast: float = 0.018,
    dilate: int = 3,
) -> torch.Tensor:
    reference_image = normalized_to_01(reference_image)
    candidate_mask = resize_mask(candidate_mask, reference_image.shape[-2:]).clamp(0, 1)
    high_energy = high_pass_filter(reference_image).abs().mean(dim=1, keepdim=True)
    chroma = reference_image.amax(dim=1, keepdim=True) - reference_image.amin(dim=1, keepdim=True)
    gray = rgb_to_gray(reference_image)
    local_gray = gaussian_blur(gray, kernel_size=15, sigma=4.0)
    bright = (gray - local_gray).clamp_min(0)
    contrast = (gray - local_gray).abs()
    material = torch.clamp(
        (chroma > float(min_chroma)).float()
        + (contrast > float(min_contrast)).float()
        + (bright > float(min_contrast) * 0.75).float(),
        0,
        1,
    )
    core = (high_energy > float(min_high)).float() * material * candidate_mask
    if dilate > 1:
        core = dilate_mask(core, int(dilate)) * candidate_mask
    fallback_empty = core.flatten(1).sum(dim=1).view(-1, 1, 1, 1) <= 1
    if fallback_mask is None:
        fallback_mask = torch.zeros_like(candidate_mask)
    fallback_mask = resize_mask(fallback_mask, reference_image.shape[-2:]).clamp(0, 1)
    return torch.where(fallback_empty, fallback_mask, core).clamp(0, 1)


def build_flat_dark_pollution_mask(
    reference_image: torch.Tensor,
    candidate_mask: torch.Tensor,
    max_gray: float = 0.22,
    max_chroma: float = 0.07,
    max_high: float = 0.018,
    dilate: int = 3,
) -> torch.Tensor:
    reference_image = normalized_to_01(reference_image)
    candidate_mask = resize_mask(candidate_mask, reference_image.shape[-2:]).clamp(0, 1)
    gray = rgb_to_gray(reference_image)
    chroma = reference_image.amax(dim=1, keepdim=True) - reference_image.amin(dim=1, keepdim=True)
    high = high_pass_filter(reference_image).abs().mean(dim=1, keepdim=True)
    flat_dark = (
        (gray < float(max_gray)).float()
        * (chroma < float(max_chroma)).float()
        * (high < float(max_high)).float()
        * candidate_mask
    )
    if dilate > 1:
        flat_dark = dilate_mask(flat_dark, int(dilate)) * candidate_mask
    return flat_dark.clamp(0, 1)


class CleanupFaceRefiner(nn.Module):
    def __init__(self, base_channels: int, hidden_channels: int = 128, strength: float = 1.0):
        super().__init__()
        self.strength = float(strength)
        self.net = nn.Sequential(
            nn.Conv2d(base_channels + 7, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, base_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        feature: torch.Tensor,
        source_01: torch.Tensor,
        target_01: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = resize_mask(mask, feature.shape[-2:])
        target_low = F.interpolate(target_01, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        # The cleanup branch should follow the SATD/target appearance, not reintroduce source skin texture.
        source_low = target_low
        delta = self.net(torch.cat([feature, source_low, target_low, mask], dim=1))
        delta = delta * mask * self.strength
        return feature + delta, mask, delta


class PostProcessModelV53(nn.Module):
    def __init__(self, args: argparse.Namespace | None = None):
        super().__init__()
        self.args = args or argparse.Namespace()

        self.encoder_face = FeatureEncoderMult(
            fs_layers=[9],
            opts=argparse.Namespace(arcface_model_path="pretrained_models/ArcFace/backbone_ir50.pth"),
        )
        if not getattr(self.args, "finetune", False):
            for param in self.encoder_face.parameters():
                param.requires_grad = False

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.latent_avg = torch.load("pretrained_models/PostProcess/latent_avg.pt", map_location=map_location)
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        self.use_mod = getattr(self.args, "use_mod", True)
        self.use_full = getattr(self.args, "use_full", False)
        self.pretrain = getattr(self.args, "pretrain", False)
        self.use_dataset_query_mask = getattr(self.args, "use_dataset_query_mask", False)
        self.use_cleanup_face_refiner = bool(
            getattr(self.args, "enable_cleanup_face_refiner", False)
            or getattr(self.args, "training_stage", "") == "cleanup_face"
        )

        if self.use_mod:
            self.to_latent_1 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
            self.to_latent_2 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
            self.pixelnorm = PixelNorm()
        else:
            self.to_latent = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm([1024]),
                nn.LeakyReLU(),
                nn.Linear(1024, 512),
            )

        feature_channels = getattr(self.args, "ear_feature_channels", 128)
        self.parsing_helper = FaceParsingHelperV53(parse_size=getattr(self.args, "ear_parse_size", 512))
        self.query_builder = build_query_builder_compat(
            ear_dilate=getattr(self.args, "ear_dilate", 21),
            hair_change_dilate=getattr(self.args, "hair_change_dilate", 25),
            earring_expand=getattr(self.args, "earring_expand", 15),
            downward_shift=getattr(self.args, "ear_downward_shift", 10),
            target_hair_dilate=getattr(self.args, "target_hair_dilate", 11),
            source_hair_block_dilate=getattr(self.args, "source_hair_block_dilate", 5),
            source_hair_block_strength=getattr(self.args, "source_hair_block_strength", 0.95),
            target_visibility_expand=getattr(self.args, "target_visibility_expand", 5),
            max_target_hair_overlap=getattr(self.args, "max_target_hair_overlap", 0.55),
            earring_lobe_dilate=getattr(self.args, "earring_lobe_dilate", 17),
            earring_lobe_down_shift=getattr(self.args, "earring_lobe_down_shift", 18),
            earring_outer_shift=getattr(self.args, "earring_outer_shift", 10),
            earring_query_floor=getattr(self.args, "earring_query_floor", 0.45),
        )
        self.hf_extractor = ShadowSuppressedHFExtractor(
            low_alpha=getattr(self.args, "ear_low_alpha", 0.15),
            feature_channels=feature_channels,
            blur_kernel=getattr(self.args, "ear_blur_kernel", 11),
            blur_sigma=getattr(self.args, "ear_blur_sigma", 3.0),
        )
        self.mask_refresher = DynamicFineMaskRefresher(
            hidden_channels=getattr(self.args, "ear_mask_hidden", 32),
            init_bias=getattr(self.args, "ear_mask_init_bias", -4.0),
        )
        self.brightness_reestimator = BrightnessReEstimator(feature_channels=feature_channels)
        self.ear_injector_64 = HFDAGatedInjectionUnit(
            base_channels=getattr(self.args, "ear_inject_channels_64", 512),
            prior_channels=feature_channels,
            strength=getattr(self.args, "ear_injection_strength", 2.5),
        )
        self.ear_injector_128 = HFDAGatedInjectionUnit(
            base_channels=getattr(self.args, "ear_inject_channels_128", 256),
            prior_channels=feature_channels,
            strength=getattr(self.args, "ear_injection_strength", 2.5),
        )
        cleanup_hidden = int(getattr(self.args, "cleanup_face_hidden", 128))
        cleanup_strength = float(getattr(self.args, "cleanup_face_strength", 1.0))
        self.cleanup_face_refiner_64 = CleanupFaceRefiner(
            base_channels=getattr(self.args, "ear_inject_channels_64", 512),
            hidden_channels=cleanup_hidden,
            strength=cleanup_strength,
        )
        self.cleanup_face_refiner_128 = CleanupFaceRefiner(
            base_channels=getattr(self.args, "ear_inject_channels_128", 256),
            hidden_channels=cleanup_hidden,
            strength=cleanup_strength,
        )

    def load_base_checkpoint(self, checkpoint_path: str | None):
        if not checkpoint_path:
            return None
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        result = self.load_state_dict(state_dict, strict=False)
        relevant_missing = [
            key for key in result.missing_keys
            if not key.startswith((
                "hf_extractor",
                "mask_refresher",
                "brightness_reestimator",
                "ear_injector_64",
                "ear_injector_128",
                "cleanup_face_refiner_64",
                "cleanup_face_refiner_128",
                "query_builder",
                "parsing_helper",
            ))
        ]
        if relevant_missing:
            print(f"[PostProcessModelV53] Missing base keys: {len(relevant_missing)}")
            print(relevant_missing[:20])
        if result.unexpected_keys:
            print(f"[PostProcessModelV53] Unexpected checkpoint keys: {len(result.unexpected_keys)}")
            print(result.unexpected_keys[:20])
        return result

    def _compute_latent(self, s_face: torch.Tensor, s_hair: torch.Tensor) -> torch.Tensor:
        if self.use_mod:
            dt_latent_face = self.pixelnorm(s_face)
            dt_latent_hair = self.pixelnorm(s_hair)

            for mod_module in self.to_latent_1:
                dt_latent_face = mod_module(dt_latent_face, s_hair)
            for mod_module in self.to_latent_2:
                dt_latent_hair = mod_module(dt_latent_hair, s_face)

            return self.latent_avg.to(s_face.device) + 0.1 * (dt_latent_face + dt_latent_hair)

        cat_s = torch.cat((s_face, s_hair), dim=-1)
        return self.latent_avg.to(s_face.device) + self.to_latent(cat_s)

    def _build_base_feature(
        self,
        f_face: torch.Tensor,
        f_hair: torch.Tensor,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_full or target_mask is None:
            cat_f = torch.cat((f_face, f_hair), dim=1)
        else:
            t_mask = F.interpolate(ensure_mask_4d(target_mask).float(), size=f_face.shape[-2:], mode="nearest")
            cat_f = torch.cat((f_face * t_mask, f_hair * (1 - t_mask)), dim=1)
        return self.to_feature(cat_f)

    def _resolve_masks(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_parsing: torch.Tensor | None,
        target_parsing: torch.Tensor | None,
        source_hair_mask: torch.Tensor | None,
        target_hair_mask: torch.Tensor | None,
        query_mask: torch.Tensor | None,
        source_ear_mask: torch.Tensor | None,
        presence_target: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        source_01 = normalized_to_01(source)
        target_01 = normalized_to_01(target)
        image_size = tuple(source_01.shape[-2:])

        if source_parsing is None:
            source_parsing = self.parsing_helper.parse(source_01, out_size=image_size)
        if target_parsing is None:
            target_parsing = self.parsing_helper.parse(target_01, out_size=image_size)

        query_info = self.query_builder(source_parsing, target_parsing, source_hair_mask, target_hair_mask)
        visible_ear_roi = query_info.get("visible_ear_roi")
        if query_mask is not None and self.use_dataset_query_mask:
            dataset_query_mask = ensure_mask_4d(query_mask).float()
            if visible_ear_roi is not None:
                dataset_query_mask = dataset_query_mask * visible_ear_roi
            query_info["query_mask"] = dataset_query_mask
        if source_ear_mask is not None:
            source_ear_mask = ensure_mask_4d(source_ear_mask).float()
            query_info["source_earring_mask"] = torch.clamp(
                query_info.get("source_earring_mask", torch.zeros_like(source_ear_mask)) + source_ear_mask,
                0,
                1,
            )
        if presence_target is not None:
            visibility_target = query_info.get("visibility_target")
            presence_target = presence_target.float()
            if visibility_target is not None:
                presence_target = presence_target * visibility_target.float()
            query_info["presence_target"] = presence_target

        query_info["source_parsing"] = source_parsing.long()
        query_info["target_parsing"] = target_parsing.long()
        return query_info

    @staticmethod
    def _parsing_label_mask(parsing: torch.Tensor | None, labels: tuple[int, ...]) -> torch.Tensor | None:
        if parsing is None:
            return None
        parsing = ensure_mask_4d(parsing).long()
        mask = torch.zeros_like(parsing, dtype=torch.bool)
        for label in labels:
            mask |= parsing == label
        return mask.float()

    @staticmethod
    def _combine_cleanup_masks(cleanup_masks: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        if not cleanup_masks:
            return None
        masks = []
        for key in CLEANUP_MASK_KEYS:
            value = cleanup_masks.get(key)
            if value is not None:
                masks.append(ensure_mask_4d(value).float())
        if not masks:
            return None
        return torch.stack(masks, dim=0).amax(dim=0).clamp(0, 1)

    def _build_cleanup_face_mask(
        self,
        aux: dict[str, torch.Tensor],
        cleanup_masks: dict[str, torch.Tensor] | None,
        revealed_skin_mask: torch.Tensor | None,
        earring_exclude_mask: torch.Tensor | None,
        fallback_size: tuple[int, int],
    ) -> torch.Tensor | None:
        target_face_surface = self._parsing_label_mask(aux.get("target_parsing"), FACE_SURFACE_LABELS)
        if target_face_surface is None:
            return None

        target_face_surface = resize_mask(target_face_surface, fallback_size)
        cleanup_mask = self._combine_cleanup_masks(cleanup_masks)
        if cleanup_mask is not None:
            cleanup_mask = resize_mask(cleanup_mask, fallback_size)

        face_cleanup = None
        halo_cleanup = None
        if cleanup_masks:
            face_cleanup = cleanup_masks.get("M_remove_face")
            halo_cleanup = cleanup_masks.get("M_remove_halo")
        if face_cleanup is not None:
            face_cleanup = resize_mask(face_cleanup, fallback_size)
        if halo_cleanup is not None:
            halo_cleanup = resize_mask(halo_cleanup, fallback_size)
        if revealed_skin_mask is not None:
            revealed_skin_mask = resize_mask(revealed_skin_mask, fallback_size)

        focused = torch.zeros_like(target_face_surface)
        if cleanup_mask is not None:
            focused = focused + 0.35 * cleanup_mask * target_face_surface
        if face_cleanup is not None:
            focused = focused + face_cleanup
        if halo_cleanup is not None:
            focused = focused + 0.5 * halo_cleanup * target_face_surface
        if revealed_skin_mask is not None:
            focused = focused + revealed_skin_mask

        mask = focused.clamp(0, 1) * target_face_surface
        target_hair_mask = aux.get("target_hair_mask")
        if target_hair_mask is not None:
            mask = mask * (1 - resize_mask(target_hair_mask, fallback_size)).clamp(0, 1)

        exclude = None
        for value in (
            aux.get("ear_roi"),
            aux.get("visible_ear_roi"),
            aux.get("query_mask"),
            aux.get("target_earring_mask"),
            aux.get("source_earring_mask"),
            aux.get("earring_confident_mask"),
            aux.get("earring_reference_mask"),
            earring_exclude_mask,
        ):
            if value is None:
                continue
            value = resize_mask(value, fallback_size)
            exclude = value if exclude is None else torch.clamp(exclude + value, 0, 1)
        if exclude is not None:
            exclude_dilate = int(getattr(self.args, "cleanup_face_exclude_earring_dilate", 13))
            mask = mask * (1 - dilate_mask(exclude, exclude_dilate)).clamp(0, 1)

        mask_dilate = int(getattr(self.args, "cleanup_face_dilate", 5))
        if mask_dilate > 1:
            mask = dilate_mask(mask, mask_dilate) * target_face_surface
            if target_hair_mask is not None:
                mask = mask * (1 - resize_mask(target_hair_mask, fallback_size)).clamp(0, 1)
            if exclude is not None:
                mask = mask * (1 - dilate_mask(exclude, exclude_dilate)).clamp(0, 1)

        if mask.flatten(1).amax(dim=1).sum().item() <= 0:
            return None
        return mask.clamp(0, 1)

    def _build_injection_mask(
        self,
        seed_mask: torch.Tensor,
    ) -> torch.Tensor:
        injection_mask = ensure_mask_4d(seed_mask).float().clamp(0, 1)

        dilate = int(getattr(self.args, "ear_injection_mask_dilate", 3))
        if dilate > 1:
            injection_mask = dilate_mask(injection_mask, dilate)

        boost = max(0.0, float(getattr(self.args, "ear_injection_mask_boost", 1.35)))
        if boost != 1.0:
            injection_mask = torch.clamp(injection_mask * boost, 0, 1)
        return injection_mask

    def _build_lateral_earring_support(
        self,
        reference_mask: torch.Tensor,
        search_roi: torch.Tensor | None,
    ) -> torch.Tensor:
        width = reference_mask.size(-1)
        ratio = max(0.20, min(0.50, float(getattr(self.args, "earring_injection_lateral_ratio", 0.42))))
        x_coords = torch.linspace(
            0,
            1,
            width,
            device=reference_mask.device,
            dtype=reference_mask.dtype,
        ).view(1, 1, 1, width)
        lateral = ((x_coords <= ratio) | (x_coords >= 1 - ratio)).float()
        support = lateral.expand_as(reference_mask)
        if search_roi is not None:
            search_roi = resize_mask(search_roi, reference_mask.shape[-2:])
            search_dilate = int(getattr(self.args, "earring_injection_search_dilate", 5))
            if search_dilate > 1:
                search_roi = dilate_mask(search_roi, search_dilate)
            support = support * search_roi
        return support.clamp(0, 1)

    def _build_material_injection_seed(
        self,
        reference_image: torch.Tensor,
        candidate_mask: torch.Tensor,
        parser_mask: torch.Tensor,
        search_roi: torch.Tensor | None,
        face_reject_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reference_image = normalized_to_01(reference_image)
        candidate_mask = resize_mask(candidate_mask, reference_image.shape[-2:]).clamp(0, 1)
        parser_mask = resize_mask(parser_mask, reference_image.shape[-2:]).clamp(0, 1)
        support = self._build_lateral_earring_support(candidate_mask, search_roi)
        face_penalty = None
        if face_reject_mask is not None:
            face_reject_mask = resize_mask(face_reject_mask, reference_image.shape[-2:]).clamp(0, 1)
            face_dilate = int(getattr(self.args, "earring_injection_face_reject_dilate", 1))
            if face_dilate > 1:
                face_reject_mask = dilate_mask(face_reject_mask, face_dilate)
            face_strength = max(
                0.0,
                min(1.0, float(getattr(self.args, "earring_injection_face_reject_strength", 0.65))),
            )
            face_penalty = (1 - face_strength * face_reject_mask).clamp(0, 1)
        candidate_mask = candidate_mask * support
        if face_penalty is not None:
            candidate_mask = candidate_mask * face_penalty
        parser_mask = parser_mask * support

        high = high_pass_filter(reference_image).abs().mean(dim=1, keepdim=True)
        chroma = reference_image.amax(dim=1, keepdim=True) - reference_image.amin(dim=1, keepdim=True)
        gray = rgb_to_gray(reference_image)
        local_gray = gaussian_blur(gray, kernel_size=15, sigma=4.0)
        bright = (gray - local_gray).clamp_min(0)
        dark = (local_gray - gray).clamp_min(0)
        contrast = (gray - local_gray).abs()

        evidence = (
            (chroma > float(getattr(self.args, "earring_injection_min_chroma", 0.026))).float()
            + (contrast > float(getattr(self.args, "earring_injection_min_contrast", 0.014))).float()
            + (bright > float(getattr(self.args, "earring_injection_min_bright", 0.012))).float()
            + (dark > float(getattr(self.args, "earring_injection_min_dark", 0.014))).float()
        )
        min_votes = max(1, int(getattr(self.args, "earring_injection_min_votes", 2)))
        material = (
            (high > float(getattr(self.args, "earring_injection_min_high", 0.010))).float()
            * (evidence >= min_votes).float()
        )
        flat_skin_like = (
            (gray > 0.20).float()
            * (gray < 0.88).float()
            * (chroma < float(getattr(self.args, "earring_injection_skin_max_chroma", 0.16))).float()
            * (high < float(getattr(self.args, "earring_injection_skin_max_high", 0.026))).float()
            * (contrast < float(getattr(self.args, "earring_injection_skin_max_contrast", 0.026))).float()
        )
        material_seed = candidate_mask * material * (1 - flat_skin_like).clamp(0, 1)

        parser_area = parser_mask.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
        parser_max_area = float(getattr(self.args, "earring_injection_parser_max_area", 420.0))
        parser_material_support = dilate_mask(material_seed, int(getattr(self.args, "earring_parser_material_dilate", 3)))
        small_parser_seed = torch.where(
            parser_area <= parser_max_area,
            parser_mask,
            parser_mask * material,
        )
        seed = torch.clamp(material_seed + small_parser_seed, 0, 1)

        max_area_frac = float(getattr(self.args, "earring_injection_seed_max_area_frac", 0.012))
        if max_area_frac > 0:
            strict_evidence = (
                (chroma > float(getattr(self.args, "earring_injection_strict_min_chroma", 0.045))).float()
                + (contrast > float(getattr(self.args, "earring_injection_strict_min_contrast", 0.024))).float()
                + (bright > float(getattr(self.args, "earring_injection_strict_min_bright", 0.022))).float()
                + (dark > float(getattr(self.args, "earring_injection_strict_min_dark", 0.022))).float()
            )
            strict_material = (
                (high > float(getattr(self.args, "earring_injection_strict_min_high", 0.018))).float()
                * (strict_evidence >= min_votes).float()
            )
            strict_seed = candidate_mask * strict_material * (1 - flat_skin_like).clamp(0, 1)
            strict_seed = torch.clamp(strict_seed + parser_mask * dilate_mask(strict_seed, 3), 0, 1)
            seed_area_frac = seed.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
            strict_area_frac = strict_seed.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
            seed = torch.where(seed_area_frac > max_area_frac, strict_seed, seed)
            seed = torch.where(strict_area_frac > max_area_frac, torch.zeros_like(seed), seed)

        seed_dilate = int(getattr(self.args, "earring_injection_seed_dilate", 1))
        if seed_dilate > 1:
            seed = dilate_mask(seed, seed_dilate) * torch.clamp(candidate_mask + parser_mask, 0, 1)
        return seed.clamp(0, 1)

    def _build_guarded_composite_seed(
        self,
        reference_image: torch.Tensor,
        parser_mask: torch.Tensor,
        highlight_mask: torch.Tensor,
        dark_candidate_mask: torch.Tensor,
        injection_seed_mask: torch.Tensor,
        search_roi: torch.Tensor | None,
        face_reject_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reference_image = normalized_to_01(reference_image)
        out_size = reference_image.shape[-2:]
        parser_mask = resize_mask(parser_mask, out_size).clamp(0, 1)
        highlight_mask = resize_mask(highlight_mask, out_size).clamp(0, 1)
        dark_candidate_mask = resize_mask(dark_candidate_mask, out_size).clamp(0, 1)
        injection_seed_mask = resize_mask(injection_seed_mask, out_size).clamp(0, 1)

        candidate_mask = torch.clamp(parser_mask + highlight_mask + dark_candidate_mask + injection_seed_mask, 0, 1)
        if not bool(getattr(self.args, "earring_composite_seed_use_search_roi", False)):
            search_roi = None
        support = self._build_lateral_earring_support(candidate_mask, search_roi)

        face_penalty = None
        if face_reject_mask is not None:
            face_reject_mask = resize_mask(face_reject_mask, out_size).clamp(0, 1)
            face_strength = max(
                0.0,
                min(1.0, float(getattr(self.args, "earring_composite_seed_face_reject_strength", 0.35))),
            )
            face_penalty = (1 - face_strength * face_reject_mask).clamp(0, 1)

        high = high_pass_filter(reference_image).abs().mean(dim=1, keepdim=True)
        chroma = reference_image.amax(dim=1, keepdim=True) - reference_image.amin(dim=1, keepdim=True)
        gray = rgb_to_gray(reference_image)
        local_gray = gaussian_blur(gray, kernel_size=15, sigma=4.0)
        bright = (gray - local_gray).clamp_min(0)
        dark = (local_gray - gray).clamp_min(0)
        contrast = (gray - local_gray).abs()
        evidence = (
            (chroma > float(getattr(self.args, "earring_composite_seed_min_chroma", 0.018))).float()
            + (contrast > float(getattr(self.args, "earring_composite_seed_min_contrast", 0.010))).float()
            + (bright > float(getattr(self.args, "earring_composite_seed_min_bright", 0.010))).float()
            + (dark > float(getattr(self.args, "earring_composite_seed_min_dark", 0.010))).float()
        )
        min_votes = max(1, int(getattr(self.args, "earring_composite_seed_min_votes", 1)))
        material = (
            (high > float(getattr(self.args, "earring_composite_seed_min_high", 0.006))).float()
            * (evidence >= min_votes).float()
        )

        parser_seed = parser_mask * support
        parser_area = parser_seed.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
        parser_max_area = float(getattr(self.args, "earring_composite_seed_parser_max_area", 520.0))
        parser_seed = torch.where(parser_area <= parser_max_area, parser_seed, parser_seed * material)

        material_seed = candidate_mask * support * material
        if face_penalty is not None:
            material_seed = material_seed * face_penalty

        anchor = torch.clamp(parser_seed + injection_seed_mask, 0, 1)
        support_dilate = int(getattr(self.args, "earring_composite_seed_support_dilate", 5))
        if support_dilate > 1:
            anchor_support = dilate_mask(anchor, support_dilate)
        else:
            anchor_support = anchor
        detail_seed = torch.clamp(highlight_mask + dark_candidate_mask, 0, 1) * support * torch.clamp(
            anchor_support + material_seed,
            0,
            1,
        )
        seed = torch.clamp(parser_seed + injection_seed_mask + detail_seed + material_seed * anchor_support, 0, 1)

        max_area_frac = float(getattr(self.args, "earring_composite_seed_max_area_frac", 0.018))
        if max_area_frac > 0:
            fallback_seed = torch.clamp(injection_seed_mask + detail_seed * material + parser_seed * material, 0, 1)
            seed_area_frac = seed.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
            fallback_area_frac = fallback_seed.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
            seed = torch.where(seed_area_frac > max_area_frac, fallback_seed, seed)
            seed = torch.where(fallback_area_frac > max_area_frac, torch.zeros_like(seed), seed)

        seed_dilate = int(getattr(self.args, "earring_composite_seed_dilate", 2))
        if seed_dilate > 1:
            seed = dilate_mask(seed, seed_dilate) * support
        return seed.clamp(0, 1)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        HT_E: torch.Tensor | None = None,
        source_parsing: torch.Tensor | None = None,
        target_parsing: torch.Tensor | None = None,
        source_hair_mask: torch.Tensor | None = None,
        target_hair_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        source_ear_mask: torch.Tensor | None = None,
        presence_target: torch.Tensor | None = None,
        cleanup_masks: dict[str, torch.Tensor] | None = None,
        revealed_skin_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if cleanup_masks is None:
            cleanup_masks = {
                key: kwargs[key]
                for key in CLEANUP_MASK_KEYS
                if key in kwargs and kwargs[key] is not None
            }
        if revealed_skin_mask is None:
            revealed_skin_mask = kwargs.get("revealed_skin_mask")

        s_face, [f_face] = self.encoder_face(source)
        if self.pretrain:
            aux = self._resolve_masks(
                source,
                target,
                source_parsing,
                target_parsing,
                source_hair_mask,
                target_hair_mask,
                query_mask,
                source_ear_mask,
                presence_target,
            )
            return self.latent_avg.to(s_face.device) + s_face, f_face, aux

        s_hair, [f_hair] = self.encoder_face(target)
        finall_s = self._compute_latent(s_face, s_hair)
        base_feature = self._build_base_feature(f_face, f_hair, target_mask)

        aux = self._resolve_masks(
            source,
            target,
            source_parsing,
            target_parsing,
            source_hair_mask,
            target_hair_mask,
            query_mask,
            source_ear_mask,
            presence_target,
        )
        source_01 = normalized_to_01(source)
        target_01 = normalized_to_01(target)

        source_hair_block = aux.get("source_hair_block_mask")
        query_mask = ensure_mask_4d(aux["query_mask"]).float()
        hair_safe_query_mask = query_mask
        if source_hair_block is not None:
            block_strength = max(0.0, min(1.0, float(getattr(self.args, "source_hair_block_strength", 0.95))))
            source_hair_block = ensure_mask_4d(source_hair_block).float()
            hair_safe_query_mask = query_mask * (1 - block_strength * source_hair_block).clamp(0, 1)

        high_floor = max(0.0, min(1.0, float(getattr(self.args, "source_hair_block_high_floor", 0.30))))
        if source_hair_block is not None and high_floor > 0:
            high_attenuation = (1 - block_strength * source_hair_block).clamp(min=high_floor, max=1)
            ear_detail_query_mask = query_mask * high_attenuation
        else:
            ear_detail_query_mask = hair_safe_query_mask

        source_earring_parser_hint = ensure_mask_4d(aux["source_earring_mask"]).float()
        source_earring_hint = source_earring_parser_hint
        weak_earring_masks = build_weak_earring_masks(
            source_01,
            aux.get("visible_ear_roi", query_mask),
            query_mask,
            source_earring_parser_hint,
            aux.get("source_hair_mask"),
            source_hair_block,
        )
        weak_dark_candidate_mask = weak_earring_masks.get(
            "earring_dark_candidate_mask",
            torch.zeros_like(weak_earring_masks["earring_confident_mask"]),
        )
        weak_confident_mask = weak_earring_masks["earring_confident_mask"]
        texture_query_boost = max(0.0, float(getattr(self.args, "earring_texture_query_boost", 0.75)))
        if texture_query_boost > 0:
            source_earring_hint = torch.clamp(
                source_earring_parser_hint + texture_query_boost * weak_confident_mask,
                0,
                1,
            )
        source_earring_raw_hint = source_earring_hint
        earring_reference_image = source_01
        earring_reference_mask = source_earring_hint
        earring_parser_reference_mask = source_earring_parser_hint
        earring_highlight_mask = weak_earring_masks["earring_highlight_mask"]
        earring_reference_raw_mask = earring_reference_mask
        earring_highlight_raw_mask = earring_highlight_mask
        if bool(getattr(self.args, "earring_align_to_target", True)):
            earring_reference_image, earring_reference_mask = build_aligned_earring_reference(
                source_01,
                source_earring_hint,
                aux.get("target_parsing"),
                aux.get("visible_ear_roi"),
                align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
                max_shift=int(getattr(self.args, "earring_align_max_shift", 26)),
                attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
            )
            _, earring_parser_reference_mask = build_aligned_earring_reference(
                source_01,
                source_earring_parser_hint,
                aux.get("target_parsing"),
                aux.get("visible_ear_roi"),
                align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
                max_shift=int(getattr(self.args, "earring_align_max_shift", 26)),
                attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
            )
            _, earring_highlight_mask = build_aligned_earring_reference(
                source_01,
                earring_highlight_mask,
                aux.get("target_parsing"),
                aux.get("visible_ear_roi"),
                align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
                max_shift=int(getattr(self.args, "earring_align_max_shift", 26)),
                attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
            )
            _, weak_confident_mask = build_aligned_earring_reference(
                source_01,
                weak_confident_mask,
                aux.get("target_parsing"),
                aux.get("visible_ear_roi"),
                align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
                max_shift=int(getattr(self.args, "earring_align_max_shift", 26)),
                attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
            )
            _, weak_dark_candidate_mask = build_aligned_earring_reference(
                source_01,
                weak_dark_candidate_mask,
                aux.get("target_parsing"),
                aux.get("visible_ear_roi"),
                align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
                max_shift=int(getattr(self.args, "earring_align_max_shift", 26)),
                attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
            )
            source_earring_hint = torch.clamp(0.25 * source_earring_hint + earring_reference_mask, 0, 1)

        earring_reference_raw_mask = earring_reference_mask
        earring_highlight_raw_mask = earring_highlight_mask
        if bool(getattr(self.args, "earring_strict_core_enable", True)):
            parser_support = dilate_mask(
                earring_parser_reference_mask,
                int(getattr(self.args, "earring_parser_support_dilate", 5)),
            )
            raw_query_support = dilate_mask(query_mask, int(getattr(self.args, "earring_recall_query_dilate", 5)))
            weak_recall_seed = torch.clamp(
                weak_confident_mask
                + weak_dark_candidate_mask
                + earring_highlight_mask
                + float(getattr(self.args, "earring_raw_query_recall_weight", 0.35)) * raw_query_support,
                0,
                1,
            )
            highlight_support = dilate_mask(
                torch.clamp(earring_reference_mask + earring_parser_reference_mask + weak_recall_seed, 0, 1),
                int(getattr(self.args, "earring_highlight_support_dilate", 5)),
            )
            core_candidate = torch.clamp(
                earring_parser_reference_mask
                + earring_reference_mask * torch.clamp(parser_support + earring_highlight_mask, 0, 1)
                + float(getattr(self.args, "earring_weak_recall_weight", 0.55)) * weak_recall_seed * raw_query_support
                + earring_highlight_mask * highlight_support,
                0,
                1,
            )
            earring_core_mask = build_strict_earring_core_mask(
                earring_reference_image,
                core_candidate,
                fallback_mask=torch.clamp(
                    earring_parser_reference_mask
                    + float(getattr(self.args, "earring_fallback_recall_weight", 0.45)) * weak_recall_seed * raw_query_support,
                    0,
                    1,
                ),
                min_high=float(getattr(self.args, "earring_core_min_high", 0.014)),
                min_chroma=float(getattr(self.args, "earring_core_min_chroma", 0.038)),
                min_contrast=float(getattr(self.args, "earring_core_min_contrast", 0.018)),
                dilate=int(getattr(self.args, "earring_core_dilate", 3)),
            )
        else:
            earring_core_mask = earring_reference_mask

        dark_reject_mask = build_flat_dark_pollution_mask(
            earring_reference_image,
            earring_core_mask,
            max_gray=float(getattr(self.args, "earring_dark_reject_max_gray", 0.22)),
            max_chroma=float(getattr(self.args, "earring_dark_reject_max_chroma", 0.07)),
            max_high=float(getattr(self.args, "earring_dark_reject_max_high", 0.018)),
            dilate=int(getattr(self.args, "earring_dark_reject_dilate", 3)),
        )
        parser_keep = dilate_mask(earring_parser_reference_mask, int(getattr(self.args, "earring_parser_keep_dilate", 3)))
        dark_keep_support = dilate_mask(
            torch.clamp(earring_parser_reference_mask + earring_reference_raw_mask + earring_highlight_mask, 0, 1),
            int(getattr(self.args, "earring_dark_keep_dilate", 3)),
        )
        dark_reject_mask = dark_reject_mask * (1 - dark_keep_support).clamp(0, 1)
        hair_reject_mask = torch.zeros_like(earring_core_mask)
        if source_hair_block is not None:
            hair_reject_mask = resize_mask(source_hair_block, earring_core_mask.shape[-2:]) * (1 - parser_keep).clamp(0, 1)
        target_hair = aux.get("target_hair_mask")
        target_hair_reject_mask = torch.zeros_like(earring_core_mask)
        if target_hair is not None:
            target_hair_reject_mask = resize_mask(target_hair, earring_core_mask.shape[-2:]) * (1 - parser_keep).clamp(0, 1)

        safe_support = dilate_mask(aux.get("visible_ear_roi", query_mask), int(getattr(self.args, "earring_safe_roi_dilate", 7)))
        safe_reject = torch.clamp(dark_reject_mask + hair_reject_mask + target_hair_reject_mask, 0, 1)
        earring_safe_mask = earring_core_mask * safe_support * (1 - safe_reject).clamp(0, 1)
        recall_safe_fallback = torch.clamp(
            earring_parser_reference_mask
            + float(getattr(self.args, "earring_safe_recall_weight", 0.45))
            * weak_confident_mask
            * dilate_mask(query_mask, int(getattr(self.args, "earring_recall_query_dilate", 5))),
            0,
            1,
        )
        parser_safe_mask = recall_safe_fallback * safe_support * (1 - safe_reject).clamp(0, 1)
        safe_empty = earring_safe_mask.flatten(1).sum(dim=1).view(-1, 1, 1, 1) <= 1
        earring_safe_mask = torch.where(safe_empty, parser_safe_mask, earring_safe_mask).clamp(0, 1)
        earring_highlight_mask = earring_highlight_mask * dilate_mask(
            earring_safe_mask,
            int(getattr(self.args, "earring_highlight_keep_dilate", 3)),
        )
        earring_reference_mask = earring_safe_mask
        source_earring_hint = torch.clamp(
            earring_safe_mask
            + float(getattr(self.args, "earring_raw_hint_residual", 0.08))
            * source_earring_raw_hint
            * dilate_mask(earring_safe_mask, int(getattr(self.args, "earring_weak_support_dilate", 5))),
            0,
            1,
        )

        earring_query_dilate = int(getattr(self.args, "earring_query_dilate", 11))
        earring_query_boost = max(0.0, float(getattr(self.args, "earring_query_boost", 1.5)))
        earring_query_hint = source_earring_hint
        if earring_query_boost > 0:
            earring_query_hint = dilate_mask(earring_query_hint, earring_query_dilate)
            ear_detail_query_mask = torch.clamp(
                ear_detail_query_mask + earring_query_boost * earring_query_hint,
                0,
                1,
            )

        weak_supported = weak_confident_mask * dilate_mask(
            earring_reference_mask,
            int(getattr(self.args, "earring_weak_support_dilate", 5)),
        )
        object_seed = torch.clamp(
            earring_reference_mask
            + float(getattr(self.args, "earring_weak_object_weight", 0.35)) * weak_supported
            + earring_highlight_mask,
            0,
            1,
        )
        object_dilate = int(getattr(self.args, "earring_object_dilate", 9))
        object_support_dilate = int(getattr(self.args, "earring_object_support_dilate", 13))
        object_support = torch.clamp(
            dilate_mask(aux.get("visible_ear_roi", query_mask), 7)
            + dilate_mask(earring_reference_mask, object_support_dilate)
            + earring_query_hint,
            0,
            1,
        )
        strict_object_support = torch.clamp(
            dilate_mask(earring_safe_mask, object_support_dilate) + earring_highlight_mask,
            0,
            1,
        )
        earring_object_mask = dilate_mask(object_seed, object_dilate) * object_support * strict_object_support
        earring_object_mask = earring_object_mask * (1 - resize_mask(safe_reject, earring_object_mask.shape[-2:])).clamp(0, 1)

        earring_recall_floor_mask = torch.clamp(
            earring_safe_mask + float(getattr(self.args, "earring_core_floor_weight", 0.35)) * earring_core_mask,
            0,
            1,
        )
        parser_injection_support = dilate_mask(earring_parser_reference_mask, int(getattr(self.args, "earring_parser_keep_dilate", 3)))
        raw_hint_near_core = earring_reference_raw_mask * dilate_mask(
            torch.clamp(earring_safe_mask + earring_core_mask + earring_highlight_mask + weak_dark_candidate_mask, 0, 1),
            int(getattr(self.args, "earring_weak_support_dilate", 5)),
        )
        injection_candidate_mask = torch.clamp(
            earring_safe_mask
            + earring_core_mask
            + earring_highlight_mask
            + weak_dark_candidate_mask
            + float(getattr(self.args, "earring_raw_hint_residual", 0.12)) * raw_hint_near_core,
            0,
            1,
        )
        align_to_target = bool(getattr(self.args, "earring_align_to_target", True))
        reference_face_reject_mask = self._parsing_label_mask(
            aux.get("target_parsing") if align_to_target else aux.get("source_parsing"),
            FACE_SURFACE_LABELS,
        )
        earring_injection_seed_mask = self._build_material_injection_seed(
            earring_reference_image,
            injection_candidate_mask,
            parser_injection_support * earring_parser_reference_mask,
            aux.get("earring_search_roi", aux.get("visible_ear_roi", query_mask)),
            face_reject_mask=reference_face_reject_mask,
        )
        earring_injection_seed_mask = earring_injection_seed_mask * (1 - resize_mask(safe_reject, earring_injection_seed_mask.shape[-2:])).clamp(0, 1)
        earring_composite_seed_mask = self._build_guarded_composite_seed(
            earring_reference_image,
            earring_parser_reference_mask,
            earring_highlight_mask,
            weak_dark_candidate_mask,
            earring_injection_seed_mask,
            aux.get("earring_search_roi", aux.get("visible_ear_roi", query_mask)),
            face_reject_mask=reference_face_reject_mask,
        )
        earring_composite_seed_mask = earring_composite_seed_mask * (1 - resize_mask(safe_reject, earring_composite_seed_mask.shape[-2:])).clamp(0, 1)
        injection_fine_mask = self._build_injection_mask(earring_injection_seed_mask)

        earring_prior_mask = torch.clamp(earring_injection_seed_mask + earring_composite_seed_mask, 0, 1)
        prior_reference_dilate = int(getattr(self.args, "earring_prior_reference_dilate", 1))
        prior_reference_mask = earring_prior_mask
        if prior_reference_dilate > 1:
            prior_reference_mask = dilate_mask(prior_reference_mask, prior_reference_dilate)
        source_for_prior = source_01 * (1 - prior_reference_mask) + earring_reference_image * prior_reference_mask

        prior_query_dilate = int(getattr(self.args, "earring_prior_query_dilate", 5))
        ear_prior_query_mask = earring_prior_mask
        if prior_query_dilate > 1:
            ear_prior_query_mask = dilate_mask(ear_prior_query_mask, prior_query_dilate)
        ear_prior_query_mask = torch.clamp(ear_prior_query_mask + earring_prior_mask, 0, 1)

        hf_outputs = self.hf_extractor(source_for_prior, ear_prior_query_mask, ear_prior_query_mask)
        mask_source_earring = torch.clamp(
            earring_prior_mask
            + float(getattr(self.args, "earring_raw_hint_residual", 0.12))
            * source_earring_raw_hint
            * dilate_mask(earring_prior_mask, int(getattr(self.args, "earring_weak_support_dilate", 5))),
            0,
            1,
        )
        mask_outputs = self.mask_refresher(
            source_for_prior,
            target_01,
            hf_outputs["high_energy"],
            ear_prior_query_mask,
            mask_source_earring,
        )
        earring_fine_mask_floor = max(0.0, min(1.0, float(getattr(self.args, "earring_fine_mask_floor", 0.12))))
        if earring_fine_mask_floor > 0:
            floor_mask = torch.clamp(earring_prior_mask + earring_fine_mask_floor * earring_recall_floor_mask, 0, 1)
            floor_mask = resize_mask(floor_mask, mask_outputs["fine_mask"].shape[-2:])
            mask_outputs["learned_fine_mask"] = mask_outputs["fine_mask"]
            mask_outputs["earring_fine_floor_mask"] = floor_mask
            mask_outputs["fine_mask"] = torch.clamp(
                mask_outputs["fine_mask"] + earring_fine_mask_floor * floor_mask,
                0,
                1,
            )
        prior_feature, brightness_outputs = self.brightness_reestimator(
            target_01,
            hf_outputs["prior_feature"],
            injection_fine_mask,
            ear_prior_query_mask,
        )
        finall_f, fine_mask_64 = self.ear_injector_64(base_feature, prior_feature, injection_fine_mask)

        cleanup_face_mask = None
        if self.use_cleanup_face_refiner:
            cleanup_face_mask = self._build_cleanup_face_mask(
                aux,
                cleanup_masks,
                revealed_skin_mask,
                earring_reference_mask,
                source_01.shape[-2:],
            )
        if cleanup_face_mask is not None:
            finall_f, cleanup_face_mask_64, cleanup_face_delta_64 = self.cleanup_face_refiner_64(
                finall_f,
                source_01,
                target_01,
                cleanup_face_mask,
            )
        else:
            cleanup_face_mask_64 = None
            cleanup_face_delta_64 = None

        aux["raw_query_mask"] = query_mask
        aux["hair_safe_query_mask"] = hair_safe_query_mask
        aux["raw_ear_detail_query_mask"] = ear_detail_query_mask
        aux["ear_detail_query_mask"] = ear_prior_query_mask
        aux["earring_reference_image"] = earring_reference_image
        aux["earring_prior_image"] = source_for_prior
        aux["earring_prior_mask"] = earring_prior_mask
        aux["earring_reference_mask"] = earring_reference_mask
        aux["earring_reference_raw_mask"] = earring_reference_raw_mask
        aux["earring_highlight_raw_mask"] = earring_highlight_raw_mask
        aux["earring_face_reject_mask"] = reference_face_reject_mask
        aux["earring_core_mask"] = earring_core_mask
        aux["earring_safe_mask"] = earring_safe_mask
        aux["earring_recall_floor_mask"] = earring_recall_floor_mask
        aux["earring_dark_candidate_mask"] = weak_dark_candidate_mask
        aux["earring_dark_reject_mask"] = dark_reject_mask
        aux["earring_hair_reject_mask"] = torch.clamp(hair_reject_mask + target_hair_reject_mask, 0, 1)
        aux["source_earring_parser_mask"] = source_earring_parser_hint
        aux["earring_parser_reference_mask"] = earring_parser_reference_mask
        aux["earring_query_hint_mask"] = earring_query_hint
        aux["source_earring_raw_mask"] = source_earring_raw_hint
        aux["source_earring_mask"] = mask_source_earring
        aux["earring_object_mask"] = earring_object_mask
        weak_confident_near_safe = (
            weak_confident_mask
            * dilate_mask(earring_safe_mask, int(getattr(self.args, "earring_weak_support_dilate", 5)))
            * (1 - resize_mask(safe_reject, weak_confident_mask.shape[-2:])).clamp(0, 1)
        )
        aux["earring_confident_mask"] = torch.clamp(
            earring_safe_mask + earring_object_mask + 0.35 * weak_confident_near_safe,
            0,
            1,
        )
        aux["earring_highlight_mask"] = earring_highlight_mask
        aux["runtime_earring_confident_mask"] = aux["earring_confident_mask"]
        aux["runtime_earring_highlight_mask"] = aux["earring_highlight_mask"]
        aux["mask_source_earring"] = mask_source_earring
        aux["earring_injection_seed_mask"] = earring_injection_seed_mask
        aux["earring_composite_seed_mask"] = earring_composite_seed_mask
        aux.update(hf_outputs)
        aux.update(mask_outputs)
        aux.update(brightness_outputs)
        aux["adjusted_prior_feature"] = prior_feature
        aux["injection_fine_mask"] = injection_fine_mask
        aux["injected_fine_mask_64"] = fine_mask_64
        aux["cleanup_face_mask"] = cleanup_face_mask
        aux["cleanup_face_mask_64"] = cleanup_face_mask_64
        aux["cleanup_face_delta_64"] = cleanup_face_delta_64
        aux["cleanup_source_01"] = source_01
        aux["cleanup_target_01"] = target_01
        aux["HT_E"] = ensure_mask_4d(HT_E).float() if HT_E is not None else None
        return finall_s, finall_f, aux

    def _apply_output_earring_guard(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not bool(getattr(self.args, "earring_output_guard", True)):
            return image

        base_image = aux.get("base_image_full")
        if base_image is None:
            return image

        out_size = image.shape[-2:]
        base_image = base_image.to(device=image.device, dtype=image.dtype)
        if base_image.ndim == 3:
            base_image = base_image.unsqueeze(0)
        if base_image.shape[-2:] != out_size:
            base_image = F.interpolate(base_image, size=out_size, mode="bilinear", align_corners=False)

        def collect_mask(keys: tuple[str, ...]) -> torch.Tensor | None:
            mask = None
            for key in keys:
                value = aux.get(key)
                if value is None:
                    continue
                value = resize_mask(value.to(device=image.device, dtype=image.dtype), out_size).clamp(0, 1)
                mask = value if mask is None else torch.clamp(mask + value, 0, 1)
            return mask

        precise_guard = collect_mask(
            (
                "injection_fine_mask",
                "injected_fine_mask_64",
                "injected_fine_mask_128",
                "earring_injection_seed_mask_full",
                "earring_composite_seed_mask_full",
                "earring_injection_seed_mask",
                "earring_composite_seed_mask",
            )
        )
        fallback_guard = collect_mask(
            (
                "earring_parser_reference_mask",
                "source_earring_parser_mask",
                "earring_reference_mask",
                "source_earring_mask",
                "earring_safe_mask",
                "earring_object_mask",
            )
        )
        broad_guard = precise_guard if precise_guard is not None else fallback_guard
        if broad_guard is None:
            alpha = torch.zeros(image.size(0), 1, *out_size, device=image.device, dtype=image.dtype)
            aux["earring_output_guard_alpha"] = alpha
            return base_image

        face_reject = aux.get("earring_face_reject_mask")
        if face_reject is not None and bool(getattr(self.args, "earring_output_guard_exclude_face", True)):
            face_reject = resize_mask(face_reject.to(device=image.device, dtype=image.dtype), out_size).clamp(0, 1)
            face_keep = collect_mask(
                (
                    "earring_parser_reference_mask",
                    "source_earring_parser_mask",
                    "earring_injection_seed_mask",
                    "earring_composite_seed_mask",
                )
            )
            if face_keep is None:
                face_keep = torch.zeros_like(broad_guard)
            keep_dilate = int(getattr(self.args, "earring_output_guard_face_keep_dilate", 3))
            if keep_dilate > 1:
                face_keep = dilate_mask(face_keep, keep_dilate)
            face_block = face_reject * (1 - face_keep).clamp(0, 1)
            broad_guard = broad_guard * (1 - face_block).clamp(0, 1)
            if precise_guard is not None:
                precise_guard = precise_guard * (1 - face_block).clamp(0, 1)
            if fallback_guard is not None:
                fallback_guard = fallback_guard * (1 - face_block).clamp(0, 1)

        min_area = float(getattr(self.args, "earring_output_guard_min_area", 4.0))
        max_area_frac = float(getattr(self.args, "earring_output_guard_max_area_frac", 0.035))
        fallback_max_area_frac = float(getattr(self.args, "earring_output_guard_fallback_max_area_frac", 0.055))

        def valid_area(mask: torch.Tensor, area_limit: float | None = None) -> torch.Tensor:
            pixel_area = mask.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
            valid = pixel_area >= min_area
            area_limit = max_area_frac if area_limit is None else area_limit
            if area_limit > 0:
                valid = valid & (mask.flatten(1).mean(dim=1).view(-1, 1, 1, 1) <= area_limit)
            return valid

        if precise_guard is not None:
            precise_valid = valid_area(precise_guard)
            if fallback_guard is not None:
                fallback_valid = valid_area(fallback_guard, fallback_max_area_frac)
                broad_guard = torch.where(precise_valid, precise_guard, fallback_guard)
                broad_guard = torch.where(precise_valid | fallback_valid, broad_guard, torch.zeros_like(broad_guard))
            else:
                broad_guard = torch.where(precise_valid, precise_guard, torch.zeros_like(precise_guard))
        else:
            has_guard = valid_area(broad_guard)
            broad_guard = torch.where(has_guard, broad_guard, torch.zeros_like(broad_guard))

        has_guard = (broad_guard.flatten(1).sum(dim=1) >= min_area).view(-1, 1, 1, 1)
        broad_guard = torch.where(has_guard, broad_guard, torch.zeros_like(broad_guard))

        dilate = int(getattr(self.args, "earring_output_guard_dilate", 21))
        if dilate > 1:
            broad_guard = dilate_mask(broad_guard, dilate)
        blur = int(getattr(self.args, "earring_output_guard_blur", 11))
        sigma = float(getattr(self.args, "earring_output_guard_sigma", 3.0))
        alpha = gaussian_blur(broad_guard, kernel_size=blur, sigma=sigma).clamp(0, 1)
        aux["earring_output_guard_alpha"] = alpha
        return base_image * (1 - alpha) + image * alpha

    def _apply_output_face_guard(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not bool(getattr(self.args, "face_output_guard", True)):
            return image

        source_face = aux.get("source_full_01", aux.get("cleanup_source_01"))
        if source_face is None:
            return image

        out_size = image.shape[-2:]
        source_face = source_face.to(device=image.device, dtype=image.dtype)
        if source_face.ndim == 3:
            source_face = source_face.unsqueeze(0)
        if source_face.shape[-2:] != out_size:
            source_face = F.interpolate(source_face, size=out_size, mode="bilinear", align_corners=False)
        source_face = source_face.clamp(0, 1)

        source_mask = self._parsing_label_mask(aux.get("source_parsing"), FACE_OUTPUT_GUARD_LABELS)
        target_mask = self._parsing_label_mask(aux.get("target_parsing"), FACE_OUTPUT_GUARD_LABELS)
        if source_mask is None and target_mask is None:
            return image
        if source_mask is None:
            face_mask = target_mask
        elif target_mask is None:
            face_mask = source_mask
        else:
            face_mask = source_mask * resize_mask(target_mask, source_mask.shape[-2:])
        face_mask = resize_mask(face_mask.to(device=image.device, dtype=image.dtype), out_size).clamp(0, 1)

        for reject_key in (
            "source_hair_mask",
            "target_hair_mask",
            "source_hair_block_mask",
            "earring_output_guard_alpha",
            "earring_composite_alpha",
            "earring_reference_mask",
            "earring_injection_seed_mask",
            "earring_composite_seed_mask",
            "target_earring_mask",
            "cleanup_face_mask",
            "revealed_skin_mask",
        ):
            reject_mask = aux.get(reject_key)
            if reject_mask is None:
                continue
            reject_mask = resize_mask(reject_mask.to(device=image.device, dtype=image.dtype), out_size).clamp(0, 1)
            reject_dilate = 3
            if "earring" in reject_key or reject_key == "earring_output_guard_alpha":
                reject_dilate = int(getattr(self.args, "face_output_guard_exclude_earring_dilate", 9))
            if reject_dilate > 1:
                reject_mask = dilate_mask(reject_mask, reject_dilate)
            face_mask = face_mask * (1 - reject_mask).clamp(0, 1)

        min_area = float(getattr(self.args, "face_output_guard_min_area", 128.0))
        has_face = (face_mask.flatten(1).sum(dim=1) >= min_area).view(-1, 1, 1, 1)
        face_mask = torch.where(has_face, face_mask, torch.zeros_like(face_mask))
        if face_mask.flatten(1).sum().item() <= 0:
            return image

        blur = int(getattr(self.args, "face_output_guard_blur", 13))
        sigma = float(getattr(self.args, "face_output_guard_sigma", 4.0))
        strength = max(0.0, min(1.0, float(getattr(self.args, "face_output_guard_strength", 0.85))))
        alpha = gaussian_blur(face_mask, kernel_size=blur, sigma=sigma).clamp(0, 1) * strength
        aux["face_output_guard_alpha"] = alpha
        image_01 = ((image + 1) / 2).clamp(0, 1)
        guarded = source_face * alpha + image_01 * (1 - alpha)
        return guarded.clamp(0, 1) * 2 - 1

    def _apply_guarded_earring_composite(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        reference_image = aux.get("earring_reference_image_full", aux.get("earring_reference_image"))
        reference_mask = aux.get("earring_reference_mask_full", aux.get("earring_reference_mask"))
        if reference_image is None or reference_mask is None:
            return image

        out_size = image.shape[-2:]
        reference_image = reference_image.to(device=image.device, dtype=image.dtype)
        if reference_image.ndim == 3:
            reference_image = reference_image.unsqueeze(0)
        reference_image = F.interpolate(reference_image, size=out_size, mode="bilinear", align_corners=False).clamp(0, 1)

        reference_mask = resize_mask(reference_mask.to(device=image.device, dtype=image.dtype), out_size)
        core_mask = aux.get(
            "earring_composite_seed_mask_full",
            aux.get(
                "earring_composite_seed_mask",
                aux.get(
                    "earring_injection_seed_mask_full",
                    aux.get(
                        "earring_injection_seed_mask",
                        aux.get("earring_core_mask_full", aux.get("earring_core_mask")),
                    ),
                ),
            ),
        )
        if core_mask is not None:
            core_mask = resize_mask(core_mask.to(device=image.device, dtype=image.dtype), out_size)
            restricted_reference_mask = torch.clamp(reference_mask + core_mask, 0, 1) * core_mask
            min_core_area = float(getattr(self.args, "earring_composite_seed_min_area", 8.0))
            has_core = (core_mask.flatten(1).sum(dim=1) >= min_core_area).view(-1, 1, 1, 1)
            reference_mask = torch.where(has_core, restricted_reference_mask, reference_mask)
        visible_ear_roi = aux.get("visible_ear_roi")
        if visible_ear_roi is not None and bool(getattr(self.args, "earring_composite_restrict_visible_roi", False)):
            reference_mask = reference_mask * dilate_mask(resize_mask(visible_ear_roi.to(image.device), out_size), 7)

        face_reject = aux.get("earring_face_reject_mask")
        if face_reject is not None and bool(getattr(self.args, "earring_composite_exclude_face", True)):
            face_reject = resize_mask(face_reject.to(image.device), out_size).clamp(0, 1)
            face_keep = None
            for keep_key in (
                "earring_parser_reference_mask",
                "source_earring_parser_mask",
                "earring_injection_seed_mask",
            ):
                keep_value = aux.get(keep_key)
                if keep_value is None:
                    continue
                keep_value = resize_mask(keep_value.to(image.device, dtype=image.dtype), out_size).clamp(0, 1)
                face_keep = keep_value if face_keep is None else torch.clamp(face_keep + keep_value, 0, 1)
            if face_keep is None:
                face_keep = torch.zeros_like(reference_mask)
            keep_dilate = int(getattr(self.args, "earring_composite_face_keep_dilate", 3))
            if keep_dilate > 1:
                face_keep = dilate_mask(face_keep, keep_dilate)
            face_block = face_reject * (1 - face_keep).clamp(0, 1)
            reference_mask = reference_mask * (1 - face_block).clamp(0, 1)

        source_hair_block = aux.get("source_hair_block_mask")
        if bool(getattr(self.args, "earring_composite_exclude_hair_block", False)) and source_hair_block is not None:
            hair_block = resize_mask(source_hair_block.to(image.device), out_size)
            reference_mask = reference_mask * (1 - hair_block).clamp(0, 1)

        target_hair = aux.get("target_hair_mask")
        if target_hair is not None and bool(getattr(self.args, "earring_composite_exclude_target_hair", False)):
            reference_mask = reference_mask * (1 - resize_mask(target_hair.to(image.device), out_size)).clamp(0, 1)

        if bool(getattr(self.args, "earring_composite_strict_mask", True)):
            high_energy = high_pass_filter(reference_image).abs().mean(dim=1, keepdim=True)
            chroma = reference_image.amax(dim=1, keepdim=True) - reference_image.amin(dim=1, keepdim=True)
            gray = rgb_to_gray(reference_image)
            local_gray = gaussian_blur(gray, kernel_size=15, sigma=4.0)
            bright = (gray - local_gray).clamp_min(0)
            dark = (local_gray - gray).clamp_min(0)
            contrast = (gray - local_gray).abs()
            strict_seed = (
                (high_energy > float(getattr(self.args, "earring_composite_min_high", 0.018))).float()
                * torch.clamp(
                    (chroma > float(getattr(self.args, "earring_composite_min_chroma", 0.045))).float()
                    + (contrast > float(getattr(self.args, "earring_composite_min_contrast", 0.022))).float()
                    + (bright > float(getattr(self.args, "earring_composite_min_contrast", 0.022)) * 0.65).float()
                    + (dark > float(getattr(self.args, "earring_composite_min_contrast", 0.022)) * 0.65).float(),
                    0,
                    1,
                )
            )
            strict_seed = dilate_mask(strict_seed * reference_mask, 3) * reference_mask
            if strict_seed.flatten(1).amax(dim=1).sum().item() > 0:
                reference_mask = strict_seed

        max_area_frac = float(getattr(self.args, "earring_composite_max_area_frac", 0.012))
        if max_area_frac > 0:
            area_frac = reference_mask.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
            reference_mask = torch.where(area_frac > max_area_frac, torch.zeros_like(reference_mask), reference_mask)

        if reference_mask.flatten(1).amax(dim=1).sum().item() <= 0:
            return image

        kernel = int(getattr(self.args, "earring_composite_feather", 3))
        sigma = float(getattr(self.args, "earring_composite_sigma", 1.2))
        strength = max(0.0, min(1.0, float(getattr(self.args, "earring_composite_strength", 0.85))))
        alpha = gaussian_blur(reference_mask, kernel_size=kernel, sigma=sigma).clamp(0, 1) * strength
        image_01 = ((image + 1) / 2).clamp(0, 1)
        composited = image_01 * (1 - alpha) + reference_image * alpha
        aux["earring_composite_alpha"] = alpha
        return composited.clamp(0, 1) * 2 - 1

    def render_refined(
        self,
        generator,
        latent_s: torch.Tensor,
        latent_f_64: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        feature_128, skip = generator(
            [latent_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=5,
            layer_in=latent_f_64,
        )

        if aux is not None and aux.get("adjusted_prior_feature") is not None and aux.get("fine_mask") is not None:
            prior_128 = F.interpolate(
                aux["adjusted_prior_feature"],
                size=feature_128.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            render_mask = aux.get("injection_fine_mask", aux["fine_mask"])
            feature_128, fine_mask_128 = self.ear_injector_128(feature_128, prior_128, render_mask)
            aux["injected_fine_mask_128"] = fine_mask_128

        if self.use_cleanup_face_refiner and aux is not None and aux.get("cleanup_face_mask") is not None:
            feature_128, cleanup_face_mask_128, cleanup_face_delta_128 = self.cleanup_face_refiner_128(
                feature_128,
                aux["cleanup_source_01"],
                aux["cleanup_target_01"],
                aux["cleanup_face_mask"],
            )
            aux["cleanup_face_mask_128"] = cleanup_face_mask_128
            aux["cleanup_face_delta_128"] = cleanup_face_delta_128

        image, _ = generator(
            [latent_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=6,
            end_layer=8,
            layer_in=feature_128,
            skip=skip,
        )
        if aux is not None:
            image = self._apply_output_earring_guard(image, aux)
            image = self._apply_output_face_guard(image, aux)
        if aux is not None and bool(getattr(self.args, "earring_guarded_composite", False)):
            image = self._apply_guarded_earring_composite(image, aux)
        return image, aux
