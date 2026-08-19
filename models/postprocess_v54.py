from __future__ import annotations

import argparse
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoders import FeatureEncoderMult, FeatureiResnet, ModulationModule
from models.ear_modules_v54 import (
    BrightnessReEstimator,
    DynamicFineMaskRefresher,
    EarAnchoredQueryBuilder,
    FaceParsingHelperV54,
    HFDAGatedInjectionUnit,
    ShadowSuppressedHFExtractor,
    build_aligned_earring_reference,
    build_earring_evidence_score,
    build_ppe1_earring_locator_masks,
    build_source_earring_object_mask,
    build_source_earring_detection_roi,
    build_target_earring_safe_zone,
    build_weak_earring_masks,
    clean_earring_candidate_mask,
    dilate_mask,
    erode_mask,
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


def build_query_builder_compat(**kwargs) -> EarAnchoredQueryBuilder:
    accepted = inspect.signature(EarAnchoredQueryBuilder.__init__).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return EarAnchoredQueryBuilder(**filtered_kwargs)


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


class PostProcessModelV54(nn.Module):
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
        self.parsing_helper = FaceParsingHelperV54(parse_size=getattr(self.args, "ear_parse_size", 512))
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
            earring_lobe_dilate=getattr(self.args, "earring_lobe_dilate", 5),
            earring_lobe_down_shift=getattr(self.args, "earring_lobe_down_shift", 18),
            earring_outer_shift=getattr(self.args, "earring_outer_shift", 6),
            earring_query_floor=getattr(self.args, "earring_query_floor", 0.45),
            earring_search_radius_y=getattr(self.args, "earring_search_radius_y", 28),
            earring_search_radius_x=getattr(self.args, "earring_search_radius_x", 18),
            earring_search_attach_y_ratio=getattr(self.args, "earring_search_attach_y_ratio", 0.78),
        )
        self.hf_extractor = ShadowSuppressedHFExtractor(
            low_alpha=getattr(self.args, "ear_low_alpha", 0.15),
            feature_channels=feature_channels,
            blur_kernel=getattr(self.args, "ear_blur_kernel", 11),
            blur_sigma=getattr(self.args, "ear_blur_sigma", 3.0),
        )
        self.mask_refresher = DynamicFineMaskRefresher(
            hidden_channels=getattr(self.args, "ear_mask_hidden", 32),
            init_bias=getattr(self.args, "ear_mask_init_bias", -2.2),
        )
        self.brightness_reestimator = BrightnessReEstimator(feature_channels=feature_channels)
        self.ear_injector_64 = HFDAGatedInjectionUnit(
            base_channels=getattr(self.args, "ear_inject_channels_64", 512),
            prior_channels=feature_channels,
        )
        self.ear_injector_128 = HFDAGatedInjectionUnit(
            base_channels=getattr(self.args, "ear_inject_channels_128", 256),
            prior_channels=feature_channels,
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
            print(f"[PostProcessModelV54] Missing base keys: {len(relevant_missing)}")
            print(relevant_missing[:20])
        if result.unexpected_keys:
            print(f"[PostProcessModelV54] Unexpected checkpoint keys: {len(result.unexpected_keys)}")
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

    @staticmethod
    def _keep_top_score_area(mask: torch.Tensor, score: torch.Tensor, max_area_frac: float, min_area: int = 8) -> torch.Tensor:
        mask = ensure_mask_4d(mask).float().clamp(0, 1)
        if max_area_frac <= 0:
            return mask
        score = resize_mask(score, mask.shape[-2:]) if score.shape[-2:] != mask.shape[-2:] else ensure_mask_4d(score).float()
        height, width = mask.shape[-2:]
        max_area = max(int(min_area), int(round(height * width * float(max_area_frac))))
        limited = torch.zeros_like(mask)
        for batch_idx in range(mask.size(0)):
            current = mask[batch_idx:batch_idx + 1]
            count = int((current > 0.02).sum().item())
            if count <= max_area:
                limited[batch_idx:batch_idx + 1] = current
                continue
            valid = current.flatten() > 0.02
            flat_score = (score[batch_idx:batch_idx + 1] * current).flatten()
            flat_score = torch.where(valid, flat_score, torch.full_like(flat_score, -1.0))
            keep_count = min(count, max_area)
            top_indices = torch.topk(flat_score, keep_count).indices
            kept = torch.zeros_like(current).flatten()
            kept[top_indices] = current.flatten()[top_indices]
            limited[batch_idx:batch_idx + 1] = kept.view_as(current)
        return limited.clamp(0, 1)

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
        earring_confident_mask: torch.Tensor | None = None,
        earring_highlight_mask: torch.Tensor | None = None,
        earring_locator_mask: torch.Tensor | None = None,
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
        if earring_confident_mask is None:
            earring_confident_mask = kwargs.get("earring_confident_mask")
        if earring_highlight_mask is None:
            earring_highlight_mask = kwargs.get("earring_highlight_mask")
        if earring_locator_mask is None:
            earring_locator_mask = kwargs.get("earring_locator_mask")

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
        placement_roi = aux.get("earring_search_roi", aux.get("visible_ear_roi", aux["query_mask"]))
        placement_roi = resize_mask(placement_roi, source_01.shape[-2:])
        target_safe_zone = None
        placement_support = dilate_mask(placement_roi, int(getattr(self.args, "earring_placement_dilate", 13)))
        locator_placement_support = dilate_mask(
            placement_roi,
            max(
                int(getattr(self.args, "earring_placement_dilate", 13)),
                int(getattr(self.args, "earring_locator_placement_dilate", 25)),
            ),
        )

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

        source_earring_hint = ensure_mask_4d(aux["source_earring_mask"]).float()
        source_detection_roi = build_source_earring_detection_roi(
            aux.get("source_parsing"),
            source_earring_hint,
            source_01.shape[-2:],
        )
        weak_earring_masks = build_weak_earring_masks(
            source_01,
            source_detection_roi,
            source_detection_roi,
            source_earring_hint,
            aux.get("source_hair_mask"),
            source_hair_block,
            aux.get("source_parsing"),
        )
        source_object_mask = build_source_earring_object_mask(
            source_01,
            aux.get("source_parsing"),
            source_earring_hint,
            aux.get("source_hair_mask"),
            max_area_frac=0.018,
            min_area=4,
        ) * source_detection_roi
        locator_masks = build_ppe1_earring_locator_masks(
            source_01,
            aux.get("source_parsing"),
            None,
            aux.get("source_hair_mask"),
            ear_roi_dilate=int(getattr(self.args, "earring_locator_ear_roi_dilate", 11)),
            parser_weak_support_dilate=int(getattr(self.args, "earring_locator_parser_support_dilate", 5)),
            object_grow_iters=int(getattr(self.args, "earring_locator_object_grow_iters", 1)),
            component_keep_top=int(getattr(self.args, "earring_locator_component_keep_top", 8)),
            component_max_area_ratio=float(getattr(self.args, "earring_locator_component_max_area_ratio", 0.025)),
            weak_high_threshold=float(getattr(self.args, "earring_locator_weak_high", 0.018)),
            weak_chroma_threshold=float(getattr(self.args, "earring_locator_weak_chroma", 0.030)),
            weak_contrast_threshold=float(getattr(self.args, "earring_locator_weak_contrast", 0.016)),
        )
        source_locator_mask = locator_masks["earring_locator_mask"].clamp(0, 1)
        if earring_locator_mask is not None:
            source_locator_mask = torch.clamp(
                source_locator_mask + resize_mask(earring_locator_mask, source_01.shape[-2:]) * source_detection_roi,
                0,
                1,
            )
        source_locator_color_object = locator_masks.get("earring_locator_color_object")
        if source_locator_color_object is not None:
            source_locator_color_object = source_locator_color_object.clamp(0, 1)
            color_object_present = (source_locator_color_object.flatten(1).sum(dim=1) >= 3).view(-1, 1, 1, 1)
            color_connected_locator = torch.clamp(
                source_locator_color_object
                + source_locator_mask * dilate_mask(
                    source_locator_color_object,
                    int(getattr(self.args, "earring_source_color_connect_dilate", 11)),
                ),
                0,
                1,
            )
            source_locator_mask = torch.where(color_object_present, color_connected_locator, source_locator_mask)
        locator_present = (source_locator_mask.flatten(1).sum(dim=1) >= 3).view(-1, 1, 1, 1)
        source_object_mask = torch.where(locator_present, source_locator_mask, source_object_mask)
        weak_confident_mask = weak_earring_masks["earring_confident_mask"]
        weak_highlight_mask = weak_earring_masks["earring_highlight_mask"]
        weak_confident_mask = torch.clamp(weak_confident_mask + source_locator_mask, 0, 1)
        weak_highlight_mask = torch.clamp(weak_highlight_mask + 0.35 * source_locator_mask, 0, 1)
        weak_confident_mask = torch.where(locator_present, source_locator_mask, weak_confident_mask)
        weak_highlight_mask = torch.where(locator_present, 0.35 * source_locator_mask, weak_highlight_mask)
        if earring_confident_mask is not None:
            weak_confident_mask = torch.clamp(
                weak_confident_mask
                + resize_mask(earring_confident_mask, source_01.shape[-2:]) * source_detection_roi,
                0,
                1,
            )
        if earring_highlight_mask is not None:
            weak_highlight_mask = torch.clamp(
                weak_highlight_mask
                + resize_mask(earring_highlight_mask, source_01.shape[-2:]) * source_detection_roi,
                0,
                1,
            )
        weak_confident_mask = clean_earring_candidate_mask(
            weak_confident_mask,
            source_01,
            support_mask=source_detection_roi,
            query_mask=source_detection_roi,
            source_parsing=aux.get("source_parsing"),
            source_hair_mask=aux.get("source_hair_mask"),
            max_area_frac=0.030,
            min_area=4,
            dilate=3,
        )
        weak_highlight_mask = weak_highlight_mask * dilate_mask(weak_confident_mask, 3)

        source_earring_hint = torch.clamp(
            weak_earring_masks.get("source_earring_clean_mask", source_earring_hint) + source_object_mask + source_locator_mask,
            0,
            1,
        )
        source_earring_hint = torch.where(locator_present, source_locator_mask, source_earring_hint)
        texture_query_boost = max(0.0, float(getattr(self.args, "earring_texture_query_boost", 0.75)))
        if texture_query_boost > 0:
            source_earring_hint = torch.clamp(
                source_earring_hint + texture_query_boost * weak_confident_mask,
                0,
                1,
            )
        source_earring_hint = clean_earring_candidate_mask(
            source_earring_hint,
            source_01,
            support_mask=source_detection_roi,
            query_mask=source_detection_roi,
            source_parsing=aux.get("source_parsing"),
            source_hair_mask=aux.get("source_hair_mask"),
            max_area_frac=0.030,
            min_area=4,
            dilate=3,
        )
        source_earring_hint = torch.clamp(source_earring_hint + source_object_mask + source_locator_mask, 0, 1) * source_detection_roi
        source_earring_hint = torch.where(locator_present, source_locator_mask, source_earring_hint)
        width = source_earring_hint.size(-1)
        x_coords = torch.linspace(0, 1, width, device=source_earring_hint.device, dtype=source_earring_hint.dtype).view(1, 1, 1, width)
        left_half = (x_coords <= 0.5).float()
        right_half = 1 - left_half
        left_present = ((source_earring_hint * left_half).flatten(1).sum(dim=1) >= 4).float().view(-1, 1, 1, 1)
        right_present = ((source_earring_hint * right_half).flatten(1).sum(dim=1) >= 4).float().view(-1, 1, 1, 1)
        side_gate = left_present * left_half + right_present * right_half
        placement_roi = (placement_roi * side_gate).clamp(0, 1)
        placement_support = dilate_mask(placement_roi, int(getattr(self.args, "earring_placement_dilate", 13)))
        locator_placement_support = dilate_mask(
            placement_roi,
            max(
                int(getattr(self.args, "earring_placement_dilate", 13)),
                int(getattr(self.args, "earring_locator_placement_dilate", 25)),
            ),
        )
        target_safe_zone = build_target_earring_safe_zone(
            aux.get("target_parsing"),
            source_01.shape[-2:],
            placement_roi=placement_roi,
            outside_margin=int(getattr(self.args, "earring_target_safe_outside_margin", 34)),
            inside_margin=int(getattr(self.args, "earring_target_safe_inside_margin", 5)),
            face_inner_erode=int(getattr(self.args, "earring_target_safe_face_erode", 15)),
            placement_dilate=int(getattr(self.args, "earring_target_safe_placement_dilate", 5)),
        )
        if target_safe_zone is not None:
            target_safe_query_zone = dilate_mask(
                target_safe_zone,
                int(getattr(self.args, "earring_target_safe_query_dilate", 3)),
            )
            placement_support = (placement_support * target_safe_query_zone).clamp(0, 1)
            locator_placement_support = (locator_placement_support * target_safe_query_zone).clamp(0, 1)
        ear_detail_query_mask = (ear_detail_query_mask * placement_support).clamp(0, 1)

        source_earring_raw_hint = source_earring_hint
        earring_reference_image = source_01
        earring_reference_mask = source_earring_hint
        earring_highlight_mask = weak_highlight_mask
        aligned_confident_mask = weak_confident_mask
        aligned_locator_mask = source_locator_mask
        if bool(getattr(self.args, "earring_align_to_target", True)):
            earring_reference_image, earring_reference_mask = build_aligned_earring_reference(
                source_01,
                source_earring_hint,
                aux.get("target_parsing"),
                placement_roi,
                align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
                max_shift=int(getattr(self.args, "earring_align_max_shift", 26)),
                attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
            )
            _, aligned_confident_mask = build_aligned_earring_reference(
                source_01,
                weak_confident_mask,
                aux.get("target_parsing"),
                placement_roi,
                align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
                max_shift=int(getattr(self.args, "earring_align_max_shift", 26)),
                attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
            )
            _, earring_highlight_mask = build_aligned_earring_reference(
                source_01,
                earring_highlight_mask,
                aux.get("target_parsing"),
                placement_roi,
                align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
                max_shift=int(getattr(self.args, "earring_align_max_shift", 26)),
                attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
            )
            _, aligned_locator_mask = build_aligned_earring_reference(
                source_01,
                source_locator_mask,
                aux.get("target_parsing"),
                placement_roi,
                align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
                max_shift=int(getattr(self.args, "earring_align_max_shift", 26)),
                attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
            )
            earring_reference_mask = (earring_reference_mask * placement_support).clamp(0, 1)
            aligned_confident_mask = (aligned_confident_mask * placement_support).clamp(0, 1)
            earring_highlight_mask = (earring_highlight_mask * placement_support).clamp(0, 1)
            aligned_locator_mask = (aligned_locator_mask * locator_placement_support).clamp(0, 1)
            source_earring_hint = earring_reference_mask

        if target_safe_zone is not None:
            earring_reference_mask = (earring_reference_mask * target_safe_zone).clamp(0, 1)
            aligned_confident_mask = (aligned_confident_mask * target_safe_zone).clamp(0, 1)
            earring_highlight_mask = (earring_highlight_mask * target_safe_zone).clamp(0, 1)
            aligned_locator_mask = (aligned_locator_mask * target_safe_zone).clamp(0, 1)
            source_earring_hint = earring_reference_mask

        aligned_locator_present = (aligned_locator_mask.flatten(1).sum(dim=1) >= 3).view(-1, 1, 1, 1)
        earring_reference_mask = torch.clamp(earring_reference_mask + aligned_locator_mask, 0, 1) * locator_placement_support
        aligned_confident_mask = torch.clamp(aligned_confident_mask + aligned_locator_mask, 0, 1) * locator_placement_support
        earring_highlight_mask = torch.clamp(earring_highlight_mask + 0.35 * aligned_locator_mask, 0, 1) * locator_placement_support
        earring_reference_mask = torch.where(aligned_locator_present, aligned_locator_mask, earring_reference_mask)
        aligned_confident_mask = torch.where(aligned_locator_present, aligned_locator_mask, aligned_confident_mask)
        earring_highlight_mask = torch.where(aligned_locator_present, 0.35 * aligned_locator_mask, earring_highlight_mask)
        if target_safe_zone is not None:
            earring_reference_mask = (earring_reference_mask * target_safe_zone).clamp(0, 1)
            aligned_confident_mask = (aligned_confident_mask * target_safe_zone).clamp(0, 1)
            earring_highlight_mask = (earring_highlight_mask * target_safe_zone).clamp(0, 1)

        aligned_positive_mask = torch.clamp(
            earring_reference_mask + aligned_locator_mask + 0.5 * aligned_confident_mask,
            0,
            1,
        )
        aligned_positive_present = (aligned_positive_mask.flatten(1).sum(dim=1) >= 3).view(-1, 1, 1, 1)
        aligned_guard = dilate_mask(
            aligned_positive_mask,
            int(getattr(self.args, "earring_aligned_guard_dilate", 9)),
        )
        earring_write_gate = aligned_guard * locator_placement_support
        if target_safe_zone is not None:
            earring_write_gate = earring_write_gate * target_safe_zone
            earring_write_gate = torch.where(aligned_positive_present, earring_write_gate, target_safe_zone)
        else:
            earring_write_gate = torch.where(aligned_positive_present, earring_write_gate, locator_placement_support)
        earring_write_gate = earring_write_gate.clamp(0, 1)
        earring_reference_mask = (earring_reference_mask * earring_write_gate).clamp(0, 1)
        aligned_confident_mask = (aligned_confident_mask * earring_write_gate).clamp(0, 1)
        earring_highlight_mask = (earring_highlight_mask * earring_write_gate).clamp(0, 1)
        aligned_locator_mask = (aligned_locator_mask * earring_write_gate).clamp(0, 1)
        source_earring_hint = earring_reference_mask

        earring_query_dilate = int(getattr(self.args, "earring_query_dilate", 11))
        earring_query_boost = max(0.0, float(getattr(self.args, "earring_query_boost", 1.5)))
        if earring_query_boost > 0:
            source_earring_hint = dilate_mask(source_earring_hint, earring_query_dilate) * earring_write_gate
            ear_detail_query_mask = torch.clamp(
                ear_detail_query_mask + earring_query_boost * source_earring_hint,
                0,
                1,
            )

        object_seed = torch.clamp(
            earring_reference_mask
            + aligned_confident_mask
            + earring_highlight_mask,
            0,
            1,
        )
        object_dilate = min(int(getattr(self.args, "earring_object_dilate", 5)), 3)
        object_support_dilate = min(int(getattr(self.args, "earring_object_support_dilate", 9)), 5)
        object_support = torch.clamp(
            earring_reference_mask
            + aligned_confident_mask
            + earring_highlight_mask
            + dilate_mask(earring_reference_mask, object_support_dilate),
            0,
            1,
        ) * earring_write_gate
        earring_object_mask = dilate_mask(object_seed, object_dilate) * object_support
        earring_score = build_earring_evidence_score(earring_reference_image)["score"]
        max_object_area = min(float(getattr(self.args, "earring_object_max_area_frac", 0.018)), 0.014)
        compact_object = erode_mask(dilate_mask(earring_reference_mask, 7), 3) * earring_write_gate
        earring_object_mask = torch.clamp(earring_object_mask + compact_object, 0, 1)
        earring_object_mask = self._keep_top_score_area(
            earring_object_mask * earring_write_gate,
            earring_score + 0.65 * earring_reference_mask,
            max_object_area,
            min_area=8,
        )
        locator_object = dilate_mask(aligned_locator_mask, max(1, int(getattr(self.args, "earring_locator_object_dilate", 3))))
        locator_object = (locator_object * earring_write_gate).clamp(0, 1)
        earring_object_mask = torch.where(aligned_locator_present, locator_object, earring_object_mask)
        earring_object_mask = (earring_object_mask * earring_write_gate).clamp(0, 1)

        source_for_prior = source_01 * (1 - earring_reference_mask) + earring_reference_image * earring_reference_mask
        fine_gate_dilate = max(1, int(getattr(self.args, "earring_fine_gate_dilate", 7)))
        locator_gate_dilate = max(fine_gate_dilate, int(getattr(self.args, "earring_locator_fine_gate_dilate", 9)))
        fine_mask_gate = torch.clamp(
            dilate_mask(earring_object_mask, fine_gate_dilate)
            + dilate_mask(earring_reference_mask, fine_gate_dilate)
            + dilate_mask(aligned_locator_mask, locator_gate_dilate),
            0,
            1,
        )
        fine_mask_gate = (fine_mask_gate * locator_placement_support).clamp(0, 1)
        fine_mask_gate = (fine_mask_gate * earring_write_gate).clamp(0, 1)
        fine_gate_present = (fine_mask_gate.flatten(1).sum(dim=1) >= 3).view(-1, 1, 1, 1)
        hf_query_mask = ear_detail_query_mask * torch.clamp(
            placement_support + dilate_mask(earring_reference_mask, 7),
            0,
            1,
        )
        hf_query_mask = torch.where(fine_gate_present, hf_query_mask * fine_mask_gate, hf_query_mask)
        hf_outputs = self.hf_extractor(source_for_prior, hf_query_mask, hf_query_mask)
        mask_source_earring = torch.clamp(source_earring_hint, 0, 1)
        mask_outputs = self.mask_refresher(
            source_for_prior,
            target_01,
            hf_outputs["high_energy"],
            hf_query_mask,
            mask_source_earring,
        )
        fine_gate_for_mask = resize_mask(fine_mask_gate, mask_outputs["fine_mask"].shape[-2:])
        mask_outputs["learned_fine_raw_mask"] = mask_outputs["fine_mask"]
        mask_outputs["fine_mask"] = mask_outputs["fine_mask"] * fine_gate_for_mask
        mask_outputs["learned_fine_mask"] = mask_outputs["fine_mask"]
        earring_fine_mask_floor = max(0.0, min(1.0, float(getattr(self.args, "earring_fine_mask_floor", 0.45))))
        if earring_fine_mask_floor > 0:
            floor_mask = earring_object_mask
            floor_mask = resize_mask(floor_mask, mask_outputs["fine_mask"].shape[-2:])
            mask_outputs["earring_fine_floor_mask"] = floor_mask
            mask_outputs["fine_mask"] = torch.clamp(
                mask_outputs["fine_mask"] + earring_fine_mask_floor * floor_mask,
                0,
                1,
            )
            mask_outputs["fine_mask"] = mask_outputs["fine_mask"] * fine_gate_for_mask
        prior_feature, brightness_outputs = self.brightness_reestimator(
            target_01,
            hf_outputs["prior_feature"],
            mask_outputs["fine_mask"],
            ear_detail_query_mask,
        )
        finall_f, fine_mask_64 = self.ear_injector_64(base_feature, prior_feature, mask_outputs["fine_mask"])

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
        aux["ear_detail_query_mask"] = hf_query_mask
        aux["earring_reference_image"] = source_for_prior
        aux["earring_reference_mask"] = earring_reference_mask
        aux["source_earring_raw_mask"] = source_earring_raw_hint
        aux["source_earring_mask"] = mask_source_earring
        aux["source_earring_object_mask"] = source_object_mask
        aux["source_earring_locator_mask"] = source_locator_mask
        aux["earring_locator_mask"] = aligned_locator_mask
        aux["earring_locator_seed"] = locator_masks["earring_locator_seed"]
        aux["earring_locator_support"] = locator_masks["earring_locator_support"]
        aux["earring_locator_color_seed"] = locator_masks.get("earring_locator_color_seed")
        aux["earring_locator_color_object"] = locator_masks.get("earring_locator_color_object")
        aux["earring_locator_color_score"] = locator_masks.get("earring_locator_color_score")
        aux["target_earring_safe_zone"] = target_safe_zone
        aux["aligned_earring_positive_mask"] = aligned_positive_mask
        aux["earring_write_gate"] = earring_write_gate
        aux["earring_object_mask"] = earring_object_mask
        aux["earring_confident_mask"] = torch.clamp(
            0.35 * aligned_confident_mask + earring_reference_mask + earring_object_mask,
            0,
            1,
        )
        aux["earring_highlight_mask"] = earring_highlight_mask
        aux["fine_mask_gate"] = fine_mask_gate
        aux["runtime_earring_confident_mask"] = aux["earring_confident_mask"]
        aux["runtime_earring_highlight_mask"] = aux["earring_highlight_mask"]
        aux["mask_source_earring"] = mask_source_earring
        aux.update(hf_outputs)
        aux.update(mask_outputs)
        aux.update(brightness_outputs)
        aux["adjusted_prior_feature"] = prior_feature
        aux["injected_fine_mask_64"] = fine_mask_64
        aux["cleanup_face_mask"] = cleanup_face_mask
        aux["cleanup_face_mask_64"] = cleanup_face_mask_64
        aux["cleanup_face_delta_64"] = cleanup_face_delta_64
        aux["cleanup_source_01"] = source_01
        aux["cleanup_target_01"] = target_01
        aux["HT_E"] = ensure_mask_4d(HT_E).float() if HT_E is not None else None
        return finall_s, finall_f, aux

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
        reference_mask = (reference_mask > 0.12).float() * reference_mask
        locator_mask = aux.get("earring_locator_mask")
        if locator_mask is not None:
            locator_mask = resize_mask(locator_mask.to(device=image.device, dtype=image.dtype), out_size)
            reference_mask = torch.clamp(reference_mask + locator_mask, 0, 1)
        target_safe_zone = aux.get("target_earring_safe_zone")
        if target_safe_zone is not None:
            target_safe_zone = resize_mask(target_safe_zone.to(device=image.device, dtype=image.dtype), out_size)
            reference_mask = (reference_mask * target_safe_zone).clamp(0, 1)
        placement_roi = aux.get("earring_search_roi", aux.get("visible_ear_roi"))
        placement_gate = None
        if placement_roi is not None:
            placement_gate = dilate_mask(
                resize_mask(placement_roi.to(device=image.device, dtype=image.dtype), out_size),
                int(getattr(self.args, "earring_composite_placement_dilate", 17)),
            )
            restrict_strength = max(
                0.0,
                min(1.0, float(getattr(self.args, "earring_composite_search_restrict_strength", 0.85))),
            )
            reference_mask = reference_mask * (1 - restrict_strength + restrict_strength * placement_gate).clamp(0, 1)

        reference_support = dilate_mask(reference_mask, 9)
        for support_key in ("earring_locator_mask", "earring_object_mask", "earring_confident_mask", "earring_highlight_mask"):
            support_mask = aux.get(support_key)
            if support_mask is not None:
                support_mask = resize_mask(support_mask.to(image.device), out_size)
                support_mask = support_mask * reference_support
                if target_safe_zone is not None:
                    support_mask = support_mask * target_safe_zone
                if placement_gate is not None:
                    support_mask = support_mask * placement_gate
                reference_mask = torch.clamp(reference_mask + 0.35 * support_mask, 0, 1)
        visible_ear_roi = aux.get("visible_ear_roi")
        if visible_ear_roi is not None:
            visible_gate = dilate_mask(resize_mask(visible_ear_roi.to(image.device), out_size), 17)
            gate_strength = max(0.0, min(1.0, float(getattr(self.args, "earring_composite_visible_gate_strength", 0.20))))
            reference_mask = reference_mask * (1 - gate_strength + gate_strength * visible_gate).clamp(0, 1)

        source_hair_block = aux.get("source_hair_block_mask")
        if bool(getattr(self.args, "earring_composite_exclude_hair_block", True)) and source_hair_block is not None:
            hair_block = resize_mask(source_hair_block.to(image.device), out_size)
            gate_strength = max(0.0, min(1.0, float(getattr(self.args, "earring_composite_hair_block_gate_strength", 0.20))))
            reference_mask = reference_mask * (1 - gate_strength * hair_block).clamp(0, 1)

        target_hair = aux.get("target_hair_mask")
        if target_hair is not None:
            target_hair_mask = resize_mask(target_hair.to(image.device), out_size)
            gate_strength = max(0.0, min(1.0, float(getattr(self.args, "earring_composite_target_hair_gate_strength", 0.15))))
            reference_mask = reference_mask * (1 - gate_strength * target_hair_mask).clamp(0, 1)

        if bool(getattr(self.args, "earring_composite_strict_mask", False)):
            high_energy = high_pass_filter(reference_image).abs().mean(dim=1, keepdim=True)
            chroma = reference_image.amax(dim=1, keepdim=True) - reference_image.amin(dim=1, keepdim=True)
            gray = rgb_to_gray(reference_image)
            local_gray = gaussian_blur(gray, kernel_size=15, sigma=4.0)
            contrast = (gray - local_gray).abs()
            strict_seed = (
                (high_energy > float(getattr(self.args, "earring_composite_min_high", 0.018))).float()
                * torch.clamp(
                    (chroma > float(getattr(self.args, "earring_composite_min_chroma", 0.045))).float()
                    + (contrast > float(getattr(self.args, "earring_composite_min_contrast", 0.022))).float(),
                    0,
                    1,
                )
            )
            strict_seed = dilate_mask(strict_seed * reference_mask, 3) * reference_mask
            min_area = float(getattr(self.args, "earring_composite_strict_min_area", 16.0))
            strict_area = strict_seed.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
            reference_mask = torch.where(strict_area >= min_area, strict_seed, reference_mask)

        max_area_frac = float(getattr(self.args, "earring_composite_max_area_frac", 0.014))
        if max_area_frac > 0:
            evidence = build_earring_evidence_score(reference_image)
            reference_score = evidence["score"] * reference_mask
            reference_mask = self._keep_top_score_area(reference_mask, reference_score, max_area_frac, min_area=8)

        if reference_mask.flatten(1).amax(dim=1).sum().item() <= 0:
            return image

        kernel = int(getattr(self.args, "earring_composite_feather", 3))
        sigma = float(getattr(self.args, "earring_composite_sigma", 1.2))
        strength = max(0.0, min(1.0, float(getattr(self.args, "earring_composite_strength", 0.95))))
        alpha_floor = max(0.0, min(1.0, float(getattr(self.args, "earring_composite_alpha_floor", 0.35))))
        alpha = torch.clamp(
            gaussian_blur(reference_mask, kernel_size=kernel, sigma=sigma).clamp(0, 1) * strength
            + alpha_floor * reference_mask,
            0,
            1,
        )
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
            feature_128, fine_mask_128 = self.ear_injector_128(feature_128, prior_128, aux["fine_mask"])
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
        if aux is not None and bool(getattr(self.args, "earring_guarded_composite", False)):
            image = self._apply_guarded_earring_composite(image, aux)
        return image, aux
