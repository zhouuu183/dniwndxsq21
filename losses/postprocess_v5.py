from __future__ import annotations

import argparse
import inspect
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoders import FeatureEncoderMult, FeatureiResnet, ModulationModule
from models.ear_modules_v5 import (
    RAW_DETAIL_LABELS,
    RAW_FACE_SURFACE_LABELS,
    BrightnessReEstimator,
    DynamicFineMaskRefresher,
    EarAnchoredQueryBuilder,
    FaceParsingHelperV5,
    HFDAGatedInjectionUnit,
    ShadowSuppressedHFExtractor,
    build_revealed_skin_mask,
    build_weak_earring_masks,
    dilate_mask,
    enhance_query_with_earring_recall,
    ensure_mask_4d,
    erode_mask,
    gaussian_blur,
    high_pass_filter,
    low_pass_filter,
    normalized_to_01,
    parsing_label_mask,
    resize_mask,
)
from models.stylegan2.model import PixelNorm


def load_checkpoint_compat(path: str | os.PathLike, map_location="cpu"):
    old_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
    finally:
        pathlib.PosixPath = old_posix_path


def build_query_builder_compat(**kwargs) -> EarAnchoredQueryBuilder:
    accepted = inspect.signature(EarAnchoredQueryBuilder.__init__).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return EarAnchoredQueryBuilder(**filtered_kwargs)


class PostProcessModelV5(nn.Module):
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
        self.parsing_helper = FaceParsingHelperV5(parse_size=getattr(self.args, "ear_parse_size", 512))
        self.query_builder = build_query_builder_compat(
            ear_dilate=getattr(self.args, "ear_dilate", 21),
            hair_change_dilate=getattr(self.args, "hair_change_dilate", 25),
            earring_expand=getattr(self.args, "earring_expand", 15),
            downward_shift=getattr(self.args, "ear_downward_shift", 10),
            target_hair_dilate=getattr(self.args, "target_hair_dilate", 11),
            earring_occlusion_dilate=getattr(self.args, "earring_occlusion_dilate", 3),
            source_hair_block_dilate=getattr(self.args, "source_hair_block_dilate", 5),
            source_hair_block_strength=getattr(self.args, "source_hair_block_strength", 0.6),
            target_visibility_expand=getattr(self.args, "target_visibility_expand", 5),
            max_target_hair_overlap=getattr(self.args, "max_target_hair_overlap", 0.55),
            min_target_visible_overlap=getattr(self.args, "min_target_visible_overlap", 0.02),
            min_target_ear_area=getattr(self.args, "min_target_ear_area", 8.0),
            earring_channel_down=getattr(self.args, "earring_channel_down", 32),
            earring_align_max_shift=getattr(self.args, "earring_align_max_shift", 12),
        )
        self.hf_extractor = ShadowSuppressedHFExtractor(
            low_alpha=getattr(self.args, "ear_low_alpha", 0.1),
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
        )
        self.ear_injector_128 = HFDAGatedInjectionUnit(
            base_channels=getattr(self.args, "ear_inject_channels_128", 256),
            prior_channels=feature_channels,
        )

    def load_base_checkpoint(self, checkpoint_path: str | None):
        if not checkpoint_path:
            return None
        checkpoint = load_checkpoint_compat(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        result = self.load_state_dict(state_dict, strict=False)
        relevant_missing = [
            key for key in result.missing_keys
            if not key.startswith(
                (
                    "hf_extractor",
                    "mask_refresher",
                    "brightness_reestimator",
                    "ear_injector_64",
                    "ear_injector_128",
                    "query_builder",
                    "parsing_helper",
                )
            )
        ]
        if relevant_missing:
            print(f"[PostProcessModelV5] Missing base keys: {len(relevant_missing)}")
            print(relevant_missing[:20])
        if result.unexpected_keys:
            print(f"[PostProcessModelV5] Unexpected checkpoint keys: {len(result.unexpected_keys)}")
            print(result.unexpected_keys[:20])
        return result

    def _enhance_query_recall(
        self,
        source_01: torch.Tensor,
        source_parsing: torch.Tensor,
        query_info: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not bool(getattr(self.args, "enable_earring_query_recall", True)):
            return query_info

        query_mask = ensure_mask_4d(query_info["query_mask"]).float()
        visible_ear_roi = query_info.get("visible_ear_roi", query_info.get("ear_roi", query_mask))
        earring_valid_roi = query_info.get("earring_valid_roi", visible_ear_roi)
        detection_ear_roi = query_info.get("ear_roi", visible_ear_roi)
        source_earring_detection_mask = query_info.get("source_earring_detection_mask")
        source_hair_block_mask = query_info.get("source_hair_block_mask")
        existing_source_mask = query_info.get("source_earring_mask")
        existing_object_mask = query_info.get("source_earring_object_mask")
        weak_masks = build_weak_earring_masks(
            source_01,
            detection_ear_roi,
            query_mask,
            source_earring_detection_mask,
            query_info.get("source_hair_mask"),
            source_hair_block_mask,
            source_parsing,
        )
        recall_info = enhance_query_with_earring_recall(
            query_mask,
            source_earring_detection_mask,
            source_hair_block_mask,
            weak_masks,
            visibility_mask=earring_valid_roi,
            recall_dilate=getattr(self.args, "earring_query_recall_dilate", 7),
            downward_shift=getattr(self.args, "earring_query_downward_shift", 18),
            lower_lobe_weight=getattr(self.args, "earring_query_lower_lobe_weight", 0.20),
            candidate_boost=getattr(self.args, "earring_query_candidate_boost", 0.90),
            block_protect=getattr(self.args, "earring_query_block_protect", 0.85),
        )
        earring_valid_mask = resize_mask(earring_valid_roi, query_mask.shape[-2:])
        if existing_source_mask is None:
            existing_source_mask = torch.zeros_like(query_mask)
        else:
            existing_source_mask = resize_mask(existing_source_mask, query_mask.shape[-2:])
        if existing_object_mask is None:
            existing_object_mask = torch.zeros_like(query_mask)
        else:
            existing_object_mask = resize_mask(existing_object_mask, query_mask.shape[-2:])
        recall_info["source_earring_mask"] = torch.clamp(
            recall_info["source_earring_mask"] + existing_source_mask + existing_object_mask,
            0,
            1,
        ) * earring_valid_mask
        existing_confident = query_info.get("earring_confident_mask")
        if existing_confident is None:
            existing_confident = torch.zeros_like(query_mask)
        else:
            existing_confident = resize_mask(existing_confident, query_mask.shape[-2:])
        confident_mask = torch.clamp(
            resize_mask(weak_masks["earring_confident_mask"], query_mask.shape[-2:])
            + recall_info["source_earring_mask"],
            0,
            1,
        )
        confident_mask = torch.clamp(confident_mask + existing_confident, 0, 1) * resize_mask(
            earring_valid_roi,
            query_mask.shape[-2:],
        )
        recall_info["earring_confident_mask"] = confident_mask
        recall_info["earring_candidate_mask"] = recall_info["online_earring_candidate_mask"]
        recall_info["earring_search_mask"] = recall_info["online_earring_search_mask"]
        recall_info["earring_highlight_mask"] = (
            resize_mask(weak_masks["earring_highlight_mask"], query_mask.shape[-2:]) * earring_valid_roi
        )
        query_info.update(recall_info)
        return query_info

    def _apply_earring_fine_mask_floor(
        self,
        fine_mask: torch.Tensor,
        query_mask: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        floor = max(0.0, min(1.0, float(getattr(self.args, "earring_fine_mask_floor", 0.18))))
        if floor <= 0:
            return fine_mask

        query_mask = ensure_mask_4d(query_mask).float()

        def aux_mask(name: str) -> torch.Tensor:
            value = aux.get(name)
            if value is None:
                return torch.zeros_like(query_mask)
            return resize_mask(value, query_mask.shape[-2:])

        support = torch.clamp(
            aux_mask("source_earring_mask")
            + aux_mask("online_earring_candidate_mask")
            + 0.35 * aux_mask("earring_query_recall_mask"),
            0,
            1,
        )
        visibility_mask = aux_mask("earring_visibility_mask")
        support = dilate_mask(support, int(getattr(self.args, "earring_fine_mask_dilate", 5)))
        support = support * query_mask * visibility_mask
        if support.detach().flatten(1).amax(dim=1).max().item() <= 0:
            return fine_mask

        aux["fine_mask_before_floor"] = fine_mask
        aux["earring_fine_floor_support"] = support
        return torch.maximum(fine_mask, floor * support)

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
        aux: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        face_scale = None
        if aux is not None:
            source_hair = aux.get("source_hair_mask")
            target_face = aux.get("target_face_surface_mask")
            target_hair = aux.get("target_hair_mask")
            if source_hair is not None and target_face is not None:
                size = f_face.shape[-2:]
                risk = resize_mask(source_hair, size) * resize_mask(target_face, size)
                max_y = max(
                    0.0,
                    min(1.0, float(getattr(self.args, "source_hair_face_suppress_max_y", 0.45))),
                )
                if max_y < 1.0:
                    y_coords = torch.linspace(
                        0,
                        1,
                        size[0],
                        device=f_face.device,
                        dtype=f_face.dtype,
                    ).view(1, 1, size[0], 1)
                    risk = risk * (y_coords <= max_y).float()
                if target_hair is not None:
                    risk = risk * (1.0 - resize_mask(target_hair, size)).clamp(0, 1)
                risk_dilate = int(getattr(self.args, "source_hair_face_suppress_dilate", 5))
                if risk_dilate > 0:
                    risk = dilate_mask(risk, risk_dilate)

                # Forehead cleanup must not suppress v58's ear/earring path.
                ear_exclude = torch.zeros_like(risk)
                for key in (
                    "ear_roi",
                    "visible_ear_roi",
                    "earring_confident_mask",
                    "source_earring_mask",
                ):
                    value = aux.get(key)
                    if value is not None:
                        ear_exclude = torch.clamp(ear_exclude + resize_mask(value, size), 0, 1)
                exclude_dilate = int(
                    getattr(self.args, "source_hair_face_suppress_ear_exclude_dilate", 9)
                )
                if exclude_dilate > 0:
                    ear_exclude = dilate_mask(ear_exclude, exclude_dilate)
                risk = risk * (1.0 - ear_exclude).clamp(0, 1)

                strength = max(
                    0.0,
                    min(1.0, float(getattr(self.args, "source_hair_face_suppress_strength", 0.65))),
                )
                face_scale = (1.0 - strength * risk).clamp(0, 1)
                aux["source_hair_face_feature_suppress_mask"] = risk
                aux["source_face_feature_scale"] = face_scale

        if face_scale is None:
            face_scale = torch.ones(
                f_face.size(0),
                1,
                f_face.size(2),
                f_face.size(3),
                device=f_face.device,
                dtype=f_face.dtype,
            )

        if self.use_full or target_mask is None:
            cat_f = torch.cat((f_face * face_scale, f_hair), dim=1)
        else:
            t_mask = F.interpolate(ensure_mask_4d(target_mask).float(), size=f_face.shape[-2:], mode="nearest")
            cat_f = torch.cat((f_face * t_mask * face_scale, f_hair * (1 - t_mask)), dim=1)
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
        source_earring_object_mask: torch.Tensor | None,
        earring_confident_mask: torch.Tensor | None,
        earring_highlight_mask: torch.Tensor | None,
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
        ear_roi = query_info.get("ear_roi", visible_ear_roi)
        earring_valid_roi = query_info.get("earring_valid_roi", visible_ear_roi)

        def gated_aux_mask(mask: torch.Tensor | None, gate: torch.Tensor | None) -> torch.Tensor | None:
            if mask is None:
                return None
            mask = resize_mask(ensure_mask_4d(mask).float(), image_size).clamp(0, 1)
            if gate is not None:
                mask = mask * resize_mask(gate, image_size)
            return mask.clamp(0, 1)

        def merge_aux_mask(name: str, value: torch.Tensor | None):
            if value is None:
                return
            current = query_info.get(name)
            if current is None:
                query_info[name] = value
            else:
                query_info[name] = torch.clamp(resize_mask(current, image_size) + value, 0, 1)

        if query_mask is not None and self.use_dataset_query_mask:
            dataset_query_mask = ensure_mask_4d(query_mask).float()
            if visible_ear_roi is not None:
                dataset_query_mask = dataset_query_mask * visible_ear_roi
            query_info["query_mask"] = dataset_query_mask
        if source_ear_mask is not None:
            merge_aux_mask("source_earring_detection_mask", gated_aux_mask(source_ear_mask, ear_roi))
            merge_aux_mask("source_earring_mask", gated_aux_mask(source_ear_mask, earring_valid_roi))
        if source_earring_object_mask is not None:
            merge_aux_mask("source_earring_detection_mask", gated_aux_mask(source_earring_object_mask, ear_roi))
            merge_aux_mask("source_earring_mask", gated_aux_mask(source_earring_object_mask, earring_valid_roi))
            merge_aux_mask("source_earring_object_mask", gated_aux_mask(source_earring_object_mask, earring_valid_roi))
        if earring_confident_mask is not None:
            merge_aux_mask("source_earring_detection_mask", gated_aux_mask(earring_confident_mask, ear_roi))
            merge_aux_mask("source_earring_mask", gated_aux_mask(earring_confident_mask, earring_valid_roi))
            merge_aux_mask("earring_confident_mask", gated_aux_mask(earring_confident_mask, earring_valid_roi))
        if earring_highlight_mask is not None:
            merge_aux_mask("earring_highlight_mask", gated_aux_mask(earring_highlight_mask, earring_valid_roi))

        recovery_seed = torch.zeros_like(query_info["query_mask"])
        for key in ("source_earring_mask", "source_earring_object_mask", "earring_confident_mask"):
            value = query_info.get(key)
            if value is not None:
                recovery_seed = torch.clamp(recovery_seed + resize_mask(value, recovery_seed.shape[-2:]), 0, 1)
        if recovery_seed.detach().flatten(1).amax(dim=1).max().item() > 0:
            query_expand = int(getattr(self.args, "earring_query_dilate", 3))
            if query_expand > 0:
                recovery_seed = dilate_mask(recovery_seed, query_expand)
            if earring_valid_roi is not None:
                recovery_seed = recovery_seed * resize_mask(earring_valid_roi, recovery_seed.shape[-2:])
            query_info["query_mask"] = torch.clamp(query_info["query_mask"] + recovery_seed, 0, 1)
        if presence_target is not None:
            visibility_target = query_info.get("visibility_target")
            presence_target = presence_target.float()
            if visibility_target is not None:
                presence_target = presence_target * visibility_target.float()
                side_presence = presence_target[:, :2]
                presence_target = torch.cat(
                    [side_presence, side_presence.amax(dim=1, keepdim=True)],
                    dim=1,
                )
            query_info["presence_target"] = presence_target

        query_info = self._enhance_query_recall(source_01, source_parsing, query_info)

        query_info["source_parsing"] = source_parsing.long()
        query_info["target_parsing"] = target_parsing.long()
        return query_info

    def _attach_revealed_skin_info(
        self,
        aux: dict[str, torch.Tensor],
        source_image: torch.Tensor,
        cleanup_masks: dict[str, torch.Tensor] | None,
        revealed_skin_mask: torch.Tensor | None,
        revealed_skin_seam_mask: torch.Tensor | None,
        source_visible_skin_reference_mask: torch.Tensor | None,
        source_skin_valid_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        image_size = tuple(aux["source_parsing"].shape[-2:])
        fallback = torch.zeros_like(aux["source_parsing"], dtype=torch.float32)
        resolved_cleanup = {}
        for key in ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck"):
            value = cleanup_masks.get(key) if isinstance(cleanup_masks, dict) else None
            resolved_cleanup[key] = fallback if value is None else resize_mask(value, image_size)

        earring_exclude = torch.zeros_like(fallback)
        for key in (
            "source_earring_mask",
            "target_earring_mask",
            "earring_confident_mask",
        ):
            value = aux.get(key)
            if value is not None:
                earring_exclude = torch.clamp(earring_exclude + resize_mask(value, image_size), 0, 1)

        info = build_revealed_skin_mask(
            resolved_cleanup,
            aux["target_parsing"],
            aux["source_parsing"],
            aux.get("target_hair_mask"),
            aux.get("source_hair_mask"),
            earring_exclude,
            source_image,
        )
        target_safe = (
            info["target_face_surface_mask"]
            * (1.0 - resize_mask(aux.get("target_hair_mask", fallback), image_size)).clamp(0, 1)
            * (1.0 - dilate_mask(earring_exclude, 3)).clamp(0, 1)
        ).clamp(0, 1)

        if revealed_skin_mask is not None:
            info["revealed_skin_mask"] = resize_mask(revealed_skin_mask, image_size) * target_safe
        if revealed_skin_seam_mask is not None:
            info["revealed_skin_seam_mask"] = resize_mask(revealed_skin_seam_mask, image_size) * target_safe

        explicit_reference = source_visible_skin_reference_mask
        if explicit_reference is None:
            explicit_reference = source_skin_valid_mask
        if explicit_reference is not None:
            # Intersect with the online strict mask so stale dataset masks can
            # never re-admit source bangs, details or earrings.
            strict_reference = info["source_visible_skin_reference_mask"]
            info["source_visible_skin_reference_mask"] = (
                resize_mask(explicit_reference, image_size) * strict_reference
            ).clamp(0, 1)
            info["source_skin_valid_mask"] = info["source_visible_skin_reference_mask"]

        info["revealed_skin_blend_mask"] = torch.maximum(
            info["revealed_skin_mask"],
            info["revealed_skin_seam_mask"],
        ).clamp(0, 1)
        aux.update(resolved_cleanup)
        aux.update(info)
        return aux

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
        source_earring_object_mask: torch.Tensor | None = None,
        earring_confident_mask: torch.Tensor | None = None,
        earring_supervision_mask: torch.Tensor | None = None,
        earring_highlight_mask: torch.Tensor | None = None,
        earring_reference: torch.Tensor | None = None,
        earring_mask_is_dataset: torch.Tensor | None = None,
        earring_reference_is_dataset: torch.Tensor | None = None,
        presence_target: torch.Tensor | None = None,
        cleanup_masks: dict[str, torch.Tensor] | None = None,
        revealed_skin_mask: torch.Tensor | None = None,
        revealed_skin_seam_mask: torch.Tensor | None = None,
        source_visible_skin_reference_mask: torch.Tensor | None = None,
        source_skin_valid_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
                source_earring_object_mask,
                earring_confident_mask if earring_confident_mask is not None else earring_supervision_mask,
                earring_highlight_mask,
                presence_target,
            )
            aux = self._attach_revealed_skin_info(
                aux,
                source,
                cleanup_masks,
                revealed_skin_mask,
                revealed_skin_seam_mask,
                source_visible_skin_reference_mask,
                source_skin_valid_mask,
            )
            return self.latent_avg.to(s_face.device) + s_face, f_face, aux

        s_hair, [f_hair] = self.encoder_face(target)
        final_s = self._compute_latent(s_face, s_hair)

        aux = self._resolve_masks(
            source,
            target,
            source_parsing,
            target_parsing,
            source_hair_mask,
            target_hair_mask,
            query_mask,
            source_ear_mask,
            source_earring_object_mask,
            earring_confident_mask if earring_confident_mask is not None else earring_supervision_mask,
            earring_highlight_mask,
            presence_target,
        )
        aux = self._attach_revealed_skin_info(
            aux,
            source,
            cleanup_masks,
            revealed_skin_mask,
            revealed_skin_seam_mask,
            source_visible_skin_reference_mask,
            source_skin_valid_mask,
        )
        base_feature = self._build_base_feature(f_face, f_hair, target_mask, aux)
        source_01 = normalized_to_01(source)
        target_01 = normalized_to_01(target)

        query_mask = ensure_mask_4d(aux["query_mask"]).float()
        source_earring_mask = ensure_mask_4d(aux["source_earring_mask"]).float()

        # Restore the v58 ear path exactly: one source-coordinate reference,
        # one learned fine mask, and no target-aligned object compositor.
        hf_outputs = self.hf_extractor(source_01, query_mask)
        mask_outputs = self.mask_refresher(
            source_01,
            target_01,
            hf_outputs["high_energy"],
            query_mask,
            source_earring_mask,
        )
        mask_outputs["fine_mask"] = self._apply_earring_fine_mask_floor(
            mask_outputs["fine_mask"],
            query_mask,
            aux,
        )
        prior_feature, brightness_outputs = self.brightness_reestimator(
            target_01,
            hf_outputs["prior_feature"],
            mask_outputs["fine_mask"],
            query_mask,
        )
        final_f, fine_mask_64 = self.ear_injector_64(base_feature, prior_feature, mask_outputs["fine_mask"])

        aux.update(hf_outputs)
        aux.update(mask_outputs)
        aux.update(brightness_outputs)
        aux["raw_query_mask"] = query_mask
        aux["ear_detail_query_mask"] = query_mask
        aux["adjusted_prior_feature"] = prior_feature
        aux["injected_fine_mask_64"] = fine_mask_64
        aux["source_01"] = source_01
        aux["detail_reference_01"] = source_01
        aux["earring_reference"] = source_01
        aux["target_01"] = target_01
        aux["HT_E"] = ensure_mask_4d(HT_E).float() if HT_E is not None else None
        return final_s, final_f, aux

    def _target_output_preserve_mask(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        target_hair = aux.get("target_hair_mask")
        if target_hair is None:
            return None

        hair_raw = resize_mask(ensure_mask_4d(target_hair).float(), size)
        hair_context = hair_raw
        hair_dilate = int(getattr(self.args, "output_target_hair_preserve_dilate", 5))
        if hair_dilate > 0:
            hair_context = dilate_mask(hair_raw, hair_dilate)

        # Preserve every parser-confirmed target-hair pixel.  Only the dilated
        # context is clipped away from the face, so light hair at the hairline
        # cannot be brightened or recolored by PP.
        target_face = aux.get("target_face_surface_mask")
        if target_face is not None:
            face_exclude = resize_mask(target_face, size)
            seam_dilate = int(getattr(self.args, "output_face_hair_seam_preserve_dilate", 7))
            if seam_dilate > 0:
                face_exclude = dilate_mask(face_exclude, seam_dilate)
            hair_context = hair_context * (1.0 - face_exclude).clamp(0, 1)
        preserve = torch.maximum(hair_raw, hair_context).clamp(0, 1)

        # Soften the hair->face hand-off at the hairline.  A hard 1->0 alpha edge
        # between target(SATD) hair and PP-generated face reads as a translucent
        # "wig-glue" line.  We erode the solid-preserve core a few pixels *into*
        # the hair, then let a Gaussian ramp fall off from that core.  Clipping
        # the ramp by the original hair region keeps the feather strictly inside
        # the hair, so it never pulls target hair color onto the face.
        seam_feather = int(getattr(self.args, "output_hairline_feather", 0))
        if seam_feather > 1:
            hair_region = preserve
            core = erode_mask(preserve, seam_feather)
            ramp = gaussian_blur(
                core,
                kernel_size=seam_feather,
                sigma=max(1.0, seam_feather / 3.0),
            ).clamp(0, 1)
            preserve = torch.maximum(core, torch.minimum(hair_region, ramp)).clamp(0, 1)

        # Hair preservation is outside the v58 earring chain.  Only v58's
        # confident pixels inside its visibility gate may remain in front of
        # target hair; the empty downward channel never grants an exemption.
        earring_keep = aux.get("earring_confident_mask")
        earring_valid_roi = aux.get("earring_valid_roi")
        if earring_keep is None or earring_valid_roi is None:
            earring_keep = torch.zeros_like(preserve)
        else:
            earring_keep = resize_mask(earring_keep, size) * resize_mask(earring_valid_roi, size)
        keep_dilate = int(getattr(self.args, "output_earring_keep_dilate", 0))
        if keep_dilate > 0:
            earring_keep = dilate_mask(earring_keep, keep_dilate)
            earring_keep = earring_keep * resize_mask(earring_valid_roi, size)

        preserve = preserve * (1.0 - earring_keep).clamp(0, 1)
        blur = int(getattr(self.args, "output_preserve_blur", 1))
        if blur > 1:
            if blur % 2 == 0:
                blur += 1
            preserve = F.avg_pool2d(
                preserve,
                kernel_size=blur,
                stride=1,
                padding=blur // 2,
            ).clamp(0, 1)
        aux["output_target_hair_preserve_mask"] = preserve
        aux["output_target_hair_earring_keep_mask"] = earring_keep
        return preserve

    @staticmethod
    def _diffuse_fill(
        value: torch.Tensor,
        known_mask: torch.Tensor,
        iterations: int,
        kernel_size: int,
        sigma: float,
    ) -> torch.Tensor:
        """Propagate known values into unknown pixels by normalized diffusion.

        This is a cheap inpainting: at each step the known content and its
        coverage are blurred together, then divided.  It spreads the recovered
        source-skin tone smoothly across the revealed forehead without any hard
        boundary, so no seam can appear between filled and reference pixels.
        """
        known_mask = (known_mask > 0.5).float()
        denom = known_mask.flatten(2).sum(dim=2, keepdim=True).clamp_min(1.0)
        known_mean = (value * known_mask).flatten(2).sum(dim=2, keepdim=True) / denom
        known_mean = known_mean.unsqueeze(-1)
        # Seed the unknown region with the global known mean, then relax by
        # repeated blur while re-anchoring the true known pixels each step.  This
        # is a stable harmonic fill: unknown pixels converge to the surrounding
        # known tone instead of drifting toward the peak value.
        estimate = value * known_mask + known_mean * (1.0 - known_mask)
        for _ in range(max(1, int(iterations))):
            estimate = gaussian_blur(estimate, kernel_size=kernel_size, sigma=sigma)
            estimate = value * known_mask + estimate * (1.0 - known_mask)
        return estimate.clamp(0, 1)

    def _harmonize_revealed_skin(
        self,
        image_01: torch.Tensor,
        aux: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Match the newly revealed forehead to the recovered source skin tone.

        The revealed forehead (source had bangs, reference does not) has no clean
        source texture to copy, so PP leaves the SATD-smoothed, whitish look and a
        visible boundary against the real skin below.  We pull the revealed
        region's low frequency toward the tone diffused from the already-recovered
        neighbouring source skin, and gently restore high-frequency micro-texture.
        Only runs at inference; training relies on the revealed-skin loss instead.
        """
        if aux is None or self.training:
            return image_01
        if not bool(getattr(self.args, "enable_revealed_skin_harmonize", True)):
            return image_01
        revealed = aux.get("revealed_skin_blend_mask", aux.get("revealed_skin_mask"))
        if revealed is None:
            return image_01

        size = image_01.shape[-2:]
        revealed = resize_mask(revealed, size).clamp(0, 1)
        if float(revealed.detach().amax().item()) <= 0:
            return image_01

        # Build the recovered-skin reference: target face skin, minus the revealed
        # region itself, target hair, facial details and earrings.  These are the
        # pixels where PP has already produced the correct source skin.
        reference = aux.get("target_face_surface_mask")
        if reference is None and aux.get("target_parsing") is not None:
            reference = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
        if reference is None:
            return image_01
        reference = resize_mask(reference, size)
        reference = reference * (1.0 - dilate_mask(revealed, 5)).clamp(0, 1)
        for key in ("target_hair_mask",):
            value = aux.get(key)
            if value is not None:
                reference = reference * (1.0 - resize_mask(value, size)).clamp(0, 1)
        target_parsing = aux.get("target_parsing")
        if target_parsing is not None:
            detail = parsing_label_mask(target_parsing, RAW_DETAIL_LABELS)
            reference = reference * (1.0 - dilate_mask(resize_mask(detail, size), 3)).clamp(0, 1)
        for key in ("earring_confident_mask", "source_earring_mask", "visible_ear_roi", "ear_roi"):
            value = aux.get(key)
            if value is not None:
                reference = reference * (1.0 - resize_mask(value, size)).clamp(0, 1)
        reference = reference.clamp(0, 1)

        area_scale = float(size[0] * size[1]) / float(256 * 256)
        if float(reference.detach().flatten(1).sum(dim=1).min().item()) < 256.0 * area_scale:
            # Too little recovered skin to define a reliable tone; leave as-is
            # rather than risk pulling the forehead toward the wrong color.
            return image_01

        work = 256
        img_small = F.interpolate(image_01, size=(work, work), mode="bilinear", align_corners=False)
        ref_small = F.interpolate(reference, size=(work, work), mode="bilinear", align_corners=False).clamp(0, 1)
        rev_small = F.interpolate(revealed, size=(work, work), mode="bilinear", align_corners=False).clamp(0, 1)

        blur_k = int(getattr(self.args, "revealed_skin_tone_kernel", 21))
        blur_s = float(getattr(self.args, "revealed_skin_tone_sigma", 7.0))
        iters = int(getattr(self.args, "revealed_skin_diffuse_iters", 24))

        ref_low = low_pass_filter(img_small, kernel_size=blur_k, sigma=blur_s)
        tone_ref = self._diffuse_fill(ref_low, ref_small, iters, blur_k, blur_s)
        cur_low = low_pass_filter(img_small, kernel_size=blur_k, sigma=blur_s)
        correction = (tone_ref - cur_low)
        # Keep the correction low-frequency and bounded so it only fixes tone,
        # never introduces edges or color blocks.
        correction = low_pass_filter(correction, kernel_size=blur_k, sigma=blur_s)
        limit = float(getattr(self.args, "revealed_skin_tone_limit", 0.35))
        correction = correction.clamp(-limit, limit)
        correction = F.interpolate(correction, size=size, mode="bilinear", align_corners=False)

        strength = max(0.0, min(1.0, float(getattr(self.args, "revealed_skin_harmonize_strength", 0.9))))
        blend = revealed * strength
        result = (image_01 + correction * blend).clamp(0, 1)

        # Optional gentle micro-texture: amplify the revealed region's own high
        # frequency so the harmonized skin does not look plastic.  Bounded and
        # local, so it cannot tile source pores or create fragments.
        gain = float(getattr(self.args, "revealed_skin_detail_gain", 1.0))
        if gain > 1.0:
            high = high_pass_filter(result, kernel_size=blur_k, sigma=blur_s)
            result = (result + (gain - 1.0) * high * blend).clamp(0, 1)

        aux["revealed_skin_harmonize_mask"] = blend
        aux["revealed_skin_tone_reference"] = F.interpolate(
            tone_ref, size=size, mode="bilinear", align_corners=False
        ).clamp(0, 1)
        return result

    def _preserve_target_output(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if aux is None:
            return image

        image_01 = ((image + 1) / 2).clamp(0, 1)
        # Harmonize the revealed forehead first, on the raw generated skin, before
        # any target-hair hand-off.  This is a no-op in training and when there is
        # no revealed region or too little recovered-skin reference.
        image_01 = self._harmonize_revealed_skin(image_01, aux)

        if not bool(getattr(self.args, "enable_output_target_preserve", True)):
            return image_01 * 2 - 1
        target = aux.get("target_01")
        if target is None:
            return image_01 * 2 - 1

        # Match v58's single generated earring path: no final RGB hard paste.
        aux["output_source_earring_composite_mask"] = torch.zeros(
            image_01.size(0),
            1,
            image_01.size(2),
            image_01.size(3),
            device=image_01.device,
            dtype=image_01.dtype,
        )
        target = F.interpolate(
            target,
            size=image_01.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        preserve = self._target_output_preserve_mask(aux, image_01.shape[-2:])
        if preserve is None:
            return image_01 * 2 - 1
        protected = image_01 * (1.0 - preserve) + target * preserve
        return protected.clamp(0, 1) * 2 - 1

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

        image, _ = generator(
            [latent_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=6,
            end_layer=8,
            layer_in=feature_128,
            skip=skip,
        )
        image = self._preserve_target_output(image, aux)
        return image, aux
