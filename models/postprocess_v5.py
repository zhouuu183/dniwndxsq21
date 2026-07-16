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
    BrightnessReEstimator,
    DynamicFineMaskRefresher,
    EarAnchoredQueryBuilder,
    FaceParsingHelperV5,
    HFDAGatedInjectionUnit,
    RAW_DETAIL_LABELS,
    RAW_FACE_SURFACE_LABELS,
    ShadowSuppressedHFExtractor,
    build_weak_earring_masks,
    dilate_mask,
    erode_mask,
    enhance_query_with_earring_recall,
    ensure_mask_4d,
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
            source_hair_block_dilate=getattr(self.args, "source_hair_block_dilate", 5),
            source_hair_block_strength=getattr(self.args, "source_hair_block_strength", 0.6),
            target_visibility_expand=getattr(self.args, "target_visibility_expand", 5),
            max_target_hair_overlap=getattr(self.args, "max_target_hair_overlap", 0.55),
            target_ear_cover_overlap=getattr(self.args, "target_ear_cover_overlap", 0.55),
            min_target_visible_overlap=getattr(self.args, "min_target_visible_overlap", 0.02),
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
        source_hair_block_mask = query_info.get("source_hair_block_mask")
        weak_masks = build_weak_earring_masks(
            source_01,
            visible_ear_roi,
            query_mask,
            query_info.get("source_earring_mask"),
            query_info.get("source_hair_mask"),
            source_hair_block_mask,
            source_parsing,
        )
        recall_info = enhance_query_with_earring_recall(
            query_mask,
            query_info.get("source_earring_mask"),
            source_hair_block_mask,
            weak_masks,
            recall_dilate=getattr(self.args, "earring_query_recall_dilate", 7),
            downward_shift=getattr(self.args, "earring_query_downward_shift", 18),
            lower_lobe_weight=getattr(self.args, "earring_query_lower_lobe_weight", 0.20),
            candidate_boost=getattr(self.args, "earring_query_candidate_boost", 0.90),
            block_protect=getattr(self.args, "earring_query_block_protect", 0.85),
        )
        confident_mask = torch.clamp(
            resize_mask(weak_masks["earring_confident_mask"], query_mask.shape[-2:])
            + recall_info["source_earring_mask"],
            0,
            1,
        )
        recall_info["earring_confident_mask"] = confident_mask
        recall_info["earring_candidate_mask"] = recall_info["online_earring_candidate_mask"]
        recall_info["earring_search_mask"] = recall_info["online_earring_search_mask"]
        recall_info["source_earring_clean_mask"] = resize_mask(
            weak_masks["source_earring_clean_mask"],
            query_mask.shape[-2:],
        )
        recall_info["earring_highlight_mask"] = resize_mask(weak_masks["earring_highlight_mask"], query_mask.shape[-2:])
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
            aux_mask("earring_supervision_mask")
            + aux_mask("earring_confident_mask")
            + aux_mask("source_earring_clean_mask")
            + 0.35 * aux_mask("earring_query_recall_mask"),
            0,
            1,
        )
        support = dilate_mask(support, int(getattr(self.args, "earring_fine_mask_dilate", 5))) * query_mask
        if support.detach().flatten(1).amax(dim=1).max().item() <= 0:
            return fine_mask

        aux["fine_mask_before_floor"] = fine_mask
        aux["earring_fine_floor_support"] = support
        return torch.maximum(fine_mask, floor * support)

    def _trusted_earring_keep_mask(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        template = None
        for key in ("query_mask", "visible_ear_roi", "ear_roi", "target_hair_mask"):
            value = aux.get(key)
            if value is not None:
                template = resize_mask(value, size)
                break
        if template is None:
            return None

        keep = torch.zeros_like(template)
        for key, weight in (
            ("earring_supervision_mask", 1.0),
            ("earring_confident_mask", 1.0),
            ("source_earring_mask", 1.0),
            ("source_earring_clean_mask", 1.0),
            ("earring_query_recall_mask", 0.35),
        ):
            value = aux.get(key)
            if value is not None:
                keep = torch.clamp(keep + weight * resize_mask(value, size), 0, 1)

        dilate = int(getattr(self.args, "trusted_earring_keep_dilate", 2))
        if dilate > 0:
            keep = dilate_mask(keep, dilate)

        valid_roi = self._earring_valid_roi(aux, size)
        if valid_roi is not None:
            keep = keep * valid_roi

        keep = keep.clamp(0, 1)
        aux["trusted_earring_keep_mask"] = keep
        return keep

    def _earring_valid_roi(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        valid = None

        ear_roi = aux.get("ear_roi")
        if ear_roi is not None:
            valid = resize_mask(ear_roi, size)
            dilate = int(getattr(self.args, "earring_valid_roi_dilate", 9))
            if dilate > 0:
                valid = dilate_mask(valid, dilate)

        visible_ear_roi = aux.get("visible_ear_roi")
        if visible_ear_roi is not None:
            visible = resize_mask(visible_ear_roi, size)
            visible_dilate = int(getattr(self.args, "earring_visible_roi_dilate", 9))
            if visible_dilate > 0:
                visible = dilate_mask(visible, visible_dilate)
            valid = visible if valid is None else torch.clamp(valid + visible, 0, 1)

        if valid is None:
            return None

        valid = valid.clamp(0, 1)
        aux["earring_valid_roi"] = valid
        return valid

    def _target_hair_ear_block_mask(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        target_hair_value = aux.get("target_hair_mask")
        covered_block = aux.get("target_covered_ear_block_mask")
        if target_hair_value is None and covered_block is None:
            return None

        if target_hair_value is None:
            target_hair_raw = torch.zeros_like(resize_mask(covered_block, size))
        else:
            target_hair_raw = resize_mask(ensure_mask_4d(target_hair_value).float(), size)

        protect = target_hair_raw
        dilate = int(getattr(self.args, "target_hair_ear_protect_dilate", 3))
        if dilate > 0:
            protect = dilate_mask(target_hair_raw, dilate)

        if covered_block is not None:
            covered_block = resize_mask(covered_block, size)
            if target_hair_value is not None:
                covered_block = covered_block * dilate_mask(
                    target_hair_raw,
                    int(getattr(self.args, "target_hair_ear_protect_dilate", 3)),
                )
            protect = torch.clamp(protect + covered_block, 0, 1)

        ear_roi = aux.get("ear_roi", aux.get("visible_ear_roi"))
        if ear_roi is not None:
            ear_scope = resize_mask(ear_roi, size)
            scope_dilate = int(getattr(self.args, "target_hair_ear_scope_dilate", 3))
            if scope_dilate > 0:
                ear_scope = dilate_mask(ear_scope, scope_dilate)
            protect = protect * ear_scope

        trusted_keep = self._trusted_earring_keep_mask(aux, size)
        if trusted_keep is not None:
            protect = protect * (1.0 - trusted_keep).clamp(0, 1)

        protect = protect.clamp(0, 1)
        aux["target_hair_ear_block_mask"] = protect
        return protect

    def _apply_target_hair_ear_protection(
        self,
        mask: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mask = ensure_mask_4d(mask).float()
        block = self._target_hair_ear_block_mask(aux, mask.shape[-2:])
        if block is None:
            return mask

        strength = max(0.0, min(1.0, float(getattr(self.args, "target_hair_ear_protect_strength", 1.0))))
        if strength <= 0:
            return mask

        return mask * (1.0 - strength * block).clamp(0, 1)

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
                max_y = max(0.0, min(1.0, float(getattr(self.args, "source_hair_face_suppress_max_y", 0.45))))
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

                exclude = torch.zeros_like(risk)
                for key in (
                    "ear_roi",
                    "visible_ear_roi",
                    "earring_supervision_mask",
                    "earring_confident_mask",
                    "source_earring_mask",
                    "source_earring_clean_mask",
                ):
                    value = aux.get(key)
                    if value is not None:
                        exclude = torch.clamp(exclude + resize_mask(value, size), 0, 1)
                exclude_dilate = int(getattr(self.args, "source_hair_face_suppress_ear_exclude_dilate", 9))
                if exclude_dilate > 0:
                    exclude = dilate_mask(exclude, exclude_dilate)
                risk = risk * (1.0 - exclude).clamp(0, 1)

                strength = max(0.0, min(1.0, float(getattr(self.args, "source_hair_face_suppress_strength", 1.0))))
                face_scale = (1.0 - strength * risk).clamp(0, 1)

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
            if visible_ear_roi is not None:
                source_ear_mask = source_ear_mask * visible_ear_roi
            query_info["source_earring_mask"] = source_ear_mask
        if presence_target is not None:
            visibility_target = query_info.get("visibility_target")
            presence_target = presence_target.float()
            if visibility_target is not None:
                presence_target = presence_target * visibility_target.float()
            query_info["presence_target"] = presence_target

        query_info = self._enhance_query_recall(source_01, source_parsing, query_info)

        query_info["source_parsing"] = source_parsing.long()
        query_info["target_parsing"] = target_parsing.long()
        return query_info

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
        earring_confident_mask: torch.Tensor | None = None,
        earring_supervision_mask: torch.Tensor | None = None,
        earring_highlight_mask: torch.Tensor | None = None,
        earring_reference: torch.Tensor | None = None,
        earring_mask_is_dataset: torch.Tensor | None = None,
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
                presence_target,
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
            presence_target,
        )
        base_feature = self._build_base_feature(f_face, f_hair, target_mask, aux)
        source_01 = normalized_to_01(source)
        target_01 = normalized_to_01(target)

        dataset_gate = None
        if torch.is_tensor(earring_mask_is_dataset):
            dataset_gate = earring_mask_is_dataset.float().view(-1, 1, 1, 1) > 0.5

        if earring_supervision_mask is not None:
            supervision_mask = ensure_mask_4d(earring_supervision_mask).float()
            supervision_mask = resize_mask(supervision_mask, aux["query_mask"].shape[-2:])
            earring_valid = self._earring_valid_roi(aux, aux["query_mask"].shape[-2:])
            if earring_valid is not None:
                supervision_mask = supervision_mask * earring_valid
            if dataset_gate is not None:
                supervision_mask = supervision_mask * dataset_gate.float()

            aux["earring_supervision_mask"] = supervision_mask.clamp(0, 1)
            aux["earring_mask_is_dataset"] = earring_mask_is_dataset
            aux["online_query_mask_before_dataset_earring"] = aux["query_mask"]
            dataset_expand = int(getattr(self.args, "dataset_earring_query_dilate", 3))
            dataset_query = aux["earring_supervision_mask"]
            if dataset_expand > 0:
                dataset_query = dilate_mask(dataset_query, dataset_expand)
            if earring_valid is not None:
                dataset_query = dataset_query * earring_valid
            aux["dataset_earring_query_mask"] = dataset_query.clamp(0, 1)
            aux["query_mask"] = torch.clamp(aux["query_mask"] + aux["dataset_earring_query_mask"], 0, 1)

        if earring_confident_mask is not None:
            dataset_earring_mask = ensure_mask_4d(earring_confident_mask).float()
            if dataset_gate is not None:
                aux["earring_confident_mask"] = torch.where(
                    dataset_gate,
                    dataset_earring_mask,
                    aux.get("earring_confident_mask", aux["source_earring_mask"]),
                )
                aux["source_earring_mask"] = torch.where(
                    dataset_gate,
                    dataset_earring_mask,
                    aux["source_earring_mask"],
                )
            else:
                aux["earring_confident_mask"] = dataset_earring_mask
                aux["source_earring_mask"] = dataset_earring_mask
            aux["earring_mask_is_dataset"] = earring_mask_is_dataset
            aux["online_query_mask_before_dataset_earring"] = aux["query_mask"]
            dataset_expand = int(getattr(self.args, "dataset_earring_query_dilate", 3))
            dataset_query = dataset_earring_mask
            if dataset_expand > 0:
                dataset_query = dilate_mask(dataset_query, dataset_expand)
            dataset_query = resize_mask(dataset_query, aux["query_mask"].shape[-2:])
            earring_valid = self._earring_valid_roi(aux, aux["query_mask"].shape[-2:])
            if earring_valid is not None:
                dataset_query = dataset_query * earring_valid
            if dataset_gate is not None:
                dataset_query = dataset_query * dataset_gate.float()
            aux["dataset_earring_query_mask"] = dataset_query
            aux["query_mask"] = torch.clamp(aux["query_mask"] + dataset_query, 0, 1)
        if earring_highlight_mask is not None:
            aux["earring_highlight_mask"] = ensure_mask_4d(earring_highlight_mask).float()
        if earring_reference is not None:
            dataset_reference = normalized_to_01(earring_reference)
            if dataset_gate is not None:
                dataset_reference = F.interpolate(
                    dataset_reference,
                    size=source_01.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0, 1)
                aux["earring_reference"] = torch.where(dataset_gate, dataset_reference, source_01)
            else:
                aux["earring_reference"] = dataset_reference

        detail_reference_01 = aux.get("earring_reference")
        if detail_reference_01 is not None:
            detail_reference_01 = F.interpolate(
                detail_reference_01,
                size=source_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0, 1)
        else:
            detail_reference_01 = source_01

        query_mask = ensure_mask_4d(aux["query_mask"]).float()
        aux["query_mask_before_target_hair_ear_protect"] = query_mask
        query_mask = self._apply_target_hair_ear_protection(query_mask, aux)
        aux["query_mask"] = query_mask

        if aux.get("earring_confident_mask") is not None:
            aux["earring_confident_mask_before_target_hair_ear_protect"] = aux["earring_confident_mask"]
            aux["earring_confident_mask"] = self._apply_target_hair_ear_protection(
                aux["earring_confident_mask"],
                aux,
            )

        source_earring_mask = ensure_mask_4d(aux["source_earring_mask"]).float()
        aux["source_earring_mask_before_target_hair_ear_protect"] = source_earring_mask
        source_earring_mask = self._apply_target_hair_ear_protection(source_earring_mask, aux)
        aux["source_earring_mask"] = source_earring_mask

        hf_outputs = self.hf_extractor(detail_reference_01, query_mask)
        mask_outputs = self.mask_refresher(
            detail_reference_01,
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
        mask_outputs["fine_mask_before_target_hair_ear_protect"] = mask_outputs["fine_mask"]
        mask_outputs["fine_mask"] = self._apply_target_hair_ear_protection(mask_outputs["fine_mask"], aux)
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
        aux["target_01"] = target_01
        aux["detail_reference_01"] = detail_reference_01
        aux["HT_E"] = ensure_mask_4d(HT_E).float() if HT_E is not None else None
        return final_s, final_f, aux

    def _target_output_preserve_mask(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        masks = []
        hard_no_keep_masks = []
        target_hair_support = None
        target_hair = aux.get("target_hair_mask")
        if target_hair is not None:
            hair_raw = resize_mask(ensure_mask_4d(target_hair).float(), size)
            target_hair_support = hair_raw
            hair = hair_raw
            hair_dilate = int(getattr(self.args, "output_target_hair_preserve_dilate", 5))
            if hair_dilate > 0:
                hair = dilate_mask(hair_raw, hair_dilate)
            surface_exclude = None
            for key in ("target_face_surface_mask", "target_left_ear_mask", "target_right_ear_mask"):
                value = aux.get(key)
                if value is None:
                    continue
                value = resize_mask(value, size)
                surface_exclude = value if surface_exclude is None else torch.clamp(surface_exclude + value, 0, 1)
            if surface_exclude is not None:
                surface_exclude_dilate = int(getattr(self.args, "output_face_hair_seam_preserve_dilate", 7))
                if surface_exclude_dilate > 0:
                    surface_exclude = dilate_mask(surface_exclude, surface_exclude_dilate)
                hair = hair * (1.0 - surface_exclude).clamp(0, 1)
            for key in ("revealed_skin_mask", "M_remove_face"):
                value = aux.get(key)
                if value is not None:
                    hair = hair * (1.0 - resize_mask(value, size)).clamp(0, 1)
            masks.append(hair)

        covered_ear = aux.get("target_covered_ear_block_mask")
        if covered_ear is not None:
            covered_ear = resize_mask(covered_ear, size)
            hard_no_keep_masks.append(covered_ear)
            if target_hair_support is not None:
                covered_hair = covered_ear * dilate_mask(
                    target_hair_support,
                    int(getattr(self.args, "output_target_hair_preserve_dilate", 5)),
                )
                masks.append(covered_hair.clamp(0, 1))

        block = aux.get("target_hair_ear_block_mask")
        if block is not None and target_hair_support is not None:
            block = resize_mask(block, size)
            ear_hair_support = dilate_mask(
                target_hair_support,
                int(getattr(self.args, "output_target_hair_ear_preserve_dilate", 3)),
            )
            block = block * ear_hair_support
            masks.append(block.clamp(0, 1))

        if bool(getattr(self.args, "enable_output_target_ear_skin_preserve", False)):
            target_ear = None
            for key in ("target_left_ear_mask", "target_right_ear_mask"):
                value = aux.get(key)
                if value is not None:
                    value = resize_mask(value, size)
                    target_ear = value if target_ear is None else torch.clamp(target_ear + value, 0, 1)
            if target_ear is not None:
                ear_dilate = int(getattr(self.args, "output_target_ear_skin_preserve_dilate", 0))
                if ear_dilate > 0:
                    target_ear = dilate_mask(target_ear, ear_dilate)
                    masks.append(target_ear.clamp(0, 1))

        if not masks:
            return None

        preserve = torch.stack(masks, dim=0).amax(dim=0).clamp(0, 1)
        hard_no_keep = (
            torch.stack(hard_no_keep_masks, dim=0).amax(dim=0).clamp(0, 1)
            if hard_no_keep_masks
            else torch.zeros_like(preserve)
        )

        earring_keep = torch.zeros_like(preserve)
        for key, weight in (
            ("earring_supervision_mask", 1.0),
            ("earring_confident_mask", 1.0),
            ("source_earring_mask", 1.0),
            ("source_earring_clean_mask", 1.0),
            ("earring_query_recall_mask", 0.35),
        ):
            value = aux.get(key)
            if value is not None:
                earring_keep = torch.clamp(earring_keep + weight * resize_mask(value, size), 0, 1)
        keep_exclude_dilate = int(getattr(self.args, "earring_target_exclude_dilate", 9))
        if keep_exclude_dilate > 0:
            earring_keep = dilate_mask(earring_keep, keep_exclude_dilate)
        keep_dilate = int(getattr(self.args, "output_earring_keep_dilate", 2))
        if keep_dilate > 0:
            earring_keep = dilate_mask(earring_keep, keep_dilate)
        earring_keep = earring_keep * (1.0 - hard_no_keep).clamp(0, 1)

        preserve = preserve * (1.0 - earring_keep).clamp(0, 1)
        blur = int(getattr(self.args, "output_preserve_blur", 1))
        if blur > 1:
            if blur % 2 == 0:
                blur += 1
            preserve = F.avg_pool2d(preserve, kernel_size=blur, stride=1, padding=blur // 2).clamp(0, 1)
        aux["output_target_preserve_mask"] = preserve
        return preserve

    def _preserve_target_output(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if aux is None or not bool(getattr(self.args, "enable_output_target_preserve", True)):
            return image
        target = aux.get("target_01")
        if target is None:
            return image

        image_01 = ((image + 1) / 2).clamp(0, 1)
        target = F.interpolate(target, size=image_01.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
        preserve = self._target_output_preserve_mask(aux, image_01.shape[-2:])
        if preserve is None:
            return image

        protected = image_01 * (1.0 - preserve) + target * preserve
        return protected.clamp(0, 1) * 2 - 1

    def _revealed_skin_repair_region(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        masks = []
        for key in ("revealed_skin_mask", "M_remove_face", "cleanup_inner_edge", "M_remove_halo"):
            value = aux.get(key)
            if value is not None:
                masks.append(resize_mask(value, size))
        if not masks:
            return None

        region = torch.stack(masks, dim=0).amax(dim=0).clamp(0, 1)
        target_face = aux.get("target_face_surface_mask")
        if target_face is not None:
            region = region * resize_mask(target_face, size)
        target_hair = aux.get("target_hair_mask")
        if target_hair is not None:
            region = region * (1.0 - resize_mask(target_hair, size)).clamp(0, 1)
        return region.clamp(0, 1)

    def _repair_revealed_skin_dark_lines(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if aux is None or not bool(getattr(self.args, "enable_revealed_dark_line_repair", False)):
            return image

        image_01 = ((image + 1) / 2).clamp(0, 1)
        region = self._revealed_skin_repair_region(aux, image_01.shape[-2:])
        if region is None or region.detach().flatten(1).amax(dim=1).max().item() <= 0:
            return image

        scope = region
        source_hair = aux.get("source_hair_mask")
        source_hair_edge = None
        if source_hair is not None:
            hair_scope = resize_mask(source_hair, image_01.shape[-2:])
            edge_dilate = int(getattr(self.args, "revealed_dark_line_source_edge_dilate", 5))
            if edge_dilate > 0:
                source_hair_edge = (dilate_mask(hair_scope, edge_dilate) - erode_mask(hair_scope, edge_dilate)).clamp(0, 1)
            hair_scope = dilate_mask(hair_scope, int(getattr(self.args, "revealed_dark_line_source_hair_dilate", 9)))
            scope = scope * hair_scope

        weights = torch.tensor([0.299, 0.587, 0.114], device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
        gray = (image_01 * weights).sum(dim=1, keepdim=True)
        local_kernel = int(getattr(self.args, "revealed_dark_line_local_kernel", 11))
        if local_kernel % 2 == 0:
            local_kernel += 1
        local_gray = F.avg_pool2d(gray, kernel_size=local_kernel, stride=1, padding=local_kernel // 2)
        dark_delta = (local_gray - gray).clamp(0, 1)
        threshold = float(getattr(self.args, "revealed_dark_line_threshold", 0.025))
        line_mask = (dark_delta > threshold).float() * scope
        if source_hair_edge is not None:
            edge_threshold = float(getattr(self.args, "revealed_dark_line_edge_threshold", max(0.008, threshold * 0.4)))
            edge_line = source_hair_edge * region * (dark_delta > edge_threshold).float()
            line_mask = torch.clamp(line_mask + edge_line, 0, 1)
        if line_mask.detach().flatten(1).amax(dim=1).max().item() <= 0:
            return image

        dilate = int(getattr(self.args, "revealed_dark_line_dilate", 5))
        if dilate > 0:
            line_mask = dilate_mask(line_mask, dilate)
        blur = int(getattr(self.args, "revealed_dark_line_blur", 3))
        if blur > 1:
            if blur % 2 == 0:
                blur += 1
            line_mask = F.avg_pool2d(line_mask, kernel_size=blur, stride=1, padding=blur // 2).clamp(0, 1)

        fill_kernel = int(getattr(self.args, "revealed_dark_line_fill_kernel", 25))
        if fill_kernel % 2 == 0:
            fill_kernel += 1
        fill_valid = torch.ones_like(scope)
        target_face = aux.get("target_face_surface_mask")
        if target_face is not None:
            fill_valid = resize_mask(target_face, image_01.shape[-2:])
        target_hair = aux.get("target_hair_mask")
        if target_hair is not None:
            fill_valid = fill_valid * (1.0 - resize_mask(target_hair, image_01.shape[-2:])).clamp(0, 1)
        if source_hair is not None:
            fill_valid = fill_valid * (1.0 - resize_mask(source_hair, image_01.shape[-2:])).clamp(0, 1)

        fill_valid = fill_valid.clamp(0, 1)
        fill_sum = F.avg_pool2d(fill_valid, kernel_size=fill_kernel, stride=1, padding=fill_kernel // 2)
        weighted_sum = F.avg_pool2d(
            image_01 * fill_valid,
            kernel_size=fill_kernel,
            stride=1,
            padding=fill_kernel // 2,
        )
        fallback_fill = F.avg_pool2d(image_01, kernel_size=fill_kernel, stride=1, padding=fill_kernel // 2)
        local_fill = torch.where(fill_sum > 1e-4, weighted_sum / fill_sum.clamp_min(1e-4), fallback_fill)
        local_fill = torch.maximum(local_fill, image_01)
        line_mask = line_mask * (fill_sum > 1e-4).float()
        repaired = image_01 * (1.0 - line_mask) + local_fill * line_mask
        aux["revealed_dark_line_repair_mask"] = line_mask
        aux["revealed_dark_line_fill_valid_mask"] = fill_valid
        return repaired.clamp(0, 1) * 2 - 1

    def _face_dark_line_repair_region(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        target_face = aux.get("target_face_surface_mask")
        if target_face is None and aux.get("target_parsing") is not None:
            target_face = parsing_label_mask(aux["target_parsing"], RAW_FACE_SURFACE_LABELS)
        if target_face is None:
            return None

        region = resize_mask(target_face, size)
        for key in ("target_hair_mask", "source_hair_mask", "target_covered_ear_block_mask", "target_hair_ear_block_mask"):
            value = aux.get(key)
            if value is not None:
                region = region * (1.0 - resize_mask(value, size)).clamp(0, 1)

        target_parsing = aux.get("target_parsing")
        if target_parsing is not None:
            detail = parsing_label_mask(target_parsing, RAW_DETAIL_LABELS)
            detail = dilate_mask(resize_mask(detail, size), int(getattr(self.args, "face_dark_line_detail_exclude_dilate", 5)))
            region = region * (1.0 - detail).clamp(0, 1)

        exclude = torch.zeros_like(region)
        for key in ("ear_roi", "visible_ear_roi", "earring_confident_mask", "source_earring_mask", "source_earring_clean_mask"):
            value = aux.get(key)
            if value is not None:
                exclude = torch.clamp(exclude + resize_mask(value, size), 0, 1)
        exclude_dilate = int(getattr(self.args, "face_dark_line_ear_exclude_dilate", 9))
        if exclude_dilate > 0:
            exclude = dilate_mask(exclude, exclude_dilate)
        region = region * (1.0 - exclude).clamp(0, 1)
        return region.clamp(0, 1)

    def _repair_face_dark_line_artifacts(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if aux is None or not bool(getattr(self.args, "enable_face_dark_line_repair", False)):
            return image

        image_01 = ((image + 1) / 2).clamp(0, 1)
        region = self._face_dark_line_repair_region(aux, image_01.shape[-2:])
        if region is None or region.detach().flatten(1).amax(dim=1).max().item() <= 0:
            return image

        weights = torch.tensor([0.299, 0.587, 0.114], device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
        gray = (image_01 * weights).sum(dim=1, keepdim=True)

        local_kernel = int(getattr(self.args, "face_dark_line_local_kernel", 13))
        if local_kernel % 2 == 0:
            local_kernel += 1
        local_gray = F.avg_pool2d(gray, kernel_size=local_kernel, stride=1, padding=local_kernel // 2)

        vertical_kernel = int(getattr(self.args, "face_dark_line_vertical_kernel", 9))
        if vertical_kernel % 2 == 0:
            vertical_kernel += 1
        vertical_gray = F.avg_pool2d(
            gray,
            kernel_size=(vertical_kernel, 1),
            stride=1,
            padding=(vertical_kernel // 2, 0),
        )

        dark_delta = torch.maximum((local_gray - gray).clamp(0, 1), (vertical_gray - gray).clamp(0, 1))
        threshold = float(getattr(self.args, "face_dark_line_threshold", 0.020))
        line_mask = (dark_delta > threshold).float() * region
        if line_mask.detach().flatten(1).amax(dim=1).max().item() <= 0:
            aux["face_dark_line_region_mask"] = region
            return image

        dilate = int(getattr(self.args, "face_dark_line_dilate", 3))
        if dilate > 0:
            line_mask = dilate_mask(line_mask, dilate) * region
        blur = int(getattr(self.args, "face_dark_line_blur", 3))
        if blur > 1:
            if blur % 2 == 0:
                blur += 1
            line_mask = F.avg_pool2d(line_mask, kernel_size=blur, stride=1, padding=blur // 2).clamp(0, 1) * region

        fill_kernel = int(getattr(self.args, "face_dark_line_fill_kernel", 21))
        if fill_kernel % 2 == 0:
            fill_kernel += 1
        fill_valid = region
        fill_sum = F.avg_pool2d(fill_valid, kernel_size=fill_kernel, stride=1, padding=fill_kernel // 2)
        weighted_sum = F.avg_pool2d(
            image_01 * fill_valid,
            kernel_size=fill_kernel,
            stride=1,
            padding=fill_kernel // 2,
        )
        local_fill = torch.where(fill_sum > 1e-4, weighted_sum / fill_sum.clamp_min(1e-4), image_01)
        local_fill = torch.maximum(local_fill, image_01)

        strength = max(0.0, min(1.0, float(getattr(self.args, "face_dark_line_repair_strength", 0.85))))
        line_mask = line_mask * strength * (fill_sum > 1e-4).float()
        repaired = image_01 * (1.0 - line_mask) + local_fill * line_mask
        aux["face_dark_line_region_mask"] = region
        aux["face_dark_line_repair_mask"] = line_mask
        return repaired.clamp(0, 1) * 2 - 1

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
        image = self._repair_revealed_skin_dark_lines(image, aux)
        image = self._repair_face_dark_line_artifacts(image, aux)
        return image, aux
