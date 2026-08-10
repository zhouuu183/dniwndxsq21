from __future__ import annotations

import argparse
import inspect
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoders import FeatureEncoderMult, FeatureiResnet, ModulationModule
from utils.hair_color_match_v8 import lab_to_rgb, rgb_to_lab
from models.ear_modules_v5 import (
    RAW_DETAIL_LABELS,
    RAW_EAR_SURFACE_LABELS,
    RAW_EARRING,
    RAW_FACE_SURFACE_LABELS,
    RAW_HAIR,
    RAW_SKIN_SURFACE_LABELS,
    BrightnessReEstimator,
    DynamicFineMaskRefresher,
    EarAnchoredQueryBuilder,
    FaceParsingHelperV5,
    HFDAGatedInjectionUnit,
    ShadowSuppressedHFExtractor,
    build_earring_search_mask,
    build_earring_write_masks,
    build_strong_earring_candidate,
    build_source_earring_instance_masks_v5,
    refine_earring_hoops_highres,
    expand_valid_roi_by_completion,
    build_revealed_skin_mask,
    build_weak_earring_masks,
    dilate_mask,
    enhance_query_with_earring_recall,
    ensure_mask_4d,
    erode_mask,
    assign_components_to_ear_sides,
    gaussian_blur,
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


def build_lower_face_skin_reference_mask(
    face_skin_mask: torch.Tensor,
    revealed_skin_mask: torch.Tensor,
    *,
    hair_mask: torch.Tensor | None = None,
    detail_mask: torch.Tensor | None = None,
    earring_mask: torch.Tensor | None = None,
    lower_start_ratio: float = 0.46,
) -> torch.Tensor:
    """Select real lower-face skin as the forehead tone fallback.

    The fallback is deliberately geometric only after semantic exclusions: it
    starts below the revealed-forehead band, removes hair/facial features/ear
    accessories and erodes the remainder away from parser boundaries.  It is
    therefore safe when local forehead reference pixels were hidden by bangs.
    """
    face = ensure_mask_4d(face_skin_mask).float().clamp(0, 1)
    revealed = resize_mask(revealed_skin_mask, face.shape[-2:]).clamp(0, 1)
    height, width = face.shape[-2:]
    y = torch.arange(height, device=face.device, dtype=face.dtype).view(1, 1, height, 1)
    base_start = int(round(max(0.0, min(0.85, float(lower_start_ratio))) * height))

    # Move the start below the actual revealed band per sample, but cap it so a
    # noisy mask cannot discard the entire cheek/chin reference.
    rows = (revealed > 0.05).amax(dim=3).float()
    row_ids = torch.arange(height, device=face.device, dtype=face.dtype).view(1, 1, height)
    last_revealed = (rows * row_ids).amax(dim=2, keepdim=True).view(-1, 1, 1, 1)
    start = torch.maximum(
        torch.full_like(last_revealed, float(base_start)),
        last_revealed + 3.0,
    ).clamp(max=float(int(0.70 * height)))
    lower = (y >= start).to(face.dtype).expand(-1, -1, -1, width)

    safe = face * lower * (1.0 - dilate_mask(revealed, 5)).clamp(0, 1)
    for mask, dilation in ((hair_mask, 5), (detail_mask, 3), (earring_mask, 5)):
        if mask is not None:
            safe = safe * (1.0 - dilate_mask(resize_mask(mask, face.shape[-2:]), dilation)).clamp(0, 1)
    return erode_mask(safe, 3).clamp(0, 1)


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
            source_hair_block_dilate=getattr(self.args, "source_hair_block_dilate", 8),
            source_hair_block_strength=getattr(self.args, "source_hair_block_strength", 0.95),
            target_visibility_expand=getattr(self.args, "target_visibility_expand", 5),
            max_target_hair_overlap=getattr(self.args, "max_target_hair_overlap", 0.30),
            min_target_visible_overlap=getattr(self.args, "min_target_visible_overlap", 0.10),
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

    @staticmethod
    def _mask_like(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        reference = ensure_mask_4d(reference).float()
        if mask is None:
            return torch.zeros_like(reference)
        return resize_mask(
            ensure_mask_4d(mask).to(device=reference.device, dtype=reference.dtype),
            reference.shape[-2:],
        )

    def _source_earring_case_masks(
        self,
        source_parsing: torch.Tensor,
        query_info: dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return parser evidence and a spatial no-earring gate per sample.

        Weak image features are deliberately not allowed to create an earring
        case.  Parser label-9, an explicit object mask, and a separately
        validated strong visual candidate are the only presence authorities.
        """

        reference = ensure_mask_4d(reference).float()
        parsing = ensure_mask_4d(source_parsing).to(device=reference.device)
        if parsing.shape[-2:] != reference.shape[-2:]:
            parsing = F.interpolate(parsing.float(), size=reference.shape[-2:], mode="nearest")
        parser_earring = (parsing.long() == RAW_EARRING).to(dtype=reference.dtype)

        # Detection/query masks are deliberately excluded from presence
        # detection.  They are search proposals and often fire on ear edges,
        # short-hair background, or shadows.  Only parser label-9 and explicit
        # object masks can enable the earring branch; otherwise a source with
        # no earring must remain a hard no-op.
        evidence = parser_earring
        for key in (
            "source_earring_explicit_mask",
            "source_earring_object_mask",
            "strong_earring_candidate_core",
        ):
            value = query_info.get(key)
            if value is not None:
                evidence = torch.clamp(evidence + self._mask_like(value, reference), 0, 1)

        height, width = reference.shape[-2:]
        area_scale = float(height * width) / float(256 * 256)
        configured_min_area = getattr(self.args, "earring_source_presence_min_area", None)
        if configured_min_area is None:
            # Four 256-space pixels is still small enough for a stud, but avoids
            # letting one or two noisy parser pixels activate the whole branch.
            configured_min_area = max(
                4.0,
                float(getattr(self.args, "earring_min_source_parser_area", 0.5)),
            )
        min_area = max(0.0, float(configured_min_area) * area_scale)
        if bool(getattr(self.args, "disable_earring_path_if_low_confidence", True)):
            present = evidence.flatten(1).sum(dim=1) >= min_area
        else:
            present = evidence.flatten(1).amax(dim=1) > 0
        no_earring = (~present).to(dtype=reference.dtype).view(-1, 1, 1, 1)
        no_earring = no_earring.expand_as(reference)
        return parser_earring.clamp(0, 1), no_earring

    def _finalize_earring_masks(
        self,
        query_info: dict[str, torch.Tensor],
        source_parsing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Turn broad earring recall into a narrow, target-safe write mask."""

        reference = ensure_mask_4d(query_info["query_mask"]).float()
        parser_earring, no_earring = self._source_earring_case_masks(
            source_parsing,
            query_info,
            reference,
        )
        active = (1.0 - no_earring).clamp(0, 1)
        query_info["source_parser_earring_mask"] = parser_earring
        query_info["no_earring_case_mask"] = no_earring
        query_info["no_earring_case"] = no_earring

        # Search may be wide enough to find an under-segmented dangling object,
        # but it never grants permission to write source pixels into target hair.
        earring_roi = query_info.get(
            "earring_valid_roi",
            query_info.get("visible_ear_roi", query_info.get("ear_roi", reference)),
        )
        earring_roi = self._mask_like(earring_roi, reference)
        online_candidate = query_info.get(
            "online_earring_candidate_mask",
            query_info.get("earring_candidate_mask"),
        )
        search_mask = build_earring_search_mask(
            parser_earring,
            earring_roi,
            online_candidate_mask=online_candidate,
            weak_recall_mask=query_info.get("earring_query_recall_mask"),
            lobe_search_mask=query_info.get("source_lobe_search_mask"),
            downward_shift=int(getattr(self.args, "earring_search_downward_shift", 10)),
            search_dilate=int(getattr(self.args, "earring_search_dilate", 7)),
            no_earring=no_earring,
        )
        search_mask = (search_mask * active).clamp(0, 1)

        # Trusted object core is deliberately narrow.  Online/search/recall
        # masks are completion proposals only and can never bypass source
        # background/hair blockers.
        trusted_object = parser_earring.clone()
        for key in (
            "source_earring_explicit_mask",
            "source_earring_object_mask",
            "strong_earring_candidate_core",
        ):
            value = query_info.get(key)
            if value is not None:
                value = self._mask_like(value, reference)
                trusted_object = torch.clamp(trusted_object + value, 0, 1)
        completion_candidate = torch.zeros_like(reference)
        for key in (
            "online_earring_candidate_mask",
            "earring_candidate_mask",
            "earring_query_recall_mask",
            "earring_object_recall_mask",
            "earring_object_detection_mask",
        ):
            value = query_info.get(key)
            if value is not None:
                completion_candidate = torch.clamp(
                    completion_candidate + self._mask_like(value, reference), 0, 1
                )
        trusted_object = trusted_object * active
        completion_candidate = completion_candidate * search_mask * active

        target_hair_occlusion = query_info.get("target_ear_hair_occlusion_mask")
        if target_hair_occlusion is None:
            target_hair = self._mask_like(query_info.get("target_hair_mask"), reference)
            ear_roi = self._mask_like(query_info.get("ear_roi"), reference)
            target_hair_occlusion = target_hair * ear_roi
        else:
            target_hair_occlusion = self._mask_like(target_hair_occlusion, reference)

        # Ear visibility is decided by the query builder's parser-confirmed OR
        # semantic-skin fallback paths.  A sparse parser ear must not close an
        # otherwise exposed side, and no image-centre split is involved.
        left_roi = self._mask_like(query_info.get("left_ear_roi"), reference)
        right_roi = self._mask_like(query_info.get("right_ear_roi"), reference)
        left_active = self._mask_like(query_info.get("left_side_active"), reference)
        right_active = self._mask_like(query_info.get("right_side_active"), reference)
        left_missing = (
            left_active.detach().flatten(1).amax(dim=1) <= 0
        ).view(-1, 1, 1, 1)
        if left_missing.any().item():
            left_ear = self._mask_like(query_info.get("target_left_ear_mask"), reference)
            left_skin = self._mask_like(query_info.get("target_skin_surface_mask"), reference)
            left_fallback = (
                (left_ear + left_skin * left_roi * (1.0 - target_hair_occlusion)).clamp(0, 1)
                .flatten(1)
                .sum(dim=1, keepdim=True)
                >= float(getattr(self.args, "min_target_ear_area", 8.0))
            ).to(reference.dtype).view(-1, 1, 1, 1)
            left_active = torch.where(left_missing, left_fallback, left_active)
        right_missing = (
            right_active.detach().flatten(1).amax(dim=1) <= 0
        ).view(-1, 1, 1, 1)
        if right_missing.any().item():
            right_ear = self._mask_like(query_info.get("target_right_ear_mask"), reference)
            right_skin = self._mask_like(query_info.get("target_skin_surface_mask"), reference)
            right_fallback = (
                (right_ear + right_skin * right_roi * (1.0 - target_hair_occlusion)).clamp(0, 1)
                .flatten(1)
                .sum(dim=1, keepdim=True)
                >= float(getattr(self.args, "min_target_ear_area", 8.0))
            ).to(reference.dtype).view(-1, 1, 1, 1)
            right_active = torch.where(right_missing, right_fallback, right_active)
        side_gate = torch.clamp(left_active * left_roi + right_active * right_roi, 0, 1)
        # ``left/right_roi`` are deliberately compact ear-local regions.  They
        # are suitable for deciding which side is visible, but are too short to
        # be a final geometric clip for a parser-missed hoop.  Keep the scalar
        # visibility decision separately so a validated outer arc can extend
        # beyond the raw ear box without opening a covered side.
        visible_side_gate = torch.clamp(left_active + right_active, 0, 1)
        earring_roi = earring_roi * side_gate

        left_lobe_anchor = self._mask_like(query_info.get("left_lobe_anchor"), reference)
        right_lobe_anchor = self._mask_like(query_info.get("right_lobe_anchor"), reference)
        earlobe_anchor = (
            left_lobe_anchor * left_active * left_roi
            + right_lobe_anchor * right_active * right_roi
        ).clamp(0, 1) * (1.0 - target_hair_occlusion).clamp(0, 1) * active
        trusted_left, trusted_right = assign_components_to_ear_sides(
            trusted_object,
            left_roi,
            right_roi,
            left_lobe_anchor,
            right_lobe_anchor,
        )
        completion_left, completion_right = assign_components_to_ear_sides(
            completion_candidate,
            left_roi,
            right_roi,
            left_lobe_anchor,
            right_lobe_anchor,
        )
        trusted_object = (
            trusted_left * left_active + trusted_right * right_active
        ).clamp(0, 1) * visible_side_gate
        completion_candidate = (
            completion_left * left_active + completion_right * right_active
        ).clamp(0, 1) * visible_side_gate

        # A parser-missed hoop's outer arc must not be clipped by the original
        # ear shell. Expansion is confined to the validated, lobe-connected
        # object itself, never to a geometric ear/background region.
        earring_roi = expand_valid_roi_by_completion(
            earring_roi,
            trusted_object,
            grow_iters=int(getattr(self.args, "earring_write_connectivity_iters", 32)),
            seed_dilate=max(1, int(getattr(self.args, "earring_write_bridge_dilate", 17))),
        ) * visible_side_gate

        source_labels = ensure_mask_4d(source_parsing).to(device=reference.device)
        if source_labels.shape[-2:] != reference.shape[-2:]:
            source_labels = F.interpolate(
                source_labels.float(),
                size=reference.shape[-2:],
                mode="nearest",
            )
        source_background = (source_labels.long() == 0).to(dtype=reference.dtype)
        source_non_earring_semantic = (
            source_labels.long() != RAW_EARRING
        ).to(dtype=reference.dtype)
        source_hair = query_info.get("source_hair_mask")
        if source_hair is None:
            source_hair = (source_labels.long() == RAW_HAIR).to(dtype=reference.dtype)

        connectivity_iters = int(getattr(self.args, "earring_write_connectivity_iters", 32))
        connectivity_kernel = int(getattr(self.args, "earring_write_connectivity_kernel", 5))
        bridge_dilate = int(getattr(self.args, "earring_write_bridge_dilate", 17))
        write_masks = build_earring_write_masks(
            trusted_object,
            completion_candidate,
            earring_roi,
            target_hair_occlusion,
            earlobe_anchor,
            no_earring=no_earring,
            source_background_mask=source_background,
            source_hair_mask=source_hair,
            source_hair_block_mask=query_info.get("source_hair_block_mask"),
            source_semantic_block_mask=source_non_earring_semantic,
            max_target_hair_overlap=float(
                getattr(self.args, "earring_write_max_target_hair_overlap", 0.30)
            ),
            source_block_dilate=int(getattr(self.args, "earring_write_source_block_dilate", 3)),
            write_dilate=int(getattr(self.args, "earring_write_dilate", 3)),
            connectivity_iters=connectivity_iters,
            connectivity_kernel=connectivity_kernel,
            bridge_dilate=bridge_dilate,
        )
        write_mask = write_masks["write_mask"] * active
        visible_segment = write_masks["earring_object_mask"] * active
        # The write mask is source-object supported.  A source-native
        # high-resolution instance pass will later provide a second, more
        # accurate hollow-hoop mask for the final RGB composite.
        hoop_hole = write_masks["hoop_hole_mask"]
        # Redundant final exclusion is intentional: later feature-mask floors
        # must never refill a hoop centre.
        write_mask = write_mask * (1.0 - hoop_hole).clamp(0, 1)
        # Hair beside/below the ear that is *not* occupied by a verified earring
        # remains protected.  This debug/constraint mask is the complement of
        # write permission inside the target-hair occlusion, not the overlap.
        target_hair_bridge = (
            target_hair_occlusion * (1.0 - write_mask).clamp(0, 1)
        ).clamp(0, 1)

        # The ordinary ear/face detail query remains intact.  Only its broad
        # earring-search extension is added here; all write-facing masks below
        # use the connected segment instead.
        search_weight = max(0.0, float(getattr(self.args, "earring_query_boost", 1.0)))
        query_info["query_mask"] = torch.clamp(reference + search_weight * search_mask, 0, 1)
        query_info["earring_search_mask"] = search_mask
        query_info["earring_write_mask"] = write_mask
        query_info["earring_visible_segment_mask"] = visible_segment * active
        query_info["earring_object_mask"] = write_masks["earring_object_mask"]
        query_info["earring_filled_mask"] = write_masks["earring_filled_mask"]
        query_info["hoop_hole_mask"] = hoop_hole
        query_info["earring_core_mask"] = write_masks["core_mask"]
        query_info["earring_completion_mask"] = write_masks["completion_mask"]
        query_info["left_core_mask"], query_info["right_core_mask"] = assign_components_to_ear_sides(
            write_masks["core_mask"], left_roi, right_roi, left_lobe_anchor, right_lobe_anchor
        )
        query_info["left_completion_mask"], query_info["right_completion_mask"] = assign_components_to_ear_sides(
            write_masks["completion_mask"], left_roi, right_roi, left_lobe_anchor, right_lobe_anchor
        )
        query_info["left_write_mask"], query_info["right_write_mask"] = assign_components_to_ear_sides(
            write_mask, left_roi, right_roi, left_lobe_anchor, right_lobe_anchor
        )
        query_info["left_search_mask"], query_info["right_search_mask"] = assign_components_to_ear_sides(
            search_mask, left_roi, right_roi, left_lobe_anchor, right_lobe_anchor
        )
        # Keep target-side openness separate from source-earring presence.
        # A missed small source earring must not erase the fact that its target
        # lobe is exposed before the high-resolution instance detector runs.
        query_info["left_target_side_open"] = left_active
        query_info["right_target_side_open"] = right_active
        query_info["left_side_active"] = left_active * active
        query_info["right_side_active"] = right_active * active
        query_info["target_hair_ear_bridge_mask"] = target_hair_bridge

        # Earring supervision/injection must consume the write mask.  Detection
        # candidates stay available under their explicit debug keys, but cannot
        # punch a hole in target hair or activate an absent-source sample.
        query_info["source_earring_mask"] = write_mask
        query_info["earring_confident_mask"] = write_mask
        highlight = query_info.get("earring_highlight_mask")
        if highlight is not None:
            query_info["earring_highlight_mask"] = self._mask_like(highlight, reference) * write_mask

        for key in (
            "source_earring_detection_mask",
            "source_earring_object_mask",
            "earring_candidate_mask",
            "online_earring_candidate_mask",
            "online_earring_search_mask",
            "earring_query_recall_mask",
            "earring_object_recall_mask",
            "earring_object_detection_mask",
            "earring_recall_lobe_support",
            "earring_recall_block_protect_mask",
        ):
            value = query_info.get(key)
            if value is not None:
                query_info[key] = self._mask_like(value, reference) * active

        query_info["earring_visibility_mask"] = earring_roi * active
        query_info["earring_valid_roi"] = earring_roi * active
        for key in ("left_earring_valid_roi", "right_earring_valid_roi"):
            value = query_info.get(key)
            if value is not None:
                query_info[key] = self._mask_like(value, reference) * active
        presence_target = query_info.get("presence_target")
        if presence_target is not None:
            sample_active = active.flatten(1).amax(dim=1, keepdim=True)
            query_info["presence_target"] = presence_target * sample_active
        return query_info

    def _enhance_query_recall(
        self,
        source_01: torch.Tensor,
        source_parsing: torch.Tensor,
        query_info: dict[str, torch.Tensor],
        target_01: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_mask = ensure_mask_4d(query_info["query_mask"]).float()
        # Build the visual proposal before deciding presence.  Only the strict
        # strong subset below may activate a parser-missed earring; all weak
        # recall products remain search-only after that decision.
        detection_ear_roi = query_info.get("ear_roi", query_mask)
        source_earring_detection_mask = self._mask_like(
            query_info.get("source_earring_detection_mask"),
            query_mask,
        )
        weak_masks = build_weak_earring_masks(
            source_01,
            detection_ear_roi,
            query_mask,
            source_earring_detection_mask,
            query_info.get("source_hair_mask"),
            query_info.get("source_hair_block_mask"),
            source_parsing,
        )
        source_labels = ensure_mask_4d(source_parsing).to(device=query_mask.device)
        if source_labels.shape[-2:] != query_mask.shape[-2:]:
            source_labels = F.interpolate(source_labels.float(), size=query_mask.shape[-2:], mode="nearest")
        strong_info = build_strong_earring_candidate(
            source_01,
            weak_masks["earring_candidate_mask"],
            # A parser-missed hoop commonly extends beyond the raw ear ROI.
            # Source-lobe support follows the real ear downward while remaining
            # semantic/side constrained, so it is safe to use for validation.
            weak_masks.get("source_lobe_search_mask", detection_ear_roi),
            query_info.get("left_ear_roi", detection_ear_roi),
            query_info.get("right_ear_roi", detection_ear_roi),
            query_info.get("left_lobe_anchor", query_info.get("target_left_ear_mask", detection_ear_roi)),
            query_info.get("right_lobe_anchor", query_info.get("target_right_ear_mask", detection_ear_roi)),
            source_background_mask=(source_labels.long() == 0).to(dtype=query_mask.dtype),
            source_hair_mask=query_info.get("source_hair_mask"),
            source_ear_mask=parsing_label_mask(source_parsing, RAW_EAR_SURFACE_LABELS),
            parser_earring_mask=source_earring_detection_mask,
        )
        query_info["strong_earring_candidate_core"] = strong_info["strong_candidate_mask"]
        query_info["left_strong_candidate"] = strong_info["left_strong_candidate"]
        query_info["right_strong_candidate"] = strong_info["right_strong_candidate"]
        query_info["left_elliptical_hoop"] = strong_info["left_elliptical_hoop"]
        query_info["right_elliptical_hoop"] = strong_info["right_elliptical_hoop"]
        query_info["elliptical_hoop_hole"] = strong_info["elliptical_hoop_hole"]
        query_info["left_elliptical_hoop_hole"] = strong_info["left_elliptical_hoop_hole"]
        query_info["right_elliptical_hoop_hole"] = strong_info["right_elliptical_hoop_hole"]
        parser_earring, no_earring = self._source_earring_case_masks(
            source_parsing,
            query_info,
            query_mask,
        )
        active = (1.0 - no_earring).clamp(0, 1)
        query_info["source_parser_earring_mask"] = parser_earring
        query_info["no_earring_case_mask"] = no_earring
        query_info["no_earring_case"] = no_earring

        # With no trusted or strong evidence the complete earring branch is a
        # hard no-op.  Weak search products must not create a lower-ear hole.
        if float(active.detach().amax().item()) <= 0:
            zero = torch.zeros_like(query_mask)
            for key in (
                "earring_search_mask",
                "earring_write_mask",
                "earring_visible_segment_mask",
                "target_hair_ear_bridge_mask",
                "source_earring_mask",
                "earring_confident_mask",
                "earring_candidate_mask",
                "online_earring_candidate_mask",
                "online_earring_search_mask",
                "earring_query_recall_mask",
                "earring_object_recall_mask",
                "earring_object_detection_mask",
            ):
                query_info[key] = zero
            return query_info

        if not bool(getattr(self.args, "enable_earring_query_recall", True)):
            return query_info

        visible_ear_roi = query_info.get("visible_ear_roi", query_info.get("ear_roi", query_mask))
        earring_valid_roi = query_info.get("earring_valid_roi", visible_ear_roi)
        source_earring_detection_mask = source_earring_detection_mask * active
        source_hair_block_mask = query_info.get("source_hair_block_mask")
        existing_source_mask = query_info.get("source_earring_mask")
        existing_object_mask = query_info.get("source_earring_object_mask")
        recall_info = enhance_query_with_earring_recall(
            query_mask,
            source_earring_detection_mask,
            source_hair_block_mask,
            weak_masks,
            visibility_mask=self._mask_like(earring_valid_roi, query_mask) * active,
            recall_dilate=getattr(self.args, "earring_query_recall_dilate", 7),
            downward_shift=getattr(self.args, "earring_query_downward_shift", 10),
            lower_lobe_weight=getattr(self.args, "earring_query_lower_lobe_weight", 0.20),
            candidate_boost=getattr(self.args, "earring_query_candidate_boost", 0.90),
            block_protect=getattr(self.args, "earring_query_block_protect", 0.95),
        )
        earring_valid_mask = resize_mask(earring_valid_roi, query_mask.shape[-2:]) * active
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
        ) * active

        # Pixel-level earring vs background separation.  The recall candidate may
        # contain background/hair pixels behind the earring (especially for long
        # dangling earrings below the lobe).  Two filters isolate true earring:
        # 1. Color-difference filter: earrings (metal/jewel) have large LAB ΔE vs
        #    target hair; background behind the earring often blends with target.
        # 2. High-frequency filter: earrings have edges (reflections, bead borders);
        #    background in the small earring ROI is typically low-frequency flat.
        # Intersection = "chromatically distinct AND structured" = true earring.
        # TEMPORARILY DISABLED: the default thresholds are too aggressive and kill
        # valid earrings.  Re-enable after tuning thresholds on actual samples.
        use_pixel_filter = bool(getattr(self.args, "earring_pixel_filter", False))  # False = disabled
        if use_pixel_filter and target_01 is not None and recall_info["source_earring_mask"].sum() > 0:
            candidate = recall_info["source_earring_mask"]
            size = candidate.shape[-2:]

            # Layer 1: color difference (LAB ΔE > threshold)
            source_lab = rgb_to_lab(source_01)
            target_lab = rgb_to_lab(target_01)
            if source_lab.shape[-2:] != size:
                source_lab = F.interpolate(source_lab, size=size, mode="bilinear", align_corners=False)
            if target_lab.shape[-2:] != size:
                target_lab = F.interpolate(target_lab, size=size, mode="bilinear", align_corners=False)
            delta_e = torch.sqrt(((source_lab - target_lab) ** 2).sum(dim=1, keepdim=True).clamp(min=0))
            color_threshold = float(getattr(self.args, "earring_color_diff_threshold", 12.0))
            color_distinct = (delta_e >= color_threshold).float()

            # Layer 2: high-frequency structure (Sobel edge magnitude > threshold)
            source_gray = source_01.mean(dim=1, keepdim=True)
            if source_gray.shape[-2:] != size:
                source_gray = F.interpolate(source_gray, size=size, mode="bilinear", align_corners=False)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=source_gray.dtype, device=source_gray.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=source_gray.dtype, device=source_gray.device).view(1, 1, 3, 3)
            edge_x = F.conv2d(source_gray, sobel_x, padding=1)
            edge_y = F.conv2d(source_gray, sobel_y, padding=1)
            edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            structure_threshold = float(getattr(self.args, "earring_structure_threshold", 0.08))
            has_structure = (edge_mag >= structure_threshold).float()

            # Apply pixel-level filter: candidate ∩ color_distinct ∩ has_structure
            # Dilate slightly to recover pixels near detected edges (earring body
            # inside the edge contour may be flat but is still part of the earring).
            structure_support = F.max_pool2d(has_structure, kernel_size=3, stride=1, padding=1)
            pixel_filtered = candidate * color_distinct * structure_support

            # Layer 3: connectivity filter — earrings hang from the lobe, so true
            # earring pixels must be spatially connected to the target ear anchor.
            # Background/hair pixels behind or around the earring are typically
            # disconnected islands after layers 1 and 2 filter out the low-contrast
            # bridge between them and the lobe.  Geodesic growth from the lobe along
            # pixel_filtered keeps only the connected component touching the ear.
            target_left_ear = query_info.get("target_left_ear_mask")
            target_right_ear = query_info.get("target_right_ear_mask")
            if target_left_ear is not None and target_right_ear is not None:
                lobe_anchor = target_left_ear + target_right_ear
                if lobe_anchor.shape[-2:] != size:
                    lobe_anchor = F.interpolate(
                        ensure_mask_4d(lobe_anchor).float(),
                        size=size,
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    lobe_anchor = ensure_mask_4d(lobe_anchor).float()
                lobe_anchor = (lobe_anchor > 0.5).float()

                # Dilate lobe slightly to bridge small parser gaps between ear and earring attachment.
                lobe_seed = F.max_pool2d(lobe_anchor, kernel_size=5, stride=1, padding=2)
                # Geodesic dilation: grow from lobe_seed along pixel_filtered for N iterations.
                grown = (lobe_seed * pixel_filtered).clamp(0, 1)
                connectivity_iters = int(getattr(self.args, "earring_connectivity_iters", 12))
                for _ in range(connectivity_iters):
                    grown = (F.max_pool2d(grown, kernel_size=3, stride=1, padding=1) * pixel_filtered).clamp(0, 1)
                    if grown.sum() == 0:
                        break
                pixel_filtered = grown

            # Fallback: if filtering kills >95% of the candidate (over-aggressive on
            # valid low-contrast earrings like pearls), keep the original candidate.
            candidate_area = candidate.flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
            filtered_area = pixel_filtered.flatten(1).sum(dim=1, keepdim=True)
            survival_ratio = filtered_area / candidate_area
            use_filtered = (survival_ratio >= 0.05).float().view(-1, 1, 1, 1)
            recall_info["source_earring_mask"] = torch.where(
                use_filtered.expand_as(candidate) > 0.5,
                pixel_filtered,
                candidate,
            )

        # Per-side visibility gating: replace pixel-wise multiply with per-side
        # scalar decision (matches pp_gen_v5 logic).  If a side is visible
        # (valid_roi non-empty), keep the FULL completed earring; if occluded,
        # zero that side.  Prevents shell from clipping completed outer arcs.
        left_earring_valid_roi = query_info.get("left_earring_valid_roi")
        right_earring_valid_roi = query_info.get("right_earring_valid_roi")
        if left_earring_valid_roi is not None and right_earring_valid_roi is not None:
            height, width = recall_info["source_earring_mask"].shape[-2:]
            area_scale = float(height * width) / float(256 * 256)
            min_visible_area = 8.0 * area_scale

            left_visible = left_earring_valid_roi.flatten(1).sum(dim=1, keepdim=True) >= min_visible_area
            right_visible = right_earring_valid_roi.flatten(1).sum(dim=1, keepdim=True) >= min_visible_area

            left_mask, right_mask = assign_components_to_ear_sides(
                recall_info["source_earring_mask"],
                query_info.get("left_ear_roi", left_earring_valid_roi),
                query_info.get("right_ear_roi", right_earring_valid_roi),
                query_info.get("left_lobe_anchor"),
                query_info.get("right_lobe_anchor"),
            )

            left_mask = left_mask * left_visible.view(-1, 1, 1, 1).float()
            right_mask = right_mask * right_visible.view(-1, 1, 1, 1).float()

            recall_info["source_earring_mask"] = torch.clamp(left_mask + right_mask, 0, 1)
        else:
            # Fallback: if left/right split not available, use the combined valid_roi
            # as a pixel-wise gate (preserves old behavior when query_info doesn't
            # provide per-side ROIs).
            recall_info["source_earring_mask"] = recall_info["source_earring_mask"] * earring_valid_mask

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
        confident_mask = torch.clamp(confident_mask + existing_confident, 0, 1) * earring_valid_mask

        # Rebuild a permissive search mask from object evidence.  This mask may
        # look below the lobe for a long earring, but remains distinct from the
        # final write mask created in _resolve_masks.
        search_mask = build_earring_search_mask(
            parser_earring,
            earring_valid_mask,
            online_candidate_mask=recall_info.get("online_earring_candidate_mask"),
            weak_recall_mask=recall_info.get("earring_query_recall_mask"),
            lobe_search_mask=recall_info.get("source_lobe_search_mask"),
            downward_shift=int(getattr(self.args, "earring_search_downward_shift", 10)),
            search_dilate=int(getattr(self.args, "earring_search_dilate", 7)),
            no_earring=no_earring,
        )
        search_mask = (search_mask * active).clamp(0, 1)
        recall_info["source_earring_mask"] = (
            recall_info["source_earring_mask"] * search_mask * active
        ).clamp(0, 1)
        recall_info["earring_confident_mask"] = (confident_mask * search_mask * active).clamp(0, 1)
        recall_info["earring_candidate_mask"] = (
            recall_info["online_earring_candidate_mask"] * search_mask * active
        ).clamp(0, 1)
        recall_info["earring_search_mask"] = search_mask
        recall_info["online_earring_search_mask"] = (
            recall_info["online_earring_search_mask"] * active
        ).clamp(0, 1)
        recall_info["earring_highlight_mask"] = (
            resize_mask(weak_masks["earring_highlight_mask"], query_mask.shape[-2:])
            * search_mask
            * active
        )
        # Never replace the ordinary visible-ear/detail query with the earring
        # branch.  Only the positive recall delta is admitted through search.
        enhanced_query = self._mask_like(recall_info.get("query_mask"), query_mask)
        recall_delta = (enhanced_query - query_mask).clamp(0, 1)
        recall_info["query_mask"] = torch.clamp(query_mask + recall_delta * search_mask * active, 0, 1)
        recall_info["earring_visibility_mask"] = earring_valid_mask
        query_info.update(recall_info)
        return query_info

    def _apply_earring_fine_mask_floor(
        self,
        fine_mask: torch.Tensor,
        query_mask: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        floor = max(0.0, min(1.0, float(getattr(self.args, "earring_fine_mask_floor", 0.18))))
        query_mask = ensure_mask_4d(query_mask).float()

        def aux_mask(name: str) -> torch.Tensor:
            value = aux.get(name)
            if value is None:
                return torch.zeros_like(query_mask)
            return resize_mask(value, query_mask.shape[-2:])

        hoop_hole = aux_mask("hoop_hole_mask").clamp(0, 1)
        # Search candidates are diagnostic/detection masks only.  They must not
        # raise the learned fine mask outside the final connected write region.
        write_mask = aux_mask("earring_write_mask").clamp(0, 1)
        support = write_mask
        visibility_mask = (write_mask > 1e-4).to(write_mask.dtype)
        support = dilate_mask(support, int(getattr(self.args, "earring_fine_mask_dilate", 5)))
        support = support * query_mask * visibility_mask
        if floor <= 0 or support.detach().flatten(1).amax(dim=1).max().item() <= 0:
            result = fine_mask * (1.0 - hoop_hole).clamp(0, 1)
            aux["earring_final_alpha"] = result * write_mask * (1.0 - hoop_hole).clamp(0, 1)
            return result

        aux["fine_mask_before_floor"] = fine_mask
        aux["earring_fine_floor_support"] = support
        result = torch.maximum(fine_mask, floor * support)
        # Second hoop constraint: downstream learned injection must retain the
        # target RGB in every enclosed ring centre.
        result = result * (1.0 - hoop_hole).clamp(0, 1)
        aux["earring_final_alpha"] = result * write_mask * (1.0 - hoop_hole).clamp(0, 1)
        return result

    def _build_source_content_gate(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        """Only genuine source pixels may be sampled for detail/earring recovery.

        The v58 ear ROI is a geometric dilation around the ears.  When the source
        has short hair or a bare ear, that dilation also covers source background,
        neck and no-hair gaps.  Because detail/earring recovery reads source
        texture at these same coordinates, an over-wide ROI copies source
        background and lighting into the target hair -> boundary seams and dark
        earring "holes".  This gate restricts every source-sampling mask to real
        source content: skin (1,10), ears (7,8) and the earring (9).  Source hair
        (17), hat, clothing and background are excluded, so nothing outside the
        actual subject can be written back onto the target.
        """
        if not bool(getattr(self.args, "enable_source_content_gate", True)):
            return None
        source_parsing = aux.get("source_parsing")
        if source_parsing is None:
            return None

        skin = aux.get("source_skin_surface_mask")
        if skin is None:
            skin = parsing_label_mask(source_parsing, RAW_SKIN_SURFACE_LABELS)
        skin = resize_mask(skin, size)
        earring = resize_mask((ensure_mask_4d(source_parsing).long() == RAW_EARRING).float(), size)

        # In the lower-ear region (where earrings hang and background/neck may
        # appear in short-hair sources), restrict content to ONLY the actual ear
        # labels (7, 8) and earring (9), excluding face (1) and neck (10).  This
        # prevents "background bleed" when recovering earrings: if the source has
        # short hair with visible ears, the gate now samples only the ear itself,
        # not the surrounding neck/background, avoiding gaps between ear and hair.
        ear_roi = aux.get("ear_roi")
        if ear_roi is not None:
            ear_roi_resized = resize_mask(ear_roi, size)
            h = ear_roi_resized.shape[-2]
            lower_ear_start = int(h * 0.55)
            lower_ear_region = torch.zeros_like(ear_roi_resized)
            lower_ear_region[..., lower_ear_start:, :] = ear_roi_resized[..., lower_ear_start:, :]

            # In the lower-ear region, only keep pixels that are actual ear or earring
            ear_labels = parsing_label_mask(source_parsing, RAW_EAR_SURFACE_LABELS)  # (7, 8)
            ear_labels_resized = resize_mask(ear_labels, size)
            strict_ear_content = torch.clamp(ear_labels_resized + earring, 0, 1)

            # Outside lower-ear: use the full skin mask; inside: use strict ear+earring only
            content = skin * (1.0 - lower_ear_region) + strict_ear_content * lower_ear_region
        else:
            content = torch.clamp(skin + earring, 0, 1)

        # Explicitly carve out source hair so a stray skin-label pixel under a
        # bang cannot re-admit occluded, wrongly-lit texture.  But protect the
        # earring label-9 (parser-confirmed real earring) from being removed by
        # source hair under-segmentation (fine hair strands covering the earring).
        source_hair = aux.get("source_hair_mask")
        if source_hair is not None:
            hair_mask = resize_mask(source_hair, size)
            # Subtract hair only from non-earring pixels.
            hair_to_subtract = hair_mask * (1.0 - earring).clamp(0, 1)
            content = content * (1.0 - hair_to_subtract).clamp(0, 1)

        # A small dilation keeps the true ear/earring edge (the parser slightly
        # under-segments thin earring wires).
        gate_dilate = int(getattr(self.args, "source_content_gate_dilate", 3))
        if gate_dilate > 0:
            content = dilate_mask(content, gate_dilate)
        # The second hair subtraction keeps background/neck from bleeding through
        # the dilation, and again protects the earring.
        if source_hair is not None:
            hair_to_subtract = hair_mask * (1.0 - earring).clamp(0, 1)
            content = content * (1.0 - hair_to_subtract).clamp(0, 1)
        return content.clamp(0, 1)

    def _apply_source_content_gate(self, aux: dict[str, torch.Tensor]) -> None:
        """Clip face-detail sampling, then restore only verified earrings.

        The gate restricts source-texture sampling to real source content (skin,
        ear, earring) so an over-wide v58 ear ROI cannot copy source background,
        neck or no-hair gaps onto the target face.

        The broad query may contain earring search/recall pixels.  Those are only
        candidates and therefore still have to pass the source-content gate.  A
        parser-missed part of a long earring can legitimately lie on pixels that
        the parser calls background, though, so the already connected and
        target-safe ``earring_write_mask`` is added back after gating.  No search,
        recall, confidence or ROI mask is allowed to bypass the gate.
        """
        query_mask = aux.get("query_mask")
        if query_mask is None:
            return
        face_query = ensure_mask_4d(query_mask).float()
        size = face_query.shape[-2:]
        gate = self._build_source_content_gate(aux, size)
        if gate is None:
            return
        gate = self._mask_like(gate, face_query).clamp(0, 1)
        verified_earring = self._mask_like(
            aux.get("earring_write_mask"),
            face_query,
        ).clamp(0, 1)
        no_earring = aux.get("no_earring_case_mask")
        if no_earring is not None:
            verified_earring = verified_earring * (
                1.0 - self._mask_like(no_earring, face_query)
            ).clamp(0, 1)

        # Target owns the ear outline and the ear/background or ear/hair
        # boundary.  Ordinary source-detail recovery may contribute only to an
        # eroded target-ear interior, and is suppressed around a verified
        # earring so two branches cannot draw two ear contours.
        ear_boundary = self._mask_like(
            aux.get("target_ear_boundary_protect_mask"), face_query
        ).clamp(0, 1)
        ear_interior = self._mask_like(aux.get("target_ear_interior_mask"), face_query)
        target_ear = torch.clamp(
            self._mask_like(aux.get("target_left_ear_mask"), face_query)
            + self._mask_like(aux.get("target_right_ear_mask"), face_query),
            0,
            1,
        )
        earring_suppress = dilate_mask(verified_earring, 3)
        ear_detail_suppress = torch.clamp(ear_boundary + earring_suppress, 0, 1)
        ordinary_query = face_query * gate
        ordinary_query = ordinary_query * (
            (1.0 - target_ear).clamp(0, 1) + ear_interior
        ).clamp(0, 1)
        aux["ear_detail_restore_mask_before"] = ordinary_query
        ordinary_query = ordinary_query * (1.0 - ear_detail_suppress).clamp(0, 1)
        aux["ear_detail_restore_mask_after"] = ordinary_query
        aux["ear_detail_suppress_mask"] = ear_detail_suppress
        aux["source_content_gate_mask"] = gate
        aux["source_gated_face_query_mask"] = ordinary_query
        aux["query_mask"] = torch.maximum(
            aux["source_gated_face_query_mask"],
            verified_earring,
        )

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
            merge_aux_mask("source_earring_explicit_mask", gated_aux_mask(source_ear_mask, ear_roi))
        if source_earring_object_mask is not None:
            merge_aux_mask("source_earring_detection_mask", gated_aux_mask(source_earring_object_mask, ear_roi))
            merge_aux_mask("source_earring_mask", gated_aux_mask(source_earring_object_mask, earring_valid_roi))
            merge_aux_mask("source_earring_object_mask", gated_aux_mask(source_earring_object_mask, earring_valid_roi))
            merge_aux_mask("source_earring_explicit_mask", gated_aux_mask(source_earring_object_mask, ear_roi))
        if earring_confident_mask is not None:
            merge_aux_mask("source_earring_detection_mask", gated_aux_mask(earring_confident_mask, ear_roi))
            merge_aux_mask("source_earring_mask", gated_aux_mask(earring_confident_mask, earring_valid_roi))
            merge_aux_mask("earring_confident_mask", gated_aux_mask(earring_confident_mask, earring_valid_roi))
            merge_aux_mask("source_earring_explicit_mask", gated_aux_mask(earring_confident_mask, ear_roi))
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

        query_info = self._enhance_query_recall(source_01, source_parsing, query_info, target_01)
        query_info = self._finalize_earring_masks(query_info, source_parsing)

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

    @staticmethod
    def _attach_authoritative_hair_highres(
        aux: dict[str, torch.Tensor],
        authoritative_hair_highres: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> None:
        """Store an explicit high-resolution hair authority in RGB [0, 1]."""

        if authoritative_hair_highres is None:
            return
        if not torch.is_tensor(authoritative_hair_highres):
            raise TypeError("authoritative_hair_highres must be a torch.Tensor.")
        authority = authoritative_hair_highres
        if authority.ndim == 3:
            authority = authority.unsqueeze(0)
        if authority.ndim != 4 or authority.size(1) != 3:
            raise ValueError(
                "authoritative_hair_highres must have shape [B,3,H,W], "
                f"got {tuple(authority.shape)}."
            )
        if authority.size(0) != reference.size(0):
            raise ValueError(
                "authoritative_hair_highres batch size must match the PP input: "
                f"{authority.size(0)} != {reference.size(0)}."
            )
        authority = authority.to(device=reference.device, dtype=reference.dtype)
        if not bool(torch.isfinite(authority).all()):
            raise ValueError("authoritative_hair_highres contains non-finite values.")

        # This interface deliberately accepts the native StyleGAN range.  An
        # explicit conversion avoids the ambiguous data-dependent range guess
        # used by generic image helpers for an all-bright sample.
        aux["authoritative_hair_highres_01"] = ((authority + 1.0) * 0.5).clamp(0, 1)

    @staticmethod
    def _attach_authoritative_target_highres(
        aux: dict[str, torch.Tensor],
        authoritative_target_highres: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> None:
        """Store the pre-PP high-resolution target for normal-face authority.

        V5 only needs to generate the verified earring object and a narrowly
        defined revealed-skin repair.  Letting the PP decoder redraw the rest
        of the face is what produced the dark facial ghosts in validation.  The
        target is in StyleGAN's ``[-1, 1]`` range, just like the optional hair
        authority above.
        """

        if authoritative_target_highres is None:
            return
        if not torch.is_tensor(authoritative_target_highres):
            raise TypeError("authoritative_target_highres must be a torch.Tensor.")
        authority = authoritative_target_highres
        if authority.ndim == 3:
            authority = authority.unsqueeze(0)
        if authority.ndim != 4 or authority.size(1) != 3:
            raise ValueError(
                "authoritative_target_highres must have shape [B,3,H,W], "
                f"got {tuple(authority.shape)}."
            )
        if authority.size(0) != reference.size(0):
            raise ValueError(
                "authoritative_target_highres batch size must match the PP input: "
                f"{authority.size(0)} != {reference.size(0)}."
            )
        authority = authority.to(device=reference.device, dtype=reference.dtype)
        if not bool(torch.isfinite(authority).all()):
            raise ValueError("authoritative_target_highres contains non-finite values.")
        aux["authoritative_target_highres_01"] = ((authority + 1.0) * 0.5).clamp(0, 1)

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
        authoritative_hair_highres: torch.Tensor | None = None,
        authoritative_target_highres: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        source_ear_mask: torch.Tensor | None = None,
        source_earring_object_mask: torch.Tensor | None = None,
        earring_confident_mask: torch.Tensor | None = None,
        earring_supervision_mask: torch.Tensor | None = None,
        earring_highlight_mask: torch.Tensor | None = None,
        earring_reference: torch.Tensor | None = None,
        source_face_reference: torch.Tensor | None = None,
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
            self._attach_authoritative_hair_highres(
                aux,
                authoritative_hair_highres,
                source,
            )
            self._attach_authoritative_target_highres(
                aux,
                authoritative_target_highres,
                source,
            )
            aux["target_01"] = normalized_to_01(target)
            self._apply_source_content_gate(aux)
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
        self._attach_authoritative_hair_highres(
            aux,
            authoritative_hair_highres,
            source,
        )
        self._attach_authoritative_target_highres(
            aux,
            authoritative_target_highres,
            source,
        )
        # Restrict every source-sampling mask to real source content (skin/ear/
        # earring) so an over-wide v58 ear ROI cannot copy source background,
        # lighting or no-hair gaps back onto the target.  This runs after mask
        # resolution and before any source texture is read.
        self._apply_source_content_gate(aux)

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
        reference_01 = source_01 if earring_reference is None else normalized_to_01(earring_reference)
        aux["earring_reference"] = reference_01
        aux["earring_reference_01"] = reference_01
        face_reference_01 = source_01 if source_face_reference is None else normalized_to_01(source_face_reference)
        aux["source_face_reference_01"] = face_reference_01
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

        # HM_X_repaired is a topology mask.  Nearest-neighbour expansion keeps
        # narrow repaired crown bridges binary instead of turning them into a
        # soft strip that a later erosion can erase.
        hair_raw = F.interpolate(
            ensure_mask_4d(target_hair).float(),
            size=size,
            mode="nearest",
        )
        hair_raw = (hair_raw > 0.5).to(dtype=hair_raw.dtype)
        hair_context = hair_raw
        hair_dilate = int(getattr(self.args, "output_target_hair_preserve_dilate", 5))
        if hair_dilate > 0:
            hair_context = dilate_mask(hair_raw, hair_dilate)

        # Preserve every parser-confirmed target-hair pixel.  Only the dilated
        # context is clipped away from the face, so light hair at the hairline
        # cannot be brightened or recolored by PP.
        target_face = aux.get("target_face_surface_mask")
        face_surface = None
        if target_face is not None:
            face_surface = (resize_mask(target_face, size) > 0.5).to(hair_raw.dtype)
            face_exclude = face_surface
            seam_dilate = int(getattr(self.args, "output_face_hair_seam_preserve_dilate", 7))
            if seam_dilate > 0:
                face_exclude = dilate_mask(face_exclude, seam_dilate)
            hair_context = hair_context * (1.0 - face_exclude).clamp(0, 1)
        preserve = torch.maximum(hair_raw, hair_context).clamp(0, 1)

        # Soften only the true face/hair seam.  Eroding ``preserve`` globally
        # used to weaken the crown and outer silhouette as well; a narrow crown
        # repair could disappear entirely with the default 9px setting.
        face_hair_seam = torch.zeros_like(hair_raw)
        seam_feather = int(getattr(self.args, "output_hairline_feather", 0))
        if seam_feather > 1 and face_surface is not None:
            face_hair_seam = (
                hair_raw * dilate_mask(face_surface, seam_feather)
            ).clamp(0, 1)
            core = erode_mask(hair_raw, seam_feather)
            ramp = gaussian_blur(
                core,
                kernel_size=seam_feather,
                sigma=max(1.0, seam_feather / 3.0),
            ).clamp(0, 1)
            seam_alpha = torch.maximum(core, torch.minimum(hair_raw, ramp)).clamp(0, 1)
            preserve = (
                preserve * (1.0 - face_hair_seam)
                + torch.minimum(preserve, seam_alpha) * face_hair_seam
            ).clamp(0, 1)

        target_hair_ear_bridge = aux.get("target_hair_ear_bridge_mask")
        if target_hair_ear_bridge is not None:
            preserve = torch.maximum(
                preserve,
                resize_mask(target_hair_ear_bridge, size),
            ).clamp(0, 1)

        # Only the connected, source-safe write mask may override target-hair
        # preservation.  Search/confidence/ROI masks are intentionally ignored:
        # they can be broad and therefore cannot be treated as write permission.
        earring_write = aux.get("earring_write_mask")
        if earring_write is None:
            earring_keep = torch.zeros_like(preserve)
        else:
            earring_keep = resize_mask(earring_write, size)
        no_earring = aux.get("no_earring_case_mask")
        if no_earring is not None:
            earring_keep = earring_keep * (1.0 - resize_mask(no_earring, size)).clamp(0, 1)
        hoop_hole = aux.get("hoop_hole_mask")
        if hoop_hole is not None:
            earring_keep = earring_keep * (1.0 - resize_mask(hoop_hole, size)).clamp(0, 1)
        keep_dilate = int(getattr(self.args, "output_earring_keep_dilate", 0))
        if keep_dilate > 0 and earring_write is not None:
            # A cosmetic dilation may smooth values *inside* the write gate, but
            # can never expand the gate onto another target-hair pixel.
            earring_keep = torch.minimum(
                dilate_mask(earring_keep, keep_dilate),
                resize_mask(earring_write, size),
            )

        # This is the only mask allowed to open the hard target-hair authority.
        # Its own soft edge supplies the earring transition; globally blurring
        # ``preserve`` would also weaken unrelated crown/outer-contour pixels.
        preserve = preserve * (1.0 - earring_keep).clamp(0, 1)
        if hoop_hole is not None:
            # A topology hole is target-owned even outside target hair.  This
            # final RGB authority is the second independent safeguard after the
            # feature-injection mask exclusion.
            preserve = torch.maximum(preserve, resize_mask(hoop_hole, size)).clamp(0, 1)
        aux["output_target_hair_face_seam_mask"] = face_hair_seam
        aux["output_target_hair_preserve_mask"] = preserve
        aux["output_target_hair_earring_keep_mask"] = earring_keep
        return preserve

    def _normal_target_face_preserve_mask(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        """Keep normal face pixels owned by the pre-PP target image.

        Only two V5 paths are allowed to alter facial pixels: the exact
        revealed-skin region (including its soft transition) and an accepted
        earring object.  This makes the decoder unable to repaint cheeks, eyes
        or the whole forehead with source-like low-frequency artefacts while
        retaining the intended local repairs.
        """

        target_face = aux.get("target_face_surface_mask")
        target_parsing = aux.get("target_parsing")
        if target_face is None and target_parsing is None:
            return None
        if target_face is None:
            preserve = torch.zeros_like(
                resize_mask(parsing_label_mask(target_parsing, RAW_DETAIL_LABELS), size)
            )
        else:
            preserve = (resize_mask(target_face, size) > 0.5).float()

        # ``face_surface`` contains skin/neck labels only.  Eyes, brows, nose
        # and mouth are separate parser classes, so protecting only skin still
        # leaves the most noticeable facial features free to be corrupted by
        # the PP decoder.  Target ears are also retained until a verified
        # earring write explicitly opens its small local edit region.
        if target_parsing is not None:
            detail = resize_mask(
                parsing_label_mask(target_parsing, RAW_DETAIL_LABELS),
                size,
            )
            preserve = torch.maximum(preserve, detail)
        for key in ("target_left_ear_mask", "target_right_ear_mask"):
            ear = aux.get(key)
            if ear is not None:
                preserve = torch.maximum(preserve, resize_mask(ear, size))
        preserve = (preserve > 0.5).float()

        target_hair = aux.get("target_hair_mask")
        if target_hair is not None:
            preserve = preserve * (1.0 - resize_mask(target_hair, size)).clamp(0, 1)

        revealed = aux.get("revealed_skin_blend_mask", aux.get("revealed_skin_mask"))
        if revealed is not None:
            revealed = resize_mask(revealed, size).clamp(0, 1)
            revealed_dilate = int(
                getattr(self.args, "output_revealed_skin_preserve_dilate", 5)
            )
            if revealed_dilate > 0:
                revealed = dilate_mask(revealed, revealed_dilate)
            preserve = preserve * (1.0 - revealed).clamp(0, 1)

        # The earring itself remains a learned local edit.  A tiny dilation
        # prevents a hard target/earring seam, but no broad ear ROI or search
        # proposal is allowed to open the normal-face authority.
        earring_write = aux.get("earring_write_mask")
        if earring_write is not None:
            earring_edit = resize_mask(earring_write, size).clamp(0, 1)
            edit_dilate = int(getattr(self.args, "output_earring_face_exclude_dilate", 2))
            if edit_dilate > 0:
                earring_edit = dilate_mask(earring_edit, edit_dilate)
            preserve = preserve * (1.0 - earring_edit).clamp(0, 1)

        hoop_hole = aux.get("hoop_hole_mask")
        if hoop_hole is not None:
            # A hoop centre is target owned even when it happens to overlap the
            # parser's face surface near an ear.
            preserve = torch.maximum(preserve, resize_mask(hoop_hole, size)).clamp(0, 1)

        aux["output_normal_face_preserve_mask"] = preserve
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

        # Semantic safety masks.  Missing parser output falls back to the
        # revealed neighbourhood rather than disabling harmonisation entirely.
        target_face = aux.get("target_face_surface_mask")
        if target_face is None and aux.get("target_parsing") is not None:
            target_face = parsing_label_mask(aux["target_parsing"], RAW_FACE_SURFACE_LABELS)
        if target_face is None:
            target_face = dilate_mask((revealed > 0.05).float(), 65)
        target_face = resize_mask(target_face, size).clamp(0, 1)
        target_hair = self._mask_like(aux.get("target_hair_mask"), revealed)

        target_parsing = aux.get("target_parsing")
        target_detail = (
            torch.zeros_like(revealed)
            if target_parsing is None
            else resize_mask(parsing_label_mask(target_parsing, RAW_DETAIL_LABELS), size)
        )
        earring_exclude = torch.zeros_like(revealed)
        for key in (
            "earring_write_mask",
            "earring_confident_mask",
            "source_earring_mask",
            "visible_ear_roi",
        ):
            value = aux.get(key)
            if value is not None:
                earring_exclude = torch.clamp(
                    earring_exclude + resize_mask(value, size), 0, 1
                )

        generated_safe = (
            target_face
            * (1.0 - dilate_mask(revealed, 5)).clamp(0, 1)
            * (1.0 - dilate_mask(target_hair, 3)).clamp(0, 1)
            * (1.0 - dilate_mask(target_detail, 3)).clamp(0, 1)
            * (1.0 - dilate_mask(earring_exclude, 5)).clamp(0, 1)
        ).clamp(0, 1)
        generated_lower = build_lower_face_skin_reference_mask(
            target_face,
            revealed,
            hair_mask=target_hair,
            detail_mask=target_detail,
            earring_mask=earring_exclude,
        )

        # The source fringe-hidden area is not valid skin ground truth.  The
        # only fixed diffusion anchors are already-normal target/PP skin around
        # the revealed region, with lower-face skin as a local fallback.
        local_skin_anchor = generated_safe * dilate_mask((revealed > 0.05).float(), 65)
        lower_face_anchor = generated_lower

        area_scale = float(size[0] * size[1]) / float(256 * 256)
        minimum_reference = float(
            getattr(self.args, "revealed_skin_min_reference_area", 96.0)
        ) * area_scale

        def enough(mask: torch.Tensor) -> torch.Tensor:
            return (mask.flatten(1).sum(dim=1) >= minimum_reference).view(-1, 1, 1, 1)

        reference_mask = torch.where(
            enough(local_skin_anchor), local_skin_anchor, lower_face_anchor
        )
        reference_mask = torch.where(
            enough(reference_mask), reference_mask, generated_safe
        ).clamp(0, 1)

        # Final non-empty safety net: use the lower portion of the semantic face
        # (or image) instead of producing a zero/black tone reference.
        height, width = size
        y = torch.arange(height, device=image_01.device).view(1, 1, height, 1)
        geometric_lower = (y >= int(0.55 * height)).to(image_01.dtype)
        geometric_lower = geometric_lower.expand(image_01.size(0), 1, height, width)
        geometric_lower = geometric_lower * torch.maximum(target_face, generated_safe)
        reference_mask = torch.where(enough(reference_mask), reference_mask, geometric_lower).clamp(0, 1)

        work = 256
        img_small = F.interpolate(image_01, size=(work, work), mode="bilinear", align_corners=False)
        ref_small = F.interpolate(
            reference_mask, size=(work, work), mode="bilinear", align_corners=False
        ).clamp(0, 1)

        blur_k = max(3, int(getattr(self.args, "revealed_skin_tone_kernel", 15)))
        if blur_k % 2 == 0:
            blur_k += 1
        blur_s = float(getattr(self.args, "revealed_skin_tone_sigma", 7.0))
        iters = int(getattr(self.args, "revealed_skin_diffuse_iters", 24))

        # Complete a low-frequency Lab field from fixed neighbouring target/PP
        # skin.  No source RGB, bangs, or source texture enters this field.
        lab_small = rgb_to_lab(img_small)
        current_low_lab = low_pass_filter(lab_small, kernel_size=blur_k, sigma=blur_s)
        skin_field_lab = self._diffuse_fill(
            current_low_lab, ref_small, iters, blur_k, blur_s
        )
        correction_lab = low_pass_filter(
            skin_field_lab - current_low_lab, kernel_size=blur_k, sigma=blur_s
        )
        limit = float(getattr(self.args, "revealed_skin_tone_limit", 0.28))
        l_limit = max(4.0, min(22.0, limit * 70.0))
        ab_limit = max(3.0, min(16.0, limit * 45.0))
        correction_lab = torch.cat(
            [
                correction_lab[:, :1].clamp(-l_limit, l_limit),
                correction_lab[:, 1:].clamp(-ab_limit, ab_limit),
            ],
            dim=1,
        )

        # Build a soft seam band entirely on safe target skin.  Tone crosses the
        # band; source detail never does, which removes black/white/grey rings and
        # half-transparent brush marks without reintroducing old bangs.
        core = (revealed > 0.05).float()
        seam_width = max(3, int(getattr(self.args, "revealed_skin_seam_band", 7)))
        seam_band = (
            dilate_mask(core, seam_width) - erode_mask(core, max(3, seam_width // 2))
        ).clamp(0, 1)
        seam_band = (
            seam_band
            * target_face
            * (1.0 - target_hair).clamp(0, 1)
            * (1.0 - target_detail).clamp(0, 1)
            * (1.0 - earring_exclude).clamp(0, 1)
        ).clamp(0, 1)
        blend_mask = torch.maximum(revealed, 0.85 * seam_band).clamp(0, 1)
        blend_mask = gaussian_blur(
            blend_mask,
            kernel_size=max(5, 2 * seam_width + 1),
            sigma=max(1.0, seam_width / 2.5),
        ).clamp(0, 1) * target_face
        blend_mask = blend_mask * (1.0 - target_hair).clamp(0, 1)

        strength = max(
            0.0,
            min(1.0, float(getattr(self.args, "revealed_skin_harmonize_strength", 0.9))),
        )
        # This is deliberately *not* a whole-face correction.  The write gate
        # is the revealed skin plus its narrow seam only; normal lower-face
        # texture/color stays exactly as emitted by PP.
        write_mask = blend_mask * torch.clamp(core + seam_band, 0, 1)
        tone_blend = (write_mask * strength).clamp(0, 1)
        write_small = F.interpolate(tone_blend, size=(work, work), mode="bilinear", align_corners=False)
        completed_lab = lab_small + correction_lab * write_small
        completed_rgb = lab_to_rgb(completed_lab)
        target_low = low_pass_filter(img_small, kernel_size=blur_k, sigma=blur_s)
        completed_low = low_pass_filter(completed_rgb, kernel_size=blur_k, sigma=blur_s)
        # Preserve target/SATD high-frequency texture.  A bounded gain only
        # restores local contrast; it never imports source bang texture.
        gain = max(0.9, min(1.15, float(getattr(self.args, "revealed_skin_detail_gain", 1.0))))
        target_high = img_small - target_low
        completed_rgb = (completed_low + gain * target_high).clamp(0, 1)
        result_small = img_small * (1.0 - write_small) + completed_rgb * write_small
        result = F.interpolate(result_small, size=size, mode="bilinear", align_corners=False).clamp(0, 1)

        # Keep the generator's own micro-texture.  Optional gain >1 is confined
        # to an eroded safe core and never samples source bangs/dark streaks.
        detail_safe = (
            erode_mask(core, 5)
            * target_face
            * (1.0 - dilate_mask(target_hair, 5)).clamp(0, 1)
            * (1.0 - dilate_mask(target_detail, 3)).clamp(0, 1)
            * (1.0 - dilate_mask(earring_exclude, 5)).clamp(0, 1)
        ).clamp(0, 1)
        aux["lower_face_skin_reference_mask"] = generated_lower
        aux["local_skin_anchor_mask"] = local_skin_anchor
        aux["lower_face_anchor_mask"] = lower_face_anchor
        aux["revealed_skin_seam_mask"] = seam_band
        aux["revealed_skin_blend_band"] = blend_mask
        aux["revealed_skin_blend_mask"] = blend_mask
        aux["revealed_skin_harmonize_mask"] = tone_blend
        aux["face_harmonize_write_mask"] = tone_blend
        aux["revealed_skin_detail_mask"] = detail_safe
        aux["skin_field_L"] = F.interpolate(
            skin_field_lab[:, :1] / 100.0, size=size, mode="bilinear", align_corners=False
        ).clamp(0, 1)
        aux["skin_field_ab"] = F.interpolate(
            torch.linalg.vector_norm(skin_field_lab[:, 1:], dim=1, keepdim=True) / 180.0,
            size=size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        aux["revealed_skin_tone_reference"] = aux["skin_field_L"]
        return result

    def _preserve_target_output(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if aux is None:
            return image

        image_01 = ((image + 1) / 2).clamp(0, 1)
        if not bool(getattr(self.args, "enable_output_target_preserve", True)):
            return image_01 * 2 - 1

        def resize_rgb(value: torch.Tensor | None, *, mode: str) -> torch.Tensor | None:
            if value is None:
                return None
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4 or value.size(1) != 3 or value.size(0) != image_01.size(0):
                raise ValueError("V5 RGB authority must have shape [B,3,H,W].")
            value = value.to(device=image_01.device, dtype=image_01.dtype)
            if value.shape[-2:] != image_01.shape[-2:]:
                value = F.interpolate(
                    value,
                    size=image_01.shape[-2:],
                    mode=mode,
                    align_corners=False,
                )
            return value.clamp(0, 1)

        def resize_earring_mask(value: torch.Tensor | None) -> torch.Tensor:
            """Upscale an accepted wire without turning it into a faint halo.

            The earring write mask is an object mask, not a feathered face
            blend.  Bilinear upsampling made a one-pixel 256px hoop into a
            mostly transparent grey trace at output resolution.  Nearest
            upsampling preserves exactly the verified pixels; it does not grow
            the object into its hollow centre or surrounding background.
            """
            if value is None:
                return torch.zeros_like(image_01[:, :1])
            value = ensure_mask_4d(value).to(device=image_01.device, dtype=image_01.dtype)
            value = value[:, :1]
            if value.shape[-2:] != image_01.shape[-2:]:
                value = F.interpolate(value, size=image_01.shape[-2:], mode="nearest")
            return (value > 0.5).to(dtype=image_01.dtype)

        # V5 is a local earring recovery stage, not a second face generator.
        # Compositing from the pre-PP target makes every non-earring pixel
        # authoritative: no decoder-produced forehead halo, eye distortion or
        # three-tone face can survive at the final output.
        target_authority = resize_rgb(
            aux.get("authoritative_target_highres_01", aux.get("target_01")),
            mode="bilinear",
        )
        if target_authority is None:
            return image_01 * 2 - 1

        earring_reference = resize_rgb(
            aux.get("earring_reference_01", aux.get("source_01")),
            mode="bilinear",
        )

        # Compute the exact earring write region before composing the protected
        # target.  Every other semantic face pixel is restored below after the
        # high-resolution hair hand-off, which prevents that hand-off from
        # bleeding colour into a parser-boundary forehead or cheek pixel.
        # The learned 256px write mask is useful for PP features but is not a
        # reliable final RGB object mask: a one-pixel parser miss can either
        # erase a real stud or expand into the source background.  The final
        # composite therefore has one authority only, the source-native object
        # locator below.  It returns actual source pixels and a separate hoop
        # hole; it never creates an ellipse or copies an ear/background crop.
        earring_edit = torch.zeros_like(image_01[:, :1])
        hoop_hole = torch.zeros_like(earring_edit)
        highres_instance = torch.zeros_like(earring_edit)
        highres_instance_hole = torch.zeros_like(earring_edit)
        highres_geometry_trace = torch.zeros_like(earring_edit)
        highres_geometry_hole = torch.zeros_like(earring_edit)
        highres_geometry_footprint = torch.zeros_like(earring_edit)
        highres_locator_roi = torch.zeros_like(earring_edit)
        highres_locator_seed = torch.zeros_like(earring_edit)
        highres_locator_support = torch.zeros_like(earring_edit)
        highres_locator_ring_support = torch.zeros_like(earring_edit)
        highres_locator_parser = torch.zeros_like(earring_edit)
        highres_locator_presence_seed = torch.zeros_like(earring_edit)
        source_earring_presence_gate = torch.zeros_like(earring_edit)
        # The native-resolution earring compositor is intentionally an
        # inference/validation operation.  Its masks are built from detached
        # source pixels and OpenCV, so running it during every training batch
        # cannot contribute gradients.  Keeping it out of the training
        # forward removes the large CPU synchronisation cost without changing
        # the learned PP path or final inference output.
        enable_highres_output = bool(
            getattr(self.args, "enable_highres_earring_output_refine", True)
        ) and not self.training
        if earring_reference is not None and enable_highres_output:
            # Ordinary earrings and hoops are intentionally separated here.
            # A strong/learned visual proposal is not a source instance: using
            # it as the ordinary-earring seed made grass, hair texture and an
            # exposed ear reappear as jewellery.  For a solid earring, parser
            # label 9 is the only direct RGB authority.  A parser-missed hoop
            # is handled separately below by a complete-ring verifier.
            parser_source_seed = resize_earring_mask(
                aux.get("source_parser_earring_mask")
            )
            no_earring = aux.get("no_earring_case_mask")
            direct_source_active = torch.ones_like(earring_edit)
            if no_earring is not None:
                direct_source_active = (
                    1.0 - resize_earring_mask(no_earring)
                ).clamp(0, 1)
            parser_seed_present = (
                parser_source_seed.flatten(1).sum(dim=1, keepdim=True) >= 1.0
            ).to(earring_edit.dtype).view(-1, 1, 1, 1)
            direct_source_active = direct_source_active * parser_seed_present
            source_instances = build_source_earring_instance_masks_v5(
                earring_reference,
                aux.get("source_parsing"),
                source_hair_mask=aux.get("source_hair_mask"),
                source_seed_mask=parser_source_seed,
            )
            left_active = resize_earring_mask(
                aux.get("left_target_side_open", aux.get("left_side_active"))
            )
            right_active = resize_earring_mask(
                aux.get("right_target_side_open", aux.get("right_side_active"))
            )
            # This is a ring-only verifier.  It returns a geometric boundary
            # and a separate interior, but neither synthetic geometry mask is
            # composited directly.  The boundary only unlocks nearby source
            # pixels already accepted by the visual locator, while the
            # interior remains target-owned.
            source_parsing = aux.get("source_parsing")
            source_ear = (
                parsing_label_mask(source_parsing, RAW_EAR_SURFACE_LABELS)
                if source_parsing is not None
                else None
            )
            highres_geometry = refine_earring_hoops_highres(
                earring_reference,
                source_instances["locator_roi"],
                source_instances["left_lobe_anchor"],
                source_instances["right_lobe_anchor"],
                source_hair_mask=aux.get("source_hair_mask"),
                source_ear_mask=source_ear,
                detection_size=max(earring_reference.shape[-2:]),
                min_axis=4.0,
                min_coverage=0.45,
            )
            # A hoop can be parser-missed, but it must pass the complete
            # lobe-connected, multi-sector geometry check in
            # ``refine_earring_hoops_highres`` before it gets any RGB write
            # permission.  Unlike an ordinary candidate, this path cannot be
            # activated by a small local texture fragment.
            left_geometry_trace = highres_geometry["left_elliptical_hoop"] * left_active
            right_geometry_trace = highres_geometry["right_elliptical_hoop"] * right_active
            left_geometry_hole = highres_geometry["left_elliptical_hoop_hole"] * left_active
            right_geometry_hole = highres_geometry["right_elliptical_hoop_hole"] * right_active
            highres_geometry_trace = torch.clamp(left_geometry_trace + right_geometry_trace, 0, 1)
            highres_geometry_hole = torch.clamp(left_geometry_hole + right_geometry_hole, 0, 1)
            # ``refine_earring_hoops_highres`` already returns only source
            # Canny-supported pixels lying on a lobe-connected ring.  A second
            # intersection with the sparse general visual support discarded
            # most of a real hoop (often leaving only one short arc).  Use the
            # verified source trace directly; it is not a rendered ellipse.
            left_geometry_observed = left_geometry_trace
            right_geometry_observed = right_geometry_trace
            left_instance = source_instances["left_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * left_active * direct_source_active
            left_instance = torch.maximum(left_instance, left_geometry_observed)
            right_instance = source_instances["right_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * right_active * direct_source_active
            right_instance = torch.maximum(right_instance, right_geometry_observed)
            highres_instance = torch.clamp(left_instance + right_instance, 0, 1)
            highres_instance_hole = (
                source_instances["left_hoop_hole_mask"].to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                ) * left_active * direct_source_active
                + source_instances["right_hoop_hole_mask"].to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                ) * right_active * direct_source_active
            ).clamp(0, 1)
            highres_instance_hole = torch.maximum(
                highres_instance_hole,
                highres_geometry_hole,
            )
            source_earring_presence_gate = (
                (highres_instance.flatten(1).sum(dim=1, keepdim=True) >= 1.0)
                .to(earring_edit.dtype)
                .view(-1, 1, 1, 1)
            ).expand_as(earring_edit)
            highres_locator_roi = source_instances["locator_roi"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            highres_locator_seed = source_instances["locator_seed"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            highres_locator_support = source_instances["locator_support"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            highres_locator_ring_support = source_instances["locator_ring_support"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            highres_locator_parser = source_instances["locator_parser_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            highres_locator_presence_seed = source_instances["locator_presence_seed"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            hoop_hole = highres_instance_hole
            earring_edit = highres_instance * (1.0 - hoop_hole).clamp(0, 1)
        # A coarse mask may still contain source ear pixels, so it cannot write
        # into the target ear interior.  A high-resolution source instance is
        # different: retaining it here preserves the thin suspension wire from
        # lobe to hoop without reopening the surrounding ear/background patch.
        ear_interior = aux.get("target_ear_interior_mask")
        if ear_interior is not None:
            ear_interior = resize_earring_mask(ear_interior)
            earring_edit = (
                earring_edit * (1.0 - ear_interior).clamp(0, 1)
                + highres_instance * ear_interior
            ).clamp(0, 1)
            earring_edit = earring_edit * (1.0 - hoop_hole).clamp(0, 1)

        protected = target_authority
        hair_authority = resize_rgb(aux.get("authoritative_hair_highres_01"), mode="bicubic")
        hair_preserve = self._target_output_preserve_mask(aux, image_01.shape[-2:])
        if hair_authority is not None and hair_preserve is not None:
            protected = protected * (1.0 - hair_preserve) + hair_authority * hair_preserve

        # ``authoritative_hair_highres`` is intentionally allowed to replace
        # generated hair, but a hair-parser dilation can overlap the forehead,
        # cheeks or an exposed ear by a few pixels.  In V5 these semantic face
        # pixels have no PP task at all.  Give them back to the pre-PP target
        # here, leaving only the already verified earring object open.
        face_authority = torch.zeros_like(earring_edit)
        for key in (
            "target_face_surface_mask",
            "target_skin_surface_mask",
            "target_left_ear_mask",
            "target_right_ear_mask",
        ):
            value = aux.get(key)
            if value is not None:
                face_authority = torch.maximum(face_authority, self._mask_like(value, face_authority))
        target_parsing = aux.get("target_parsing")
        if target_parsing is not None:
            face_labels = parsing_label_mask(
                target_parsing,
                RAW_FACE_SURFACE_LABELS + RAW_SKIN_SURFACE_LABELS + RAW_DETAIL_LABELS,
            )
            face_authority = torch.maximum(face_authority, self._mask_like(face_labels, face_authority))
        face_authority = face_authority * (1.0 - earring_edit).clamp(0, 1)
        protected = protected * (1.0 - face_authority) + target_authority * face_authority

        # Preserve facial identity detail without copying the source's
        # low-frequency illumination.  Direct source RGB replacement made the
        # source/SATD boundary visible as a coloured tile whenever their skin
        # tones differed.  The transferred target owns colour and shading;
        # source contributes only high-frequency pores, eyes and facial detail.
        # Hair, ears and accessories remain explicitly target-owned.
        face_restore = torch.zeros_like(face_authority)
        if bool(getattr(self.args, "enable_direct_face_skin_restore", True)):
            source_reference = resize_rgb(aux.get("source_face_reference_01"), mode="bilinear")
            source_parsing = aux.get("source_parsing")
            target_parsing = aux.get("target_parsing")
            if source_reference is not None and source_parsing is not None and target_parsing is not None:
                source_face = self._mask_like(
                    parsing_label_mask(source_parsing, RAW_FACE_SURFACE_LABELS + RAW_DETAIL_LABELS),
                    face_authority,
                )
                target_face = self._mask_like(
                    parsing_label_mask(target_parsing, RAW_FACE_SURFACE_LABELS + RAW_DETAIL_LABELS),
                    face_authority,
                )
                source_hair = self._mask_like(aux.get("source_hair_mask"), face_authority)
                target_hair = self._mask_like(aux.get("target_hair_mask"), face_authority)
                source_hair = torch.maximum(
                    source_hair,
                    self._mask_like(parsing_label_mask(source_parsing, (RAW_HAIR,)), face_authority),
                )
                target_hair = torch.maximum(
                    target_hair,
                    self._mask_like(parsing_label_mask(target_parsing, (RAW_HAIR,)), face_authority),
                )
                source_earring = self._mask_like(
                    parsing_label_mask(source_parsing, (RAW_EARRING,)),
                    face_authority,
                )
                face_restore = (
                    source_face
                    * target_face
                    * (1.0 - dilate_mask(source_hair, 17)).clamp(0, 1)
                    * (1.0 - dilate_mask(target_hair, 9)).clamp(0, 1)
                    * (1.0 - dilate_mask(source_earring + earring_edit, 5)).clamp(0, 1)
                ).clamp(0, 1)
                face_restore = erode_mask(face_restore, 5).clamp(0, 1)
                # Keep the transition inside the semantic face and feather
                # only the texture contribution.  This cannot introduce a
                # different skin colour across an eyebrow/forehead boundary.
                face_alpha = gaussian_blur(face_restore, kernel_size=17, sigma=3.5)
                face_alpha = (face_alpha * target_face * face_authority).clamp(0, 1)
                source_low = low_pass_filter(source_reference, kernel_size=31, sigma=6.0)
                source_detail = source_reference - source_low
                detail_gain = max(
                    0.0,
                    min(1.0, float(getattr(self.args, "direct_face_detail_gain", 0.80))),
                )
                protected = (protected + detail_gain * source_detail * face_alpha).clamp(0, 1)
                face_restore = face_alpha

        if bool(getattr(self.args, "enable_direct_earring_restore", True)):
            if earring_reference is not None:
                protected = protected * (1.0 - earring_edit) + earring_reference * earring_edit
        else:
            protected = protected * (1.0 - earring_edit) + image_01 * earring_edit
        aux["output_face_target_authority_mask"] = face_authority
        aux["output_direct_face_skin_restore_mask"] = face_restore
        aux["output_source_earring_composite_mask"] = earring_edit
        aux["output_v5_earring_edit_mask"] = earring_edit
        aux["output_source_earring_presence_gate"] = source_earring_presence_gate
        aux["output_highres_earring_instance"] = highres_instance
        aux["output_highres_earring_hole"] = highres_instance_hole
        aux["output_highres_earring_geometry_seed"] = highres_geometry_trace
        aux["output_highres_earring_geometry_hole"] = highres_geometry_hole
        aux["output_highres_earring_geometry_footprint"] = highres_geometry_footprint
        aux["output_source_earring_locator_roi"] = highres_locator_roi
        aux["output_source_earring_locator_seed"] = highres_locator_seed
        aux["output_source_earring_locator_support"] = highres_locator_support
        aux["output_source_earring_locator_ring_support"] = highres_locator_ring_support
        aux["output_source_earring_locator_parser"] = highres_locator_parser
        aux["output_source_earring_locator_presence_seed"] = highres_locator_presence_seed
        aux["output_highres_earring_output_refine_enabled"] = torch.full_like(
            earring_edit,
            float(enable_highres_output),
        )
        # Compatibility debug aliases.  They now show the real instance, not
        # a synthetic ellipse, so existing visualisation scripts stay useful.
        aux["output_highres_hoop_trace"] = highres_instance
        aux["output_highres_hoop_hole"] = highres_instance_hole
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
