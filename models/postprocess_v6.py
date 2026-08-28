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
    RAW_LEFT_EAR,
    RAW_RIGHT_EAR,
    RAW_EARRING,
    RAW_FACE_SURFACE_LABELS,
    RAW_HAIR,
    RAW_HAT,
    RAW_NECK_SURFACE_LABELS,
    RAW_SKIN_SURFACE_LABELS,
    align_earring_reference_to_target,
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
    build_earlobe_anchor,
    refine_earring_instances_highres,
    refine_earring_hoops_highres,
    expand_valid_roi_by_completion,
    build_revealed_skin_mask,
    build_weak_earring_masks,
    compute_earring_hole_mask,
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
    shift_mask,
    shift_tensor_per_batch,
)
from models.earring_foreground_v6 import (
    EarringCoordinateSpace,
    EarringNativeInstanceV6,
    align_earring_instance_v6,
    composite_earring_v6,
    enforce_exclusive_earring_sides_v6,
    extract_source_native_earring_v6,
    retain_single_earring_group_v6,
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


class NativeSourceDetailConditioner(nn.Module):
    """Encode native source detail for one global PP decode.

    The baseline PP encoder always canonicalizes its input to 256px.  That is
    sufficient for identity and geometry, but it discards pores, freckles and
    small eye detail before the StyleGAN decode starts.  This branch keeps
    source-native samples as a feature condition at the 128px generator layer.
    It deliberately has no semantic face/hair mask: applying a source-bang
    mask here would again make the observed face and a newly exposed forehead
    decode from different conditioning fields.
    """

    def __init__(self, channels: int = 64, work_size: int = 384):
        super().__init__()
        self.work_size = max(128, int(work_size))
        self.work_size -= self.work_size % 2
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 48, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, source_native: torch.Tensor) -> torch.Tensor:
        source_native = normalized_to_01(source_native).clamp(0, 1).detach()
        target_size = (self.work_size, self.work_size)
        if source_native.shape[-2:] != target_size:
            source_native = F.interpolate(
                source_native,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        # Pixel-unshuffle preserves the native 2x2 sampling pattern as feature
        # channels instead of blurring it before the learned encoder sees it.
        return self.encoder(F.pixel_unshuffle(source_native, downscale_factor=2))


class PostProcessModelV6(nn.Module):
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
            earring_align_max_shift=getattr(self.args, "earring_align_max_shift", 16),
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
        native_detail_channels = max(
            16,
            int(getattr(self.args, "native_source_detail_channels", 64)),
        )
        self.native_source_detail_enabled = bool(
            getattr(self.args, "enable_native_source_detail", False)
        )
        self.native_source_detail_conditioner = NativeSourceDetailConditioner(
            channels=native_detail_channels,
            work_size=getattr(self.args, "native_source_detail_work_size", 384),
        )
        # The zero-initialized residual keeps a pretrained PP unchanged until
        # this new condition has received training signal.  Unlike a face mask,
        # its gate is all ones at the 128px StyleGAN feature layer.
        self.native_source_detail_injector_128 = HFDAGatedInjectionUnit(
            base_channels=getattr(self.args, "ear_inject_channels_128", 256),
            prior_channels=native_detail_channels,
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
                    "native_source_detail_conditioner",
                    "native_source_detail_injector_128",
                    "query_builder",
                    "parsing_helper",
                )
            )
        ]
        if relevant_missing:
            print(f"[PostProcessModelV6] Missing base keys: {len(relevant_missing)}")
            print(relevant_missing[:20])
        if result.unexpected_keys:
            print(f"[PostProcessModelV6] Unexpected checkpoint keys: {len(result.unexpected_keys)}")
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

    @staticmethod
    def _per_sample_flag(
        value: torch.Tensor,
        reference: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        """Normalize a scalar flag to ``[B, 1, 1, 1]``.

        ``earring_reference_is_dataset`` describes the coordinate system of an
        entire reference image.  Treating it as a spatial alpha would combine
        source-coordinate and target-coordinate RGB within the same sample,
        which can make the locator see a displaced or duplicate accessory.
        Keep the contract explicit and fail before a malformed batch reaches
        the high-resolution compositor.
        """

        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor.")

        batch_size = reference.size(0)
        flag = value.to(device=reference.device, dtype=reference.dtype)
        if flag.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    f"{name} is scalar but the PP input batch has {batch_size} samples."
                )
            flag = flag.reshape(1, 1, 1, 1)
        elif flag.ndim == 1:
            if flag.size(0) != batch_size:
                raise ValueError(
                    f"{name} batch size must match the PP input: "
                    f"{flag.size(0)} != {batch_size}."
                )
            flag = flag.view(batch_size, 1, 1, 1)
        elif flag.ndim == 2 and tuple(flag.shape) == (batch_size, 1):
            flag = flag.view(batch_size, 1, 1, 1)
        elif flag.ndim == 3 and tuple(flag.shape) == (batch_size, 1, 1):
            flag = flag.view(batch_size, 1, 1, 1)
        elif flag.ndim == 4 and tuple(flag.shape) == (batch_size, 1, 1, 1):
            pass
        else:
            raise ValueError(
                f"{name} must be one scalar per sample with shape [B], [B,1], "
                f"[B,1,1], or [B,1,1,1]; got {tuple(flag.shape)}."
            )
        if not bool(torch.isfinite(flag).all()):
            raise ValueError(f"{name} contains non-finite values.")
        return (flag > 0.5).to(dtype=reference.dtype)

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

    def _finalize_dataset_earring_masks(
        self,
        query_info: dict[str, torch.Tensor],
        source_parsing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Use generated instance labels as the full training authority.

        Fresh PP data already has a source-verified, target-aligned object
        alpha.  Re-running parser/GrabCut recall at training time replaces a
        negative example with a false positive and reduces a complete hoop to
        its ear-local arc.  This path intentionally preserves the saved object
        and its target-owned hole exactly.
        """

        reference = ensure_mask_4d(
            query_info.get("dataset_earring_base_query_mask", query_info["query_mask"])
        ).float()
        # Preserve the target-only exposure decision created by
        # ``EarAnchoredQueryBuilder`` before replacing the source object masks
        # with the generated dataset authority below.  ``left/right_present``
        # calculated in this method describe *source* earring components, not
        # whether the corresponding target earlobe is visible.  Conflating the
        # two made a parser-missed source accessory close an otherwise exposed
        # target side, so the final native compositor had no legal side on
        # which to restore it.
        target_left_open = self._mask_like(
            query_info.get("left_target_side_open", query_info.get("left_side_active")),
            reference,
        )
        target_right_open = self._mask_like(
            query_info.get("right_target_side_open", query_info.get("right_side_active")),
            reference,
        )
        instance = self._mask_like(query_info.get("dataset_earring_instance_mask"), reference)
        hole = self._mask_like(query_info.get("dataset_hoop_hole_mask"), reference)
        instance = instance * (1.0 - hole).clamp(0, 1)
        present = (instance.flatten(1).sum(dim=1, keepdim=True) >= 1.0).to(reference.dtype)
        active = present.view(-1, 1, 1, 1).expand_as(reference)
        no_earring = (1.0 - active).clamp(0, 1)

        left_roi = self._mask_like(query_info.get("left_ear_roi"), reference)
        right_roi = self._mask_like(query_info.get("right_ear_roi"), reference)
        left_anchor = self._mask_like(query_info.get("left_lobe_anchor"), reference)
        right_anchor = self._mask_like(query_info.get("right_lobe_anchor"), reference)
        left_instance, right_instance = assign_components_to_ear_sides(
            instance,
            left_roi,
            right_roi,
            left_anchor,
            right_anchor,
        )
        left_present = (left_instance.flatten(1).sum(dim=1, keepdim=True) >= 1.0).to(reference.dtype)
        right_present = (right_instance.flatten(1).sum(dim=1, keepdim=True) >= 1.0).to(reference.dtype)
        left_active = left_present.view(-1, 1, 1, 1).expand_as(reference)
        right_active = right_present.view(-1, 1, 1, 1).expand_as(reference)

        search_dilate = max(0, int(getattr(self.args, "earring_query_dilate", 3)))
        search_mask = instance if search_dilate <= 0 else dilate_mask(instance, search_dilate)
        search_mask = search_mask * active
        target_hair_occlusion = self._mask_like(
            query_info.get("target_ear_hair_occlusion_mask"), reference
        )
        parser = ensure_mask_4d(source_parsing).to(device=reference.device)
        if parser.shape[-2:] != reference.shape[-2:]:
            parser = F.interpolate(parser.float(), size=reference.shape[-2:], mode="nearest")
        parser_earring = (parser.long() == RAW_EARRING).to(dtype=reference.dtype)
        zero = torch.zeros_like(reference)

        query_info["query_mask"] = torch.clamp(reference + search_mask, 0, 1)
        query_info["source_parser_earring_mask"] = parser_earring
        query_info["no_earring_case_mask"] = no_earring
        query_info["no_earring_case"] = no_earring
        query_info["earring_search_mask"] = search_mask
        query_info["earring_write_mask"] = instance
        query_info["earring_visible_segment_mask"] = instance
        query_info["earring_object_mask"] = instance
        query_info["earring_filled_mask"] = torch.clamp(instance + hole, 0, 1)
        query_info["earring_core_mask"] = instance
        query_info["earring_completion_mask"] = zero
        query_info["hoop_hole_mask"] = hole
        query_info["source_earring_mask"] = instance
        query_info["source_earring_object_mask"] = instance
        query_info["source_earring_detection_mask"] = instance
        query_info["source_earring_explicit_mask"] = instance
        query_info["earring_confident_mask"] = instance
        query_info["earring_candidate_mask"] = zero
        query_info["online_earring_candidate_mask"] = zero
        query_info["online_earring_search_mask"] = zero
        query_info["earring_query_recall_mask"] = zero
        query_info["earring_object_recall_mask"] = zero
        query_info["earring_object_detection_mask"] = zero
        query_info["left_core_mask"] = left_instance
        query_info["right_core_mask"] = right_instance
        query_info["left_completion_mask"] = zero
        query_info["right_completion_mask"] = zero
        query_info["left_write_mask"] = left_instance
        query_info["right_write_mask"] = right_instance
        query_info["left_search_mask"] = search_mask * left_active
        query_info["right_search_mask"] = search_mask * right_active
        query_info["left_target_side_open"] = target_left_open
        query_info["right_target_side_open"] = target_right_open
        query_info["left_side_active"] = target_left_open * left_active
        query_info["right_side_active"] = target_right_open * right_active
        query_info["target_hair_ear_bridge_mask"] = (
            target_hair_occlusion * (1.0 - instance).clamp(0, 1)
        )
        query_info["earring_visibility_mask"] = self._mask_like(
            query_info.get("earring_valid_roi", query_info.get("visible_ear_roi")), reference
        )
        presence_target = query_info.get("presence_target")
        if presence_target is not None:
            query_info["presence_target"] = torch.cat(
                (left_present, right_present, torch.maximum(left_present, right_present)), dim=1
            )
        return query_info

    def _finalize_earring_masks(
        self,
        query_info: dict[str, torch.Tensor],
        source_parsing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Turn broad earring recall into a narrow, target-safe write mask."""

        reference = ensure_mask_4d(query_info["query_mask"]).float()
        dataset_authority = query_info.get("dataset_earring_mask_authority")
        if dataset_authority is not None:
            dataset_authority = ensure_mask_4d(dataset_authority).to(reference.device)
            if bool((dataset_authority > 0.5).flatten(1).all(dim=1).all().item()):
                return self._finalize_dataset_earring_masks(query_info, source_parsing)
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
        ).clamp(0, 1) * active
        # ``left/right_lobe_anchor`` was already built from the target ear
        # after ``target_hair_lobe_core`` removed an actually covered lobe.
        # Do not subtract ``target_hair_occlusion`` here a second time: that
        # mask spans the whole ear-side ROI, including long hair *below* an
        # exposed lobe.  Its dilation was zeroing the connectivity seed and
        # consequently rejecting the entire verified pendant whenever its
        # lower body would sit in front of transferred target hair.
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
        dataset_authority = query_info.get("dataset_earring_mask_authority")
        if dataset_authority is not None:
            dataset_authority = self._mask_like(dataset_authority, query_mask)
            if bool((dataset_authority > 0.5).flatten(1).all(dim=1).all().item()):
                # Fresh generated data contains a verified instance alpha for
                # every sample, including an explicit all-zero no-earring one.
                # Running Canny/GrabCut recall here is both slower and wrong:
                # it substitutes detector noise for the supervision target.
                return query_info
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
        """Return the unmodified author-baseline PP feature.

        The original PP has no source-hair, SATD, target-hair, or earring mask
        in this operation: it concatenates both complete encoder feature maps
        and decodes them once.  Any mask here splits the newly revealed skin
        from the observed face before StyleGAN has a chance to make one
        continuous result.
        """
        del target_mask, aux
        return self.to_feature(torch.cat((f_face, f_hair), dim=1))

    def lock_baseline_face_core_eval(self) -> None:
        """Keep frozen baseline modules out of train-mode running-stat updates."""
        self.encoder_face.eval()
        self.to_feature.eval()
        if self.use_mod:
            self.to_latent_1.eval()
            self.to_latent_2.eval()
        else:
            self.to_latent.eval()

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
        hoop_hole_mask: torch.Tensor | None,
        earring_mask_is_dataset: torch.Tensor | None,
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

        def gated_instance_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
            """Gate an explicit object per ear side without cropping its extent.

            Dataset instances are already source-verified.  Multiplying them
            pixelwise by the compact target ear ROI turns a complete hoop into
            a lobe-adjacent arc before the learned mask and losses can see it.
            The ROI therefore decides only whether a side is visible; component
            assignment retains every verified pixel on that accepted side.
            """

            if mask is None:
                return None
            instance = resize_mask(ensure_mask_4d(mask).float(), image_size).clamp(0, 1)
            # This is SOURCE_NATIVE object evidence.  Do not multiply it by a
            # target ear ROI here: when SATD rewrites the background beside an
            # exposed lobe, the low-resolution target ROI can be empty even
            # though the source object is valid.  Target visibility is applied
            # later as one scalar gate per side in the strict compositor.
            if visible_ear_roi is None or ear_roi is None:
                return instance
            left_roi = self._mask_like(query_info.get("left_ear_roi"), instance)
            right_roi = self._mask_like(query_info.get("right_ear_roi"), instance)
            left_anchor = self._mask_like(query_info.get("left_lobe_anchor"), instance)
            right_anchor = self._mask_like(query_info.get("right_lobe_anchor"), instance)
            left_instance, right_instance = assign_components_to_ear_sides(
                instance,
                left_roi,
                right_roi,
                left_anchor,
                right_anchor,
            )
            # Preserve both source sides.  The final target-side lobe gate is
            # computed from the completed transfer and decides whether either
            # side is actually written; applying a preliminary pixel/ROI gate
            # here was the reason SATD-region earrings disappeared entirely.
            return torch.clamp(left_instance + right_instance, 0, 1)

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
            instance_mask = gated_instance_mask(source_earring_object_mask)
            merge_aux_mask("source_earring_detection_mask", instance_mask)
            merge_aux_mask("source_earring_mask", instance_mask)
            merge_aux_mask("source_earring_object_mask", instance_mask)
            merge_aux_mask("source_earring_explicit_mask", instance_mask)
        if earring_confident_mask is not None:
            instance_mask = gated_instance_mask(earring_confident_mask)
            merge_aux_mask("source_earring_detection_mask", instance_mask)
            merge_aux_mask("source_earring_mask", instance_mask)
            merge_aux_mask("earring_confident_mask", instance_mask)
            merge_aux_mask("source_earring_explicit_mask", instance_mask)
        if earring_mask_is_dataset is not None and earring_confident_mask is not None:
            dataset_gate = earring_mask_is_dataset.to(
                device=source_01.device,
                dtype=source_01.dtype,
            )
            if dataset_gate.ndim == 1:
                dataset_gate = dataset_gate.view(-1, 1, 1, 1)
            else:
                dataset_gate = ensure_mask_4d(dataset_gate)
            dataset_gate = (dataset_gate > 0.5).to(source_01.dtype).expand_as(query_info["query_mask"])
            exact_instance = gated_instance_mask(earring_confident_mask)
            exact_hole = self._mask_like(hoop_hole_mask, exact_instance)
            exact_instance = exact_instance * (1.0 - exact_hole).clamp(0, 1)
            # Preserve the unmodified base query for the authoritative branch.
            # Online recall is skipped later, so no false candidate can enlarge
            # a saved zero-mask or overwrite a complete hoop with a short arc.
            query_info["dataset_earring_base_query_mask"] = query_info["query_mask"]
            query_info["dataset_earring_mask_authority"] = dataset_gate
            query_info["dataset_earring_instance_mask"] = exact_instance
            query_info["dataset_hoop_hole_mask"] = exact_hole
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

    @staticmethod
    def _attach_satd_background_highres(
        aux: dict[str, torch.Tensor],
        satd_background_highres: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> None:
        """Store SATD's candidate image for the final background residual only."""
        if satd_background_highres is None:
            return
        if not torch.is_tensor(satd_background_highres):
            raise TypeError("satd_background_highres must be a torch.Tensor.")
        candidate = satd_background_highres
        if candidate.ndim == 3:
            candidate = candidate.unsqueeze(0)
        if candidate.ndim != 4 or candidate.size(1) != 3:
            raise ValueError(
                "satd_background_highres must have shape [B,3,H,W], "
                f"got {tuple(candidate.shape)}."
            )
        if candidate.size(0) != reference.size(0):
            raise ValueError(
                "satd_background_highres batch size must match the PP input: "
                f"{candidate.size(0)} != {reference.size(0)}."
            )
        candidate = candidate.to(device=reference.device, dtype=reference.dtype)
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError("satd_background_highres contains non-finite values.")
        aux["satd_background_highres_01"] = ((candidate + 1.0) * 0.5).clamp(0, 1)

    @staticmethod
    def _attach_direct_satd_flag(
        aux: dict[str, torch.Tensor],
        direct_satd_pp_input,
        reference: torch.Tensor,
    ) -> None:
        """Record whether ``target`` is already the SATD PP input.

        In direct mode the SATD image is encoded by PP and must not be used a
        second time as a decoded-image residual candidate.  Keep this as a
        per-sample tensor so batches remain explicit and serialization/debug
        paths can inspect the decision.
        """
        if direct_satd_pp_input is None:
            return
        if torch.is_tensor(direct_satd_pp_input):
            flag = direct_satd_pp_input.to(device=reference.device, dtype=reference.dtype)
        else:
            flag = torch.full(
                (reference.size(0), 1, 1, 1),
                float(bool(direct_satd_pp_input)),
                device=reference.device,
                dtype=reference.dtype,
            )
        if flag.ndim == 0:
            flag = flag.view(1, 1, 1, 1).expand(reference.size(0), 1, 1, 1)
        elif flag.ndim == 1:
            flag = flag.view(-1, 1, 1, 1)
        elif flag.ndim == 2:
            flag = flag.view(flag.size(0), flag.size(1), 1, 1)
        if flag.size(0) == 1 and reference.size(0) != 1:
            flag = flag.expand(reference.size(0), *flag.shape[1:])
        if flag.size(0) != reference.size(0):
            raise ValueError(
                "direct_satd_pp_input batch size must match the PP input: "
                f"{flag.size(0)} != {reference.size(0)}."
            )
        aux["direct_satd_pp_input"] = (flag[:, :1] > 0.5).to(reference.dtype)

    def _attach_native_source_detail(
        self,
        aux: dict[str, torch.Tensor],
        source_face_reference: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> None:
        """Publish a global native-source feature for the 128px decoder layer."""
        if not self.native_source_detail_enabled or source_face_reference is None:
            return
        if not torch.is_tensor(source_face_reference):
            raise TypeError("source_face_reference must be a torch.Tensor.")
        native = source_face_reference
        if native.ndim == 3:
            native = native.unsqueeze(0)
        if native.ndim != 4 or native.size(1) != 3:
            raise ValueError(
                "source_face_reference must have shape [B,3,H,W], "
                f"got {tuple(native.shape)}."
            )
        if native.size(0) != reference.size(0):
            raise ValueError(
                "source_face_reference batch size must match the PP input: "
                f"{native.size(0)} != {reference.size(0)}."
            )
        native = native.to(device=reference.device, dtype=reference.dtype)
        if not bool(torch.isfinite(native).all()):
            raise ValueError("source_face_reference contains non-finite values.")
        aux["native_source_detail_feature"] = self.native_source_detail_conditioner(native)

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
        satd_background_highres: torch.Tensor | None = None,
        direct_satd_pp_input=None,
        query_mask: torch.Tensor | None = None,
        source_ear_mask: torch.Tensor | None = None,
        source_earring_object_mask: torch.Tensor | None = None,
        source_instance_verified_left: torch.Tensor | None = None,
        source_instance_verified_right: torch.Tensor | None = None,
        earring_confident_mask: torch.Tensor | None = None,
        hoop_hole_mask: torch.Tensor | None = None,
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
        if self.pretrain:
            s_face, [f_face] = self.encoder_face(source)
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
                hoop_hole_mask,
                earring_mask_is_dataset,
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
            if source_instance_verified_left is not None:
                aux["source_instance_verified_left"] = source_instance_verified_left
            if source_instance_verified_right is not None:
                aux["source_instance_verified_right"] = source_instance_verified_right
            self._attach_satd_background_highres(
                aux,
                satd_background_highres,
                source,
            )
            self._attach_direct_satd_flag(aux, direct_satd_pp_input, source)
            self._attach_native_source_detail(aux, source_face_reference, source)
            aux["target_01"] = normalized_to_01(target)
            self._apply_source_content_gate(aux)
            return self.latent_avg.to(s_face.device) + s_face, f_face, aux

        # V6 computes these only to retain its auxiliary ear-loss contract.
        # Its S/F are not used for the global image: that comes from a separate
        # no-suffix PostProcessModel instance in the caller.
        s_face, [f_face] = self.encoder_face(source)
        s_hair, [f_hair] = self.encoder_face(target)
        final_s = self._compute_latent(s_face, s_hair)
        base_feature = self._build_base_feature(f_face, f_hair, target_mask, aux=None)

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
            hoop_hole_mask,
            earring_mask_is_dataset,
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
        if source_instance_verified_left is not None:
            aux["source_instance_verified_left"] = source_instance_verified_left
        if source_instance_verified_right is not None:
            aux["source_instance_verified_right"] = source_instance_verified_right
        self._attach_satd_background_highres(
            aux,
            satd_background_highres,
            source,
        )
        self._attach_direct_satd_flag(aux, direct_satd_pp_input, source)
        self._attach_native_source_detail(aux, source_face_reference, source)
        # Restrict every source-sampling mask to real source content (skin/ear/
        # earring) so an over-wide v58 ear ROI cannot copy source background,
        # lighting or no-hair gaps back onto the target.  This runs after mask
        # resolution and before any source texture is read.
        self._apply_source_content_gate(aux)

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
        # The accessory heads may learn/query their own masks, but they must
        # never alter the global baseline PP feature.  Even a spatially gated
        # feature residual propagates through StyleGAN convolutions and changes
        # forehead, eye and cheek texture outside the ear ROI.
        final_f = base_feature
        fine_mask_64 = F.interpolate(
            mask_outputs["fine_mask"],
            size=base_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

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
        # The V5 native path has no coordinate-ambiguous earring field.  The parser seed is
        # source-canonical evidence; dataset labels/reference are target
        # aligned supervision only and are never eligible for native extraction.
        source_parser_for_seed = aux.get("source_parsing")
        aux["source_native_earring_seed"] = (
            parsing_label_mask(source_parser_for_seed, (RAW_EARRING,))
            if source_parser_for_seed is not None
            else torch.zeros_like(source_01[:, :1])
        )
        aux["source_native_earring_seed_space"] = EarringCoordinateSpace.SOURCE_CANONICAL
        aux["source_native_earring_alpha"] = None
        aux["source_native_earring_rgb"] = source_01
        dataset_aligned_alpha = aux.get("earring_confident_mask")
        aux["dataset_target_aligned_earring_alpha"] = (
            ensure_mask_4d(dataset_aligned_alpha)
            if dataset_aligned_alpha is not None
            else None
        )
        aux["dataset_target_aligned_earring_rgb"] = reference_01
        aux["target_aligned_earring_alpha"] = aux["dataset_target_aligned_earring_alpha"]
        aux["target_aligned_earring_rgb"] = reference_01
        # Generated PP samples store an object-only target-frame reference.
        # It is valid learned supervision, but not source-coordinate content.
        if earring_reference_is_dataset is None:
            aux["earring_reference_is_dataset"] = torch.zeros(
                source_01.size(0),
                1,
                1,
                1,
                device=source_01.device,
                dtype=source_01.dtype,
            )
        else:
            aux["earring_reference_is_dataset"] = self._per_sample_flag(
                earring_reference_is_dataset,
                source_01,
                name="earring_reference_is_dataset",
            )
        face_reference_01 = source_01 if source_face_reference is None else normalized_to_01(source_face_reference)
        aux["source_face_reference_01"] = face_reference_01
        aux["target_01"] = target_01
        aux["HT_E"] = ensure_mask_4d(HT_E).float() if HT_E is not None else None
        aux["baseline_face_core_locked"] = torch.ones_like(source_01[:, :1])
        return final_s, final_f, aux

    def _target_output_preserve_mask(
        self,
        aux: dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | None:
        target_hair = aux.get("target_hair_mask")
        if target_hair is None:
            return None

        # The V5 compositor publishes a high-resolution, anti-aliased hair
        # alpha after parsing the completed transfer.  Use that directly when
        # available.  Falling back to the old binary HM_X path is retained only
        # for legacy callers; its dilation was the source of the thick,
        # staircase-like face-side and background-side hair rim.
        published_hair = aux.get("output_v5_face_target_hair_soft_alpha")
        if published_hair is not None:
            hair_raw = resize_mask(published_hair, size).clamp(0, 1)
            hair_binary = (hair_raw > 0.5).to(hair_raw.dtype)
            hair_context = hair_raw
            hair_dilate = 0
        else:
            hair_raw = F.interpolate(
                ensure_mask_4d(target_hair).float(),
                size=size,
                mode="nearest",
            )
            hair_raw = (hair_raw > 0.5).to(dtype=hair_raw.dtype)
            hair_context = hair_raw
            hair_binary = hair_raw
            hair_dilate = int(getattr(self.args, "output_target_hair_preserve_dilate", 5))
            if hair_dilate > 0:
                hair_context = dilate_mask(hair_raw, hair_dilate)

        # Preserve every parser-confirmed target-hair pixel.  When the alpha was
        # published by the V6 compositor it came from the completed
        # high-resolution target itself.  The old 256px target-face mask is not
        # allowed to clip it: that mask can overlap a real right-side hair
        # strand, causing the strand to be replaced by the PP decoder and
        # exposing source clothing underneath.  Legacy callers still use the
        # clipped path below.
        target_face = aux.get("target_face_surface_mask")
        face_surface = None
        if target_face is not None:
            face_surface = (resize_mask(target_face, size) > 0.5).to(hair_raw.dtype)
            face_exclude = torch.zeros_like(face_surface) if published_hair is not None else face_surface
            seam_dilate = 0 if published_hair is not None else int(
                getattr(self.args, "output_face_hair_seam_preserve_dilate", 7)
            )
            if seam_dilate > 0:
                face_exclude = dilate_mask(face_exclude, seam_dilate)
            hair_context = hair_context * (1.0 - face_exclude).clamp(0, 1)
        preserve = (
            hair_context if published_hair is not None
            else torch.maximum(hair_binary, hair_context)
        ).clamp(0, 1)

        # Soften only the true face/hair seam.  Eroding ``preserve`` globally
        # used to weaken the crown and outer silhouette as well; a narrow crown
        # repair could disappear entirely with the default 9px setting.
        face_hair_seam = torch.zeros_like(hair_raw)
        seam_feather = int(getattr(self.args, "output_hairline_feather", 0))
        if seam_feather > 1 and face_surface is not None:
            face_hair_seam = (
                hair_binary * dilate_mask(face_surface, seam_feather)
            ).clamp(0, 1)
            core = erode_mask(hair_binary, seam_feather)
            ramp = gaussian_blur(
                core,
                kernel_size=seam_feather,
                sigma=max(1.0, seam_feather / 3.0),
            ).clamp(0, 1)
            seam_alpha = torch.maximum(core, torch.minimum(hair_binary, ramp)).clamp(0, 1)
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

        # ``earring_keep`` is a source-safe, lobe-connected foreground target,
        # not permission to paste source RGB.  The PP decoder is trained to
        # render this complete object, including a pendant in front of target
        # hair.  Keeping it open at evaluation is required: otherwise every
        # learned lower segment is overwritten by target-hair preservation and
        # only the much stricter native RGB fragment can remain.
        if self.training or bool(getattr(self.args, "enable_earring_pp_foreground", True)):
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

        # V19 has no normal-face target authority.  Keeping this compatibility
        # entry point as a no-op prevents a debug/external caller from
        # reintroducing the old split-face hard composite.
        return None

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

    @staticmethod
    def _diffuse_signed_face_field(
        value: torch.Tensor,
        known_mask: torch.Tensor,
        face_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Diffuse a low-frequency signed correction over one face surface.

        ``_diffuse_fill`` is deliberately bounded for RGB image repair.  The
        final PP join instead needs an RGB *difference* field, which is signed:
        at a hairline the decoded PP face must meet the completed transfer's
        skin tone without copying either image's high-frequency content across
        the semantic boundary.  Work at a small resolution because the field
        is explicitly low frequency and because this compositor also runs
        during V5 training previews.
        """
        native_size = tuple(value.shape[-2:])
        longest = max(native_size)
        work_longest = min(256, longest)
        work_size = (
            max(1, int(round(native_size[0] * work_longest / longest))),
            max(1, int(round(native_size[1] * work_longest / longest))),
        )

        def to_work(tensor: torch.Tensor, *, mode: str) -> torch.Tensor:
            if tuple(tensor.shape[-2:]) == work_size:
                return tensor
            if mode == "area":
                return F.interpolate(tensor, size=work_size, mode="area")
            return F.interpolate(tensor, size=work_size, mode="bilinear", align_corners=False)

        work_value = to_work(value, mode="bilinear")
        work_face = (to_work(face_mask, mode="area") > 0.10).to(work_value.dtype)
        work_known = (
            (to_work(known_mask, mode="area") > 0.10).to(work_value.dtype)
            * work_face
        )
        known_area = work_known.flatten(2).sum(dim=2, keepdim=True)
        known_mean = (
            (work_value * work_known).flatten(2).sum(dim=2, keepdim=True)
            / known_area.clamp_min(1.0)
        ).unsqueeze(-1)
        estimate = (
            work_value * work_known
            + known_mean * (work_face - work_known).clamp_min(0.0)
        )

        # Normalize every relaxation by valid face coverage.  Otherwise hair
        # or background RGB outside the face would bias the correction field
        # and recreate the very mask-coloured ring this field is meant to
        # remove.
        for _ in range(8):
            numerator = gaussian_blur(estimate * work_face, kernel_size=17, sigma=4.5)
            denominator = gaussian_blur(work_face, kernel_size=17, sigma=4.5).clamp_min(1e-5)
            relaxed = numerator / denominator
            estimate = (
                work_value * work_known
                + relaxed * (work_face - work_known).clamp_min(0.0)
            )

        estimate = estimate * (known_area > 1.0).to(estimate.dtype).view(-1, 1, 1, 1)
        if work_size != native_size:
            estimate = F.interpolate(estimate, size=native_size, mode="bilinear", align_corners=False)
        return estimate * face_mask

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
        corrected = F.interpolate(
            result_small,
            size=size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        # The correction field is estimated at 256px, but the output is not a
        # 256px image.  Returning the upsampled working image flattened every
        # untouched facial region into the third, smooth face state.  Blend
        # only the explicit revealed-skin/seam field back over the native
        # output so lower-face texture, pores and lighting stay continuous.
        result = image_01 * (1.0 - tone_blend) + corrected * tone_blend

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

    def _compose_final_v19_legacy_v19(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor],
        *,
        include_earring: bool = True,
    ) -> torch.Tensor:
        """Compose the sharp source-face path, optionally with its legacy earring pass."""
        image_01 = ((image + 1.0) * 0.5).clamp(0, 1)
        size = image_01.shape[-2:]

        def rgb(value, fallback=None):
            value = fallback if value is None else value
            if value is None:
                return None
            value = normalized_to_01(value)
            if value.shape[-2:] != size:
                value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            return value.to(device=image_01.device, dtype=image_01.dtype).clamp(0, 1)

        def mask(value, target_size=size):
            if value is None:
                return torch.zeros(image_01.size(0), 1, *target_size, device=image_01.device, dtype=image_01.dtype)
            value = ensure_mask_4d(value).to(device=image_01.device, dtype=image_01.dtype)
            if value.shape[-2:] != target_size:
                value = F.interpolate(value[:, :1], size=target_size, mode="nearest")
            return (value[:, :1] > 0.5).to(image_01.dtype)

        # V19 has one continuous face authority: the PP image.  The pre-PP
        # transfer remains authoritative only outside the semantic face and on
        # transferred hair/background.  Do not turn normal skin back into the
        # target while leaving revealed skin in the PP image: that was the
        # ownership split behind the white forehead / grey band / textured
        # lower-face failure.
        target_rgb = rgb(
            aux.get("authoritative_target_highres_01"),
            aux.get("target_01"),
        )
        hair_mask_soft = resize_mask(aux.get("target_hair_mask"), size).to(
            device=image_01.device,
            dtype=image_01.dtype,
        ) if aux.get("target_hair_mask") is not None else torch.zeros_like(image_01[:, :1])
        hair_mask = (hair_mask_soft > 0.5).to(image_01.dtype)
        target_parsing = aux.get("target_parsing")
        target_face = aux.get("target_face_surface_mask")
        if target_face is None and target_parsing is not None:
            target_face = parsing_label_mask(target_parsing, RAW_FACE_SURFACE_LABELS)
        if target_face is None:
            face_pp_alpha = torch.zeros_like(hair_mask)
        else:
            face_pp_alpha = resize_mask(target_face, size).to(
                device=image_01.device,
                dtype=image_01.dtype,
            )
            # Facial landmarks are part of the face result too.  Leaving eyes,
            # brows, nose and mouth in the pre-PP image while skin comes from
            # a different image is another form of split-face compositor.
            if target_parsing is not None:
                face_pp_alpha = torch.maximum(
                    face_pp_alpha,
                    resize_mask(
                        parsing_label_mask(target_parsing, RAW_DETAIL_LABELS),
                        size,
                    ).to(device=image_01.device, dtype=image_01.dtype),
                )
            # ``resize_mask`` supplies the only narrow antialiased edge here.
            # No face-wide blur, dilated hair guard, or revealed-region switch
            # is allowed to introduce a second skin state.
            face_pp_alpha = face_pp_alpha * (1.0 - hair_mask).clamp(0, 1)

        if target_rgb is None:
            base = image_01
        else:
            base = target_rgb * (1.0 - face_pp_alpha) + image_01 * face_pp_alpha
            # The explicit high-resolution repaired hair is more reliable than
            # a parser edge, but it never clips a confirmed earring later.
            hair_rgb = rgb(aux.get("authoritative_hair_highres_01"), target_rgb)
            if hair_rgb is not None:
                base = base * (1.0 - hair_mask) + hair_rgb * hair_mask
            if target_parsing is not None:
                target_background = mask((ensure_mask_4d(target_parsing).long() == 0).float())
                base = base * (1.0 - target_background) + target_rgb * target_background

        aux["output_v19_base"] = base
        aux["output_v19_face_pp_alpha"] = face_pp_alpha
        # Retain these names for existing V19 diagnostic consumers.  They now
        # describe the single full-strength PP face field, not a weak residual.
        aux["output_v19_face_residual_mask"] = face_pp_alpha
        aux["output_v19_face_residual_alpha"] = face_pp_alpha

        source_value = aux.get("source_full_01")
        source_01 = rgb(source_value, aux.get("source_face_reference_01", aux.get("source_01")))
        native = aux.get("source_full_01")
        if native is None:
            native = aux.get("source_face_reference_01", aux.get("source_01"))
        native = normalized_to_01(native) if native is not None else None
        source_low = aux.get("source_01")
        native_is_highres = (
            native is not None
            and source_low is not None
            and tuple(native.shape[-2:]) != tuple(source_low.shape[-2:])
        )
        if native is None:
            native = source_01
        if native is None or native.shape[0] != image_01.shape[0]:
            alpha = torch.zeros_like(image_01[:, :1])
            aux["output_source_earring_composite_mask"] = alpha
            aux["output_v19_unified_face"] = face_pp_alpha
            return base * 2.0 - 1.0

        native_size = tuple(native.shape[-2:])

        native = native.to(device=image_01.device, dtype=image_01.dtype)
        source_parsing = aux.get("source_parsing")
        if native_is_highres or source_parsing is None or tuple(ensure_mask_4d(source_parsing).shape[-2:]) != native_size:
            source_parsing_native = self.parsing_helper.parse(native, out_size=native_size)
        else:
            source_parsing_native = ensure_mask_4d(source_parsing).to(device=native.device).long()

        # V19 face ownership is intentionally source-first for pixels that are
        # truly visible in the source.  The previous high-frequency-only
        # residual left the SATD/PP low-frequency face underneath, so a real
        # source face became a smooth white forehead and a different lower
        # face.  Source hair, earrings and the immediately adjacent hair edge
        # remain excluded.  They are the only pixels for which source RGB is
        # not valid face content.
        source_skin_native = parsing_label_mask(
            source_parsing_native,
            RAW_FACE_SURFACE_LABELS,
        ).to(device=native.device, dtype=native.dtype)
        source_detail_native = parsing_label_mask(
            source_parsing_native,
            RAW_DETAIL_LABELS,
        ).to(device=native.device, dtype=native.dtype)
        source_hair_native = parsing_label_mask(
            source_parsing_native,
            (RAW_HAIR,),
        ).to(device=native.device, dtype=native.dtype)
        source_earring_native = parsing_label_mask(
            source_parsing_native,
            (RAW_EARRING,),
        ).to(device=native.device, dtype=native.dtype)
        native_scale = max(native_size) / 256.0
        hair_guard = dilate_mask(
            source_hair_native,
            max(1, int(round(3.0 * native_scale))),
        )
        earring_guard = dilate_mask(
            source_earring_native,
            max(1, int(round(2.0 * native_scale))),
        )
        face_window_native = resize_mask(face_pp_alpha, native_size).to(
            device=native.device,
            dtype=native.dtype,
        )
        source_skin_alpha_native = (
            source_skin_native
            * face_window_native
            * (1.0 - hair_guard).clamp(0, 1)
            * (1.0 - earring_guard).clamp(0, 1)
        ).clamp(0, 1)
        # Eyes, brows, nose and mouth carry the source identity too.  They use
        # the same native frame but are kept separate from the skin-tone field
        # below, which prevents a dark eyebrow or lip from tinting a revealed
        # forehead.
        source_detail_alpha_native = (
            source_detail_native
            * face_window_native
            * (1.0 - hair_guard).clamp(0, 1)
            * (1.0 - earring_guard).clamp(0, 1)
        ).clamp(0, 1)
        source_face_alpha_native = torch.maximum(
            source_skin_alpha_native,
            source_detail_alpha_native,
        )

        source_skin_alpha = resize_mask(source_skin_alpha_native, size).to(
            device=image_01.device,
            dtype=image_01.dtype,
        ).clamp(0, 1)
        source_detail_alpha = resize_mask(source_detail_alpha_native, size).to(
            device=image_01.device,
            dtype=image_01.dtype,
        ).clamp(0, 1)
        source_face_alpha = torch.maximum(source_skin_alpha, source_detail_alpha)
        source_rgb = rgb(native)

        # Source hair-hidden skin has no source RGB ground truth.  It remains
        # owned by the one PP decode.  Do not low-pass, diffuse, or tone-fill it
        # from visible skin: that creates a third, airbrushed facial state.
        missing_skin = (
            resize_mask(target_face, size).to(device=image_01.device, dtype=image_01.dtype)
            if target_face is not None
            else torch.zeros_like(face_pp_alpha)
        )
        missing_skin = missing_skin * (1.0 - source_face_alpha).clamp(0, 1)
        missing_skin = missing_skin * (1.0 - hair_mask).clamp(0, 1)
        tone_correction = torch.zeros_like(base)

        # Copy the complete observed source face, not merely a high-frequency
        # residual.  The PP result remains only under source bangs/occlusions,
        # where source face RGB would be invalid.  This order gives one
        # continuous skin field instead of the old white/grey/textured bands.
        if source_rgb is not None:
            base = source_rgb * source_face_alpha + base * (1.0 - source_face_alpha)
        aux["output_v19_base"] = base
        aux["output_v19_source_face_alpha"] = source_face_alpha
        aux["output_v19_source_skin_alpha"] = source_skin_alpha
        aux["output_v19_source_detail_alpha"] = source_detail_alpha
        aux["output_v19_revealed_skin_alpha"] = missing_skin
        aux["output_v19_face_tone_correction"] = tone_correction

        # The sharp face path and the old V19 earring recovery have separate
        # responsibilities.  The latter was replaced because its permissive
        # object extraction could recall source background or a covered lobe.
        # Returning here preserves the exact high-detail face result while the
        # caller applies the current strict earring compositor afterwards.
        if not include_earring:
            aux["output_v19_unified_face"] = face_pp_alpha
            aux["output_face_target_authority_mask"] = torch.zeros_like(source_face_alpha)
            aux["output_direct_face_skin_restore_mask"] = source_face_alpha
            return base.clamp(0, 1) * 2.0 - 1.0

        # Ear visibility is derived solely from the completed high-resolution
        # transfer.  A 256px query flag must never reopen a lobe that the final
        # hairstyle covers.
        target_for_geometry = target_rgb if target_rgb is not None else image_01
        target_parsing_output = self.parsing_helper.parse(target_for_geometry, out_size=size)
        target_hair_output = parsing_label_mask(target_parsing_output, (RAW_HAIR,))
        if native_size == size:
            target_parsing_native = target_parsing_output
        else:
            target_native = F.interpolate(
                target_for_geometry,
                size=native_size,
                mode="bilinear",
                align_corners=False,
            )
            target_parsing_native = self.parsing_helper.parse(target_native, out_size=native_size)
        target_hair_native = parsing_label_mask(target_parsing_native, (RAW_HAIR,))
        # The final parser can miss a thin transferred strand or classify it
        # as background/skin.  The target transfer mask is the authoritative
        # hairstyle geometry already used by the blending stage; union it
        # here before deciding whether an ear is exposed.  This closes a
        # fully-covered ear even when the high-resolution parser leaves a few
        # false ear pixels at the hair boundary.
        target_hair_hint_native = aux.get("target_hair_mask")
        if target_hair_hint_native is not None:
            target_hair_hint_native = resize_mask(
                target_hair_hint_native,
                native_size,
            ).to(device=native.device, dtype=native.dtype)
            target_hair_native = torch.maximum(
                target_hair_native,
                (target_hair_hint_native > 0.25).to(native.dtype),
            ).clamp(0, 1)
        target_left_ear = (
            parsing_label_mask(target_parsing_native, (7,))
            * (1.0 - target_hair_native).clamp(0, 1)
        )
        target_right_ear = (
            parsing_label_mask(target_parsing_native, (8,))
            * (1.0 - target_hair_native).clamp(0, 1)
        )

        # Extract one source-native object instance.  Search and recall masks
        # can help locate a parser-missed pendant, but cannot themselves become
        # write alpha.  The extractor keeps only lobe-connected components and
        # explicitly removes source background, source hair and hoop interiors.
        native_parser_seed = parsing_label_mask(
            source_parsing_native,
            (RAW_EARRING,),
        ).to(device=native.device, dtype=native.dtype)
        native_recall_hint = torch.zeros_like(native[:, :1])
        for key in (
            "source_native_earring_seed",
            "source_parser_earring_mask",
            "strong_earring_candidate_core",
        ):
            value = aux.get(key)
            if value is not None:
                native_recall_hint = torch.maximum(
                    native_recall_hint,
                    resize_mask(value, native_size).to(device=native.device, dtype=native.dtype),
                )
        native_extracted = extract_source_native_earring_v6(
            native,
            source_parsing_native,
            source_native_seed=native_parser_seed,
            seed_space=EarringCoordinateSpace.SOURCE_NATIVE,
            source_native_recall_hint=native_recall_hint,
            recall_hint_space=EarringCoordinateSpace.SOURCE_NATIVE,
            max_graph_depth=max(
                1,
                int(getattr(self.args, "earring_component_max_depth", 4)),
            ),
            max_cumulative_cost=max(
                0.1,
                float(getattr(self.args, "earring_component_max_cumulative_cost", 1.85)),
            ),
            # Keep the source instance contiguous and shape-preserving.  The
            # r9 continuation graph admitted neighbouring highlights/background
            # components and produced deformed earrings; long pendants are
            # recovered by the verified structured source-instance fallback
            # below instead of by a permissive component union.
            allow_long_continuation=False,
        )
        native_alpha = native_extracted["source_native_earring_alpha"]
        extracted = {
            "source_alpha": native_alpha,
            "left_source_alpha": native_extracted["source_native_left_alpha"],
            "right_source_alpha": native_extracted["source_native_right_alpha"],
            "source_rgb": native_extracted["source_native_earring_rgb"],
            "presence_state": native_extracted["source_native_presence_state"],
            "instance_confidence": native_extracted["source_native_presence_score"],
            "localization_roi": native_extracted["localization_core"],
            "foreground_seed": native_extracted["parser_earring_seed"],
            "raw_foreground": native_extracted["grabcut_raw"],
            "boundary_band": (
                dilate_mask(native_alpha, 3) - erode_mask(native_alpha, 3)
            ).clamp(0, 1),
            "hole_mask": native_extracted["source_native_hole_alpha"],
            **native_extracted,
        }
        source_instance = EarringNativeInstanceV6(
            alpha=extracted["source_alpha"],
            rgb=extracted["source_rgb"],
            hole_alpha=extracted["hole_mask"],
            left_alpha=extracted["left_source_alpha"],
            right_alpha=extracted["right_source_alpha"],
            left_hole_alpha=native_extracted["source_native_left_hole_alpha"],
            right_hole_alpha=native_extracted["source_native_right_hole_alpha"],
        )
        source_left_ear = parsing_label_mask(source_parsing_native, (7,))
        source_right_ear = parsing_label_mask(source_parsing_native, (8,))
        aligned = align_earring_instance_v6(
            source_instance,
            source_left_ear,
            source_right_ear,
            target_left_ear,
            target_right_ear,
            max_shift=max(
                0,
                int(
                    round(
                        float(getattr(self.args, "earring_align_max_shift", 16))
                        * max(native_size)
                        / 256.0
                    )
                ),
            ),
        )
        native_base = F.interpolate(base, size=native_size, mode="bilinear", align_corners=False)
        native_result, native_alpha = composite_earring_v6(
            native_base,
            aligned,
            target_left_ear,
            target_right_ear,
            min_visible_area=max(
                2.0,
                float(getattr(self.args, "min_target_ear_area", 8.0))
                * (max(native_size) / max(1, int(getattr(self.parsing_helper, "parse_size", 512)))) ** 2,
            ),
        )
        result = (
            native_result
            if native_size == size
            else F.interpolate(native_result, size=size, mode="bilinear", align_corners=False)
        )
        alpha = (
            native_alpha
            if native_size == size
            else F.interpolate(native_alpha, size=size, mode="bilinear", align_corners=False)
        )

        # Retain V19 loss/debug keys while publishing the coordinate-explicit
        # native fields used by the strict compositor.
        aux["source_native_earring_alpha"] = extracted["source_alpha"]
        aux["source_native_earring_rgb"] = extracted["source_rgb"]
        aux["source_native_earring_seed"] = native_parser_seed
        aux["source_native_earring_recall_hint"] = native_recall_hint
        aux["source_native_earring_seed_space"] = EarringCoordinateSpace.SOURCE_NATIVE
        aux["target_aligned_earring_alpha"] = aligned["target_aligned_earring_alpha"]
        aux["target_aligned_earring_rgb"] = aligned["target_aligned_earring_rgb"]
        aux["output_source_earring_composite_mask"] = alpha
        aux["output_v19_source_alpha"] = alpha
        aux["output_v19_source_rgb"] = F.interpolate(
            aligned["target_aligned_earring_rgb"],
            size=size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        aux["output_v19_source_detail_mask"] = source_detail_alpha
        aux["output_v19_presence_state"] = extracted["presence_state"]
        aux["output_v19_instance_confidence"] = extracted["instance_confidence"]
        aux["output_v19_localization_roi"] = resize_mask(extracted["localization_roi"], size)
        aux["output_v19_foreground_seed"] = resize_mask(extracted["foreground_seed"], size)
        aux["output_v19_raw_foreground"] = resize_mask(extracted["raw_foreground"], size)
        aux["output_v19_boundary_band"] = resize_mask(extracted["boundary_band"], size)
        aux["output_v19_hole_mask"] = resize_mask(extracted["hole_mask"], size)
        aux["output_v19_alpha_leak"] = torch.zeros_like(alpha)
        alpha_flat = alpha.flatten(1)
        rows = (alpha > 0.5).amax(dim=3).float()
        row_ids = torch.arange(size[0], device=alpha.device, dtype=alpha.dtype).view(1, 1, size[0])
        first_row = torch.where(rows > 0, row_ids, torch.full_like(row_ids, float(size[0]))).amin(dim=2)
        last_row = (rows * row_ids).amax(dim=2)
        aux["output_v19_alpha_area"] = alpha_flat.sum(dim=1, keepdim=True).view(-1, 1, 1, 1)
        aux["output_v19_vertical_extent"] = ((last_row - first_row + 1.0).clamp_min(0) / max(1.0, float(size[0]))).view(-1, 1, 1, 1)
        aux["output_v19_hair_overlap"] = (
            alpha * resize_mask(target_hair_output, size)
        ).flatten(1).sum(dim=1, keepdim=True).view(-1, 1, 1, 1)
        source_background_native = (source_parsing_native == 0).to(native.dtype)
        aux["output_v19_source_background_leak"] = (
            extracted["source_alpha"] * source_background_native
        ).flatten(1).sum(dim=1, keepdim=True).view(-1, 1, 1, 1)
        aux["output_v19_unified_face"] = face_pp_alpha
        aux["output_face_target_authority_mask"] = torch.zeros_like(alpha)
        aux["output_direct_face_skin_restore_mask"] = torch.zeros_like(alpha)
        aux["output_target_hair_preserve_mask"] = resize_mask(target_hair_output, size)
        return result.clamp(0, 1) * 2.0 - 1.0

    @staticmethod
    def _resolve_v6_target_ear_masks(
        target_parsing_native: torch.Tensor,
        target_hair_native: torch.Tensor,
        source_parsing_native: torch.Tensor | None,
        aux: dict[str, torch.Tensor],
        native: torch.Tensor,
        native_size: tuple[int, int],
        min_target_area: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve exposed target ears without turning an ROI into an ear.

        Parser labels are the spatial alignment authority.  The query-builder
        masks can only open a side when a small parser miss is backed by an
        exposed, low-hair target skin corridor; the compositor receives scalar
        side gates so long earrings are never clipped to the ear shell.
        """
        parser_left = (
            parsing_label_mask(target_parsing_native, (7,))
            * (1.0 - target_hair_native).clamp(0, 1)
        )
        parser_right = (
            parsing_label_mask(target_parsing_native, (8,))
            * (1.0 - target_hair_native).clamp(0, 1)
        )

        # SATD can legitimately rewrite the background immediately beside an
        # exposed lobe.  At that point the target parser may label the lobe as
        # background even though the face/ear geometry itself has not moved.
        # Use the source face's ear labels as a geometry-only fallback, then
        # subtract the final target hairstyle below.  This opens a side only
        # when the corresponding source ear exists; source earring presence is
        # still decided independently by the native object extractor.
        # Use the same source-native parse that the object extractor consumes.
        # The former aux fallback was normally a 256px PP parse enlarged to
        # 1024px.  When SATD removed a thin earring/lobe edge, that coarse map
        # contained no usable ear geometry and the target side was closed even
        # though the final image visibly exposed the lobe.
        if source_parsing_native is not None:
            source_parsing_native = ensure_mask_4d(source_parsing_native).to(device=native.device)
            if tuple(source_parsing_native.shape[-2:]) != native_size:
                source_parsing_native = F.interpolate(
                    source_parsing_native.float(), size=native_size, mode="nearest"
                ).long()
            source_left_ear_geometry = parsing_label_mask(
                source_parsing_native, (RAW_LEFT_EAR,)
            ).to(device=native.device, dtype=native.dtype)
            source_right_ear_geometry = parsing_label_mask(
                source_parsing_native, (RAW_RIGHT_EAR,)
            ).to(device=native.device, dtype=native.dtype)
        else:
            source_left_ear_geometry = torch.zeros_like(parser_left)
            source_right_ear_geometry = torch.zeros_like(parser_right)

        def aux_mask(name: str) -> torch.Tensor:
            value = aux.get(name)
            if value is None:
                return torch.zeros_like(parser_left)
            return resize_mask(value, native_size).to(
                device=native.device,
                dtype=native.dtype,
            ).clamp(0, 1)

        # The final parser and HM_X jointly define occlusion.  SATD can make an
        # exposed lobe look like background to the parser, while the transfer
        # mask remains the authoritative record of where the new hairstyle
        # actually covers the source-coordinate ear.
        target_hair_occlusion = target_hair_native.clamp(0, 1)
        target_hair_hint_native = aux.get("target_hair_mask")
        if target_hair_hint_native is not None:
            target_hair_hint_native = resize_mask(
                target_hair_hint_native,
                native_size,
            ).to(device=native.device, dtype=native.dtype)
            target_hair_occlusion = torch.maximum(
                target_hair_occlusion,
                (target_hair_hint_native > 0.25).to(native.dtype),
            ).clamp(0, 1)
        left_roi = aux_mask("left_ear_roi")
        right_roi = aux_mask("right_ear_roi")
        # Merge the final target parser's hair evidence into the lower
        # resolution occlusion mask.  Otherwise a stale query mask can report
        # an ear as open even though the completed target image now has hair
        # over the lobe.
        target_skin = torch.maximum(
            aux_mask("target_skin_surface_mask"),
            parsing_label_mask(
                target_parsing_native,
                RAW_FACE_SURFACE_LABELS + RAW_DETAIL_LABELS + (RAW_LEFT_EAR, RAW_RIGHT_EAR),
            ).to(device=native.device, dtype=native.dtype),
        ).clamp(0, 1)
        target_skin = torch.maximum(
            target_skin,
            torch.maximum(source_left_ear_geometry, source_right_ear_geometry),
        ).clamp(0, 1)
        if target_hair_occlusion.flatten(1).amax(dim=1).max().item() <= 0:
            target_hair_occlusion = (
                target_hair_native * torch.clamp(left_roi + right_roi, 0, 1)
            ).clamp(0, 1)

        # Use the source ear geometry only as a target visibility fallback.
        # Coordinates are shared by the aligned face; subtracting HM_X/final
        # hair keeps a fully covered lobe closed.  This restores the case where
        # SATD cleaned the lobe/background boundary and the target parser no
        # longer emits label 7/8, without granting source RGB or a new object.
        parser_left = torch.maximum(
            parser_left,
            source_left_ear_geometry * (1.0 - target_hair_occlusion).clamp(0, 1),
        ).clamp(0, 1)
        parser_right = torch.maximum(
            parser_right,
            source_right_ear_geometry * (1.0 - target_hair_occlusion).clamp(0, 1),
        ).clamp(0, 1)

        left_skin = (
            target_skin * left_roi * (1.0 - target_hair_occlusion).clamp(0, 1)
        ).clamp(0, 1)
        right_skin = (
            target_skin * right_roi * (1.0 - target_hair_occlusion).clamp(0, 1)
        ).clamp(0, 1)

        def side_masks(
            parser_ear: torch.Tensor,
            skin_candidate: torch.Tensor,
            side_name: str,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # A low-resolution query can miss a small, but genuinely exposed,
            # high-resolution earlobe.  Its result is therefore a useful
            # opening signal, not the only one.  The completed target parser
            # provides the independent fallback: only a real exposed ear label
            # after target-hair subtraction can reopen that side.  Generic
            # skin in the broad ear ROI remains forbidden here, so a cheek or
            # neck cannot make an earring appear through covered hair.
            # ``left/right_side_active`` are intentionally multiplied by
            # source-earring presence inside the low-resolution query branch.
            # They therefore answer “is there already a detected source
            # object?”, not “is this target lobe visible?”.  Using them here
            # closed an actually exposed target ear before the native source
            # verifier could recover a parser-missed pearl or metal hoop.
            # The separate target-only key is produced before that source
            # gate; retain the older field only as a legacy fallback.
            query_side_active = aux_mask(
                "left_target_side_open" if side_name == "left" else "right_target_side_open"
            )
            if query_side_active.flatten(1).amax().item() <= 0:
                query_side_active = aux_mask(
                    "left_side_active" if side_name == "left" else "right_side_active"
                )
            query_side_open = (
                query_side_active.flatten(1).amax(dim=1, keepdim=True) > 0.5
            ).view(-1, 1, 1, 1)
            parser_area = parser_ear.flatten(1).sum(dim=1, keepdim=True)
            skin_area = skin_candidate.flatten(1).sum(dim=1, keepdim=True)
            side_roi = aux_mask("left_ear_roi" if side_name == "left" else "right_ear_roi")
            hair_area = (target_hair_occlusion * side_roi).flatten(1).sum(dim=1, keepdim=True)
            corridor_area = ((skin_candidate + target_hair_occlusion) * side_roi).flatten(1).sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            lobe_hint = aux_mask(
                "left_lobe_anchor" if side_name == "left" else "right_lobe_anchor"
            )
            lobe_hint_present = (
                lobe_hint.flatten(1).sum(dim=1, keepdim=True) >= 0.5
            ).view(-1, 1, 1, 1)
            # A small exposed lobe can sit inside a mostly hair-covered ear.
            # Evaluate that local lobe separately instead of rejecting the
            # side from the corridor-wide hair ratio alone.
            lobe_region = torch.where(
                lobe_hint_present,
                lobe_hint,
                skin_candidate,
            )
            lobe_window = dilate_mask(lobe_region, 5)
            lobe_skin_area = (
                target_skin * lobe_window * (1.0 - target_hair_occlusion).clamp(0, 1)
            ).flatten(1).sum(
                dim=1, keepdim=True
            )
            lobe_hair_area = (target_hair_occlusion * lobe_window).flatten(1).sum(
                dim=1, keepdim=True
            )
            lobe_corridor_area = (
                (target_skin + target_hair_occlusion) * lobe_window
            ).flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
            # Keep the measurements as diagnostics.  They are deliberately no
            # longer a second opening rule: they can be contaminated by the
            # opposite side when a profile face has a broad ROI.
            hair_ratio = hair_area / corridor_area
            del skin_area, hair_ratio
            # A narrow exposed lobe can be only a few pixels at the parser
            # resolution.  Requiring the full ear-area threshold here was the
            # reason visible lobes were treated as covered.  Keep a tiny
            # absolute floor, while also allowing a local exposed-skin proof
            # when the parser labels that lobe as skin instead of ear.
            # A one/two-pixel parser island is commonly a hair/cheek error at
            # native resolution, not an exposed earlobe.  Require a small but
            # real lower-lobe footprint on every side; this is still far below
            # the full-ear threshold and preserves genuinely exposed lobes.
            lobe_visibility_floor = max(4.0, 0.125 * float(min_target_area))
            parser_visible = (parser_area >= lobe_visibility_floor).view(-1, 1, 1, 1)
            lobe_skin_visible = (
                lobe_skin_area >= max(3.0, 0.75 * lobe_visibility_floor)
            ).view(-1, 1, 1, 1)
            lobe_hair_ratio = lobe_hair_area / lobe_corridor_area
            aux[f"target_lobe_hair_cover_ratio_{side_name}"] = lobe_hair_ratio.detach()
            aux[f"target_lobe_exposed_area_{side_name}"] = lobe_skin_area.detach()
            # A parser ear island above the lobe is not enough to prove that
            # the target earlobe is exposed.  Recompute a target-only lower
            # lobe anchor from the final target ear mask (with skin fallback)
            # and require that anchor to survive target-hair occlusion.
            # Use the query lobe only as a spatial hint for skin-labelled
            # lobes; never treat the entire broad skin ROI as an exposed ear.
            target_lobe_fallback = skin_candidate * dilate_mask(lobe_hint, 5)
            target_lobe_anchor = build_earlobe_anchor(
                parser_ear,
                fallback_skin_mask=target_lobe_fallback,
                ear_roi=side_roi,
                lower_ratio=0.70,
                dilate=3,
            ) * (1.0 - target_hair_occlusion).clamp(0, 1)
            target_lobe_present = (
                target_lobe_anchor.flatten(1).sum(dim=1, keepdim=True)
                >= max(4.0, 0.75 * lobe_visibility_floor)
            ).view(-1, 1, 1, 1)
            # The final target lobe is closed only when it is genuinely
            # occluded: at least 85% of its local probe is hair and there is
            # no meaningful exposed ear/skin evidence.  A small exposed lower
            # lobe must remain eligible even if most of the ear shell is under
            # the transferred hairstyle.  The old <=25% rule rejected exactly
            # those visible-lobe cases and caused missed long earrings.
            meaningful_exposure = parser_visible | lobe_skin_visible
            fully_covered = (
                (lobe_hair_ratio >= 0.85).view(-1, 1, 1, 1)
                & ~meaningful_exposure
            )
            aux[f"target_lobe_fully_covered_{side_name}"] = fully_covered.to(native.dtype).detach()
            # The query-builder flag is pre-decode information.  It cannot
            # veto the final target RGB/parser decision.
            del query_side_open
            open_side = (target_lobe_present & ~fully_covered).to(native.dtype)
            # The parser ear is a visibility test, not an attachment point:
            # its centroid is often in the middle of the ear shell.  When the
            # query builder has a lobe anchor, use that compact target-lobe
            # mask for alignment even if parser pixels are also present.  The
            # scalar gate below still comes from the exposed ear decision, so
            # this cannot enlarge or redraw the target lobe.
            # Alignment must use the final target ear geometry, not the
            # query-builder lobe hint (which may contain source-ear pixels and
            # leaves a recovered earring at its old source coordinate).
            # ``target_lobe_anchor`` is already lower-lobe-only and hair-free.
            align_base = target_lobe_anchor
            align_mask = align_base * open_side.view(-1, 1, 1, 1)
            composite_gate = open_side.view(-1, 1, 1, 1).expand_as(parser_ear)
            return align_mask, composite_gate, open_side.view(-1, 1, 1, 1)

        left_align, left_gate, left_open = side_masks(parser_left, left_skin, "left")
        right_align, right_gate, right_open = side_masks(parser_right, right_skin, "right")
        return left_align, right_align, left_gate, right_gate, torch.cat(
            (left_open, right_open), dim=1
        )

    def _compose_strict_source_native_earring_v6(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Keep one global PP face and write only verified native earrings.

        The decoded image is the sole RGB authority for face, hair, neck and
        background.  The completed transfer is parsed only to decide whether
        each target lobe is actually visible.  This prevents a face-shaped
        source RGB overlay from creating a mask boundary while retaining the
        one permitted local exception: a confirmed earring foreground.
        """
        image_01 = normalized_to_01(image).clamp(0, 1)
        size = tuple(image_01.shape[-2:])
        native = aux.get("source_full_01")
        if native is None:
            native = aux.get("source_face_reference_01", aux.get("source_01"))
        if native is None:
            aux["output_source_earring_composite_mask"] = torch.zeros_like(image_01[:, :1])
            return image
        native = normalized_to_01(native).to(device=image_01.device, dtype=image_01.dtype)
        if native.shape[0] != image_01.shape[0]:
            aux["output_source_earring_composite_mask"] = torch.zeros_like(image_01[:, :1])
            return image
        native_size = tuple(native.shape[-2:])

        source_parsing = aux.get("source_parsing")
        if source_parsing is None or tuple(ensure_mask_4d(source_parsing).shape[-2:]) != native_size:
            source_parsing_native = self.parsing_helper.parse(native, out_size=native_size)
        else:
            source_parsing_native = ensure_mask_4d(source_parsing).to(device=native.device).long()

        # Visibility must be measured on the exact RGB canvas that will receive
        # the earring.  ``authoritative_target_highres_01`` is a comparison
        # artifact from the transfer stage and can have a different hairline;
        # using it here reopened/closed the wrong target side.
        target_geometry = image_01
        target_parsing_output = self.parsing_helper.parse(target_geometry, out_size=size)
        if native_size == size:
            target_parsing_native = target_parsing_output
        else:
            target_native = F.interpolate(target_geometry, size=native_size, mode="bilinear", align_corners=False)
            target_parsing_native = self.parsing_helper.parse(target_native, out_size=native_size)
        target_hair_native = parsing_label_mask(target_parsing_native, (RAW_HAIR,))
        # The final parser can miss a thin transferred strand or classify it
        # as background/skin.  The target transfer mask is the authoritative
        # hairstyle geometry already used by the blending stage; union it
        # here before deciding whether an ear is exposed.  This closes a
        # fully-covered ear even when the high-resolution parser leaves a few
        # false ear pixels at the hair boundary.
        native_area_scale = (
            max(native_size) / max(1, int(getattr(self.parsing_helper, "parse_size", 512)))
        ) ** 2
        min_target_area = max(
            2.0,
            float(getattr(self.args, "min_target_ear_area", 8.0)) * native_area_scale,
        )
        (
            target_left_ear,
            target_right_ear,
            target_left_gate,
            target_right_gate,
            target_side_open,
        ) = self._resolve_v6_target_ear_masks(
            target_parsing_native,
            target_hair_native,
            source_parsing_native,
            aux,
            native,
            native_size,
            min_target_area,
        )
        aux["v6_target_left_exposed_ear"] = target_left_ear.detach()
        aux["v6_target_right_exposed_ear"] = target_right_ear.detach()
        aux["v6_target_left_ear_open"] = target_side_open[:, :1].detach()
        aux["v6_target_right_ear_open"] = target_side_open[:, 1:2].detach()

        # The final target parse can miss a solid interior island of transferred
        # hair.  Supplement it with only the eroded interior of the 256px
        # transfer mask; this cannot move the hairline, but it stops the PP
        # decoder from replacing an existing hair region with source-conditioned
        # clothing.  The high-resolution transfer remains the RGB authority for
        # these pixels, while PP keeps ownership of the face and neck.
        raw_target_hair_hint = aux.get("target_hair_mask")
        target_hair_hint = (
            resize_mask(raw_target_hair_hint, size)
            if raw_target_hair_hint is not None
            else torch.zeros_like(image_01[:, :1])
        )
        target_hair_hint_core = erode_mask(
            (target_hair_hint > 0.5).to(image_01.dtype),
            9,
        )
        target_hair_output = torch.maximum(
            parsing_label_mask(target_parsing_output, (RAW_HAIR,)),
            target_hair_hint_core,
        )
        target_hair_binary = (target_hair_output > 0.5).to(image_01.dtype)
        aux["v6_target_hair_preserve_mask"] = target_hair_binary.detach()

        native_parser_seed = parsing_label_mask(
            source_parsing_native,
            (RAW_EARRING,),
        ).to(device=native.device, dtype=native.dtype)
        native_recall_hint = torch.zeros_like(native[:, :1])
        for key in (
            "source_native_earring_seed",
            "source_parser_earring_mask",
            "strong_earring_candidate_core",
        ):
            value = aux.get(key)
            if value is not None:
                native_recall_hint = torch.maximum(
                    native_recall_hint,
                    resize_mask(value, native_size).to(device=native.device, dtype=native.dtype),
                )
        extracted = extract_source_native_earring_v6(
            native,
            source_parsing_native,
            source_native_seed=native_parser_seed,
            seed_space=EarringCoordinateSpace.SOURCE_NATIVE,
            source_native_recall_hint=native_recall_hint,
            recall_hint_space=EarringCoordinateSpace.SOURCE_NATIVE,
            # Allow the bounded continuation pass so a long pendant is not
            # reduced to its bright attachment rim.  Candidate components
            # still pass the source-lobe, hair and background gates below.
            max_graph_depth=max(
                1,
                int(getattr(self.args, "earring_component_max_depth", 4)),
            ),
            max_cumulative_cost=max(
                0.1,
                float(getattr(self.args, "earring_component_max_cumulative_cost", 1.85)),
            ),
            allow_long_continuation=True,
        )
        # Preserve extraction decisions for trace/validation.  These are
        # diagnostics only; no query/label map is used as source RGB alpha.
        for key in (
            "component_labels",
            "component_scores",
            "selected_components",
            "accepted_component_ids",
            "rejected_component_ids",
            "root_component_id",
            "accepted_component_count",
            "rejected_component_count",
            "reject_reason",
        ):
            if key in extracted:
                aux[f"source_{key}"] = extracted[key].detach()

        # The native graph can still keep only a high-contrast rim when a
        # pendant's interior is low contrast.  Reuse the measured structured
        # source-instance fallback already used by V5.  It is source-native,
        # hair-aware, area-capped, and carries a separate hoop hole; it never
        # pastes an ear/background crop or invents a geometric ellipse.
        # A strict 256px strong candidate is a localisation hint, not RGB
        # alpha.  It must nevertheless reach this native verifier.  Passing
        # only ``native_parser_seed`` meant a parser-missed but visually
        # obvious earring could never enter the structured path at all.
        structured_seed = torch.clamp(native_parser_seed + native_recall_hint, 0, 1)
        structured_instances = build_source_earring_instance_masks_v5(
            native,
            source_parsing_native,
            source_hair_mask=parsing_label_mask(source_parsing_native, (RAW_HAIR,)),
            source_seed_mask=structured_seed,
        )
        output_area = float(native_size[0] * native_size[1])
        # Large hoops/pendants legitimately occupy more than the old 5.5%
        # cap.  The structured instance is still source-connected and
        # background/hair filtered, so raise only this object-area ceiling
        # rather than opening an ear ROI.
        fallback_area_cap = 0.10 * output_area
        minimum_fallback_area = max(
            8.0,
            2.0 * (max(native_size) / 256.0) ** 2,
        )
        source_hair_native = parsing_label_mask(source_parsing_native, (RAW_HAIR,)).to(
            device=native.device,
            dtype=native.dtype,
        )
        source_background_native = (
            (source_parsing_native.long() == 0).to(device=native.device, dtype=native.dtype)
        )
        # Only parser label 9 (earring) and verified parser-background metal
        # are valid source RGB.  Source face, ear skin, neck, necklace, cloth,
        # hat and hair must remain target-owned; allowing any of those labels
        # into this alpha changes the target lobe shape and creates the red
        # cheek/background contour reported in V6 output.
        source_labels_native = source_parsing_native.long()
        # Ear/skin labels can contain a genuine parser-missed accessory.  Only
        # labels that are never valid earring RGB (hair, neck, necklace, cloth,
        # hat) form the hard semantic veto; independently verified object
        # pixels may pass face/ear/skin labels without importing the subject.
        source_non_ear_subject_native = parsing_label_mask(
            source_parsing_native,
            tuple(RAW_NECK_SURFACE_LABELS) + (16, RAW_HAIR, RAW_HAT),
        ).to(device=native.device, dtype=native.dtype)
        # Parser label 9 can include a one-pixel backdrop fringe.  Keep its
        # eroded core, but reject low-contrast label-9 boundary pixels whose
        # colour is indistinguishable from the surrounding source image; this
        # is the red cheek contour seen in the reported composites.
        source_parser_earring_native = parsing_label_mask(
            source_parsing_native,
            (RAW_EARRING,),
        ).to(device=native.device, dtype=native.dtype)
        source_parser_earring_core = erode_mask(source_parser_earring_native, 3)
        source_local_delta = (
            native
            - gaussian_blur(native, kernel_size=21, sigma=5.0)
        ).abs().mean(dim=1, keepdim=True)
        source_label9_halo_block = (
            source_parser_earring_native
            * (1.0 - source_parser_earring_core).clamp(0, 1)
            * (source_local_delta < 0.015).to(native.dtype)
        ).clamp(0, 1)

        # The native extractor's broad inspection envelope can contain a
        # shoulder or shirt.  Build a narrower, source-lobe-connected rail for
        # parser-background pixels.  It follows the same long-pendant offsets
        # as the structured V5 locator, but it is only a write guard and never
        # creates alpha on its own.
        native_scale = max(native_size) / 256.0

        def scaled_rail(value: float, minimum: int = 3) -> int:
            scaled_value = max(minimum, int(round(value * native_scale)))
            return scaled_value if scaled_value % 2 else scaled_value + 1

        def source_lobe_rail(side: str) -> torch.Tensor:
            context_key = "left_context" if side == "left" else "right_context"
            anchor_key = "left_lobe_anchor" if side == "left" else "right_lobe_anchor"
            context = structured_instances.get(context_key, torch.zeros_like(native[:, :1]))
            anchor = structured_instances.get(anchor_key, torch.zeros_like(native[:, :1]))
            return torch.clamp(
                context
                + shift_mask(
                    dilate_mask(anchor, scaled_rail(9)),
                    down=scaled_rail(48, 1),
                )
                + shift_mask(
                    dilate_mask(anchor, scaled_rail(7)),
                    down=scaled_rail(72, 1),
                ),
                0,
                1,
            ).to(device=native.device, dtype=native.dtype)

        source_left_lobe_rail = source_lobe_rail("left")
        source_right_lobe_rail = source_lobe_rail("right")

        def choose_structured_side(
            direct: torch.Tensor,
            structured: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            direct_area = direct.flatten(1).sum(dim=1, keepdim=True)
            structured_area = structured.flatten(1).sum(dim=1, keepdim=True)
            direct_hair_ratio = (
                (direct * source_hair_native).flatten(1).sum(dim=1, keepdim=True)
                / direct_area.clamp_min(1.0)
            )
            direct_background_ratio = (
                (direct * source_background_native).flatten(1).sum(dim=1, keepdim=True)
                / direct_area.clamp_min(1.0)
            )
            use_structured_area = (
                (structured_area >= minimum_fallback_area)
                & (structured_area <= fallback_area_cap)
            )
            # The native extractor owns a valid source-resolution contour.
            # A structured/parser fallback must not replace it merely because
            # a lower-resolution mask is larger or longer: that replacement
            # was the source of truncated/warped long pendants.  Fall back
            # only when native extraction produced no usable object at all.
            prefer_structured = (
                use_structured_area
                & (direct_area < minimum_fallback_area)
            ).view(-1, 1, 1, 1)
            selected = torch.where(prefer_structured, structured, direct)
            return selected, prefer_structured

        selected_left, use_structured_left = choose_structured_side(
            extracted["source_native_left_alpha"],
            structured_instances["left_instance_mask"].to(
                device=native.device,
                dtype=native.dtype,
            ),
        )
        selected_right, use_structured_right = choose_structured_side(
            extracted["source_native_right_alpha"],
            structured_instances["right_instance_mask"].to(
                device=native.device,
                dtype=native.dtype,
            ),
        )
        # A semantic label-9 source earring is already a verified accessory
        # pixel.  If GrabCut and the visual instance pass both reject it (a
        # common failure for tiny studs or low-contrast metal), retain that
        # source-native label as a last-resort object-only fallback.  It never
        # copies source ear/background pixels and still receives topology-hole
        # protection below.
        source_left_ear = parsing_label_mask(source_parsing_native, (7,))
        source_right_ear = parsing_label_mask(source_parsing_native, (8,))
        parser_left_seed, parser_right_seed = assign_components_to_ear_sides(
            native_parser_seed,
            source_left_ear,
            source_right_ear,
            source_left_ear,
            source_right_ear,
        )
        # All source-native masks participate in broadcasted fallback writes;
        # normalize legacy dataset/extractor outputs that may omit channel 1.
        parser_left_seed = ensure_mask_4d(parser_left_seed).to(
            device=native.device, dtype=native.dtype
        )
        parser_right_seed = ensure_mask_4d(parser_right_seed).to(
            device=native.device, dtype=native.dtype
        )
        # A semantic earring label may contain a long pendant beyond the
        # short source-lobe rail used for parser-background verification.
        # Keep exactly its one lobe-connected component/vertical chain.  This
        # is deliberately label-9-only: label-0 pixels still need the stricter
        # material verifier below and therefore cannot copy source backdrop.
        parser_chain_gap = max(3, int(round(6.0 * max(native_size) / 256.0)))
        parser_chain_extent = None
        parser_left_chain = retain_single_earring_group_v6(
            parser_left_seed,
            source_left_ear,
            max_gap=parser_chain_gap,
            max_downward_extent=parser_chain_extent,
        )
        parser_right_chain = retain_single_earring_group_v6(
            parser_right_seed,
            source_right_ear,
            max_gap=parser_chain_gap,
            max_downward_extent=parser_chain_extent,
        )
        parser_left_seed = parser_left_chain
        parser_right_seed = parser_right_chain
        aux["v6_source_left_label9_chain"] = parser_left_chain.detach()
        aux["v6_source_right_label9_chain"] = parser_right_chain.detach()
        # Parser-background metal is legal only when it is close to an
        # independently verified source object/label-9 seed.  The previous
        # rail-only rule still admitted a red backdrop strip running beside
        # the cheek because that strip happened to lie inside the lobe rail.
        # This support keeps parser-missed thin metal while rejecting that
        # disconnected background edge.
        # Background-labelled pixels are eligible only when they also carry
        # measurable object texture.  This removes the smooth backdrop/strand
        # pixels that happened to fall inside a lobe rail while preserving
        # parser-missed metal wires (label-9 pixels remain unaffected).
        # Background-labelled pixels require a strong local object signal;
        # the previous low threshold admitted thin source backdrop strands.
        source_background_material = (source_local_delta >= 0.035).to(native.dtype)
        # Label-0 is never expanded from a label-9 seed.  That old dilation
        # was intended to keep parser-missed metal, but it also promoted the
        # red backdrop/fine source hairs beside an earring to source RGB.  A
        # background-labelled pixel is valid only when a native foreground or
        # structured object extractor already accepted that exact pixel; a
        # semantic label-9 pendant follows its own chain above.
        source_left_bg_verified = torch.clamp(
            extracted["source_native_left_alpha"]
            + structured_instances["left_instance_mask"].to(
                device=native.device,
                dtype=native.dtype,
            ),
            0,
            1,
        ) * source_background_native * source_background_material
        source_right_bg_verified = torch.clamp(
            extracted["source_native_right_alpha"]
            + structured_instances["right_instance_mask"].to(
                device=native.device,
                dtype=native.dtype,
            ),
            0,
            1,
        ) * source_background_native * source_background_material
        parser_fallback_min_area = max(
            2.0,
            0.5 * (max(native_size) / max(1, int(getattr(self.parsing_helper, "parse_size", 512)))) ** 2,
        )
        left_needs_parser = (
            selected_left.flatten(1).sum(dim=1, keepdim=True) < parser_fallback_min_area
        ).view(-1, 1, 1, 1)
        right_needs_parser = (
            selected_right.flatten(1).sum(dim=1, keepdim=True) < parser_fallback_min_area
        ).view(-1, 1, 1, 1)
        parser_left_area = parser_left_seed.flatten(1).sum(dim=1, keepdim=True)
        parser_right_area = parser_right_seed.flatten(1).sum(dim=1, keepdim=True)
        parser_left_ok = (parser_left_area >= parser_fallback_min_area).view(-1, 1, 1, 1)
        parser_right_ok = (parser_right_area >= parser_fallback_min_area).view(-1, 1, 1, 1)
        visual_left_present = structured_instances.get(
            "left_visual_recall_present",
            torch.zeros_like(parser_left_ok),
        ).to(device=native.device, dtype=native.dtype)
        visual_right_present = structured_instances.get(
            "right_visual_recall_present",
            torch.zeros_like(parser_right_ok),
        ).to(device=native.device, dtype=native.dtype)
        explicit_source = torch.zeros_like(native_parser_seed)
        # Prefer the exact object mask.  ``source_earring_mask`` is a legacy
        # search/ROI mask and can include the lobe surrounding a long pendant;
        # using it as the write alpha recreates the coloured ear blobs this V6
        # path is intended to avoid.
        explicit_value = aux.get("source_earring_object_mask")
        # Only an explicitly verified SOURCE_NATIVE object mask may be used as
        # a source RGB fallback.  Legacy query/ROI fields are intentionally not
        # accepted here: they are target-canonical search masks, and treating
        # them as alpha is the direct cause of duplicate earrings and copied
        # source strands.
        if explicit_value is not None:
            explicit_source = resize_mask(explicit_value, native_size).to(
                device=native.device,
                dtype=native.dtype,
            ).clamp(0, 1)
        explicit_left_value = aux.get("source_native_left_alpha")
        explicit_right_value = aux.get("source_native_right_alpha")
        if torch.is_tensor(explicit_left_value) and torch.is_tensor(explicit_right_value):
            explicit_left = resize_mask(explicit_left_value, native_size).to(
                device=native.device, dtype=native.dtype
            ).clamp(0, 1)
            explicit_right = resize_mask(explicit_right_value, native_size).to(
                device=native.device, dtype=native.dtype
            ).clamp(0, 1)
        else:
            explicit_left, explicit_right = assign_components_to_ear_sides(
                explicit_source,
                source_left_ear,
                source_right_ear,
                source_left_ear,
                source_right_ear,
            )
        explicit_left = ensure_mask_4d(explicit_left).to(
            device=native.device, dtype=native.dtype
        )
        explicit_right = ensure_mask_4d(explicit_right).to(
            device=native.device, dtype=native.dtype
        )
        explicit_left_before_v6_filter = explicit_left.clone()
        explicit_right_before_v6_filter = explicit_right.clone()
        verified_left_value = aux.get("source_instance_verified_left")
        verified_right_value = aux.get("source_instance_verified_right")
        def verified_side(value):
            if not torch.is_tensor(value):
                return torch.zeros(native.size(0), 1, 1, 1, device=native.device, dtype=torch.bool)
            value = value.to(device=native.device, dtype=native.dtype)
            if value.ndim == 0:
                value = value.view(1)
            if value.ndim == 1:
                return value.gt(0.5).view(-1, 1, 1, 1)
            return value.flatten(1).amax(dim=1, keepdim=True).gt(0.5).view(-1, 1, 1, 1)
        verified_left = (
            verified_side(verified_left_value)
        )
        verified_right = (
            verified_side(verified_right_value)
        )
        # Dataset object masks are source-native and were verified during
        # generation.  They must be a real alpha fallback, not merely a side
        # permission flag: visual/GrabCut extraction can otherwise return an
        # empty mask for small studs or only a short rim of a long pendant.
        # Keep object pixels, remove source-hair fringe, and preserve hoop
        # interiors as holes so no source background is pasted back.
        explicit_hair_block = dilate_mask(source_hair_native, 1)
        # The explicit dataset mask is object-only.  Keep its native contour
        # intact except for the source-hair fringe; the earring interior/hole
        # logic below remains separate from RGB alpha.
        explicit_left = explicit_left * (1.0 - explicit_hair_block).clamp(0, 1)
        explicit_right = explicit_right * (1.0 - explicit_hair_block).clamp(0, 1)
        # Exclude source neck/necklace/cloth/hat unconditionally.  A genuine
        # thin/parser-background earring is retained only in the narrow rail
        # attached to its source lobe, so a broad white shirt cannot become
        # write alpha through the explicit dataset fallback.
        explicit_left = explicit_left * (1.0 - source_non_ear_subject_native).clamp(0, 1)
        explicit_right = explicit_right * (1.0 - source_non_ear_subject_native).clamp(0, 1)
        explicit_left = explicit_left * (1.0 - source_label9_halo_block).clamp(0, 1)
        explicit_right = explicit_right * (1.0 - source_label9_halo_block).clamp(0, 1)
        explicit_left = explicit_left * (
            (1.0 - source_background_native).clamp(0, 1)
            + source_background_native * source_left_lobe_rail
        ).clamp(0, 1)
        explicit_right = explicit_right * (
            (1.0 - source_background_native).clamp(0, 1)
            + source_background_native * source_right_lobe_rail
        ).clamp(0, 1)
        explicit_left = explicit_left * (
            (1.0 - source_background_native).clamp(0, 1)
            + source_background_native * source_left_bg_verified
        ).clamp(0, 1)
        explicit_right = explicit_right * (
            (1.0 - source_background_native).clamp(0, 1)
            + source_background_native * source_right_bg_verified
        ).clamp(0, 1)
        # A verified dataset alpha is still source RGB and must obey the same
        # semantic contract as the online extractor.  The previous verified
        # fast-path bypassed the clothing/background guards; when an old or
        # imperfect mask contained a nearby shirt/strand, that bypass pasted
        # the source background as a second earring.  Preserve parser label-9
        # metal and only parser-background pixels supported by the measured
        # source object; never preserve source ear/skin/neck/cloth pixels.
        # Parser label 9 is useful semantic evidence, but a mistaken label-9
        # fringe beside the cheek must not become source RGB.  Keep only the
        # lobe-connected source label-9 chain.  Unlike ``source_*_lobe_rail``,
        # this follows a verified long pendant past 140px at the 256 reference
        # size; parser-background metal remains subject to the stricter
        # material support below.
        source_label9_left_allowed = (
            (source_labels_native == RAW_EARRING).to(native.dtype)
            * parser_left_chain
        )
        source_label9_right_allowed = (
            (source_labels_native == RAW_EARRING).to(native.dtype)
            * parser_right_chain
        )
        parser_label9 = (source_labels_native == RAW_EARRING).to(native.dtype)
        explicit_left = explicit_left * (
            (1.0 - parser_label9).clamp(0, 1) + source_label9_left_allowed
        ).clamp(0, 1)
        explicit_right = explicit_right * (
            (1.0 - parser_label9).clamp(0, 1) + source_label9_right_allowed
        ).clamp(0, 1)
        explicit_left = torch.where(
            verified_left,
            explicit_left_before_v6_filter,
            explicit_left,
        )
        explicit_right = torch.where(
            verified_right,
            explicit_right_before_v6_filter,
            explicit_right,
        )
        aux["v6_source_explicit_alpha_before_clothing_filter"] = torch.clamp(
            explicit_left_before_v6_filter + explicit_right_before_v6_filter,
            0,
            1,
        ).detach()
        aux["v6_source_explicit_clothing_reject_mask"] = torch.clamp(
            (explicit_left_before_v6_filter - explicit_left).relu()
            + (explicit_right_before_v6_filter - explicit_right).relu(),
            0,
            1,
        ).detach()
        # ``source_native_earring_alpha`` is already an object-only matte
        # produced by the native verifier.  Parser label 0 is not a safe
        # rejection rule here: thin metal/wire pixels are frequently labelled
        # background even though they are the actual earring body.  Hair is
        # still blocked above, and the native extractor's local background
        # halo test remains responsible for rejecting surrounding strands.
        explicit_left_hole = compute_earring_hole_mask(explicit_left)
        explicit_right_hole = compute_earring_hole_mask(explicit_right)
        source_left_allowed = (
            (parser_left_area > 0.5).view(-1, 1, 1, 1)
            | (visual_left_present > 0.5)
            | (extracted["source_native_left_alpha"].flatten(1).sum(dim=1, keepdim=True) > 0.5).view(-1, 1, 1, 1)
            | (explicit_left.flatten(1).sum(dim=1, keepdim=True) > 0.5).view(-1, 1, 1, 1)
        )
        source_right_allowed = (
            (parser_right_area > 0.5).view(-1, 1, 1, 1)
            | (visual_right_present > 0.5)
            | (extracted["source_native_right_alpha"].flatten(1).sum(dim=1, keepdim=True) > 0.5).view(-1, 1, 1, 1)
            | (explicit_right.flatten(1).sum(dim=1, keepdim=True) > 0.5).view(-1, 1, 1, 1)
        )
        selected_left = selected_left * source_left_allowed.to(native.dtype)
        selected_right = selected_right * source_right_allowed.to(native.dtype)
        aux["v6_source_left_earring_allowed"] = source_left_allowed.to(native.dtype).detach()
        aux["v6_source_right_earring_allowed"] = source_right_allowed.to(native.dtype).detach()
        parser_left_hole = compute_earring_hole_mask(parser_left_seed)
        parser_right_hole = compute_earring_hole_mask(parser_right_seed)
        explicit_left_area = explicit_left.flatten(1).sum(dim=1, keepdim=True)
        explicit_right_area = explicit_right.flatten(1).sum(dim=1, keepdim=True)
        selected_left_area = selected_left.flatten(1).sum(dim=1, keepdim=True)
        selected_right_area = selected_right.flatten(1).sum(dim=1, keepdim=True)
        use_explicit_left = (
            (explicit_left_area >= parser_fallback_min_area)
            & (selected_left_area < parser_fallback_min_area)
        ).view(-1, 1, 1, 1)
        use_explicit_right = (
            (explicit_right_area >= parser_fallback_min_area)
            & (selected_right_area < parser_fallback_min_area)
        ).view(-1, 1, 1, 1)
        # An object mask without a per-side SOURCE_NATIVE verification bit is
        # diagnostic/query data only.  In particular, never revive an old
        # schema's broad target mask through this branch.
        use_explicit_left = use_explicit_left & verified_left
        use_explicit_right = use_explicit_right & verified_right
        selected_left = ensure_mask_4d(selected_left).to(
            device=native.device, dtype=native.dtype
        )
        selected_right = ensure_mask_4d(selected_right).to(
            device=native.device, dtype=native.dtype
        )
        selected_left = torch.where(use_explicit_left, explicit_left, selected_left)
        selected_right = torch.where(use_explicit_right, explicit_right, selected_right)
        # Parser label-9 is the final source fallback.  It cannot replace a
        # non-empty native/verified contour merely because it is larger.
        use_parser_left = (
            parser_left_ok
            & ~use_explicit_left
            & (selected_left_area < parser_fallback_min_area).view(-1, 1, 1, 1)
        )
        use_parser_right = (
            parser_right_ok
            & ~use_explicit_right
            & (selected_right_area < parser_fallback_min_area).view(-1, 1, 1, 1)
        )
        parser_left_write = (
            use_parser_left & ~use_explicit_left
        ).expand_as(parser_left_seed)
        parser_right_write = (
            use_parser_right & ~use_explicit_right
        ).expand_as(parser_right_seed)
        selected_left = torch.where(
            parser_left_write,
            parser_left_seed * (1.0 - parser_left_hole).clamp(0, 1),
            selected_left,
        )
        selected_right = torch.where(
            parser_right_write,
            parser_right_seed * (1.0 - parser_right_hole).clamp(0, 1),
            selected_right,
        )
        # A complete semantic label-9 pendant is a valid source object even
        # when its thin boundary is low contrast.  Keep that parser-authority
        # path intact (apart from real source hair) so a long earring is not
        # reduced to its bright root by the generic halo filter.  Unverified
        # visual/structured candidates still receive the stricter filters.
        selected_before_v6_clothing_filter = torch.clamp(
            selected_left + selected_right,
            0,
            1,
        )
        # The selected contour is already a verified source-native object.
        # Parser hair/neck/cloth/background labels were component evidence in
        # the verifier; they are not a per-pixel veto here.  Pixel masking at
        # this stage shaved long pendants whenever their lower body crossed a
        # coarse hair/neck/background label.  Only the accepted object alpha
        # can write RGB, so this does not authorize an ROI or source backdrop.
        selected_left_filtered = selected_left
        selected_right_filtered = selected_right
        selected_left = selected_left_filtered
        selected_right = selected_right_filtered
        selected_after_v6_clothing_filter = torch.clamp(
            selected_left + selected_right,
            0,
            1,
        )
        aux["v6_source_alpha_before_clothing_filter"] = selected_before_v6_clothing_filter.detach()
        aux["v6_source_alpha_after_clothing_filter"] = selected_after_v6_clothing_filter.detach()
        aux["v6_source_clothing_reject_mask"] = torch.clamp(
            selected_before_v6_clothing_filter - selected_after_v6_clothing_filter,
            0,
            1,
        ).detach()
        # Enforce one physical source accessory per ear side while retaining
        # nearby vertical continuation pieces of a long pendant.
        group_gap = max(3, int(round(6.0 * max(native_size) / 256.0)))
        max_pendant_extent = None
        selected_left = retain_single_earring_group_v6(
            selected_left,
            source_left_ear,
            max_gap=group_gap,
            max_downward_extent=max_pendant_extent,
        )
        selected_right = retain_single_earring_group_v6(
            selected_right,
            source_right_ear,
            max_gap=group_gap,
            max_downward_extent=max_pendant_extent,
        )
        # Independent source-side fallbacks can still describe one physical
        # earring. Decide that source ambiguity before target-lobe anchoring;
        # otherwise one object is shifted to two targets and looks duplicated.
        selected_left, selected_right, removed_duplicate_left, removed_duplicate_right = (
            enforce_exclusive_earring_sides_v6(
                selected_left,
                selected_right,
                source_left_ear,
                source_right_ear,
            )
        )
        # Do not blanket-reject parser-background pixels after the native
        # extraction: a real thin earring is often label 0.  Reapplying that
        # semantic filter here was the main reason exposed earrings vanished.
        # Source hair remains blocked, and native extraction has already
        # removed disconnected background components/halos.
        aux["v6_parser_earring_fallback"] = torch.clamp(
            use_parser_left.to(native.dtype) + use_parser_right.to(native.dtype),
            0,
            1,
        ).detach()
        aux["v6_dataset_earring_fallback"] = torch.clamp(
            use_explicit_left.to(native.dtype) + use_explicit_right.to(native.dtype),
            0,
            1,
        ).detach()
        aux["v6_source_left_earring_area"] = selected_left.flatten(1).sum(dim=1, keepdim=True).detach()
        aux["v6_source_right_earring_area"] = selected_right.flatten(1).sum(dim=1, keepdim=True).detach()
        aux["v6_source_duplicate_left_removed"] = removed_duplicate_left.detach()
        aux["v6_source_duplicate_right_removed"] = removed_duplicate_right.detach()
        aux["v6_source_selected_left_alpha"] = selected_left.detach()
        aux["v6_source_selected_right_alpha"] = selected_right.detach()
        selected_left_hole = torch.where(
            use_explicit_left,
            explicit_left_hole,
            torch.where(
                use_parser_left,
                parser_left_hole,
                torch.where(
                    use_structured_left,
                    structured_instances["left_hoop_hole_mask"].to(
                        device=native.device,
                        dtype=native.dtype,
                    ),
                    extracted["source_native_left_hole_alpha"],
                ),
            ),
        )
        selected_right_hole = torch.where(
            use_explicit_right,
            explicit_right_hole,
            torch.where(
                use_parser_right,
                parser_right_hole,
                torch.where(
                    use_structured_right,
                    structured_instances["right_hoop_hole_mask"].to(
                        device=native.device,
                        dtype=native.dtype,
                    ),
                    extracted["source_native_right_hole_alpha"],
                ),
            ),
        )
        # A discarded duplicate side must not leave behind a translated hoop
        # hole on the other target side.
        selected_left_hole = selected_left_hole * (1.0 - removed_duplicate_left).clamp(0, 1)
        selected_right_hole = selected_right_hole * (1.0 - removed_duplicate_right).clamp(0, 1)
        # Hole ownership follows the final, arbitrated object contour.  A
        # hole left over from a rejected duplicate or a shorter pre-filter
        # parser component otherwise makes a target-owned empty patch beside
        # the real earring.  Recompute it after all source-side decisions;
        # the hole remains alpha=0, so no source background is copied.
        selected_left_hole = compute_earring_hole_mask(selected_left)
        selected_right_hole = compute_earring_hole_mask(selected_right)
        extracted["source_native_left_alpha"] = selected_left
        extracted["source_native_right_alpha"] = selected_right
        extracted["source_native_left_hole_alpha"] = selected_left_hole
        extracted["source_native_right_hole_alpha"] = selected_right_hole
        extracted["source_native_earring_alpha"] = torch.clamp(
            selected_left + selected_right,
            0,
            1,
        )
        extracted["source_native_hole_alpha"] = torch.clamp(
            selected_left_hole + selected_right_hole,
            0,
            1,
        )
        aux["v6_structured_earring_fallback"] = torch.clamp(
            use_structured_left.to(dtype=native.dtype)
            + use_structured_right.to(dtype=native.dtype),
            0,
            1,
        ).detach()
        instance = EarringNativeInstanceV6(
            alpha=extracted["source_native_earring_alpha"],
            rgb=extracted["source_native_earring_rgb"],
            hole_alpha=extracted["source_native_hole_alpha"],
            left_alpha=extracted["source_native_left_alpha"],
            right_alpha=extracted["source_native_right_alpha"],
            left_hole_alpha=extracted["source_native_left_hole_alpha"],
            right_hole_alpha=extracted["source_native_right_hole_alpha"],
        )
        aligned = align_earring_instance_v6(
            instance,
            parsing_label_mask(source_parsing_native, (7,)),
            parsing_label_mask(source_parsing_native, (8,)),
            target_left_ear,
            target_right_ear,
            max_shift=max(
                0,
                int(
                    round(
                        float(getattr(self.args, "earring_align_max_shift", 16))
                        * max(native_size)
                        / 256.0
                    )
                ),
            ),
        )
        # Keep the completed PP canvas at its decoded resolution.  Resampling
        # the whole image down to source-native size and back softened every
        # face/hair pixel even though the earring was the only intended edit.
        # Only the already verified earring object and its masks cross the
        # native/output resolution boundary.
        def object_mask_at_output(value: torch.Tensor, *, soft: bool) -> torch.Tensor:
            value = ensure_mask_4d(value).to(device=image_01.device, dtype=image_01.dtype)
            if value.shape[-2:] == size:
                return value.clamp(0, 1)
            return F.interpolate(
                value,
                size=size,
                mode="bilinear" if soft else "nearest",
                align_corners=False if soft else None,
            ).clamp(0, 1)

        def object_rgb_at_output(value: torch.Tensor) -> torch.Tensor:
            value = normalized_to_01(value).to(device=image_01.device, dtype=image_01.dtype)
            if value.shape[-2:] != size:
                value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            return value.clamp(0, 1)

        aligned_output = dict(aligned)
        for key in (
            "target_aligned_earring_alpha",
            "target_aligned_left_alpha",
            "target_aligned_right_alpha",
        ):
            aligned_output[key] = object_mask_at_output(aligned[key], soft=True)
        aligned_output["target_aligned_hole_alpha"] = object_mask_at_output(
            aligned["target_aligned_hole_alpha"], soft=False
        )
        aligned_output["target_aligned_earring_rgb"] = object_rgb_at_output(
            aligned["target_aligned_earring_rgb"]
        )
        target_left_gate_output = object_mask_at_output(target_left_gate, soft=False)
        target_right_gate_output = object_mask_at_output(target_right_gate, soft=False)

        # A PP decode can hallucinate label-9 accessory pixels even when the
        # completed transfer did not contain one.  Restore only those detached
        # pixels from the completed transfer, then write the one verified
        # source object below.  This is a normal RGB replacement, never a
        # zero/black mask clear; genuine transfer accessories and the verified
        # source instance remain untouched.
        completed_target = aux.get("authoritative_target_highres_01", aux.get("target_01"))
        if torch.is_tensor(completed_target):
            completed_target = normalized_to_01(completed_target).to(
                device=image_01.device, dtype=image_01.dtype
            )
            if completed_target.shape[-2:] != size:
                completed_target = F.interpolate(
                    completed_target, size=size, mode="bilinear", align_corners=False
                )
            completed_target_parsing = self.parsing_helper.parse(completed_target, out_size=size)
            completed_earring = parsing_label_mask(completed_target_parsing, (RAW_EARRING,))
        else:
            completed_target = image_01
            completed_earring = torch.zeros_like(image_01[:, :1])
        pp_existing_earring = parsing_label_mask(target_parsing_output, (RAW_EARRING,))
        target_lobe_zone = dilate_mask(
            torch.maximum(
                torch.maximum(target_left_gate_output, target_right_gate_output),
                aligned_output["target_aligned_earring_alpha"],
            ),
            17,
        )
        verified_object_zone = dilate_mask(
            aligned_output["target_aligned_earring_alpha"], 3
        )
        pp_duplicate_clear = (
            pp_existing_earring
            * target_lobe_zone
            * (1.0 - completed_earring).clamp(0, 1)
            * (1.0 - verified_object_zone).clamp(0, 1)
        ).clamp(0, 1)
        base_before_object = (
            image_01 * (1.0 - pp_duplicate_clear)
            + completed_target * pp_duplicate_clear
        ).clamp(0, 1)
        result, alpha = composite_earring_v6(
            base_before_object,
            aligned_output,
            target_left_gate_output,
            target_right_gate_output,
            min_visible_area=max(2.0, min_target_area),
        )

        aux["source_native_earring_alpha"] = extracted["source_native_earring_alpha"]
        aux["source_native_earring_rgb"] = extracted["source_native_earring_rgb"]
        aux["source_native_earring_seed"] = native_parser_seed
        aux["source_native_earring_recall_hint"] = native_recall_hint
        aux["source_native_earring_seed_space"] = EarringCoordinateSpace.SOURCE_NATIVE
        aux["target_aligned_earring_alpha"] = aligned_output["target_aligned_earring_alpha"]
        aux["target_aligned_earring_rgb"] = aligned_output["target_aligned_earring_rgb"]
        aux["v6_target_aligned_left_alpha"] = aligned_output["target_aligned_left_alpha"]
        aux["v6_target_aligned_right_alpha"] = aligned_output["target_aligned_right_alpha"]
        aux["v6_target_left_shift_y"] = aligned["left_shift_y"]
        aux["v6_target_left_shift_x"] = aligned["left_shift_x"]
        aux["v6_target_right_shift_y"] = aligned["right_shift_y"]
        aux["v6_target_right_shift_x"] = aligned["right_shift_x"]
        aux["alignment_raw_left_dy"] = aligned["left_raw_shift_y"]
        aux["alignment_raw_left_dx"] = aligned["left_raw_shift_x"]
        aux["alignment_raw_right_dy"] = aligned["right_raw_shift_y"]
        aux["alignment_raw_right_dx"] = aligned["right_raw_shift_x"]
        aux["alignment_valid_left"] = aligned["left_alignment_valid"]
        aux["alignment_valid_right"] = aligned["right_alignment_valid"]
        aux["fallback_zero_shift_used_left"] = aligned["fallback_zero_shift_used_left"]
        aux["fallback_zero_shift_used_right"] = aligned["fallback_zero_shift_used_right"]
        aux["output_source_earring_composite_mask"] = alpha
        aux["output_v19_source_alpha"] = alpha
        aux["output_v19_source_rgb"] = aligned_output["target_aligned_earring_rgb"]
        aux["output_v19_hole_mask"] = aligned_output["target_aligned_hole_alpha"]
        aux["final_hole_alpha"] = aux["output_v19_hole_mask"]
        aux["output_v19_target_left_visible_ear"] = target_left_gate_output
        aux["output_v19_target_right_visible_ear"] = target_right_gate_output
        aux["pp_existing_earring_mask"] = pp_existing_earring.detach()
        aux["pp_duplicate_clear_mask"] = pp_duplicate_clear.detach()
        aux["v6_outside_authorized_write_area"] = alpha.new_zeros(
            alpha.size(0), 1, 1, 1
        )
        return result.clamp(0, 1) * 2.0 - 1.0

    def _compose_final_v5(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """V5 final compositor: immutable completed transfer plus native earring.

        The completed high-resolution transfer owns the full canvas.  PP is
        permitted only in a guarded face interior, never at a hair, ear, or
        background contour.  The earring path is deliberately one-way:
        source-native instance -> one alignment -> target composite.
        """
        image_01 = normalized_to_01(image).clamp(0, 1)
        size = tuple(image_01.shape[-2:])

        def image_at(value: torch.Tensor | None, target_size: tuple[int, int], fallback: torch.Tensor | None = None):
            value = fallback if value is None else value
            if value is None:
                return None
            value = normalized_to_01(value).to(device=image_01.device, dtype=image_01.dtype)
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            return value.clamp(0, 1)

        def mask_at(value: torch.Tensor | None, target_size: tuple[int, int], *, soft: bool = False):
            if value is None:
                return torch.zeros(image_01.size(0), 1, *target_size, device=image_01.device, dtype=image_01.dtype)
            value = ensure_mask_4d(value).to(device=image_01.device, dtype=image_01.dtype)[:, :1]
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear" if soft else "nearest", align_corners=False if soft else None)
            return value.clamp(0, 1)

        def parser_edge_alpha(mask: torch.Tensor) -> torch.Tensor:
            """Recover the parser's native-scale antialiasing for RGB joins.

            FaceParsingHelperV5 emits a 512px discrete label map and expands it
            with nearest-neighbour interpolation.  That map is correct for
            semantic decisions but turns into visible two-pixel stairs when it
            is used directly as a high-resolution RGB composite alpha.  This
            helper is intentionally only for the face/hair colour join: the
            original hard masks still control hair ownership and ear exposure.
            """
            parse_size = max(1, int(getattr(self.parsing_helper, "parse_size", 512)))
            work_size = (min(size[0], parse_size), min(size[1], parse_size))
            value = mask.to(device=image_01.device, dtype=image_01.dtype)
            if value.shape[-2:] != work_size:
                value = F.interpolate(value, size=work_size, mode="area")
            if work_size != size:
                value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            return value.clamp(0, 1)

        # Visibility and geometry decisions belong to the final decoded canvas;
        # the authoritative transfer image is retained for diagnostics only.
        target_rgb = image_01
        target_for_geometry = image_01

        native_source = aux.get("source_full_01")
        if native_source is None:
            native_source = aux.get("source_face_reference_01", aux.get("source_01"))
        native = image_at(
            native_source,
            tuple(native_source.shape[-2:]),
            image_01,
        ) if native_source is not None else image_01
        native = native.to(device=image_01.device, dtype=image_01.dtype)
        native_size = tuple(native.shape[-2:])
        source_parsing = aux.get("source_parsing")
        if source_parsing is None or tuple(ensure_mask_4d(source_parsing).shape[-2:]) != native_size:
            source_parsing_native = self.parsing_helper.parse(native, out_size=native_size)
        else:
            source_parsing_native = ensure_mask_4d(source_parsing).to(device=native.device).long()

        # The final target image owns all face/ear geometry decisions.  The
        # query builder's 256px masks are training/search auxiliaries and must
        # not reopen an ear that high-resolution transferred hair has covered.
        target_parsing_output = self.parsing_helper.parse(target_for_geometry, out_size=size)
        target_face = torch.maximum(
            parsing_label_mask(target_parsing_output, RAW_FACE_SURFACE_LABELS),
            parsing_label_mask(target_parsing_output, RAW_DETAIL_LABELS),
        )
        target_hair_parser = parsing_label_mask(target_parsing_output, (RAW_HAIR,))
        # The completed transfer is the sole owner of its hair geometry.  A
        # 256px HM_X/topology mask is not a valid final-resolution geometry
        # authority: when enlarged it can cover a visibly exposed lobe or move
        # the face/hair cut several pixels off the actual strand boundary.
        # That was the cause of both the black fragmented ear fill and the
        # coarse face-side hair rim.  Keep its use confined to the PP training
        # path; output ownership comes only from this final-image parser.
        # The final parser can miss a solid interior island of transferred hair
        # even when the 256px transfer mask sees it.  Use only the eroded
        # interior of that mask as a supplemental ownership signal.  It cannot
        # move the hairline or paint new RGB, but it prevents PP from replacing
        # an existing hair region with source-conditioned clothing.
        target_hair_hint = mask_at(aux.get("target_hair_mask"), size)
        target_hair_hint_core = erode_mask(
            (target_hair_hint > 0.5).to(image_01.dtype),
            9,
        )
        target_hair_output = torch.maximum(target_hair_parser, target_hair_hint_core)
        if native_size == size:
            target_parsing_native = target_parsing_output
        else:
            target_for_alignment = image_at(image_01, native_size, target_for_geometry)
            target_parsing_native = self.parsing_helper.parse(
                target_for_alignment,
                out_size=native_size,
            )
        # Earring eligibility is intentionally derived from the same final
        # target parse.  Do not let a coarse transfer mask veto a lobe which is
        # visibly open in the completed image.
        target_hair_native = parsing_label_mask(target_parsing_native, (RAW_HAIR,))
        target_left_ear = (
            parsing_label_mask(target_parsing_native, (7,))
            * (1.0 - target_hair_native).clamp(0, 1)
        )
        target_right_ear = (
            parsing_label_mask(target_parsing_native, (8,))
            * (1.0 - target_hair_native).clamp(0, 1)
        )

        # Keep one continuous face repair field.  Source pixels below are used
        # exclusively for the exact earring foreground extractor; they never
        # become a second RGB authority for a subset of facial skin.
        target_hair_binary = (target_hair_output > 0.5).to(image_01.dtype)
        # The transfer itself stays untouched.  Only the two-pixel join to the
        # PP face/background is antialiased, so the binary parser contour
        # cannot leave a staircase or a hard dark rim around otherwise real
        # transferred strands.
        # Do not erode before feathering.  That previous construction removed
        # several pixels of real transferred hair around the whole face, then
        # exposed decoder RGB in the gap as a visibly painted inner rim.  A
        # direct, clipped blur changes only the one/two-pixel antialiasing
        # fringe while all interior transferred strands remain byte-identical.
        target_hair_alpha = torch.minimum(
            target_hair_binary,
            gaussian_blur(target_hair_binary, kernel_size=5, sigma=0.90),
        ).clamp(0, 1)
        target_hair_core = (target_hair_alpha >= 0.995).to(image_01.dtype)
        face_surface = target_face * (1.0 - target_hair_binary).clamp(0, 1)
        # Hair shape and colour transfer are complete before V5 post-process.
        # The completed image is the default final canvas, not merely a source
        # for a hair-mask paste.  The latter split one physical image into PP
        # face/background and transferred hair, which created the visibly
        # straight face-side hairline even though the pre-PP transfer already
        # had a natural boundary.  Keep every hair and surrounding boundary
        # pixel from the completed transfer exactly as it was generated.
        completed_hair = image_at(aux.get("authoritative_target_highres_01"), size)
        base = image_01 if completed_hair is None else completed_hair

        # The decoded PP face and completed transfer have different RGB bases.
        # No feather width can hide that fact: a hard mask makes a staircase,
        # while a broad one makes a pale filter ring.  Match only PP's
        # low-frequency face field to the completed transfer in a thin band
        # immediately inside the final hair boundary, then diffuse that signed
        # correction across the face surface.  PP's pores/features remain its
        # own high-frequency signal; transferred hair/background RGB is never
        # copied into the face.
        face_boundary_band = (
            face_surface * dilate_mask(target_hair_binary, 17)
        ).clamp(0, 1)
        completed_low = gaussian_blur(base, kernel_size=31, sigma=8.0)
        pp_low = gaussian_blur(image_01, kernel_size=31, sigma=8.0)
        face_lowfreq_delta = self._diffuse_signed_face_field(
            (completed_low - pp_low).clamp(-0.35, 0.35),
            face_boundary_band,
            face_surface,
        ).clamp(-0.35, 0.35)
        harmonized_pp_face = (
            image_01 + face_lowfreq_delta * face_surface
        ).clamp(0, 1)

        # This is strictly parser-native antialiasing (roughly one output
        # pixel), not a face-mask feather.  The two RGB fields already agree
        # at the boundary above, so this only removes the parser's 512px stair
        # steps without creating another visible processing band.
        pp_face_alpha = parser_edge_alpha(face_surface)
        base = (
            base * (1.0 - pp_face_alpha)
            + harmonized_pp_face * pp_face_alpha
        ).clamp(0, 1)

        source_skin = mask_at(
            parsing_label_mask(source_parsing_native, RAW_FACE_SURFACE_LABELS),
            size,
        )
        source_detail = mask_at(
            parsing_label_mask(source_parsing_native, RAW_DETAIL_LABELS),
            size,
        )
        source_hair = mask_at(
            parsing_label_mask(source_parsing_native, (RAW_HAIR,)),
            size,
        )
        source_earring = mask_at(
            parsing_label_mask(source_parsing_native, (RAW_EARRING,)),
            size,
        )
        source_uncertain = dilate_mask(source_hair + source_earring, 5)
        source_valid_skin = source_skin * face_surface * (1.0 - source_uncertain).clamp(0, 1)
        source_valid_detail = source_detail * face_surface * (1.0 - source_uncertain).clamp(0, 1)
        revealed = mask_at(aux.get("revealed_skin_mask"), size, soft=True) * face_surface
        face = {
            "face_result": base,
            "face_surface": face_surface,
            "source_valid_skin": source_valid_skin,
            "source_valid_detail": source_valid_detail,
            "source_hair_guard_uncertain": source_uncertain,
            "revealed_skin": revealed,
            "continuous_lowfreq_delta": face_lowfreq_delta,
            "source_highfreq_residual": torch.zeros_like(image_01),
            "face_after_lowfreq": harmonized_pp_face,
            "face_after_detail": image_01,
            "target_hair_binary": target_hair_binary,
            "target_hair_soft_alpha": target_hair_alpha,
            "face_boundary_band": face_boundary_band,
        }

        # Only ``result`` below keeps the generator backward graph.  Saved
        # diagnostics/masks are consumed as fixed loss inputs or visual output;
        # retaining every intermediate face field here needlessly keeps the
        # 1024px convolution graph alive until the batch finishes.
        aux["output_v5_base"] = base.detach()
        aux["output_v5_unified_face"] = face["face_result"].detach()
        aux["output_v5_face_pp_alpha"] = pp_face_alpha.detach()
        aux["output_v5_face_residual_mask"] = face["source_valid_detail"].detach()
        aux["output_v5_face_residual_alpha"] = face["source_valid_detail"].detach()
        for key, value in face.items():
            aux[f"output_v5_face_{key}"] = value.detach()

        # Use the same bounded source-native component extractor as dataset
        # generation.  The older GrabCut foreground path could promote a
        # background component touching an earring into RGB alpha, which is
        # exactly the source-background/transparent-hole failure seen in
        # inference.  Native components retain the real pearl/metal pixels and
        # only graph-connect appearance-compatible pieces from a lobe root.
        native_parser_seed = parsing_label_mask(
            source_parsing_native,
            (RAW_EARRING,),
        ).to(device=native.device, dtype=native.dtype)
        native_recall_hint = torch.zeros_like(native[:, :1])
        for key in (
            "source_native_earring_seed",
            "source_parser_earring_mask",
            "strong_earring_candidate_core",
        ):
            value = aux.get(key)
            if value is not None:
                # These tensors originate from the 256px PP path.  They may
                # find a missed accessory but their enlarged blocks are never
                # legal object alpha at source-native resolution.
                native_recall_hint = torch.maximum(
                    native_recall_hint,
                    mask_at(value, native_size),
                )
        native_extracted = extract_source_native_earring_v6(
            native,
            source_parsing_native,
            source_native_seed=native_parser_seed,
            seed_space=EarringCoordinateSpace.SOURCE_NATIVE,
            source_native_recall_hint=native_recall_hint,
            recall_hint_space=EarringCoordinateSpace.SOURCE_NATIVE,
            max_graph_depth=max(
                1,
                int(getattr(self.args, "earring_component_max_depth", 4)),
            ),
            max_cumulative_cost=max(
                0.1,
                float(getattr(self.args, "earring_component_max_cumulative_cost", 1.85)),
            ),
            allow_long_continuation=False,
        )
        native_alpha = native_extracted["source_native_earring_alpha"]
        extracted = {
            "source_alpha": native_alpha,
            "left_source_alpha": native_extracted["source_native_left_alpha"],
            "right_source_alpha": native_extracted["source_native_right_alpha"],
            "source_rgb": native_extracted["source_native_earring_rgb"],
            "presence_state": native_extracted["source_native_presence_state"],
            "instance_confidence": native_extracted["source_native_presence_score"],
            "localization_roi": native_extracted["localization_core"],
            "foreground_seed": native_extracted["parser_earring_seed"],
            "raw_foreground": native_extracted["grabcut_raw"],
            "boundary_band": (
                dilate_mask(native_alpha, 3) - erode_mask(native_alpha, 3)
            ).clamp(0, 1),
            "hole_mask": native_extracted["source_native_hole_alpha"],
            **native_extracted,
        }
        source_instance = EarringNativeInstanceV6(
            alpha=extracted["source_alpha"],
            rgb=extracted["source_rgb"],
            hole_alpha=extracted["hole_mask"],
            left_alpha=extracted["left_source_alpha"],
            right_alpha=extracted["right_source_alpha"],
            # A hoop hole is intentionally disjoint from the ring alpha.
            # Multiplying it by the ring made every per-side hole zero before
            # alignment, so the source background inside a large hoop could
            # survive the final composite.  Carry the extractor's explicit
            # side holes unchanged and keep those pixels target-owned.
            left_hole_alpha=native_extracted["source_native_left_hole_alpha"],
            right_hole_alpha=native_extracted["source_native_right_hole_alpha"],
        )
        source_left_ear = parsing_label_mask(source_parsing_native, (7,))
        source_right_ear = parsing_label_mask(source_parsing_native, (8,))
        align_shift = max(
            0,
            int(
                round(
                    float(getattr(self.args, "earring_align_max_shift", 16))
                    * max(native_size)
                    / 256.0
                )
            ),
        )
        aligned = align_earring_instance_v6(
            source_instance,
            source_left_ear,
            source_right_ear,
            target_left_ear,
            target_right_ear,
            max_shift=align_shift,
        )
        native_base = F.interpolate(base, size=native_size, mode="bilinear", align_corners=False)
        native_result, native_alpha = composite_earring_v6(
            native_base,
            aligned,
            target_left_ear,
            target_right_ear,
            min_visible_area=max(
                2.0,
                float(getattr(self.args, "min_target_ear_area", 8.0))
                * (max(native_size) / max(1, int(getattr(self.parsing_helper, "parse_size", 512)))) ** 2,
            ),
        )
        result = native_result if native_size == size else F.interpolate(native_result, size=size, mode="bilinear", align_corners=False)
        alpha = native_alpha if native_size == size else F.interpolate(native_alpha, size=size, mode="bilinear", align_corners=False)

        # Coordinate-explicit fields are the only V5 authority/debug data.
        aux["source_native_earring_alpha"] = extracted["source_alpha"]
        aux["source_native_earring_rgb"] = extracted["source_rgb"]
        aux["source_native_earring_seed"] = native_parser_seed
        aux["source_native_earring_recall_hint"] = native_recall_hint
        aux["source_native_earring_seed_space"] = EarringCoordinateSpace.SOURCE_NATIVE
        aux["target_aligned_earring_alpha"] = aligned["target_aligned_earring_alpha"]
        aux["target_aligned_earring_rgb"] = aligned["target_aligned_earring_rgb"]
        aux["output_source_earring_composite_mask"] = alpha
        aux["output_v5_source_alpha"] = alpha.detach()
        aux["output_v5_source_rgb"] = image_at(aligned["target_aligned_earring_rgb"], size, base).detach()
        aux["output_v5_source_detail_mask"] = face["source_valid_detail"].detach()
        for key, value in extracted.items():
            aux[f"output_v5_earring_{key}"] = value
        for key, value in aligned.items():
            aux[f"output_v5_{key}"] = value
        aux["output_v5_target_aligned_earring_alpha"] = alpha.detach()
        aux["output_v5_target_aligned_earring_rgb"] = image_at(aligned["target_aligned_earring_rgb"], size, base).detach()
        aux["output_v5_target_hair_overlap"] = (alpha * face["target_hair_soft_alpha"]).flatten(1).sum(dim=1, keepdim=True).view(-1, 1, 1, 1)
        source_background = (source_parsing_native == 0).to(native.dtype)
        aux["output_v5_source_background_overlap"] = (
            extracted["source_alpha"] * source_background
        ).flatten(1).sum(dim=1, keepdim=True).view(-1, 1, 1, 1)
        write_outside = ((native_result - native_base).abs().amax(dim=1, keepdim=True) > 1e-6).to(native.dtype)
        aux["output_v5_outside_alpha_write_area"] = (
            write_outside * (1.0 - native_alpha)
        ).flatten(1).sum(dim=1, keepdim=True).view(-1, 1, 1, 1)
        aux["output_v5_target_parsing_native"] = target_parsing_native
        aux["output_v5_target_left_visible_ear"] = target_left_ear
        aux["output_v5_target_right_visible_ear"] = target_right_ear
        aux["output_target_hair_preserve_mask"] = face["target_hair_soft_alpha"].detach()
        return result.clamp(0, 1) * 2.0 - 1.0

    def _preserve_target_output_legacy_v18(
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
        # ``earring_reference`` may be the target-aligned, object-only image
        # saved by PP dataset generation.  That tensor is deliberately used by
        # the training losses, but it cannot be interpreted with source parsing
        # or source ear masks.  More importantly, validation receives ``source``
        # at 256px for the PP encoder and stores the native source separately as
        # ``source_full_01`` just before rendering.  The former selection only
        # used that native image when the reference happened to be a dataset
        # tensor, so normal validation silently re-expanded a 256px source for
        # both the instance alpha and its RGB.  A small stud could therefore
        # never retain its original contour in the saved high-resolution result.
        #
        # Prefer an explicitly supplied native source whenever it exists.  In
        # production, ``earring_reference`` is already that source image; for a
        # dataset-aligned reference without ``source_full_01``, the separately
        # supplied source-face reference remains the only valid source frame.
        native_source_value = aux.get("source_full_01")
        native_source_is_fullres = native_source_value is not None
        if native_source_value is None:
            source_face_reference = aux.get("source_face_reference_01")
            source_01 = aux.get("source_01")
            # Production passes the source face explicitly rather than through
            # ``source_full_01``.  Treat it as native only when it really is
            # larger than the PP encoder input; the default 256px fallback
            # must retain the existing dataset-coordinate selection below.
            if (
                source_face_reference is not None
                and source_01 is not None
                and source_face_reference.shape[-2] > source_01.shape[-2]
                and source_face_reference.shape[-1] > source_01.shape[-1]
            ):
                native_source_value = source_face_reference
                native_source_is_fullres = True
            else:
                native_source_value = source_face_reference
        native_source_reference = (
            None
            if native_source_value is None
            else resize_rgb(normalized_to_01(native_source_value), mode="bilinear")
        )
        dataset_reference_gate = aux.get("earring_reference_is_dataset")
        native_reference_gate = torch.zeros_like(image_01[:, :1])
        if native_source_reference is None:
            locator_reference = earring_reference
        elif native_source_is_fullres:
            # Validation's full-size source is source-coordinate RGB for every
            # sample, regardless of whether it also carries a dataset label.
            locator_reference = native_source_reference
            native_reference_gate = torch.ones_like(image_01[:, :1])
        elif dataset_reference_gate is not None:
            dataset_reference_gate = self._per_sample_flag(
                dataset_reference_gate,
                image_01,
                name="earring_reference_is_dataset",
            )
            locator_reference = (
                native_source_reference * dataset_reference_gate
                + earring_reference * (1.0 - dataset_reference_gate)
            ).clamp(0, 1)
            native_reference_gate = dataset_reference_gate.expand_as(image_01[:, :1])
        else:
            locator_reference = earring_reference

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
        highres_native_instance = torch.zeros_like(earring_edit)
        highres_native_parser_instance = torch.zeros_like(earring_edit)
        highres_native_visual_recall = torch.zeros_like(earring_edit)
        highres_refined_instance = torch.zeros_like(earring_edit)
        highres_refined_hole = torch.zeros_like(earring_edit)
        highres_connector = torch.zeros_like(earring_edit)
        highres_interior_authority = torch.zeros_like(earring_edit)
        highres_observed_instance = torch.zeros_like(earring_edit)
        source_earring_presence_gate = torch.zeros_like(earring_edit)
        target_side_earring_gate = torch.zeros_like(earring_edit)
        native_source_parser_earring = torch.zeros_like(earring_edit)
        native_source_parser_is_fullres = torch.zeros_like(earring_edit)
        # The native-resolution earring compositor is intentionally an
        # inference/validation operation.  Its masks are built from detached
        # source pixels and OpenCV, so running it during every training batch
        # cannot contribute gradients.  Keeping it out of the training
        # forward removes the large CPU synchronisation cost without changing
        # the learned PP path or final inference output.
        enable_highres_output = bool(
            getattr(self.args, "enable_highres_earring_output_refine", True)
        ) and not self.training
        if locator_reference is not None and enable_highres_output:
            # ``aux["source_parsing"]`` is the 256px PP parser result.  It is
            # sufficient for the learned feature branch, but it cannot be
            # reused for source-native earring extraction: upsampling that
            # coarse label removes a stud outright and truncates a pendant to
            # the few pixels that survived at 256px.  Parse the actual source
            # RGB in the same output frame used by the native compositor.
            #
            # This is deliberately inference-only.  The native OpenCV path is
            # already excluded from training because it has no gradients; this
            # change gives it the correct evidence rather than changing PP
            # losses or broadening any target-side write permission.
            source_parsing_for_native = aux.get("source_parsing")
            source_hair_for_native = aux.get("source_hair_mask")
            if native_source_is_fullres:
                source_parsing_for_native = self.parsing_helper.parse(
                    locator_reference,
                    out_size=tuple(locator_reference.shape[-2:]),
                )
                source_hair_for_native = parsing_label_mask(
                    source_parsing_for_native,
                    (RAW_HAIR,),
                )
                native_source_parser_is_fullres = torch.ones_like(earring_edit)
            if source_parsing_for_native is not None:
                native_source_parser_earring = parsing_label_mask(
                    source_parsing_for_native,
                    (RAW_EARRING,),
                ).to(device=earring_edit.device, dtype=earring_edit.dtype)
            # Geometry alone is not source-earring evidence.  Without this
            # gate, a curved grass or hair edge beside an ear can be fitted as
            # a hoop and copied to an image whose source has no accessory.
            # These are source-aligned, independently accepted low-resolution
            # seeds; they guide high-resolution localisation but are never
            # composited at their blocky low-resolution shape.
            reliable_source_seed = torch.zeros_like(earring_edit)
            for key in (
                "source_parser_earring_mask",
                "strong_earring_candidate_core",
            ):
                value = aux.get(key)
                if value is not None:
                    reliable_source_seed = torch.maximum(
                        reliable_source_seed,
                        resize_earring_mask(value),
                    )
            no_earring = aux.get("no_earring_case_mask")
            source_case_allowed = torch.ones_like(earring_edit)
            if no_earring is not None:
                source_case_allowed = (
                    1.0 - resize_earring_mask(no_earring)
                ).clamp(0, 1)
            source_instances = build_source_earring_instance_masks_v5(
                locator_reference,
                source_parsing_for_native,
                source_hair_mask=source_hair_for_native,
                source_seed_mask=reliable_source_seed,
            )
            # During validation/training the dataset carries a verified
            # source-native object mask.  Reuse that exact mask as the native
            # instance authority instead of running a second visual detector
            # that can choose a nearby strand or shorten a low-contrast
            # pendant.  The side split is source-lobe based and the mask is
            # still subject to the target-side visibility gates below.
            exact_native = aux.get("source_earring_object_mask")
            exact_verified = aux.get("source_earring_object_mask_verified_native")
            if exact_native is not None and exact_verified is not None:
                exact_native = resize_earring_mask(exact_native).to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                )
                exact_native = exact_native * (1.0 - resize_earring_mask(source_hair_for_native)).clamp(0, 1)
                exact_left, exact_right = assign_components_to_ear_sides(
                    exact_native,
                    source_instances["left_context"],
                    source_instances["right_context"],
                    source_instances["left_lobe_anchor"],
                    source_instances["right_lobe_anchor"],
                )
                verified_gate = exact_verified.to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                )
                if verified_gate.ndim == 1:
                    verified_gate = verified_gate.view(-1, 1, 1, 1)
                else:
                    verified_gate = ensure_mask_4d(verified_gate).flatten(1).amax(
                        dim=1,
                        keepdim=True,
                    ).view(-1, 1, 1, 1)
                source_instances["left_instance_mask"] = torch.where(
                    verified_gate > 0.5,
                    exact_left,
                    source_instances["left_instance_mask"],
                )
                source_instances["right_instance_mask"] = torch.where(
                    verified_gate > 0.5,
                    exact_right,
                    source_instances["right_instance_mask"],
                )
                source_instances["instance_mask"] = torch.clamp(
                    source_instances["left_instance_mask"]
                    + source_instances["right_instance_mask"],
                    0,
                    1,
                )
                source_instances["left_parser_instance_mask"] = source_instances["left_instance_mask"]
                source_instances["right_parser_instance_mask"] = source_instances["right_instance_mask"]
            highres_native_instance = source_instances["instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            highres_native_parser_instance = torch.clamp(
                source_instances["left_parser_instance_mask"].to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                )
                + source_instances["right_parser_instance_mask"].to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                ),
                0,
                1,
            )
            highres_native_visual_recall = source_instances.get(
                "visual_recall_instance_mask",
                torch.zeros_like(earring_edit),
            ).to(device=earring_edit.device, dtype=earring_edit.dtype)
            # Presence is per source ear.  Split the low-resolution evidence
            # in the real source-side corridors before it is upsampled; using
            # a shared broad ROI let a genuine earring on one side unlock a
            # second fabricated earring on the opposite exposed ear.
            left_source_seed = reliable_source_seed * source_instances["left_context"]
            right_source_seed = reliable_source_seed * source_instances["right_context"]
            # The source instance builder has already associated these parser
            # components with a real lobe.  Preserve them as per-side evidence
            # as well as the compact seed: a true long pendant or outer hoop
            # arc often lives beyond the small corridor used to initialise the
            # low-resolution seed, and discarding it here was a direct cause
            # of short arcs and one-sided misses.
            left_parser_evidence = source_instances["left_parser_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            right_parser_evidence = source_instances["right_parser_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            # A parser-missed ordinary accessory is allowed to activate only
            # after the native locator has accepted a compact, source-lobe
            # connected visual instance.  This is deliberately separate from
            # hoop evidence: a visual ordinary-instance candidate must never
            # authorize a contour search over nearby grass, hair or background.
            left_native_recall = source_instances.get(
                "left_visual_recall_instance_mask",
                torch.zeros_like(earring_edit),
            ).to(device=earring_edit.device, dtype=earring_edit.dtype)
            right_native_recall = source_instances.get(
                "right_visual_recall_instance_mask",
                torch.zeros_like(earring_edit),
            ).to(device=earring_edit.device, dtype=earring_edit.dtype)
            left_visual_hoop_seed = source_instances.get(
                "left_visual_recall_hoop_seed_mask",
                torch.zeros_like(earring_edit),
            ).to(device=earring_edit.device, dtype=earring_edit.dtype)
            right_visual_hoop_seed = source_instances.get(
                "right_visual_recall_hoop_seed_mask",
                torch.zeros_like(earring_edit),
            ).to(device=earring_edit.device, dtype=earring_edit.dtype)
            # A low-resolution strong proposal is a locator hint only.  It is
            # deliberately excluded from hoop authority: a textured grass or
            # hair edge can satisfy that proposal but cannot prove a native
            # closed ring.  Parser pixels and a closed-loop native visual seed
            # are the only source evidence allowed to start the hoop verifier.
            left_hoop_evidence = torch.clamp(
                left_parser_evidence + left_visual_hoop_seed,
                0,
                1,
            )
            right_hoop_evidence = torch.clamp(
                right_parser_evidence + right_visual_hoop_seed,
                0,
                1,
            )
            left_source_evidence = torch.clamp(
                left_hoop_evidence + left_native_recall,
                0,
                1,
            )
            right_source_evidence = torch.clamp(
                right_hoop_evidence + right_native_recall,
                0,
                1,
            )
            # Source extraction and target visibility live in different image
            # coordinates.  A target-side *map* must never crop a source-side
            # object before alignment: that was cutting real hoops into short
            # arcs whenever the two parsers disagreed by a few pixels.  Reduce
            # target visibility to a per-side decision here, then apply only
            # source-native masks during instance extraction.
            def side_is_open(value: torch.Tensor | None) -> torch.Tensor:
                """Return a scalar target-side visibility decision per sample."""

                value = resize_earring_mask(value)
                return (
                    value.flatten(1).amax(dim=1, keepdim=True) > 0.5
                ).to(earring_edit.dtype).view(-1, 1, 1, 1)

            def side_has_source_evidence(
                seed: torch.Tensor,
                instance: torch.Tensor,
                bootstrap_seed: torch.Tensor,
            ) -> torch.Tensor:
                """Require independent source evidence for one ear side.

                All inputs are in the source frame.  ``bootstrap_seed`` is an
                accepted, lobe-associated low-resolution locator result.  It
                is allowed to open high-resolution *inspection* when the
                parser only retained a tiny rim (or nothing), but is never
                source RGB authority: the native refiner below must still
                return a measured object before anything can be composited.
                The return value is intentionally a scalar, so it may decide
                whether the target side is eligible without ever spatially
                clipping the source object before alignment.
                """

                seed_present = (
                    seed.flatten(1).sum(dim=1, keepdim=True) >= side_seed_min_area
                )
                instance_present = (
                    torch.clamp(instance + bootstrap_seed, 0, 1)
                    .flatten(1)
                    .sum(dim=1, keepdim=True)
                    >= side_seed_min_area
                )
                return (seed_present & instance_present).to(earring_edit.dtype).view(
                    -1, 1, 1, 1
                )

            # This scalar only opens native inspection for an already
            # source-lobe-associated candidate.  Area/shape verification
            # happens in the native extractor, so resolution-scaled coarse
            # label area here must not suppress a real one-pixel stud seed.
            side_seed_min_area = 1.0
            left_accepted_instance = source_instances["left_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            right_accepted_instance = source_instances["right_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            )
            # The native extractor associates this seed with a concrete source
            # lobe and rejects unrelated image-half proposals.  Keep it split
            # by that same side context.  It can bootstrap native-resolution
            # object extraction but must never be pasted at its blocky seed
            # shape, which is what previously produced background patches.
            accepted_presence_seed = source_instances.get(
                "locator_presence_seed",
                torch.zeros_like(earring_edit),
            ).to(device=earring_edit.device, dtype=earring_edit.dtype)
            left_bootstrap_seed = accepted_presence_seed * source_instances[
                "left_context"
            ].to(device=earring_edit.device, dtype=earring_edit.dtype)
            right_bootstrap_seed = accepted_presence_seed * source_instances[
                "right_context"
            ].to(device=earring_edit.device, dtype=earring_edit.dtype)
            left_source_evidence = torch.clamp(
                left_source_evidence + left_bootstrap_seed,
                0,
                1,
            )
            right_source_evidence = torch.clamp(
                right_source_evidence + right_bootstrap_seed,
                0,
                1,
            )
            left_source_present = side_has_source_evidence(
                left_source_evidence,
                left_accepted_instance,
                left_bootstrap_seed,
            )
            right_source_present = side_has_source_evidence(
                right_source_evidence,
                right_accepted_instance,
                right_bootstrap_seed,
            )
            # ``no_earring_case_mask`` is a sample-level source decision.  Keep
            # it scalar as well, rather than letting its 256px representation
            # act as a target-space crop on a native-resolution source object.
            # ``no_earring_case_mask`` is a hard negative from the source
            # frame.  Native visual/structured fallbacks may recover a parser
            # miss, but they must never override an explicit no-earring case;
            # doing so pasted bright source hair/background into the target.
            source_case_present = side_is_open(source_case_allowed)
            left_target_open = side_is_open(
                aux.get("left_target_side_open", aux.get("left_side_active"))
            )
            right_target_open = side_is_open(
                aux.get("right_target_side_open", aux.get("right_side_active"))
            )
            left_active_gate = left_source_present * left_target_open * source_case_present
            right_active_gate = right_source_present * right_target_open * source_case_present
            # Expand only after the source/target eligibility decision has been
            # made.  The expanded form is debug-friendly; all source extraction
            # below consumes the scalar gates above.
            left_active = left_active_gate.expand_as(earring_edit)
            right_active = right_active_gate.expand_as(earring_edit)
            source_earring_presence_gate = torch.clamp(left_active + right_active, 0, 1)
            # The learned 256px completion mask is spatial, unlike the native
            # instance-presence scalar above.  Keep its permission tied to the
            # matching target-ear corridor; otherwise a real left-side object
            # could authorize a spurious low-resolution write at the exposed
            # right ear (or vice versa).
            left_target_corridor = resize_earring_mask(
                aux.get("left_earring_valid_roi", aux.get("left_ear_roi"))
            )
            right_target_corridor = resize_earring_mask(
                aux.get("right_earring_valid_roi", aux.get("right_ear_roi"))
            )
            target_side_earring_gate = torch.clamp(
                left_target_corridor * left_active_gate
                + right_target_corridor * right_active_gate,
                0,
                1,
            )
            # This is a ring-only instance extractor.  It accepts only a
            # source-native pair of complete inner/outer contours and returns
            # their annular alpha plus a separate target-owned inner hole.
            # It is deliberately independent of the coarse source seed: a
            # parser-missed hoop may be absent at 256px, but a no-earring
            # source still cannot pass the paired-contour validation.
            source_parsing = source_parsing_for_native
            source_ear = (
                parsing_label_mask(source_parsing, RAW_EAR_SURFACE_LABELS)
                if source_parsing is not None
                else None
            )
            # The compact locator above is deliberately conservative.  Run a
            # second source-native pass to recover an ordinary stud, pendant
            # or suspension wire that the parser reduced to a few pixels.  Its
            # seed can initialise the segmentation but cannot be returned as
            # RGB by itself (see ``refine_earring_instances_highres``).
            highres_refined = refine_earring_instances_highres(
                locator_reference,
                source_parsing,
                # A lobe-associated coarse seed may initialise native
                # inspection when parser/visual extraction retained only the
                # attachment fragment.  ``refine_earring_instances_highres``
                # returns only source-measured pixels after its structural
                # checks; the seed itself never reaches RGB compositing.
                torch.clamp(
                    source_instances["instance_mask"] + accepted_presence_seed,
                    0,
                    1,
                ),
                source_instances["locator_roi"],
                source_instances["left_lobe_anchor"],
                source_instances["right_lobe_anchor"],
                left_active_gate,
                right_active_gate,
                source_hair_mask=source_hair_for_native,
                source_ear_mask=source_ear,
            )
            refined_left = highres_refined["left_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * left_active_gate
            refined_right = highres_refined["right_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * right_active_gate
            highres_refined_instance = torch.clamp(refined_left + refined_right, 0, 1)
            refined_left_hole = highres_refined["left_hoop_hole_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * left_active_gate
            refined_right_hole = highres_refined["right_hoop_hole_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * right_active_gate
            # The parser-miss visual-recall branch is deliberately an
            # ordinary-solid candidate.  Do not let the generic flood-fill
            # topology pass reinterpret a noisy interior as a hoop hole; true
            # hollow accessories are authorized only by the paired-contour
            # verifier below.
            left_visual_recall_mask = source_instances.get(
                "left_visual_recall_instance_mask",
                torch.zeros_like(earring_edit),
            ).to(device=earring_edit.device, dtype=earring_edit.dtype)
            right_visual_recall_mask = source_instances.get(
                "right_visual_recall_instance_mask",
                torch.zeros_like(earring_edit),
            ).to(device=earring_edit.device, dtype=earring_edit.dtype)
            refined_left_hole = refined_left_hole * (
                1.0 - dilate_mask(left_visual_recall_mask, 3)
            ).clamp(0, 1)
            refined_right_hole = refined_right_hole * (
                1.0 - dilate_mask(right_visual_recall_mask, 3)
            ).clamp(0, 1)
            highres_refined_hole = torch.clamp(refined_left_hole + refined_right_hole, 0, 1)
            refined_left_connector = highres_refined["left_connector_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * left_active_gate
            refined_right_connector = highres_refined["right_connector_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * right_active_gate
            # A hoop verifier is never a detector by itself.  It receives only
            # the accepted source evidence for the matching side; a curved
            # grass/background contour therefore cannot manufacture an earring.
            highres_geometry = {
                "left_elliptical_hoop": torch.zeros_like(earring_edit),
                "right_elliptical_hoop": torch.zeros_like(earring_edit),
                "left_elliptical_hoop_hole": torch.zeros_like(earring_edit),
                "right_elliptical_hoop_hole": torch.zeros_like(earring_edit),
                "left_elliptical_hoop_footprint": torch.zeros_like(earring_edit),
                "right_elliptical_hoop_footprint": torch.zeros_like(earring_edit),
                "left_elliptical_hoop_connector": torch.zeros_like(earring_edit),
                "right_elliptical_hoop_connector": torch.zeros_like(earring_edit),
            }
            hoop_evidence_gate = torch.clamp(
                left_hoop_evidence + right_hoop_evidence,
                0,
                1,
            )
            if bool(hoop_evidence_gate.flatten(1).amax().item() > 0):
                highres_geometry = refine_earring_hoops_highres(
                    locator_reference,
                    source_instances["locator_ring_support"],
                    source_instances["left_lobe_anchor"],
                    source_instances["right_lobe_anchor"],
                    left_source_evidence=left_hoop_evidence * left_active_gate,
                    right_source_evidence=right_hoop_evidence * right_active_gate,
                    source_hair_mask=aux.get("source_hair_mask"),
                    source_ear_mask=source_ear,
                    detection_size=max(locator_reference.shape[-2:]),
                    min_axis=4.0,
                    min_coverage=0.36,
                )
            left_geometry_trace = highres_geometry["left_elliptical_hoop"] * left_active_gate
            right_geometry_trace = highres_geometry["right_elliptical_hoop"] * right_active_gate
            left_geometry_hole = highres_geometry["left_elliptical_hoop_hole"] * left_active_gate
            right_geometry_hole = highres_geometry["right_elliptical_hoop_hole"] * right_active_gate
            left_geometry_connector = highres_geometry.get(
                "left_elliptical_hoop_connector", torch.zeros_like(left_geometry_trace)
            ) * left_active_gate
            right_geometry_connector = highres_geometry.get(
                "right_elliptical_hoop_connector", torch.zeros_like(right_geometry_trace)
            ) * right_active_gate
            left_geometry_footprint = highres_geometry.get(
                "left_elliptical_hoop_footprint",
                torch.zeros_like(left_geometry_trace),
            ) * left_active_gate
            right_geometry_footprint = highres_geometry.get(
                "right_elliptical_hoop_footprint",
                torch.zeros_like(right_geometry_trace),
            ) * right_active_gate
            highres_geometry_trace = torch.clamp(left_geometry_trace + right_geometry_trace, 0, 1)
            highres_geometry_hole = torch.clamp(left_geometry_hole + right_geometry_hole, 0, 1)
            highres_geometry_footprint = torch.clamp(
                left_geometry_footprint + right_geometry_footprint,
                0,
                1,
            )

            def choose_regular_instance(
                base: torch.Tensor,
                refined: torch.Tensor,
                side_seed: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Keep the observed instance and add only a verified continuation.

                The parser-first source locator is the established path that
                recovers solid studs and pendants.  Replacing it wholesale
                with a GrabCut proposal made a real earring turn into source
                background fragments, but treating any non-empty parser result
                as final also reduced an under-segmented pendant or hoop to a
                dot.  The high-resolution result can therefore *extend* the
                base only when it is continuous with it and remains within the
                small object budget already enforced by the native extractor.
                """

                base_area = base.flatten(1).sum(dim=1, keepdim=True)
                refined_area = refined.flatten(1).sum(dim=1, keepdim=True)
                seed_overlap = (
                    refined * dilate_mask(side_seed, 3)
                ).flatten(1).sum(dim=1, keepdim=True)
                base_overlap = (
                    refined * dilate_mask(base, 5)
                ).flatten(1).sum(dim=1, keepdim=True)
                # The native refinement has its own local maximum-area guard.
                # This second guard only rejects a grab-cut spill that is far
                # larger than both the observed component and a normal compact
                # accessory at the output resolution.
                output_area = float(earring_edit.shape[-2] * earring_edit.shape[-1])
                compact_budget = torch.maximum(
                    # ``refined`` has already passed source-lobe connection,
                    # foreground, contour and area checks at native
                    # resolution.  The former stud-sized budget discarded a
                    # verified large pendant/ornament and fell back to its
                    # thin parser edge.  Allow a realistic accessory body,
                    # while retaining an image-relative cap against any
                    # accidental background component.
                    base_area * 16.0,
                    torch.full_like(
                        base_area,
                        1024.0 * (max(earring_edit.shape[-2:]) / 256.0) ** 2,
                    ),
                )
                compact_budget = torch.minimum(
                    compact_budget,
                    torch.full_like(base_area, 0.055 * output_area),
                )
                use_refined_fallback = (
                    (base_area < 1.0)
                    &
                    (refined_area >= 1.0)
                    & (seed_overlap >= 1.0)
                )
                use_refined_continuation = (
                    (base_area >= 1.0)
                    & (refined_area >= 1.0)
                    & (base_overlap >= 1.0)
                    & (refined_area <= compact_budget)
                )
                use_refined = (
                    use_refined_fallback | use_refined_continuation
                ).view(-1, 1, 1, 1)
                return torch.clamp(base + refined * use_refined, 0, 1), use_refined

            left_base_instance = source_instances["left_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * left_active_gate
            right_base_instance = source_instances["right_instance_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * right_active_gate
            left_regular_instance, left_use_refined = choose_regular_instance(
                left_base_instance,
                refined_left,
                left_source_seed,
            )
            right_regular_instance, right_use_refined = choose_regular_instance(
                right_base_instance,
                refined_right,
                right_source_seed,
            )

            def verified_regular_hole(
                instance: torch.Tensor,
                candidate: torch.Tensor,
            ) -> torch.Tensor:
                # A small hoop that missed the stricter paired-contour branch
                # must still keep a truly enclosed centre target-owned.  Do not
                # treat every accidental enclosed parser blob (a stud, gem or
                # ear fold) as a hoop: require a meaningful annular area ratio.
                closed = compute_earring_hole_mask(dilate_mask(instance, 3))
                object_area = instance.flatten(1).sum(dim=1, keepdim=True)
                hole_area = closed.flatten(1).sum(dim=1, keepdim=True)
                annular_ratio = hole_area / (object_area + hole_area).clamp_min(1.0)
                topology_gate = (
                    (hole_area >= 4.0)
                    & (annular_ratio >= 0.06)
                    & (annular_ratio <= 0.72)
                ).to(instance.dtype).view(-1, 1, 1, 1)
                return candidate * closed * topology_gate

            left_base_hole = source_instances["left_hoop_hole_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * left_active
            right_base_hole = source_instances["right_hoop_hole_mask"].to(
                device=earring_edit.device,
                dtype=earring_edit.dtype,
            ) * right_active
            left_regular_hole = verified_regular_hole(
                left_regular_instance,
                torch.where(
                    left_use_refined,
                    torch.maximum(refined_left_hole, left_base_hole),
                    left_base_hole,
                ),
            )
            right_regular_hole = verified_regular_hole(
                right_regular_instance,
                torch.where(
                    right_use_refined,
                    torch.maximum(refined_right_hole, right_base_hole),
                    right_base_hole,
                ),
            )
            left_base_connector = (
                source_instances["left_parser_instance_mask"].to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                )
                * dilate_mask(source_instances["left_lobe_anchor"], 5)
                * left_active_gate
            )
            right_base_connector = (
                source_instances["right_parser_instance_mask"].to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                )
                * dilate_mask(source_instances["right_lobe_anchor"], 5)
                * right_active_gate
            )
            left_regular_connector = torch.where(
                left_use_refined,
                torch.maximum(refined_left_connector, left_base_connector),
                left_base_connector,
            ) * left_regular_instance
            right_regular_connector = torch.where(
                right_use_refined,
                torch.maximum(refined_right_connector, right_base_connector),
                right_base_connector,
            ) * right_regular_instance

            # A complete contour-verified hoop owns its side exclusively.  Do
            # not union it with generic/GrabCut output, which is how duplicate
            # rings, thick halos and black fragments were reintroduced.
            left_geometry_present = (
                left_geometry_trace.flatten(1).sum(dim=1, keepdim=True) >= 1.0
            ).view(-1, 1, 1, 1)
            right_geometry_present = (
                right_geometry_trace.flatten(1).sum(dim=1, keepdim=True) >= 1.0
            ).view(-1, 1, 1, 1)

            def regular_body_overrides_hoop(
                regular: torch.Tensor,
                geometry_footprint: torch.Tensor,
            ) -> torch.Tensor:
                """Keep a proven solid ornament out of the hoop-only path.

                The hoop tracer correctly returns just an annulus.  A large
                solid disc can nevertheless contain nested decorative contours
                and satisfy that tracer, at which point the old unconditional
                preference for geometry erased the disc body and left only its
                rim.  A regular instance that already covers most of the same
                measured footprint and has no enclosed topology is stronger
                evidence for a solid source object, so it keeps RGB authority.
                """

                footprint_area = geometry_footprint.flatten(1).sum(dim=1, keepdim=True)
                regular_inside = (regular * geometry_footprint).flatten(1).sum(
                    dim=1,
                    keepdim=True,
                )
                body_coverage = regular_inside / footprint_area.clamp_min(1.0)
                regular_hole = compute_earring_hole_mask(dilate_mask(regular, 3))
                hole_coverage = (
                    regular_hole * geometry_footprint
                ).flatten(1).sum(dim=1, keepdim=True) / footprint_area.clamp_min(1.0)
                minimum_footprint = max(
                    12.0,
                    3.0 * (max(earring_edit.shape[-2:]) / 256.0) ** 2,
                )
                return (
                    (footprint_area >= minimum_footprint)
                    & (body_coverage >= 0.62)
                    & (hole_coverage <= 0.08)
                ).view(-1, 1, 1, 1)

            left_regular_is_solid = regular_body_overrides_hoop(
                left_regular_instance,
                left_geometry_footprint,
            )
            right_regular_is_solid = regular_body_overrides_hoop(
                right_regular_instance,
                right_geometry_footprint,
            )
            left_use_geometry = left_geometry_present & ~left_regular_is_solid
            right_use_geometry = right_geometry_present & ~right_regular_is_solid
            # A verified contour owns the hoop annulus, but it can miss the
            # tiny source-observed hook between the lobe and the ring.  Keep
            # only that connector from the regular path; never union the
            # complete regular proposal with the hoop, which would recreate
            # thick halos or a second earring.
            left_geometry_instance = torch.clamp(
                left_geometry_trace + left_regular_connector,
                0,
                1,
            )
            right_geometry_instance = torch.clamp(
                right_geometry_trace + right_regular_connector,
                0,
                1,
            )
            left_instance = torch.where(
                left_use_geometry,
                left_geometry_instance,
                left_regular_instance,
            )
            right_instance = torch.where(
                right_use_geometry,
                right_geometry_instance,
                right_regular_instance,
            )
            left_hole = torch.where(left_use_geometry, left_geometry_hole, left_regular_hole)
            right_hole = torch.where(right_use_geometry, right_geometry_hole, right_regular_hole)
            left_connector = torch.where(
                left_use_geometry,
                torch.maximum(left_geometry_connector, left_regular_connector),
                left_regular_connector,
            )
            right_connector = torch.where(
                right_use_geometry,
                torch.maximum(right_geometry_connector, right_regular_connector),
                right_regular_connector,
            )
            # Parser pixels are a source-side locator, not permission to paste
            # an earlobe-shaped low-resolution block into the target ear.  The
            # native refiner returns a connector only after it has verified a
            # compact in-ear stud or a measured attachment at source resolution.
            # Keep that exact alpha separate from the parser connector used for
            # regular/hoop association above.
            left_interior_authority = torch.where(
                left_use_geometry,
                left_geometry_connector,
                refined_left_connector * left_use_refined,
            ) * left_instance
            right_interior_authority = torch.where(
                right_use_geometry,
                right_geometry_connector,
                refined_right_connector * right_use_refined,
            ) * right_instance
            # Keep the raw parser-supported subset separately.  It is the only
            # ordinary object evidence that may overlap an ear interior after
            # alignment; the broader refined/geometry alpha stays outside the
            # ear to prevent copied ear folds and shadows.
            left_observed_instance = torch.clamp(
                source_instances["left_parser_instance_mask"].to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                )
                + source_instances.get(
                    "left_visual_recall_attachment_mask",
                    torch.zeros_like(earring_edit),
                ).to(device=earring_edit.device, dtype=earring_edit.dtype),
                0,
                1,
            ) * left_active_gate
            right_observed_instance = torch.clamp(
                source_instances["right_parser_instance_mask"].to(
                    device=earring_edit.device,
                    dtype=earring_edit.dtype,
                )
                + source_instances.get(
                    "right_visual_recall_attachment_mask",
                    torch.zeros_like(earring_edit),
                ).to(device=earring_edit.device, dtype=earring_edit.dtype),
                0,
                1,
            ) * right_active_gate
            highres_instance = torch.clamp(left_instance + right_instance, 0, 1)
            highres_instance_hole = torch.clamp(left_hole + right_hole, 0, 1)
            highres_connector = torch.clamp(left_connector + right_connector, 0, 1)

            # Dataset generation aligns the source object, RGB and hole per
            # ear.  Final inference must use the same coordinate contract;
            # otherwise an old location can survive while a second earring is
            # pasted at the shifted location.
            output_scale = max(locator_reference.shape[-2:]) / 256.0
            alignment = align_earring_reference_to_target(
                locator_reference,
                highres_instance,
                resize_earring_mask(aux.get("source_left_ear_mask")),
                resize_earring_mask(aux.get("source_right_ear_mask")),
                resize_earring_mask(aux.get("target_left_ear_mask")),
                resize_earring_mask(aux.get("target_right_ear_mask")),
                source_instances["left_context"],
                source_instances["right_context"],
                target_left_roi=resize_earring_mask(aux.get("left_ear_roi")),
                target_right_roi=resize_earring_mask(aux.get("right_ear_roi")),
                max_vertical_shift=max(
                    0,
                    int(round(float(getattr(self.args, "earring_align_max_shift", 16)) * output_scale)),
                ),
                max_horizontal_shift=max(
                    0,
                    int(round(float(getattr(self.args, "earring_align_max_shift", 16)) * output_scale / 2.0)),
                ),
                reference_base=target_authority,
                source_left_instance_mask=left_instance,
                source_right_instance_mask=right_instance,
                source_left_hole_mask=left_hole,
                source_right_hole_mask=right_hole,
            )
            earring_reference = alignment["earring_reference"]
            highres_instance = alignment["earring_confident_mask"]
            highres_instance_hole = alignment["hoop_hole_mask"]
            highres_connector = torch.clamp(
                shift_tensor_per_batch(
                    left_connector,
                    alignment["left_earring_shift_y"].view(-1),
                    alignment["left_earring_shift_x"].view(-1),
                )
                + shift_tensor_per_batch(
                    right_connector,
                    alignment["right_earring_shift_y"].view(-1),
                    alignment["right_earring_shift_x"].view(-1),
                ),
                0,
                1,
            ) * highres_instance
            highres_interior_authority = torch.clamp(
                shift_tensor_per_batch(
                    left_interior_authority,
                    alignment["left_earring_shift_y"].view(-1),
                    alignment["left_earring_shift_x"].view(-1),
                )
                + shift_tensor_per_batch(
                    right_interior_authority,
                    alignment["right_earring_shift_y"].view(-1),
                    alignment["right_earring_shift_x"].view(-1),
                ),
                0,
                1,
            ) * highres_instance
            highres_observed_instance = torch.clamp(
                shift_tensor_per_batch(
                    left_observed_instance,
                    alignment["left_earring_shift_y"].view(-1),
                    alignment["left_earring_shift_x"].view(-1),
                )
                + shift_tensor_per_batch(
                    right_observed_instance,
                    alignment["right_earring_shift_y"].view(-1),
                    alignment["right_earring_shift_x"].view(-1),
                ),
                0,
                1,
            ) * highres_instance
            # Keep the topology-verifier output in the same target frame as
            # the final RGB instance.  A completed hoop can legitimately span
            # the cheek-side of an exposed ear; the generic face guard below
            # must not cut that already verified annulus back to a short arc.
            highres_geometry_trace = torch.clamp(
                shift_tensor_per_batch(
                    left_geometry_trace,
                    alignment["left_earring_shift_y"].view(-1),
                    alignment["left_earring_shift_x"].view(-1),
                )
                + shift_tensor_per_batch(
                    right_geometry_trace,
                    alignment["right_earring_shift_y"].view(-1),
                    alignment["right_earring_shift_x"].view(-1),
                ),
                0,
                1,
            ) * highres_instance
            highres_geometry_hole = torch.clamp(
                shift_tensor_per_batch(
                    left_geometry_hole,
                    alignment["left_earring_shift_y"].view(-1),
                    alignment["left_earring_shift_x"].view(-1),
                )
                + shift_tensor_per_batch(
                    right_geometry_hole,
                    alignment["right_earring_shift_y"].view(-1),
                    alignment["right_earring_shift_x"].view(-1),
                ),
                0,
                1,
            )
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
        # A direct source RGB object may not reopen a target ear based on an
        # upsampled parser fragment.  ``highres_interior_authority`` is the
        # source-native, lobe-linked alpha returned by the compact-stud/connector
        # verifier, so it can preserve a real stud without copying ear folds or
        # a whole source earlobe.
        ear_interior = aux.get("target_ear_interior_mask")
        target_lobe = torch.clamp(
            resize_earring_mask(aux.get("left_lobe_anchor"))
            + resize_earring_mask(aux.get("right_lobe_anchor")),
            0,
            1,
        )
        attachment_band = dilate_mask(target_lobe, 5)
        if ear_interior is not None:
            ear_interior = resize_earring_mask(ear_interior)
            ear_interior_allow = highres_interior_authority
            earring_edit = (
                earring_edit * (1.0 - ear_interior).clamp(0, 1)
                + earring_edit * ear_interior * ear_interior_allow
            ).clamp(0, 1)
            earring_edit = earring_edit * (1.0 - hoop_hole).clamp(0, 1)

        # A direct earring RGB paste must never own cheek or temple skin unless
        # its source-native instance alpha proves that exact pixel is jewellery.
        # The target earring corridor remains a learned-fallback constraint
        # below; it is not a geometric clip for a verified native instance.
        non_ear_face = torch.zeros_like(earring_edit)
        for key in ("target_face_surface_mask", "target_skin_surface_mask"):
            value = aux.get(key)
            if value is not None:
                non_ear_face = torch.maximum(non_ear_face, resize_earring_mask(value))
        target_earring_corridor = resize_earring_mask(
            aux.get("earring_valid_roi", aux.get("visible_ear_roi"))
        )
        face_allow = torch.clamp(
            attachment_band
            # A source-native instance is already object alpha, not a broad
            # ROI.  The target corridor decides whether the *side* is open,
            # but must not clip verified dangling earrings or solid hoops at
            # the cheek/neck boundary.  Expanding this alpha would reintroduce
            # a dark source edge, so keep its exact measured pixels only.
            + earring_edit
            # This is not a broad face exception: it contains only the measured
            # annulus accepted by the nested-contour verifier for this source
            # ear.  Letting it survive preserves the full hoop circumference.
            + highres_geometry_trace * earring_edit,
            0,
            1,
        )
        earring_edit = earring_edit * (
            1.0 - non_ear_face * (1.0 - face_allow)
        ).clamp(0, 1)

        # Mirror the PP foreground contract used by
        # ``_target_output_preserve_mask``.  That helper owns the target-hair
        # layer; the ear authority below also needs this local final-resolution
        # copy so it cannot cover the trained object inside the exposed lobe.
        pp_earring_foreground = resize_earring_mask(aux.get("earring_write_mask"))
        no_earring = aux.get("no_earring_case_mask")
        if no_earring is not None:
            pp_earring_foreground = pp_earring_foreground * (
                1.0 - resize_earring_mask(no_earring)
            ).clamp(0, 1)
        learned_hoop_hole = aux.get("hoop_hole_mask")
        if learned_hoop_hole is not None:
            pp_earring_foreground = pp_earring_foreground * (
                1.0 - resize_earring_mask(learned_hoop_hole)
            ).clamp(0, 1)

        # The author decode is the sole authority for face, hair, ears and
        # neck.  Do not paste target hair/skin/ears or harmonize a revealed
        # forehead here: each of those operations would replace part of the
        # author's single image field with a second image state and recreate a
        # visible seam.  This compositor may write only a verified earring
        # alpha; background SATD is applied by its caller.
        # Preserve the completed high-resolution transferred hair before the
        # source-native earring write.  Without this hand-off the PP decoder's
        # source-conditioned background/cloth response can erase a right-side
        # hair strand and expose the source white shirt.
        protected = (
            image_01 * (1.0 - target_hair_binary)
            + target_geometry * target_hair_binary
        ).clamp(0, 1)
        # PP can hallucinate an extra accessory around an exposed lobe. Clear
        # that lobe-local band back to the completed target transfer before
        # writing the one verified source-native instance below. The verified
        # instance is excluded from the band and is written immediately after.
        target_ear_region = resize_mask(
            target_left_gate + target_right_gate,
            size,
        ).to(device=image_01.device, dtype=image_01.dtype)
        target_ear_clear_dilate = max(
            1,
            int(getattr(self.args, "target_ear_accessory_clear_dilate", 28)),
        )
        target_ear_clear = dilate_mask(target_ear_region, target_ear_clear_dilate)
        target_ear_clear = target_ear_clear * (
            1.0 - dilate_mask(earring_edit, 2)
        ).clamp(0, 1)
        protected = (
            protected * (1.0 - target_ear_clear)
            + target_geometry * target_ear_clear
        ).clamp(0, 1)
        # The PP decoder can hallucinate a low-resolution accessory near the
        # exposed ear.  Clear only a small band around the verified native
        # source object back to the target geometry before writing the source
        # RGB.  This removes duplicate/offset earrings without changing the
        # face, hair or any no-earring sample.
        native_clear_band = dilate_mask(
            earring_edit,
            max(1, int(getattr(self.args, "output_earring_keep_dilate", 4)) or 4),
        )
        protected = (
            protected * (1.0 - native_clear_band)
            + target_geometry * native_clear_band
        ).clamp(0, 1)
        face_authority = torch.zeros_like(earring_edit)
        face_restore = torch.zeros_like(earring_edit)

        if bool(getattr(self.args, "enable_direct_earring_restore", True)):
            if earring_reference is not None:
                protected = protected * (1.0 - earring_edit) + earring_reference * earring_edit
        else:
            protected = protected * (1.0 - earring_edit) + image_01 * earring_edit

        # Keep a learned fallback for a genuine low-resolution accessory when
        # the source-native CV locator is inconclusive.  This is the only
        # route through which the PP earring branch can improve recall after
        # training.  Its permission is still the narrow, source/target-safe
        # ``earring_write_mask``; a broad ear ROI, parser neighbourhood or
        # contour candidate is never sufficient.  A verified hoop hole stays
        # target-owned, so this fallback cannot recreate the old ear-side
        # background cavity.
        learned_earring_mask = aux.get("earring_write_mask")
        if learned_earring_mask is None:
            learned_earring_mask = aux.get("earring_confident_mask")
        if learned_earring_mask is not None:
            learned_earring_mask = resize_earring_mask(learned_earring_mask)
            learned_hole = resize_earring_mask(aux.get("hoop_hole_mask"))
            learned_earring_mask = learned_earring_mask * (1.0 - learned_hole).clamp(0, 1)
            # A native source instance or its verified hoop hole is reserved
            # before semantic guards run.  Using only ``earring_edit`` here
            # let the learned low-resolution fallback write source RGB back
            # into deliberately rejected ear/face pixels and even refill a
            # target-owned hoop centre, causing black chunks, duplicate
            # earrings and background inside hollow rings.
            native_reserved = torch.clamp(
                highres_instance + highres_instance_hole,
                0,
                1,
            )
            learned_earring_mask = learned_earring_mask * (
                1.0 - native_reserved
            ).clamp(0, 1)
            if enable_highres_output:
                # Native source-side evidence is per exposed side.  A learned
                # mask is useful only as a small completion for such a side;
                # it must not manufacture an accessory on a source with no
                # accepted earring instance.
                learned_earring_mask = learned_earring_mask * target_side_earring_gate
                # The learned mask is a completion, never an independent
                # detector.  Restrict it to a small band around a measured
                # native source instance.  Without this adjacency gate, a
                # stale/overconfident PP mask can invent a second earring or a
                # black fragment on an accessory-free ear.
                native_completion_band = dilate_mask(highres_instance, 5)
                learned_earring_mask = learned_earring_mask * native_completion_band
            # Keep fallback writes in the exposed target-ear corridor and out
            # of cheek/temple skin.  The lobe attachment band is the only
            # allowed semantic-face exception for a stud/connector.
            learned_corridor = resize_earring_mask(
                aux.get("earring_valid_roi", aux.get("visible_ear_roi"))
            )
            learned_earring_mask = learned_earring_mask * learned_corridor
            learned_face_allow = torch.clamp(
                attachment_band + learned_corridor,
                0,
                1,
            )
            learned_earring_mask = learned_earring_mask * (
                1.0 - non_ear_face * (1.0 - learned_face_allow)
            ).clamp(0, 1)
            learned_alpha = max(
                0.0,
                # Native source-instance compositing is the only default RGB
                # authority.  A missing option must not reactivate the old
                # low-resolution learned paste, whose mask can include source
                # background around a long earring or inside a hollow one.
                min(1.0, float(getattr(self.args, "earring_learned_fallback_alpha", 0.0))),
            )
            learned_earring_mask = learned_earring_mask * learned_alpha
            protected = (
                protected * (1.0 - learned_earring_mask)
                + image_01 * learned_earring_mask
            ).clamp(0, 1)
        aux["output_face_target_authority_mask"] = face_authority
        aux["output_direct_face_skin_restore_mask"] = face_restore
        aux["output_source_earring_composite_mask"] = earring_edit
        aux["output_pp_earring_foreground_mask"] = pp_earring_foreground
        aux["output_source_earring_native_reference_gate"] = native_reference_gate
        aux["output_source_earring_native_reference"] = locator_reference
        aux["output_source_earring_native_parse_earring"] = native_source_parser_earring
        aux["output_source_earring_native_parse_is_fullres"] = native_source_parser_is_fullres
        aux["output_learned_earring_fallback_mask"] = learned_earring_mask if learned_earring_mask is not None else torch.zeros_like(earring_edit)
        aux["output_v5_earring_edit_mask"] = earring_edit
        aux["output_source_earring_presence_gate"] = source_earring_presence_gate
        aux["output_target_side_earring_gate"] = target_side_earring_gate
        aux["output_highres_earring_instance"] = highres_instance
        aux["output_highres_earring_hole"] = highres_instance_hole
        aux["output_highres_earring_refined_instance"] = highres_refined_instance
        aux["output_highres_earring_refined_hole"] = highres_refined_hole
        aux["output_highres_earring_interior_authority"] = highres_interior_authority
        aux["output_highres_earring_geometry_seed"] = highres_geometry_trace
        aux["output_highres_earring_geometry_hole"] = highres_geometry_hole
        aux["output_highres_earring_geometry_footprint"] = highres_geometry_footprint
        aux["output_source_earring_locator_roi"] = highres_locator_roi
        aux["output_source_earring_locator_seed"] = highres_locator_seed
        aux["output_source_earring_locator_support"] = highres_locator_support
        aux["output_source_earring_locator_ring_support"] = highres_locator_ring_support
        aux["output_source_earring_locator_parser"] = highres_locator_parser
        aux["output_source_earring_locator_presence_seed"] = highres_locator_presence_seed
        aux["output_source_earring_native_instance"] = highres_native_instance
        aux["output_source_earring_native_parser_instance"] = highres_native_parser_instance
        aux["output_source_earring_native_visual_recall"] = highres_native_visual_recall
        aux["output_highres_earring_output_refine_enabled"] = torch.full_like(
            earring_edit,
            float(enable_highres_output),
        )
        # Compatibility debug aliases.  They now show the real instance, not
        # a synthetic ellipse, so existing visualisation scripts stay useful.
        aux["output_highres_hoop_trace"] = highres_instance
        aux["output_highres_hoop_hole"] = highres_instance_hole
        return protected.clamp(0, 1) * 2 - 1

    def _apply_satd_background_residual(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply SATD residual only to supported target background pixels.

        Neither image used here is encoded by PP at this stage.  Face, details,
        hair, ears, earrings and neck remain read-only; ``M_remove`` support
        can only authorize an edit after it intersects the parsed background.
        """
        candidate = aux.get("satd_background_highres_01")
        base_target = aux.get("authoritative_target_highres_01")
        image_01 = normalized_to_01(image).clamp(0, 1)
        if candidate is None or base_target is None:
            aux["satd_background_cleanup_alpha"] = torch.zeros_like(image_01[:, :1])
            aux["satd_pp_before_01"] = image_01.detach()
            aux["satd_pp_after_01"] = image_01.detach()
            return image

        size = tuple(image_01.shape[-2:])
        candidate = normalized_to_01(candidate).to(
            device=image_01.device,
            dtype=image_01.dtype,
        )
        base_target = normalized_to_01(base_target).to(
            device=image_01.device,
            dtype=image_01.dtype,
        )
        if candidate.shape[-2:] != size:
            candidate = F.interpolate(candidate, size=size, mode="bilinear", align_corners=False)
        if base_target.shape[-2:] != size:
            base_target = F.interpolate(base_target, size=size, mode="bilinear", align_corners=False)

        target_parsing = self.parsing_helper.parse(base_target, out_size=size)
        target_labels = target_parsing.long()
        background = (target_labels == 0).to(dtype=image_01.dtype)
        parsed_hair = (target_labels == RAW_HAIR).to(dtype=image_01.dtype)
        # The high-resolution parser sees the *old source-hair shadow* in an
        # exposed background patch and commonly labels it as RAW_HAIR.  The
        # transfer's author target-hair mask is the geometry authority for
        # real, retained target hair.  Pixels that the parser calls hair but
        # lie outside that protected target-hair support are ghost residue,
        # not target hair; M_remove may clean only those pixels.
        raw_target_hair_hint = aux.get("target_hair_mask")
        if raw_target_hair_hint is None:
            target_hair_authority = parsed_hair
        else:
            target_hair_authority = resize_mask(
                raw_target_hair_hint,
                size,
            ).to(device=image_01.device, dtype=image_01.dtype).clamp(0, 1)
        target_hair_authority = dilate_mask(
            (target_hair_authority > 0.25).to(image_01.dtype),
            max(
                1,
                int(
                    getattr(
                        self.args,
                        "satd_background_target_hair_protect_dilate",
                        5,
                    )
                ),
            ),
        )
        ghost_hair = (parsed_hair * (1.0 - target_hair_authority)).clamp(0, 1)
        # A candidate cleanup can cover both ordinary parsed background and
        # the false-hair residue above.  All real subject semantics and the
        # authoritative target hairstyle remain hard-blocked below.
        safe_region = torch.clamp(background + ghost_hair, 0, 1)
        # SATD's candidate is already a latent-space cleanup result.  The
        # previous 17px exclusion ring removed nearly all pixels next to the
        # transferred hair, which is exactly where the residual shadow lives;
        # it made SATD appear disabled.  Keep a narrow safety ring so no
        # semantic subject pixel is edited, while allowing nearby background
        # cleanup to survive.
        subject_hint = (
            (target_labels != 0) & (target_labels != RAW_HAIR)
        ).to(dtype=image_01.dtype)
        subject_hint = torch.maximum(subject_hint, target_hair_authority)
        # The high-resolution parser is the primary geometry source, but a
        # thin hair/face contour can be labelled background at one pass.  Add
        # the already-computed target subject masks as a second guard before
        # SATD is allowed to touch any pixel.
        for key in (
            "target_face_surface_mask",
            "target_skin_surface_mask",
            "target_ear_mask",
            "target_ear_hair_occlusion_mask",
        ):
            value = aux.get(key)
            if value is not None:
                subject_hint = torch.maximum(
                    subject_hint,
                    resize_mask(value, size).to(
                        device=image_01.device,
                        dtype=image_01.dtype,
                    ),
                )
        subject_guard = dilate_mask(
            subject_hint,
            max(1, int(getattr(self.args, "satd_background_exclude_dilate", 2))),
        )

        cleanup_support = torch.zeros_like(background)
        for key in (
            # Boundary/context masks are still SATD-authorized support.  They
            # are safe to use here because the semantic subject guard below
            # remains the final authority on every individual pixel.
            "M_boundary",
            "M_remove",
            "M_remove_halo",
            "M_remove_tail",
            "M_remove_face",
            "M_remove_neck",
            "M_remove_context",
        ):
            value = aux.get(key)
            if value is not None:
                cleanup_support = torch.maximum(
                    cleanup_support,
                    resize_mask(value, size).to(device=image_01.device, dtype=image_01.dtype),
                )

        # M_remove is usually conservative around the outer shadow boundary.
        # Expand only its support mask before intersecting with semantic
        # background below; this recovers the remaining shadow tail while the
        # subject guard still blocks every face/hair/ear/neck pixel.
        cleanup_expand = max(
            0,
            # M_remove is intentionally conservative.  Expand its support
            # farther into the background so residual shadow tails are also
            # cleaned; the semantic subject guard below still hard-blocks
            # face, hair, ears and neck pixels.
            int(getattr(self.args, "satd_background_cleanup_dilate", 110)),
        )
        if cleanup_expand > 0:
            cleanup_support = dilate_mask(cleanup_support, cleanup_expand)

        # The semantic guard above intentionally protects hair plus a small
        # surrounding ring.  That ring also contains the original-hair shadow
        # on the *background* side of the transferred contour, which is the
        # white/transparent-looking strip seen along an outer hair edge.  Open
        # only the background pixels in that contour and only when an existing
        # M_remove support lies nearby.  This never grants SATD access to hair,
        # face, neck, ears or an unsupported background region.
        hair_edge_radius = max(
            1,
            int(getattr(self.args, "satd_background_hair_edge_dilate", 16)),
        )
        hair_edge_support_radius = max(
            hair_edge_radius,
            int(getattr(self.args, "satd_background_hair_edge_support_dilate", 24)),
        )
        hair_edge_background = (
            dilate_mask(target_hair_authority, hair_edge_radius)
            * background
        ).clamp(0, 1)
        nearby_cleanup = dilate_mask(cleanup_support, hair_edge_support_radius)
        hair_edge_cleanup = (
            hair_edge_background
            * nearby_cleanup
            * float(
                max(
                    0.0,
                    min(1.0, float(getattr(self.args, "satd_background_hair_edge_strength", 1.0))),
                )
            )
        ).clamp(0, 1)
        effective_cleanup_support = torch.maximum(cleanup_support, hair_edge_cleanup)
        # Keep the subject guard intact on this band.  The earlier version
        # removed it to chase a hair-edge shadow, which let SATD write a
        # discontinuous white contour into parsed hair/neck pixels.  Cleanup
        # now reaches the supported background through the expanded mask while
        # the semantic guard remains authoritative at the boundary.

        earring_guard = torch.zeros_like(background)
        # Only target-coordinate masks can guard the target output.  The
        # source-native fields are used for extraction/alignment and must not
        # be resized into this target frame (that creates a misplaced SATD
        # exclusion ring around an unrelated background patch).
        for key in (
            "target_earring_mask",
            "target_aligned_earring_alpha",
            "earring_confident_mask",
        ):
            value = aux.get(key)
            if value is not None:
                earring_guard = torch.maximum(
                    earring_guard,
                    resize_mask(value, size).to(device=image_01.device, dtype=image_01.dtype),
                )
        earring_guard = dilate_mask(
            earring_guard,
            max(1, int(getattr(self.args, "satd_background_earring_exclude_dilate", 6))),
        )

        # M_remove is the authority for residual cleanup.  Its write area is
        # ordinary background plus parser-misclassified ghost hair, while the
        # semantic guard deliberately excludes face, ears, neck and the true
        # author target hairstyle.
        alpha = (
            effective_cleanup_support.clamp(0, 1)
            * safe_region
            * (1.0 - subject_guard).clamp(0, 1)
            * (1.0 - earring_guard).clamp(0, 1)
        ).clamp(0, 1)
        # A source-hair shadow is primarily a low-frequency background error.
        # Transfer that continuous SATD reference instead of relying only on
        # the small per-pixel difference, while retaining a limited fraction
        # of the native residual for texture.  This is still a delta from the
        # authoritative pre-SATD target, never a flat white fill.
        candidate_low = gaussian_blur(candidate, kernel_size=31, sigma=9.0)
        base_target_low = gaussian_blur(base_target, kernel_size=31, sigma=9.0)

        # SATD predicts the direction of the correction, but a raw candidate
        # can still be brighter/darker than the surrounding background.  Build
        # a local reference from clean background pixels outside the approved
        # remove support, then combine it with SATD's low-frequency result.
        # This is an actual continuity operation: the removed shadow receives
        # the colour field of nearby unoccluded background rather than a white
        # or uniformly faded patch.  Subject and target-hair guards remain in
        # force for both the samples and the write alpha.
        clean_background = (
            safe_region
            * (1.0 - subject_guard).clamp(0, 1)
            * (1.0 - earring_guard).clamp(0, 1)
            * (1.0 - effective_cleanup_support).clamp(0, 1)
        ).clamp(0, 1)
        reference_kernel = 51
        clean_weight = gaussian_blur(
            clean_background,
            kernel_size=reference_kernel,
            sigma=15.0,
        )
        clean_reference = gaussian_blur(
            base_target * clean_background,
            kernel_size=reference_kernel,
            sigma=15.0,
        ) / clean_weight.clamp_min(1e-3)
        clean_reference_valid = (clean_weight >= 0.04).to(image_01.dtype)
        continuity_low = (
            clean_reference * clean_reference_valid
            + candidate_low * (1.0 - clean_reference_valid)
        ).clamp(0, 1)
        # SATD remains the primary learned correction; the local field fixes
        # its colour drift without erasing the background texture carried by
        # the already-decoded PP image.
        reference_low = (
            0.65 * candidate_low + 0.35 * continuity_low
        ).clamp(0, 1)
        low_frequency_residual = reference_low - base_target_low
        raw_residual = candidate - base_target
        # Prefer the smooth SATD/background estimate for continuity, retaining
        # only a small native residual fraction so the result is not a flat
        # white patch.
        residual = 0.85 * low_frequency_residual + 0.15 * raw_residual
        strength = max(
            0.0,
            min(1.5, float(getattr(self.args, "satd_background_residual_strength", 1.25))),
        )
        feather_kernel = max(
            1,
            int(getattr(self.args, "satd_background_alpha_feather", 7)),
        )
        if feather_kernel > 1:
            if feather_kernel % 2 == 0:
                feather_kernel += 1
            # Feather only *inside* the already approved background support;
            # it cannot spill the cleanup into hair, face, ears or neck.
            alpha = (
                gaussian_blur(
                    alpha,
                    kernel_size=feather_kernel,
                    sigma=max(0.5, feather_kernel / 4.0),
                )
                * effective_cleanup_support
                * safe_region
                * (1.0 - subject_guard).clamp(0, 1)
                * (1.0 - earring_guard).clamp(0, 1)
            ).clamp(0, 1)
        result = (image_01 + strength * residual * alpha).clamp(0, 1)
        aux["satd_pp_before_01"] = image_01.detach()
        aux["satd_pp_after_01"] = result.detach()
        aux["satd_background_cleanup_alpha"] = alpha.detach()
        aux["satd_background_ghost_hair_support"] = ghost_hair.detach()
        aux["satd_background_target_hair_authority"] = target_hair_authority.detach()
        aux["satd_background_hair_edge_background"] = hair_edge_background.detach()
        aux["satd_background_hair_edge_cleanup"] = hair_edge_cleanup.detach()
        aux["satd_background_effective_cleanup_support"] = effective_cleanup_support.detach()
        aux["satd_background_cleanup_residual"] = residual.detach()
        aux["satd_background_continuity_reference_01"] = continuity_low.detach()
        aux["satd_background_clean_reference_weight"] = clean_weight.detach()
        aux["satd_background_cleanup_alpha_area"] = alpha.flatten(1).sum(
            dim=1, keepdim=True
        ).view(-1, 1, 1, 1).detach()
        aux["satd_background_cleanup_residual_mean"] = (
            (residual.abs() * alpha).flatten(1).sum(dim=1, keepdim=True)
            / alpha.flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
        ).view(-1, 1, 1, 1).detach()
        aux["satd_background_candidate_delta_mean"] = raw_residual.abs().flatten(1).mean(
            dim=1, keepdim=True
        ).view(-1, 1, 1, 1).detach()
        aux["satd_background_cleanup_support_area"] = cleanup_support.flatten(1).sum(
            dim=1, keepdim=True
        ).view(-1, 1, 1, 1).detach()
        aux["satd_background_applied_delta_mean"] = (
            (result - image_01).abs().flatten(1).mean(dim=1, keepdim=True)
        ).view(-1, 1, 1, 1).detach()
        return result * 2.0 - 1.0

    def _preserve_target_output(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Keep the single author PP field; only clean background and earrings.

        V5's face compositor pasted source-face RGB into selected regions of
        the PP decode.  That creates the visible three-band boundary whenever
        a source fringe hid part of the forehead.  V6 deliberately has no
        face/skin compositor: its complete face comes from one PP decode.
        """
        if aux is None:
            return image
        direct_flag = aux.get("direct_satd_pp_input")
        if torch.is_tensor(direct_flag):
            direct_flag = direct_flag.to(device=image.device, dtype=image.dtype)
            if direct_flag.ndim == 1:
                direct_flag = direct_flag.view(-1, 1, 1, 1)
            elif direct_flag.ndim == 0:
                direct_flag = direct_flag.view(1, 1, 1, 1).expand(
                    image.size(0), 1, 1, 1
                )
            direct_flag = direct_flag[:, :1].clamp(0, 1)
        direct_mode = bool(
            torch.is_tensor(direct_flag)
            and direct_flag.detach().float().amin().item() > 0.5
        )
        if direct_mode:
            # SATD has already been encoded in the PP target.  Applying the
            # same candidate again after StyleGAN would double the correction
            # and is the source of the washed-out/halo output seen in prior
            # runs.  Keep the decoded PP image untouched before earring-only
            # composition.
            background_cleaned = image
            image_01 = normalized_to_01(image).clamp(0, 1)
            aux["satd_pp_before_01"] = image_01.detach()
            aux["satd_pp_after_01"] = image_01.detach()
            aux["satd_background_cleanup_alpha"] = torch.zeros_like(image_01[:, :1])
        elif torch.is_tensor(direct_flag) and direct_flag.detach().float().amax().item() > 0.5:
            # Mixed batches are uncommon but valid: apply the legacy residual
            # only to non-direct samples, preserving direct samples exactly.
            residual_image = self._apply_satd_background_residual(image, aux)
            direct_rgb = direct_flag.expand(-1, image.size(1), image.size(2), image.size(3))
            background_cleaned = image * direct_rgb + residual_image * (1.0 - direct_rgb)
        else:
            background_cleaned = self._apply_satd_background_residual(image, aux)
        return self._compose_strict_source_native_earring_v6(background_cleaned, aux)

    def compose_post_decode(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Apply V6's permitted post-decode edits to an author-decoded image.

        ``image`` must already be the output of the unmodified author
        Blending/PP path.  This method contains no encoder, latent, feature or
        StyleGAN operation, which keeps V6 from changing face or hair detail.
        """
        if aux is not None:
            aux["pp_global_raw_01"] = ((image.detach() + 1.0) * 0.5).clamp(0, 1)
        return self._preserve_target_output(image, aux), aux

    def render_refined(
        self,
        generator,
        latent_s: torch.Tensor,
        latent_f_64: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        # This is deliberately the author's one-call PP decode from
        # ``models/Blending.py``.  V5 originally split layer 5 from layers
        # 6--8 for an ear-feature injection; that injection is no longer part
        # of the global face path, so retaining the split needlessly changed
        # the execution path that must synthesize bang-hidden facial detail.
        image, _ = generator(
            [latent_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=latent_f_64,
        )
        return self.compose_post_decode(image, aux)
