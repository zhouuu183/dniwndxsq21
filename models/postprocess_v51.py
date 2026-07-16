from __future__ import annotations

import argparse
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoders import FeatureEncoderMult, FeatureiResnet, ModulationModule
from models.ear_modules_v51 import (
    BrightnessReEstimator,
    DynamicFineMaskRefresher,
    EarAnchoredQueryBuilder,
    FaceParsingHelperV51,
    HFDAGatedInjectionUnit,
    ShadowSuppressedHFExtractor,
    dilate_mask,
    gaussian_blur,
    ensure_mask_4d,
    normalized_to_01,
    resize_mask,
)
from models.stylegan2.model import PixelNorm


def build_query_builder_compat(**kwargs) -> EarAnchoredQueryBuilder:
    accepted = inspect.signature(EarAnchoredQueryBuilder.__init__).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return EarAnchoredQueryBuilder(**filtered_kwargs)


class PostProcessModelV51(nn.Module):
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
        self.parsing_helper = FaceParsingHelperV51(parse_size=getattr(self.args, "ear_parse_size", 512))
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
                "query_builder",
                "parsing_helper",
            ))
        ]
        if relevant_missing:
            print(f"[PostProcessModelV51] Missing base keys: {len(relevant_missing)}")
            print(relevant_missing[:20])
        if result.unexpected_keys:
            print(f"[PostProcessModelV51] Unexpected checkpoint keys: {len(result.unexpected_keys)}")
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
            if visible_ear_roi is not None:
                source_ear_mask = source_ear_mask * visible_ear_roi
            query_info["source_earring_mask"] = source_ear_mask
        if presence_target is not None:
            visibility_target = query_info.get("visibility_target")
            presence_target = presence_target.float()
            if visibility_target is not None:
                presence_target = presence_target * visibility_target.float()
            query_info["presence_target"] = presence_target

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
        query_mask = aux["query_mask"]
        hair_safe_query_mask = query_mask
        if source_hair_block is not None:
            block_strength = max(0.0, min(1.0, float(getattr(self.args, "source_hair_block_strength", 0.95))))
            source_hair_block = ensure_mask_4d(source_hair_block).float()
            hair_safe_query_mask = query_mask * (1 - block_strength * source_hair_block).clamp(0, 1)

        source_earring_mask = (ensure_mask_4d(aux["source_earring_mask"]).float() > 0.05).float()
        earring_query_boost = max(0.0, float(getattr(self.args, "earring_query_boost", 1.0)))
        earring_query_dilate = int(getattr(self.args, "earring_query_dilate", 5))
        earring_injection_dilate = int(getattr(self.args, "earring_injection_dilate", 1))
        earring_prior_dilate = int(getattr(self.args, "earring_prior_dilate", 1))
        earring_write_mask = dilate_mask(source_earring_mask, earring_injection_dilate)
        earring_prior_mask = dilate_mask(source_earring_mask, earring_prior_dilate)
        ear_detail_query_mask = torch.clamp(
            hair_safe_query_mask + earring_query_boost * dilate_mask(source_earring_mask, earring_query_dilate),
            0,
            1,
        )

        hf_outputs = self.hf_extractor(source_01, ear_detail_query_mask, earring_prior_mask)
        mask_outputs = self.mask_refresher(
            source_01,
            target_01,
            hf_outputs["high_energy"],
            ear_detail_query_mask,
            source_earring_mask,
        )
        learned_fine_mask = mask_outputs["fine_mask"]
        write_mask = resize_mask(earring_write_mask, learned_fine_mask.shape[-2:])
        mask_outputs["learned_fine_mask"] = learned_fine_mask
        mask_outputs["earring_write_mask"] = write_mask
        mask_outputs["fine_mask"] = learned_fine_mask * write_mask

        earring_fine_mask_floor = max(0.0, min(1.0, float(getattr(self.args, "earring_fine_mask_floor", 0.75))))
        if earring_fine_mask_floor > 0:
            floor_mask = resize_mask(source_earring_mask, mask_outputs["fine_mask"].shape[-2:])
            mask_outputs["earring_fine_floor_mask"] = floor_mask
            mask_outputs["fine_mask"] = torch.clamp(
                mask_outputs["fine_mask"] + earring_fine_mask_floor * floor_mask,
                0,
                1,
            )
        prior_feature, brightness_outputs = self.brightness_reestimator(
            target_01,
            hf_outputs["prior_feature"],
            mask_outputs["fine_mask"],
            ear_detail_query_mask,
        )
        finall_f, fine_mask_64 = self.ear_injector_64(base_feature, prior_feature, mask_outputs["fine_mask"])

        aux["raw_query_mask"] = query_mask
        aux["hair_safe_query_mask"] = hair_safe_query_mask
        aux["ear_detail_query_mask"] = ear_detail_query_mask
        aux["earring_prior_mask"] = earring_prior_mask
        aux.update(hf_outputs)
        aux.update(mask_outputs)
        aux.update(brightness_outputs)
        aux["adjusted_prior_feature"] = prior_feature
        aux["injected_fine_mask_64"] = fine_mask_64
        aux["source_01"] = source_01
        aux["HT_E"] = ensure_mask_4d(HT_E).float() if HT_E is not None else None
        return finall_s, finall_f, aux

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
        if aux is not None and bool(getattr(self.args, "enable_direct_earring_restore", True)):
            image = self._apply_direct_earring_restore(image, aux)
        return image, aux

    def _apply_direct_earring_restore(
        self,
        image: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        source = aux.get("source_full_01", aux.get("source_01"))
        mask = aux.get("source_earring_recall_mask", aux.get("earring_write_mask", aux.get("source_earring_mask")))
        if source is None or mask is None:
            return image

        out_size = image.shape[-2:]
        source = source.to(device=image.device, dtype=image.dtype)
        if source.ndim == 3:
            source = source.unsqueeze(0)
        source = F.interpolate(source, size=out_size, mode="bilinear", align_corners=False).clamp(0, 1)

        mask = (ensure_mask_4d(mask).to(device=image.device, dtype=image.dtype) > 0.05).float()
        write_mask = aux.get("earring_write_mask")
        if write_mask is not None:
            write_mask = (ensure_mask_4d(write_mask).to(device=image.device, dtype=image.dtype) > 0.05).float()
            mask = mask * resize_mask(write_mask, mask.shape[-2:])
        mask = resize_mask(mask, out_size)

        if mask.flatten(1).amax(dim=1).sum().item() <= 0:
            return image

        kernel = int(getattr(self.args, "direct_earring_restore_feather", 3))
        sigma = float(getattr(self.args, "direct_earring_restore_sigma", 1.0))
        strength = max(0.0, min(1.0, float(getattr(self.args, "direct_earring_restore_strength", 0.95))))
        alpha = gaussian_blur(mask, kernel_size=kernel, sigma=sigma).clamp(0, 1) * strength
        image_01 = ((image + 1) / 2).clamp(0, 1)
        restored = image_01 * (1 - alpha) + source * alpha
        aux["direct_earring_restore_alpha"] = alpha
        return restored.clamp(0, 1) * 2 - 1

