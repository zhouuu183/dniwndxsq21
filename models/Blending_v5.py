from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from models.Blending_v8 import Blending_v8
from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.postprocess_v5 import PostProcessModelV5, load_checkpoint_compat
from utils.bicubic import BicubicDownSample
from utils.blending_checkpoint_v8 import validate_blending_checkpoint_policy_v8
from utils.hair_color_match_v8 import gaussian_blur2d, lab_to_rgb, rgb_to_lab
from utils.image_utils import DilateErosion
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents, save_vis_mask


def hair_color_debug_ab_to_rgb(image: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Render a Lab A/B debug tensor without depending on Blending_v8 internals."""
    was_normalized = bool(image.detach().amin() < -0.05)
    image_01 = ((image + 1.0) * 0.5 if was_normalized else image).clamp(0, 1)
    lab = rgb_to_lab(image_01)
    if ab.shape[-2:] != lab.shape[-2:]:
        ab = F.interpolate(ab, size=lab.shape[-2:], mode="bilinear", align_corners=False)
    result = lab_to_rgb(torch.cat((lab[:, :1], ab.to(dtype=lab.dtype)), dim=1))
    return result * 2.0 - 1.0 if was_normalized else result


class BlendingV5(Blending_v8):
    """
    V8-cleaned blending + V5 refinement entry.

    The alignment input is produced by Alignment_v8/SATD. This class mirrors
    the v8 blending mask logic, then replaces only the final PP stage with the
    ear/detail-aware PostProcessModelV5.
    """

    def __init__(self, opts, net=None):
        # BlendingV5 owns a different PP module, so initialize the shared
        # nn.Module state without running Blending_v8's baseline constructor.
        nn.Module.__init__(self)
        self.opts = opts
        self.net = Net(self.opts) if net is None else net

        blending_checkpoint = torch.load(self.opts.blending_checkpoint, map_location="cpu")
        self.blending_color_policy_v8 = validate_blending_checkpoint_policy_v8(
            blending_checkpoint,
            self.opts,
            checkpoint_path=self.opts.blending_checkpoint,
        )
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get("clip", "ViT-B/32"))
        blending_load = self.blending_encoder.load_state_dict(
            blending_checkpoint["model_state_dict"], strict=False
        )
        if blending_load.missing_keys or blending_load.unexpected_keys:
            print(
                "[BlendingV5] blending checkpoint mismatch: "
                f"missing={len(blending_load.missing_keys)}, "
                f"unexpected={len(blending_load.unexpected_keys)}"
            )
        self.blending_encoder.to(self.opts.device).eval()

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

        checkpoint_path = getattr(self.opts, "pp_v5_checkpoint", None) or getattr(self.opts, "pp_checkpoint", None)
        checkpoint = load_checkpoint_compat(checkpoint_path, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {}) or {}

        def pp_value(name: str, default):
            # Runtime policy is authoritative.  The old PP checkpoint stores
            # historical mask/tone defaults (wide earring write, weak hair
            # blocking, disabled forehead harmonisation); replaying those values
            # would silently undo this refactor.  Checkpoint values are only a
            # fallback when the current entry point has no such option.
            return getattr(self.opts, name, checkpoint_args.get(name, default))

        def pp_policy_value(name: str, default):
            """Read non-learned compositing policy without checkpoint replay."""
            return getattr(self.opts, name, default)

        pp_args = argparse.Namespace(
            use_mod=checkpoint_args.get("use_mod", getattr(self.opts, "pp_v5_use_mod", True)),
            use_full=checkpoint_args.get("use_full", getattr(self.opts, "pp_v5_use_full", True)),
            pretrain=False,
            finetune=False,
            ear_parse_size=pp_value("ear_parse_size", 512),
            ear_feature_channels=pp_value("ear_feature_channels", 128),
            ear_low_alpha=pp_value("ear_low_alpha", 0.1),
            ear_dilate=pp_value("ear_dilate", 21),
            hair_change_dilate=pp_value("hair_change_dilate", 25),
            earring_expand=pp_value("earring_expand", 15),
            ear_downward_shift=pp_value("ear_downward_shift", 10),
            target_hair_dilate=pp_value("target_hair_dilate", 11),
            earring_occlusion_dilate=pp_value("earring_occlusion_dilate", 3),
            source_hair_block_dilate=pp_policy_value("source_hair_block_dilate", 8),
            source_hair_block_strength=pp_policy_value("source_hair_block_strength", 0.95),
            target_visibility_expand=pp_value("target_visibility_expand", 5),
            max_target_hair_overlap=pp_policy_value("max_target_hair_overlap", 0.30),
            min_target_visible_overlap=pp_policy_value("min_target_visible_overlap", 0.10),
            min_target_ear_area=pp_value("min_target_ear_area", 8.0),
            earring_channel_down=pp_value("earring_channel_down", 32),
            earring_align_max_shift=pp_value("earring_align_max_shift", 12),
            ear_blur_kernel=pp_value("ear_blur_kernel", 11),
            ear_blur_sigma=pp_value("ear_blur_sigma", 3.0),
            ear_mask_hidden=pp_value("ear_mask_hidden", 32),
            ear_mask_init_bias=pp_value("ear_mask_init_bias", -4.0),
            use_dataset_query_mask=pp_value("use_dataset_query_mask", False),
            enable_earring_query_recall=pp_value("enable_earring_query_recall", True),
            earring_query_recall_dilate=pp_value("earring_query_recall_dilate", 7),
            earring_query_downward_shift=pp_policy_value("earring_query_downward_shift", 10),
            earring_query_lower_lobe_weight=pp_value("earring_query_lower_lobe_weight", 0.20),
            earring_query_candidate_boost=pp_value("earring_query_candidate_boost", 0.90),
            earring_query_block_protect=pp_policy_value("earring_query_block_protect", 0.95),
            disable_earring_path_if_low_confidence=pp_policy_value("disable_earring_path_if_low_confidence", True),
            earring_source_presence_min_area=pp_policy_value("earring_source_presence_min_area", 4.0),
            earring_search_downward_shift=pp_policy_value("earring_search_downward_shift", 10),
            earring_search_dilate=pp_policy_value("earring_search_dilate", 7),
            earring_write_max_target_hair_overlap=pp_policy_value(
                "earring_write_max_target_hair_overlap", 0.30
            ),
            earring_write_source_block_dilate=pp_policy_value(
                "earring_write_source_block_dilate", 3
            ),
            earring_write_dilate=pp_policy_value("earring_write_dilate", 3),
            earring_write_connectivity_iters=pp_policy_value(
                "earring_write_connectivity_iters", 32
            ),
            earring_write_connectivity_kernel=pp_policy_value(
                "earring_write_connectivity_kernel", 5
            ),
            # This is a connectivity-only radius.  It follows a thin hoop over
            # parser/antialiasing gaps but never widens the final RGB write mask.
            earring_write_bridge_dilate=pp_policy_value("earring_write_bridge_dilate", 17),
            earring_anchor_visible_dilate=pp_policy_value("earring_anchor_visible_dilate", 3),
            earring_fine_mask_floor=pp_value("earring_fine_mask_floor", 0.18),
            earring_fine_mask_dilate=pp_value("earring_fine_mask_dilate", 5),
            enable_source_content_gate=pp_policy_value("enable_source_content_gate", True),
            source_content_gate_dilate=pp_policy_value("source_content_gate_dilate", 3),
            source_hair_face_suppress_dilate=pp_policy_value("source_hair_face_suppress_dilate", 5),
            source_hair_face_suppress_strength=pp_policy_value("source_hair_face_suppress_strength", 0.65),
            source_hair_face_suppress_max_y=pp_policy_value("source_hair_face_suppress_max_y", 0.45),
            source_hair_face_suppress_ear_exclude_dilate=pp_policy_value(
                "source_hair_face_suppress_ear_exclude_dilate",
                9,
            ),
            # V5 always gives the final ear boundary, hoop hole and repaired
            # hairstyle geometry back to the target/blending image.  Older PP
            # checkpoint args may contain ``False`` from the broad-paste path;
            # honoring that flag would reopen ear holes at inference.
            enable_output_target_preserve=True,
            # Final output is target-authoritative outside the validated
            # earring object.  This policy must not be inherited from an old
            # PP checkpoint that allowed the decoder to redraw the face.
            enable_direct_earring_restore=True,
            enable_direct_face_skin_restore=True,
            output_target_hair_preserve_dilate=pp_policy_value(
                "output_target_hair_preserve_dilate", 5
            ),
            output_face_hair_seam_preserve_dilate=pp_policy_value(
                "output_face_hair_seam_preserve_dilate",
                7,
            ),
            output_earring_keep_dilate=pp_policy_value("output_earring_keep_dilate", 0),
            output_preserve_blur=pp_policy_value("output_preserve_blur", 1),
            output_hairline_feather=pp_policy_value("output_hairline_feather", 9),
            enable_revealed_skin_harmonize=pp_policy_value("enable_revealed_skin_harmonize", True),
            revealed_skin_harmonize_strength=pp_policy_value("revealed_skin_harmonize_strength", 0.9),
            revealed_skin_tone_kernel=pp_policy_value("revealed_skin_tone_kernel", 15),
            revealed_skin_tone_sigma=pp_policy_value("revealed_skin_tone_sigma", 7.0),
            revealed_skin_diffuse_iters=pp_policy_value("revealed_skin_diffuse_iters", 24),
            revealed_skin_tone_limit=pp_policy_value("revealed_skin_tone_limit", 0.28),
            revealed_skin_detail_gain=pp_policy_value("revealed_skin_detail_gain", 1.0),
            revealed_skin_seam_band=pp_policy_value("revealed_skin_seam_band", 7),
            revealed_skin_detail_safe_ratio=pp_policy_value("revealed_skin_detail_safe_ratio", 0.35),
            revealed_skin_min_reference_area=pp_policy_value(
                "revealed_skin_min_reference_area", 96.0
            ),
        )
        self.post_process = PostProcessModelV5(pp_args).to(self.opts.device).eval()

        result = self.post_process.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        if result.missing_keys:
            print(f"[BlendingV5] Missing PP keys: {len(result.missing_keys)}")
            print(result.missing_keys[:20])
        if result.unexpected_keys:
            print(f"[BlendingV5] Unexpected PP keys: {len(result.unexpected_keys)}")
            print(result.unexpected_keys[:20])

    @staticmethod
    def _as_mask(mask: torch.Tensor, size: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask[:, :1].to(dtype=dtype)
        if mask.shape[-2:] != size:
            mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
        return mask.clamp(0, 1)

    @torch.inference_mode()
    def _prepare_color_transfer(self, align_shape, name_to_embed):
        """Keep V5 runnable against the standalone V8 blending implementation.

        V5 uses V8 as a base class, but V8 intentionally exposes only its public
        ``blend_images`` API.  This copy is the narrow common pre-PP contract V5
        needs; keeping it here prevents a V8 refactor from breaking PP dataset
        generation at import/runtime.
        """
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_2 = name_to_embed["shape"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        face_mask, _ = filter_parsing_to_primary_subject(name_to_embed["face"]["mask"])
        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        HM_1 = (face_mask == 13).to(dtype=I_1.dtype)
        HM_3 = (color_mask == 13).to(dtype=I_1.dtype)
        HM_1D, _ = self.dilate_erosion.mask(HM_1)
        HM_3D, HM_3E = self.dilate_erosion.mask(HM_3)

        latent_S_1 = name_to_embed["face"]["S"]
        latent_S_3 = name_to_embed["color"]["S"]
        latent_F_align = align_shape["latent_F_align"]
        HM_X = align_shape.get("HM_X_repaired", align_shape["HM_X"])
        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        if I_1 is not I_3 or I_1 is not I_2:
            S_blend_6_18 = self.blending_encoder(
                latent_S_1[:, 6:],
                latent_S_3[:, 6:],
                I_1 * target_mask,
                I_3 * HM_3E,
            )
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        return I_1, I_2, I_3, latent_F_align, HM_3E, target_mask, S_blend

    @torch.inference_mode()
    def _reference_dominant_color(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        target_hair_mask: torch.Tensor,
        reference_hair_mask: torch.Tensor,
        kwargs: dict,
        *,
        return_debug: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Transfer global colour while retaining the generated hair's local sheen.

        Reference and target heads are not pixel-aligned.  Consequently the
        reference may control Lab statistics, but never where a highlight lands.
        The target's high-frequency Lab residual is retained, so this operation
        cannot turn all hair regions into one flat colour.
        """
        if bool(getattr(self.opts, "disable_reference_dominant_hair_color_v8", False)):
            return image, None

        strength = float(kwargs.get(
            "hair_color_reference_strength_v8",
            getattr(self.opts, "hair_color_reference_strength_v8", 0.9),
        ))
        if strength <= 0:
            return image, None

        image_is_normalized = bool(image.detach().amin() < -0.05)
        image_01 = ((image + 1.0) * 0.5 if image_is_normalized else image).clamp(0, 1)
        reference_01 = ((reference + 1.0) * 0.5 if reference.detach().amin() < -0.05 else reference).clamp(0, 1)
        if reference_01.shape[-2:] != image_01.shape[-2:]:
            reference_01 = F.interpolate(
                reference_01,
                size=image_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        target_mask = self._as_mask(target_hair_mask, image_01.shape[-2:], image_01.dtype)
        reference_mask = self._as_mask(reference_hair_mask, reference_01.shape[-2:], image_01.dtype)
        if target_mask.flatten(1).sum(dim=1).min().item() < 4 or reference_mask.flatten(1).sum(dim=1).min().item() < 4:
            return image, None

        target_lab = rgb_to_lab(image_01)
        reference_lab = rgb_to_lab(reference_01)
        target_ab, reference_ab = target_lab[:, 1:], reference_lab[:, 1:]
        target_l, reference_l = target_lab[:, :1], reference_lab[:, :1]
        target_area = target_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=4.0)
        reference_area = reference_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=4.0)

        def masked_stats(values: torch.Tensor, mask: torch.Tensor, area: torch.Tensor):
            mean = (values * mask).sum(dim=(-2, -1), keepdim=True) / area
            variance = ((values - mean).square() * mask).sum(dim=(-2, -1), keepdim=True) / area
            return mean, variance.add(1e-6).sqrt()

        target_ab_mean, target_ab_std = masked_stats(target_ab, target_mask, target_area)
        reference_ab_mean, reference_ab_std = masked_stats(reference_ab, reference_mask, reference_area)
        chroma_std_gain = float(kwargs.get(
            "hair_color_detail_chroma_gain_v8",
            getattr(self.opts, "hair_color_detail_chroma_gain_v8", 1.0),
        ))
        chroma_std_gain = max(0.0, min(1.5, chroma_std_gain))
        matched_ab = (target_ab - target_ab_mean) * (
            1.0 + chroma_std_gain * (reference_ab_std / target_ab_std.clamp(min=1e-4) - 1.0)
        ) + reference_ab_mean

        radius = max(0, int(kwargs.get(
            "hair_color_low_frequency_radius_v8",
            getattr(self.opts, "hair_color_low_frequency_radius_v8", 15),
        )))
        sigma = kwargs.get(
            "hair_color_low_frequency_sigma_v8",
            getattr(self.opts, "hair_color_low_frequency_sigma_v8", None),
        )
        target_low_l = gaussian_blur2d(target_l, radius=radius, sigma=sigma)
        reference_low_l = gaussian_blur2d(reference_l, radius=radius, sigma=sigma)
        target_low_ab = gaussian_blur2d(target_ab, radius=radius, sigma=sigma)
        reference_low_ab = gaussian_blur2d(reference_ab, radius=radius, sigma=sigma)
        target_l_mean, target_l_std = masked_stats(target_low_l, target_mask, target_area)
        reference_l_mean, reference_l_std = masked_stats(reference_low_l, reference_mask, reference_area)
        luma_strength = max(0.0, min(1.0, float(kwargs.get(
            "hair_color_luma_reference_strength_v8",
            getattr(self.opts, "hair_color_luma_reference_strength_v8", 0.30),
        ))))
        mean_limit = max(0.0, float(kwargs.get(
            "hair_color_luma_mean_limit_v8",
            getattr(self.opts, "hair_color_luma_mean_limit_v8", 6.0),
        )))
        std_limit = max(1.0, float(kwargs.get(
            "hair_color_luma_std_ratio_limit_v8",
            getattr(self.opts, "hair_color_luma_std_ratio_limit_v8", 1.25),
        )))
        mean_delta = (reference_l_mean - target_l_mean).clamp(-mean_limit, mean_limit)
        std_ratio = (reference_l_std / target_l_std.clamp(min=1e-4)).clamp(
            1.0 / std_limit,
            std_limit,
        )
        matched_low_l = (target_low_l - target_l_mean) * std_ratio + target_l_mean + mean_delta
        matched_l = target_l + luma_strength * (matched_low_l - target_low_l)

        matched_rgb = lab_to_rgb(torch.cat((matched_l, matched_ab), dim=1))
        feather = max(0, int(kwargs.get(
            "hair_color_feather_radius_v8",
            getattr(self.opts, "hair_color_feather_radius_v8", 5),
        )))
        alpha = gaussian_blur2d(target_mask, radius=feather) if feather > 0 else target_mask
        alpha = (alpha * strength).clamp(0, 1)
        output_01 = (matched_rgb * alpha + image_01 * (1.0 - alpha)).clamp(0, 1)
        output = output_01 * 2.0 - 1.0 if image_is_normalized else output_01
        debug = None
        if return_debug:
            debug = {
                "target_low_ab": target_low_ab.detach(),
                "reference_low_ab": reference_low_ab.detach(),
            }
        return output, debug

    @torch.inference_mode()
    def _restore_authoritative_color_after_pp(
        self,
        image: torch.Tensor,
        color_before_pp: torch.Tensor,
        target_hair_mask: torch.Tensor,
        kwargs: dict,
        earring_exclude_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift the 256px colour decision to the high-res generated hair.

        The correction is low-frequency only.  This preserves the original
        high-resolution strand texture and highlight layout while preventing PP
        from pulling the transferred hair back toward the source colour.
        """
        low = self.downsample_256(image)
        if low.shape[-2:] != color_before_pp.shape[-2:]:
            low = F.interpolate(low, size=color_before_pp.shape[-2:], mode="bicubic", align_corners=False)
        desired = color_before_pp.to(device=image.device, dtype=image.dtype)
        low_delta = desired - low
        delta = F.interpolate(low_delta, size=image.shape[-2:], mode="bicubic", align_corners=False)
        mask = self._as_mask(target_hair_mask, image.shape[-2:], image.dtype)
        if earring_exclude_mask is not None:
            # A recovered earring can sit in front of target hair.  The final
            # hair-colour correction must not recolour that exact object after
            # V5 restored it.
            exclude = self._as_mask(
                earring_exclude_mask,
                image.shape[-2:],
                image.dtype,
            )
            mask = mask * (1.0 - exclude).clamp(0, 1)
        feather = max(0, int(kwargs.get(
            "hair_color_feather_radius_v8",
            getattr(self.opts, "hair_color_feather_radius_v8", 5),
        )))
        if feather > 0:
            scale = max(1, round(image.shape[-1] / max(1, desired.shape[-1])))
            mask = gaussian_blur2d(mask, radius=feather * scale)
        mask = mask.clamp(0, 1)
        return (image + delta * mask).clamp(-1, 1), mask

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        # Reuse the v8 color-transfer state verbatim so the target-generation
        # path cannot silently drift from the standalone v8 pipeline.  This uses
        # align_shape["HM_X"] (the shape hair region) for target_mask, which is
        # what the blending encoder was trained against.
        I_1, I_2, I_3, latent_F_align, HM_3E, target_mask, S_blend = self._prepare_color_transfer(
            align_shape,
            name_to_embed,
        )

        I_blend, _ = self.net.generator([S_blend], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=latent_F_align)
        shape_only = I_blend
        if bool(getattr(self.opts, "save_all", False)):
            shape_only, _ = self.net.generator(
                [name_to_embed["face"]["S"]],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_align,
            )
        I_blend_256_raw = self.downsample_256(I_blend)
        # ``_prepare_color_transfer`` canonicalises both aliases, but use the
        # explicit repaired field at this hand-off so PP cannot regress to a
        # stale/raw crown topology.
        HM_X = align_shape.get("HM_X_repaired", align_shape["HM_X"])
        save_all = bool(getattr(self.opts, "save_all", False))
        save_color_debug = save_all and bool(kwargs.get(
            "debug_save_intermediate_color",
            getattr(self.opts, "debug_save_intermediate_color", True),
        ))
        color_before_pp, color_transfer_debug = self._reference_dominant_color(
            I_blend_256_raw,
            I_3,
            HM_X,
            HM_3E,
            kwargs,
            return_debug=save_color_debug,
        )
        I_blend_256 = color_before_pp

        delta_masks = align_shape.get("delta_masks", {}) if isinstance(align_shape, dict) else {}
        cleanup_masks = {
            key: delta_masks[key]
            for key in ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")
            if torch.is_tensor(delta_masks.get(key))
        } if isinstance(delta_masks, dict) else {}

        # Build this once at the formal stage boundary as well as in the final
        # path.  PP dataset generation can then carry the repaired v8 hair
        # topology forward instead of reparsing a 256px image and reviving an
        # old crown/ear gap.
        authoritative_hair_highres, _ = self._restore_authoritative_color_after_pp(
            I_blend,
            color_before_pp,
            HM_X,
            kwargs,
        )

        # PP dataset generation needs the *actual* inference tensor at this
        # boundary.  Expose it explicitly instead of monkey-patching
        # ``downsample_256``/``post_process`` and raising an exception: the old
        # hook stopped before the new colour transfer and silently trained PP on
        # a different colour distribution.
        return_stage = kwargs.get("return_stage")
        stop_before_pp = bool(kwargs.get("stop_before_pp", False))
        if return_stage is not None and return_stage not in {
            "color_before_pp",
            "before_pp",
        }:
            raise ValueError(
                "BlendingV5 return_stage must be 'color_before_pp'/'before_pp', "
                f"got {return_stage!r}."
            )
        if stop_before_pp or return_stage is not None:
            color_stage = ((color_before_pp[0] + 1.0) * 0.5).clamp(0, 1)
            raw_stage = ((I_blend_256_raw[0] + 1.0) * 0.5).clamp(0, 1)
            return {
                "stage": "color_before_pp",
                "image": color_stage,
                "color_before_pp": color_stage,
                "pre_reference_color": raw_stage,
                "cleanup_masks": cleanup_masks,
                "target_hair_mask": HM_X[0].detach().cpu(),
                "authoritative_hair_highres": authoritative_hair_highres[0].detach().cpu(),
            }

        S_final, F_final, aux = self.post_process(
            I_1,
            I_blend_256,
            target_mask,
            HM_3E,
            target_hair_mask=HM_X,
            authoritative_hair_highres=authoritative_hair_highres,
            authoritative_target_highres=I_blend,
            # Keep accessory recovery in the original aligned source frame.
            # I_1 is the 256px encoder tensor; using it here made thin metal
            # hoops fade before the final high-resolution composite even when
            # their write mask was accepted.
            earring_reference=name_to_embed["face"].get("image_1024"),
            source_face_reference=name_to_embed["face"].get("image_1024"),
            cleanup_masks=cleanup_masks,
        )
        I_final, _ = self.post_process.render_refined(self.net.generator, S_final, F_final, aux)
        # PP checkpoints trained with the old colour path can still shift the
        # hair toward the source.  Reuse the exact same transform, restricted
        # to the repaired target-hair mask, for a deterministic final hand-off.
        I_final_raw = I_final
        # The V5 final compositor has already declared the semantic face and
        # the recovered accessory target-owned.  Exclude both from the last
        # high-resolution colour correction too; otherwise a dilated hair mask
        # can reintroduce a narrow forehead/cheek colour band after PP has
        # restored the correct target pixels.
        final_color_exclude = torch.zeros(
            I_final.shape[0],
            1,
            I_final.shape[-2],
            I_final.shape[-1],
            device=I_final.device,
            dtype=I_final.dtype,
        )
        for key in (
            "earring_write_mask",
            "output_face_target_authority_mask",
            "output_direct_face_skin_restore_mask",
        ):
            value = aux.get(key)
            if value is not None:
                final_color_exclude = torch.maximum(
                    final_color_exclude,
                    self._as_mask(value, I_final.shape[-2:], I_final.dtype),
                )
        I_final, _ = self._restore_authoritative_color_after_pp(
            I_final_raw,
            color_before_pp,
            HM_X,
            kwargs,
            earring_exclude_mask=final_color_exclude,
        )

        if save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending", "blending.png", I_blend)
            save_gen_image(output_dir, "Stages", "shape_only.png", shape_only)
            save_gen_image(output_dir, "Stages", "blending_before_pp.png", color_before_pp)
            save_gen_image(output_dir, "Stages", "color_before_pp.png", color_before_pp)
            save_gen_image(
                output_dir,
                "Stages",
                "authoritative_hair_highres.png",
                authoritative_hair_highres,
            )
            save_gen_image(
                output_dir,
                "Stages",
                "pp_raw_before_color_restore.png",
                I_final_raw,
            )
            save_gen_image(output_dir, "Stages", "final_after_pp.png", I_final)
            save_gen_image(output_dir, "Stages", "hair_target_before_color.png", I_blend_256_raw)
            save_gen_image(output_dir, "Stages", "hair_ref_color.png", I_3)
            save_gen_image(output_dir, "Stages", "hair_final_before_pp.png", color_before_pp)
            if color_transfer_debug is not None:
                save_gen_image(
                    output_dir,
                    "Stages",
                    "hair_lowfreq_target.png",
                    hair_color_debug_ab_to_rgb(
                        I_blend_256_raw,
                        color_transfer_debug["target_low_ab"],
                    ),
                )
                save_gen_image(
                    output_dir,
                    "Stages",
                    "hair_lowfreq_ref.png",
                    hair_color_debug_ab_to_rgb(
                        I_3,
                        color_transfer_debug["reference_low_ab"],
                    ),
                )
                save_gen_image(
                    output_dir,
                    "Stages",
                    "hair_after_color_transfer.png",
                    color_before_pp,
                )
            save_latents(output_dir, "Blending", "blending.npz", S_blend=S_blend)
            save_gen_image(output_dir, "Final", "final.png", I_final)
            save_latents(output_dir, "Final", "final.npz", S_final=S_final, F_final=F_final)
            if aux.get("detail_reference_01") is not None:
                save_gen_image(
                    output_dir,
                    "PostProcessV5",
                    "earring_detail_reference.png",
                    aux["detail_reference_01"] * 2 - 1,
                )
            for mask_name in (
                "source_earring_mask",
                "source_left_parser_earring",
                "source_right_parser_earring",
                "earring_search_mask",
                "earring_write_mask",
                "earring_core_mask",
                "earring_completion_mask",
                "earring_object_mask",
                "earring_filled_mask",
                "hoop_hole_mask",
                "earring_final_alpha",
                "left_strong_candidate",
                "right_strong_candidate",
                "target_left_ear",
                "target_right_ear",
                "left_lobe_anchor",
                "right_lobe_anchor",
                "left_parser_visible",
                "right_parser_visible",
                "left_fallback_visible",
                "right_fallback_visible",
                "left_side_active",
                "right_side_active",
                "left_search_mask",
                "right_search_mask",
                "left_core_mask",
                "right_core_mask",
                "left_completion_mask",
                "right_completion_mask",
                "left_write_mask",
                "right_write_mask",
                "earring_visible_segment_mask",
                "no_earring_case_mask",
                "no_earring_case",
                "target_hair_ear_bridge_mask",
                "source_earring_detection_mask",
                "ear_detail_query_mask",
                "query_mask_before_recall",
                "earring_query_recall_mask",
                "online_earring_candidate_mask",
                "online_earring_search_mask",
                "source_lobe_search_mask",
                "earring_confident_mask",
                "earring_visibility_mask",
                "target_ear_hair_occlusion_mask",
                "source_hair_block_mask",
                "fine_mask_before_floor",
                "earring_fine_floor_support",
                "output_source_earring_composite_mask",
                "output_v5_earring_edit_mask",
                "output_face_target_authority_mask",
                "output_direct_face_skin_restore_mask",
                "fine_mask",
                "prior_mask",
                "source_hair_face_feature_suppress_mask",
                "source_face_feature_scale",
                "target_ear_boundary_protect_mask",
                "ear_detail_restore_mask_before",
                "ear_detail_restore_mask_after",
                "ear_detail_suppress_mask",
                "output_target_hair_preserve_mask",
                "output_target_hair_face_seam_mask",
                "output_target_hair_earring_keep_mask",
                "revealed_skin_mask",
                "revealed_skin_seam_mask",
                "revealed_skin_blend_mask",
                "revealed_skin_blend_band",
                "source_visible_skin_reference_mask",
                "lower_face_skin_reference_mask",
                "local_skin_anchor_mask",
                "lower_face_anchor_mask",
                "revealed_skin_harmonize_mask",
                "face_harmonize_write_mask",
                "revealed_skin_detail_mask",
                "revealed_skin_tone_reference",
                "skin_field_L",
                "skin_field_ab",
            ):
                if aux.get(mask_name) is not None:
                    save_vis_mask(output_dir, "PostProcessV5Masks", f"{mask_name}.png", aux[mask_name])

        return ((I_final[0] + 1) / 2).clip(0, 1)
