from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

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


def load_blending_checkpoint_v5(checkpoint_path: str) -> dict:
    """Load the required blending checkpoint with diagnostics for bad artifacts."""
    path = Path(checkpoint_path).expanduser()
    try:
        stat = path.stat()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "[BlendingV5] Blending checkpoint was not found: "
            f"{path}. Pass a valid --blending_checkpoint path."
        ) from error

    if not path.is_file():
        raise ValueError(f"[BlendingV5] Blending checkpoint is not a file: {path}")
    if stat.st_size == 0:
        raise ValueError(
            f"[BlendingV5] Blending checkpoint is empty: {path}. "
            "Re-copy the verified checkpoint artifact."
        )

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as error:
        try:
            with path.open("rb") as handle:
                magic = handle.read(16).hex() or "empty"
        except OSError:
            magic = "unreadable"
        raise RuntimeError(
            "[BlendingV5] Unable to read the configured blending checkpoint.\n"
            f"  path: {path}\n"
            f"  size: {stat.st_size:,} bytes\n"
            f"  first_16_bytes_hex: {magic}\n"
            f"  torch.load error: {error}\n"
            "The file is truncated, corrupted, or not a PyTorch checkpoint. "
            "Re-copy the verified blending checkpoint at this exact path, or pass "
            "a verified compatible file with --blending_checkpoint. V5 will not "
            "fall back to a different checkpoint automatically."
        ) from error

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "[BlendingV5] Blending checkpoint has an unsupported payload: "
            f"{path}. Expected a dictionary containing 'model_state_dict'."
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "[BlendingV5] Blending checkpoint is missing 'model_state_dict': "
            f"{path}. This is not a compatible blending checkpoint."
        )
    return checkpoint


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

        blending_checkpoint = load_blending_checkpoint_v5(self.opts.blending_checkpoint)
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
            # HairFast keeps the source face in the same image frame.  Moving
            # a recovered earring to chase two noisy parser centroids creates
            # a second earring; zero is the correct default contract.
            earring_align_max_shift=pp_value("earring_align_max_shift", 0),
            # V5 structural compositor controls.  They are runtime policy,
            # not checkpoint metadata, so an older V5 weight cannot reactivate
            # a source-face paste or target-aligned seed path.
            face_continuity_work_size=pp_value("face_continuity_work_size", 512),
            face_detail_soft_edge=pp_value("face_detail_soft_edge", 6),
            face_hair_soft_edge=pp_value("face_hair_soft_edge", 4),
            earring_component_max_depth=pp_value("earring_component_max_depth", 3),
            earring_component_max_cumulative_cost=pp_value(
                "earring_component_max_cumulative_cost", 1.55
            ),
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
            # The learned PP output may fill a verified low-resolution
            # ordinary-earring mask only where native source extraction has
            # no confident instance.  This is a V5 runtime policy, never an
            # old-checkpoint default.
            earring_learned_fallback_alpha=pp_policy_value(
                "earring_learned_fallback_alpha",
                0.0,
            ),
            # V5 uses one PP face base; normal-face hard authority is disabled.
            enable_direct_face_skin_restore=False,
            output_target_hair_preserve_dilate=pp_policy_value(
                "output_target_hair_preserve_dilate", 5
            ),
            output_face_hair_seam_preserve_dilate=pp_policy_value(
                "output_face_hair_seam_preserve_dilate",
                7,
            ),
            output_earring_keep_dilate=pp_policy_value("output_earring_keep_dilate", 0),
            output_preserve_blur=pp_policy_value("output_preserve_blur", 1),
            # A feather mixes the PP image and the target-authoritative hair
            # at the parser hairline.  Those images can have different colour
            # statistics, which presents as a coloured ring around the face.
            output_hairline_feather=pp_policy_value("output_hairline_feather", 0),
            enable_revealed_skin_harmonize=pp_policy_value("enable_revealed_skin_harmonize", False),
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
        # V5 must not spread a colour transfer through the hair boundary.  At
        # 256px the old 5px Gaussian alpha became a broad face-side and
        # outer-hair halo; after the final 1024px restore it appeared as two
        # concentric fake-hair rings.  Colour changes stay inside the semantic
        # hair ownership mask.  The generated RGB itself keeps strand detail.
        alpha = ((target_mask > 0.5).to(image_01.dtype) * strength).clamp(0, 1)
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
        mask = (self._as_mask(target_hair_mask, image.shape[-2:], image.dtype) > 0.5).to(
            dtype=image.dtype
        )
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
        # Do not feather this authority mask.  It is the second V5 colour
        # hand-off; blurring a 256px radius by the output scale was changing
        # non-hair pixels on both sides of the hairline after PP completed.
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

        # ``color_before_pp`` is a 256px conditioning image.  Lifting its
        # colour delta back onto the 1024px result replaces the native strand
        # residual with an enlarged low-frequency field, which is precisely the
        # waxy/filter-like hair failure.  Keep the generated high-resolution
        # transfer as the only final hair RGB authority.  The colour-adjusted
        # 256px tensor remains the PP conditioning/training target.
        authoritative_hair_highres = I_blend

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
                # The high-resolution hair transfer is already complete at
                # this point.  PP receives the 256px colour-stage tensor, but
                # its final compositor must preserve these original pixels
                # rather than decode a new hairstyle for validation/inference.
                "completed_hair_highres": ((I_blend[0] + 1.0) * 0.5).clamp(0, 1).detach().cpu(),
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
            earring_reference=name_to_embed["face"].get(
                "image_v5_native",
                name_to_embed["face"].get("image_1024"),
            ),
            source_face_reference=name_to_embed["face"].get(
                "image_v5_native",
                name_to_embed["face"].get("image_1024"),
            ),
            cleanup_masks=cleanup_masks,
        )
        I_final, _ = self.post_process.render_refined(self.net.generator, S_final, F_final, aux)
        I_final_raw = I_final
        # Do not reapply the 256px colour delta after face/earring composition.
        # The compositor has already selected the generated high-resolution
        # hair, direct source face pixels, and exact source earring alpha.
        # A second upsampled colour pass corrupts all three at semantic edges.

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
            if aux.get("output_source_earring_native_reference") is not None:
                save_gen_image(
                    output_dir,
                    "PostProcessV5",
                    "earring_native_reference.png",
                    aux["output_source_earring_native_reference"] * 2 - 1,
                )
            if aux.get("output_v5_base") is not None:
                save_gen_image(
                    output_dir,
                    "PostProcessV5",
                    "v5_face_base_before_earring.png",
                    aux["output_v5_base"] * 2 - 1,
                )
            if aux.get("output_v5_source_rgb") is not None:
                save_gen_image(
                    output_dir,
                    "PostProcessV5",
                    "v5_source_earring_rgb.png",
                    aux["output_v5_source_rgb"] * 2 - 1,
                )
            for image_name in (
                "output_v5_face_face_after_lowfreq",
                "output_v5_face_face_after_detail",
                "output_v5_face_source_highfreq_residual",
                "output_v5_target_aligned_earring_rgb",
            ):
                if aux.get(image_name) is not None:
                    category = "earring" if "earring" in image_name else "face"
                    save_gen_image(
                        output_dir,
                        f"res/v5_structural/{category}",
                        f"{image_name}.png",
                        aux[image_name] * 2 - 1,
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
                "left_elliptical_hoop",
                "right_elliptical_hoop",
                "left_elliptical_hoop_hole",
                "right_elliptical_hoop_hole",
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
                "left_target_side_open",
                "right_target_side_open",
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
                "output_v19_unified_face",
                "output_v19_face_pp_alpha",
                "output_v19_face_residual_mask",
                "output_v19_face_residual_alpha",
                "output_v19_localization_roi",
                "output_v19_foreground_seed",
                "output_v19_raw_foreground",
                "output_v19_source_alpha",
                "output_v19_source_detail_mask",
                "output_v19_source_face_alpha",
                "output_v19_source_skin_alpha",
                "output_v19_source_detail_alpha",
                "output_v19_revealed_skin_alpha",
                "output_v19_face_tone_correction",
                "output_v19_boundary_band",
                "output_v19_hole_mask",
                "output_v19_alpha_leak",
                "output_v5_earring_edit_mask",
                "output_source_earring_native_reference_gate",
                "output_source_earring_native_parse_earring",
                "output_source_earring_native_parse_is_fullres",
                "output_source_earring_presence_gate",
                "output_highres_earring_instance",
                "output_highres_earring_hole",
                "output_highres_earring_refined_instance",
                "output_highres_earring_refined_hole",
                "output_highres_earring_interior_authority",
                "output_highres_earring_geometry_seed",
                "output_highres_earring_geometry_hole",
                "output_highres_earring_geometry_footprint",
                "output_source_earring_locator_roi",
                "output_source_earring_locator_seed",
                "output_source_earring_locator_support",
                "output_source_earring_locator_ring_support",
                "output_source_earring_locator_parser",
                "output_source_earring_locator_presence_seed",
                "output_source_earring_native_instance",
                "output_source_earring_native_parser_instance",
                "output_source_earring_native_visual_recall",
                "output_highres_earring_output_refine_enabled",
                "output_highres_hoop_trace",
                "output_highres_hoop_hole",
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
                "output_v5_face_source_valid_skin",
                "output_v5_face_source_valid_detail",
                "output_v5_face_source_hair_guard_uncertain",
                "output_v5_face_revealed_skin",
                "output_v5_face_revealed_skin_microtexture",
                "output_v5_face_revealed_skin_microtexture_alpha",
                "output_v5_face_face_surface",
                "output_v5_face_target_hair_soft_alpha",
                "output_v5_face_face_boundary_band",
                "output_v5_target_aligned_earring_alpha",
                "output_v5_target_hair_overlap",
                "output_v5_outside_alpha_write_area",
                "output_v5_target_left_visible_ear",
                "output_v5_target_right_visible_ear",
                "output_v5_earring_source_native_earring_alpha",
                "output_v5_earring_source_native_hole_alpha",
                "output_v5_earring_source_native_left_alpha",
                "output_v5_earring_source_native_right_alpha",
                "output_v5_earring_source_native_left_hole_alpha",
                "output_v5_earring_source_native_right_hole_alpha",
                "output_v5_earring_source_native_recall_hint",
                "output_v5_earring_parser_earring_seed",
                "output_v5_earring_lobe_anchor",
                "output_v5_earring_localization_core",
                "output_v5_earring_localization_adaptive",
                "output_v5_earring_probable_visual_evidence",
                "output_v5_earring_grabcut_raw",
                "output_v5_earring_selected_components",
            ):
                if aux.get(mask_name) is not None:
                    mask_path = (
                        "res/v5_structural/diagnostics"
                        if mask_name.startswith("output_v5_")
                        else "PostProcessV5Masks"
                    )
                    save_vis_mask(output_dir, mask_path, f"{mask_name}.png", aux[mask_name])

            # Persist scalar P0 diagnostics next to the high-resolution masks.
            # They are deliberately descriptive rather than a single quality
            # score: visual inspection must catch upper/mid/lower face seams
            # and no-earring false positives before any long training run.
            def mean_value(name):
                value = aux.get(name)
                if value is None:
                    return None
                return float(value.detach().float().mean().cpu().item())

            def area_value(name):
                value = aux.get(name)
                if value is None:
                    return None
                return float((value.detach().float() > 0.5).sum().cpu().item())

            face_delta = aux.get("output_v5_face_continuous_lowfreq_delta")
            face_delta_l = None
            face_delta_ab = None
            if face_delta is not None:
                face_delta = face_delta.detach().float()
                face_delta_l = float(face_delta[:, :1].abs().mean().cpu().item())
                face_delta_ab = float(face_delta[:, 1:].abs().mean().cpu().item())
            face_detail = aux.get("output_v5_face_source_highfreq_residual")
            face_energy = {"upper": None, "mid": None, "lower": None}
            if face_detail is not None:
                energy = face_detail.detach().float().abs().mean(dim=1, keepdim=True)
                height = energy.size(-2)
                thirds = ((0, height // 3), (height // 3, 2 * height // 3), (2 * height // 3, height))
                for key, (start, end) in zip(face_energy, thirds):
                    face_energy[key] = float(energy[:, :, start:end].mean().cpu().item())
            diagnostic = {
                "face": {
                    "source_valid_face_area": area_value("output_v5_face_source_valid_skin"),
                    "revealed_skin_area": area_value("output_v5_face_revealed_skin"),
                    "face_boundary_lowfreq_delta_L": face_delta_l,
                    "face_boundary_lowfreq_delta_ab": face_delta_ab,
                    "face_boundary_gradient_jump": mean_value("output_v5_face_face_boundary_band"),
                    "upper_face_HF_energy": face_energy["upper"],
                    "mid_face_HF_energy": face_energy["mid"],
                    "lower_face_HF_energy": face_energy["lower"],
                    "source_valid_HF_retention": mean_value("output_v5_face_source_valid_detail"),
                },
                "earring": {
                    "presence_state": aux.get("output_v5_earring_source_native_presence_state", torch.empty(0)).detach().cpu().tolist(),
                    "presence_score": aux.get("output_v5_earring_source_native_presence_score", torch.empty(0)).detach().cpu().tolist(),
                    "native_alpha_area": area_value("output_v5_earring_source_native_earring_alpha"),
                    "native_component_count": aux.get("output_v5_earring_component_count", torch.empty(0)).detach().cpu().tolist(),
                    "selected_component_area": area_value("output_v5_earring_selected_components"),
                    "max_graph_depth_used": aux.get("output_v5_earring_max_graph_depth_used", torch.empty(0)).detach().cpu().tolist(),
                    "cumulative_graph_cost": aux.get("output_v5_earring_cumulative_graph_cost", torch.empty(0)).detach().cpu().tolist(),
                    "aligned_alpha_area": area_value("output_v5_target_aligned_earring_alpha"),
                    "target_hair_overlap": mean_value("output_v5_target_hair_overlap"),
                    "outside_alpha_write_area": mean_value("output_v5_outside_alpha_write_area"),
                },
            }
            diagnostic_dir = output_dir / "res" / "v5_structural" / "diagnostics"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            with (diagnostic_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
                json.dump(diagnostic, handle, ensure_ascii=True, indent=2)

        return ((I_final[0] + 1) / 2).clip(0, 1)
