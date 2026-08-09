from __future__ import annotations

import argparse

import torch
from torch import nn

from models.Blending_v8 import Blending_v8, hair_color_debug_ab_to_rgb
from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.postprocess_v5 import PostProcessModelV5, load_checkpoint_compat
from utils.bicubic import BicubicDownSample
from utils.blending_checkpoint_v8 import validate_blending_checkpoint_policy_v8
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents, save_vis_mask


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
            earring_write_bridge_dilate=pp_policy_value("earring_write_bridge_dilate", 5),
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
            enable_output_target_preserve=pp_policy_value("enable_output_target_preserve", True),
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
        HM_X = align_shape["HM_X_repaired"]
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
            cleanup_masks=cleanup_masks,
        )
        I_final, _ = self.post_process.render_refined(self.net.generator, S_final, F_final, aux)
        # PP checkpoints trained with the old colour path can still shift the
        # hair toward the source.  Reuse the exact same transform, restricted
        # to the repaired target-hair mask, for a deterministic final hand-off.
        I_final_raw = I_final
        I_final, _ = self._restore_authoritative_color_after_pp(
            I_final_raw,
            color_before_pp,
            HM_X,
            kwargs,
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
                "earring_search_mask",
                "earring_write_mask",
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
                "fine_mask",
                "prior_mask",
                "source_hair_face_feature_suppress_mask",
                "source_face_feature_scale",
                "output_target_hair_preserve_mask",
                "output_target_hair_face_seam_mask",
                "output_target_hair_earring_keep_mask",
                "revealed_skin_mask",
                "revealed_skin_seam_mask",
                "revealed_skin_blend_mask",
                "revealed_skin_blend_band",
                "source_visible_skin_reference_mask",
                "lower_face_skin_reference_mask",
                "revealed_skin_harmonize_mask",
                "revealed_skin_detail_mask",
                "revealed_skin_tone_reference",
            ):
                if aux.get(mask_name) is not None:
                    save_vis_mask(output_dir, "PostProcessV5Masks", f"{mask_name}.png", aux[mask_name])

        return ((I_final[0] + 1) / 2).clip(0, 1)
