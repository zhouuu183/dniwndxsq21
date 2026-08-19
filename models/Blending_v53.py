from __future__ import annotations

import argparse

import torch
from torch import nn

from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.postprocess_v53 import PostProcessModelV53
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")


class BlendingV53(nn.Module):
    """
    V8-cleaned blending + V53 refinement entry.

    The alignment input is produced by Alignment_v8/SATD. This class mirrors
    the v8 blending mask logic, then replaces only the final PP stage with the
    ear/detail-aware PostProcessModelV53.
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.net = Net(self.opts) if net is None else net

        blending_checkpoint = torch.load(self.opts.blending_checkpoint, map_location=self.opts.device)
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get("clip", "ViT-B/32"))
        self.blending_encoder.load_state_dict(blending_checkpoint["model_state_dict"], strict=False)
        self.blending_encoder.to(self.opts.device).eval()

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

        pp_args = argparse.Namespace(
            use_mod=getattr(self.opts, "pp_v53_use_mod", True),
            use_full=getattr(self.opts, "pp_v53_use_full", True),
            pretrain=False,
            finetune=False,
            ear_parse_size=getattr(self.opts, "ear_parse_size", 512),
            ear_feature_channels=getattr(self.opts, "ear_feature_channels", 128),
            ear_low_alpha=getattr(self.opts, "ear_low_alpha", 0.15),
            ear_dilate=getattr(self.opts, "ear_dilate", 21),
            hair_change_dilate=getattr(self.opts, "hair_change_dilate", 25),
            earring_expand=getattr(self.opts, "earring_expand", 15),
            ear_downward_shift=getattr(self.opts, "ear_downward_shift", 10),
            target_hair_dilate=getattr(self.opts, "target_hair_dilate", 11),
            source_hair_block_dilate=getattr(self.opts, "source_hair_block_dilate", 5),
            source_hair_block_strength=getattr(self.opts, "source_hair_block_strength", 0.95),
            source_hair_block_high_floor=getattr(self.opts, "source_hair_block_high_floor", 0.30),
            target_visibility_expand=getattr(self.opts, "target_visibility_expand", 5),
            max_target_hair_overlap=getattr(self.opts, "max_target_hair_overlap", 0.55),
            earring_lobe_dilate=getattr(self.opts, "earring_lobe_dilate", 17),
            earring_lobe_down_shift=getattr(self.opts, "earring_lobe_down_shift", 18),
            earring_outer_shift=getattr(self.opts, "earring_outer_shift", 10),
            earring_query_floor=getattr(self.opts, "earring_query_floor", 0.45),
            ear_injection_strength=getattr(self.opts, "ear_injection_strength", 2.5),
            ear_injection_mask_dilate=getattr(self.opts, "ear_injection_mask_dilate", 3),
            ear_injection_mask_boost=getattr(self.opts, "ear_injection_mask_boost", 1.35),
            earring_injection_lateral_ratio=getattr(self.opts, "earring_injection_lateral_ratio", 0.34),
            earring_injection_search_dilate=getattr(self.opts, "earring_injection_search_dilate", 1),
            earring_injection_min_high=getattr(self.opts, "earring_injection_min_high", 0.010),
            earring_injection_min_chroma=getattr(self.opts, "earring_injection_min_chroma", 0.026),
            earring_injection_min_contrast=getattr(self.opts, "earring_injection_min_contrast", 0.014),
            earring_injection_min_bright=getattr(self.opts, "earring_injection_min_bright", 0.012),
            earring_injection_min_dark=getattr(self.opts, "earring_injection_min_dark", 0.014),
            earring_injection_min_votes=getattr(self.opts, "earring_injection_min_votes", 2),
            earring_injection_skin_max_chroma=getattr(self.opts, "earring_injection_skin_max_chroma", 0.16),
            earring_injection_skin_max_high=getattr(self.opts, "earring_injection_skin_max_high", 0.026),
            earring_injection_skin_max_contrast=getattr(self.opts, "earring_injection_skin_max_contrast", 0.026),
            earring_injection_face_reject_dilate=getattr(self.opts, "earring_injection_face_reject_dilate", 1),
            earring_injection_face_reject_strength=getattr(self.opts, "earring_injection_face_reject_strength", 0.65),
            earring_injection_parser_max_area=getattr(self.opts, "earring_injection_parser_max_area", 260.0),
            earring_injection_seed_max_area_frac=getattr(self.opts, "earring_injection_seed_max_area_frac", 0.012),
            earring_injection_strict_min_high=getattr(self.opts, "earring_injection_strict_min_high", 0.018),
            earring_injection_strict_min_chroma=getattr(self.opts, "earring_injection_strict_min_chroma", 0.045),
            earring_injection_strict_min_contrast=getattr(self.opts, "earring_injection_strict_min_contrast", 0.024),
            earring_injection_strict_min_bright=getattr(self.opts, "earring_injection_strict_min_bright", 0.022),
            earring_injection_strict_min_dark=getattr(self.opts, "earring_injection_strict_min_dark", 0.022),
            earring_injection_seed_dilate=getattr(self.opts, "earring_injection_seed_dilate", 1),
            ear_blur_kernel=getattr(self.opts, "ear_blur_kernel", 11),
            ear_blur_sigma=getattr(self.opts, "ear_blur_sigma", 3.0),
            ear_mask_hidden=getattr(self.opts, "ear_mask_hidden", 32),
            ear_mask_init_bias=getattr(self.opts, "ear_mask_init_bias", -4.0),
            earring_query_dilate=getattr(self.opts, "earring_query_dilate", 11),
            earring_query_boost=getattr(self.opts, "earring_query_boost", 1.75),
            earring_texture_query_boost=getattr(self.opts, "earring_texture_query_boost", 1.0),
            earring_fine_mask_floor=getattr(self.opts, "earring_fine_mask_floor", 0.12),
            earring_object_dilate=getattr(self.opts, "earring_object_dilate", 9),
            earring_object_support_dilate=getattr(self.opts, "earring_object_support_dilate", 13),
            earring_align_to_target=getattr(self.opts, "earring_align_to_target", True),
            earring_align_strength=getattr(self.opts, "earring_align_strength", 1.0),
            earring_align_max_shift=getattr(self.opts, "earring_align_max_shift", 26),
            earring_attach_y_ratio=getattr(self.opts, "earring_attach_y_ratio", 0.78),
            earring_strict_core_enable=getattr(self.opts, "earring_strict_core_enable", True),
            earring_core_min_high=getattr(self.opts, "earring_core_min_high", 0.014),
            earring_core_min_chroma=getattr(self.opts, "earring_core_min_chroma", 0.038),
            earring_core_min_contrast=getattr(self.opts, "earring_core_min_contrast", 0.018),
            earring_core_dilate=getattr(self.opts, "earring_core_dilate", 3),
            earring_recall_query_dilate=getattr(self.opts, "earring_recall_query_dilate", 5),
            earring_raw_query_recall_weight=getattr(self.opts, "earring_raw_query_recall_weight", 0.35),
            earring_weak_recall_weight=getattr(self.opts, "earring_weak_recall_weight", 0.55),
            earring_fallback_recall_weight=getattr(self.opts, "earring_fallback_recall_weight", 0.45),
            earring_safe_recall_weight=getattr(self.opts, "earring_safe_recall_weight", 0.45),
            earring_core_floor_weight=getattr(self.opts, "earring_core_floor_weight", 0.35),
            earring_parser_support_dilate=getattr(self.opts, "earring_parser_support_dilate", 5),
            earring_parser_keep_dilate=getattr(self.opts, "earring_parser_keep_dilate", 3),
            earring_safe_roi_dilate=getattr(self.opts, "earring_safe_roi_dilate", 7),
            earring_dark_reject_max_gray=getattr(self.opts, "earring_dark_reject_max_gray", 0.22),
            earring_dark_reject_max_chroma=getattr(self.opts, "earring_dark_reject_max_chroma", 0.07),
            earring_dark_reject_max_high=getattr(self.opts, "earring_dark_reject_max_high", 0.018),
            earring_dark_reject_dilate=getattr(self.opts, "earring_dark_reject_dilate", 3),
            earring_highlight_support_dilate=getattr(self.opts, "earring_highlight_support_dilate", 5),
            earring_highlight_keep_dilate=getattr(self.opts, "earring_highlight_keep_dilate", 3),
            earring_raw_hint_residual=getattr(self.opts, "earring_raw_hint_residual", 0.12),
            earring_weak_support_dilate=getattr(self.opts, "earring_weak_support_dilate", 5),
            earring_weak_object_weight=getattr(self.opts, "earring_weak_object_weight", 0.35),
            earring_guarded_composite=getattr(self.opts, "earring_guarded_composite", True),
            earring_composite_exclude_face=getattr(self.opts, "earring_composite_exclude_face", True),
            earring_composite_face_keep_dilate=getattr(self.opts, "earring_composite_face_keep_dilate", 3),
            earring_composite_max_area_frac=getattr(self.opts, "earring_composite_max_area_frac", 0.012),
            earring_composite_exclude_target_hair=getattr(self.opts, "earring_composite_exclude_target_hair", False),
            earring_composite_restrict_visible_roi=getattr(self.opts, "earring_composite_restrict_visible_roi", False),
            earring_composite_seed_dilate=getattr(self.opts, "earring_composite_seed_dilate", 2),
            earring_composite_seed_max_area_frac=getattr(self.opts, "earring_composite_seed_max_area_frac", 0.018),
            earring_composite_seed_parser_max_area=getattr(self.opts, "earring_composite_seed_parser_max_area", 520.0),
            earring_composite_seed_use_search_roi=getattr(self.opts, "earring_composite_seed_use_search_roi", False),
            earring_composite_seed_face_reject_strength=getattr(
                self.opts,
                "earring_composite_seed_face_reject_strength",
                0.35,
            ),
            earring_prior_reference_dilate=getattr(self.opts, "earring_prior_reference_dilate", 1),
            earring_prior_query_dilate=getattr(self.opts, "earring_prior_query_dilate", 5),
            earring_output_guard=getattr(self.opts, "earring_output_guard", True),
            earring_output_guard_dilate=getattr(self.opts, "earring_output_guard_dilate", 21),
            earring_output_guard_blur=getattr(self.opts, "earring_output_guard_blur", 11),
            earring_output_guard_sigma=getattr(self.opts, "earring_output_guard_sigma", 3.0),
            earring_output_guard_min_area=getattr(self.opts, "earring_output_guard_min_area", 4.0),
            earring_output_guard_max_area_frac=getattr(self.opts, "earring_output_guard_max_area_frac", 0.035),
            earring_output_guard_fallback_max_area_frac=getattr(
                self.opts,
                "earring_output_guard_fallback_max_area_frac",
                0.055,
            ),
            face_output_guard=getattr(self.opts, "face_output_guard", True),
            face_output_guard_strength=getattr(self.opts, "face_output_guard_strength", 0.85),
            face_output_guard_blur=getattr(self.opts, "face_output_guard_blur", 13),
            face_output_guard_sigma=getattr(self.opts, "face_output_guard_sigma", 4.0),
            face_output_guard_min_area=getattr(self.opts, "face_output_guard_min_area", 128.0),
            face_output_guard_exclude_earring_dilate=getattr(
                self.opts,
                "face_output_guard_exclude_earring_dilate",
                9,
            ),
            earring_composite_strength=getattr(self.opts, "earring_composite_strength", 0.85),
            earring_composite_feather=getattr(self.opts, "earring_composite_feather", 3),
            earring_composite_sigma=getattr(self.opts, "earring_composite_sigma", 1.2),
            earring_composite_strict_mask=getattr(self.opts, "earring_composite_strict_mask", True),
            earring_composite_exclude_hair_block=getattr(self.opts, "earring_composite_exclude_hair_block", False),
            earring_composite_min_high=getattr(self.opts, "earring_composite_min_high", 0.018),
            earring_composite_min_chroma=getattr(self.opts, "earring_composite_min_chroma", 0.045),
            earring_composite_min_contrast=getattr(self.opts, "earring_composite_min_contrast", 0.022),
            enable_cleanup_face_refiner=getattr(self.opts, "enable_cleanup_face_refiner", False),
            cleanup_face_hidden=getattr(self.opts, "cleanup_face_hidden", 128),
            cleanup_face_strength=getattr(self.opts, "cleanup_face_strength", 1.0),
            cleanup_face_dilate=getattr(self.opts, "cleanup_face_dilate", 5),
            cleanup_face_exclude_earring_dilate=getattr(self.opts, "cleanup_face_exclude_earring_dilate", 13),
        )
        self.post_process = PostProcessModelV53(pp_args).to(self.opts.device).eval()

        checkpoint_path = getattr(self.opts, "pp_v53_checkpoint", None) or getattr(self.opts, "pp_checkpoint", None)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.post_process.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

    @staticmethod
    def _extract_cleanup_masks(align_shape):
        if not isinstance(align_shape, dict):
            return None
        delta_masks = align_shape.get("delta_masks")
        if not isinstance(delta_masks, dict):
            return None
        cleanup_masks = {
            key: value.float()
            for key in CLEANUP_MASK_KEYS
            if torch.is_tensor(value := delta_masks.get(key))
        }
        return cleanup_masks or None

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_2 = name_to_embed["shape"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        face_mask, _ = filter_parsing_to_primary_subject(name_to_embed["face"]["mask"])
        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        HM_1 = torch.where(face_mask == 13, torch.ones_like(face_mask), torch.zeros_like(face_mask)).float()
        HM_3 = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        HM_1D, _ = self.dilate_erosion.mask(HM_1)
        HM_3D, HM_3E = self.dilate_erosion.mask(HM_3)

        latent_S_1, latent_F_align = name_to_embed["face"]["S"], align_shape["latent_F_align"]
        HM_X = align_color["HM_X"]
        latent_S_3 = name_to_embed["color"]["S"]

        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        if I_1 is not I_3 or I_1 is not I_2:
            S_blend_6_18 = self.blending_encoder(latent_S_1[:, 6:], latent_S_3[:, 6:], I_1 * target_mask, I_3 * HM_3E)
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        I_blend, _ = self.net.generator([S_blend], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=latent_F_align)
        I_blend_256 = self.downsample_256(I_blend)

        S_final, F_final, aux = self.post_process(
            I_1,
            I_blend_256,
            target_mask,
            HM_3E,
            cleanup_masks=self._extract_cleanup_masks(align_shape),
        )
        aux["base_image_full"] = I_blend.detach()
        I_final, _ = self.post_process.render_refined(self.net.generator, S_final, F_final, aux)

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending", "blending.png", I_blend)
            save_latents(output_dir, "Blending", "blending.npz", S_blend=S_blend)
            save_gen_image(output_dir, "Final", "final.png", I_final)
            save_latents(output_dir, "Final", "final.npz", S_final=S_final, F_final=F_final)

        return ((I_final[0] + 1) / 2).clip(0, 1)
