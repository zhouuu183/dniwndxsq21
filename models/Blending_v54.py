from __future__ import annotations

import argparse

import torch
from torch import nn

from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.postprocess_v54 import PostProcessModelV54
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")


class BlendingV54(nn.Module):
    """
    V8-cleaned blending + restored V54 refinement entry.

    The alignment input is produced by Alignment_v8/SATD. This class mirrors
    the v8 blending mask logic, then replaces only the final PP stage with the
    ear/detail-aware PostProcessModelV54.
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
            use_mod=getattr(self.opts, "pp_v54_use_mod", True),
            use_full=getattr(self.opts, "pp_v54_use_full", True),
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
            ear_blur_kernel=getattr(self.opts, "ear_blur_kernel", 11),
            ear_blur_sigma=getattr(self.opts, "ear_blur_sigma", 3.0),
            ear_mask_hidden=getattr(self.opts, "ear_mask_hidden", 32),
            ear_mask_init_bias=getattr(self.opts, "ear_mask_init_bias", -2.2),
            earring_query_dilate=getattr(self.opts, "earring_query_dilate", 11),
            earring_query_boost=getattr(self.opts, "earring_query_boost", 1.25),
            earring_texture_query_boost=getattr(self.opts, "earring_texture_query_boost", 0.70),
            earring_fine_mask_floor=getattr(self.opts, "earring_fine_mask_floor", 0.45),
            earring_object_dilate=getattr(self.opts, "earring_object_dilate", 3),
            earring_object_support_dilate=getattr(self.opts, "earring_object_support_dilate", 5),
            earring_object_max_area_frac=getattr(self.opts, "earring_object_max_area_frac", 0.014),
            earring_placement_dilate=getattr(self.opts, "earring_placement_dilate", 13),
            earring_lobe_dilate=getattr(self.opts, "earring_lobe_dilate", 5),
            earring_lobe_down_shift=getattr(self.opts, "earring_lobe_down_shift", 18),
            earring_outer_shift=getattr(self.opts, "earring_outer_shift", 6),
            earring_query_floor=getattr(self.opts, "earring_query_floor", 0.45),
            earring_search_radius_y=getattr(self.opts, "earring_search_radius_y", 34),
            earring_search_radius_x=getattr(self.opts, "earring_search_radius_x", 22),
            earring_search_attach_y_ratio=getattr(self.opts, "earring_search_attach_y_ratio", 0.78),
            earring_align_to_target=getattr(self.opts, "earring_align_to_target", True),
            earring_align_strength=getattr(self.opts, "earring_align_strength", 1.0),
            earring_align_max_shift=getattr(self.opts, "earring_align_max_shift", 26),
            earring_attach_y_ratio=getattr(self.opts, "earring_attach_y_ratio", 0.78),
            earring_guarded_composite=getattr(self.opts, "earring_guarded_composite", True),
            earring_composite_strength=getattr(self.opts, "earring_composite_strength", 0.95),
            earring_composite_feather=getattr(self.opts, "earring_composite_feather", 3),
            earring_composite_sigma=getattr(self.opts, "earring_composite_sigma", 1.2),
            earring_composite_strict_mask=getattr(self.opts, "earring_composite_strict_mask", False),
            earring_composite_visible_gate_strength=getattr(self.opts, "earring_composite_visible_gate_strength", 0.45),
            earring_composite_target_hair_gate_strength=getattr(self.opts, "earring_composite_target_hair_gate_strength", 0.35),
            earring_composite_hair_block_gate_strength=getattr(self.opts, "earring_composite_hair_block_gate_strength", 0.45),
            earring_composite_alpha_floor=getattr(self.opts, "earring_composite_alpha_floor", 0.30),
            earring_composite_max_area_frac=getattr(self.opts, "earring_composite_max_area_frac", 0.020),
            earring_composite_placement_dilate=getattr(self.opts, "earring_composite_placement_dilate", 17),
            earring_composite_search_restrict_strength=getattr(
                self.opts,
                "earring_composite_search_restrict_strength",
                1.0,
            ),
            earring_locator_ear_roi_dilate=getattr(self.opts, "earring_locator_ear_roi_dilate", 11),
            earring_locator_parser_support_dilate=getattr(self.opts, "earring_locator_parser_support_dilate", 5),
            earring_locator_object_grow_iters=getattr(self.opts, "earring_locator_object_grow_iters", 1),
            earring_locator_component_keep_top=getattr(self.opts, "earring_locator_component_keep_top", 8),
            earring_locator_component_max_area_ratio=getattr(self.opts, "earring_locator_component_max_area_ratio", 0.025),
            earring_locator_weak_high=getattr(self.opts, "earring_locator_weak_high", 0.018),
            earring_locator_weak_chroma=getattr(self.opts, "earring_locator_weak_chroma", 0.030),
            earring_locator_weak_contrast=getattr(self.opts, "earring_locator_weak_contrast", 0.016),
            earring_locator_placement_dilate=getattr(self.opts, "earring_locator_placement_dilate", 25),
            earring_locator_object_dilate=getattr(self.opts, "earring_locator_object_dilate", 3),
            earring_fine_gate_dilate=getattr(self.opts, "earring_fine_gate_dilate", 7),
            earring_locator_fine_gate_dilate=getattr(self.opts, "earring_locator_fine_gate_dilate", 9),
            earring_source_color_connect_dilate=getattr(self.opts, "earring_source_color_connect_dilate", 11),
            earring_aligned_guard_dilate=getattr(self.opts, "earring_aligned_guard_dilate", 9),
            earring_target_safe_outside_margin=getattr(self.opts, "earring_target_safe_outside_margin", 34),
            earring_target_safe_inside_margin=getattr(self.opts, "earring_target_safe_inside_margin", 5),
            earring_target_safe_face_erode=getattr(self.opts, "earring_target_safe_face_erode", 15),
            earring_target_safe_placement_dilate=getattr(self.opts, "earring_target_safe_placement_dilate", 5),
            earring_target_safe_query_dilate=getattr(self.opts, "earring_target_safe_query_dilate", 3),
            enable_cleanup_face_refiner=getattr(self.opts, "enable_cleanup_face_refiner", False),
            cleanup_face_hidden=getattr(self.opts, "cleanup_face_hidden", 128),
            cleanup_face_strength=getattr(self.opts, "cleanup_face_strength", 1.0),
            cleanup_face_dilate=getattr(self.opts, "cleanup_face_dilate", 5),
            cleanup_face_exclude_earring_dilate=getattr(self.opts, "cleanup_face_exclude_earring_dilate", 13),
        )
        self.post_process = PostProcessModelV54(pp_args).to(self.opts.device).eval()

        checkpoint_path = getattr(self.opts, "pp_v54_checkpoint", None) or getattr(self.opts, "pp_checkpoint", None)
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
        I_final, _ = self.post_process.render_refined(self.net.generator, S_final, F_final, aux)

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending", "blending.png", I_blend)
            save_latents(output_dir, "Blending", "blending.npz", S_blend=S_blend)
            save_gen_image(output_dir, "Final", "final.png", I_final)
            save_latents(output_dir, "Final", "final.npz", S_final=S_final, F_final=F_final)

        return ((I_final[0] + 1) / 2).clip(0, 1)
