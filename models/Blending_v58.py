from __future__ import annotations

import argparse

import torch
from torch import nn

from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.postprocess_v58 import PostProcessModelV5, load_checkpoint_compat
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents, save_vis_mask


class BlendingV5(nn.Module):
    """
    V8-cleaned blending + V5 refinement entry.

    The alignment input is produced by Alignment_v8/SATD. This class mirrors
    the v8 blending mask logic, then replaces only the final PP stage with the
    ear/detail-aware PostProcessModelV5.
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

        checkpoint_path = (
            getattr(self.opts, "pp_v58_checkpoint", None)
            or getattr(self.opts, "pp_v5_checkpoint", None)
            or getattr(self.opts, "pp_checkpoint", None)
        )
        checkpoint = load_checkpoint_compat(checkpoint_path, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {}) or {}

        def pp_value(name: str, default):
            return checkpoint_args.get(name, getattr(self.opts, name, default))

        pp_args = argparse.Namespace(
            use_mod=checkpoint_args.get("use_mod", getattr(self.opts, "pp_v58_use_mod", True)),
            use_full=checkpoint_args.get("use_full", getattr(self.opts, "pp_v58_use_full", True)),
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
            source_hair_block_dilate=pp_value("source_hair_block_dilate", 5),
            source_hair_block_strength=pp_value("source_hair_block_strength", 0.6),
            target_visibility_expand=pp_value("target_visibility_expand", 5),
            max_target_hair_overlap=pp_value("max_target_hair_overlap", 0.55),
            min_target_visible_overlap=pp_value("min_target_visible_overlap", 0.02),
            min_target_ear_area=pp_value("min_target_ear_area", 8.0),
            earring_channel_down=pp_value("earring_channel_down", 32),
            ear_blur_kernel=pp_value("ear_blur_kernel", 11),
            ear_blur_sigma=pp_value("ear_blur_sigma", 3.0),
            ear_mask_hidden=pp_value("ear_mask_hidden", 32),
            ear_mask_init_bias=pp_value("ear_mask_init_bias", -4.0),
            use_dataset_query_mask=pp_value("use_dataset_query_mask", False),
            enable_earring_query_recall=pp_value("enable_earring_query_recall", True),
            earring_query_recall_dilate=pp_value("earring_query_recall_dilate", 7),
            earring_query_downward_shift=pp_value("earring_query_downward_shift", 18),
            earring_query_lower_lobe_weight=pp_value("earring_query_lower_lobe_weight", 0.20),
            earring_query_candidate_boost=pp_value("earring_query_candidate_boost", 0.90),
            earring_query_block_protect=pp_value("earring_query_block_protect", 0.85),
            earring_fine_mask_floor=pp_value("earring_fine_mask_floor", 0.18),
            earring_fine_mask_dilate=pp_value("earring_fine_mask_dilate", 5),
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

        S_final, F_final, aux = self.post_process(I_1, I_blend_256, target_mask, HM_3E)
        I_final, _ = self.post_process.render_refined(self.net.generator, S_final, F_final, aux)

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending", "blending.png", I_blend)
            save_latents(output_dir, "Blending", "blending.npz", S_blend=S_blend)
            save_gen_image(output_dir, "Final", "final.png", I_final)
            save_latents(output_dir, "Final", "final.npz", S_final=S_final, F_final=F_final)
            for mask_name in (
                "source_earring_mask",
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
                "fine_mask",
                "prior_mask",
            ):
                if aux.get(mask_name) is not None:
                    save_vis_mask(output_dir, "PostProcessV5Masks", f"{mask_name}.png", aux[mask_name])

        return ((I_final[0] + 1) / 2).clip(0, 1)
