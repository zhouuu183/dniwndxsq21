from __future__ import annotations

import argparse

import torch
from torch import nn

from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.postprocess_v51 import PostProcessModelV51
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents


class BlendingV51(nn.Module):
    """
    V8-cleaned blending + V51 refinement entry.

    The alignment input is produced by Alignment_v8/SATD. This class mirrors
    the v8 blending mask logic, then replaces only the final PP stage with the
    ear/detail-aware PostProcessModelV51.
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
            use_mod=getattr(self.opts, "pp_v51_use_mod", True),
            use_full=getattr(self.opts, "pp_v51_use_full", True),
            pretrain=False,
            finetune=False,
            ear_parse_size=getattr(self.opts, "ear_parse_size", 512),
            ear_feature_channels=getattr(self.opts, "ear_feature_channels", 128),
            ear_low_alpha=getattr(self.opts, "ear_low_alpha", 0.1),
            ear_dilate=getattr(self.opts, "ear_dilate", 21),
            hair_change_dilate=getattr(self.opts, "hair_change_dilate", 25),
            earring_expand=getattr(self.opts, "earring_expand", 15),
            ear_downward_shift=getattr(self.opts, "ear_downward_shift", 10),
            target_hair_dilate=getattr(self.opts, "target_hair_dilate", 11),
            source_hair_block_dilate=getattr(self.opts, "source_hair_block_dilate", 5),
            source_hair_block_strength=getattr(self.opts, "source_hair_block_strength", 0.95),
            target_visibility_expand=getattr(self.opts, "target_visibility_expand", 5),
            max_target_hair_overlap=getattr(self.opts, "max_target_hair_overlap", 0.55),
            ear_blur_kernel=getattr(self.opts, "ear_blur_kernel", 11),
            ear_blur_sigma=getattr(self.opts, "ear_blur_sigma", 3.0),
            ear_mask_hidden=getattr(self.opts, "ear_mask_hidden", 32),
        )
        self.post_process = PostProcessModelV51(pp_args).to(self.opts.device).eval()

        checkpoint_path = getattr(self.opts, "pp_v51_checkpoint", None) or getattr(self.opts, "pp_checkpoint", None)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.post_process.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

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

        return ((I_final[0] + 1) / 2).clip(0, 1)

