from __future__ import annotations

import argparse

import torch

from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.postprocess_v6 import PostProcessModelV6
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_formula_v6 import ensure_mask_4d
from utils.save_utils import save_gen_image, save_latents


class Blending_v6(torch.nn.Module):
    """
    v6 keeps the original blending encoder, but swaps in the new alignment masks
    and the new post-process refinement.
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.net = Net(self.opts) if net is None else net

        blending_checkpoint = torch.load(self.opts.blending_checkpoint, map_location="cpu")
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get("clip", "ViT-B/32"))
        self.blending_encoder.load_state_dict(blending_checkpoint["model_state_dict"], strict=False)
        self.blending_encoder.to(self.opts.device).eval()

        pp_args = argparse.Namespace(
            use_mod=getattr(self.opts, "pp_v6_use_mod", True),
            pretrain=False,
            finetune=getattr(self.opts, "pp_v6_finetune", False),
            diff_mask_dilate=getattr(self.opts, "diff_mask_dilate", 5),
            diff_mask_blur_kernel=getattr(self.opts, "diff_mask_blur_kernel", 11),
            diff_mask_blur_sigma=getattr(self.opts, "diff_mask_blur_sigma", 0.0),
        )
        self.post_process = PostProcessModelV6(pp_args).to(self.opts.device).eval()
        self.post_process.load_base_checkpoint(getattr(self.opts, "pp_v6_checkpoint", self.opts.pp_checkpoint))

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        return_aux = kwargs.get("return_aux", False)

        i_face = name_to_embed["face"]["image_norm_256"]
        i_shape = name_to_embed["shape"]["image_norm_256"]
        i_color = name_to_embed["color"]["image_norm_256"]

        face_mask = torch.where(
            name_to_embed["face"]["mask"] == 13,
            torch.ones_like(name_to_embed["face"]["mask"]),
            torch.zeros_like(name_to_embed["face"]["mask"]),
        ).float()
        color_mask = torch.where(
            name_to_embed["color"]["mask"] == 13,
            torch.ones_like(name_to_embed["color"]["mask"]),
            torch.zeros_like(name_to_embed["color"]["mask"]),
        ).float()

        hm_1d, _ = self.dilate_erosion.mask(face_mask)
        _, hm_3e = self.dilate_erosion.mask(color_mask)
        hm_x = ensure_mask_4d(align_color.get("HM_X", align_shape["HM_X"])).float()
        hm_xd, _ = self.dilate_erosion.mask(hm_x)
        default_face_mask = ((1.0 - hm_1d) * (1.0 - hm_xd)).clamp(0, 1)
        face_guidance_mask = ensure_mask_4d(align_shape.get("M_valid", default_face_mask)).float()

        latent_s_face = name_to_embed["face"]["S"]
        latent_s_color = name_to_embed["color"]["S"]
        latent_f_align = align_shape["latent_F_align"]

        if i_face is not i_color or i_face is not i_shape:
            s_blend_6_18 = self.blending_encoder(
                latent_s_face[:, 6:],
                latent_s_color[:, 6:],
                i_face * face_guidance_mask,
                i_color * hm_3e,
            )
            s_blend = torch.cat((latent_s_face[:, :6], s_blend_6_18), dim=1)
        else:
            s_blend = latent_s_face

        i_blend, _ = self.net.generator(
            [s_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_f_align,
        )
        i_blend_256 = self.downsample_256(i_blend)

        s_final, f_final, pp_aux = self.post_process(
            i_face,
            i_blend_256,
            target_mask=default_face_mask,
            HT_E=hm_3e,
            diff_mask=align_shape.get("M_diff"),
            valid_mask=align_shape.get("M_valid"),
            source_hair_mask=align_shape.get("source_hair_mask", face_mask),
            target_hair_mask=hm_xd,
        )
        i_final, _ = self.net.generator(
            [s_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=f_final,
        )

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v6", "blending.png", i_blend)
            save_latents(output_dir, "Blending_v6", "blending.npz", S_blend=s_blend, F_align=latent_f_align)
            save_gen_image(output_dir, "Final_v6", "final.png", i_final)
            save_latents(output_dir, "Final_v6", "final.npz", S_final=s_final, F_final=f_final)

        final_image = ((i_final[0] + 1) / 2).clip(0, 1)
        if not return_aux:
            return final_image

        return {
            "final_image": final_image,
            "pre_pp_image_1024": i_blend,
            "pre_pp_image_256": i_blend_256,
            "blend_mask": default_face_mask,
            "hair_color_mask": hm_3e,
            "align_shape": align_shape,
            "align_color": align_color,
            "pp_aux": pp_aux,
            "S_blend": s_blend,
            "S_final": s_final,
            "F_final": f_final,
        }
