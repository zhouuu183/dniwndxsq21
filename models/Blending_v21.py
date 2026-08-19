import torch

from models.Blending import Blending
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents


class Blending_v8(Blending):
    """
    v8 keeps the stable blending branch, but makes the implementation
    self-contained so it does not depend on Blending_v4 from the server.
    """

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
        hair_color_mask = HM_3E

        latent_S_1 = name_to_embed["face"]["S"]
        latent_S_3 = name_to_embed["color"]["S"]
        latent_F_align = align_shape["latent_F_align"]
        HM_X = align_shape["HM_X"]

        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        if I_1 is not I_3 or I_1 is not I_2:
            S_blend_6_18_raw = self.blending_encoder(
                latent_S_1[:, 6:],
                latent_S_3[:, 6:],
                I_1 * target_mask,
                I_3 * hair_color_mask,
            )
            S_blend_6_18 = latent_S_1[:, 6:] + self.blend_color_strength * (S_blend_6_18_raw - latent_S_1[:, 6:])
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        I_blend, _ = self.net.generator(
            [S_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_align,
        )
        I_blend_256 = self.downsample_256(I_blend)

        S_final, F_final = self.post_process(I_1, I_blend_256)
        I_final, _ = self.net.generator(
            [S_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=F_final,
        )

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get("exp_name")) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v8", "blending.png", I_blend)
            save_latents(output_dir, "Blending_v8", "blending.npz", S_blend=S_blend)
            save_gen_image(output_dir, "Final_v8", "final.png", I_final)
            save_latents(output_dir, "Final_v8", "final.npz", S_final=S_final, F_final=F_final)

        return ((I_final[0] + 1) / 2).clamp(0, 1)
