import torch
from torch import nn

from models.Encoders import ClipBlendingModel, PostProcessModel
from models.Net import Net
from utils.bicubic import BicubicDownSample
from utils.blending_checkpoint_v8 import validate_blending_checkpoint_policy_v8
from utils.hair_color_match_v8 import match_hair_color_lab_v8
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents


class Blending(nn.Module):
    """
    Module for transferring the desired hair color and post processing
    """

    requires_blending_checkpoint_policy_v8 = False

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net

        blending_checkpoint = torch.load(self.opts.blending_checkpoint, map_location="cpu")
        if self.requires_blending_checkpoint_policy_v8:
            self.blending_color_policy_v8 = validate_blending_checkpoint_policy_v8(
                blending_checkpoint,
                self.opts,
                checkpoint_path=self.opts.blending_checkpoint,
            )
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get('clip', "ViT-B/32"))
        blending_load = self.blending_encoder.load_state_dict(
            blending_checkpoint['model_state_dict'], strict=False
        )
        if blending_load.missing_keys or blending_load.unexpected_keys:
            print(
                "[Blending] checkpoint mismatch: "
                f"missing={len(blending_load.missing_keys)}, "
                f"unexpected={len(blending_load.unexpected_keys)}"
            )
        self.blending_encoder.to(self.opts.device).eval()
        blend_strength_override = float(getattr(self.opts, "blend_color_strength_v8", 0.0) or 0.0)
        if blend_strength_override > 0:
            self.blend_color_strength = blend_strength_override
        else:
            self.blend_color_strength = float(blending_checkpoint.get("blend_color_strength", 1.0))

        self.post_process = PostProcessModel().to(self.opts.device).eval()
        self.post_process.load_state_dict(
            torch.load(self.opts.pp_checkpoint, map_location="cpu")['model_state_dict']
        )

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        I_1 = name_to_embed['face']['image_norm_256']
        I_2 = name_to_embed['shape']['image_norm_256']
        I_3 = name_to_embed['color']['image_norm_256']

        mask_de = self.dilate_erosion.hair_from_mask(
            torch.cat([name_to_embed[x]['mask'] for x in ['face', 'color']], dim=0)
        )
        HM_1D, _ = mask_de[0][0].unsqueeze(0), mask_de[1][0].unsqueeze(0)
        HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)

        latent_S_1, latent_F_align = name_to_embed['face']['S'], align_shape['latent_F_align']
        HM_X = align_color['HM_X']

        latent_S_3 = name_to_embed['color']["S"]

        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        # Blending
        if I_1 is not I_3 or I_1 is not I_2:
            S_blend_6_18_raw = self.blending_encoder(
                latent_S_1[:, 6:],
                latent_S_3[:, 6:],
                I_1 * target_mask,
                I_3 * HM_3E,
            )
            S_blend_6_18 = latent_S_1[:, 6:] + self.blend_color_strength * (S_blend_6_18_raw - latent_S_1[:, 6:])
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        I_blend, _ = self.net.generator([S_blend], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=latent_F_align)
        I_blend_256 = self.downsample_256(I_blend)

        # Post Process
        S_final, F_final = self.post_process(I_1, I_blend_256)
        I_final, _ = self.net.generator([S_final], input_is_latent=True, return_latents=False,
                                         start_layer=5, end_layer=8, layer_in=F_final)
        I_final_01 = ((I_final + 1) / 2).clamp(0, 1)
        if not getattr(self.opts, "disable_exact_hair_color_match_v8", False):
            I_final_01 = match_hair_color_lab_v8(
                I_final_01,
                I_3,
                HM_X,
                HM_3E,
                strength=float(getattr(self.opts, "exact_hair_color_strength_v8", 1.0)),
                chroma_strength=float(getattr(self.opts, "exact_hair_color_chroma_strength_v8", 1.0)),
                luma_strength=float(getattr(self.opts, "exact_hair_color_luma_strength_v8", 1.0)),
                std_strength=float(getattr(self.opts, "exact_hair_color_std_strength_v8", 1.0)),
                alpha_blur_radius=int(getattr(self.opts, "exact_hair_color_alpha_blur_v8", 5)),
            )

        I_final_to_save = I_final_01 * 2 - 1

        if self.opts.save_all:
            exp_name = kwargs.get('exp_name')
            exp_name = exp_name if exp_name is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, 'Blending', 'blending.png', I_blend)
            save_latents(output_dir, 'Blending', 'blending.npz', S_blend=S_blend)

            save_gen_image(output_dir, 'Final', 'final.png', I_final_to_save)
            save_latents(output_dir, 'Final', 'final.npz', S_final=S_final, F_final=F_final)

        return I_final_01[0]
