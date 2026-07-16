import torch
from torch import nn

from models.EarringDirectRestore_e1 import EarringDirectRestore
from models.Encoders import ClipBlendingModel, PostProcessModel
from models.Net import Net
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents


class Blending(nn.Module):
    """
    Module for transferring the desired hair color and post processing
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net

        blending_checkpoint = torch.load(self.opts.blending_checkpoint)
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get('clip', "ViT-B/32"))
        self.blending_encoder.load_state_dict(blending_checkpoint['model_state_dict'], strict=False)
        self.blending_encoder.to(self.opts.device).eval()
        self.blend_color_strength = float(blending_checkpoint.get("blend_color_strength", 1.0))

        self.post_process = PostProcessModel().to(self.opts.device).eval()
        self.post_process.load_state_dict(torch.load(self.opts.pp_checkpoint)['model_state_dict'])
        self.earring_restore = EarringDirectRestore(self.opts).to(self.opts.device).eval()

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

    @staticmethod
    def _debug_to_tanh_image(value):
        if value is None:
            return None
        value = value.detach().float().clamp(0, 1)
        return value * 2 - 1

    def _save_earring_restore_debug(self, output_dir, debug):
        for key in (
            "source",
            "pp_out",
            "source_ear_roi",
            "object_mask",
            "weak_seed",
            "weak_support",
            "aligned_object_mask",
            "visibility_gate",
            "safe_region",
            "restore_mask",
            "alpha",
            "final",
        ):
            value = self._debug_to_tanh_image(debug.get(key))
            if value is not None:
                save_gen_image(output_dir, "EarringDirectRestore", f"{key}.png", value)

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

        earring_debug = {}
        use_earring_restore = kwargs.get(
            "use_earring_direct_restore",
            getattr(self.opts, "use_earring_direct_restore", False),
        )
        if use_earring_restore:
            source_full_01 = kwargs.get("source_full_01", name_to_embed["face"]["image_256"])
            I_final_01, earring_debug = self.earring_restore(
                source_image=source_full_01,
                pp_out=I_final_01,
                source_parsing=kwargs.get("earring_source_parsing", name_to_embed["face"]["mask"]),
                target_parsing=kwargs.get("earring_target_parsing", name_to_embed["face"]["mask"]),
                target_hair_mask=kwargs.get("earring_target_hair_mask", align_shape.get("HM_X", HM_X)),
                manual_earring_mask=kwargs.get(
                    "earring_manual_mask",
                    kwargs.get("earring_restore_manual_mask", None),
                ),
                enabled=True,
            )
        I_final_to_save = I_final_01 * 2 - 1

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get('exp_name')) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, 'Blending', 'blending.png', I_blend)
            save_latents(output_dir, 'Blending', 'blending.npz', S_blend=S_blend)

            save_gen_image(output_dir, 'Final', 'final.png', I_final_to_save)
            save_latents(output_dir, 'Final', 'final.npz', S_final=S_final, F_final=F_final)
            if earring_debug and bool(getattr(self.opts, "earring_restore_debug", False)):
                self._save_earring_restore_debug(output_dir, earring_debug)

        return I_final_01[0]
