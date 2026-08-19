import torch

from models.Blending import Blending
from models.FringeRefiner_v7 import FringeRefinerV7
from utils.save_utils import save_gen_image, save_latents, save_vis_mask


class Blending_v7(Blending):
    def __init__(self, opts, net=None):
        super().__init__(opts, net=net)
        self.fringe_refiner = FringeRefinerV7(opts).to(self.opts.device)

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

        if I_1 is not I_3 or I_1 is not I_2:
            S_blend_6_18 = self.blending_encoder(latent_S_1[:, 6:], latent_S_3[:, 6:], I_1 * target_mask, I_3 * HM_3E)
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        I_blend, _ = self.net.generator([S_blend], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=latent_F_align)
        if getattr(self.opts, 'disable_post_process', False):
            final_image = ((I_blend[0] + 1) / 2).clip(0, 1)
            return final_image

        I_blend_256 = self.downsample_256(I_blend)
        S_final, F_final = self.post_process(I_1, I_blend_256)
        reference_image = 0.5 * (I_2 + I_3)
        fringe_outputs = self.fringe_refiner(
            source_mask=name_to_embed['face']['mask'],
            target_hair_mask=HM_X,
            source_image=I_1,
            reference_image=reference_image,
            blending_image=I_blend_256,
            global_style=S_final,
            global_feature=F_final,
            generator=self.net.generator,
        )
        I_final = fringe_outputs['final_image']

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get('exp_name')) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name

            save_gen_image(output_dir, 'Blending_v7', 'blending.png', I_blend)
            save_latents(
                output_dir,
                'Blending_v7',
                'blending.npz',
                S_blend=S_blend,
                S_final=S_final,
                F_final=F_final,
                F_fused=fringe_outputs['fused_feature'],
            )
            save_vis_mask(output_dir, 'Blending_v7', 'fringe_roi_mask.png', fringe_outputs['roi_mask'])
            save_vis_mask(output_dir, 'Blending_v7', 'fringe_boundary_mask.png', fringe_outputs['boundary_mask'])
            save_vis_mask(output_dir, 'Blending_v7', 'fringe_mask.png', fringe_outputs['fringe_mask_256'])
            save_gen_image(output_dir, 'Blending_v7', 'global_fringe.png', fringe_outputs['global_image'])
            save_gen_image(output_dir, 'Blending_v7', 'refined_256.png', fringe_outputs['refined_image_256'])

            save_gen_image(output_dir, 'Final_v7', 'final.png', I_final)
            save_latents(output_dir, 'Final_v7', 'final.npz', S_final=S_final, F_final=fringe_outputs['fused_feature'])

        final_image = ((I_final[0] + 1) / 2).clip(0, 1)
        return final_image
