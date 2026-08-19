import torch
import torch.nn.functional as F

from models.Alignment import Alignment
from models.sean_codes.models.pix2pix_model import encode_sean, decode_sean
from utils.save_utils import save_gen_image, save_latents


class Alignment_v7(Alignment):
    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]['image_256']
        img2_in = name_to_embed[im_name2]['image_256']
        latent_S_1, latent_F_1 = name_to_embed[im_name1]["S"], name_to_embed[im_name1]["F"]
        latent_S_2, latent_F_2 = name_to_embed[im_name2]["S"], name_to_embed[im_name2]["F"]

        if img1_in is img2_in:
            hair_mask_target = self.shape_module(im_name1, im_name2, name_to_embed, only_target=True, **kwargs)['HM_X']
            return {'latent_F_align': latent_F_1, 'HM_X': hair_mask_target}

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = (
            self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        )

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([inp_mask1, inp_mask2], dim=0)
        img1_code, img2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask)
        gen2_sean = decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask)

        enc_imgs = self.latent_encoder([gen1_sean, gen2_sean])
        intermediate_align, latent_inter = enc_imgs["F"][0].unsqueeze(0), enc_imgs["W"][0].unsqueeze(0)
        latent_F_out_new, latent_out = enc_imgs["F"][1].unsqueeze(0), enc_imgs["W"][1].unsqueeze(0)

        masks = [
            1 - (1 - hair_mask1) * (1 - hair_mask_target),
            hair_mask_target,
            hair_mask2 * hair_mask_target
        ]
        masks = torch.cat(masks, dim=0)

        dilate, erosion = self.dilate_erosion.mask(masks)
        free_mask = [
            dilate[0],
            erosion[1],
            erosion[2]
        ]
        free_mask = torch.stack(free_mask, dim=0)
        align_mask_size = getattr(self.opts, 'align_mask_size', 32)
        free_mask_low = F.interpolate(free_mask.float(), size=(align_mask_size, align_mask_size), mode='bicubic')
        interpolation_low = 1 - free_mask_low

        latent_F_align = intermediate_align + interpolation_low[0] * (latent_F_1 - intermediate_align)
        latent_F_align = latent_F_out_new + interpolation_low[1] * (latent_F_align - latent_F_out_new)
        latent_F_align = latent_F_2 + interpolation_low[2] * (latent_F_align - latent_F_2)

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get('exp_name')) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name

            save_gen_image(output_dir, 'Align_v7', f'{im_name1}_{im_name2}_SEAN.png', gen1_sean)
            save_gen_image(output_dir, 'Align_v7', f'{im_name2}_{im_name1}_SEAN.png', gen2_sean)

            img1_e4e = self.net.generator([latent_inter], input_is_latent=True, return_latents=False, start_layer=4,
                                          end_layer=8, layer_in=intermediate_align)[0]
            img2_e4e = self.net.generator([latent_out], input_is_latent=True, return_latents=False, start_layer=4,
                                          end_layer=8, layer_in=latent_F_out_new)[0]

            save_gen_image(output_dir, 'Align_v7', f'{im_name1}_{im_name2}_e4e.png', img1_e4e)
            save_gen_image(output_dir, 'Align_v7', f'{im_name2}_{im_name1}_e4e.png', img2_e4e)

            gen_im, _ = self.net.generator([latent_S_1], input_is_latent=True, return_latents=False, start_layer=4,
                                           end_layer=8, layer_in=latent_F_align)

            save_gen_image(output_dir, 'Align_v7', f'{im_name1}_{im_name2}_output.png', gen_im)
            save_latents(output_dir, 'Align_v7', f'{im_name1}_{im_name2}_F.npz', latent_F_align=latent_F_align)

        return {'latent_F_align': latent_F_align, 'HM_X': hair_mask_target}
