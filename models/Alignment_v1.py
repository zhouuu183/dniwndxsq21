import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn

from models.CtrlHair.shape_branch.config import cfg as cfg_mask
from models.CtrlHair.shape_branch.solver import Solver as SolverMask
from models.CtrlHair.shape_branch.solver import get_hair_face_code, get_new_shape
from models.Encoders import RotateModel
from models.Net import Net, get_segmentation
from models.sean_codes.models.pix2pix_model import Pix2PixModel, SEAN_OPT, decode_sean, encode_sean
from utils.image_utils import DilateErosion
from utils.mask_delta_v1 import binary_dilate, compute_delta_masks
from utils.save_utils import save_gen_image, save_latents, save_vis_mask


# v1 modification:
# Alignment now returns explicit edit-delta masks.
# The original code only exposed HM_X. v1 decomposes the edit into
# source/target/add/remove/keep/boundary masks for the new blending stage.


class Alignment(nn.Module):
    def __init__(self, opts, latent_encoder=None, net=None):
        super().__init__()
        self.opts = opts
        self.latent_encoder = latent_encoder
        self.net = net if net is not None else Net(self.opts)

        self.sean_model = Pix2PixModel(SEAN_OPT)
        self.sean_model.eval()

        solver_mask = SolverMask(cfg_mask, device=self.opts.device, local_rank=-1, training=False)
        self.mask_generator = solver_mask.gen
        self.mask_generator.load_state_dict(torch.load("pretrained_models/ShapeAdaptor/mask_generator.pth"))

        self.rotate_model = RotateModel()
        self.rotate_model.load_state_dict(torch.load(self.opts.rotate_checkpoint)["model_state_dict"])
        self.rotate_model.to(self.opts.device).eval()

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.to_bisenet = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def _binary_hair_mask(self, parsing_mask: torch.Tensor) -> torch.Tensor:
        return torch.where(parsing_mask == 13, torch.ones_like(parsing_mask), torch.zeros_like(parsing_mask)).float()

    def _binary_accessory_mask(self, parsing_mask: torch.Tensor) -> torch.Tensor:
        # v1 refinement:
        # Treat hats and head-covering cloth near the upper head area as
        # removable source content. This helps long-to-short edits and reduces
        # hat/headscarf remnants in the blending stage.
        hair = self._binary_hair_mask(parsing_mask)
        hat = torch.where(parsing_mask == 14, torch.ones_like(parsing_mask), torch.zeros_like(parsing_mask)).float()
        cloth = torch.where(parsing_mask == 18, torch.ones_like(parsing_mask), torch.zeros_like(parsing_mask)).float()

        height, width = parsing_mask.shape[-2:]
        y_coords = torch.linspace(0.0, 1.0, steps=height, device=parsing_mask.device).view(1, 1, height, 1)
        x_coords = torch.linspace(0.0, 1.0, steps=width, device=parsing_mask.device).view(1, 1, 1, width)
        upper_head_region = ((y_coords < 0.72) & (x_coords > 0.10) & (x_coords < 0.90)).float()
        hair_neighborhood = binary_dilate(hair, iterations=15)

        # v1 refinement:
        # Keep this conservative. We only treat cloth as removable if it is in
        # the upper head zone or tightly attached to the source hair region,
        # which avoids deleting unrelated clothing or side people.
        cloth_head_cover = cloth * ((upper_head_region + hair_neighborhood) > 0).float()
        return (hat + cloth_head_cover).clamp(0.0, 1.0)

    def _binary_source_edit_mask(self, parsing_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hair = self._binary_hair_mask(parsing_mask)
        accessory = self._binary_accessory_mask(parsing_mask)
        editable = (hair + accessory).clamp(0.0, 1.0)
        return editable, hair, accessory

    @torch.inference_mode()
    def shape_module(self, im_name1: str, im_name2: str, name_to_embed, **kwargs):
        device = self.opts.device

        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]

        latent_W_1 = name_to_embed[im_name1]["W"]
        latent_W_2 = name_to_embed[im_name2]["W"]

        inp_mask1 = name_to_embed[im_name1]["mask"]
        inp_mask2 = name_to_embed[im_name2]["mask"]

        if img1_in is not img2_in:
            rotate_to = self.rotate_model(latent_W_2[:, :6], latent_W_1[:, :6])
            rotate_to = torch.cat((rotate_to, latent_W_2[:, 6:]), dim=1)
            I_rot, _ = self.net.generator([rotate_to], input_is_latent=True, return_latents=False)

            I_rot_to_seg = self.to_bisenet(((I_rot + 1) / 2).clip(0, 1))
            rot_mask = get_segmentation(I_rot_to_seg)
        else:
            I_rot = None
            rot_mask = inp_mask2

        if img1_in is not img2_in:
            face_1, _ = get_hair_face_code(self.mask_generator, inp_mask1[0, 0, ...])
            _, hair_2 = get_hair_face_code(self.mask_generator, rot_mask[0, 0, ...])
            target_mask = get_new_shape(self.mask_generator, face_1, hair_2)[None, None]
        else:
            target_mask = inp_mask1

        hair_mask_source_edit, hair_mask_source, accessory_mask_source = self._binary_source_edit_mask(inp_mask1)
        hair_mask_source_edit = hair_mask_source_edit.to(device)
        hair_mask_source = hair_mask_source.to(device)
        accessory_mask_source = accessory_mask_source.to(device)
        hair_mask_reference = self._binary_hair_mask(inp_mask2).to(device)
        hair_mask_target = self._binary_hair_mask(target_mask).to(device)

        delta_masks = compute_delta_masks(
            hair_mask_source_edit,
            hair_mask_target,
            boundary_width=max(1, getattr(self.opts, "delta_boundary", self.opts.smooth)),
        )
        delta_masks["M_accessory"] = accessory_mask_source
        delta_masks["M_src_hair"] = hair_mask_source
        delta_masks["M_ref"] = hair_mask_reference
        delta_masks["M_rot_ref"] = self._binary_hair_mask(rot_mask).to(device)

        result = {
            "source_mask": inp_mask1,
            "reference_mask": inp_mask2,
            "rotated_reference_mask": rot_mask,
            "target_mask": target_mask,
            "HM_X": hair_mask_target,
            "delta_masks": delta_masks,
        }

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            if I_rot is not None:
                save_gen_image(output_dir, "Shape_v1", f"{im_name2}_rotate_to_{im_name1}.png", I_rot)
            save_vis_mask(output_dir, "Shape_v1", f"mask_{im_name1}.png", inp_mask1)
            save_vis_mask(output_dir, "Shape_v1", f"mask_{im_name2}.png", inp_mask2)
            save_vis_mask(output_dir, "Shape_v1", f"mask_{im_name1}_{im_name2}_target.png", target_mask)

        return result

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]

        latent_S_1, latent_F_1 = name_to_embed[im_name1]["S"], name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]

        shape_info = self.shape_module(im_name1, im_name2, name_to_embed, **kwargs)
        inp_mask1 = shape_info["source_mask"]
        inp_mask2 = shape_info["reference_mask"]
        target_mask = shape_info["target_mask"]
        delta_masks = shape_info["delta_masks"]
        hair_mask_source = delta_masks["M_src"]
        hair_mask_reference = delta_masks["M_ref"]
        hair_mask_target = delta_masks["M_tgt"]

        if img1_in is img2_in:
            return {
                "latent_F_align": latent_F_1,
                "HM_X": hair_mask_target,
                "delta_masks": delta_masks,
                "source_mask": inp_mask1,
                "target_mask": target_mask,
            }

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([inp_mask1, inp_mask2], dim=0)

        gen1_code, gen2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = decode_sean(self.sean_model, gen1_code.unsqueeze(0), target_mask)
        gen2_sean = decode_sean(self.sean_model, gen2_code.unsqueeze(0), target_mask)

        enc_imgs = self.latent_encoder([gen1_sean, gen2_sean])
        intermediate_align = enc_imgs["F"][0].unsqueeze(0)
        latent_inter = enc_imgs["W"][0].unsqueeze(0)
        latent_F_out_new = enc_imgs["F"][1].unsqueeze(0)
        latent_out = enc_imgs["W"][1].unsqueeze(0)

        masks = [
            1 - (1 - hair_mask_source) * (1 - hair_mask_target),
            hair_mask_target,
            hair_mask_reference * hair_mask_target,
        ]
        masks = torch.cat(masks, dim=0)

        dilate, erosion = self.dilate_erosion.mask(masks)
        free_mask = torch.stack([dilate[0], erosion[1], erosion[2]], dim=0)
        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode="bicubic")
        interpolation_low = 1 - free_mask_down_32

        latent_F_align = intermediate_align + interpolation_low[0] * (latent_F_1 - intermediate_align)
        latent_F_align = latent_F_out_new + interpolation_low[1] * (latent_F_align - latent_F_out_new)
        latent_F_align = latent_F_2 + interpolation_low[2] * (latent_F_align - latent_F_2)

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Align_v1", f"{im_name1}_{im_name2}_SEAN.png", gen1_sean)
            save_gen_image(output_dir, "Align_v1", f"{im_name2}_{im_name1}_SEAN.png", gen2_sean)

            img1_e4e = self.net.generator(
                [latent_inter],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=intermediate_align,
            )[0]
            img2_e4e = self.net.generator(
                [latent_out],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_out_new,
            )[0]

            save_gen_image(output_dir, "Align_v1", f"{im_name1}_{im_name2}_e4e.png", img1_e4e)
            save_gen_image(output_dir, "Align_v1", f"{im_name2}_{im_name1}_e4e.png", img2_e4e)
            save_latents(output_dir, "Align_v1", f"{im_name1}_{im_name2}_F.npz", latent_F_align=latent_F_align)

        return {
            "latent_F_align": latent_F_align,
            "HM_X": hair_mask_target,
            "delta_masks": delta_masks,
            "source_mask": inp_mask1,
            "target_mask": target_mask,
            "latent_S_face": latent_S_1,
        }
