from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn

from models.CtrlHair.shape_branch.config import cfg as cfg_mask
from models.CtrlHair.shape_branch.solver import get_hair_face_code, get_new_shape, Solver as SolverMask
from models.Encoders import RotateModel
from models.Net import Net, get_segmentation
from models.sean_codes.models.pix2pix_model import Pix2PixModel, SEAN_OPT, encode_sean, decode_sean
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents, save_vis_mask


def _binary_hair_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask == 13, torch.ones_like(mask), torch.zeros_like(mask)).float()


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def _morphological_boundary(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = _odd_kernel(kernel_size)
    if kernel_size <= 1:
        return torch.zeros_like(mask)
    dilated = F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    eroded = 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (dilated - eroded).clamp(0.0, 1.0)


def _blur_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = _odd_kernel(kernel_size)
    if kernel_size <= 1:
        return mask.float().clamp(0.0, 1.0)
    return F.avg_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).clamp(0.0, 1.0)


class Alignment_v10(nn.Module):
    """
    Baseline HairFast alignment with v10 consistency and boundary-alpha additions.

    The implementation keeps the baseline SEAN -> E4E -> F-space alignment path, but
    exposes the masks/features needed by the v10 blending trainer:
    source_hair_mask, donor_hair_mask, target_hair_mask, latent_F_hair, and
    boundary_alpha_256.
    """

    def __init__(self, opts, latent_encoder=None, net=None):
        super().__init__()
        self.opts = opts
        self.latent_encoder = latent_encoder
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net

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

    def _boundary_width(self, kwargs: dict) -> int:
        return int(kwargs.get("alpha_boundary_width_v10", getattr(self.opts, "alpha_boundary_width_v10", 9)))

    def _boundary_strength(self, kwargs: dict) -> float:
        return float(kwargs.get("alpha_boundary_strength_v10", getattr(self.opts, "alpha_boundary_strength_v10", 0.65)))

    def _resolve_alpha_256(self, im_name: str, name_to_embed, donor_hair_mask: torch.Tensor, kwargs: dict) -> torch.Tensor:
        alpha_mattes = kwargs.get("alpha_mattes")
        if isinstance(alpha_mattes, dict) and im_name in alpha_mattes:
            alpha = alpha_mattes[im_name]
        else:
            alpha = name_to_embed.get(im_name, {}).get("alpha_256")

        if alpha is None:
            blur_kernel = int(kwargs.get("alpha_fallback_blur_v10", getattr(self.opts, "alpha_fallback_blur_v10", 9)))
            alpha = _blur_mask(donor_hair_mask, blur_kernel)
        if alpha.dim() == 3:
            alpha = alpha.unsqueeze(0)
        if alpha.shape[-2:] != donor_hair_mask.shape[-2:]:
            alpha = F.interpolate(alpha.float(), size=donor_hair_mask.shape[-2:], mode="bilinear", align_corners=False)
        return alpha.to(donor_hair_mask.device, donor_hair_mask.dtype).clamp(0.0, 1.0)

    def _make_shape_info(self, inp_mask1: torch.Tensor, inp_mask2: torch.Tensor, target_mask: torch.Tensor,
                         im_name2: str, name_to_embed, kwargs: dict) -> dict[str, torch.Tensor]:
        source_hair_mask = _binary_hair_mask(inp_mask1)
        donor_hair_mask = _binary_hair_mask(inp_mask2)
        target_hair_mask = _binary_hair_mask(target_mask)
        boundary_mask = _morphological_boundary(target_hair_mask, self._boundary_width(kwargs))
        donor_alpha = self._resolve_alpha_256(im_name2, name_to_embed, donor_hair_mask, kwargs)
        boundary_alpha = (donor_alpha * boundary_mask).clamp(0.0, 1.0)
        return {
            "source_parsing_mask": inp_mask1,
            "donor_parsing_mask": inp_mask2,
            "target_parsing_mask": target_mask,
            "source_hair_mask": source_hair_mask,
            "donor_hair_mask": donor_hair_mask,
            "target_hair_mask": target_hair_mask,
            "HM_X": target_hair_mask,
            "boundary_mask_256": boundary_mask,
            "boundary_alpha_256": boundary_alpha,
        }

    @torch.inference_mode()
    def shape_module(self, im_name1: str, im_name2: str, name_to_embed, only_target=True, **kwargs):
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
            I_rot_to_seg = ((I_rot + 1) / 2).clip(0, 1)
            I_rot_to_seg = self.to_bisenet(I_rot_to_seg)
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

        target_mask = target_mask.to(device)
        shape_info = self._make_shape_info(inp_mask1, inp_mask2, target_mask, im_name2, name_to_embed, kwargs)

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            if I_rot is not None:
                save_gen_image(output_dir, "Shape_v10", f"{im_name2}_rotate_to_{im_name1}.png", I_rot)
            save_vis_mask(output_dir, "Shape_v10", f"mask_{im_name1}.png", inp_mask1)
            save_vis_mask(output_dir, "Shape_v10", f"mask_{im_name2}.png", inp_mask2)
            save_vis_mask(output_dir, "Shape_v10", f"mask_{im_name2}_rotate_to_{im_name1}.png", rot_mask)
            save_vis_mask(output_dir, "Shape_v10", f"mask_{im_name1}_{im_name2}_target.png", target_mask)

        if only_target:
            return shape_info
        return shape_info

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_S_1 = name_to_embed[im_name1]["S"]
        latent_F_1 = name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]

        shape_info = self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        if img1_in is img2_in:
            return {
                "latent_F_align": latent_F_1,
                "latent_F_src": latent_F_1,
                "latent_F_hair": latent_F_1,
                **shape_info,
            }

        inp_mask1 = shape_info["source_parsing_mask"]
        inp_mask2 = shape_info["donor_parsing_mask"]
        target_mask = shape_info["target_parsing_mask"]
        source_hair_mask = shape_info["source_hair_mask"]
        donor_hair_mask = shape_info["donor_hair_mask"]
        target_hair_mask = shape_info["target_hair_mask"]

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([inp_mask1, inp_mask2], dim=0)

        img1_code, img2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask)
        gen2_sean = decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask)

        enc_imgs = self.latent_encoder([gen1_sean, gen2_sean])
        intermediate_align = enc_imgs["F"][0].unsqueeze(0)
        latent_inter = enc_imgs["W"][0].unsqueeze(0)
        latent_F_out_new = enc_imgs["F"][1].unsqueeze(0)
        latent_out = enc_imgs["W"][1].unsqueeze(0)

        masks = [
            1 - (1 - source_hair_mask) * (1 - target_hair_mask),
            target_hair_mask,
            donor_hair_mask * target_hair_mask,
        ]
        masks = torch.cat(masks, dim=0)
        dilate, erosion = self.dilate_erosion.mask(masks)
        free_mask = torch.stack([dilate[0], erosion[1], erosion[2]], dim=0)
        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode="bicubic").clamp(0.0, 1.0)
        interpolation_low = 1 - free_mask_down_32

        latent_F_align = intermediate_align + interpolation_low[0] * (latent_F_1 - intermediate_align)
        latent_F_align = latent_F_out_new + interpolation_low[1] * (latent_F_align - latent_F_out_new)
        latent_F_align = latent_F_2 + interpolation_low[2] * (latent_F_align - latent_F_2)

        boundary_mask_32 = F.interpolate(shape_info["boundary_mask_256"], size=(32, 32), mode="bicubic").clamp(0.0, 1.0)
        boundary_alpha_32 = F.interpolate(shape_info["boundary_alpha_256"], size=(32, 32), mode="bicubic").clamp(0.0, 1.0)
        boundary_strength = self._boundary_strength(kwargs)
        boundary_weight = (boundary_strength * boundary_mask_32).clamp(0.0, 1.0)
        boundary_feature = boundary_alpha_32 * latent_F_2 + (1.0 - boundary_alpha_32) * latent_F_1
        latent_F_align = latent_F_align * (1.0 - boundary_weight) + boundary_feature * boundary_weight

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Align_v10", f"{im_name1}_{im_name2}_SEAN.png", gen1_sean)
            save_gen_image(output_dir, "Align_v10", f"{im_name2}_{im_name1}_SEAN.png", gen2_sean)
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
            save_gen_image(output_dir, "Align_v10", f"{im_name1}_{im_name2}_e4e.png", img1_e4e)
            save_gen_image(output_dir, "Align_v10", f"{im_name2}_{im_name1}_e4e.png", img2_e4e)
            gen_im, _ = self.net.generator(
                [latent_S_1],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_align,
            )
            save_gen_image(output_dir, "Align_v10", f"{im_name1}_{im_name2}_output.png", gen_im)
            save_latents(
                output_dir,
                "Align_v10",
                f"{im_name1}_{im_name2}_F.npz",
                latent_F_align=latent_F_align,
                latent_F_hair=latent_F_2,
                target_hair_mask=target_hair_mask,
                boundary_alpha_256=shape_info["boundary_alpha_256"],
            )

        return {
            "latent_F_align": latent_F_align,
            "latent_F_src": latent_F_1,
            "latent_F_hair": latent_F_2,
            **shape_info,
        }
