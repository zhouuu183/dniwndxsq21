from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn

from models.CtrlHair.shape_branch.config import cfg as cfg_mask
from models.CtrlHair.shape_branch.solver import Solver as SolverMask
from models.CtrlHair.shape_branch.solver import get_hair_face_code, get_new_shape
from models.Encoders import RotateModel
from models.Net import Net, get_segmentation
from models.ShapeAdapter_v12 import TopologyF32Adapter_v12, V12_MASK_KEYS, load_shape_adapter_v12
from models.sean_codes.models.pix2pix_model import SEAN_OPT, Pix2PixModel, decode_sean, encode_sean
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


class Alignment_v12(nn.Module):
    """
    Baseline HairFast alignment plus v12 topology-mask outputs and optional F32 adapter.

    Unlike v10, this class does not directly blend alpha into F32. Alpha is a
    topology cue for the trainable adapter, not a final fusion rule.
    """

    def __init__(self, opts, latent_encoder=None, net=None):
        super().__init__()
        self.opts = opts
        self.latent_encoder = latent_encoder
        self.net = Net(self.opts) if net is None else net

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

        self.shape_adapter = TopologyF32Adapter_v12(
            feature_channels=int(getattr(opts, "shape_adapter_channels_v12", 512)),
            hidden_channels=int(getattr(opts, "shape_adapter_hidden_v12", 512)),
        ).to(self.opts.device)
        self._load_adapter(getattr(opts, "shape_adapter_v12_checkpoint", ""))

    def _load_adapter(self, checkpoint_path: str):
        if not checkpoint_path:
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.opts.device)
        load_shape_adapter_v12(self.shape_adapter, checkpoint)

    def _boundary_width(self, kwargs: dict) -> int:
        return int(kwargs.get("topology_boundary_width_v12", getattr(self.opts, "topology_boundary_width_v12", 9)))

    def _alpha_fallback_blur(self, kwargs: dict) -> int:
        return int(kwargs.get("alpha_fallback_blur_v12", getattr(self.opts, "alpha_fallback_blur_v12", 9)))

    def _bang_top_ratio(self, kwargs: dict) -> float:
        return float(kwargs.get("bang_top_ratio_v12", getattr(self.opts, "bang_top_ratio_v12", 0.56)))

    def _adapter_strength(self, kwargs: dict) -> float:
        return float(kwargs.get("shape_adapter_strength_v12", getattr(self.opts, "shape_adapter_strength_v12", 1.0)))

    def _resolve_alpha_256(self, im_name: str, name_to_embed, donor_hair_mask: torch.Tensor, kwargs: dict) -> torch.Tensor:
        alpha_mattes = kwargs.get("alpha_mattes")
        if isinstance(alpha_mattes, dict) and im_name in alpha_mattes:
            alpha = alpha_mattes[im_name]
        else:
            alpha = name_to_embed.get(im_name, {}).get("alpha_256")
        if alpha is None:
            alpha = _blur_mask(donor_hair_mask, self._alpha_fallback_blur(kwargs))
        if alpha.dim() == 3:
            alpha = alpha.unsqueeze(0)
        if alpha.shape[-2:] != donor_hair_mask.shape[-2:]:
            alpha = F.interpolate(alpha.float(), size=donor_hair_mask.shape[-2:], mode="bilinear", align_corners=False)
        return alpha.to(donor_hair_mask.device, donor_hair_mask.dtype).clamp(0.0, 1.0)

    def _make_topology_masks(
        self,
        source_hair_mask: torch.Tensor,
        donor_hair_mask: torch.Tensor,
        target_hair_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_alpha: torch.Tensor,
        kwargs: dict,
    ) -> dict[str, torch.Tensor]:
        height, width = target_hair_mask.shape[-2:]
        y = torch.linspace(0.0, 1.0, height, device=target_hair_mask.device, dtype=target_hair_mask.dtype).view(1, 1, height, 1)
        upper = (y <= self._bang_top_ratio(kwargs)).float().expand_as(target_hair_mask)

        m_add = (target_hair_mask * (1.0 - source_hair_mask)).clamp(0.0, 1.0)
        m_remove = (source_hair_mask * (1.0 - target_hair_mask)).clamp(0.0, 1.0)
        m_keep = (source_hair_mask * target_hair_mask).clamp(0.0, 1.0)
        m_bang = ((target_hair_mask + donor_hair_mask) * upper * (boundary_mask + m_add + m_keep)).clamp(0.0, 1.0)
        m_face_protect = (1.0 - (target_hair_mask + m_add + 0.5 * boundary_mask).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        m_alpha = boundary_alpha.clamp(0.0, 1.0)
        m_edit = (m_add + m_remove + boundary_mask + m_bang).clamp(0.0, 1.0)
        return {
            "M_target_hair": target_hair_mask,
            "M_source_hair": source_hair_mask,
            "M_donor_hair": donor_hair_mask,
            "M_add": m_add,
            "M_remove": m_remove,
            "M_keep": m_keep,
            "M_boundary": boundary_mask,
            "M_bang": m_bang,
            "M_face_protect": m_face_protect,
            "M_alpha": m_alpha,
            "M_edit": m_edit,
        }

    def _make_shape_info(
        self,
        inp_mask1: torch.Tensor,
        inp_mask2: torch.Tensor,
        target_mask: torch.Tensor,
        im_name2: str,
        name_to_embed,
        kwargs: dict,
    ) -> dict[str, torch.Tensor]:
        source_hair_mask = _binary_hair_mask(inp_mask1)
        donor_hair_mask = _binary_hair_mask(inp_mask2)
        target_hair_mask = _binary_hair_mask(target_mask)
        boundary_mask = _morphological_boundary(target_hair_mask, self._boundary_width(kwargs))
        donor_alpha = self._resolve_alpha_256(im_name2, name_to_embed, donor_hair_mask, kwargs)
        boundary_alpha = (donor_alpha * boundary_mask).clamp(0.0, 1.0)
        topology_masks = self._make_topology_masks(
            source_hair_mask,
            donor_hair_mask,
            target_hair_mask,
            boundary_mask,
            boundary_alpha,
            kwargs,
        )
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
            **topology_masks,
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

        target_mask = target_mask.to(device)
        shape_info = self._make_shape_info(inp_mask1, inp_mask2, target_mask, im_name2, name_to_embed, kwargs)

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            if I_rot is not None:
                save_gen_image(output_dir, "Shape_v12", f"{im_name2}_rotate_to_{im_name1}.png", I_rot)
            save_vis_mask(output_dir, "Shape_v12", f"mask_{im_name1}.png", inp_mask1)
            save_vis_mask(output_dir, "Shape_v12", f"mask_{im_name2}.png", inp_mask2)
            save_vis_mask(output_dir, "Shape_v12", f"mask_{im_name2}_rotate_to_{im_name1}.png", rot_mask)
            save_vis_mask(output_dir, "Shape_v12", f"mask_{im_name1}_{im_name2}_target.png", target_mask)

        return shape_info

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_S_1 = name_to_embed[im_name1]["S"]
        latent_F_1 = name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]
        latent_F_shape_aligned = latent_F_2

        shape_info = self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        if img1_in is img2_in:
            latent_F_base = latent_F_1
        else:
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
            latent_F_shape_aligned = latent_F_out_new

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

            latent_F_base = intermediate_align + interpolation_low[0] * (latent_F_1 - intermediate_align)
            latent_F_base = latent_F_out_new + interpolation_low[1] * (latent_F_base - latent_F_out_new)
            latent_F_base = latent_F_2 + interpolation_low[2] * (latent_F_base - latent_F_2)

            if self.opts.save_all:
                exp_name = kwargs.get("exp_name") or ""
                output_dir = self.opts.save_all_dir / exp_name
                save_gen_image(output_dir, "Align_v12", f"{im_name1}_{im_name2}_SEAN.png", gen1_sean)
                save_gen_image(output_dir, "Align_v12", f"{im_name2}_{im_name1}_SEAN.png", gen2_sean)
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
                save_gen_image(output_dir, "Align_v12", f"{im_name1}_{im_name2}_e4e.png", img1_e4e)
                save_gen_image(output_dir, "Align_v12", f"{im_name2}_{im_name1}_e4e.png", img2_e4e)

        latent_F_align = latent_F_base
        adapter_outputs = {}
        use_adapter = kwargs.get("use_shape_adapter_v12", getattr(self.opts, "use_shape_adapter_v12", False))
        if use_adapter:
            masks_256 = {key: shape_info[key] for key in V12_MASK_KEYS}
            adapter_outputs = self.shape_adapter(
                F_base=latent_F_base,
                F_src=latent_F_1,
                F_shape=latent_F_shape_aligned,
                masks_256=masks_256,
                strength=self._adapter_strength(kwargs),
                shape_prior_strength=float(
                    kwargs.get("shape_prior_strength_v12", getattr(self.opts, "shape_prior_strength_v12", 0.0))
                ),
            )
            latent_F_align = adapter_outputs["latent_F_refined"]

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            gen_im, _ = self.net.generator(
                [latent_S_1],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_align,
            )
            save_gen_image(output_dir, "Align_v12", f"{im_name1}_{im_name2}_output.png", gen_im)
            save_latents(
                output_dir,
                "Align_v12",
                f"{im_name1}_{im_name2}_F.npz",
                latent_F_align=latent_F_align,
                latent_F_base=latent_F_base,
                latent_F_src=latent_F_1,
                latent_F_shape=latent_F_shape_aligned,
                latent_F_shape_raw=latent_F_2,
                target_hair_mask=shape_info["target_hair_mask"],
            )

        return {
            "latent_F_align": latent_F_align,
            "latent_F_base": latent_F_base,
            "latent_F_src": latent_F_1,
            "latent_F_shape": latent_F_shape_aligned,
            "latent_F_shape_raw": latent_F_2,
            "latent_F_hair": latent_F_shape_aligned,
            **shape_info,
            **adapter_outputs,
        }
