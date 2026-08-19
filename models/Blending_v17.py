import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoders import PostProcessModel
from models.Net import Net
from models.SID_Blending_v17 import SIDBlendingModel_v17, gray01
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v17 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents


def load_compatible_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    model_state = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model_state.update(compatible)
    module.load_state_dict(model_state, strict=False)
    return sorted(compatible.keys())


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=width)


def build_lock_mask_v17(align_info: dict[str, object]) -> torch.Tensor:
    delta_masks = align_info.get("delta_masks")
    if not isinstance(delta_masks, dict):
        hm_x = align_info["HM_X"]
        return torch.zeros_like(hm_x).float()

    remove = delta_masks["M_remove"].float()
    zero = torch.zeros_like(remove)
    lock = (
        1.00 * remove
        + 0.95 * delta_masks.get("M_remove_halo", zero).float()
        + 0.92 * delta_masks.get("M_remove_face", zero).float()
        + 0.92 * delta_masks.get("M_remove_neck", zero).float()
        + 0.88 * delta_masks.get("M_remove_tail", zero).float()
        + 0.82 * delta_masks.get("M_remove_context", zero).float()
        + 0.45 * delta_masks.get("M_boundary", zero).float()
        + 0.35 * delta_masks.get("M_body_preserve", zero).float()
    )
    return lock.clamp(0, 1)


def build_safe_mask_v17(hair_mask: torch.Tensor, lock_mask: torch.Tensor) -> torch.Tensor:
    hair_mask = hair_mask.float().clamp(0, 1)
    lock_mask = lock_mask.float().clamp(0, 1)
    lock_guard = _dilate(lock_mask, 1)
    safe = hair_mask * (1.0 - 0.75 * lock_guard)
    safe = torch.maximum(safe, 0.18 * (hair_mask - lock_mask).clamp(0, 1))
    return safe.clamp(0, 1)


class Blending_v17(nn.Module):
    """
    SID-Blending v17:
    illumination-decoupled color injection on top of SATD-cleaned alignment.
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net

        ckpt_path = getattr(self.opts, "blending_checkpoint_v17", getattr(self.opts, "blending_checkpoint", ""))
        blending_checkpoint = torch.load(ckpt_path, map_location=self.opts.device) if ckpt_path else {}
        clip_name = blending_checkpoint.get("clip", getattr(self.opts, "clip_model_v17", "ViT-B/32"))

        self.blending_encoder = SIDBlendingModel_v17(clip_name).to(self.opts.device).eval()
        if blending_checkpoint:
            load_compatible_state_dict(self.blending_encoder, blending_checkpoint.get("model_state_dict", blending_checkpoint))

        self.post_process = PostProcessModel().to(self.opts.device).eval()
        self.post_process.load_state_dict(torch.load(self.opts.pp_checkpoint, map_location=self.opts.device)["model_state_dict"])

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        del align_color

        I_face = name_to_embed["face"]["image_norm_256"]
        I_color = name_to_embed["color"]["image_norm_256"]
        latent_S_face = name_to_embed["face"]["S"]
        latent_S_color = name_to_embed["color"]["S"]

        face_mask, _ = filter_parsing_to_primary_subject(name_to_embed["face"]["mask"])
        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        HM_face = torch.where(face_mask == 13, torch.ones_like(face_mask), torch.zeros_like(face_mask)).float()
        HM_color = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        HM_face_d, _ = self.dilate_erosion.mask(HM_face)
        _, HM_color_e = self.dilate_erosion.mask(HM_color)

        latent_F_align = align_shape["latent_F_align"]
        H_align = align_shape["HM_X"].float()
        M_lock = build_lock_mask_v17(align_shape)
        M_safe = build_safe_mask_v17(H_align, M_lock)

        H_align_d, _ = self.dilate_erosion.mask(H_align)
        face_target_mask = (1.0 - HM_face_d) * (1.0 - H_align_d)

        I_satd, _ = self.net.generator(
            [latent_S_face],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_align,
        )
        I_satd_256 = self.downsample_256(I_satd)
        I_satd_luma = gray01((I_satd_256 + 1.0) * 0.5)

        if I_face is not I_color:
            S_blend_6_18, F_blend, sid_info = self.blending_encoder(
                latent_face=latent_S_face[:, 6:],
                latent_color=latent_S_color[:, 6:],
                target_face=I_face * face_target_mask,
                reference_hair_image=I_color,
                reference_hair_mask=HM_color_e,
                safe_mask=M_safe,
                lock_mask=M_lock,
                align_hair_mask=H_align,
                satd_luma=I_satd_luma,
                align_f=latent_F_align,
                feature_strength=getattr(self.opts, "feature_strength_v17", 1.5),
            )
            S_blend = torch.cat((latent_S_face[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_face
            F_blend = latent_F_align
            sid_info = {}

        I_blend, _ = self.net.generator(
            [S_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=F_blend,
        )
        I_blend_256 = self.downsample_256(I_blend)

        S_final, F_final = self.post_process(I_face, I_blend_256)
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
            save_gen_image(output_dir, "SID_Blending_v17", "satd_seed.png", I_satd)
            save_gen_image(output_dir, "SID_Blending_v17", "sid_blending.png", I_blend)
            save_latents(
                output_dir,
                "SID_Blending_v17",
                "sid_blending.npz",
                S_blend=S_blend,
                F_blend=F_blend,
                H_align=H_align,
                M_lock=M_lock,
                M_safe=M_safe,
            )
            save_gen_image(output_dir, "Final_v17", "final.png", I_final)
            save_latents(output_dir, "Final_v17", "final.npz", S_final=S_final, F_final=F_final)

        final_image = ((I_final[0] + 1) / 2).clamp(0, 1)
        return final_image
