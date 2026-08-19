from __future__ import annotations

import argparse
import inspect

import torch
import torch.nn.functional as F
from torch import nn

from models.Net import Net
from models.SG_IDCT_v16 import SG_IDCT_v16
from models.postprocess_v5 import PostProcessModelV5
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents


def _zero_like_delta(delta_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    for value in delta_masks.values():
        if isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
    raise ValueError("delta_masks does not contain any tensor values.")


def _mask_from_delta(delta_masks: dict[str, torch.Tensor], keys: tuple[str, ...], weights: tuple[float, ...] | None = None):
    zero = _zero_like_delta(delta_masks)
    weights = tuple(1.0 for _ in keys) if weights is None else weights
    mask = torch.zeros_like(zero)
    for key, weight in zip(keys, weights):
        mask = mask + float(weight) * delta_masks.get(key, zero)
    return mask.float().clamp(0, 1)


def _dilate_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    return F.max_pool2d(mask.float(), kernel_size=2 * width + 1, stride=1, padding=width).clamp(0, 1)


def build_sg_idct_masks_v16(align_shape: dict[str, object]) -> dict[str, torch.Tensor]:
    delta_masks = align_shape.get("delta_masks")
    H_align = align_shape["HM_X"].float().clamp(0, 1)

    if not isinstance(delta_masks, dict):
        zero = torch.zeros_like(H_align)
        return {
            "H_align": H_align,
            "M_remove": zero,
            "M_face": zero,
            "M_neck": zero,
            "M_ear": zero,
            "M_final_lock": (1.0 - H_align).clamp(0, 1),
        }

    M_remove = _mask_from_delta(
        delta_masks,
        (
            "M_remove",
            "M_remove_halo",
            "M_remove_tail",
            "M_remove_context",
            "M_boundary",
        ),
        (1.00, 0.95, 0.90, 0.85, 0.35),
    )
    M_face = _mask_from_delta(
        delta_masks,
        ("M_face_region", "M_face_surface", "M_remove_face"),
        (1.00, 0.85, 1.00),
    )
    M_neck = _mask_from_delta(
        delta_masks,
        ("M_neck_region", "M_remove_neck"),
        (1.00, 1.00),
    )
    M_ear = _mask_from_delta(
        delta_masks,
        ("M_ear_surface",),
        (1.00,),
    )
    H_transfer = (_dilate_mask(H_align, 4) * (1.0 - M_face) * (1.0 - M_neck) * (1.0 - M_ear)).clamp(0, 1)
    M_remove_lock = (M_remove * (1.0 - H_transfer)).clamp(0, 1)
    M_final_lock = (M_remove_lock + M_face + M_neck + (1.0 - H_transfer) * (1.0 - M_ear)).clamp(0, 1)
    return {
        "H_align": H_align,
        "H_transfer": H_transfer,
        "M_remove": M_remove,
        "M_remove_lock": M_remove_lock,
        "M_face": M_face,
        "M_neck": M_neck,
        "M_ear": M_ear,
        "M_final_lock": M_final_lock,
    }


class Blending_v16(nn.Module):
    """
    SG-IDCT v16 image-space color transfer followed by v5-style ear-aware
    refinement.
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.net = Net(self.opts) if net is None else net
        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

        self.sg_idct = SG_IDCT_v16(
            beta=getattr(self.opts, "sg_idct_beta_v16", 0.05),
            chroma_strength=getattr(self.opts, "sg_idct_chroma_strength_v16", 0.90),
            chroma_std_strength=getattr(self.opts, "sg_idct_chroma_std_strength_v16", 0.0),
            chroma_delta_limit=getattr(self.opts, "sg_idct_chroma_delta_limit_v16", 24.0),
            texture_strength=getattr(self.opts, "sg_idct_texture_strength_v16", 0.90),
            luma_bins=getattr(self.opts, "sg_idct_luma_bins_v16", 9),
            alpha_strength=getattr(self.opts, "sg_idct_alpha_strength_v16", 1.05),
            alpha_dilate=getattr(self.opts, "sg_idct_alpha_dilate_v16", 6),
            alpha_blur_radius=getattr(self.opts, "sg_idct_alpha_blur_radius_v16", 7),
            learnable=False,
            use_gamut_map=getattr(self.opts, "sg_idct_use_gamut_map_v16", False),
        ).to(self.opts.device).eval()

        sg_idct_checkpoint = getattr(self.opts, "sg_idct_checkpoint_v16", "")
        if sg_idct_checkpoint:
            checkpoint = torch.load(sg_idct_checkpoint, map_location=self.opts.device)
            state_dict = checkpoint.get("sg_idct_v16_state_dict", checkpoint.get("model_state_dict", checkpoint))
            self.sg_idct.load_state_dict(state_dict, strict=False)

        self.skip_refinement = getattr(self.opts, "sg_idct_skip_refinement_v16", True)
        self.post_process = None
        if not self.skip_refinement:
            pp_args = argparse.Namespace(
                use_mod=getattr(self.opts, "pp_v16_use_mod", True),
                use_full=getattr(self.opts, "pp_v16_use_full", True),
                pretrain=False,
                finetune=False,
                ear_parse_size=getattr(self.opts, "ear_parse_size", 512),
                ear_feature_channels=getattr(self.opts, "ear_feature_channels", 128),
                ear_low_alpha=getattr(self.opts, "ear_low_alpha", 0.1),
                ear_dilate=getattr(self.opts, "ear_dilate", 21),
                hair_change_dilate=getattr(self.opts, "hair_change_dilate", 25),
                earring_expand=getattr(self.opts, "earring_expand", 15),
                ear_downward_shift=getattr(self.opts, "ear_downward_shift", 10),
                target_hair_dilate=getattr(self.opts, "target_hair_dilate", 11),
                source_hair_block_dilate=getattr(self.opts, "source_hair_block_dilate", 5),
                source_hair_block_strength=getattr(self.opts, "source_hair_block_strength", 0.6),
                target_visibility_expand=getattr(self.opts, "target_visibility_expand", 5),
                max_target_hair_overlap=getattr(self.opts, "max_target_hair_overlap", 0.55),
                ear_blur_kernel=getattr(self.opts, "ear_blur_kernel", 11),
                ear_blur_sigma=getattr(self.opts, "ear_blur_sigma", 3.0),
                ear_mask_hidden=getattr(self.opts, "ear_mask_hidden", 32),
            )
            self.post_process = PostProcessModelV5(pp_args).to(self.opts.device).eval()

            checkpoint_path = getattr(self.opts, "pp_v16_checkpoint", None) or getattr(self.opts, "pp_checkpoint", None)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.post_process.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
            self._patch_postprocess_compat()

    def _patch_postprocess_compat(self):
        """Keep v16 usable with older v5 ear modules without editing v5 files."""
        if self.post_process is None:
            return
        hf_forward = self.post_process.hf_extractor.forward
        hf_params = inspect.signature(hf_forward).parameters
        if len(hf_params) >= 3:
            return

        def hf_forward_v16_compat(source_01, query_mask, prior_mask=None):
            del prior_mask
            return hf_forward(source_01, query_mask)

        self.post_process.hf_extractor.forward = hf_forward_v16_compat

    @staticmethod
    def _to_norm(rgb01: torch.Tensor) -> torch.Tensor:
        return rgb01.clamp(0, 1) * 2.0 - 1.0

    @staticmethod
    def _to_rgb(norm: torch.Tensor) -> torch.Tensor:
        return ((norm + 1.0) * 0.5).clamp(0, 1)

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        mask_de = self.dilate_erosion.hair_from_mask(
            torch.cat([name_to_embed[x]["mask"] for x in ["face", "color"]], dim=0)
        )
        HM_1D, _ = mask_de[0][0].unsqueeze(0), mask_de[1][0].unsqueeze(0)
        HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)

        latent_S_1 = name_to_embed["face"]["S"]
        latent_F_align = align_shape["latent_F_align"]

        I_satd, _ = self.net.generator(
            [latent_S_1],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_align,
        )
        I_satd_256 = self.downsample_256(I_satd)

        HM_X = align_shape["HM_X"]
        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        sg_masks = build_sg_idct_masks_v16(align_shape)
        I_ct_256, sg_aux = self.sg_idct(
            I_satd_256,
            I_3,
            sg_masks["H_transfer"],
            sg_masks["M_remove"],
            sg_masks["M_face"],
            sg_masks["M_neck"],
            sg_masks["M_ear"],
            H_color=HM_3E,
            return_aux=True,
        )
        I_ct_norm_256 = self._to_norm(I_ct_256)

        if self.skip_refinement:
            I_final = F.interpolate(I_ct_norm_256, size=I_satd.shape[-2:], mode="bicubic", align_corners=False)
            S_final = latent_S_1
            F_final = latent_F_align
            pp_aux = {}
        else:
            S_final, F_final, pp_aux = self.post_process(I_1, I_ct_norm_256, target_mask, HM_3E)
            I_final, _ = self.post_process.render_refined(self.net.generator, S_final, F_final, pp_aux)

        final_lock = F.interpolate(
            sg_masks["M_final_lock"].float(),
            size=I_final.shape[-2:],
            mode="nearest",
        ).clamp(0, 1)
        I_final = (I_final * (1.0 - final_lock) + I_satd * final_lock).clamp(-1, 1)

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "SATD_v16", "satd.png", I_satd)
            save_gen_image(output_dir, "SG_IDCT_v16", "ct.png", I_ct_norm_256)
            save_gen_image(output_dir, "Final_v16", "final.png", I_final)
            save_latents(
                output_dir,
                "SG_IDCT_v16",
                "masks.npz",
                H_align=sg_masks["H_align"],
                H_transfer=sg_masks["H_transfer"],
                H_color=HM_3E,
                M_remove=sg_masks["M_remove"],
                M_face=sg_masks["M_face"],
                M_neck=sg_masks["M_neck"],
                M_ear=sg_masks["M_ear"],
                M_safe=sg_aux["M_safe"],
                A_hair=sg_aux["A_hair"],
                target_mask=target_mask,
            )
            save_latents(output_dir, "Final_v16", "final.npz", S_final=S_final, F_final=F_final)

        final_image = self._to_rgb(I_final)[0]
        if kwargs.get("return_intermediates_v16", False):
            return {
                "final": final_image,
                "satd": self._to_rgb(I_satd_256)[0],
                "ct": I_ct_256[0].clamp(0, 1),
                "color": self._to_rgb(I_3)[0],
                "H_align": sg_masks["H_align"][0],
                "H_transfer": sg_masks["H_transfer"][0],
                "H_color": HM_3E[0],
                "M_remove": sg_masks["M_remove"][0],
                "M_face": sg_masks["M_face"][0],
                "M_neck": sg_masks["M_neck"][0],
                "M_ear": sg_masks["M_ear"][0],
                "M_safe": sg_aux["M_safe"][0],
                "A_hair": sg_aux["A_hair"][0],
            }
        return final_image
