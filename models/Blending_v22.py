import torch
from torch import nn

from models.ColorTransfer_v22 import V22ColorTransfer, build_v22_lock_mask, norm_from_rgb01
from models.Net import Net, get_segmentation
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents


def load_v22_color_transfer(opts) -> V22ColorTransfer:
    ckpt_path = getattr(opts, "v22_checkpoint", "")
    checkpoint = None
    ckpt_config = {}
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    def value(name: str, default):
        return ckpt_config.get(name, getattr(opts, f"v22_{name}", default))

    model = V22ColorTransfer(
        ab_strength=value("ab_strength", 0.58),
        ab_std_strength=value("ab_std_strength", 0.10),
        luma_boost_scale=value("luma_boost_scale", 0.20),
        max_luma_shift=value("max_luma_shift", 5.0),
        min_luma_shift=value("min_luma_shift", -3.0),
        saturation_compress=value("saturation_compress", 0.92),
        extreme_ab_scale=value("extreme_ab_scale", 0.74),
        luma_gap_threshold=value("luma_gap_threshold", 0.12),
        extreme_luma_gap=value("extreme_luma_gap", 0.25),
        dark_luma_threshold=value("dark_luma_threshold", 28.0),
        light_luma_threshold=value("light_luma_threshold", 62.0),
        shadow_threshold=value("shadow_threshold", 18.0),
        highlight_threshold=value("highlight_threshold", 82.0),
        highlight_shadow_ab_scale=value("highlight_shadow_ab_scale", 0.42),
        bins=value("bins", 7),
        target_erode=value("target_erode", 1),
        lock_dilate=value("lock_dilate", 1),
        alpha_blur_radius=value("alpha_blur_radius", 3),
        ref_erode=value("ref_erode", 2),
        ref_l_min=value("ref_l_min", 5.0),
        ref_l_max=value("ref_l_max", 95.0),
        ref_min_sat=value("ref_min_sat", 0.02),
        learnable=False,
    )
    if checkpoint is not None:
        state_dict = checkpoint.get("v22_state_dict", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
    return model


class Blending_v22(nn.Module):
    """
    v22 recolors only a strict safe hair region on top of the SATD-cleaned image.
    Non-safe pixels are hard-preserved from I_satd.
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.net = Net(opts) if net is None else net
        self.dilate_erosion = DilateErosion(dilate_erosion=opts.smooth, device=opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.color_transfer = load_v22_color_transfer(opts).to(opts.device).eval()

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        del align_color
        i_color = name_to_embed["color"]["image_norm_256"]
        latent_s_face = name_to_embed["face"]["S"]
        latent_f_align = align_shape["latent_F_align"]

        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        hm_color = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        _, hm_color_e = self.dilate_erosion.mask(hm_color)

        i_satd, _ = self.net.generator(
            [latent_s_face],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_f_align,
        )
        i_satd_01 = ((i_satd + 1.0) * 0.5).clamp(0, 1)
        lock_mask = build_v22_lock_mask(align_shape, size=align_shape["HM_X"].shape[-2:])

        satd_mask, _ = filter_parsing_to_primary_subject(get_segmentation(i_satd_01, resize=True))
        hm_satd = torch.where(satd_mask == 13, torch.ones_like(satd_mask), torch.zeros_like(satd_mask)).float()
        target_hair_mask = (hm_satd + align_shape["HM_X"]).clamp(0, 1)

        final_01, aux = self.color_transfer(
            i_satd_01,
            i_color,
            target_hair_mask,
            hm_color_e,
            lock_mask,
            return_aux=True,
        )

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "SATD_v22", "satd.png", i_satd)
            save_gen_image(output_dir, "Final_v22", "final.png", norm_from_rgb01(final_01))
            save_latents(
                output_dir,
                "V22_Masks",
                "masks.npz",
                HM_X=align_shape["HM_X"],
                target_hair_mask=target_hair_mask,
                lock_mask=lock_mask,
                core=aux["core"],
                alpha=aux["alpha"],
                ref_mask=aux["ref_mask"],
                luma_gap=aux["luma_gap"],
                luma_shift=aux["luma_shift"],
            )

        return final_01[0]
