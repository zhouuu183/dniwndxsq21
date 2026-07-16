import torch
from torch import nn

from models.ColorTransfer_v23 import V23ColorTransfer, build_v23_protect_masks, norm_from_rgb01
from models.Net import Net
from utils.image_utils import DilateErosion
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents


def load_v23_color_transfer(opts) -> V23ColorTransfer:
    ckpt_path = getattr(opts, "v23_checkpoint", "")
    checkpoint = None
    ckpt_config = {}
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    def value(name: str, default):
        return ckpt_config.get(name, getattr(opts, f"v23_{name}", default))

    model = V23ColorTransfer(
        work_size=value("work_size", 256),
        bins=value("bins", 7),
        ab_strength=value("ab_strength", 1.0),
        ab_std_strength=value("ab_std_strength", 0.12),
        main_color_strength=value("main_color_strength", 0.18),
        chroma_floor_strength=value("chroma_floor_strength", 0.72),
        chroma_floor_min_ref=value("chroma_floor_min_ref", 8.0),
        luma_boost_scale=value("luma_boost_scale", 0.18),
        darken_scale=value("darken_scale", 0.16),
        max_luma_shift=value("max_luma_shift", 8.0),
        min_luma_shift=value("min_luma_shift", -4.0),
        shadow_threshold=value("shadow_threshold", 18.0),
        highlight_threshold=value("highlight_threshold", 84.0),
        extreme_luma_chroma_scale=value("extreme_luma_chroma_scale", 0.68),
        extreme_luma_shift_scale=value("extreme_luma_shift_scale", 0.20),
        target_erode=value("target_erode", 1),
        target_band_dilate=value("target_band_dilate", 1),
        band_alpha=value("band_alpha", 0.45),
        soft_lock_strength=value("soft_lock_strength", 0.65),
        alpha_blur_radius=value("alpha_blur_radius", 3),
        ref_erode=value("ref_erode", 2),
        ref_l_min=value("ref_l_min", 5.0),
        ref_l_max=value("ref_l_max", 95.0),
        ref_min_sat=value("ref_min_sat", 0.025),
        learnable=False,
    )
    if checkpoint is not None:
        state_dict = checkpoint.get("v23_state_dict", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
    return model


class Blending_v23(nn.Module):
    """
    v23 uses v8/SATD alignment to generate a clean I_satd, then recolors only
    the target hair alpha in image space. It does not use ClipBlendingModel or
    PostProcessModel.
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.net = Net(opts) if net is None else net
        self.dilate_erosion = DilateErosion(dilate_erosion=opts.smooth, device=opts.device)
        self.color_transfer = load_v23_color_transfer(opts).to(opts.device).eval()

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

        protect_masks = build_v23_protect_masks(align_shape, size=align_shape["HM_X"].shape[-2:])
        final_01, aux = self.color_transfer(
            i_satd_01,
            i_color,
            align_shape["HM_X"],
            hm_color_e,
            protect_masks["hard_lock"],
            protect_masks["soft_lock"],
            return_aux=True,
        )

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "SATD_v23", "satd.png", i_satd)
            save_gen_image(output_dir, "Final_v23", "final.png", norm_from_rgb01(final_01))
            save_latents(
                output_dir,
                "V23_Masks",
                "masks.npz",
                HM_X=align_shape["HM_X"],
                reference_hair_mask=hm_color_e,
                hard_lock=protect_masks["hard_lock"],
                soft_lock=protect_masks["soft_lock"],
                core=aux["core"],
                band=aux["band"],
                alpha=aux["alpha"],
                alpha_full=aux["alpha_full"],
                ref_mask=aux["ref_mask"],
                target_l_mean=aux["target_l_mean"],
                ref_l_mean=aux["ref_l_mean"],
                target_ab_mean=aux["target_ab_mean"],
                ref_ab_mean=aux["ref_ab_mean"],
            )

        return final_01[0]
