import torch
import torch.nn.functional as F

from models.Blending import Blending
from models.Net import get_segmentation
from utils.hair_color_completion_v8 import (
    apply_pp_hair_guard_v8,
    compute_color_diagnostics_v8,
    reference_color_completion_v8,
    save_color_debug_v8,
)
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.hair_color_match_v8 import lab_to_rgb, rgb_to_lab
from utils.save_utils import save_gen_image, save_latents


def hair_color_debug_ab_to_rgb(image: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Render a Lab A/B debug tensor for V5/V8 compatibility.

    Older V5 entry points import this helper from ``Blending_v8``.  Keeping the
    helper here is intentionally side-effect free: it is used only for optional
    debug images and does not participate in the V8 colour-transfer path.
    """

    was_normalized = bool(image.detach().amin() < -0.05)
    image_01 = ((image + 1.0) * 0.5 if was_normalized else image).clamp(0, 1)
    lab = rgb_to_lab(image_01)
    if ab.shape[-2:] != lab.shape[-2:]:
        ab = F.interpolate(ab, size=lab.shape[-2:], mode="bilinear", align_corners=False)
    result = lab_to_rgb(torch.cat((lab[:, :1], ab.to(dtype=lab.dtype)), dim=1))
    return result * 2.0 - 1.0 if was_normalized else result


class Blending_v8(Blending):
    """
    v8 keeps the stable blending branch, but makes the implementation
    self-contained so it does not depend on Blending_v4 from the server.
    """

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_2 = name_to_embed["shape"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        face_mask, _ = filter_parsing_to_primary_subject(name_to_embed["face"]["mask"])
        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        HM_1 = torch.where(face_mask == 13, torch.ones_like(face_mask), torch.zeros_like(face_mask)).float()
        HM_3 = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        HM_1D, _ = self.dilate_erosion.mask(HM_1)
        HM_3D, HM_3E = self.dilate_erosion.mask(HM_3)
        hair_color_mask = HM_3E

        latent_S_1 = name_to_embed["face"]["S"]
        latent_S_3 = name_to_embed["color"]["S"]
        latent_F_align = align_shape["latent_F_align"]
        HM_X = align_shape["HM_X"]

        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        if I_1 is not I_3 or I_1 is not I_2:
            S_blend_6_18 = self.blending_encoder(
                latent_S_1[:, 6:],
                latent_S_3[:, 6:],
                I_1 * target_mask,
                I_3 * hair_color_mask,
            )
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        I_blend, _ = self.net.generator(
            [S_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_align,
        )

        use_completion = kwargs.get(
            "v8_color_completion",
            getattr(self.opts, "v8_color_completion", True),
        )
        use_pp_guard = kwargs.get(
            "v8_pp_hair_guard",
            getattr(self.opts, "v8_pp_hair_guard", True),
        )
        color_fixed = I_blend
        completion_masks = None
        completion_stats = None
        if use_completion:
            generated_parsing = get_segmentation(((I_blend + 1.0) * 0.5).clamp(0, 1))
            generated_hair = (generated_parsing == 13).float()
            color_fixed, completion_masks, completion_stats = reference_color_completion_v8(
                source=I_1,
                reference=I_3,
                blend_raw=I_blend,
                source_hair=HM_1,
                reference_hair=HM_3E,
                target_hair=HM_X,
                protect_masks=align_shape.get("delta_masks"),
                generated_hair=generated_hair,
                target_progress=getattr(self.opts, "v8_color_target_progress", 0.88),
                chroma_gain=getattr(self.opts, "v8_color_chroma_gain", 0.75),
                luma_gain=getattr(self.opts, "v8_color_luma_gain", 0.20),
                max_chroma_shift=getattr(self.opts, "v8_color_max_chroma_shift", 0.075),
                max_luma_shift=getattr(self.opts, "v8_color_max_luma_shift", 0.035),
                vivid_boost=getattr(self.opts, "v8_color_vivid_boost", 1.0),
                highlight_chroma_gain=getattr(self.opts, "v8_color_highlight_chroma_gain", 0.85),
                highlight_luma_gain=getattr(self.opts, "v8_color_highlight_luma_gain", 0.35),
                global_palette_lock=getattr(self.opts, "v8_color_global_palette_lock", 0.95),
                texture_preserve=getattr(self.opts, "v8_color_texture_preserve", 0.35),
                edge_width=getattr(self.opts, "v8_color_edge_width", 3),
            )

        I_fixed_256 = self.downsample_256(color_fixed)

        S_final, F_final = self.post_process(I_1, I_fixed_256)
        I_pp, _ = self.net.generator(
            [S_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=F_final,
        )
        I_final = I_pp

        if use_pp_guard and completion_masks is not None:
            I_final = apply_pp_hair_guard_v8(I_final, color_fixed, completion_masks)
        debug_enabled = bool(
            self.opts.save_all or getattr(self.opts, "v8_save_color_debug", False)
        )
        if debug_enabled and completion_masks is not None and completion_stats is not None:
            completion_stats.update(
                compute_color_diagnostics_v8(
                    I_1,
                    I_3,
                    {
                        "color_fixed": color_fixed,
                        "pp_raw": I_pp,
                        "final": I_final,
                    },
                    completion_masks,
                )
            )

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name")
            exp_name = exp_name if exp_name is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v8", "blending.png", I_blend)
            save_latents(output_dir, "Blending_v8", "blending.npz", S_blend=S_blend)
            save_gen_image(output_dir, "Final_v8", "final.png", I_final)
            save_latents(output_dir, "Final_v8", "final.npz", S_final=S_final, F_final=F_final)

        if debug_enabled and completion_masks is not None:
            exp_name = kwargs.get("exp_name")
            exp_name = exp_name if exp_name is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_color_debug_v8(
                output_dir / "ColorV8",
                I_blend,
                completion_masks,
                color_fixed,
                I_pp,
                completion_masks["pp_guard"],
                I_final,
                completion_stats or {},
            )

        return ((I_final[0] + 1) / 2).clamp(0, 1)
