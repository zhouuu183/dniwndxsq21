import os

import torch
import torch.nn.functional as F
from torch import nn

from models.Encoders_v1 import DeltaAwareBlendingModelV1, DeltaAwarePostProcessModelV1
from models.Net import Net
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents


# v1 modification:
# The blending stage is no longer a single global latent mixer.
# It now uses:
# 1) a delta-aware hair branch for complex texture,
# 2) a context branch for remove/disocclusion regions,
# 3) a boundary gate for shadow harmonization,
# followed by a lightly adapted pp model.


class Blending(nn.Module):
    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.net = net if net is not None else Net(self.opts)

        self.blending_encoder = DeltaAwareBlendingModelV1().to(self.opts.device).eval()
        self.post_process = DeltaAwarePostProcessModelV1().to(self.opts.device).eval()
        self.has_blending_checkpoint = False
        self.has_pp_checkpoint = False

        blending_checkpoint = getattr(self.opts, "blending_checkpoint_v1", None)
        # v1 bugfix:
        # Do not silently run with randomly initialized v1 modules.
        # A missing blending checkpoint is always an error for the v1 pipeline.
        if not blending_checkpoint:
            raise ValueError("blending_checkpoint_v1 is required for HairFast_v1.")
        if not os.path.isfile(blending_checkpoint):
            raise FileNotFoundError(f"Cannot find blending checkpoint: {blending_checkpoint}")
        checkpoint = torch.load(blending_checkpoint, map_location=self.opts.device)
        self.blending_encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.has_blending_checkpoint = True

        pp_checkpoint = getattr(self.opts, "pp_checkpoint_v1", None)
        if pp_checkpoint:
            if not os.path.isfile(pp_checkpoint):
                raise FileNotFoundError(f"Cannot find post-process checkpoint: {pp_checkpoint}")
            checkpoint = torch.load(pp_checkpoint, map_location=self.opts.device)
            self.post_process.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.has_pp_checkpoint = True

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

    def _prepare_masks(self, align_shape, name_to_embed):
        delta_masks = {key: value.float() for key, value in align_shape["delta_masks"].items() if key.startswith("M_")}

        color_seg = name_to_embed["color"]["mask"]
        color_hair = torch.where(color_seg == 13, torch.ones_like(color_seg), torch.zeros_like(color_seg)).float()
        delta_masks["M_color"] = color_hair
        return delta_masks

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        del align_color  # v1 uses shape-driven delta masks as the primary routing signal.

        I_1 = name_to_embed["face"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        latent_S_1 = name_to_embed["face"]["S"]
        latent_S_3 = name_to_embed["color"]["S"]
        latent_F_align = align_shape["latent_F_align"]
        delta_masks = self._prepare_masks(align_shape, name_to_embed)

        blend_state = self.blending_encoder(
            latent_S_1[:, 6:],
            latent_S_3[:, 6:],
            I_1,
            I_3,
            delta_masks,
        )

        S_blend = torch.cat((latent_S_1[:, :6], blend_state["latent_tail"]), dim=1)
        remove_down = F.interpolate(delta_masks["M_remove"], size=(32, 32), mode="bilinear", align_corners=False)
        # v1 refinement:
        # Restrict the context/disocclusion residual to the remove side and to
        # a small non-hair boundary ring. Let the hair branch handle target-hair
        # appearance so the F-space residual does not create gray veils.
        boundary_bg = delta_masks["M_boundary"] * (1.0 - delta_masks["M_tgt"])
        boundary_bg_down = F.interpolate(boundary_bg, size=(32, 32), mode="bilinear", align_corners=False)
        context_weight = (remove_down + 0.15 * boundary_bg_down).clamp(0, 1) * blend_state["boundary_gate"]
        latent_F_blend = latent_F_align + context_weight * blend_state["delta_F"]

        I_blend, _ = self.net.generator(
            [S_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_blend,
        )
        I_blend_256 = self.downsample_256(I_blend)

        pre_pp_image = ((I_blend[0] + 1) / 2).clip(0, 1)
        pre_pp_image_256 = ((I_blend_256[0] + 1) / 2).clip(0, 1)

        intermediates = {
            "pre_pp_image": pre_pp_image,
            "pre_pp_image_256": pre_pp_image_256,
            "delta_masks": {key: value.detach() for key, value in delta_masks.items()},
            "boundary_gate": blend_state["boundary_gate"].detach(),
            "latent_F_blend": latent_F_blend.detach(),
            "context_weight": context_weight.detach(),
        }

        if kwargs.get("stop_before_pp", False):
            if self.opts.save_all:
                exp_name = kwargs.get("exp_name") or ""
                output_dir = self.opts.save_all_dir / exp_name
                save_gen_image(output_dir, "Blending_v1", "pre_pp.png", I_blend)
            if kwargs.get("return_intermediates", False):
                return pre_pp_image, intermediates
            return pre_pp_image

        # v1 bugfix:
        # Full inference requires a trained pp checkpoint. Fail loudly instead of
        # silently using a random refinement module.
        if not self.has_pp_checkpoint:
            raise ValueError("pp_checkpoint_v1 is required for full HairFast_v1 inference.")

        S_final, F_final = self.post_process(
            I_1,
            I_blend_256,
            remove_mask=delta_masks["M_remove"],
            boundary_mask=delta_masks["M_boundary"],
        )
        I_final, _ = self.net.generator(
            [S_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=F_final,
        )

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v1", "pre_pp.png", I_blend)
            save_latents(output_dir, "Blending_v1", "pre_pp_latents.npz", S_blend=S_blend, latent_F_blend=latent_F_blend)
            save_gen_image(output_dir, "Final_v1", "final.png", I_final)
            save_latents(output_dir, "Final_v1", "final.npz", S_final=S_final, F_final=F_final)

        final_image = ((I_final[0] + 1) / 2).clip(0, 1)
        if kwargs.get("return_intermediates", False):
            intermediates["final_image"] = final_image
            return final_image, intermediates
        return final_image
