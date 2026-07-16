import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Alignment_v18 import Alignment_v18
from models.Embedding import Embedding
from models.Encoders import ClipBlendingModel, PostProcessModel
from models.Net import Net
from models.SATD_v18 import SATD_v18
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v18 import filter_parsing_to_primary_subject
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


def resolve_checkpoint_state_dict(ckpt: dict[str, object], preferred_key: str) -> dict[str, torch.Tensor]:
    if preferred_key in ckpt:
        return ckpt[preferred_key]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    for key, value in ckpt.items():
        if key.endswith("_state_dict") and isinstance(value, dict):
            return value
    return ckpt


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)


def build_lock_mask_v18(align_info: dict[str, object]) -> torch.Tensor:
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
        + 0.30 * delta_masks.get("M_boundary", zero).float()
        + 0.35 * delta_masks.get("M_body_preserve", zero).float()
    )
    return lock.clamp(0, 1)


def build_safe_mask_v18(hair_mask: torch.Tensor, lock_mask: torch.Tensor) -> torch.Tensor:
    hair_mask = hair_mask.float().clamp(0, 1)
    lock_mask = lock_mask.float().clamp(0, 1)
    lock_guard = _dilate(lock_mask, 1)
    safe = hair_mask * (1.0 - 0.85 * lock_guard)
    safe = torch.maximum(safe, 0.08 * (hair_mask - lock_mask).clamp(0, 1))
    return safe.clamp(0, 1)


class Blending_v18(nn.Module):
    """
    v18 color pipeline:
    SATD-cleaned alignment -> stable baseline-style color transfer.
    """

    def __init__(self, opts, net=None, embed: Embedding | None = None, align: Alignment_v18 | None = None):
        super().__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net
        self.embed = embed if embed is not None else Embedding(self.opts, net=self.net)
        self.align = (
            align
            if align is not None
            else Alignment_v18(self.opts, latent_encoder=self.embed.get_e4e_embed, net=self.net)
        )
        self.align.eval()

        ckpt_path = getattr(self.opts, "blending_checkpoint_v18", getattr(self.opts, "blending_checkpoint", ""))
        blending_checkpoint = torch.load(ckpt_path, map_location=self.opts.device) if ckpt_path else {}
        clip_name = blending_checkpoint.get("clip", getattr(self.opts, "clip_model_v18", "ViT-B/32"))

        self.blending_encoder = ClipBlendingModel(clip_name)
        if blending_checkpoint:
            load_compatible_state_dict(self.blending_encoder, blending_checkpoint.get("model_state_dict", blending_checkpoint))
        self.blending_encoder.to(self.opts.device).eval()

        self.satd_model_v18 = None
        if getattr(self.opts, "use_post_satd_v18", False):
            self.satd_model_v18 = SATD_v18().to(self.opts.device).eval()
            satd_ckpt_path = getattr(self.opts, "satd_checkpoint_v18", "")
            if satd_ckpt_path:
                ckpt = torch.load(satd_ckpt_path, map_location=self.opts.device)
                load_compatible_state_dict(self.satd_model_v18, resolve_checkpoint_state_dict(ckpt, "satd_v18_state_dict"))

        self.post_process = None
        pp_checkpoint = getattr(self.opts, "pp_checkpoint", "")
        if pp_checkpoint:
            self.post_process = PostProcessModel().to(self.opts.device).eval()
            self.post_process.load_state_dict(torch.load(pp_checkpoint, map_location=self.opts.device)["model_state_dict"])

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

    @torch.inference_mode()
    def _reembed_generated_image(self, image_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        image_01 = ((image_norm + 1.0) * 0.5).clamp(0, 1)
        image_key = image_01[0].detach().cpu()
        return self.embed.embedding_images({image_key: ["post_blend"]})["post_blend"]

    def _post_color_satd(
        self,
        align_shape: dict[str, object],
        latent_F_color_base: torch.Tensor,
        color_blend_256: torch.Tensor,
        enabled: bool,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not enabled or self.satd_model_v18 is None:
            return latent_F_color_base, {}
        required = ("latent_F_src", "latent_F_src_inpaint", "cleanup_masks_256", "delta_masks", "HM_X")
        if any(key not in align_shape for key in required):
            return latent_F_color_base, {}

        satd_out, satd_aux = self.satd_model_v18(
            F_base=latent_F_color_base,
            F_src=align_shape["latent_F_src"],
            F_src_inpaint=align_shape["latent_F_src_inpaint"],
            cleanup_masks_256=align_shape["cleanup_masks_256"],
            source_rgb_256=color_blend_256,
        )
        cleanup_support = Alignment_v18._protected_cleanup_support_from_delta_masks(
            align_shape["delta_masks"],
            target_hair_mask=align_shape["HM_X"],
            out_hw=latent_F_color_base.shape[-2:],
        )
        post_satd_blend = kwargs.get(
            "post_satd_blend_v18",
            getattr(self.opts, "post_satd_blend_v18", 0.28),
        )
        latent_F_clean = latent_F_color_base + post_satd_blend * cleanup_support * (satd_out - latent_F_color_base)
        satd_aux = dict(satd_aux)
        satd_aux["post_satd_support"] = cleanup_support
        return latent_F_clean, satd_aux

    @torch.inference_mode()
    def _build_color_blend_v18(self, align_shape, align_color, name_to_embed):
        i_face = name_to_embed["face"]["image_norm_256"]
        i_shape = name_to_embed["shape"]["image_norm_256"]
        i_color = name_to_embed["color"]["image_norm_256"]

        face_mask, _ = filter_parsing_to_primary_subject(name_to_embed["face"]["mask"])
        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        hm_face = torch.where(face_mask == 13, torch.ones_like(face_mask), torch.zeros_like(face_mask)).float()
        hm_color = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        hm_face_d, _ = self.dilate_erosion.mask(hm_face)
        hm_color_d, hm_color_e = self.dilate_erosion.mask(hm_color)

        latent_s_face = name_to_embed["face"]["S"]
        latent_s_color = name_to_embed["color"]["S"]
        latent_f_author = align_shape["latent_F_align"]
        target_hair_mask = align_color["HM_X"]

        target_hair_mask_d, _ = self.dilate_erosion.mask(target_hair_mask)
        target_mask = (1 - hm_face_d) * (1 - hm_color_d) * (1 - target_hair_mask_d)

        color_enabled = i_face is not i_color or i_face is not i_shape
        if color_enabled:
            s_blend_6_18 = self.blending_encoder(
                latent_s_face[:, 6:],
                latent_s_color[:, 6:],
                i_face * target_mask,
                i_color * hm_color_e,
            )
            s_blend = torch.cat((latent_s_face[:, :6], s_blend_6_18), dim=1)
        else:
            s_blend = latent_s_face

        i_color_blend, _ = self.net.generator(
            [s_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_f_author,
        )
        i_color_blend_256 = self.downsample_256(i_color_blend)
        return {
            "color_enabled": color_enabled,
            "S_blend": s_blend,
            "I_color_blend": i_color_blend,
            "I_color_blend_256": i_color_blend_256,
            "latent_F_author": latent_f_author,
            "target_mask": target_mask,
            "HM_3E": hm_color_e,
        }

    @torch.inference_mode()
    def build_post_blend_cleanup_inputs_v18(
        self,
        name_to_embed,
        face_name="face",
        shape_name="shape",
        color_name="color",
        align_shape=None,
        align_color=None,
        **kwargs,
    ):
        if align_shape is None:
            align_shape = self.align.align_images(
                face_name,
                shape_name,
                name_to_embed,
                satd_boundary_v18=kwargs.get("satd_boundary_v18", getattr(self.opts, "satd_boundary_v18", 8)),
                eq8_reference_blend_v18=kwargs.get(
                    "eq8_reference_blend_v18",
                    getattr(self.opts, "eq8_reference_blend_v18", 0.0),
                ),
            )
        if align_color is None:
            align_color = self.align.shape_module(face_name, color_name, name_to_embed)
        role_embed = {
            "face": name_to_embed[face_name],
            "shape": name_to_embed[shape_name],
            "color": name_to_embed[color_name],
        }
        color_blend = self._build_color_blend_v18(align_shape, align_color, role_embed)

        use_reembed_cleanup = kwargs.get(
            "use_reembed_cleanup_v18",
            getattr(self.opts, "use_reembed_cleanup_v18", False),
        )
        if color_blend["color_enabled"] and use_reembed_cleanup:
            post_blend_embed = self._reembed_generated_image(color_blend["I_color_blend"])
            latent_s_cleanup = post_blend_embed["S"]
            latent_f_cleanup_base = post_blend_embed["F"]
            cleanup_context_256 = post_blend_embed["image_norm_256"]
        else:
            latent_s_cleanup = color_blend["S_blend"]
            latent_f_cleanup_base = color_blend["latent_F_author"]
            cleanup_context_256 = color_blend["I_color_blend_256"]

        return {
            "align_shape": align_shape,
            "align_color": align_color,
            "color_enabled": color_blend["color_enabled"],
            "color_blend_256": color_blend["I_color_blend_256"],
            "cleanup_context_256": cleanup_context_256,
            "latent_s_cleanup": latent_s_cleanup,
            "latent_f_cleanup_base": latent_f_cleanup_base,
            "color_image_256": role_embed["color"]["image_norm_256"],
            "target_mask": color_blend["target_mask"],
        }

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        i_face = name_to_embed["face"]["image_norm_256"]
        cleanup_inputs = self.build_post_blend_cleanup_inputs_v18(
            {"face": name_to_embed["face"], "shape": name_to_embed["shape"], "color": name_to_embed["color"]},
            face_name="face",
            shape_name="shape",
            color_name="color",
            align_shape=align_shape,
            align_color=align_color,
            **kwargs,
        )
        color_enabled = cleanup_inputs["color_enabled"]
        i_color_blend_256 = cleanup_inputs["color_blend_256"]
        latent_S_cleanup = cleanup_inputs["latent_s_cleanup"]
        latent_F_cleanup_base = cleanup_inputs["latent_f_cleanup_base"]
        cleanup_context_256 = cleanup_inputs["cleanup_context_256"]
        latent_f_author = cleanup_inputs["align_shape"]["latent_F_align"]

        F_blend, satd_info = self._post_color_satd(
            align_shape=cleanup_inputs["align_shape"],
            latent_F_color_base=latent_F_cleanup_base,
            color_blend_256=cleanup_context_256,
            enabled=color_enabled and getattr(self.opts, "use_post_satd_v18", False),
            **kwargs,
        )

        I_blend, _ = self.net.generator(
            [latent_S_cleanup],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=F_blend,
        )
        I_blend_256 = self.downsample_256(I_blend)

        use_postprocess = kwargs.get(
            "use_postprocess_v18",
            getattr(self.opts, "use_postprocess_v18", False),
        )
        if use_postprocess and self.post_process is None:
            raise RuntimeError("pp_checkpoint is required when use_postprocess_v18=True.")
        if use_postprocess:
            postprocess_face_source = kwargs.get(
                "postprocess_face_source_v18",
                getattr(self.opts, "postprocess_face_source_v18", False),
            )
            postprocess_source = i_face if postprocess_face_source else i_color_blend_256
            S_final, F_final = self.post_process(postprocess_source, I_blend_256)
            I_final, _ = self.net.generator(
                [S_final],
                input_is_latent=True,
                return_latents=False,
                start_layer=5,
                end_layer=8,
                layer_in=F_final,
            )
        else:
            S_final = latent_S_cleanup
            F_final = F_blend
            I_final = I_blend

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get("exp_name")) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v18", "author_color_blend.png", i_color_blend_256)
            save_gen_image(output_dir, "Blending_v18", "post_blend_reencoded_cleanup.png", I_blend)
            save_gen_image(output_dir, "Blending_v18", "post_color_satd_blending.png", I_blend)
            save_latents(
                output_dir,
                "Blending_v18",
                "post_color_satd_blending.npz",
                S_blend=latent_S_cleanup,
                S_cleanup=latent_S_cleanup,
                F_author=latent_f_author,
                F_cleanup_base=latent_F_cleanup_base,
                F_blend=F_blend,
                H_align=align_shape["HM_X"],
                M_lock=build_lock_mask_v18(align_shape),
                M_safe=build_safe_mask_v18(align_shape["HM_X"], build_lock_mask_v18(align_shape)),
                post_satd_support=satd_info.get("post_satd_support", torch.zeros_like(align_shape["HM_X"])),
            )
            save_gen_image(output_dir, "Final_v18", "final.png", I_final)
            save_latents(output_dir, "Final_v18", "final.npz", S_final=S_final, F_final=F_final)

        final_image = ((I_final[0] + 1) / 2).clamp(0, 1)
        return final_image
