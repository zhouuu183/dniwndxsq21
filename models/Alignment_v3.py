import torch
import torch.nn.functional as F

from models.Alignment import Alignment
from models.SATD_v3 import SATD_v3
from utils.mask_delta_v3 import build_delta_masks, stack_satd_masks


class Alignment_v3(Alignment):
    """
    v3 alignment keeps original hard Eq.(8) fusion as a prior and exposes
    real shape-alignment features for SATD training.
    """

    def __init__(self, opts, latent_encoder=None, net=None, satd_model: SATD_v3 | None = None):
        super().__init__(opts, latent_encoder=latent_encoder, net=net)
        self.satd_model = satd_model

        if self.satd_model is None and getattr(self.opts, "use_satd_v3", False):
            self.satd_model = SATD_v3().to(self.opts.device).eval()
            ckpt_path = getattr(self.opts, "satd_checkpoint_v3", None)
            if ckpt_path:
                ckpt = torch.load(ckpt_path, map_location=self.opts.device)
                state_dict = ckpt.get("satd_state_dict", ckpt.get("model_state_dict", ckpt))
                self.satd_model.load_state_dict(state_dict, strict=False)

    def _encode_decode_sean(self, images, labels, target_mask):
        from models.sean_codes.models.pix2pix_model import encode_sean, decode_sean

        img1_code, img2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask)
        gen2_sean = decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask)
        return gen1_sean, gen2_sean

    @staticmethod
    def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
        # SEAN decode returns [3, H, W], while the rest of the pipeline expects [1, 3, H, W].
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

    def _eq8_fusion(self, latent_F_1, latent_F_2, intermediate_align, latent_F_out_new, hair_mask1, hair_mask2,
                    hair_mask_target):
        masks = [
            1 - (1 - hair_mask1) * (1 - hair_mask_target),
            hair_mask_target,
            hair_mask2 * hair_mask_target,
        ]
        masks = torch.cat(masks, dim=0)
        dilate, erosion = self.dilate_erosion.mask(masks)
        free_mask = torch.stack([dilate[0], erosion[1], erosion[2]], dim=0)
        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode="bicubic")
        interpolation_low = 1 - free_mask_down_32

        latent_F_align = intermediate_align + interpolation_low[0] * (latent_F_1 - intermediate_align)
        latent_F_align = latent_F_out_new + interpolation_low[1] * (latent_F_align - latent_F_out_new)
        latent_F_align = latent_F_2 + interpolation_low[2] * (latent_F_align - latent_F_2)
        return latent_F_align

    @torch.inference_mode()
    def prepare_satd_features(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_S_1, latent_F_1 = name_to_embed[im_name1]["S"], name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]
        boundary_width = kwargs.get("satd_boundary_v3", getattr(self.opts, "satd_boundary_v3", self.opts.smooth))

        if img1_in is img2_in:
            hair_mask_target = self.shape_module(im_name1, im_name2, name_to_embed, only_target=True, **kwargs)["HM_X"]
            hair_mask1 = torch.where(name_to_embed[im_name1]["mask"] == 13, torch.ones_like(name_to_embed[im_name1]["mask"]),
                                     torch.zeros_like(name_to_embed[im_name1]["mask"])).float()
            delta_masks = build_delta_masks(hair_mask1, hair_mask_target, hair_mask1, boundary_width=boundary_width)
            satd_masks = stack_satd_masks(delta_masks)
            return {
                "latent_F_eq8": latent_F_1,
                "latent_F_src": latent_F_1,
                "latent_F_ref": latent_F_2,
                "latent_F_src_inpaint": latent_F_1,
                "latent_F_shape_inpaint": latent_F_1,
                "delta_masks": delta_masks,
                "satd_masks_256": satd_masks,
                "satd_masks_32": F.interpolate(satd_masks, size=(32, 32), mode="nearest"),
                "HM_X": hair_mask_target,
                "source_image_256": img1_in,
                "shape_image_256": img2_in,
                "source_inpaint_256": img1_in,
                "shape_inpaint_256": img2_in,
                "latent_S_src": latent_S_1,
            }

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = (
            self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        )

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([inp_mask1, inp_mask2], dim=0)
        gen1_sean, gen2_sean = self._encode_decode_sean(images, labels, target_mask)
        gen1_sean = self._ensure_image_batch(gen1_sean)
        gen2_sean = self._ensure_image_batch(gen2_sean)

        # Embedding.get_e4e_embed expects a list of single images shaped [3, H, W].
        enc_imgs = self.latent_encoder([gen1_sean[0], gen2_sean[0]])
        intermediate_align = enc_imgs["F"][0].unsqueeze(0)
        latent_F_out_new = enc_imgs["F"][1].unsqueeze(0)

        latent_F_eq8 = self._eq8_fusion(
            latent_F_1=latent_F_1,
            latent_F_2=latent_F_2,
            intermediate_align=intermediate_align,
            latent_F_out_new=latent_F_out_new,
            hair_mask1=hair_mask1,
            hair_mask2=hair_mask2,
            hair_mask_target=hair_mask_target,
        )

        delta_masks = build_delta_masks(
            hair_mask1.float(),
            hair_mask_target.float(),
            ref_hair=hair_mask2.float(),
            boundary_width=boundary_width,
        )
        satd_masks_256 = stack_satd_masks(delta_masks)

        gen1_sean_256 = F.interpolate(gen1_sean, size=img1_in.shape[-2:], mode="bicubic", align_corners=False)
        gen2_sean_256 = F.interpolate(gen2_sean, size=img2_in.shape[-2:], mode="bicubic", align_corners=False)

        return {
            "latent_F_eq8": latent_F_eq8,
            "latent_F_src": latent_F_1,
            "latent_F_ref": latent_F_2,
            "latent_F_src_inpaint": intermediate_align,
            "latent_F_shape_inpaint": latent_F_out_new,
            "delta_masks": delta_masks,
            "satd_masks_256": satd_masks_256,
            "satd_masks_32": F.interpolate(satd_masks_256, size=(32, 32), mode="nearest"),
            "HM_X": hair_mask_target,
            "source_image_256": img1_in,
            "shape_image_256": img2_in,
            "source_inpaint_256": gen1_sean_256,
            "shape_inpaint_256": gen2_sean_256,
            "latent_S_src": latent_S_1,
        }

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        satd_features = self.prepare_satd_features(im_name1, im_name2, name_to_embed, **kwargs)
        latent_F_align = satd_features["latent_F_eq8"]

        use_satd = kwargs.get("use_satd_v3", getattr(self.opts, "use_satd_v3", False))
        if use_satd and self.satd_model is not None and name_to_embed[im_name1]["image_256"] is not name_to_embed[im_name2]["image_256"]:
            satd_out, _ = self.satd_model(
                F_src=satd_features["latent_F_src"],
                F_src_inpaint=satd_features["latent_F_src_inpaint"],
                F_shape_inpaint=satd_features["latent_F_shape_inpaint"],
                F_ref=satd_features["latent_F_ref"],
                masks_256=satd_features["satd_masks_256"],
                source_rgb_256=satd_features["source_image_256"] * 2 - 1,
            )
            satd_blend = kwargs.get("satd_blend_v3", getattr(self.opts, "satd_blend_v3", 0.35))
            latent_F_align = (1 - satd_blend) * latent_F_align + satd_blend * satd_out

        return {
            "latent_F_align": latent_F_align,
            "HM_X": satd_features["HM_X"],
            "delta_masks": satd_features["delta_masks"],
            "source_inpaint_256": satd_features["source_inpaint_256"],
            "shape_inpaint_256": satd_features["shape_inpaint_256"],
        }
