import torch

from models.Alignment_v8 import Alignment_v8
from models.DGSTA_v9 import DGSTA_v9
from utils.semantic_topology_masks_v9 import enrich_delta_masks_for_dgsta_v9, stack_dgsta_masks_v9


class Alignment_v9(Alignment_v8):
    """
    v9 replaces SATD as the main correction path with DGSTA-lite:
    a residual alignment corrector on top of the original HairFast Eq.(8)
    feature alignment.
    """

    def __init__(self, opts, latent_encoder=None, net=None, dgsta_model_v9: DGSTA_v9 | None = None):
        super().__init__(opts, latent_encoder=latent_encoder, net=net, satd_model_v8=None)
        self.dgsta_model_v9 = dgsta_model_v9

        if self.dgsta_model_v9 is None and getattr(self.opts, "use_dgsta_v9", False):
            self.dgsta_model_v9 = DGSTA_v9().to(self.opts.device).eval()
            ckpt_path = getattr(self.opts, "dgsta_checkpoint_v9", "")
            if ckpt_path:
                ckpt = torch.load(ckpt_path, map_location=self.opts.device)
                state_dict = ckpt.get("dgsta_v9_state_dict", ckpt.get("model_state_dict", ckpt))
                self._load_compatible_state_dict(self.dgsta_model_v9, state_dict)

    def prepare_dgsta_features(self, im_name1: str, im_name2: str, name_to_embed, **kwargs):
        info = super().prepare_satd_features(im_name1, im_name2, name_to_embed, **kwargs)
        delta_masks = enrich_delta_masks_for_dgsta_v9(info["delta_masks"])
        info["delta_masks"] = delta_masks
        info["dgsta_masks_256"] = stack_dgsta_masks_v9(delta_masks)
        return info

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        dgsta_features = self.prepare_dgsta_features(im_name1, im_name2, name_to_embed, **kwargs)
        author_align = self.author_align_images(im_name1, im_name2, name_to_embed, **kwargs)
        latent_F_base = author_align["latent_F_align"]
        latent_F_align = latent_F_base

        use_dgsta = kwargs.get("use_dgsta_v9", getattr(self.opts, "use_dgsta_v9", False))
        if (
            use_dgsta
            and self.dgsta_model_v9 is not None
            and name_to_embed[im_name1]["image_256"] is not name_to_embed[im_name2]["image_256"]
        ):
            dgsta_out, _ = self.dgsta_model_v9(
                F_base=latent_F_base,
                F_src=dgsta_features["latent_F_src"],
                F_ref=dgsta_features["latent_F_ref"],
                F_src_inpaint=dgsta_features["latent_F_src_inpaint"],
                F_shape_inpaint=dgsta_features["latent_F_shape_inpaint"],
                dgsta_masks_256=dgsta_features["dgsta_masks_256"],
                source_rgb_256=dgsta_features["source_image_256"],
                shape_rgb_256=dgsta_features["shape_image_256"],
            )
            blend = kwargs.get("dgsta_blend_v9", getattr(self.opts, "dgsta_blend_v9", 1.0))
            latent_F_align = latent_F_base + blend * (dgsta_out - latent_F_base)

        return {
            "latent_F_align": latent_F_align,
            "HM_X": author_align["HM_X"],
            "delta_masks": dgsta_features["delta_masks"],
            "dgsta_masks_256": dgsta_features["dgsta_masks_256"],
            "source_image_256": dgsta_features["source_image_256"],
            "shape_image_256": dgsta_features["shape_image_256"],
            "source_inpaint_256": dgsta_features["source_inpaint_256"],
            "shape_inpaint_256": dgsta_features["shape_inpaint_256"],
        }
