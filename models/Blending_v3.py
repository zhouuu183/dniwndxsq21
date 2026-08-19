import torch

from models.Blending import Blending


class Blending_v3(Blending):
    """
    Prefer the shape-side target hair mask when shape/color references differ.
    """

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        if "HM_X" in align_shape:
            align_color = dict(align_color)
            align_color["HM_X"] = align_shape["HM_X"]
        return super().blend_images(align_shape, align_color, name_to_embed, **kwargs)
