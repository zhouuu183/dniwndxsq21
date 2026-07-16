import torch
import torch.nn.functional as F

from losses.pp_losses import LossBuilder, LossBuilderMulti, normalize


# v1 modification:
# The new pp loss keeps the original objectives and adds explicit supervision
# for remove/disocclusion regions and boundary harmonization.


class LossBuilderMultiV1(LossBuilderMulti):
    def __call__(
        self,
        source,
        target,
        target_mask,
        HT_E,
        gen_w,
        F_w,
        gen_F,
        F_gen,
        remove_mask=None,
        boundary_mask=None,
        **kwargs,
    ):
        losses = super().__call__(source, target, target_mask, HT_E, gen_w, F_w, gen_F, F_gen, **kwargs)
        gen_F_256 = self.downsample_256(gen_F)

        if remove_mask is not None:
            remove_mask = remove_mask.float()
            losses["delta remove lpips"] = 0.5 * self.LPIPS(normalize(source) * remove_mask, gen_F_256 * remove_mask)
            losses["delta remove l1"] = 2.0 * F.l1_loss(((gen_F_256 + 1) / 2) * remove_mask, source * remove_mask)

        if boundary_mask is not None:
            boundary_mask = boundary_mask.float()
            losses["delta boundary l1"] = 1.5 * F.l1_loss(((gen_F_256 + 1) / 2) * boundary_mask, source * boundary_mask)

        return losses


__all__ = ["LossBuilder", "LossBuilderMultiV1"]
