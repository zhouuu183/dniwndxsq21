import torch

from models.Blending import Blending
from utils.hair_color_match_v8 import rgb_to_lab, lab_to_rgb
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents


class Blending_v8(Blending):
    """
    v8 keeps the stable blending branch, but makes the implementation
    self-contained so it does not depend on Blending_v4 from the server.
    """

    @torch.inference_mode()
    def _prepare_color_transfer(self, align_shape, name_to_embed):
        """Build the prepared v8 color-transfer state for both v8 and v5.

        The v5 refinement stage must consume exactly the same hair mask and
        latent color blend as the standalone v8 pipeline.  Keeping this part
        here prevents the two pipelines from silently drifting apart.
        """
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_2 = name_to_embed["shape"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        face_mask, _ = filter_parsing_to_primary_subject(name_to_embed["face"]["mask"])
        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        HM_1 = torch.where(face_mask == 13, torch.ones_like(face_mask), torch.zeros_like(face_mask)).float()
        HM_3 = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        HM_1D, _ = self.dilate_erosion.mask(HM_1)
        HM_3D, HM_3E = self.dilate_erosion.mask(HM_3)

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
                I_3 * HM_3E,
            )
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        return I_1, I_2, I_3, latent_F_align, HM_3E, target_mask, S_blend

    @staticmethod
    def _correct_hair_chroma_drift(
        image: torch.Tensor,
        reference: torch.Tensor,
        hair_mask: torch.Tensor,
        strength: float = 0.5,
    ) -> torch.Tensor:
        """Correct A/B channel (hue/chroma) drift in the blended hair region.

        After SATD modifies latent_F_align, even with L-channel (brightness)
        correction via F-space statistics restoration, the A/B channels (hue)
        still drift because the generator's colour rendering depends on the
        interaction between S_blend and the SATD-modified F spatial patterns,
        not just F statistics.  The blending_encoder cannot fully compensate
        for this drift because it does not observe F directly.

        This function applies a LOCAL mean-shift correction in LAB A/B space:
        within the hair mask, it shifts the blended image's A/B channel means
        toward the reference's A/B means.  Only MEAN is corrected (not variance)
        to avoid the "fake highlight" and "painted filter" artifacts seen in
        v16/v22 (which did full histogram matching).  The L channel is untouched
        (already corrected by F-space statistics preservation).

        strength=0.0 : no correction (same as before).
        strength=0.5 : default — corrects half the hue drift, avoids over-correction.
        strength=1.0 : full mean-shift to reference hue.
        """
        if strength <= 0.0:
            return image

        # Ensure 4D [B, C, H, W] and resize mask to image resolution
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if reference.dim() == 3:
            reference = reference.unsqueeze(0)
        m = hair_mask.float()
        while m.dim() < 4:
            m = m.unsqueeze(0)
        if m.shape[1] != 1:
            m = m[:, :1]
        if m.shape[-2:] != image.shape[-2:]:
            m = torch.nn.functional.interpolate(
                m, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        m = m.clamp(0, 1)
        if m.size(0) == 1 and image.size(0) > 1:
            m = m.expand(image.size(0), -1, -1, -1)

        area = m.sum(dim=(-2, -1), keepdim=True).clamp(min=4.0)
        if m.max().item() < 1e-3:
            return image

        # Convert to LAB
        image_lab = rgb_to_lab(image)
        ref_lab = rgb_to_lab(reference)

        # Compute masked mean of A/B channels in hair region
        m_ch = m.expand_as(image_lab)
        image_ab = image_lab[:, 1:3]
        ref_ab = ref_lab[:, 1:3]

        image_ab_mean = (image_ab * m_ch[:, 1:3]).sum(dim=(-2, -1), keepdim=True) / area
        ref_ab_mean = (ref_ab * m_ch[:, 1:3]).sum(dim=(-2, -1), keepdim=True) / area

        # Apply mean-shift correction to A/B channels only (L untouched)
        delta_ab = ref_ab_mean - image_ab_mean
        corrected_ab = image_ab + float(strength) * delta_ab * m_ch[:, 1:3]

        # Reconstruct LAB with corrected A/B and original L
        corrected_lab = torch.cat([image_lab[:, 0:1], corrected_ab], dim=1)
        corrected_rgb = lab_to_rgb(corrected_lab)

        # Blend correction into original image only in hair region
        out = corrected_rgb * m + image * (1.0 - m)
        return out.clamp(0, 1)

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        I_1, I_2, I_3, latent_F_align, HM_3E, _, S_blend = self._prepare_color_transfer(
            align_shape,
            name_to_embed,
        )
        I_blend, _ = self.net.generator(
            [S_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_align,
        )
        I_blend_256 = self.downsample_256(I_blend)

        # Correct A/B channel (hue/chroma) drift in the hair region caused by
        # SATD's F modification.  The L channel (brightness) was already fixed
        # by F-space statistics restoration in Alignment_v8; this step fixes
        # the remaining hue drift that the blending_encoder cannot compensate
        # for (because it does not observe F directly).  Only mean-shift is
        # applied (no variance scaling) to avoid v16/v22 artifacts.
        chroma_correct_strength = float(kwargs.get(
            "blend_chroma_correct_strength",
            getattr(self.opts, "blend_chroma_correct_strength", 0.0),
        ))
        if chroma_correct_strength > 0.0:
            # Use the aligned shape's hair mask as the correction region
            HM_X = align_shape.get("HM_X")
            if HM_X is not None:
                I_3_256 = self.downsample_256(I_3)
                I_blend_256 = self._correct_hair_chroma_drift(
                    I_blend_256,
                    I_3_256,
                    HM_X,
                    strength=chroma_correct_strength,
                )

        S_final, F_final = self.post_process(I_1, I_blend_256)
        I_final, _ = self.net.generator(
            [S_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=F_final,
        )

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name")
            exp_name = exp_name if exp_name is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v8", "blending.png", I_blend)
            save_latents(output_dir, "Blending_v8", "blending.npz", S_blend=S_blend)
            save_gen_image(output_dir, "Final_v8", "final.png", I_final)
            save_latents(output_dir, "Final_v8", "final.npz", S_final=S_final, F_final=F_final)

        return ((I_final[0] + 1) / 2).clamp(0, 1)
