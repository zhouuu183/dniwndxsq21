import torch
import torch.nn.functional as F

from models.Alignment import Alignment
from models.CtrlHair.shape_branch.solver import get_hair_face_code, get_new_shape
from models.Net import get_segmentation
from models.SATD_v8 import SATD_v8
from models.sean_codes.models.pix2pix_model import decode_sean, encode_sean
from models.earring_foreground_v6 import (
    EarringCoordinateSpace,
    extract_source_native_earring_v6,
)
from utils.mask_delta_v8 import (
    build_delta_masks,
    drop_parsing_labels,
    enrich_delta_masks_with_halo,
    filter_parsing_to_primary_subject,
    protect_cleanup_masks_v8,
    stack_cleanup_masks_v8,
)


class AlignmentV6(Alignment):
    """
    V6 keeps the author baseline as the shape/colour prior and applies SATD to
    the blended F feature.  The resulting decoded image is the direct PP
    input; no second decoded-image residual is needed.

    This implementation is intentionally self-contained so it does not depend
    on whatever Alignment_v4 version exists on the target server.
    """

    def __init__(
        self,
        opts,
        latent_encoder=None,
        net=None,
        satd_model_v8: SATD_v8 | None = None,
        base_alignment: Alignment | None = None,
    ):
        if base_alignment is None:
            super().__init__(opts, latent_encoder=latent_encoder, net=net)
        else:
            # Share the actual author modules instead of constructing a second
            # alignment stack.  HairFastV6 never invokes this object's base
            # alignment methods; it uses it solely for the SATD side branch.
            torch.nn.Module.__init__(self)
            self.opts = base_alignment.opts
            self.latent_encoder = base_alignment.latent_encoder
            self.net = base_alignment.net
            self.sean_model = base_alignment.sean_model
            self.mask_generator = base_alignment.mask_generator
            self.rotate_model = base_alignment.rotate_model
            self.dilate_erosion = base_alignment.dilate_erosion
            self.to_bisenet = base_alignment.to_bisenet
        self.satd_model_v8 = satd_model_v8

        if self.satd_model_v8 is None and getattr(self.opts, "use_satd_v8", False):
            self.satd_model_v8 = SATD_v8().to(self.opts.device).eval()
            ckpt_path = getattr(self.opts, "satd_checkpoint_v8", "")
            if ckpt_path:
                ckpt = torch.load(ckpt_path, map_location=self.opts.device)
                state_dict = ckpt.get("satd_v8_state_dict", ckpt.get("model_state_dict", ckpt))
                self._load_compatible_state_dict(self.satd_model_v8, state_dict)

    @staticmethod
    def _load_compatible_state_dict(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]):
        model_state = module.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        model_state.update(compatible)
        module.load_state_dict(model_state, strict=False)

    @staticmethod
    def _ensure_image_batch_v8(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

    @staticmethod
    def _cleanup_support_mask(cleanup_masks_256: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        (
            boundary,
            remove,
            remove_halo,
            remove_tail,
            remove_face,
            remove_neck,
            body_preserve,
        ) = torch.chunk(cleanup_masks_256, 7, dim=1)
        support = (
            0.20 * boundary
            + 0.55 * remove
            + 0.95 * remove_halo
            + 0.94 * remove_tail
            + 0.62 * remove_face
            + 0.86 * remove_neck
        ).clamp(0, 1)
        support = (support * (1.0 - 0.90 * body_preserve)).clamp(0, 1)
        return F.interpolate(support.float(), size=out_hw, mode="bicubic", align_corners=False).clamp(0, 1)

    @staticmethod
    def _cleanup_support_from_delta_masks(delta_masks: dict[str, torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:
        remove = delta_masks["M_remove"]
        zero = torch.zeros_like(remove)
        boundary = delta_masks.get("M_boundary", zero)
        remove_halo = delta_masks.get("M_remove_halo", zero)
        remove_tail = delta_masks.get("M_remove_tail", zero)
        remove_face = delta_masks.get("M_remove_face", zero)
        remove_neck = delta_masks.get("M_remove_neck", zero)
        body_preserve = delta_masks.get("M_body_preserve", zero)
        detail_protect = delta_masks.get("M_detail_protect", zero)
        face_cleanup_surface = delta_masks.get("M_face_cleanup_surface", torch.ones_like(remove_face))

        support = (
            0.16 * boundary
            + 0.56 * remove
            + 0.98 * remove_halo
            + 1.18 * remove_tail
            + 0.86 * remove_face * face_cleanup_surface
            + 1.14 * remove_neck
        ).clamp(0, 1)
        support = (support * (1.0 - 0.84 * body_preserve)).clamp(0, 1)
        support = (support * (1.0 - 0.72 * detail_protect)).clamp(0, 1)
        return F.interpolate(support.float(), size=out_hw, mode="bicubic", align_corners=False).clamp(0, 1)

    def _source_earring_satd_protection(
        self,
        name_to_embed: dict[str, dict[str, torch.Tensor]],
        image_key: str,
        out_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a source-coordinate earring guard for the SATD F edit.

        ``M_remove`` describes source-hair pixels that disappear in the target
        hairstyle.  A source earring can occupy those same pixels, especially
        a long pendant over source hair/background.  SATD must not rewrite that
        evidence before the later native-object compositor has a chance to
        restore it.  The guard is source-only and therefore never authorizes
        an ear/face/background crop as output alpha.
        """
        image_256 = name_to_embed[image_key]["image_256"]
        # SATD consumes a 256px mask, but the evidence used to build that mask
        # must come from the original aligned source.  Running the object
        # verifier on ``image_256`` permanently removes thin wires and small
        # stones before the guard is constructed; those are exactly the
        # earrings that disappear when they overlap M_remove.
        image = name_to_embed[image_key].get("image_1024", image_256)
        source_mask = name_to_embed[image_key].get("mask")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image_256.ndim == 3:
            image_256 = image_256.unsqueeze(0)
        if source_mask is None:
            zero = torch.zeros(
                image_256.size(0), 1, *image_256.shape[-2:],
                device=image_256.device,
                dtype=image_256.dtype,
            )
            return zero, F.interpolate(zero, size=out_hw, mode="nearest")
        parsing = source_mask
        if parsing.ndim == 3:
            parsing = parsing.unsqueeze(1)
        parsing = parsing.to(device=image.device).long()
        # Preserve the original RGB sampling grid.  The parsing labels are
        # categorical and can be enlarged safely with nearest-neighbour; the
        # reverse operation (shrinking native RGB to the parser/PP grid) loses
        # the very foreground evidence this guard exists to preserve.
        work_image = image.to(device=image_256.device, dtype=image_256.dtype)
        if parsing.shape[-2:] != work_image.shape[-2:]:
            parsing = F.interpolate(
                parsing.float(),
                size=work_image.shape[-2:],
                mode="nearest",
            ).long()

        # Label 9 is only a source-object seed.  The native extractor is the
        # authority used for the SATD guard; a structured search corridor is
        # intentionally not used here because it also contains background.
        parser_seed = (parsing == 9).to(dtype=work_image.dtype)
        source_object = torch.zeros_like(parser_seed)
        try:
            extracted = extract_source_native_earring_v6(
                work_image,
                parsing,
                source_native_seed=parser_seed,
                seed_space=EarringCoordinateSpace.SOURCE_CANONICAL,
                max_graph_depth=4,
                max_cumulative_cost=1.85,
                allow_long_continuation=True,
            )
            source_object = extracted["source_native_earring_alpha"].to(
                device=work_image.device,
                dtype=work_image.dtype,
            ).clamp(0, 1)
        except (RuntimeError, ValueError, KeyError):
            source_object = torch.zeros_like(parser_seed)

        # If native extraction is unavailable, retain only exact parser pixels
        # as a conservative fallback.  Do not widen this fallback before the
        # overlap test, otherwise the entire ear/background corridor is frozen.
        if float(source_object.detach().sum().item()) <= 0.0:
            source_object = parser_seed

        protect_radius = max(0, int(getattr(self.opts, "satd_earring_protect_dilate", 3)))
        if protect_radius == 0:
            protected_256 = F.adaptive_max_pool2d(
                source_object, output_size=image_256.shape[-2:]
            ).clamp(0, 1)
            protected_feature = F.interpolate(
                source_object, size=out_hw, mode="area"
            ).clamp(0, 1)
            return protected_256, protected_feature
        protect_radius = max(
            1,
            int(round(protect_radius * max(parsing.shape[-2:]) / 256.0)),
        )
        protected_256 = F.max_pool2d(
            source_object,
            kernel_size=2 * protect_radius + 1,
            stride=1,
            padding=protect_radius,
        ).clamp(0, 1)
        protected_256 = F.adaptive_max_pool2d(
            protected_256,
            output_size=(image_256.shape[-2], image_256.shape[-1]),
        ).clamp(0, 1)
        protected_feature = F.adaptive_max_pool2d(
            protected_256,
            output_size=out_hw,
        ).clamp(0, 1)
        return protected_256, protected_feature

    @torch.inference_mode()
    def shape_module(self, im_name1: str, im_name2: str, name_to_embed, only_target=True, **kwargs):
        device = self.opts.device

        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_W_1 = name_to_embed[im_name1]["W"]
        latent_W_2 = name_to_embed[im_name2]["W"]

        inp_mask1, source_support = filter_parsing_to_primary_subject(name_to_embed[im_name1]["mask"])
        inp_mask2, reference_support = filter_parsing_to_primary_subject(name_to_embed[im_name2]["mask"])

        if img1_in is not img2_in:
            rotate_to = self.rotate_model(latent_W_2[:, :6], latent_W_1[:, :6])
            rotate_to = torch.cat((rotate_to, latent_W_2[:, 6:]), dim=1)
            I_rot, _ = self.net.generator([rotate_to], input_is_latent=True, return_latents=False)

            I_rot_to_seg = self.to_bisenet(((I_rot + 1) / 2).clamp(0, 1))
            rot_mask = get_segmentation(I_rot_to_seg)
            rot_mask, _ = filter_parsing_to_primary_subject(rot_mask, subject_support=reference_support)
        else:
            rot_mask = inp_mask2

        inp_mask1_shape = drop_parsing_labels(inp_mask1)
        rot_mask_shape = drop_parsing_labels(rot_mask)

        if img1_in is not img2_in:
            face_1, _ = get_hair_face_code(self.mask_generator, inp_mask1_shape[0, 0, ...])
            _, hair_2 = get_hair_face_code(self.mask_generator, rot_mask_shape[0, 0, ...])
            target_mask = get_new_shape(self.mask_generator, face_1, hair_2)[None, None]
        else:
            target_mask = inp_mask1_shape

        target_mask, _ = filter_parsing_to_primary_subject(target_mask, subject_support=source_support)
        hair_mask_target = torch.where(
            target_mask == 13,
            torch.ones_like(target_mask, device=device),
            torch.zeros_like(target_mask, device=device),
        ).float()

        if only_target:
            return {"HM_X": hair_mask_target}

        hair_mask1 = torch.where(inp_mask1 == 13, torch.ones_like(inp_mask1, device=device), torch.zeros_like(inp_mask1, device=device)).float()
        hair_mask2 = torch.where(inp_mask2 == 13, torch.ones_like(inp_mask2, device=device), torch.zeros_like(inp_mask2, device=device)).float()
        return inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target

    def _eq8_fusion_v8(
        self,
        latent_F_1: torch.Tensor,
        latent_F_2: torch.Tensor,
        intermediate_align: torch.Tensor,
        latent_F_out_new: torch.Tensor,
        hair_mask1: torch.Tensor,
        hair_mask2: torch.Tensor,
        hair_mask_target: torch.Tensor,
    ) -> torch.Tensor:
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
    def _pure_author_shape_module(self, im_name1: str, im_name2: str, name_to_embed, only_target=True, **kwargs):
        device = self.opts.device

        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_W_1 = name_to_embed[im_name1]["W"]
        latent_W_2 = name_to_embed[im_name2]["W"]

        inp_mask1 = name_to_embed[im_name1]["mask"]
        inp_mask2 = name_to_embed[im_name2]["mask"]

        if img1_in is not img2_in:
            rotate_to = self.rotate_model(latent_W_2[:, :6], latent_W_1[:, :6])
            rotate_to = torch.cat((rotate_to, latent_W_2[:, 6:]), dim=1)
            I_rot, _ = self.net.generator([rotate_to], input_is_latent=True, return_latents=False)

            I_rot_to_seg = self.to_bisenet(((I_rot + 1) / 2).clamp(0, 1))
            rot_mask = get_segmentation(I_rot_to_seg)
        else:
            rot_mask = inp_mask2

        if img1_in is not img2_in:
            face_1, _ = get_hair_face_code(self.mask_generator, inp_mask1[0, 0, ...])
            _, hair_2 = get_hair_face_code(self.mask_generator, rot_mask[0, 0, ...])
            target_mask = get_new_shape(self.mask_generator, face_1, hair_2)[None, None]
        else:
            target_mask = inp_mask1

        hair_mask_target = torch.where(
            target_mask == 13,
            torch.ones_like(target_mask, device=device),
            torch.zeros_like(target_mask, device=device),
        ).float()

        if only_target:
            return {"HM_X": hair_mask_target}

        hair_mask1 = torch.where(inp_mask1 == 13, torch.ones_like(inp_mask1, device=device), torch.zeros_like(inp_mask1, device=device)).float()
        hair_mask2 = torch.where(inp_mask2 == 13, torch.ones_like(inp_mask2, device=device), torch.zeros_like(inp_mask2, device=device)).float()
        return inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target

    @torch.inference_mode()
    def author_align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_F_1 = name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]

        if img1_in is img2_in:
            hair_mask_target = self._pure_author_shape_module(im_name1, im_name2, name_to_embed, only_target=True, **kwargs)["HM_X"]
            return {"latent_F_align": latent_F_1, "HM_X": hair_mask_target}

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = (
            self._pure_author_shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        )

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([inp_mask1, inp_mask2], dim=0)
        img1_code, img2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = self._ensure_image_batch_v8(decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask))
        gen2_sean = self._ensure_image_batch_v8(decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask))

        enc_imgs = self.latent_encoder([gen1_sean[0], gen2_sean[0]])
        intermediate_align = enc_imgs["F"][0].unsqueeze(0)
        latent_F_out_new = enc_imgs["F"][1].unsqueeze(0)

        latent_F_align = self._eq8_fusion_v8(
            latent_F_1=latent_F_1,
            latent_F_2=latent_F_2,
            intermediate_align=intermediate_align,
            latent_F_out_new=latent_F_out_new,
            hair_mask1=hair_mask1,
            hair_mask2=hair_mask2,
            hair_mask_target=hair_mask_target,
        )
        return {"latent_F_align": latent_F_align, "HM_X": hair_mask_target}

    @torch.inference_mode()
    def v8_baseline_align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_F_1 = name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]

        if img1_in is img2_in:
            hair_mask_target = self.shape_module(im_name1, im_name2, name_to_embed, only_target=True, **kwargs)["HM_X"]
            return {"latent_F_align": latent_F_1, "HM_X": hair_mask_target}

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = (
            self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        )

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([inp_mask1, inp_mask2], dim=0)
        img1_code, img2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = self._ensure_image_batch_v8(decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask))
        gen2_sean = self._ensure_image_batch_v8(decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask))

        enc_imgs = self.latent_encoder([gen1_sean[0], gen2_sean[0]])
        intermediate_align = enc_imgs["F"][0].unsqueeze(0)
        latent_F_out_new = enc_imgs["F"][1].unsqueeze(0)

        latent_F_align = self._eq8_fusion_v8(
            latent_F_1=latent_F_1,
            latent_F_2=latent_F_2,
            intermediate_align=intermediate_align,
            latent_F_out_new=latent_F_out_new,
            hair_mask1=hair_mask1,
            hair_mask2=hair_mask2,
            hair_mask_target=hair_mask_target,
        )
        return {"latent_F_align": latent_F_align, "HM_X": hair_mask_target}

    @staticmethod
    def _reference_enhance_eq8(
        latent_F_eq8: torch.Tensor,
        latent_F_ref: torch.Tensor,
        latent_F_shape_inpaint: torch.Tensor,
        delta_masks: dict[str, torch.Tensor],
        blend: float,
    ) -> torch.Tensor:
        if blend <= 0:
            return latent_F_eq8

        add = delta_masks["M_add"]
        keep = delta_masks.get("M_keep", torch.zeros_like(add))
        boundary = delta_masks.get("M_boundary", torch.zeros_like(add))
        ref_overlap = delta_masks.get("M_ref_overlap", torch.zeros_like(add))
        body_preserve = delta_masks.get("M_body_preserve", torch.zeros_like(add))

        target_region = (add + 0.30 * keep + 0.95 * ref_overlap + 0.45 * boundary).clamp(0, 1)
        target_region = (target_region * (1 - 0.15 * body_preserve)).clamp(0, 1)
        target_region = F.interpolate(
            target_region.float(),
            size=latent_F_eq8.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)
        ref_overlap_32 = F.interpolate(
            ref_overlap.float(),
            size=latent_F_eq8.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)

        reference_anchor = latent_F_shape_inpaint + 0.40 * ref_overlap_32 * (latent_F_ref - latent_F_shape_inpaint)
        return latent_F_eq8 + blend * target_region * (reference_anchor - latent_F_eq8)

    @torch.inference_mode()
    def prepare_satd_features(self, im_name1: str, im_name2: str, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_S_1 = name_to_embed[im_name1]["S"]
        latent_F_1 = name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]
        boundary_width = kwargs.get("satd_boundary_v8", getattr(self.opts, "satd_boundary_v8", self.opts.smooth))

        if img1_in is img2_in:
            source_mask, source_support = filter_parsing_to_primary_subject(name_to_embed[im_name1]["mask"])
            hair_mask_target = self.shape_module(im_name1, im_name2, name_to_embed, only_target=True, **kwargs)["HM_X"]
            hair_mask1 = torch.where(
                source_mask == 13,
                torch.ones_like(source_mask),
                torch.zeros_like(source_mask),
            ).float()
            delta_masks = build_delta_masks(hair_mask1, hair_mask_target, hair_mask1, boundary_width=boundary_width)
            delta_masks = enrich_delta_masks_with_halo(source_mask, delta_masks, subject_support=source_support)
            cleanup_masks_256 = stack_cleanup_masks_v8(delta_masks)
            img1_norm = img1_in * 2 - 1
            return {
                "latent_F_eq8": latent_F_1,
                "latent_F_base": latent_F_1,
                "latent_F_src": latent_F_1,
                "latent_F_ref": latent_F_2,
                "latent_F_src_inpaint": latent_F_1,
                "latent_F_shape_inpaint": latent_F_1,
                "delta_masks": delta_masks,
                "cleanup_masks_256": cleanup_masks_256,
                "cleanup_masks_32": F.interpolate(cleanup_masks_256, size=(32, 32), mode="nearest"),
                "HM_X": hair_mask_target,
                "source_image_256": img1_norm,
                "shape_image_256": img1_norm,
                "source_inpaint_256": img1_norm,
                "shape_inpaint_256": img1_norm,
                "latent_S_src": latent_S_1,
            }

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = (
            self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        )

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([drop_parsing_labels(inp_mask1), drop_parsing_labels(inp_mask2)], dim=0)
        target_mask_clean = drop_parsing_labels(target_mask)
        img1_code, img2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = self._ensure_image_batch_v8(decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask_clean))
        gen2_sean = self._ensure_image_batch_v8(decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask_clean))

        enc_imgs = self.latent_encoder([gen1_sean[0], gen2_sean[0]])
        intermediate_align = enc_imgs["F"][0].unsqueeze(0)
        latent_F_out_new = enc_imgs["F"][1].unsqueeze(0)

        latent_F_eq8 = self._eq8_fusion_v8(
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
        delta_masks = enrich_delta_masks_with_halo(inp_mask1, delta_masks)
        cleanup_masks_256 = stack_cleanup_masks_v8(delta_masks)
        latent_F_base = self._reference_enhance_eq8(
            latent_F_eq8=latent_F_eq8,
            latent_F_ref=latent_F_2,
            latent_F_shape_inpaint=latent_F_out_new,
            delta_masks=delta_masks,
            blend=kwargs.get(
                "eq8_reference_blend_v8",
                getattr(self.opts, "eq8_reference_blend_v8", 0.0),
            ),
        )

        gen1_sean_256 = F.interpolate(gen1_sean, size=img1_in.shape[-2:], mode="bicubic", align_corners=False)
        gen2_sean_256 = F.interpolate(gen2_sean, size=img2_in.shape[-2:], mode="bicubic", align_corners=False)

        return {
            "latent_F_eq8": latent_F_eq8,
            "latent_F_base": latent_F_base,
            "latent_F_src": latent_F_1,
            "latent_F_ref": latent_F_2,
            "latent_F_src_inpaint": intermediate_align,
            "latent_F_shape_inpaint": latent_F_out_new,
            "delta_masks": delta_masks,
            "cleanup_masks_256": cleanup_masks_256,
            "cleanup_masks_32": F.interpolate(cleanup_masks_256, size=(32, 32), mode="nearest"),
            "HM_X": hair_mask_target,
            "source_image_256": img1_in * 2 - 1,
            "shape_image_256": img2_in * 2 - 1,
            "source_inpaint_256": gen1_sean_256,
            "shape_inpaint_256": gen2_sean_256,
            "latent_S_src": latent_S_1,
        }

    @torch.inference_mode()
    def build_satd_background_candidate(
        self,
        im_name1,
        im_name2,
        name_to_embed,
        baseline_alignment: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Build SATD's isolated image from an already-final baseline F.

        ``baseline_alignment`` is produced by :class:`models.Alignment.Alignment`
        and is never replaced.  SATD therefore changes only F before the
        StyleGAN render; the resulting image is the PP encoder input while the
        author shape/colour path remains unchanged.
        """
        satd_features = self.prepare_satd_features(im_name1, im_name2, name_to_embed, **kwargs)
        latent_F_base = baseline_alignment["latent_F_align"]
        latent_F_satd = latent_F_base
        satd_applied = False

        use_satd = kwargs.get("use_satd_v8", getattr(self.opts, "use_satd_v8", False))
        if (
            use_satd
            and self.satd_model_v8 is not None
            and name_to_embed[im_name1]["image_256"] is not name_to_embed[im_name2]["image_256"]
        ):
            # Prevent SATD from editing an earring that happens to overlap
            # M_remove.  This guard must be applied to the network's input
            # masks, not only to the residual after SATD has already changed F.
            earring_protect_256, earring_protect_feature = self._source_earring_satd_protection(
                name_to_embed,
                im_name1,
                out_hw=latent_F_base.shape[-2:],
            )
            satd_masks_256 = satd_features["cleanup_masks_256"]
            if tuple(earring_protect_256.shape[-2:]) != tuple(satd_masks_256.shape[-2:]):
                earring_protect_256 = F.interpolate(
                    earring_protect_256,
                    size=satd_masks_256.shape[-2:],
                    mode="nearest",
                )
            # Only the overlap between a verified source object and the
            # actual source-hair removal support is protected.  A full earring
            # ROI would freeze adjacent background and leave the old shadow
            # visibly dark; pixels outside M_remove remain SATD-editable.
            removal_support = satd_features["delta_masks"].get(
                "M_remove", torch.zeros_like(earring_protect_256)
            )
            if tuple(removal_support.shape[-2:]) != tuple(earring_protect_256.shape[-2:]):
                removal_support = F.interpolate(
                    removal_support,
                    size=earring_protect_256.shape[-2:],
                    mode="nearest",
                )
            earring_overlap_guard = (
                earring_protect_256 * (removal_support > 0.05).to(earring_protect_256.dtype)
            ).clamp(0, 1)
            # ``stack_cleanup_masks_v8`` stores six edit-authority channels
            # followed by M_body_preserve.  Remove earring support only from
            # the overlapping edit pixels and add preserve authority there.
            satd_masks_256 = protect_cleanup_masks_v8(
                satd_masks_256,
                earring_overlap_guard,
            )
            satd_out, _ = self.satd_model_v8(
                F_base=latent_F_base,
                F_src=satd_features["latent_F_src"],
                F_src_inpaint=satd_features["latent_F_src_inpaint"],
                cleanup_masks_256=satd_masks_256,
                source_rgb_256=satd_features["source_image_256"],
            )
            protected_delta_masks = dict(satd_features["delta_masks"])
            for key in (
                "M_boundary", "M_remove", "M_remove_halo", "M_remove_tail",
                "M_remove_face", "M_remove_neck", "M_remove_context",
            ):
                value = protected_delta_masks.get(key)
                if torch.is_tensor(value):
                    guard = earring_overlap_guard
                    if tuple(guard.shape[-2:]) != tuple(value.shape[-2:]):
                        guard = F.interpolate(guard, size=value.shape[-2:], mode="nearest")
                    protected_delta_masks[key] = value * (1.0 - guard).clamp(0, 1)
            # Preserve channels have the opposite polarity from M_remove.
            # Publish the same contract to every downstream consumer so a
            # serialized dataset or residual pass cannot re-authorize cleanup
            # at the protected source earring.
            for key in ("M_body_preserve", "M_detail_protect"):
                value = protected_delta_masks.get(key)
                if torch.is_tensor(value):
                    guard = earring_overlap_guard
                    if tuple(guard.shape[-2:]) != tuple(value.shape[-2:]):
                        guard = F.interpolate(guard, size=value.shape[-2:], mode="nearest")
                    protected_delta_masks[key] = torch.maximum(value, guard).clamp(0, 1)
            cleanup_support = self._cleanup_support_from_delta_masks(
                protected_delta_masks,
                out_hw=latent_F_base.shape[-2:],
            )
            # This is the critical ordering: protect the source earring before
            # applying SATD's residual to F.  The later target-side compositor
            # still decides whether the lobe is visible, but an earring that
            # happens to overlap M_remove is no longer erased upstream.
            cleanup_support = (
                cleanup_support * (
                    1.0
                    - F.interpolate(
                        earring_overlap_guard,
                        size=cleanup_support.shape[-2:],
                        mode="nearest",
                    )
                ).clamp(0, 1)
            ).clamp(0, 1)
            satd_blend = kwargs.get("satd_blend_v8", getattr(self.opts, "satd_blend_v8", 0.28))
            latent_F_satd = latent_F_base + satd_blend * cleanup_support * (satd_out - latent_F_base)
            satd_applied = True
            satd_delta_masks = protected_delta_masks
        else:
            earring_protect_256 = torch.zeros(
                name_to_embed[im_name1]["image_256"].size(0),
                1,
                *name_to_embed[im_name1]["image_256"].shape[-2:],
                device=latent_F_base.device,
                dtype=latent_F_base.dtype,
            )
            earring_protect_feature = torch.zeros(
                latent_F_base.size(0), 1, *latent_F_base.shape[-2:],
                device=latent_F_base.device,
                dtype=latent_F_base.dtype,
            )
            satd_delta_masks = satd_features["delta_masks"]

        return {
            "latent_F_base": latent_F_base,
            "latent_F_satd": latent_F_satd,
            "satd_applied": satd_applied,
            "HM_X": baseline_alignment["HM_X"],
            "delta_masks": satd_delta_masks,
            "source_image_256": satd_features["source_image_256"],
            "source_inpaint_256": satd_features["source_inpaint_256"],
            "shape_inpaint_256": satd_features["shape_inpaint_256"],
            "earring_protect_mask_256": earring_protect_256,
            "earring_protect_mask_feature": earring_protect_feature,
        }

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        """Run the explicit author alignment as V6's base path."""
        return self.author_align_images(im_name1, im_name2, name_to_embed, **kwargs)
