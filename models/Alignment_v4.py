import inspect

import torch
import torch.nn.functional as F

from models.Alignment import Alignment
from models.CtrlHair.shape_branch.solver import get_hair_face_code, get_new_shape
from models.Net import get_segmentation
from models.SATD_v4 import SATD_v4
from models.ear_modules_v5 import EarAnchoredQueryBuilder, FaceParsingHelperV5
from models.sean_codes.models.pix2pix_model import decode_sean, encode_sean
from utils.mask_delta_v4 import (
    build_delta_masks,
    drop_parsing_labels,
    enrich_delta_masks_with_halo,
    filter_parsing_to_primary_subject,
    stack_satd_masks,
)


class Alignment_v4(Alignment):
    """
    v4 keeps the original Eq.(8) hard fusion as the stable prior and adds
    SATD as a residual/soft refinement over the real alignment features.
    """

    def __init__(self, opts, latent_encoder=None, net=None, satd_model: SATD_v4 | None = None):
        super().__init__(opts, latent_encoder=latent_encoder, net=net)
        self.satd_model = satd_model
        self.ear_parsing_helper = FaceParsingHelperV5(
            parse_size=getattr(self.opts, "ear_parse_size_v4", 512),
        )
        self.ear_query_builder = self._build_ear_query_builder()

        if self.satd_model is None and getattr(self.opts, "use_satd_v4", False):
            self.satd_model = SATD_v4().to(self.opts.device).eval()
            ckpt_path = getattr(self.opts, "satd_checkpoint_v4", "")
            if ckpt_path:
                ckpt = torch.load(ckpt_path, map_location=self.opts.device)
                state_dict = ckpt.get("satd_state_dict", ckpt.get("model_state_dict", ckpt))
                self._load_compatible_state_dict(self.satd_model, state_dict)

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

    def _build_ear_query_builder(self) -> EarAnchoredQueryBuilder:
        candidate_kwargs = {
            "ear_dilate": getattr(self.opts, "ear_dilate_v4", 21),
            "hair_change_dilate": getattr(self.opts, "hair_change_dilate_v4", 25),
            "earring_expand": getattr(self.opts, "earring_expand_v4", 15),
            "downward_shift": getattr(self.opts, "ear_downward_shift_v4", 10),
            "target_hair_dilate": getattr(self.opts, "target_hair_dilate_v4", 11),
            "target_visibility_expand": getattr(self.opts, "target_visibility_expand_v4", 5),
            "max_target_hair_overlap": getattr(self.opts, "max_target_hair_overlap_v4", 0.55),
        }
        supported = inspect.signature(EarAnchoredQueryBuilder.__init__).parameters
        filtered_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in supported
        }
        return EarAnchoredQueryBuilder(**filtered_kwargs)

    @staticmethod
    def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

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

            I_rot_to_seg = self.to_bisenet(((I_rot + 1) / 2).clip(0, 1))
            rot_mask = get_segmentation(I_rot_to_seg)
            rot_mask, _ = filter_parsing_to_primary_subject(rot_mask, subject_support=reference_support)
        else:
            I_rot = None
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
        hair_mask_target = torch.where(target_mask == 13, torch.ones_like(target_mask, device=device), torch.zeros_like(target_mask, device=device)).float()

        if only_target:
            return {"HM_X": hair_mask_target}

        hair_mask1 = torch.where(inp_mask1 == 13, torch.ones_like(inp_mask1, device=device), torch.zeros_like(inp_mask1, device=device)).float()
        hair_mask2 = torch.where(inp_mask2 == 13, torch.ones_like(inp_mask2, device=device), torch.zeros_like(inp_mask2, device=device)).float()
        return inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target

    def _eq8_fusion(
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
    def author_align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]

        latent_F_1 = name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]

        if img1_in is img2_in:
            hair_mask_target = Alignment.shape_module(self, im_name1, im_name2, name_to_embed, only_target=True, **kwargs)["HM_X"]
            return {"latent_F_align": latent_F_1, "HM_X": hair_mask_target}

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = (
            Alignment.shape_module(self, im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        )

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([inp_mask1, inp_mask2], dim=0)

        img1_code, img2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = self._ensure_image_batch(decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask))
        gen2_sean = self._ensure_image_batch(decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask))

        enc_imgs = self.latent_encoder([gen1_sean[0], gen2_sean[0]])
        intermediate_align = enc_imgs["F"][0].unsqueeze(0)
        latent_F_out_new = enc_imgs["F"][1].unsqueeze(0)

        latent_F_align = self._eq8_fusion(
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

    @staticmethod
    def _satd_residual_masks(
        delta_masks: dict[str, torch.Tensor],
        out_hw: tuple[int, int],
        ear_query_info: dict[str, torch.Tensor] | None = None,
        cleanup_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        add = delta_masks["M_add"]
        remove = delta_masks["M_remove"]
        keep = delta_masks.get("M_keep", torch.zeros_like(remove))
        remove_halo = delta_masks.get("M_remove_halo", torch.zeros_like(remove))
        remove_tail = delta_masks.get("M_remove_tail", torch.zeros_like(remove))
        remove_face = delta_masks.get("M_remove_face", torch.zeros_like(remove))
        remove_neck = delta_masks.get("M_remove_neck", torch.zeros_like(remove))
        boundary = delta_masks["M_boundary"]
        ref_overlap = delta_masks.get("M_ref_overlap", torch.zeros_like(add))
        body_preserve = delta_masks.get("M_body_preserve", torch.zeros_like(add))

        cleanup = (
            remove
            + 0.90 * remove_halo
            + 0.60 * remove_tail
            + 0.65 * remove_face
            + 0.50 * remove_neck
        ).clamp(0, 1)

        if cleanup_only:
            transfer = torch.zeros_like(add)
            cleanup = (cleanup * (1 - 0.92 * body_preserve)).clamp(0, 1)
        else:
            transfer = (add + 0.30 * keep + 0.70 * ref_overlap + 0.35 * boundary).clamp(0, 1)
            cleanup = (cleanup + 0.10 * boundary).clamp(0, 1)

            transfer = (transfer * (1 - 0.12 * body_preserve)).clamp(0, 1)
            cleanup = (cleanup * (1 - 0.80 * body_preserve) * (1 - 0.30 * transfer)).clamp(0, 1)

        transfer = F.interpolate(transfer.float(), size=out_hw, mode="bicubic", align_corners=False).clamp(0, 1)
        cleanup = F.interpolate(cleanup.float(), size=out_hw, mode="bicubic", align_corners=False).clamp(0, 1)

        if ear_query_info is not None:
            visible_ear_roi = ear_query_info.get("visible_ear_roi")
            query_mask = ear_query_info.get("query_mask")
            source_earring_mask = ear_query_info.get("source_earring_mask")

            if visible_ear_roi is not None:
                visible_ear_roi = F.interpolate(
                    visible_ear_roi.float(),
                    size=out_hw,
                    mode="bicubic",
                    align_corners=False,
                ).clamp(0, 1)
            else:
                visible_ear_roi = torch.zeros_like(transfer)

            if query_mask is not None:
                query_mask = F.interpolate(
                    query_mask.float(),
                    size=out_hw,
                    mode="bicubic",
                    align_corners=False,
                ).clamp(0, 1)
            else:
                query_mask = torch.zeros_like(transfer)

            if source_earring_mask is not None:
                source_earring_mask = F.interpolate(
                    source_earring_mask.float(),
                    size=out_hw,
                    mode="bicubic",
                    align_corners=False,
                ).clamp(0, 1)
            else:
                source_earring_mask = torch.zeros_like(transfer)

            ear_protect = (visible_ear_roi * (source_earring_mask + 0.20 * query_mask)).clamp(0, 1)
            transfer = (transfer + 0.30 * ear_protect).clamp(0, 1)
            cleanup = (cleanup * (1 - 0.85 * ear_protect)).clamp(0, 1)

        return transfer, cleanup

    @torch.inference_mode()
    def _build_ear_query_info(
        self,
        source_image_256: torch.Tensor,
        target_image_256: torch.Tensor,
        source_hair_mask: torch.Tensor,
        target_hair_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_size = tuple(source_image_256.shape[-2:])
        source_parsing = self.ear_parsing_helper.parse(source_image_256, out_size=image_size)
        target_parsing = self.ear_parsing_helper.parse(target_image_256, out_size=image_size)
        return self.ear_query_builder(
            source_parsing,
            target_parsing,
            source_hair_mask=source_hair_mask.float(),
            target_hair_mask=target_hair_mask.float(),
        )

    @torch.inference_mode()
    def prepare_satd_features(self, im_name1: str, im_name2: str, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_S_1, latent_F_1 = name_to_embed[im_name1]["S"], name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]
        boundary_width = kwargs.get("satd_boundary_v4", getattr(self.opts, "satd_boundary_v4", self.opts.smooth))

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
            satd_masks = stack_satd_masks(delta_masks)
            img1_norm = img1_in * 2 - 1
            ear_query_info = self._build_ear_query_info(
                source_image_256=img1_in,
                target_image_256=img1_in,
                source_hair_mask=hair_mask1,
                target_hair_mask=hair_mask_target,
            )
            return {
                "latent_F_eq8": latent_F_1,
                "latent_F_base": latent_F_1,
                "latent_F_src": latent_F_1,
                "latent_F_ref": latent_F_2,
                "latent_F_src_inpaint": latent_F_1,
                "latent_F_shape_inpaint": latent_F_1,
                "delta_masks": delta_masks,
                "satd_masks_256": satd_masks,
                "satd_masks_32": F.interpolate(satd_masks, size=(32, 32), mode="nearest"),
                "HM_X": hair_mask_target,
                "source_image_256": img1_norm,
                "shape_image_256": img1_norm,
                "source_inpaint_256": img1_norm,
                "shape_inpaint_256": img1_norm,
                "latent_S_src": latent_S_1,
                "ear_query_info": ear_query_info,
            }

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = (
            self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        )

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([drop_parsing_labels(inp_mask1), drop_parsing_labels(inp_mask2)], dim=0)
        target_mask_clean = drop_parsing_labels(target_mask)
        img1_code, img2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = self._ensure_image_batch(decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask_clean))
        gen2_sean = self._ensure_image_batch(decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask_clean))

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
        delta_masks = enrich_delta_masks_with_halo(inp_mask1, delta_masks)
        satd_masks_256 = stack_satd_masks(delta_masks)
        latent_F_base = self._reference_enhance_eq8(
            latent_F_eq8=latent_F_eq8,
            latent_F_ref=latent_F_2,
            latent_F_shape_inpaint=latent_F_out_new,
            delta_masks=delta_masks,
            blend=kwargs.get(
                "eq8_reference_blend_v4",
                getattr(self.opts, "eq8_reference_blend_v4", 0.0),
            ),
        )

        gen1_sean_256 = F.interpolate(gen1_sean, size=img1_in.shape[-2:], mode="bicubic", align_corners=False)
        gen2_sean_256 = F.interpolate(gen2_sean, size=img2_in.shape[-2:], mode="bicubic", align_corners=False)
        ear_query_info = self._build_ear_query_info(
            source_image_256=img1_in,
            target_image_256=((gen2_sean_256 + 1) / 2).clamp(0, 1),
            source_hair_mask=hair_mask1,
            target_hair_mask=hair_mask_target,
        )

        return {
            "latent_F_eq8": latent_F_eq8,
            "latent_F_base": latent_F_base,
            "latent_F_src": latent_F_1,
            "latent_F_ref": latent_F_2,
            "latent_F_src_inpaint": intermediate_align,
            "latent_F_shape_inpaint": latent_F_out_new,
            "delta_masks": delta_masks,
            "satd_masks_256": satd_masks_256,
            "satd_masks_32": F.interpolate(satd_masks_256, size=(32, 32), mode="nearest"),
            "HM_X": hair_mask_target,
            "source_image_256": img1_in * 2 - 1,
            "shape_image_256": img2_in * 2 - 1,
            "source_inpaint_256": gen1_sean_256,
            "shape_inpaint_256": gen2_sean_256,
            "latent_S_src": latent_S_1,
            "ear_query_info": ear_query_info,
        }

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        satd_features = self.prepare_satd_features(im_name1, im_name2, name_to_embed, **kwargs)
        cleanup_only = kwargs.get("satd_cleanup_only_v4", getattr(self.opts, "satd_cleanup_only_v4", False))
        author_align = self.author_align_images(im_name1, im_name2, name_to_embed, **kwargs) if cleanup_only else None
        latent_F_base = author_align["latent_F_align"] if cleanup_only else satd_features.get("latent_F_base", satd_features["latent_F_eq8"])
        latent_F_align = latent_F_base

        use_satd = kwargs.get("use_satd_v4", getattr(self.opts, "use_satd_v4", False))
        if (
            use_satd
            and self.satd_model is not None
            and name_to_embed[im_name1]["image_256"] is not name_to_embed[im_name2]["image_256"]
        ):
            satd_out, satd_aux = self.satd_model(
                F_eq8=latent_F_base,
                F_src=satd_features["latent_F_src"],
                F_src_inpaint=satd_features["latent_F_src_inpaint"],
                F_shape_inpaint=satd_features["latent_F_shape_inpaint"],
                F_ref=satd_features["latent_F_ref"],
                masks_256=satd_features["satd_masks_256"],
                source_rgb_256=satd_features["source_image_256"],
            )
            satd_blend = kwargs.get("satd_blend_v4", getattr(self.opts, "satd_blend_v4", 0.25))
            transfer_mask, cleanup_mask = self._satd_residual_masks(
                satd_features["delta_masks"],
                out_hw=latent_F_align.shape[-2:],
                ear_query_info=satd_features.get("ear_query_info"),
                cleanup_only=cleanup_only,
            )
            default_delta = satd_out - latent_F_base
            transfer_delta = satd_aux.get("transfer_delta", default_delta)
            cleanup_delta = satd_aux.get("cleanup_delta", default_delta)
            if cleanup_only:
                cleanup_blend = kwargs.get(
                    "satd_cleanup_blend_v4",
                    getattr(self.opts, "satd_cleanup_blend_v4", 0.24),
                )
                latent_F_align = latent_F_align + cleanup_blend * cleanup_mask * cleanup_delta
            else:
                latent_F_align = latent_F_align + satd_blend * transfer_mask * transfer_delta
                latent_F_align = latent_F_align + (0.60 * satd_blend) * cleanup_mask * cleanup_delta

        return {
            "latent_F_align": latent_F_align,
            "HM_X": author_align["HM_X"] if author_align is not None else satd_features["HM_X"],
            "delta_masks": satd_features["delta_masks"],
            "source_image_256": satd_features["source_image_256"],
            "source_inpaint_256": satd_features["source_inpaint_256"],
            "shape_inpaint_256": satd_features["shape_inpaint_256"],
        }
