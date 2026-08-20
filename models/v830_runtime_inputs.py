"""Shared parsing and tensor contract for V2.30 diagnostic/runtime parity."""

from __future__ import annotations

import torch
import torch.nn.functional as F


PARSER_LABELS_V830 = {
    "background": 0,
    "skin": 1,
    "nose": 2,
    "eye_glasses": 3,
    "left_eye": 4,
    "right_eye": 5,
    "left_brow": 6,
    "right_brow": 7,
    "left_ear": 8,
    "right_ear": 9,
    "mouth": 10,
    "upper_lip": 11,
    "lower_lip": 12,
    "hair": 13,
    "hat": 14,
    "ear_ring": 15,
    "necklace": 16,
    "neck": 17,
    "cloth": 18,
}

FACE_LABELS_V830 = tuple(range(1, 13))
SKIN_LABELS_V830 = (1,)
EAR_LABELS_V830 = (8, 9, 15)
NECK_LABELS_V830 = (16, 17)
CLOTH_LABELS_V830 = (18,)
SUBJECT_LABELS_V830 = tuple(range(1, 19))


def _resize_mask(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.dim() == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.dim() == 3:
        value = value.unsqueeze(1)
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value.float(), size=reference.shape[-2:], mode="nearest")
    return value.to(device=reference.device, dtype=reference.dtype).clamp(0, 1)


def _labels_mask(labels: torch.Tensor, selected: tuple[int, ...]) -> torch.Tensor:
    result = torch.zeros_like(labels, dtype=torch.bool)
    for label in selected:
        result |= labels == label
    return result.float()


def build_parser_regions_v830(
    parser_labels: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if parser_labels.dim() == 2:
        parser_labels = parser_labels.unsqueeze(0).unsqueeze(0)
    elif parser_labels.dim() == 3:
        parser_labels = parser_labels.unsqueeze(1)
    parser_labels = parser_labels.to(device=reference.device)
    if parser_labels.shape[-2:] != reference.shape[-2:]:
        parser_labels = F.interpolate(
            parser_labels.float(), size=reference.shape[-2:], mode="nearest"
        ).long()
    return {
        "source_subject_mask": _labels_mask(parser_labels, SUBJECT_LABELS_V830).to(reference.dtype),
        "source_face_mask": _labels_mask(parser_labels, FACE_LABELS_V830).to(reference.dtype),
        "source_skin_mask": _labels_mask(parser_labels, SKIN_LABELS_V830).to(reference.dtype),
        "source_ear_mask": _labels_mask(parser_labels, EAR_LABELS_V830).to(reference.dtype),
        "source_neck_mask": _labels_mask(parser_labels, NECK_LABELS_V830).to(reference.dtype),
        "source_cloth_mask": _labels_mask(parser_labels, CLOTH_LABELS_V830).to(reference.dtype),
    }


def parser_region_audit_v830() -> dict[str, object]:
    return {
        "true_skin_mask_available": True,
        "source": "models.CtrlHair.global_value_utils.PARSING_LABEL_LIST via FaceParsing_tensor.swap_parsing_label_to_celeba_mask",
        "labels": PARSER_LABELS_V830,
        "face_labels": list(FACE_LABELS_V830),
        "skin_labels": list(SKIN_LABELS_V830),
        "ear_labels": list(EAR_LABELS_V830),
        "neck_labels": list(NECK_LABELS_V830),
        "cloth_labels": list(CLOTH_LABELS_V830),
        "hair_label": PARSER_LABELS_V830["hair"],
        "passed": True,
    }


def build_v830_runtime_inputs(
    *,
    base_rgb: torch.Tensor,
    anchor_rgb: torch.Tensor,
    v226_rgb: torch.Tensor,
    v229_prepp_rgb: torch.Tensor,
    pp_original_rgb: torch.Tensor,
    target_hair_mask: torch.Tensor,
    target_hair_eroded: torch.Tensor,
    target_hair_dilated: torch.Tensor,
    parser_labels: torch.Tensor,
    final_unlock_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    images = {
        "base_rgb": base_rgb,
        "anchor_rgb": anchor_rgb,
        "v226_rgb": v226_rgb,
        "v229_prepp_rgb": v229_prepp_rgb,
    }
    for name, value in images.items():
        if value.shape != base_rgb.shape:
            raise ValueError(f"V2.30 {name} must match base_rgb")
        images[name] = value.to(device=base_rgb.device, dtype=base_rgb.dtype).clamp(0, 1)
    hair = _resize_mask(target_hair_mask, base_rgb[:, :1])
    regions = build_parser_regions_v830(parser_labels, hair)
    unlock = torch.zeros_like(hair) if final_unlock_mask is None else _resize_mask(
        final_unlock_mask, hair
    )
    pp_original_rgb = pp_original_rgb.to(
        device=base_rgb.device, dtype=base_rgb.dtype
    ).clamp(0, 1)
    if pp_original_rgb.size(0) != base_rgb.size(0) or pp_original_rgb.size(1) != 3:
        raise ValueError("V2.30 pp_original_rgb must have batch RGB channels")
    return {
        **images,
        "pp_original_rgb": pp_original_rgb,
        "target_hair_mask": hair,
        "target_hair_eroded": _resize_mask(target_hair_eroded, hair),
        "target_hair_dilated": _resize_mask(target_hair_dilated, hair),
        "final_unlock_mask": unlock,
        **regions,
    }


__all__ = [
    "PARSER_LABELS_V830",
    "build_parser_regions_v830",
    "build_v830_runtime_inputs",
    "parser_region_audit_v830",
]
