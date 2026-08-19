import torch
import torch.nn.functional as F


DGSTA_MASK_KEYS_V9 = (
    "M_remove",
    "M_add",
    "M_keep",
    "M_boundary",
    "M_remove_halo",
    "M_remove_face",
    "M_remove_neck",
    "M_remove_tail",
    "M_body_preserve",
    "M_visible_body_anchor",
    "M_body_reveal_only",
    "M_context_reveal_only",
    "M_reveal_overlap",
    "M_detail_protect",
    "M_bang",
    "M_bundle",
)


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=width)


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size=kernel, stride=1, padding=width)


def _soft_boundary(mask: torch.Tensor, width: int = 2) -> torch.Tensor:
    return (_dilate(mask, width) - _erode(mask, width)).clamp(0, 1)


def enrich_delta_masks_for_dgsta_v9(delta_masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remove = delta_masks["M_remove"]
    zero = torch.zeros_like(remove)
    tgt = delta_masks.get("M_tgt", zero)
    add = delta_masks.get("M_add", zero)
    keep = delta_masks.get("M_keep", zero)
    boundary = delta_masks.get("M_boundary", zero)
    face_surface = delta_masks.get("M_face_surface", zero)
    face_region = delta_masks.get("M_face_region", zero)
    visible_body_anchor = delta_masks.get("M_visible_body_anchor", zero)
    detail_protect = delta_masks.get("M_detail_protect", zero)
    ref_overlap = delta_masks.get("M_ref_overlap", zero)

    _, _, height, _ = remove.shape
    y_coords = torch.linspace(0.0, 1.0, steps=height, device=remove.device, dtype=remove.dtype).view(1, 1, height, 1)
    upper_face_band = (y_coords < 0.56).float()

    face_near_hair = (_dilate(face_surface + 0.35 * face_region, 18) * upper_face_band).clamp(0, 1)
    target_hair_edge = _soft_boundary(tgt, width=2)
    bang = (tgt * face_near_hair * (target_hair_edge + 0.35 * ref_overlap + 0.25 * keep)).clamp(0, 1)
    bang = (bang * (1.0 - 0.55 * detail_protect)).clamp(0, 1)

    target_topology_edge = _soft_boundary(tgt, width=3)
    add_keep_topology = (add + 0.55 * keep + 0.45 * boundary + 0.65 * ref_overlap).clamp(0, 1)
    bundle = (add_keep_topology * (0.55 + 0.45 * target_topology_edge)).clamp(0, 1)
    bundle = (bundle * (1.0 - 0.45 * visible_body_anchor)).clamp(0, 1)

    enriched = dict(delta_masks)
    enriched.update(
        {
            "M_bang": bang,
            "M_bundle": bundle,
        }
    )
    return enriched


def stack_dgsta_masks_v9(delta_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    remove = delta_masks["M_remove"]
    zero = torch.zeros_like(remove)
    return torch.cat([delta_masks.get(key, zero) for key in DGSTA_MASK_KEYS_V9], dim=1)


__all__ = [
    "DGSTA_MASK_KEYS_V9",
    "enrich_delta_masks_for_dgsta_v9",
    "stack_dgsta_masks_v9",
]
