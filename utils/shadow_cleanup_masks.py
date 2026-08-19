import torch
import torch.nn.functional as F


HAIR_LABELS = (13,)
NECK_LABELS = (16, 17)
CLOTH_LABELS = (18,)


def _ensure_batch(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return mask


def _binary(mask: torch.Tensor) -> torch.Tensor:
    return (_ensure_batch(mask) > 0.5).float()


def _label_mask(parsing_mask: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    parsing_mask = _ensure_batch(parsing_mask)
    mask = torch.zeros_like(parsing_mask, dtype=torch.bool)
    for label in labels:
        mask |= parsing_mask == label
    return mask.float()


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)


def build_shadow_cleanup_masks(
    source_parsing: torch.Tensor,
    target_hair_mask: torch.Tensor,
    ring_width: int = 7,
    halo_width: int = 9,
    protect_width: int = 2,
) -> dict[str, torch.Tensor]:
    parsing = _ensure_batch(source_parsing)
    target = _binary(target_hair_mask)

    src_hair = _label_mask(parsing, HAIR_LABELS)
    neck = _label_mask(parsing, NECK_LABELS)
    cloth = _label_mask(parsing, CLOTH_LABELS)
    body = (neck + cloth).clamp(0, 1)

    target_protect = _dilate(target, protect_width)
    source_visible = ((1 - src_hair) * (1 - target_protect)).clamp(0, 1)

    remove = (src_hair * (1 - target)).clamp(0, 1)
    shadow_ring = (_dilate(target, ring_width) - target).clamp(0, 1)
    shadow_ring = (shadow_ring * source_visible).clamp(0, 1)

    remove_halo = (_dilate(remove, halo_width) - remove).clamp(0, 1)
    remove_halo = (remove_halo * (1 - target) * source_visible).clamp(0, 1)

    _, _, height, _ = parsing.shape
    y_coords = torch.linspace(0.0, 1.0, steps=height, device=parsing.device).view(1, 1, height, 1)
    lower_region = (y_coords > 0.58).float()
    body_preserve = (body * (1 - target)).clamp(0, 1)
    remove_tail = ((remove + 0.85 * remove_halo) * lower_region * (1 - body) * (1 - target)).clamp(0, 1)

    cleanup = (shadow_ring + 0.65 * remove_halo + 0.35 * remove_tail).clamp(0, 1)
    cleanup = (cleanup * (1 - 0.70 * body_preserve)).clamp(0, 1)

    source_copy = (shadow_ring + 0.35 * remove_halo).clamp(0, 1)
    source_copy = (source_copy * source_visible * (1 - 0.75 * body_preserve)).clamp(0, 1)

    return {
        "M_tgt": target,
        "M_src_hair": src_hair,
        "M_remove": remove,
        "M_shadow_ring": shadow_ring,
        "M_remove_halo": remove_halo,
        "M_remove_tail": remove_tail,
        "M_body_preserve": body_preserve,
        "M_source_visible": source_visible,
        "M_source_copy": source_copy,
        "M_cleanup": cleanup,
    }
