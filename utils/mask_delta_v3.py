import torch
import torch.nn.functional as F


def _ensure_4d(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask.float()


def _binary(mask: torch.Tensor) -> torch.Tensor:
    return (_ensure_4d(mask) > 0.5).float()


def boundary_band(mask: torch.Tensor, width: int = 5) -> torch.Tensor:
    mask = _binary(mask)
    if width <= 0:
        return torch.zeros_like(mask)

    dilated = mask
    eroded = mask
    for _ in range(width):
        dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
        eroded = 1 - F.max_pool2d(1 - eroded, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


def build_delta_masks(
    source_hair: torch.Tensor,
    target_hair: torch.Tensor,
    ref_hair: torch.Tensor | None = None,
    boundary_width: int = 5,
) -> dict[str, torch.Tensor]:
    src = _binary(source_hair)
    tgt = _binary(target_hair)
    ref = _binary(ref_hair) if ref_hair is not None else tgt.clone()

    keep = src * tgt
    add = tgt * (1 - src)
    remove = src * (1 - tgt)
    ref_overlap = ref * tgt
    boundary = boundary_band((add + remove).clamp(0, 1), width=boundary_width)

    return {
        "M_src": src,
        "M_tgt": tgt,
        "M_add": add,
        "M_remove": remove,
        "M_keep": keep,
        "M_boundary": boundary,
        "M_ref_overlap": ref_overlap,
    }


def stack_satd_masks(delta_masks: dict[str, torch.Tensor], size: tuple[int, int] | None = None) -> torch.Tensor:
    masks = torch.cat(
        [
            delta_masks["M_add"],
            delta_masks["M_remove"],
            delta_masks["M_keep"],
            delta_masks["M_boundary"],
            delta_masks["M_ref_overlap"],
        ],
        dim=1,
    ).float()

    if size is not None:
        masks = F.interpolate(masks, size=size, mode="nearest")
    return masks
