import torch
import torch.nn.functional as F


# v1 modification:
# This helper centralizes the delta-mask computation used by the new
# blending/pp pipeline. The original project only passes HM_X; v1 explicitly
# builds source/target/add/remove/keep/boundary masks for edit decomposition.


def ensure_binary_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return (mask > 0.5).float()


def binary_dilate(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    out = ensure_binary_mask(mask)
    for _ in range(max(1, iterations)):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
    return (out > 0).float()


def binary_erode(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    out = ensure_binary_mask(mask)
    for _ in range(max(1, iterations)):
        out = 1.0 - F.max_pool2d(1.0 - out, kernel_size=3, stride=1, padding=1)
    return (out > 0.5).float()


def downsample_mask(mask: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(mask.float(), size=size, mode=mode)
    return F.interpolate(mask.float(), size=size, mode=mode, align_corners=False)


def compute_delta_masks(
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    boundary_width: int = 5,
) -> dict[str, torch.Tensor]:
    src = ensure_binary_mask(source_mask)
    tgt = ensure_binary_mask(target_mask)

    add = (tgt * (1.0 - src)).float()
    remove = (src * (1.0 - tgt)).float()
    keep = (src * tgt).float()
    change = (add + remove).clamp(0.0, 1.0)

    boundary_outer = binary_dilate(change, iterations=boundary_width)
    boundary_inner = binary_erode(change, iterations=max(1, boundary_width // 2))
    boundary = (boundary_outer - boundary_inner).clamp(0.0, 1.0)

    return {
        "M_src": src,
        "M_tgt": tgt,
        "M_add": add,
        "M_remove": remove,
        "M_keep": keep,
        "M_boundary": boundary,
        "M_change": change,
    }
