from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# CelebA-style labels after FaceParsing_tensor.swap_parsing_label_to_celeba_mask().
BACKGROUND_LABEL = 0
FACE_SURFACE_LABELS = (1, 2)
DETAIL_BLOCK_LABELS = (3, 4, 5, 6, 7, 10, 11, 12, 14)
EAR_LABELS = (8, 9)
EARRING_LABELS = (15,)
HAIR_LABELS = (13,)
NECK_CLOTH_LABELS = (16, 17, 18)


@dataclass(frozen=True)
class RestoreConfig:
    ear_roi_dilate: int = 9
    ear_lobe_shift: int = 10
    source_hair_exclude_dilate: int = 2
    target_hair_exclude_dilate: int = 5
    weak_high_threshold: float = 0.030
    weak_chroma_threshold: float = 0.050
    weak_contrast_threshold: float = 0.026
    support_high_threshold: float = 0.012
    support_chroma_threshold: float = 0.028
    support_contrast_threshold: float = 0.014
    object_seed_dilate: int = 1
    object_support_dilate: int = 2
    object_grow_iters: int = 1
    weak_fallback: bool = True
    parser_weak_support_dilate: int = 4
    component_min_area: int = 3
    component_max_area_ratio: float = 0.020
    component_keep_top: int = 6
    safe_region_dilate: int = 2
    face_contact_dilate: int = 1
    align_max_shift: int = 0
    alpha_strength: float = 0.92
    alpha_core_dilate: int = 0
    alpha_feather_dilate: int = 1
    alpha_feather_strength: float = 0.35
    alpha_blur_kernel: int = 3
    alpha_blur_sigma: float = 1.4


def _ensure_bchw(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def _normalize_image(image: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = _ensure_bchw(image).to(device=device, dtype=dtype)
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    if image.max().item() > 2.0:
        image = image / 255.0
    if image.min().item() < -0.05:
        image = (image + 1.0) / 2.0
    return image.clamp(0, 1)


def _resize_image(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if image.shape[-2:] == size:
        return image
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False).clamp(0, 1)


def _resize_mask(mask: torch.Tensor | None, size: tuple[int, int], mode: str = "nearest") -> torch.Tensor | None:
    mask = _ensure_bchw(mask)
    if mask is None:
        return None
    if mask.shape[-2:] == size:
        return mask.float()
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    return F.interpolate(mask.float(), size=size, mode=mode, align_corners=align_corners).clamp(0, 1)


def _resize_label_map(label_map: torch.Tensor | None, size: tuple[int, int]) -> torch.Tensor | None:
    label_map = _ensure_bchw(label_map)
    if label_map is None:
        return None
    if label_map.shape[-2:] == size:
        return label_map.long()
    return F.interpolate(label_map.float(), size=size, mode="nearest").long()


def _label_mask(parsing: torch.Tensor | None, labels: tuple[int, ...]) -> torch.Tensor | None:
    if parsing is None:
        return None
    parsing = _ensure_bchw(parsing).long()
    mask = torch.zeros_like(parsing, dtype=torch.bool)
    for label in labels:
        mask |= parsing == label
    return mask.float()


def _dilate(mask: torch.Tensor | None, kernel_size: int) -> torch.Tensor | None:
    mask = _ensure_bchw(mask)
    if mask is None:
        return None
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return mask.float().clamp(0, 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.max_pool2d(mask.float(), kernel_size, stride=1, padding=kernel_size // 2).clamp(0, 1)


def _scaled(value: int | float, size: tuple[int, int], base: int = 256, minimum: int = 0) -> int:
    scale = max(size) / float(base)
    return max(minimum, int(round(float(value) * scale)))


def _scaled_area(value: int | float, size: tuple[int, int], base: int = 256) -> int:
    scale = max(size) / float(base)
    return max(1, int(round(float(value) * scale * scale)))


def _translate_tensor(tensor: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    tensor = _ensure_bchw(tensor)
    if dx == 0 and dy == 0:
        return tensor

    out = torch.zeros_like(tensor)
    _, _, h, w = tensor.shape
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx) if dx < 0 else w
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy) if dy < 0 else h

    if src_x1 > src_x0 and src_y1 > src_y0 and dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[:, :, src_y0:src_y1, src_x0:src_x1]
    return out


def _mask_center(mask: torch.Tensor) -> tuple[float, float] | None:
    mask = _ensure_bchw(mask).float()[0, 0]
    total = mask.sum().item()
    if total <= 1e-6:
        return None
    ys = torch.arange(mask.shape[0], device=mask.device, dtype=mask.dtype)
    xs = torch.arange(mask.shape[1], device=mask.device, dtype=mask.dtype)
    cy = (mask.sum(dim=1) * ys).sum().item() / total
    cx = (mask.sum(dim=0) * xs).sum().item() / total
    return cx, cy


def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-6)


def _gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    image = _ensure_bchw(image).float()
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(float(sigma), 1e-3)
    kernel = _gaussian_kernel(kernel_size, sigma, image.device, image.dtype)
    channels = image.shape[1]
    pad = kernel_size // 2
    kernel_x = kernel.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
    out = F.conv2d(F.pad(image, (pad, pad, 0, 0), mode="reflect"), kernel_x, groups=channels)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), kernel_y, groups=channels)
    return out


def _rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    return image[:, 0:1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114


def _filter_components(
    mask: torch.Tensor,
    min_area: int,
    max_area_ratio: float,
    keep_top: int,
) -> torch.Tensor:
    mask = (_ensure_bchw(mask).float() > 0.5).float()
    if mask.flatten(1).amax(dim=1).sum().item() <= 0:
        return mask

    kept = torch.zeros_like(mask)
    b, _, h, w = mask.shape
    max_area = int(max_area_ratio * h * w) if max_area_ratio > 0 else h * w
    max_area = max(max_area, min_area)

    for batch_idx in range(b):
        arr = mask[batch_idx, 0].detach().cpu().numpy().astype(np.bool_)
        visited = np.zeros_like(arr, dtype=np.bool_)
        components: list[list[tuple[int, int]]] = []
        ys, xs = np.nonzero(arr)

        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for ny in range(max(0, y - 1), min(h, y + 2)):
                    for nx in range(max(0, x - 1), min(w, x + 2)):
                        if not visited[ny, nx] and arr[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            area = len(comp)
            if min_area <= area <= max_area:
                components.append(comp)

        components.sort(key=len, reverse=True)
        for comp in components[: max(1, int(keep_top))]:
            ys_comp, xs_comp = zip(*comp)
            kept[batch_idx, 0, list(ys_comp), list(xs_comp)] = 1.0

    return kept


class EarringDirectRestore(nn.Module):
    """
    Untrained direct earring restore after the author's original post-process.

    The module only uses parser/texture masks to decide alpha. The pasted RGB is
    always taken from the source image, while the original PP output remains the
    base image.
    """

    def __init__(self, opts=None):
        super().__init__()
        self.opts = opts
        self.enabled = bool(getattr(opts, "use_earring_direct_restore", False)) if opts is not None else False
        self.config = RestoreConfig(
            ear_roi_dilate=int(getattr(opts, "earring_restore_ear_roi_dilate", 9)),
            ear_lobe_shift=int(getattr(opts, "earring_restore_ear_lobe_shift", 10)),
            source_hair_exclude_dilate=int(getattr(opts, "earring_restore_source_hair_exclude", 2)),
            target_hair_exclude_dilate=int(getattr(opts, "earring_restore_target_hair_exclude", 5)),
            weak_high_threshold=float(getattr(opts, "earring_restore_weak_high", 0.030)),
            weak_chroma_threshold=float(getattr(opts, "earring_restore_weak_chroma", 0.050)),
            weak_contrast_threshold=float(getattr(opts, "earring_restore_weak_contrast", 0.026)),
            support_high_threshold=float(getattr(opts, "earring_restore_support_high", 0.012)),
            support_chroma_threshold=float(getattr(opts, "earring_restore_support_chroma", 0.028)),
            support_contrast_threshold=float(getattr(opts, "earring_restore_support_contrast", 0.014)),
            object_seed_dilate=int(getattr(opts, "earring_restore_object_seed_dilate", 1)),
            object_support_dilate=int(getattr(opts, "earring_restore_object_support_dilate", 2)),
            object_grow_iters=int(getattr(opts, "earring_restore_object_grow_iters", 1)),
            weak_fallback=not bool(getattr(opts, "disable_earring_restore_weak_fallback", False)),
            parser_weak_support_dilate=int(getattr(opts, "earring_restore_parser_weak_support_dilate", 4)),
            component_min_area=int(getattr(opts, "earring_restore_component_min_area", 3)),
            component_max_area_ratio=float(getattr(opts, "earring_restore_component_max_area_ratio", 0.020)),
            component_keep_top=int(getattr(opts, "earring_restore_component_keep_top", 6)),
            safe_region_dilate=int(getattr(opts, "earring_restore_safe_region_dilate", 2)),
            face_contact_dilate=int(getattr(opts, "earring_restore_face_contact_dilate", 1)),
            align_max_shift=int(getattr(opts, "earring_restore_align_max_shift", 0)),
            alpha_strength=float(getattr(opts, "earring_restore_alpha_strength", 0.92)),
            alpha_core_dilate=int(getattr(opts, "earring_restore_alpha_core_dilate", 0)),
            alpha_feather_dilate=int(getattr(opts, "earring_restore_alpha_feather_dilate", 1)),
            alpha_feather_strength=float(getattr(opts, "earring_restore_alpha_feather_strength", 0.35)),
            alpha_blur_kernel=int(getattr(opts, "earring_restore_alpha_blur_kernel", 3)),
            alpha_blur_sigma=float(getattr(opts, "earring_restore_alpha_blur_sigma", 1.4)),
        )

    def _build_ear_roi(self, source_parsing: torch.Tensor, source_size: tuple[int, int]) -> torch.Tensor:
        cfg = self.config
        source_ear = _label_mask(source_parsing, EAR_LABELS)
        source_earring = _label_mask(source_parsing, EARRING_LABELS)
        ear_base = torch.zeros_like(source_parsing, dtype=torch.float32)
        for value in (source_ear, source_earring):
            if value is not None:
                ear_base = torch.clamp(ear_base + value, 0, 1)

        lobe_shift = _scaled(cfg.ear_lobe_shift, source_size)
        lobe_probe = _translate_tensor(source_ear if source_ear is not None else ear_base, 0, lobe_shift)
        ear_base = torch.clamp(ear_base + 0.75 * lobe_probe, 0, 1)
        return _dilate(ear_base, _scaled(cfg.ear_roi_dilate, source_size, minimum=1)).clamp(0, 1)

    def _weak_earring_masks(
        self,
        source_01: torch.Tensor,
        source_parsing: torch.Tensor,
        source_ear_roi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        gray = _rgb_to_gray(source_01)
        local_small = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        local_large = F.avg_pool2d(gray, kernel_size=21, stride=1, padding=10)
        high = (gray - local_small).abs()
        contrast = (gray - local_large).abs()
        chroma = source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)

        weak_seed = (
            (high > cfg.weak_high_threshold).float()
            * torch.clamp(
                (chroma > cfg.weak_chroma_threshold).float()
                + (contrast > cfg.weak_contrast_threshold).float(),
                0,
                1,
            )
        )
        weak_support = (
            (high > cfg.support_high_threshold).float()
            * torch.clamp(
                (chroma > cfg.support_chroma_threshold).float()
                + (contrast > cfg.support_contrast_threshold).float(),
                0,
                1,
            )
        )

        source_hair = _label_mask(source_parsing, HAIR_LABELS)
        source_earring = _label_mask(source_parsing, EARRING_LABELS)
        source_face = _label_mask(source_parsing, FACE_SURFACE_LABELS)
        if source_hair is not None:
            hair_block = _dilate(source_hair, _scaled(cfg.source_hair_exclude_dilate, source_01.shape[-2:]))
            weak_seed = weak_seed * (1 - hair_block).clamp(0, 1)
            weak_support = weak_support * (1 - hair_block).clamp(0, 1)
        if source_face is not None:
            face_block = _dilate(source_face, 3)
            weak_seed = weak_seed * (1 - face_block).clamp(0, 1)
            weak_support = weak_support * (1 - source_face).clamp(0, 1)
        if source_earring is not None:
            weak_seed = torch.clamp(weak_seed + source_earring, 0, 1)
            weak_support = torch.clamp(weak_support + _dilate(source_earring, 3), 0, 1)

        return weak_seed * source_ear_roi, weak_support * source_ear_roi

    def _build_object_mask(
        self,
        source_01: torch.Tensor,
        source_parsing: torch.Tensor,
        source_ear_roi: torch.Tensor,
        manual_earring_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        work_size = source_01.shape[-2:]
        parser_earring = _label_mask(source_parsing, EARRING_LABELS)
        weak_seed, weak_support = self._weak_earring_masks(source_01, source_parsing, source_ear_roi)

        if manual_earring_mask is not None:
            manual_earring_mask = manual_earring_mask.to(device=source_01.device, dtype=source_01.dtype)
            manual_earring_mask = _resize_mask(manual_earring_mask, work_size, mode="nearest")
            parser_earring = torch.clamp((parser_earring if parser_earring is not None else 0) + manual_earring_mask, 0, 1)

        parser_earring = torch.zeros_like(source_ear_roi) if parser_earring is None else parser_earring * source_ear_roi
        parser_area = parser_earring.flatten(1).sum(dim=1).max().item()
        has_parser_earring = parser_area >= _scaled_area(cfg.component_min_area, work_size)

        if has_parser_earring:
            parser_near = _dilate(
                parser_earring,
                _scaled(cfg.parser_weak_support_dilate, work_size, minimum=1),
            ) * source_ear_roi
            seed = _dilate(parser_earring, _scaled(cfg.object_seed_dilate, work_size, minimum=1)) * source_ear_roi
            support = torch.clamp(
                parser_near
                * torch.clamp(weak_support + _dilate(parser_earring, _scaled(cfg.object_support_dilate, work_size, minimum=1)), 0, 1),
                0,
                1,
            )
        elif cfg.weak_fallback:
            seed = _dilate(weak_seed, _scaled(cfg.object_seed_dilate, work_size, minimum=1)) * source_ear_roi
            support = weak_support * source_ear_roi
        else:
            empty = torch.zeros_like(source_ear_roi)
            return empty, weak_seed, weak_support

        object_mask = seed
        for _ in range(max(0, cfg.object_grow_iters)):
            object_mask = _dilate(object_mask, 3) * support
        object_mask = torch.clamp(object_mask + seed, 0, 1) * source_ear_roi
        object_mask = _filter_components(
            object_mask,
            min_area=_scaled_area(cfg.component_min_area, work_size),
            max_area_ratio=cfg.component_max_area_ratio,
            keep_top=cfg.component_keep_top,
        )
        return object_mask, weak_seed, weak_support

    def _resolve_shift(
        self,
        source_parsing: torch.Tensor,
        target_parsing: torch.Tensor | None,
        source_object: torch.Tensor,
        work_size: tuple[int, int],
    ) -> tuple[int, int]:
        if target_parsing is None:
            return 0, 0
        cfg = self.config
        source_anchor = torch.clamp(_label_mask(source_parsing, EAR_LABELS) + source_object, 0, 1)
        target_anchor = _label_mask(target_parsing, EAR_LABELS)
        if source_anchor is None or target_anchor is None:
            return 0, 0
        source_center = _mask_center(source_anchor)
        target_center = _mask_center(target_anchor)
        if source_center is None or target_center is None:
            return 0, 0
        max_shift = _scaled(cfg.align_max_shift, work_size)
        dx = int(round(max(-max_shift, min(max_shift, target_center[0] - source_center[0]))))
        dy = int(round(max(-max_shift, min(max_shift, target_center[1] - source_center[1]))))
        return dx, dy

    def _build_visibility_gate(
        self,
        target_hair_mask: torch.Tensor | None,
        work_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cfg = self.config
        if target_hair_mask is None:
            return torch.ones((1, 1, *work_size), device=device, dtype=dtype)
        target_hair_mask = _resize_mask(target_hair_mask.to(device=device, dtype=dtype), work_size, mode="nearest")
        hair_block = _dilate(target_hair_mask, _scaled(cfg.target_hair_exclude_dilate, work_size, minimum=1))
        return (1 - hair_block).clamp(0, 1)

    def _build_safe_region(
        self,
        target_parsing: torch.Tensor | None,
        source_ear_roi: torch.Tensor,
        aligned_object: torch.Tensor,
        visibility_gate: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        work_size = source_ear_roi.shape[-2:]
        safe = _dilate(
            torch.clamp(source_ear_roi + aligned_object, 0, 1),
            _scaled(cfg.safe_region_dilate, work_size, minimum=1),
        )
        if target_parsing is not None:
            face_surface = _label_mask(target_parsing, FACE_SURFACE_LABELS)
            detail_block = _label_mask(target_parsing, DETAIL_BLOCK_LABELS)
            if face_surface is not None:
                contact = _dilate(aligned_object, _scaled(cfg.face_contact_dilate, work_size, minimum=1))
                safe = safe * (1 - face_surface).clamp(0, 1) + safe * face_surface * contact
            if detail_block is not None:
                safe = safe * (1 - detail_block).clamp(0, 1)
        return (safe * visibility_gate).clamp(0, 1)

    def _build_alpha(self, restore_mask: torch.Tensor, safe_region: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        cfg = self.config
        restore_out = _resize_mask(restore_mask, out_size, mode="nearest")
        safe_out = _resize_mask(safe_region, out_size, mode="nearest")

        core = restore_out
        core_dilate = _scaled(cfg.alpha_core_dilate, out_size)
        if core_dilate > 1:
            core = _dilate(core, core_dilate) * safe_out

        feather_support = _dilate(restore_out, _scaled(cfg.alpha_feather_dilate, out_size, minimum=1)) * safe_out
        blur_kernel = _scaled(cfg.alpha_blur_kernel, out_size, minimum=1)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blur_sigma = max(float(cfg.alpha_blur_sigma) * max(out_size) / 256.0, 1e-3)
        soft = _gaussian_blur(feather_support, kernel_size=blur_kernel, sigma=blur_sigma).clamp(0, 1)
        core_alpha = core * max(0.0, min(1.0, cfg.alpha_strength))
        feather_alpha = soft * max(0.0, min(1.0, cfg.alpha_feather_strength))
        alpha = torch.maximum(core_alpha, feather_alpha)
        return (alpha * feather_support).clamp(0, 1)

    @torch.inference_mode()
    def forward(
        self,
        source_image: torch.Tensor,
        pp_out: torch.Tensor,
        source_parsing: torch.Tensor | None = None,
        target_parsing: torch.Tensor | None = None,
        target_hair_mask: torch.Tensor | None = None,
        manual_earring_mask: torch.Tensor | None = None,
        enabled: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        use_restore = self.enabled if enabled is None else bool(enabled)
        pp_out = _ensure_bchw(pp_out).float()
        if not use_restore or source_parsing is None:
            return pp_out.clamp(0, 1), {}

        device = pp_out.device
        dtype = pp_out.dtype
        pp_out = _normalize_image(pp_out, device=device, dtype=dtype)
        out_size = pp_out.shape[-2:]

        source_image = _normalize_image(source_image, device=device, dtype=dtype)
        source_image = _resize_image(source_image, out_size)

        source_parsing = _resize_label_map(source_parsing.to(device=device), source_parsing.shape[-2:])
        work_size = source_parsing.shape[-2:]
        target_parsing = _resize_label_map(target_parsing.to(device=device), work_size) if target_parsing is not None else None
        source_work = _resize_image(source_image, work_size)

        source_ear_roi = self._build_ear_roi(source_parsing, work_size)
        object_mask, weak_seed, weak_support = self._build_object_mask(
            source_work,
            source_parsing,
            source_ear_roi,
            manual_earring_mask=manual_earring_mask,
        )
        if object_mask.flatten(1).amax(dim=1).sum().item() <= 0:
            debug = {
                "source_ear_roi": source_ear_roi,
                "object_mask": object_mask,
                "weak_seed": weak_seed,
                "weak_support": weak_support,
            }
            return pp_out, debug

        dx, dy = self._resolve_shift(source_parsing, target_parsing, object_mask, work_size)
        aligned_object = _translate_tensor(object_mask, dx, dy)
        aligned_ear_roi = _translate_tensor(source_ear_roi, dx, dy)
        visibility_gate = self._build_visibility_gate(target_hair_mask, work_size, device, dtype)
        safe_parsing = target_parsing if target_parsing is not None else source_parsing
        safe_region = self._build_safe_region(safe_parsing, aligned_ear_roi, aligned_object, visibility_gate)

        restore_mask = (aligned_object * visibility_gate * safe_region).clamp(0, 1)
        restore_mask = _filter_components(
            restore_mask,
            min_area=_scaled_area(self.config.component_min_area, work_size),
            max_area_ratio=self.config.component_max_area_ratio,
            keep_top=self.config.component_keep_top,
        )
        if restore_mask.flatten(1).amax(dim=1).sum().item() <= 0:
            debug = {
                "source_ear_roi": source_ear_roi,
                "object_mask": object_mask,
                "visibility_gate": visibility_gate,
                "safe_region": safe_region,
                "restore_mask": restore_mask,
            }
            return pp_out, debug

        alpha = self._build_alpha(restore_mask, safe_region, out_size)
        scale_x = out_size[1] / float(work_size[1])
        scale_y = out_size[0] / float(work_size[0])
        shifted_source = _translate_tensor(source_image, int(round(dx * scale_x)), int(round(dy * scale_y)))
        final = pp_out * (1 - alpha) + shifted_source * alpha

        debug = {
            "source": source_image,
            "pp_out": pp_out,
            "source_ear_roi": source_ear_roi,
            "object_mask": object_mask,
            "weak_seed": weak_seed,
            "weak_support": weak_support,
            "aligned_object_mask": aligned_object,
            "visibility_gate": visibility_gate,
            "safe_region": safe_region,
            "restore_mask": restore_mask,
            "alpha": alpha,
            "final": final.clamp(0, 1),
        }
        return final.clamp(0, 1), debug
