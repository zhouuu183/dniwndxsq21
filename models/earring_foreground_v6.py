"""Source-native earring instances for the V6 final compositor.

The module has one coordinate rule: source-native evidence may create a source
instance once; only that instance may be aligned into target coordinates.
Search masks, target-aligned labels and semantic regions are never write alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # pragma: no cover - runtime has OpenCV, tests exercise the contract without it.
    cv2 = None


RAW_LEFT_EAR = 7
RAW_RIGHT_EAR = 8
RAW_EARRING = 9
RAW_HAIR = 17
RAW_FACE_SURFACE_LABELS = (1, 10)
RAW_DETAIL_LABELS = (2, 3, 4, 5, 6, 11, 12, 13)


class EarringCoordinateSpace(str, Enum):
    SOURCE_NATIVE = "SOURCE_NATIVE"
    SOURCE_CANONICAL = "SOURCE_CANONICAL"
    TARGET_CANONICAL = "TARGET_CANONICAL"
    TARGET_OUTPUT = "TARGET_OUTPUT"


@dataclass(frozen=True)
class EarringNativeInstanceV6:
    alpha: torch.Tensor
    rgb: torch.Tensor
    hole_alpha: torch.Tensor
    left_alpha: torch.Tensor
    right_alpha: torch.Tensor
    left_hole_alpha: torch.Tensor
    right_hole_alpha: torch.Tensor
    space: EarringCoordinateSpace = EarringCoordinateSpace.SOURCE_NATIVE


def assert_source_seed_space(space: EarringCoordinateSpace | str) -> None:
    value = EarringCoordinateSpace(space)
    if value not in (EarringCoordinateSpace.SOURCE_NATIVE, EarringCoordinateSpace.SOURCE_CANONICAL):
        raise AssertionError(
            "A target-aligned earring mask must never be used as a source-native seed; "
            f"got {value.value}."
        )


def _image_01(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(0)
    return ((value + 1.0) * 0.5 if value.detach().amin() < -0.05 else value).clamp(0, 1)


def _mask(value: torch.Tensor | None, size: tuple[int, int], like: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros(like.size(0), 1, *size, device=like.device, dtype=like.dtype)
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    value = value[:, :1].to(device=like.device, dtype=like.dtype)
    if value.shape[-2:] != size:
        value = F.interpolate(value, size=size, mode="nearest")
    return value.clamp(0, 1)


def _parsing_mask(parsing: torch.Tensor | None, labels: tuple[int, ...], size: tuple[int, int], like: torch.Tensor) -> torch.Tensor:
    if parsing is None:
        return torch.zeros(like.size(0), 1, *size, device=like.device, dtype=like.dtype)
    if parsing.ndim == 3:
        parsing = parsing.unsqueeze(1)
    parsing = parsing[:, :1].to(device=like.device)
    if parsing.shape[-2:] != size:
        parsing = F.interpolate(parsing.float(), size=size, mode="nearest").long()
    else:
        parsing = parsing.long()
    result = torch.zeros_like(parsing, dtype=like.dtype)
    for label in labels:
        result = torch.maximum(result, (parsing == label).to(like.dtype))
    return result


def _shift_per_batch(value: torch.Tensor, shifts_y: torch.Tensor, shifts_x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    height, width = value.shape[-2:]
    for index in range(value.size(0)):
        dy = int(shifts_y[index].item())
        dx = int(shifts_x[index].item())
        src_y0, src_y1 = max(0, -dy), min(height, height - dy)
        src_x0, src_x1 = max(0, -dx), min(width, width - dx)
        dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
        dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
        if src_y1 > src_y0 and src_x1 > src_x0:
            result[index, :, dst_y0:dst_y1, dst_x0:dst_x1] = value[index, :, src_y0:src_y1, src_x0:src_x1]
    return result


def _centroid(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = mask.shape[-2:]
    y = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    x = torch.arange(width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
    area = mask.flatten(1).sum(dim=1)
    valid = area > 0.5
    cy = (mask * y).flatten(1).sum(dim=1) / area.clamp_min(1.0)
    cx = (mask * x).flatten(1).sum(dim=1) / area.clamp_min(1.0)
    return cy, cx, valid


def _lobe_anchor(mask: torch.Tensor) -> torch.Tensor:
    """Return the lower ear/lobe portion used as an accessory attachment point."""
    height = mask.shape[-2]
    y = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    area = mask.flatten(1).sum(dim=1, keepdim=True)
    y_min = torch.where(
        mask > 0.5,
        y.expand_as(mask),
        torch.full_like(mask, float(height)),
    ).flatten(1).amin(dim=1, keepdim=True)
    y_max = torch.where(
        mask.flatten(1).amax(dim=1, keepdim=True) > 0,
        (mask * y).flatten(1).amax(dim=1, keepdim=True),
        y_min,
    )
    threshold = y_min + 0.52 * (y_max - y_min).clamp_min(1.0)
    lower = mask * (y >= threshold.view(-1, 1, 1, 1)).to(mask.dtype)
    lower_area = lower.flatten(1).sum(dim=1, keepdim=True)
    # A partially visible lobe can contain only a few parser pixels.  In that
    # case the complete ear centroid is a more stable fallback than a noisy
    # one-pixel lower-edge centroid.
    use_lower = (lower_area >= torch.maximum(torch.ones_like(area), area * 0.05)).to(mask.dtype)
    return lower * use_lower.view(-1, 1, 1, 1) + mask * (1.0 - use_lower.view(-1, 1, 1, 1))


def _ear_bottom_attachment(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the visible lower-lobe attachment point for each sample.

    The full ear centroid is not a useful hanger location: it moves upward
    when only the upper ear is visible and it places a recovered pendant in
    the middle of the lobe.  A narrow bottom band follows the actual target
    lobe without changing any source-earring pixels.
    """
    mask = (mask > 0.5).to(mask.dtype)
    height = mask.shape[-2]
    y = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    y_min = torch.where(
        mask > 0,
        y.expand_as(mask),
        torch.full_like(mask, float(height)),
    ).flatten(1).amin(dim=1, keepdim=True)
    y_max = (mask * y).flatten(1).amax(dim=1, keepdim=True)
    band_height = torch.maximum(
        torch.full_like(y_max, 2.0),
        (y_max - y_min).clamp_min(1.0) * 0.14,
    )
    bottom = mask * (y >= (y_max - band_height).view(-1, 1, 1, 1)).to(mask.dtype)
    bottom_area = bottom.flatten(1).sum(dim=1, keepdim=True)
    use_bottom = (bottom_area > 0.5).to(mask.dtype)
    return _centroid(
        bottom * use_bottom.view(-1, 1, 1, 1)
        + mask * (1.0 - use_bottom.view(-1, 1, 1, 1))
    )


def _earring_top_attachment(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the top attachment band of a confirmed source earring."""
    mask = (mask > 0.5).to(mask.dtype)
    height = mask.shape[-2]
    y = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    y_min = torch.where(
        mask > 0,
        y.expand_as(mask),
        torch.full_like(mask, float(height)),
    ).flatten(1).amin(dim=1, keepdim=True)
    y_max = (mask * y).flatten(1).amax(dim=1, keepdim=True)
    band_height = torch.maximum(
        torch.full_like(y_min, 2.0),
        (y_max - y_min).clamp_min(1.0) * 0.10,
    )
    top = mask * (y <= (y_min + band_height).view(-1, 1, 1, 1)).to(mask.dtype)
    top_area = top.flatten(1).sum(dim=1, keepdim=True)
    use_top = (top_area > 0.5).to(mask.dtype)
    return _centroid(
        top * use_top.view(-1, 1, 1, 1)
        + mask * (1.0 - use_top.view(-1, 1, 1, 1))
    )


def canonical_target_visibility_v6(
    source_ear: torch.Tensor,
    source_earring_alpha: torch.Tensor,
    target_hair: torch.Tensor,
    *,
    covered_threshold: float = 0.85,
    probe_dilate: int = 3,
) -> dict[str, torch.Tensor]:
    """Decide target visibility in HairFast's shared canonical coordinates.

    HairFast changes hairstyle while retaining the source face geometry.  A
    final target ear parser is therefore unnecessary for positioning the
    source accessory and can be actively harmful after SATD changes the local
    ear/background contrast.  Probe the source lower lobe (or, if the parser
    missed it, the verified earring's top attachment) against the *target hair
    authority*.  The side closes only when that attachment is almost fully
    covered.  Visibility is object-level rather than pixel-level: clipping the
    upper half of an otherwise valid stud/pendant makes it look smaller and
    lower than the source.  A closed side writes nothing; an open side writes
    the complete verified jewellery object only, never source ear/background.
    """
    if source_ear.ndim == 3:
        source_ear = source_ear.unsqueeze(1)
    if source_earring_alpha.ndim == 3:
        source_earring_alpha = source_earring_alpha.unsqueeze(1)
    if target_hair.ndim == 3:
        target_hair = target_hair.unsqueeze(1)
    reference = source_earring_alpha[:, :1]
    size = tuple(reference.shape[-2:])
    ear = _mask(source_ear, size, reference)
    earring = _mask(source_earring_alpha, size, reference)
    hair = _mask(target_hair, size, reference)
    hair = (hair > 0.35).to(reference.dtype)

    height = size[0]
    y = torch.arange(
        height,
        device=reference.device,
        dtype=reference.dtype,
    ).view(1, 1, height, 1)

    # Lower source-ear band: the actual attachment neighbourhood, not the
    # whole ear shell or a broad ear ROI.
    ear_binary = (ear > 0.5).to(reference.dtype)
    ear_y_min = torch.where(
        ear_binary > 0,
        y.expand_as(ear_binary),
        torch.full_like(ear_binary, float(height)),
    ).flatten(1).amin(dim=1, keepdim=True)
    ear_y_max = (ear_binary * y).flatten(1).amax(dim=1, keepdim=True)
    lower_start = ear_y_min + 0.62 * (ear_y_max - ear_y_min).clamp_min(1.0)
    lobe_probe = ear_binary * (
        y >= lower_start.view(-1, 1, 1, 1)
    ).to(reference.dtype)

    # Parser-missed ear fallback: only the verified object's narrow top band.
    # This is a visibility probe and never becomes output RGB alpha.
    object_binary = (earring > 0.01).to(reference.dtype)
    object_y_min = torch.where(
        object_binary > 0,
        y.expand_as(object_binary),
        torch.full_like(object_binary, float(height)),
    ).flatten(1).amin(dim=1, keepdim=True)
    object_y_max = (object_binary * y).flatten(1).amax(dim=1, keepdim=True)
    top_height = torch.maximum(
        torch.full_like(object_y_min, 2.0),
        0.10 * (object_y_max - object_y_min).clamp_min(1.0),
    )
    object_top = object_binary * (
        y <= (object_y_min + top_height).view(-1, 1, 1, 1)
    ).to(reference.dtype)

    lobe_present = (
        lobe_probe.flatten(1).sum(dim=1, keepdim=True) >= 2.0
    ).view(-1, 1, 1, 1)
    probe = torch.where(lobe_present, lobe_probe, object_top)
    radius = max(0, int(probe_dilate))
    if radius > 0:
        probe = F.max_pool2d(
            probe,
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )
    probe = probe.clamp(0, 1)
    probe_area = probe.flatten(1).sum(dim=1, keepdim=True)
    hair_cover_ratio = (
        (probe * hair).flatten(1).sum(dim=1, keepdim=True)
        / probe_area.clamp_min(1.0)
    )
    threshold = max(0.0, min(1.0, float(covered_threshold)))
    side_open = (
        (probe_area >= 1.0)
        & (hair_cover_ratio < threshold)
    ).view(-1, 1, 1, 1)
    visible_alpha = (
        earring * side_open.to(reference.dtype)
    ).clamp(0, 1)
    return {
        "probe": probe,
        "hair_cover_ratio": hair_cover_ratio.view(-1, 1, 1, 1),
        "side_open": side_open.to(reference.dtype),
        "visible_alpha": visible_alpha,
        "target_hair": hair,
    }


def _side_open(mask: torch.Tensor, min_visible_area: float) -> torch.Tensor:
    """Open a side only from final-resolution visible ear/lobe pixels.

    A previous low-resolution ``side_open`` flag could override a fully hair-
    covered ear.  Final compositing must instead be gated by the target image's
    own semantic ear/lobe region after target-hair exclusion.
    """
    opened = (mask.flatten(1).sum(dim=1) >= max(1.0, float(min_visible_area))).to(mask.dtype)
    return opened.view(-1, 1, 1, 1)


def _empty_output(source: torch.Tensor) -> dict[str, torch.Tensor]:
    zeros = torch.zeros_like(source[:, :1])
    state = torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype)
    return {
        "source_native_earring_alpha": zeros,
        "source_native_earring_rgb": source,
        "source_native_hole_alpha": zeros,
        "source_native_left_alpha": zeros.clone(),
        "source_native_right_alpha": zeros.clone(),
        "source_native_left_hole_alpha": zeros.clone(),
        "source_native_right_hole_alpha": zeros.clone(),
        "source_native_presence_state": state,
        "source_native_presence_score": state.clone(),
        "parser_earring_seed": zeros.clone(),
        "source_native_recall_hint": zeros.clone(),
        "lobe_anchor": zeros.clone(),
        "localization_core": zeros.clone(),
        "localization_adaptive": zeros.clone(),
        "probable_visual_evidence": zeros.clone(),
        "grabcut_raw": zeros.clone(),
        "component_labels": zeros.clone(),
        "component_scores": zeros.clone(),
        "selected_components": zeros.clone(),
        "accepted_component_ids": zeros.clone(),
        "rejected_component_ids": zeros.clone(),
        "max_graph_depth_used": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
        "cumulative_graph_cost": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
        "component_count": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
        "root_component_id": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
        "accepted_component_count": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
        "rejected_component_count": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
        # 0=none, 1=remote, 2=lateral, 3=background, 4=hair, 5=skin,
        # 6=gap, 7=extent.
        "reject_reason": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
    }


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    if cv2 is None or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    inverse = (~mask).astype(np.uint8)
    flood = inverse.copy()
    flood_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 0)
    return (inverse.astype(bool) & ~flood.astype(bool))


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(ax - (bx + bw), bx - (ax + aw), 0)
    dy = max(ay - (by + bh), by - (ay + ah), 0)
    return math.hypot(dx, dy)


def _component_angle(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    centered = points.astype(np.float32) - points.astype(np.float32).mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    vector = vectors[:, int(np.argmax(values))]
    return float(math.atan2(float(vector[0]), float(vector[1])))


def _verified_hair_continuation_v6(
    root: dict[str, object],
    current: dict[str, object],
    candidate: dict[str, object],
    *,
    gap: float,
    scale: float,
    enabled: bool,
) -> bool:
    """Accept only jewellery-like continuation pixels mislabeled as source hair.

    This is deliberately narrower than generic long-object continuation.  A
    verified lobe root must already exist, and the new component must be a
    close, downward, axis-aligned, high-material/high-edge segment.  The rule
    never promotes a hair-labelled component into a root by itself.
    """
    if not enabled or float(candidate.get("hair_overlap", 0.0)) < 0.30:
        return False
    root_authority = (
        float(root.get("parser_overlap", 0.0)) > 0.02
        or float(root.get("seed_overlap", 0.0)) > 0.02
        or (
            float(root.get("recall_overlap", 0.0)) > 0.05
            and float(root.get("material_score", 0.0)) >= 0.55
            and float(root.get("edge_energy", 0.0)) >= 0.22
        )
    )
    if not root_authority:
        return False
    if gap > max(2.0, 6.0 * scale):
        return False
    if float(candidate.get("centroid_y", 0.0)) < float(current.get("centroid_y", 0.0)) - 2.0 * scale:
        return False
    if abs(float(candidate.get("centroid_x", 0.0)) - float(current.get("centroid_x", 0.0))) > 10.0 * scale:
        return False
    material = float(candidate.get("material_score", 0.0))
    edge = float(candidate.get("edge_energy", 0.0))
    elongation = float(candidate.get("elongation", 0.0))
    fill_ratio = float(candidate.get("fill_ratio", 1.0))
    area = float(candidate.get("area", float("inf")))
    # Long earrings are often segmented into a thin hook plus one or more
    # compact pearls/stones.  Permit a small, very strong compact pendant as
    # well as a thin wire; the root, direction, gap and x-axis checks above
    # still prevent an unrelated bright hair/background speck from becoming
    # a second object.
    compact_jewel = (
        area <= max(4.0, 140.0 * scale * scale)
        and material >= 0.82
        and edge >= 0.30
    )
    return (
        material >= 0.70
        and edge >= 0.26
        and (elongation >= 1.65 or fill_ratio <= 0.42 or compact_jewel)
    )


def retain_single_earring_group_v6(
    mask: torch.Tensor,
    lobe_anchor: torch.Tensor,
    *,
    max_gap: int = 48,
    max_downward_extent: int | None = None,
) -> torch.Tensor:
    """Keep one lobe-connected source accessory group per side.

    A native candidate can contain several disconnected visual components.
    Keep the component attached to the lobe and only nearby, vertically
    aligned continuation pieces. This preserves segmented long pendants while
    rejecting a second same-side accessory or a detached background strand.
    The accepted chain always has a finite downward extent; it is not an
    unbounded vertical rail through the source background.
    """
    if cv2 is None:
        return torch.zeros_like(mask)
    value = _mask(mask, tuple(mask.shape[-2:]), mask).detach()
    anchor = _mask(lobe_anchor, tuple(value.shape[-2:]), value).detach()
    output = torch.zeros_like(value)
    scale = max(value.shape[-2:]) / 256.0
    # These limits are deliberately small in 256px coordinates.  They are
    # connection limits, not a rectangular search rail.
    gap_limit = max(1.0, float(max_gap))
    for batch_index, array in enumerate(value[:, 0].cpu().numpy()):
        binary = array > 0.01
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), 8
        )
        if count <= 1:
            continue
        anchor_binary = anchor[batch_index, 0].cpu().numpy() > 0.01
        anchor_points = np.argwhere(anchor_binary)
        candidates = []
        for component_id in range(1, count):
            component = labels == component_id
            points = np.argwhere(component)
            if not points.size:
                continue
            if anchor_points.size:
                distance = float(np.min(np.linalg.norm(
                    points[:, None, :].astype(np.float32)
                    - anchor_points[None, :, :].astype(np.float32), axis=2)))
            else:
                distance = float("inf")
            overlap = float((component & anchor_binary).sum())
            area = float(stats[component_id, cv2.CC_STAT_AREA])
            candidates.append((overlap, distance, area, component_id))
        if not candidates:
            continue
        touching = [item for item in candidates if item[0] > 0.0]
        root_info = max(touching, key=lambda item: (item[0], -item[1], item[2])) if touching else min(candidates, key=lambda item: (item[1], -item[2]))
        if not touching and root_info[1] > gap_limit:
            continue
        root_id = root_info[3]
        root = labels == root_id
        root_points = np.argwhere(root)
        root_x0 = float(stats[root_id, cv2.CC_STAT_LEFT])
        root_x1 = root_x0 + float(stats[root_id, cv2.CC_STAT_WIDTH])
        root_y0 = float(stats[root_id, cv2.CC_STAT_TOP])
        root_y1 = root_y0 + float(stats[root_id, cv2.CC_STAT_HEIGHT])
        root_height = max(1.0, root_y1 - root_y0)
        keep = root.copy()
        keep_x0, keep_x1, keep_y0, keep_y1 = root_x0, root_x1, root_y0, root_y1
        # A connected component may contain a background strand.  Restrict a
        # very wide root to the lobe-centred object corridor; this is only a
        # decontamination step and cannot add pixels outside the input mask.
        if anchor_points.size:
            anchor_x = float(anchor_points[:, 1].mean())
            corridor = max(4.0 * scale, min(16.0 * scale, 0.65 * max(1.0, root_x1 - root_x0)))
            root &= np.abs(np.indices(root.shape)[1] - anchor_x) <= corridor
            keep = root
            ys, xs = np.where(keep)
            if not len(ys):
                continue
            keep_x0, keep_x1 = float(xs.min()), float(xs.max() + 1)
            keep_y0, keep_y1 = float(ys.min()), float(ys.max() + 1)
        # Extend only downward, in short locally connected steps.  The extent
        # grows from the accepted object, so a remote lateral strand cannot be
        # admitted merely because it lies in a long fixed rail.
        # 150px at 256px resolution is large enough for an ordinary long
        # pendant while still preventing a lobe-rooted component from walking
        # down an arbitrary background chain.  Callers can request a smaller
        # finite limit, but never an unbounded one.
        dynamic_limit = (
            max(16.0 * scale, min(150.0 * scale, 0.60 * value.shape[-2]))
            if max_downward_extent is None
            else max(1.0, float(max_downward_extent))
        )
        accepted_ids = {root_id}
        changed = True
        while changed:
            changed = False
            best = None
            best_distance = float("inf")
            for _, _, _, component_id in candidates:
                if component_id in accepted_ids:
                    continue
                component = labels == component_id
                x0 = float(stats[component_id, cv2.CC_STAT_LEFT])
                x1 = x0 + float(stats[component_id, cv2.CC_STAT_WIDTH])
                y0 = float(stats[component_id, cv2.CC_STAT_TOP])
                y1 = y0 + float(stats[component_id, cv2.CC_STAT_HEIGHT])
                gap_x = max(keep_x0 - x1, x0 - keep_x1, 0.0)
                gap_y = max(y0 - keep_y1, keep_y0 - y1, 0.0)
                cx = float(centroids[component_id][0])
                cy = float(centroids[component_id][1])
                # Pendants go down from the accepted chain.  A component which
                # begins beside/above the root is treated as a second object.
                if cy < keep_y1 - max(1.0, 0.25 * gap_limit):
                    continue
                if gap_x > 0.75 * gap_limit or gap_y > gap_limit:
                    continue
                if abs(cx - 0.5 * (keep_x0 + keep_x1)) > max(2.0 * scale, gap_limit):
                    continue
                if dynamic_limit is not None and y1 > root_y1 + dynamic_limit:
                    continue
                distance = gap_x + gap_y
                if distance < best_distance:
                    best_distance, best = distance, component_id
            if best is None:
                break
            component = labels == best
            keep |= component
            x0 = float(stats[best, cv2.CC_STAT_LEFT])
            x1 = x0 + float(stats[best, cv2.CC_STAT_WIDTH])
            y0 = float(stats[best, cv2.CC_STAT_TOP])
            y1 = y0 + float(stats[best, cv2.CC_STAT_HEIGHT])
            keep_x0, keep_x1 = min(keep_x0, x0), max(keep_x1, x1)
            keep_y0, keep_y1 = min(keep_y0, y0), max(keep_y1, y1)
            accepted_ids.add(best)
            changed = True
        output[batch_index, 0] = torch.from_numpy(array * keep.astype(np.float32)).to(
            output.device, output.dtype
        )
    return output


def enforce_exclusive_earring_sides_v6(
    left_alpha: torch.Tensor,
    right_alpha: torch.Tensor,
    source_left_ear: torch.Tensor,
    source_right_ear: torch.Tensor,
    *,
    proximity_radius: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ensure one physical source object is aligned to at most one target side."""
    size = tuple(left_alpha.shape[-2:])
    like = left_alpha
    left = _mask(left_alpha, size, like)
    right = _mask(right_alpha, size, like)
    left_ear = _mask(source_left_ear, size, like)
    right_ear = _mask(source_right_ear, size, like)
    small_size = (min(256, size[0]), min(256, size[1]))
    left_small = F.adaptive_max_pool2d((left > 0.01).to(like.dtype), small_size)
    right_small = F.adaptive_max_pool2d((right > 0.01).to(like.dtype), small_size)
    left_ear_small = F.adaptive_max_pool2d((left_ear > 0.01).to(like.dtype), small_size)
    right_ear_small = F.adaptive_max_pool2d((right_ear > 0.01).to(like.dtype), small_size)
    radius = max(6, int(round(40.0 * max(small_size) / 256.0)))
    if proximity_radius is not None:
        radius = max(1, int(round(float(proximity_radius) * max(small_size) / 256.0)))
    near_right = F.max_pool2d(right_small, 2 * radius + 1, stride=1, padding=radius)
    duplicate = (
        (left_small.flatten(1).sum(dim=1, keepdim=True) > 0.5)
        & (right_small.flatten(1).sum(dim=1, keepdim=True) > 0.5)
        & ((left_small * near_right).flatten(1).sum(dim=1, keepdim=True) > 0.5)
    ).view(-1, 1, 1, 1)
    attach_radius = max(3, int(round(14.0 * max(small_size) / 256.0)))
    left_band = F.max_pool2d(left_ear_small, 2 * attach_radius + 1, stride=1, padding=attach_radius)
    right_band = F.max_pool2d(right_ear_small, 2 * attach_radius + 1, stride=1, padding=attach_radius)
    left_own_contact = (left_small * left_band).flatten(1).sum(dim=1, keepdim=True)
    left_cross_contact = (left_small * right_band).flatten(1).sum(dim=1, keepdim=True)
    right_own_contact = (right_small * right_band).flatten(1).sum(dim=1, keepdim=True)
    right_cross_contact = (right_small * left_band).flatten(1).sum(dim=1, keepdim=True)
    left_score = 8.0 * left_own_contact - 3.0 * left_cross_contact + 0.001 * left_small.flatten(1).sum(dim=1, keepdim=True)
    right_score = 8.0 * right_own_contact - 3.0 * right_cross_contact + 0.001 * right_small.flatten(1).sum(dim=1, keepdim=True)
    keep_left = (left_score >= right_score).view(-1, 1, 1, 1)
    remove_left = duplicate & ~keep_left
    remove_right = duplicate & keep_left

    # Near-overlap is not the only split failure.  A long source object can
    # emit a lobe-attached root on one side and a detached fragment on the
    # other, placing that fragment on a second target ear after alignment.
    # A genuine bilateral pair has own-lobe contact on *both* sides.  Remove
    # only an unanchored opposite-side fragment when the other side has a
    # real source-ear attachment; this preserves true left/right earrings and
    # does not alter target-side visibility.
    association_radius = max(6, int(round(48.0 * max(small_size) / 256.0)))
    left_association = F.max_pool2d(
        left_ear_small, 2 * association_radius + 1, stride=1, padding=association_radius
    )
    right_association = F.max_pool2d(
        right_ear_small, 2 * association_radius + 1, stride=1, padding=association_radius
    )
    left_associated_area = (left_small * left_association).flatten(1).sum(dim=1, keepdim=True)
    right_associated_area = (right_small * right_association).flatten(1).sum(dim=1, keepdim=True)
    left_present = left_small.flatten(1).sum(dim=1, keepdim=True) > 0.5
    right_present = right_small.flatten(1).sum(dim=1, keepdim=True) > 0.5
    left_rooted = left_associated_area > 0.5
    right_rooted = right_associated_area > 0.5
    remove_left_unanchored = left_present & right_present & ~left_rooted & right_rooted
    remove_right_unanchored = left_present & right_present & left_rooted & ~right_rooted
    remove_left = remove_left | remove_left_unanchored.view(-1, 1, 1, 1)
    remove_right = remove_right | remove_right_unanchored.view(-1, 1, 1, 1)
    dtype = like.dtype
    return left * (1.0 - remove_left.to(dtype)), right * (1.0 - remove_right.to(dtype)), remove_left.to(dtype), remove_right.to(dtype)


def extract_source_native_earring_v6(
    source_rgb: torch.Tensor,
    source_parsing: torch.Tensor | None,
    *,
    source_native_seed: torch.Tensor | None = None,
    seed_space: EarringCoordinateSpace | str = EarringCoordinateSpace.SOURCE_NATIVE,
    source_native_recall_hint: torch.Tensor | None = None,
    recall_hint_space: EarringCoordinateSpace | str = EarringCoordinateSpace.SOURCE_CANONICAL,
    max_graph_depth: int = 3,
    max_cumulative_cost: float = 1.55,
    allow_long_continuation: bool = False,
    allow_semantic_hair_continuation: bool = False,
    native_matte_radius_px: int = 1,
    native_boundary_alpha_floor: float = 0.85,
) -> dict[str, torch.Tensor]:
    """Extract one source-native foreground instance per visible source side.

    Generic visual evidence is never a write mask.  It creates/scored connected
    components; final alpha contains only components accepted from a root with
    bounded cumulative appearance-compatible graph cost.
    """
    assert_source_seed_space(seed_space)
    assert_source_seed_space(recall_hint_space)
    source = _image_01(source_rgb)
    if source_parsing is None or cv2 is None:
        return _empty_output(source)

    batch, _, height, width = source.shape
    device, dtype = source.device, source.dtype
    size = (height, width)
    parsing = source_parsing
    if parsing.ndim == 3:
        parsing = parsing.unsqueeze(1)
    if parsing.shape[-2:] != size:
        parsing = F.interpolate(parsing.float(), size=size, mode="nearest").long()
    else:
        parsing = parsing.long()
    parsing_np = parsing.detach().cpu().numpy()[:, 0]
    seed = _mask(source_native_seed, size, source)
    seed_np_all = seed.detach().cpu().numpy()[:, 0] > 0.5
    recall_hint = _mask(source_native_recall_hint, size, source)
    recall_hint_np_all = recall_hint.detach().cpu().numpy()[:, 0] > 0.5
    source_np = np.clip(source.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0, 0, 255).astype(np.uint8)

    output = _empty_output(source)
    foreground_rgb = source.detach().cpu().permute(0, 2, 3, 1).numpy().copy()
    result_arrays = {name: np.zeros((batch, height, width), dtype=np.float32) for name in (
        "source_native_earring_alpha", "source_native_hole_alpha", "source_native_left_alpha",
        "source_native_right_alpha", "source_native_left_hole_alpha", "source_native_right_hole_alpha",
        "parser_earring_seed", "source_native_recall_hint", "lobe_anchor", "localization_core",
        "localization_adaptive", "probable_visual_evidence", "grabcut_raw", "component_labels",
        "component_scores", "selected_components",
    )}
    state = np.zeros((batch, 2), dtype=np.float32)
    confidence = np.zeros((batch, 2), dtype=np.float32)
    max_depth_out = np.zeros((batch, 2), dtype=np.float32)
    cumulative_cost_out = np.zeros((batch, 2), dtype=np.float32)
    component_count_out = np.zeros((batch, 2), dtype=np.float32)
    root_component_id_out = np.zeros((batch, 2), dtype=np.float32)
    accepted_component_count_out = np.zeros((batch, 2), dtype=np.float32)
    rejected_component_count_out = np.zeros((batch, 2), dtype=np.float32)
    reject_reason_out = np.zeros((batch, 2), dtype=np.float32)
    scale = max(height, width) / 256.0

    def radius(value: float, minimum: int = 1) -> int:
        return max(minimum, int(round(value * scale)))

    for batch_index in range(batch):
        image = source_np[batch_index]
        labels = parsing_np[batch_index]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        blur = cv2.GaussianBlur(image, (2 * radius(4) + 1, 2 * radius(4) + 1), 0)
        residual = np.abs(image.astype(np.int16) - blur.astype(np.int16)).mean(axis=2).astype(np.float32)
        chroma = (image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)).astype(np.float32)
        edges = cv2.Canny(gray, 45, 120) > 0
        side_ears = ((labels == RAW_LEFT_EAR), (labels == RAW_RIGHT_EAR))
        parser_earring = labels == RAW_EARRING
        semantic_skin = np.isin(labels, RAW_FACE_SURFACE_LABELS + RAW_DETAIL_LABELS + (RAW_LEFT_EAR, RAW_RIGHT_EAR))
        semantic_hair = labels == RAW_HAIR
        semantic_background = labels == 0

        for side_index, ear in enumerate(side_ears):
            if not ear.any():
                continue
            ys, xs = np.where(ear)
            cutoff = np.quantile(ys, 0.58)
            anchor = ear & (np.indices((height, width))[0] >= cutoff)
            anchor = cv2.dilate(anchor.astype(np.uint8), np.ones((2 * radius(2) + 1, 2 * radius(2) + 1), np.uint8), 1).astype(bool)
            anchor_points = np.argwhere(anchor)
            anchor_y, anchor_x = anchor_points.mean(axis=0)
            core = cv2.dilate(ear.astype(np.uint8), np.ones((2 * radius(18) + 1, 2 * radius(18) + 1), np.uint8), 1).astype(bool)
            y0 = max(0, int(round(anchor_y - radius(18))))
            y1 = min(height, int(round(anchor_y + radius(68))))
            x0 = max(0, int(round(anchor_x - radius(44))))
            x1 = min(width, int(round(anchor_x + radius(44))))
            core[y0:y1, x0:x1] = True
            # This is only a bounded component-inspection envelope.  Unlike the
            # old fixed rail, it does not itself become localization or alpha.
            ay0 = max(0, int(round(anchor_y - radius(18))))
            ay1 = min(height, int(round(anchor_y + radius(210))))
            ax0 = max(0, int(round(anchor_x - radius(82))))
            ax1 = min(width, int(round(anchor_x + radius(82))))
            envelope = np.zeros((height, width), dtype=bool)
            envelope[ay0:ay1, ax0:ax1] = True
            parser_side = parser_earring & envelope
            seed_side = seed_np_all[batch_index] & envelope
            recall_hint_side = recall_hint_np_all[batch_index] & envelope
            local_values = residual[envelope]
            chroma_values = chroma[envelope]
            residual_limit = max(6.0, float(np.percentile(local_values, 76))) if local_values.size else float("inf")
            chroma_limit = max(12.0, float(np.percentile(chroma_values, 82))) if chroma_values.size else float("inf")
            probable = envelope & cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool) & (
                (residual >= residual_limit) | (chroma >= chroma_limit)
            )
            # The recall hint only steers candidate discovery.  It is never
            # made definite foreground and is never ORed into final alpha.
            hint_neighborhood = cv2.dilate(
                recall_hint_side.astype(np.uint8),
                np.ones((2 * radius(2) + 1, 2 * radius(2) + 1), np.uint8),
                1,
            ).astype(bool) & envelope

            # Parser label 9 and a native seed locate the accessory but are
            # not its final contour.  Mark only an eroded core definite and
            # leave its boundary probable so GrabCut can follow the actual
            # pearl/metal/stone edge in the source image.  Treating the whole
            # coarse label as definite foreground was why small earrings were
            # restored as flat white or coloured blobs.
            trusted = parser_side | seed_side
            trusted_core = cv2.erode(
                trusted.astype(np.uint8),
                np.ones((2 * radius(1) + 1, 2 * radius(1) + 1), np.uint8),
                1,
            ).astype(bool)
            if not trusted_core.any() and trusted.any():
                trusted_core = trusted
            trusted_support = cv2.dilate(
                trusted.astype(np.uint8),
                np.ones((2 * radius(2) + 1, 2 * radius(2) + 1), np.uint8),
                1,
            ).astype(bool) & envelope
            gc = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
            gc[envelope] = cv2.GC_PR_BGD
            gc[probable | hint_neighborhood | trusted_support] = cv2.GC_PR_FGD
            gc[trusted_core] = cv2.GC_FGD
            if (parser_side | seed_side | recall_hint_side).any():
                try:
                    cv2.grabCut(image, gc, None, None, None, 3, cv2.GC_INIT_WITH_MASK)
                    grabcut = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
                except cv2.error:
                    grabcut = parser_side | seed_side | (probable & hint_neighborhood)
            else:
                # Unseeded evidence remains an UNCERTAIN component candidate,
                # never an automatically accepted foreground object.
                grabcut = np.zeros_like(envelope)
            # All final pixels must survive native image segmentation.  The
            # parser/seed contour is an object prior, however, not merely an
            # eroded centre: limiting it to ``trusted_core`` turns a solid stud
            # or the body of a long pendant into an edge fragment, which then
            # lets target hair show through the intended jewellery.  GrabCut
            # still has to accept every pixel, and dilated search hints remain
            # discovery-only, so this does not authorize the surrounding source
            # ear or background as foreground.
            candidate = grabcut & (probable | trusted) & envelope
            if not candidate.any():
                # Do not fall back to the coarse parser/seed footprint here.
                # If GrabCut cannot confirm the core, this side is uncertain;
                # returning a broad label is worse than returning no alpha.
                candidate = grabcut & probable & envelope

            # A semantic label-9 component is already an accessory authority.
            # GrabCut can nevertheless discard its low-contrast lower pendant
            # after locking onto the bright root.  In V6, retain the parser
            # component as an additional *candidate* when it is source-lobe
            # associated; all later halo, hair, skin and hole checks still
            # apply before it becomes RGB alpha.
            # Parser label 9 is a locator, not a blanket RGB mask.  Keep only
            # the single label-9 component nearest the lobe; detached label-9
            # fragments (often cheek contours or source wisps) are discarded.
            parser_component = np.zeros_like(envelope, dtype=bool)
            parser_labels_count, parser_labels, parser_stats, _ = cv2.connectedComponentsWithStats(
                parser_side.astype(np.uint8), 8
            )
            parser_candidates = []
            for parser_id in range(1, parser_labels_count):
                parser_part = parser_labels == parser_id
                parser_points = np.argwhere(parser_part)
                if not parser_points.size:
                    continue
                parser_distance = float(np.min(np.linalg.norm(
                    parser_points[:, None, :].astype(np.float32)
                    - anchor_points[None, :, :].astype(np.float32), axis=2
                ))) if anchor_points.size else float("inf")
                parser_candidates.append((parser_distance, -float(parser_stats[parser_id, cv2.CC_STAT_AREA]), parser_id))
            if parser_candidates:
                parser_id = min(parser_candidates)[2]
                parser_component = parser_labels == parser_id
            # Label-9 is a locator/seed, never a blanket RGB alpha.  In
            # particular, do not OR the complete parser component back into
            # the GrabCut result: parser label-9 can contain a cheek contour,
            # source wisps, or a backdrop fringe.  A low-contrast long pendant
            # is recovered later by the independently verified structured
            # source-instance path, which still applies lobe, material and
            # background checks before becoming write alpha.
            parser_candidate_area = int(parser_component.sum())
            candidate_area = int(candidate.sum())

            count, component_labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
            components: list[dict[str, object]] = []
            for component_id in range(1, count):
                component = component_labels == component_id
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                if area <= 0 or area > int((0.10 if allow_long_continuation else 0.055) * height * width):
                    continue
                points = np.argwhere(component)
                centroid_y, centroid_x = points.mean(axis=0)
                bbox = (
                    int(stats[component_id, cv2.CC_STAT_LEFT]),
                    int(stats[component_id, cv2.CC_STAT_TOP]),
                    int(stats[component_id, cv2.CC_STAT_WIDTH]),
                    int(stats[component_id, cv2.CC_STAT_HEIGHT]),
                )
                edge_energy = float(edges[component].mean())
                material_score = float(((residual >= residual_limit) | (chroma >= chroma_limit))[component].mean())
                parser_overlap = float(parser_side[component].mean())
                seed_overlap = float(seed_side[component].mean())
                recall_overlap = float(recall_hint_side[component].mean())
                skin_overlap = float(semantic_skin[component].mean())
                hair_overlap = float(semantic_hair[component].mean())
                background_overlap = float(semantic_background[component].mean())
                lobe_distance = float(math.hypot(centroid_y - anchor_y, centroid_x - anchor_x))
                mean_lab = lab[component].mean(axis=0).astype(np.float32)
                axis = _component_angle(points)
                max_dim = max(bbox[2], bbox[3])
                min_dim = max(1, min(bbox[2], bbox[3]))
                elongation = float(max_dim / min_dim)
                fill_ratio = float(area / max(1, bbox[2] * bbox[3]))
                lobe_score = max(0.0, 1.0 - lobe_distance / max(1.0, radius(80)))
                size_penalty = max(0.0, area / max(1.0, 0.025 * height * width) - 1.0)
                # A long pendant may be labelled as source background because
                # it hangs outside the ear shell.  Keep it eligible only when
                # its own native material/edge evidence is strong; otherwise
                # background texture is penalised before it can join the graph.
                background_penalty = 1.25 * background_overlap
                if background_overlap >= 0.45:
                    background_penalty += 0.90 * max(0.0, 0.58 - material_score)
                    background_penalty += 0.90 * max(0.0, 0.22 - edge_energy)
                semantic_penalty = 0.10 * skin_overlap + 0.16 * hair_overlap + background_penalty
                score = (
                    2.1 * seed_overlap
                    + 2.0 * parser_overlap
                    + 1.05 * recall_overlap
                    + 0.65 * material_score
                    + 0.55 * edge_energy
                    + 0.75 * lobe_score
                    - semantic_penalty
                    - 0.65 * size_penalty
                )
                compact_near_lobe = (
                    area <= max(4, radius(22) * radius(22))
                    and lobe_distance <= radius(15)
                    and edge_energy >= 0.12
                    and material_score >= 0.10
                )
                # A continuation outside the parser's earring label can be a
                # real long metal wire, but it must look like one.  This
                # rejects broad, background-labelled regions even when their
                # texture happened to be connected to a genuine hoop.
                background_only = (
                    background_overlap >= 0.55
                    and parser_overlap < 0.02
                    and seed_overlap < 0.02
                )
                background_continuation_ok = (
                    # Parser-missed metal may be labelled background, but a
                    # source-background strand beside the cheek is much more
                    # common.  Require a clearly object-like, elongated edge
                    # before a background component may continue an earring.
                    material_score >= 0.68
                    and edge_energy >= 0.28
                    and (elongation >= 1.65 or fill_ratio <= 0.42)
                )
                if background_only and not background_continuation_ok:
                    continue
                # A long pendant may extend down from the lobe, but a
                # background-labelled continuation above the lobe is an ear-
                # side hair/background filament, never jewellery.  This keeps
                # the locator from following the bright strings beside a hoop
                # into the transferred target hair.
                if background_only and centroid_y < anchor_y - radius(1):
                    continue
                components.append({
                    "id": component_id,
                    "mask": component,
                    "bbox": bbox,
                    "area": area,
                    "centroid_y": centroid_y,
                    "centroid_x": centroid_x,
                    "mean_lab": mean_lab,
                    "axis": axis,
                    "elongation": elongation,
                    "fill_ratio": fill_ratio,
                    "score": score,
                    "seed_overlap": seed_overlap,
                    "parser_overlap": parser_overlap,
                    "recall_overlap": recall_overlap,
                    "material_score": material_score,
                    "edge_energy": edge_energy,
                    # Keep semantic overlap scores on every component.  The
                    # V6 long-pendant continuation pass uses these values to
                    # reject hair/skin/background components; omitting them
                    # made that pass fail with a KeyError during generation.
                    "skin_overlap": skin_overlap,
                    "hair_overlap": hair_overlap,
                    "background_overlap": background_overlap,
                    "lobe_distance": lobe_distance,
                    "semantic_penalty": semantic_penalty,
                    "compact_near_lobe": compact_near_lobe,
                })

            component_count_out[batch_index, side_index] = len(components)
            roots = [
                item for item in components
                if (
                    item["seed_overlap"] > 0
                    or item["parser_overlap"] > 0
                    or item["compact_near_lobe"]
                    or (
                        item["recall_overlap"] > 0.05
                        and item["material_score"] >= 0.15
                        and item["edge_energy"] >= 0.10
                    )
                )
                and item["score"] >= 0.85
            ]
            # One source ear has at most one physical accessory.  Starting
            # the graph from every parser/highlight component let unrelated
            # background strands become a second object on the same side.
            # Keep the strongest lobe-associated root and attach compatible
            # lower pieces through the bounded graph below.
            if len(roots) > 1:
                roots.sort(
                    key=lambda item: (
                        float(item["score"]),
                        float(item["parser_overlap"]),
                        float(item["seed_overlap"]),
                        float(item["material_score"]),
                        int(item["area"]),
                    ),
                    reverse=True,
                )
                roots = roots[:1]
            if not roots:
                # Evidence exists, but it did not form an object-consistent
                # root.  Keep this side UNCERTAIN with alpha exactly zero.
                if components:
                    state[batch_index, side_index] = 1.0
                    confidence[batch_index, side_index] = float(max(item["score"] for item in components))
                    rejected_component_count_out[batch_index, side_index] = len(components)
                    best_component = max(components, key=lambda item: float(item["score"]))
                    if float(best_component.get("hair_overlap", 0.0)) >= 0.30:
                        reject_reason_out[batch_index, side_index] = 4.0
                    elif float(best_component.get("skin_overlap", 0.0)) >= 0.78:
                        reject_reason_out[batch_index, side_index] = 5.0
                    elif float(best_component.get("background_overlap", 0.0)) >= 0.55:
                        reject_reason_out[batch_index, side_index] = 3.0
                    else:
                        reject_reason_out[batch_index, side_index] = 2.0
                result_arrays["parser_earring_seed"][batch_index] = np.maximum(
                    result_arrays["parser_earring_seed"][batch_index], (parser_side | seed_side).astype(np.float32)
                )
                result_arrays["source_native_recall_hint"][batch_index] = np.maximum(
                    result_arrays["source_native_recall_hint"][batch_index], recall_hint_side.astype(np.float32)
                )
                result_arrays["lobe_anchor"][batch_index] = np.maximum(result_arrays["lobe_anchor"][batch_index], anchor.astype(np.float32))
                result_arrays["localization_core"][batch_index] = np.maximum(result_arrays["localization_core"][batch_index], core.astype(np.float32))
                result_arrays["probable_visual_evidence"][batch_index] = np.maximum(result_arrays["probable_visual_evidence"][batch_index], probable.astype(np.float32))
                continue

            root_component_id_out[batch_index, side_index] = float(roots[0]["id"])
            root = roots[0]

            selected: dict[int, tuple[int, float]] = {}
            frontier: list[tuple[dict[str, object], int, float]] = []
            for root in roots:
                component_id = int(root["id"])
                if component_id not in selected or selected[component_id][1] > 0:
                    selected[component_id] = (0, 0.0)
                    frontier.append((root, 0, 0.0))
            while frontier:
                current, depth, cumulative_cost = frontier.pop(0)
                if depth >= max(1, int(max_graph_depth)):
                    continue
                for candidate_item in components:
                    candidate_id = int(candidate_item["id"])
                    if candidate_id in selected:
                        continue
                    gap = _bbox_gap(current["bbox"], candidate_item["bbox"])
                    # A long pendant can contain separately detected metal,
                    # jewel and highlight components.  This is a connection
                    # test only; it never expands the final write alpha.
                    max_gap = max(
                        radius(3),
                        radius(8 if allow_long_continuation else 6) - 2 * depth,
                    )
                    if gap > max_gap:
                        continue
                    lab_distance = float(np.linalg.norm(current["mean_lab"] - candidate_item["mean_lab"]))
                    appearance = math.exp(-lab_distance / 38.0)
                    axis_delta = abs(math.sin(float(current["axis"]) - float(candidate_item["axis"])))
                    axis_consistency = 1.0 - axis_delta
                    vertical = 1.0 if float(candidate_item["centroid_y"]) >= float(current["centroid_y"]) - radius(8) else 0.45
                    edge_score = 0.44 * appearance + 0.30 * (1.0 - gap / max(1.0, max_gap)) + 0.16 * axis_consistency + 0.10 * vertical
                    if edge_score < (0.44 if allow_long_continuation else 0.56):
                        continue
                    # Once a semantic label-9 root exists, an unlabelled
                    # component is allowed to continue it only with strong
                    # native object evidence.  Broad visual components beside
                    # the ear are the usual source of copied background
                    # strands and duplicate earrings.
                    hair_continuation_ok = _verified_hair_continuation_v6(
                        root,
                        current,
                        candidate_item,
                        gap=gap,
                        scale=scale,
                        enabled=allow_semantic_hair_continuation,
                    )
                    if (
                        float(root.get("parser_overlap", 0.0)) > 0.02
                        and float(candidate_item.get("parser_overlap", 0.0)) <= 0.02
                        and not (
                            (
                                float(candidate_item.get("background_overlap", 1.0)) <= 0.18
                                and float(candidate_item.get("hair_overlap", 1.0)) <= 0.10
                                and float(candidate_item.get("material_score", 0.0)) >= 0.55
                                and float(candidate_item.get("edge_energy", 0.0)) >= 0.25
                                and gap <= radius(8)
                            )
                            or hair_continuation_ok
                        )
                    ):
                        continue
                    next_cost = cumulative_cost + (1.0 - edge_score) + 0.18 * float(candidate_item["semantic_penalty"])
                    cost_limit = float(max_cumulative_cost) * (1.8 if allow_long_continuation else 1.0)
                    if next_cost > cost_limit:
                        continue
                    selected[candidate_id] = (depth + 1, next_cost)
                    frontier.append((candidate_item, depth + 1, next_cost))

            selected_mask = np.zeros((height, width), dtype=bool)
            label_debug = np.zeros((height, width), dtype=np.float32)
            score_debug = np.zeros((height, width), dtype=np.float32)
            adaptive = np.zeros((height, width), dtype=bool)
            max_depth_used = 0
            max_cost_used = 0.0
            for item in components:
                item_mask = item["mask"]
                label_debug[item_mask] = float(item["id"])
                score_debug[item_mask] = float(item["score"])
                item_id = int(item["id"])
                if item_id in selected:
                    selected_mask |= item_mask
                    depth, item_cost = selected[item_id]
                    max_depth_used = max(max_depth_used, depth)
                    max_cost_used = max(max_cost_used, item_cost)
                    adaptive |= cv2.dilate(item_mask.astype(np.uint8), np.ones((2 * radius(2) + 1, 2 * radius(2) + 1), np.uint8), 1).astype(bool)

            if allow_long_continuation and selected_mask.any():
                # A long pendant is often split into several low-contrast
                # components.  The graph above may reject a legitimate lower
                # segment even after its distance/cost budget is relaxed.  Add
                # only a component that is below an accepted segment, close in
                # colour, narrow/elongated, and still materially distinct from
                # the local background.  This is a source-lobe continuation,
                # never a free rectangular rail; hair and broad background
                # components remain excluded.
                selected_items = [
                    item for item in components if int(item["id"]) in selected
                ]
                continuation_limit = int(0.05 * height * width)
                continuation_area = int(selected_mask.sum())
                changed = True
                while changed and continuation_area < continuation_limit:
                    changed = False
                    best_item = None
                    best_score = -1.0
                    for candidate_item in components:
                        candidate_id = int(candidate_item["id"])
                        if candidate_id in selected:
                            continue
                        if float(candidate_item["centroid_y"]) < min(
                            float(item["centroid_y"]) for item in selected_items
                        ) - radius(4):
                            continue
                        if float(candidate_item["background_overlap"] if "background_overlap" in candidate_item else 0.0) >= 0.86:
                            continue
                        if float(candidate_item["skin_overlap"]) >= 0.78:
                            continue
                        if float(candidate_item["material_score"]) < 0.30 or float(candidate_item["edge_energy"]) < 0.10:
                            continue
                        nearest = min(
                            selected_items,
                            key=lambda item: _bbox_gap(item["bbox"], candidate_item["bbox"]),
                        )
                        gap = _bbox_gap(nearest["bbox"], candidate_item["bbox"])
                        if gap > radius(8):
                            continue
                        if (
                            float(candidate_item["hair_overlap"]) >= 0.30
                            and not _verified_hair_continuation_v6(
                                root,
                                nearest,
                                candidate_item,
                                gap=gap,
                                scale=scale,
                                enabled=allow_semantic_hair_continuation,
                            )
                        ):
                            continue
                        lab_distance = float(np.linalg.norm(nearest["mean_lab"] - candidate_item["mean_lab"]))
                        if lab_distance > 68.0:
                            continue
                        x_distance = abs(float(candidate_item["centroid_x"]) - float(nearest["centroid_x"]))
                        if x_distance > radius(10):
                            continue
                        score = (
                            0.45 * float(candidate_item["material_score"])
                            + 0.35 * float(candidate_item["edge_energy"])
                            + 0.20 * max(0.0, 1.0 - lab_distance / 68.0)
                        )
                        if score > best_score:
                            best_item = candidate_item
                            best_score = score
                    if best_item is None:
                        break
                    best_id = int(best_item["id"])
                    selected[best_id] = (1, 0.0)
                    selected_items.append(best_item)
                    selected_mask |= best_item["mask"]
                    adaptive |= cv2.dilate(
                        best_item["mask"].astype(np.uint8),
                        np.ones((2 * radius(2) + 1, 2 * radius(2) + 1), np.uint8),
                        1,
                    ).astype(bool)
                    continuation_area = int(selected_mask.sum())
                    changed = True

            accepted_component_count_out[batch_index, side_index] = float(len(selected))
            rejected_component_count_out[batch_index, side_index] = float(
                max(0, len(components) - len(selected))
            )

            # Decontaminate the selected boundary before exporting RGB alpha.
            # GrabCut correctly finds the broad hoop, but antialiased source
            # edges can still contain a one-pixel strip of the green/bright
            # background.  Compare boundary pixels with a local outside-
            # background estimate and remove only pixels that are both
            # colour-identical to that outside ring.  Keep the trusted object
            # core intact, then alpha-matte its antialiased boundary below.
            # Deleting every near-background boundary pixel made thin long
            # earrings and small studs collapse into fragments.
            if selected_mask.any():
                trim_kernel = np.ones((2 * radius(2) + 1, 2 * radius(2) + 1), np.uint8)
                trim_window = (int(trim_kernel.shape[1]), int(trim_kernel.shape[0]))
                # Matte/contour operations must stay at one or two native
                # pixels.  ``radius(1)`` scales to four pixels on a 1024px
                # image, which visibly shrinks small studs and softens long
                # pendant edges.  The wider trim window above is only a local
                # background estimator and does not change the matte width.
                # Keep the matte to one *native* pixel by default.  Scaling a
                # one-pixel operation with image resolution turns a thin
                # pendant into nothing but boundary and makes the recovered
                # object look smaller/translucent.
                matte_radius = max(1, min(2, int(native_matte_radius_px)))
                boundary = selected_mask & ~cv2.erode(
                    selected_mask.astype(np.uint8),
                    np.ones((2 * matte_radius + 1, 2 * matte_radius + 1), np.uint8),
                    1,
                ).astype(bool)
                outside = cv2.dilate(
                    selected_mask.astype(np.uint8),
                    trim_kernel,
                    1,
                ).astype(bool) & ~selected_mask
                # The source backdrop behind an earring is not always parser
                # label 0.  Long pendants frequently hang in front of source
                # hair.  Treat that *outside* hair as a local matte backdrop
                # too, so a mixed edge is reconstructed as jewellery over the
                # target image rather than copied as a bright source-hair rim.
                # Source skin is deliberately excluded: it is not a stable
                # backdrop estimate and must remain target-owned.
                outside_background = outside & (semantic_background | semantic_hair)
                bg_support = cv2.boxFilter(
                    outside_background.astype(np.float32),
                    cv2.CV_32F,
                    trim_window,
                    normalize=False,
                )
                bg_lab = np.stack(
                    [
                        cv2.boxFilter(
                            (lab[:, :, channel].astype(np.float32) * outside_background.astype(np.float32)),
                            cv2.CV_32F,
                            trim_window,
                            normalize=False,
                        )
                        for channel in range(3)
                    ],
                    axis=2,
                ) / np.maximum(bg_support[:, :, None], 1.0)
                bg_rgb = np.stack(
                    [
                        cv2.boxFilter(
                            image[:, :, channel].astype(np.float32) * outside_background.astype(np.float32),
                            cv2.CV_32F,
                            trim_window,
                            normalize=False,
                        )
                        for channel in range(3)
                    ],
                    axis=2,
                ) / np.maximum(bg_support[:, :, None], 1.0)
                bg_distance = np.linalg.norm(
                    lab.astype(np.float32) - bg_lab,
                    axis=2,
                )
                background_halo = (
                    boundary
                    & ~trusted_core
                    & (bg_support >= 2.0)
                    & (bg_distance <= 20.0)
                )
                # Parser labels were already considered while scoring the
                # lobe-rooted components.  They must not become a final
                # pixel-level veto: a valid long pendant is often labelled
                # hair, neck or background after it leaves the ear shell.
                # Keep only the measured local-background halo rejection;
                # this removes source backdrop fringe without shaving a
                # verified object contour.
                selected_mask &= ~background_halo

            final_alpha = selected_mask.astype(np.float32)
            # The source boundary is generally antialiased over its original
            # backdrop.  A binary paste copies that backdrop as a green/bright
            # rim; deleting the boundary instead cuts the actual jewellery.
            # Estimate a local foreground colour from the component core and
            # turn only the boundary into a soft alpha.  The final composite
            # then supplies the target's already-transferred background (black
            # hair in the reported case) behind the full object.
            if selected_mask.any():
                core = cv2.erode(
                    selected_mask.astype(np.uint8),
                    np.ones((2 * matte_radius + 1, 2 * matte_radius + 1), np.uint8),
                    1,
                ).astype(bool)
                core_support = cv2.boxFilter(
                    core.astype(np.float32),
                    cv2.CV_32F,
                    trim_window,
                    normalize=False,
                )
                core_rgb = np.stack(
                    [
                        cv2.boxFilter(
                            image[:, :, channel].astype(np.float32) * core.astype(np.float32),
                            cv2.CV_32F,
                            trim_window,
                            normalize=False,
                        )
                        for channel in range(3)
                    ],
                    axis=2,
                ) / np.maximum(core_support[:, :, None], 1.0)
                matte_region = (
                    boundary
                    & selected_mask
                    & (bg_support >= 2.0)
                    & (core_support >= 1.0)
                )
                if matte_region.any():
                    observed = image.astype(np.float32)
                    fg_distance = np.linalg.norm(core_rgb - bg_rgb, axis=2)
                    observed_distance = np.linalg.norm(observed - bg_rgb, axis=2)
                    matte_alpha = np.clip(
                        observed_distance / np.maximum(fg_distance, 1.0),
                        0.0,
                        1.0,
                    )
                    # Do not apply the uncertain colour unmixing where the
                    # local foreground and background are indistinguishable.
                    matte_region &= fg_distance >= 12.0
                    # A non-zero alpha floor writes a visible strip of the
                    # source backdrop around every antialiased edge.  Keep the
                    # mathematically estimated coverage instead: at a pure
                    # source-background edge alpha is zero and the completed
                    # target (for example its black hair) owns that pixel.
                    matte_alpha = np.clip(matte_alpha, 0.0, 1.0)
                    if matte_region.any():
                        alpha_values = matte_alpha[matte_region]
                        stable_alpha = np.maximum(alpha_values, 1e-3)
                        reconstructed = (
                            observed[matte_region]
                            - (1.0 - alpha_values[:, None]) * bg_rgb[matte_region]
                        ) / stable_alpha[:, None]
                        foreground_rgb[batch_index, matte_region] = np.clip(
                            reconstructed,
                            0.0,
                            255.0,
                        ) / 255.0
                        # After unmixing, the RGB no longer contains the source
                        # backdrop.  Retain a visible coverage floor for a
                        # verified one-pixel wire/stone edge; discard only a
                        # nearly pure-background subpixel.  This restores the
                        # object's apparent size without pasting a background
                        # halo or filling a hoop centre.
                        alpha_floor = max(
                            0.0,
                            min(1.0, float(native_boundary_alpha_floor)),
                        )
                        final_alpha[matte_region] = np.where(
                            alpha_values >= 0.08,
                            np.maximum(alpha_values, alpha_floor),
                            0.0,
                        )
            parser_or_seed = parser_side | seed_side
            has_native_object = final_alpha.any()
            if not has_native_object:
                state[batch_index, side_index] = 1.0
            else:
                state[batch_index, side_index] = 2.0
            root_strength = max(float(item["score"]) for item in roots)
            confidence[batch_index, side_index] = min(1.0, max(0.0, root_strength / 3.5 + 0.15 * len(selected)))
            max_depth_out[batch_index, side_index] = max_depth_used
            cumulative_cost_out[batch_index, side_index] = max_cost_used
            # A real hoop can have one- or two-pixel segmentation gaps.  Close
            # only that microscopic contour gap before finding the hole, then
            # keep the centre target-owned.  This cannot add RGB pixels and
            # prevents source background from leaking through a nearly closed
            # ring into the output.
            topology = cv2.morphologyEx(
                (final_alpha > 0.5).astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((2 * max(1, min(2, int(round(scale * 0.5)))) + 1,
                         2 * max(1, min(2, int(round(scale * 0.5)))) + 1), np.uint8),
            ).astype(bool)
            hole = _fill_holes(topology)
            final_alpha *= (~hole).astype(np.float32)
            alpha_key = "source_native_left_alpha" if side_index == 0 else "source_native_right_alpha"
            result_arrays[alpha_key][batch_index] = final_alpha.astype(np.float32)
            result_arrays["source_native_earring_alpha"][batch_index] = np.maximum(result_arrays["source_native_earring_alpha"][batch_index], final_alpha.astype(np.float32))
            result_arrays["source_native_hole_alpha"][batch_index] = np.maximum(result_arrays["source_native_hole_alpha"][batch_index], hole.astype(np.float32))
            hole_key = "source_native_left_hole_alpha" if side_index == 0 else "source_native_right_hole_alpha"
            result_arrays[hole_key][batch_index] = hole.astype(np.float32)
            result_arrays["parser_earring_seed"][batch_index] = np.maximum(result_arrays["parser_earring_seed"][batch_index], parser_or_seed.astype(np.float32))
            result_arrays["source_native_recall_hint"][batch_index] = np.maximum(
                result_arrays["source_native_recall_hint"][batch_index], recall_hint_side.astype(np.float32)
            )
            result_arrays["lobe_anchor"][batch_index] = np.maximum(result_arrays["lobe_anchor"][batch_index], anchor.astype(np.float32))
            result_arrays["localization_core"][batch_index] = np.maximum(result_arrays["localization_core"][batch_index], core.astype(np.float32))
            result_arrays["localization_adaptive"][batch_index] = np.maximum(result_arrays["localization_adaptive"][batch_index], adaptive.astype(np.float32))
            result_arrays["probable_visual_evidence"][batch_index] = np.maximum(result_arrays["probable_visual_evidence"][batch_index], probable.astype(np.float32))
            result_arrays["grabcut_raw"][batch_index] = np.maximum(result_arrays["grabcut_raw"][batch_index], grabcut.astype(np.float32))
            result_arrays["component_labels"][batch_index] = np.maximum(result_arrays["component_labels"][batch_index], label_debug)
            result_arrays["component_scores"][batch_index] = np.maximum(result_arrays["component_scores"][batch_index], score_debug)
            result_arrays["selected_components"][batch_index] = np.maximum(result_arrays["selected_components"][batch_index], selected_mask.astype(np.float32))

    # The per-side inspection envelopes can overlap on near-frontal faces.
    # Without a final arbitration, one physical source earring may be emitted
    # in both side tensors and later aligned twice.  Resolve only genuinely
    # overlapping/near-identical objects; distinct left/right earrings are
    # left untouched.  This is a source-coordinate guard, so it cannot invent
    # an object or alter target visibility.
    for batch_index in range(batch):
        left = result_arrays["source_native_left_alpha"][batch_index]
        right = result_arrays["source_native_right_alpha"][batch_index]
        left_binary = left > 0.01
        right_binary = right > 0.01
        # These areas are used by duplicate-side arbitration below even when
        # the two candidate masks do not overlap.  Compute them before the
        # overlap-only branch so valid single/two-earring samples never hit an
        # uninitialised local variable during SATD protection.
        left_area = float(left_binary.sum())
        right_area = float(right_binary.sum())
        overlap = left_binary & right_binary
        if overlap.any():
            labels = parsing_np[batch_index]
            left_ear_support = float((left_binary & (labels == RAW_LEFT_EAR)).sum())
            right_ear_support = float((right_binary & (labels == RAW_RIGHT_EAR)).sum())
            # Keep the side with stronger native ear/parser support.  If both
            # supports tie, retain the larger verified object and remove only
            # the duplicate overlap pixels from the other side.
            keep_left = (left_ear_support, left_area) >= (right_ear_support, right_area)
            if keep_left:
                right[overlap] = 0.0
            else:
                left[overlap] = 0.0

        left_points = np.argwhere(left > 0.01)
        right_points = np.argwhere(right > 0.01)
        if left_points.size and right_points.size:
            left_min = left_points.min(axis=0)
            left_max = left_points.max(axis=0)
            right_min = right_points.min(axis=0)
            right_max = right_points.max(axis=0)
            gap_y = max(int(left_min[0]) - int(right_max[0]), int(right_min[0]) - int(left_max[0]), 0)
            gap_x = max(int(left_min[1]) - int(right_max[1]), int(right_min[1]) - int(left_max[1]), 0)
            bbox_gap = math.hypot(gap_y, gap_x)
            left_center = left_points.mean(axis=0)
            right_center = right_points.mean(axis=0)
            center_gap = float(np.linalg.norm(left_center - right_center))
            # Two different ears are far apart in source coordinates.  A
            # near-identical pair is the duplicated-envelope failure case.
            duplicate_limit = max(8.0, 18.0 * scale)
            if bbox_gap <= duplicate_limit and center_gap <= 28.0 * scale:
                labels = parsing_np[batch_index]
                left_score = float((left > 0.01).astype(np.float32)[labels == RAW_LEFT_EAR].sum())
                right_score = float((right > 0.01).astype(np.float32)[labels == RAW_RIGHT_EAR].sum())
                if (left_score, left_area) >= (right_score, right_area):
                    right[...] = 0.0
                    result_arrays["source_native_right_hole_alpha"][batch_index] = 0.0
                else:
                    left[...] = 0.0
                    result_arrays["source_native_left_hole_alpha"][batch_index] = 0.0
        # Keep the combined alpha consistent with the arbitrated per-side
        # fields (the original loop populated it before this final guard).
        result_arrays["source_native_earring_alpha"][batch_index] = np.maximum(left, right)
        # Presence is published after arbitration as well.  Otherwise a
        # duplicate side removed above would still be marked positive in the
        # dataset and could reappear through the learned supervision path.
        state[batch_index, 0] = 2.0 if np.any(left > 0.01) else 0.0
        state[batch_index, 1] = 2.0 if np.any(right > 0.01) else 0.0
        if state[batch_index, 0] == 0.0:
            confidence[batch_index, 0] = 0.0
        if state[batch_index, 1] == 0.0:
            confidence[batch_index, 1] = 0.0

    for name, value in result_arrays.items():
        output[name] = torch.from_numpy(value).to(device=device, dtype=dtype).unsqueeze(1)
    output["source_native_earring_rgb"] = torch.from_numpy(foreground_rgb).to(
        device=device,
        dtype=dtype,
    ).permute(0, 3, 1, 2).contiguous()
    output["source_native_presence_state"] = torch.from_numpy(state).to(device=device, dtype=dtype)
    output["source_native_presence_score"] = torch.from_numpy(confidence).to(device=device, dtype=dtype)
    output["max_graph_depth_used"] = torch.from_numpy(max_depth_out).to(device=device, dtype=dtype)
    output["cumulative_graph_cost"] = torch.from_numpy(cumulative_cost_out).to(device=device, dtype=dtype)
    output["component_count"] = torch.from_numpy(component_count_out).to(device=device, dtype=dtype)
    output["root_component_id"] = torch.from_numpy(root_component_id_out).to(device=device, dtype=dtype)
    output["accepted_component_count"] = torch.from_numpy(accepted_component_count_out).to(device=device, dtype=dtype)
    output["rejected_component_count"] = torch.from_numpy(rejected_component_count_out).to(device=device, dtype=dtype)
    output["reject_reason"] = torch.from_numpy(reject_reason_out).to(device=device, dtype=dtype)
    output["accepted_component_ids"] = output["selected_components"].clone()
    output["rejected_component_ids"] = (
        output["component_labels"]
        * (1.0 - (output["selected_components"] > 0.5).to(dtype))
    ).clamp_min(0)
    return output


def align_earring_instance_v6(
    instance: EarringNativeInstanceV6,
    source_left_ear: torch.Tensor,
    source_right_ear: torch.Tensor,
    target_left_ear: torch.Tensor,
    target_right_ear: torch.Tensor,
    *,
    max_shift: int = 0,
    max_vertical_shift: int | None = None,
    max_horizontal_shift: int | None = None,
    identity_tolerance: int = 0,
    force_identity_alignment: bool = False,
) -> dict[str, torch.Tensor]:
    """Align one confirmed source-native instance once into target coordinates."""
    if instance.space not in (
        EarringCoordinateSpace.SOURCE_NATIVE,
        EarringCoordinateSpace.SOURCE_CANONICAL,
    ):
        raise AssertionError(
            "Expected a source-coordinate instance for alignment, "
            f"got {instance.space.value}."
        )
    source = _image_01(instance.rgb)
    size = tuple(source.shape[-2:])
    left = _mask(source_left_ear, size, source)
    right = _mask(source_right_ear, size, source)
    target_left = _mask(target_left_ear, size, source)
    target_right = _mask(target_right_ear, size, source)
    # Per-side alphas are carried explicitly.  A midpoint split is not a
    # coordinate contract: a visible ear can lie on either half of a crop and
    # a long pendant can cross the image centre.
    left_instance = _mask(instance.left_alpha, size, source)
    right_instance = _mask(instance.right_alpha, size, source)
    # Align the source object's top attachment band to the target lobe's
    # visible bottom band.  Aligning ear/earring centroids pulled a long
    # pendant upward and put the source accessory in the middle of the lobe.
    src_ly, src_lx, src_lvalid = _earring_top_attachment(left_instance)
    src_ry, src_rx, src_rvalid = _earring_top_attachment(right_instance)
    tgt_ly, tgt_lx, tgt_lvalid = _ear_bottom_attachment(target_left)
    tgt_ry, tgt_rx, tgt_rvalid = _ear_bottom_attachment(target_right)
    left_valid = src_lvalid & tgt_lvalid & (left_instance.flatten(1).sum(dim=1) > 0.5)
    right_valid = src_rvalid & tgt_rvalid & (right_instance.flatten(1).sum(dim=1) > 0.5)
    # Validate the raw translation before applying it.  Clamping an erroneous
    # cheek/face anchor to ``max_shift`` still writes a real earring at a wrong
    # location; an invalid correspondence must produce no object instead.
    raw_left_dy = torch.round(tgt_ly - src_ly)
    raw_left_dx = torch.round(tgt_lx - src_lx)
    raw_right_dy = torch.round(tgt_ry - src_ry)
    raw_right_dx = torch.round(tgt_rx - src_rx)
    vertical_limit = max(0, int(max_shift if max_vertical_shift is None else max_vertical_shift))
    horizontal_limit = max(0, int(max_shift if max_horizontal_shift is None else max_horizontal_shift))
    left_in_range = (
        (raw_left_dy.abs() <= float(vertical_limit))
        & (raw_left_dx.abs() <= float(horizontal_limit))
    )
    right_in_range = (
        (raw_right_dy.abs() <= float(vertical_limit))
        & (raw_right_dx.abs() <= float(horizontal_limit))
    )
    # HairFast keeps the source face/canonical coordinates.  If the extracted
    # attachment already lies close to the final exposed lobe, moving it to a
    # noisy parser centroid only places it on the cheek.  Prefer the exact
    # source coordinate in that case; this is translation-only and never
    # scales or redraws the object.
    identity_limit = max(0, int(identity_tolerance))
    if bool(force_identity_alignment):
        # HairFast's face is source-canonical after hairstyle transfer.  The
        # target masks supplied by this mode are visibility probes only; do
        # not let their parser centroids translate a correct source object.
        left_identity = left_valid
        right_identity = right_valid
    elif identity_limit > 0:
        left_identity = left_valid & (
            torch.maximum(raw_left_dy.abs(), raw_left_dx.abs()) <= float(identity_limit)
        )
        right_identity = right_valid & (
            torch.maximum(raw_right_dy.abs(), raw_right_dx.abs()) <= float(identity_limit)
        )
    else:
        left_identity = torch.zeros_like(left_valid)
        right_identity = torch.zeros_like(right_valid)
    # For max_shift=0, permit only an already coincident attachment band
    # (within one native pixel).  A remote/face anchor remains rejected;
    # clamping it would place the source object at an arbitrary location.
    left_zero_fallback = left_valid & ~left_in_range & (
        torch.maximum(raw_left_dy.abs(), raw_left_dx.abs()) <= 1.0
    )
    right_zero_fallback = right_valid & ~right_in_range & (
        torch.maximum(raw_right_dy.abs(), raw_right_dx.abs()) <= 1.0
    )
    left_shift_valid = left_valid & (left_identity | left_in_range | left_zero_fallback)
    right_shift_valid = right_valid & (right_identity | right_in_range | right_zero_fallback)
    left_zero = left_identity | left_zero_fallback
    right_zero = right_identity | right_zero_fallback
    left_dy = torch.where(
        left_shift_valid & ~left_zero,
        raw_left_dy,
        torch.zeros_like(raw_left_dy),
    )
    left_dx = torch.where(
        left_shift_valid & ~left_zero,
        raw_left_dx,
        torch.zeros_like(raw_left_dx),
    )
    right_dy = torch.where(
        right_shift_valid & ~right_zero,
        raw_right_dy,
        torch.zeros_like(raw_right_dy),
    )
    right_dx = torch.where(
        right_shift_valid & ~right_zero,
        raw_right_dx,
        torch.zeros_like(raw_right_dx),
    )
    left_alpha = _shift_per_batch(left_instance, left_dy, left_dx)
    right_alpha = _shift_per_batch(right_instance, right_dy, right_dx)
    left_hole = _shift_per_batch(_mask(instance.left_hole_alpha, size, source), left_dy, left_dx)
    right_hole = _shift_per_batch(_mask(instance.right_hole_alpha, size, source), right_dy, right_dx)
    left_alpha = left_alpha * left_shift_valid.view(-1, 1, 1, 1).to(left_alpha.dtype)
    right_alpha = right_alpha * right_shift_valid.view(-1, 1, 1, 1).to(right_alpha.dtype)
    left_hole = left_hole * left_shift_valid.view(-1, 1, 1, 1).to(left_hole.dtype)
    right_hole = right_hole * right_shift_valid.view(-1, 1, 1, 1).to(right_hole.dtype)
    aligned_alpha = (left_alpha + right_alpha).clamp(0, 1)
    aligned_hole = (left_hole + right_hole).clamp(0, 1)
    aligned_alpha = aligned_alpha * (1.0 - aligned_hole).clamp(0, 1)
    # Keep RGB straight (not premultiplied).  The legacy binary compositor
    # multiplied before alignment and again during compositing, which happened
    # to be harmless for alpha 0/1 but squares a real matte edge into a dark,
    # broken-looking pendant.  Weight only while combining two possible sides;
    # ``composite_earring_v6`` applies the one final target alpha.
    left_rgb = _shift_per_batch(source, left_dy, left_dx)
    right_rgb = _shift_per_batch(source, right_dy, right_dx)
    rgb_weight = (left_alpha + right_alpha).clamp_min(1e-6)
    aligned_rgb = (
        left_rgb * left_alpha + right_rgb * right_alpha
    ) / rgb_weight
    return {
        "target_aligned_earring_alpha": aligned_alpha,
        "target_aligned_earring_rgb": aligned_rgb.clamp(0, 1),
        "target_aligned_hole_alpha": aligned_hole,
        "target_aligned_left_alpha": left_alpha,
        "target_aligned_right_alpha": right_alpha,
        "left_shift_y": left_dy.view(-1, 1),
        "left_shift_x": left_dx.view(-1, 1),
        "right_shift_y": right_dy.view(-1, 1),
        "right_shift_x": right_dx.view(-1, 1),
        "left_raw_shift_y": raw_left_dy.view(-1, 1),
        "left_raw_shift_x": raw_left_dx.view(-1, 1),
        "right_raw_shift_y": raw_right_dy.view(-1, 1),
        "right_raw_shift_x": raw_right_dx.view(-1, 1),
        "left_alignment_valid": left_shift_valid.view(-1, 1).to(source.dtype),
        "right_alignment_valid": right_shift_valid.view(-1, 1).to(source.dtype),
        "fallback_zero_shift_used_left": left_zero_fallback.view(-1, 1).to(source.dtype),
        "fallback_zero_shift_used_right": right_zero_fallback.view(-1, 1).to(source.dtype),
        "identity_alignment_used_left": left_identity.view(-1, 1).to(source.dtype),
        "identity_alignment_used_right": right_identity.view(-1, 1).to(source.dtype),
    }


def composite_earring_v6(
    base: torch.Tensor,
    aligned: dict[str, torch.Tensor],
    target_left_ear: torch.Tensor,
    target_right_ear: torch.Tensor,
    *,
    min_visible_area: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Final object-only write from an already target-gated native instance."""
    base_01 = _image_01(base)
    size = tuple(base_01.shape[-2:])
    left_gate = _side_open(_mask(target_left_ear, size, base_01), min_visible_area)
    right_gate = _side_open(_mask(target_right_ear, size, base_01), min_visible_area)
    alpha = (
        _mask(aligned.get("target_aligned_left_alpha"), size, base_01) * left_gate
        + _mask(aligned.get("target_aligned_right_alpha"), size, base_01) * right_gate
    ).clamp(0, 1)
    hole = _mask(aligned.get("target_aligned_hole_alpha"), size, base_01)
    alpha = alpha * (1.0 - hole).clamp(0, 1)
    rgb = aligned["target_aligned_earring_rgb"].to(device=base_01.device, dtype=base_01.dtype)
    if rgb.shape[-2:] != size:
        rgb = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False)
    return (base_01 * (1.0 - alpha) + rgb.clamp(0, 1) * alpha).clamp(0, 1), alpha
