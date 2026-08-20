"""Source-native earring instances for the V5 final compositor.

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
class EarringNativeInstanceV5:
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
        "max_graph_depth_used": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
        "cumulative_graph_cost": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
        "component_count": torch.zeros(source.size(0), 2, device=source.device, dtype=source.dtype),
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


def extract_source_native_earring_v5(
    source_rgb: torch.Tensor,
    source_parsing: torch.Tensor | None,
    *,
    source_native_seed: torch.Tensor | None = None,
    seed_space: EarringCoordinateSpace | str = EarringCoordinateSpace.SOURCE_NATIVE,
    source_native_recall_hint: torch.Tensor | None = None,
    recall_hint_space: EarringCoordinateSpace | str = EarringCoordinateSpace.SOURCE_CANONICAL,
    max_graph_depth: int = 3,
    max_cumulative_cost: float = 1.55,
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
            anchor_y, anchor_x = np.argwhere(anchor).mean(axis=0)
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

            count, component_labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
            components: list[dict[str, object]] = []
            for component_id in range(1, count):
                component = component_labels == component_id
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                if area <= 0 or area > int(0.055 * height * width):
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
                    material_score >= 0.58
                    and edge_energy >= 0.20
                    and (elongation >= 1.30 or fill_ratio <= 0.48)
                )
                if background_only and not background_continuation_ok:
                    continue
                # A long pendant may extend down from the lobe, but a
                # background-labelled continuation above the lobe is an ear-
                # side hair/background filament, never jewellery.  This keeps
                # the locator from following the bright strings beside a hoop
                # into the transferred target hair.
                if background_only and centroid_y < anchor_y - radius(3):
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
            if not roots:
                # Evidence exists, but it did not form an object-consistent
                # root.  Keep this side UNCERTAIN with alpha exactly zero.
                if components:
                    state[batch_index, side_index] = 1.0
                    confidence[batch_index, side_index] = float(max(item["score"] for item in components))
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
                    max_gap = max(radius(6), radius(18) - 2 * depth)
                    if gap > max_gap:
                        continue
                    lab_distance = float(np.linalg.norm(current["mean_lab"] - candidate_item["mean_lab"]))
                    appearance = math.exp(-lab_distance / 38.0)
                    axis_delta = abs(math.sin(float(current["axis"]) - float(candidate_item["axis"])))
                    axis_consistency = 1.0 - axis_delta
                    vertical = 1.0 if float(candidate_item["centroid_y"]) >= float(current["centroid_y"]) - radius(8) else 0.45
                    edge_score = 0.44 * appearance + 0.30 * (1.0 - gap / max(1.0, max_gap)) + 0.16 * axis_consistency + 0.10 * vertical
                    if edge_score < 0.56:
                        continue
                    next_cost = cumulative_cost + (1.0 - edge_score) + 0.18 * float(candidate_item["semantic_penalty"])
                    if next_cost > float(max_cumulative_cost):
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
                boundary = selected_mask & ~cv2.erode(
                    selected_mask.astype(np.uint8),
                    np.ones((2 * radius(1) + 1, 2 * radius(1) + 1), np.uint8),
                    1,
                ).astype(bool)
                outside = cv2.dilate(
                    selected_mask.astype(np.uint8),
                    trim_kernel,
                    1,
                ).astype(bool) & ~selected_mask
                outside_background = outside & semantic_background
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
                hair_leak = (
                    selected_mask
                    & semantic_hair
                )
                selected_mask &= ~(background_halo | hair_leak)

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
                    np.ones((2 * radius(1) + 1, 2 * radius(1) + 1), np.uint8),
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
                        reconstructed = (
                            observed[matte_region]
                            - (1.0 - alpha_values[:, None]) * bg_rgb[matte_region]
                        ) / alpha_values[:, None]
                        foreground_rgb[batch_index, matte_region] = np.clip(
                            reconstructed,
                            0.0,
                            255.0,
                        ) / 255.0
                        final_alpha[matte_region] = alpha_values
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
                np.ones((2 * radius(1) + 1, 2 * radius(1) + 1), np.uint8),
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
    return output


def align_earring_instance_v5(
    instance: EarringNativeInstanceV5,
    source_left_ear: torch.Tensor,
    source_right_ear: torch.Tensor,
    target_left_ear: torch.Tensor,
    target_right_ear: torch.Tensor,
    *,
    max_shift: int = 0,
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
    # Align the object by the ear lobe, never by ``ear + earring``.  The old
    # mixed centroid was pulled toward a long pendant and placed the recovered
    # object beside the lobe instead of at its attachment point.
    source_left_anchor = _lobe_anchor(left)
    source_right_anchor = _lobe_anchor(right)
    target_left_anchor = _lobe_anchor(target_left)
    target_right_anchor = _lobe_anchor(target_right)
    src_ly, src_lx, src_lvalid = _centroid(source_left_anchor)
    src_ry, src_rx, src_rvalid = _centroid(source_right_anchor)
    tgt_ly, tgt_lx, tgt_lvalid = _centroid(target_left_anchor)
    tgt_ry, tgt_rx, tgt_rvalid = _centroid(target_right_anchor)
    left_valid = src_lvalid & tgt_lvalid & (left_instance.flatten(1).sum(dim=1) > 0.5)
    right_valid = src_rvalid & tgt_rvalid & (right_instance.flatten(1).sum(dim=1) > 0.5)
    left_dy = torch.where(left_valid, torch.round(tgt_ly - src_ly), torch.zeros_like(src_ly)).clamp(-max_shift, max_shift)
    left_dx = torch.where(left_valid, torch.round(tgt_lx - src_lx), torch.zeros_like(src_lx)).clamp(-max_shift, max_shift)
    right_dy = torch.where(right_valid, torch.round(tgt_ry - src_ry), torch.zeros_like(src_ry)).clamp(-max_shift, max_shift)
    right_dx = torch.where(right_valid, torch.round(tgt_rx - src_rx), torch.zeros_like(src_rx)).clamp(-max_shift, max_shift)
    left_alpha = _shift_per_batch(left_instance, left_dy, left_dx)
    right_alpha = _shift_per_batch(right_instance, right_dy, right_dx)
    left_hole = _shift_per_batch(_mask(instance.left_hole_alpha, size, source), left_dy, left_dx)
    right_hole = _shift_per_batch(_mask(instance.right_hole_alpha, size, source), right_dy, right_dx)
    aligned_alpha = (left_alpha + right_alpha).clamp(0, 1)
    aligned_hole = (left_hole + right_hole).clamp(0, 1)
    aligned_alpha = aligned_alpha * (1.0 - aligned_hole).clamp(0, 1)
    # Keep RGB straight (not premultiplied).  The legacy binary compositor
    # multiplied before alignment and again during compositing, which happened
    # to be harmless for alpha 0/1 but squares a real matte edge into a dark,
    # broken-looking pendant.  Weight only while combining two possible sides;
    # ``composite_earring_v5`` applies the one final target alpha.
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
    }


def composite_earring_v5(
    base: torch.Tensor,
    aligned: dict[str, torch.Tensor],
    target_left_ear: torch.Tensor,
    target_right_ear: torch.Tensor,
    *,
    min_visible_area: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Final object-only write.  Target hair is deliberately not a clip mask."""
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
