from __future__ import annotations

from collections import deque

import torch

import models.ear_modules_v5 as ear_modules_v5
from models.ear_modules_v5 import MorphologyInteriorAnalyzer, binary_fill_holes


def _reference_binary_fill_holes(mask: torch.Tensor) -> torch.Tensor:
    """Original implementation retained as an output-equivalence oracle."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    binary = (mask > 0.5).float()
    filled = binary.clone()
    for batch_index in range(binary.size(0)):
        height, width = binary.shape[-2:]
        background = binary[batch_index, 0] < 0.5
        outside = torch.zeros_like(background)
        queue = deque()
        for x in range(width):
            if bool(background[0, x]):
                queue.append((0, x))
            if bool(background[height - 1, x]):
                queue.append((height - 1, x))
        for y in range(height):
            if bool(background[y, 0]):
                queue.append((y, 0))
            if bool(background[y, width - 1]):
                queue.append((y, width - 1))
        while queue:
            y, x = queue.popleft()
            if bool(outside[y, x]) or not bool(background[y, x]):
                continue
            outside[y, x] = True
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and not bool(outside[ny, nx]):
                    queue.append((ny, nx))
        filled[batch_index, 0] = (~outside).float()
    return filled


def _assert_matches_reference(mask: torch.Tensor) -> None:
    expected = _reference_binary_fill_holes(mask)
    actual = binary_fill_holes(mask)
    assert torch.equal(actual.cpu(), expected.cpu())
    assert actual.device == mask.device


def test_binary_fill_holes_matches_edge_cases():
    cases = []

    empty = torch.zeros(1, 1, 15, 17)
    cases.append(empty)

    full = torch.ones(1, 1, 15, 17)
    cases.append(full)

    ring = torch.zeros(1, 1, 15, 17)
    ring[:, :, 3:12, 4:13] = 1
    ring[:, :, 5:10, 6:11] = 0
    cases.append(ring)

    open_ring = ring.clone()
    open_ring[:, :, 3:7, 8] = 0
    cases.append(open_ring)

    border_ring = torch.ones(1, 1, 15, 17)
    border_ring[:, :, 2:13, 2:15] = 0
    border_ring[:, :, 5:10, 5:12] = 1
    cases.append(border_ring)

    diagonal_gap = torch.zeros(1, 1, 9, 9)
    diagonal_gap[:, :, 2:7, 2:7] = 1
    diagonal_gap[:, :, 3:6, 3:6] = 0
    diagonal_gap[:, :, 2, 2] = 0
    cases.append(diagonal_gap)

    batched = torch.cat((ring, open_ring, empty), dim=0)
    cases.append(batched)

    multi_channel = torch.cat((ring, torch.rand_like(ring)), dim=1)
    cases.append(multi_channel)

    for case in cases:
        _assert_matches_reference(case)


def test_binary_fill_holes_matches_random_masks():
    generator = torch.Generator().manual_seed(3407)
    for height, width in ((8, 9), (17, 23), (48, 41)):
        for probability in (0.05, 0.25, 0.50, 0.80):
            mask = (torch.rand((3, 1, height, width), generator=generator) < probability).float()
            _assert_matches_reference(mask)


def test_morphology_analyzer_outputs_are_unchanged():
    generator = torch.Generator().manual_seed(3407)
    source = torch.rand((2, 3, 72, 80), generator=generator)
    object_mask = torch.zeros((2, 1, 72, 80))
    object_mask[0, :, 10:60, 12:68] = 1
    object_mask[0, :, 20:50, 24:56] = 0
    object_mask[1, :, 8:64, 18:62] = 1
    object_mask[1, :, 24:48, 30:50] = 0
    object_mask[1, :, 8:35, 40] = 0
    material = torch.rand((2, 1, 72, 80), generator=generator)
    proposal = torch.rand((2, 1, 72, 80), generator=generator)
    parsing = torch.randint(0, 19, (2, 1, 72, 80), generator=generator)
    analyzer = MorphologyInteriorAnalyzer()

    fast_fill = ear_modules_v5.binary_fill_holes
    ear_modules_v5.binary_fill_holes = _reference_binary_fill_holes
    try:
        expected = analyzer(source, object_mask, material, proposal, parsing)
    finally:
        ear_modules_v5.binary_fill_holes = fast_fill
    actual = analyzer(source, object_mask, material, proposal, parsing)

    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        if torch.is_tensor(expected_value):
            assert torch.equal(actual[key], expected_value), key
        else:
            assert actual[key] == expected_value, key


if __name__ == "__main__":
    test_binary_fill_holes_matches_edge_cases()
    test_binary_fill_holes_matches_random_masks()
    test_morphology_analyzer_outputs_are_unchanged()
