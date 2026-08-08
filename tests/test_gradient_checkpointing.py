from __future__ import annotations

import copy

import torch

from models.ear_modules_v5 import MorphologyAwareDualBranchRenderer
from tests.v3c_test_utils import renderer_data


def test_renderer_checkpointing_preserves_forward_and_gradients():
    torch.manual_seed(3407)
    direct = MorphologyAwareDualBranchRenderer(
        crop_size=32,
        large_crop_size=32,
        base_channels=4,
        gate_mode="learned",
        gradient_checkpointing=False,
    )
    checkpointed = copy.deepcopy(direct)
    checkpointed.gradient_checkpointing = True
    direct.train()
    checkpointed.train()

    raw_pp = torch.rand(1, 3, 64, 64)
    source = torch.rand(1, 3, 64, 64)
    data = renderer_data(64)

    direct_final, direct_aux = direct(raw_pp, source, data)
    checkpointed_final, checkpointed_aux = checkpointed(raw_pp, source, data)
    assert torch.equal(direct_final, checkpointed_final)
    assert torch.equal(direct_aux["delta_rgb"], checkpointed_aux["delta_rgb"])

    (direct_final.mean() + direct_aux["delta_rgb"].mean()).backward()
    (checkpointed_final.mean() + checkpointed_aux["delta_rgb"].mean()).backward()
    for direct_parameter, checkpointed_parameter in zip(direct.parameters(), checkpointed.parameters()):
        assert direct_parameter.grad is not None
        assert checkpointed_parameter.grad is not None
        assert torch.equal(direct_parameter.grad, checkpointed_parameter.grad)


if __name__ == "__main__":
    test_renderer_checkpointing_preserves_forward_and_gradients()
