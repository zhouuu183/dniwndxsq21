from __future__ import annotations

import torch

from losses.pp_losses_v5 import masked_color_statistics


def test_color_statistics_zero_mask_has_finite_gradient():
    value = torch.rand(1, 2, 8, 8, requires_grad=True)
    target = torch.rand(1, 2, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    loss = masked_color_statistics(value, target, mask)
    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert torch.isfinite(value.grad).all()
    assert torch.equal(value.grad, torch.zeros_like(value.grad))


def test_color_statistics_constant_region_has_finite_gradient():
    value = torch.full((1, 2, 8, 8), 0.4, requires_grad=True)
    target = torch.full((1, 2, 8, 8), 0.2)
    mask = torch.ones(1, 1, 8, 8)
    loss = masked_color_statistics(value, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(value.grad).all()


if __name__ == "__main__":
    test_color_statistics_zero_mask_has_finite_gradient()
    test_color_statistics_constant_region_has_finite_gradient()
