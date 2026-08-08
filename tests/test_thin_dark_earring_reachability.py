import torch

from models.ear_modules_v5 import compute_adaptive_residual_scales


def test_dark_thin_high_missingness_has_reachable_low_frequency_delta():
    source = torch.zeros(1, 3, 8, 8)
    ones = torch.ones(1, 1, 8, 8)
    zeros = torch.zeros_like(ones)
    scales = compute_adaptive_residual_scales(ones, ones, source, zeros, zeros, ones)
    assert float(scales["adaptive_low_delta_scale"].max()) >= 0.15
    assert float(scales["adaptive_detail_delta_scale"].max()) >= 0.31

