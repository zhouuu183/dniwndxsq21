import torch

from models.ear_modules_v5 import compute_adaptive_residual_scales


def test_high_missingness_and_confidence_increase_delta_ceiling():
    source = torch.full((2, 3, 8, 8), 0.5)
    missingness = torch.stack((torch.zeros(1, 8, 8), torch.ones(1, 8, 8)))
    confidence = torch.ones_like(missingness)
    trusted = torch.ones_like(missingness)
    zeros = torch.zeros_like(missingness)
    scales = compute_adaptive_residual_scales(missingness, confidence, source, trusted, zeros, zeros)
    assert float(scales["adaptive_low_delta_scale"][1].mean()) > float(scales["adaptive_low_delta_scale"][0].mean())
    assert float(scales["adaptive_detail_delta_scale"][1].mean()) > float(scales["adaptive_detail_delta_scale"][0].mean())

