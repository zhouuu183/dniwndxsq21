import torch

from models.ear_modules_v5 import PollutionSafeProposalBuilder, PollutionSafeProposalConfig


def test_local_multiscale_redetects_small_high_contrast_object():
    config = PollutionSafeProposalConfig(local_scale_size=64)
    builder = PollutionSafeProposalBuilder(config)
    source = torch.zeros(1, 3, 32, 32)
    source[:, :, 15:17, 8:10] = 1
    left = torch.zeros(1, 1, 32, 32)
    left[:, :, 4:28, :16] = 1
    zeros = torch.zeros_like(left)
    roi = {"left_ear_roi_1024": left, "right_ear_roi_1024": zeros}
    corridor = {"earlobe_anchor_1024": left, "earring_corridor_1024": left}
    candidate = builder._local_multiscale_candidate(source, roi, corridor)
    assert float(candidate.sum()) > 0

