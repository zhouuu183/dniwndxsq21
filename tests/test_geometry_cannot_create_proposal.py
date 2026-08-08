import torch

from models.ear_modules_v5 import ImageSideEarROIBuilder, PollutionSafeProposalBuilder


def test_geometry_without_visual_evidence_cannot_create_candidate():
    source = torch.full((1, 3, 64, 64), 0.5)
    parsing = torch.zeros(1, 1, 64, 64, dtype=torch.long)
    parsing[:, :, 24:40, 10:18] = 7
    parsing[:, :, 24:40, 46:54] = 8
    roi = ImageSideEarROIBuilder()(source, parsing)
    proposal = PollutionSafeProposalBuilder()(source, parsing, roi)
    assert torch.count_nonzero(proposal["weak_candidate_1024"]) == 0
    assert torch.count_nonzero(proposal["strong_candidate_1024"]) == 0

