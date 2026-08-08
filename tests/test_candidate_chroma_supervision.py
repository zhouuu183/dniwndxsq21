import torch

from losses.pp_losses_v5 import EarAwareLossBuilder
from models.ear_modules_v5 import EAR_UNCERTAIN
from tests.v3c_test_utils import renderer_data


def test_uncertain_candidate_receives_source_chroma_supervision():
    data = renderer_data()
    candidate = data["visible_trusted_core_1024"].clone()
    data["visible_trusted_core_1024"].zero_()
    data["visible_solid_interior_1024"].zero_()
    data["visible_candidate_core_1024"] = candidate
    data["candidate_support_1024"] = candidate
    data["trusted_support_1024"].zero_()
    data["component_restore_confidence_map_1024"] = candidate
    data["color_supervision_mask_1024"] = candidate
    data["earring_status"] = torch.tensor([EAR_UNCERTAIN])
    source = torch.zeros(1, 3, 64, 64)
    source[:, 0] = 1
    gray = torch.full_like(source, 0.5)
    losses = EarAwareLossBuilder()(gray, source, gray, data, stage=1)
    assert float(losses["tho_v3c_chroma"]) > 0

