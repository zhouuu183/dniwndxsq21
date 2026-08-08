import torch

from losses.pp_losses_v5 import EarAwareLossBuilder
from tests.v3c_test_utils import renderer_data


def test_chroma_loss_detects_trusted_color_shift():
    data = renderer_data()
    data.update({
        "topology_skeleton_1024": data["visible_trusted_core_1024"],
        "occluded_candidate_support_1024": torch.zeros_like(data["visible_trusted_core_1024"]),
        "target_hair_core_1024": torch.zeros_like(data["visible_trusted_core_1024"]),
        "component_quality_map_1024": data["visible_trusted_core_1024"],
        "skin_edge_negative_1024": torch.zeros_like(data["visible_trusted_core_1024"]),
        "shoulder_negative_1024": torch.zeros_like(data["visible_trusted_core_1024"]),
        "neck_negative_1024": torch.zeros_like(data["visible_trusted_core_1024"]),
        "residual_gate": data["visible_trusted_core_1024"],
        "delta_rgb": torch.zeros(1, 3, 64, 64),
        "delta_low": torch.zeros(1, 3, 64, 64),
        "ear_local_roi_1024": torch.ones_like(data["visible_trusted_core_1024"]),
    })
    source = torch.zeros(1, 3, 64, 64)
    source[:, 0] = 0.9
    source[:, 1] = 0.2
    source[:, 2] = 0.1
    shifted = source.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    losses = EarAwareLossBuilder()(shifted, source, shifted, data)
    assert float(losses["tho_v3c_chroma"]) > 0.0
