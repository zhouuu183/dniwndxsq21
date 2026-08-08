import torch

from models.ear_modules_v5 import ComponentSafetyRollback
from tests.v3c_test_utils import renderer_data


def test_shoulder_dark_component_triggers_rollback():
    rollback = ComponentSafetyRollback()
    data = renderer_data()
    gate = data["visible_trusted_core_1024"].clone()
    data["shoulder_negative_1024"] = gate.clone()
    data["component_quality_map_1024"] = gate.clone()
    raw = torch.full((1, 3, 64, 64), 0.5)
    provisional = raw - 0.15 * gate
    final, mask, reasons = rollback(raw, provisional, gate, data)
    assert float(mask.sum()) > 0
    assert torch.equal(final, raw)
    assert any("shoulder_overlap" in reason for reason in reasons[0])
