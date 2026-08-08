import inspect

from models.ear_modules_v5 import MorphologyAwareDualBranchRenderer


def test_renderer_has_no_target_rgb_input():
    parameters = inspect.signature(MorphologyAwareDualBranchRenderer.forward).parameters
    assert "target_1024_01" not in parameters
    renderer = MorphologyAwareDualBranchRenderer(crop_size=32, large_crop_size=32, base_channels=4)
    assert renderer.context_branch.enc1[0].block[0].in_channels == 17

