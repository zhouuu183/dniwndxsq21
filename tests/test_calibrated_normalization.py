import torch

from models.ear_modules_v5 import calibrated_normalize_in_rois


def test_low_signal_roi_is_zero_and_strong_signal_is_preserved():
    roi = torch.ones(1, 1, 32, 32)
    weak = torch.full_like(roi, 0.01) + 1e-4 * torch.rand_like(roi)
    normalized, diagnostics = calibrated_normalize_in_rois(
        weak, (roi,), min_dynamic_range=0.01, min_absolute_peak=0.05
    )
    assert torch.count_nonzero(normalized) == 0
    assert float(diagnostics["cue_valid"].max()) == 0.0
    strong = torch.zeros_like(roi)
    strong[:, :, 8:24, 15:17] = 0.5
    normalized, diagnostics = calibrated_normalize_in_rois(
        strong, (roi,), min_dynamic_range=0.01, min_absolute_peak=0.05
    )
    assert float(normalized.max()) == 1.0
    assert float(diagnostics["cue_valid"].max()) == 1.0

