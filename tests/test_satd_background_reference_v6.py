from argparse import Namespace

import pytest
import torch

pytest.importorskip("clip")

from models.postprocess_v6 import PostProcessModelV6
from models.ear_modules_v5 import RAW_FACE_SURFACE_LABELS


class _AllBackgroundParser:
    parse_size = 64

    def parse(self, image, out_size):
        labels = torch.zeros(
            image.size(0), 1, out_size[0], out_size[1],
            dtype=torch.long, device=image.device,
        )
        # A small face patch must stay read-only even when it overlaps the
        # broad cleanup support supplied by the SATD branch.
        labels[:, :, 4:16, 4:16] = RAW_FACE_SURFACE_LABELS[0]
        return labels


def _model():
    model = PostProcessModelV6.__new__(PostProcessModelV6)
    model.args = Namespace(
        satd_background_cleanup_dilate=5,
        satd_background_exclude_dilate=2,
        satd_background_earring_exclude_dilate=2,
        satd_background_target_hair_protect_dilate=2,
        satd_background_hair_edge_dilate=2,
        satd_background_hair_edge_support_dilate=3,
        satd_background_reference_size=64,
        satd_background_reference_kernel=31,
        satd_background_reference_sigma=8.0,
        satd_background_reference_weight=1.0,
        satd_background_residual_strength=1.0,
        satd_background_alpha_feather=1,
    )
    model.parsing_helper = _AllBackgroundParser()
    return model


def test_background_reference_restores_shadow_and_protects_face():
    height = width = 64
    clean = torch.tensor([0.10, 0.78, 0.20]).view(1, 3, 1, 1)
    shadow = torch.tensor([0.08, 0.30, 0.12]).view(1, 3, 1, 1)
    base = clean.expand(1, 3, height, width).clone()
    support = torch.zeros(1, 1, height, width)
    support[:, :, 24:48, 20:44] = 1.0
    base = base * (1.0 - support) + shadow * support
    image = base.clone()

    result = _model()._apply_satd_background_residual(
        image * 2.0 - 1.0,
        {
            "satd_background_highres_01": base,
            "satd_background_reference_highres_01": base,
            "M_remove": support,
        },
    )
    result = (result + 1.0) * 0.5

    # The supported shadow should move toward the unoccluded green field.
    center_before = image[:, 1, 32, 32]
    center_after = result[:, 1, 32, 32]
    assert center_after.item() > center_before.item() + 0.10
    assert torch.allclose(
        result[:, :, 0:4, 0:4],
        image[:, :, 0:4, 0:4],
        atol=1e-5,
    )
    # The semantic face patch is outside the permitted write area.
    assert torch.allclose(
        result[:, :, 4:16, 4:16],
        image[:, :, 4:16, 4:16],
        atol=1e-5,
    )
