import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.v223_metrics import (
    aggregate_v223_records,
    build_v223_normal_color_manifest,
    classify_v223,
    full_stat_error,
    strict_masked_mean_per_sample,
)


def record(index, **updates):
    value = {
        "sample_id": f"sample-{index}",
        "safe_fraction": 0.8,
        "pseudo_to_reference_ab_error": 1.0,
        "pseudo_to_reference_hue_error": 3.0,
        "ref_base_ab_distance": 2.0,
        "reference_chroma_magnitude": 12.0,
        "base_ref_l_error": 30.0,
        "anchor_ref_l_error": 3.0,
        "selective_ref_l_error": 5.0,
        "ref_base_l_distance": 30.0,
        "base_ref_ab_error": 13.0,
        "anchor_ref_ab_error": 4.0,
        "base_full_stat_error": 24.0,
        "anchor_full_stat_error": 5.0,
        "selective_full_stat_error": 7.0,
        "pseudo_deltaE76": 6.0,
        "full_color_retention": 0.85,
        "raw_full_anchor_positive_gain": True,
        "ab_retention": 0.90,
        "raw_anchor_positive_gain": True,
        "l_progress": 0.8,
        "l_progress_valid": True,
        "selective_ref_ab_error": 4.0,
        "selective_ref_hue_error": 5.0,
        "selective_ref_chroma_error": 3.0,
        "full_color_improvement_ratio": 0.7,
        "raw_anchor_edge_luma_excess_mean": 5.0,
        "edge_luma_excess_mean": 2.0,
        "edge_hf_excess": 1.0,
        "edge_transfer_ratio": 0.55,
        "face_keep_l1": 0.0,
        "outer_bg_keep_l1": 0.0,
        "outside_hair_max_abs_delta": 0.0,
        "hard_protect_max_abs_delta": 0.0,
        "mean_luma_transfer_weight": 0.8,
        "mean_chroma_transfer_weight": 0.9,
    }
    value.update(updates)
    return value


def main():
    value = torch.tensor(
        [
            [[[1.0, 3.0], [5.0, 7.0]]],
            [[[10.0, 20.0], [30.0, 40.0]]],
            [[[2.0, 4.0], [8.0, 16.0]]],
        ]
    )
    mask = torch.tensor(
        [
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 1.0]]],
            [[[1.0, 1.0], [0.0, 0.0]]],
        ]
    )
    vectorized = strict_masked_mean_per_sample(value, mask)
    expected = torch.tensor([3.0, 30.0, 3.0])
    assert torch.allclose(vectorized, expected)
    for index in range(3):
        selected = value[index][mask[index].expand_as(value[index]) > 0]
        assert abs(float(selected.mean()) - float(vectorized[index])) < 1e-6
    squeezed = strict_masked_mean_per_sample(value[:, 0], mask)
    assert torch.allclose(squeezed, expected)

    assert torch.allclose(
        full_stat_error(torch.tensor([3.0]), torch.tensor([4.0])),
        torch.tensor([5.0]),
    )
    records = [record(0), record(1, selective_ref_l_error=30.0)]
    manifest = build_v223_normal_color_manifest(
        [
            record(0, reference_chroma_magnitude=10.0),
            record(1, base_full_stat_error=20.0, reference_chroma_magnitude=11.0),
            record(2, base_full_stat_error=19.0, reference_chroma_magnitude=9.0),
        ],
        0.35,
    )
    assert manifest["status"] == "OK"
    assert manifest["normal_count"] == 1
    assert manifest["entries"][2]["normal_color"]
    assert not manifest["entries"][2]["near_no_edit"]
    summary = aggregate_v223_records(records)
    assert summary["count"] == 2
    assert abs(float(summary["median_selective_ref_l_error"]) - 17.5) < 1e-6
    assert classify_v223(summary)["decision"] == "V223_FULL_COLOR_PASS"
    failed = aggregate_v223_records(
        [record(0, selective_ref_l_error=30.0, l_progress=0.1)]
    )
    assert classify_v223(failed)["decision"] == "V223_LUMA_RETENTION_FAIL"
    print("v2.23 metrics tests passed")


if __name__ == "__main__":
    main()
