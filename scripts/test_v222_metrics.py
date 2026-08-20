import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.v222_metrics import (
    aggregate_v222_records,
    classify_v222_pretrain,
    v222_checkpoint_score,
)


def record(index, *, retention=0.85, outside=0.0, hard=0.0, edge=0.2, raw_edge=1.0):
    return {
        "sample_id": f"sample-{index}",
        "base_to_ref_ab": 10.0,
        "anchor_to_ref_ab": 4.0,
        "selective_to_ref_ab": 4.9,
        "color_improvement_ratio": 0.51,
        "raw_anchor_positive_gain": True,
        "color_retention_ratio": retention,
        "parallel_progress": 0.8,
        "orthogonal_error": 1.0,
        "selective_to_ref_hue": 4.0,
        "selective_to_ref_chroma": 2.0,
        "anchor_edge_luma_excess_mean": raw_edge,
        "anchor_face_keep_l1": 0.08,
        "anchor_outer_bg_keep_l1": 0.07,
        "edge_luma_excess_mean": edge,
        "edge_luma_excess_fraction": 0.1,
        "edge_hf_excess": 0.2,
        "face_keep_l1": 0.0,
        "outer_bg_keep_l1": 0.0,
        "hard_protect_max_abs_delta": hard,
        "outside_hair_max_abs_delta": outside,
        "parallel_gain": 1.0,
        "orth_keep": 0.1,
        "boundary_strength": 0.22,
        "luma_strength": 0.12,
        "mean_A_core": 0.7,
        "mean_A_edge": 0.1,
        "mean_halo_gate": 0.8,
        "overshoot_fraction": 0.1,
        "negative_parallel_fraction": 0.02,
    }


def main():
    records = [record(index) for index in range(4)]
    summary = aggregate_v222_records(records, {item["sample_id"] for item in records})
    assert summary["count"] == 4
    assert summary["median_color_retention_ratio"] == 0.85
    assert v222_checkpoint_score(summary) > 0.0
    decision = classify_v222_pretrain(summary)
    assert decision["decision"] == "SELECTIVE_PRETRAIN_PROMISING"
    assert not decision["abort"]

    structural = aggregate_v222_records(
        [record(0, outside=1e-3)], {"sample-0"}
    )
    structural_decision = classify_v222_pretrain(structural)
    assert structural_decision["decision"] == "MASK_PROTECTION_BUG"
    assert structural_decision["abort"]

    weak = aggregate_v222_records(
        [record(index, retention=0.4) for index in range(4)],
        {f"sample-{index}" for index in range(4)},
    )
    weak_decision = classify_v222_pretrain(weak)
    assert weak_decision["decision"] == "SELECTIVE_COLOR_LOSS_TOO_LARGE"
    assert not weak_decision["abort"]
    print("v2.22 metrics tests passed")


if __name__ == "__main__":
    main()
