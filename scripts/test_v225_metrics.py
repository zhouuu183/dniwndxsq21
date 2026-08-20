import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.v225_metrics import classify_v225


def main():
    summary = {
        "median_v224_selective_full_stat_error": 4.0,
        "median_selective_full_stat_error": 4.1,
        "median_full_color_retention": 0.95,
        "median_ab_retention": 0.9,
        "median_l_progress": 0.8,
        "mean_edge_reference_parallel_mag": 5.0,
        "median_v225_edge_transfer_ab": 0.5,
        "median_v225_edge_transfer_l": 0.5,
        "median_v225_edge_transfer_full": 0.5,
        "median_edge_reference_ab_progress": 0.6,
        "median_edge_reference_l_progress": 0.6,
        "mean_edge_ab_direction_agreement": 0.9,
        "mean_edge_l_direction_agreement": 0.9,
        "edge_halo_ratio": 0.2,
    }
    assert classify_v225(summary, parity_max_diff=0, outside_max_delta=0, hard_protect_max_delta=0)["decision"] == "V225_REFERENCE_BOUNDARY_PASS"
    assert classify_v225(summary, parity_max_diff=2e-6, outside_max_delta=0, hard_protect_max_delta=0)["decision"] == "V225_PROTECTION_OR_PARITY_BUG"
    print("test_v225_metrics: PASS")


if __name__ == "__main__":
    main()
