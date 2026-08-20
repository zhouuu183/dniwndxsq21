"""Acceptance decision tests for V2.24 boundary diagnostics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.v224_metrics import classify_v224


def _passing_summary() -> dict[str, float]:
    return {
        "median_v223_old_selective_full_stat_error": 4.0,
        "median_v224_selective_full_stat_error": 4.1,
        "median_full_color_retention": 0.95,
        "median_ab_retention": 0.90,
        "median_l_progress": 0.80,
        "median_v224_edge_transfer_full": 0.50,
        "median_v224_edge_transfer_ab": 0.55,
        "edge_halo_ratio": 0.40,
    }


def main() -> None:
    summary = _passing_summary()
    assert classify_v224(
        summary, parity_max_diff=0.0, outside_max_delta=0.0,
        hard_protect_max_delta=0.0,
    )["decision"] == "V224_BOUNDARY_PASS"
    assert classify_v224(
        summary, parity_max_diff=2e-6, outside_max_delta=0.0,
        hard_protect_max_delta=0.0,
    )["decision"] == "V224_PROTECTION_OR_PARITY_BUG"
    weak = {**summary, "median_v224_edge_transfer_full": 0.30}
    assert classify_v224(
        weak, parity_max_diff=0.0, outside_max_delta=0.0,
        hard_protect_max_delta=0.0,
    )["decision"] == "V224_BOUNDARY_WEAK"
    halo = {**summary, "edge_halo_ratio": 0.61}
    assert classify_v224(
        halo, parity_max_diff=0.0, outside_max_delta=0.0,
        hard_protect_max_delta=0.0,
    )["decision"] == "V224_HALO_FAIL"
    print("test_v224_metrics: PASS")


if __name__ == "__main__":
    main()
