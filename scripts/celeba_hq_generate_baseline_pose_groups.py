"""Generate baseline outputs for pose-group manifests.

Run celeba_hq_group_pose_from_star.py first.  This wrapper calls the existing
baseline evaluator once for each requested group, so the author baseline,
checkpoint paths, RGB conversion, real-source export, and safety checks remain
identical to the ordinary baseline FID run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.celeba_hq_generate_baseline_fid as baseline


# ========================= User config: edit here only ========================
USER_MODE = "full"  # mode used to create the original fixed manifest
USER_GROUPS = ("medium", "hard")
USER_GROUP_MANIFEST_ROOT = Path("input/eval_pairs_pose_v1")
USER_OUTPUT_ROOT = Path("output/celeba_hq_baseline_pose_fid")
USER_RUN_PREFIX = "author_baseline_pose"
USER_EXPECTED_GROUP_COUNT = None  # None: accept the count written by each group manifest
# ==============================================================================


def count_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"Non-object row in group manifest: {path}")
                count += 1
    return count


def main() -> None:
    mode = USER_MODE.strip().lower()
    if mode not in {"both", "full"}:
        raise ValueError("USER_MODE must be 'both' or 'full'.")
    if not USER_GROUPS:
        raise ValueError("USER_GROUPS cannot be empty.")

    repo_root = REPO_ROOT
    manifest_root = (
        USER_GROUP_MANIFEST_ROOT
        if USER_GROUP_MANIFEST_ROOT.is_absolute()
        else repo_root / USER_GROUP_MANIFEST_ROOT
    )
    output_root = (
        USER_OUTPUT_ROOT
        if USER_OUTPUT_ROOT.is_absolute()
        else repo_root / USER_OUTPUT_ROOT
    )

    for group in USER_GROUPS:
        group = str(group).strip().lower()
        if group not in {"easy", "medium", "hard"}:
            raise ValueError(f"Unsupported pose group: {group!r}")
        manifest_path = manifest_root / f"celeba_hq_{mode}_seed3407_pose_{group}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing group manifest: {manifest_path}. Run "
                "celeba_hq_group_pose_from_star.py first."
            )
        group_count = count_rows(manifest_path)
        if group_count <= 0:
            raise RuntimeError(f"Pose group manifest is empty: {manifest_path}")
        if USER_EXPECTED_GROUP_COUNT is not None and group_count != USER_EXPECTED_GROUP_COUNT:
            raise RuntimeError(
                f"{manifest_path} has {group_count} rows, expected "
                f"{USER_EXPECTED_GROUP_COUNT}."
            )

        # Configure the already-validated baseline evaluator for this group.
        baseline.USER_MODE = mode
        baseline.USER_MANIFEST_PATH = manifest_path
        baseline.USER_EXPECTED_SAMPLE_COUNT = group_count
        baseline.USER_OUTPUT_ROOT = output_root
        baseline.USER_RUN_NAME = f"{USER_RUN_PREFIX}_{group}"
        baseline.USER_SKIP_INVALID_INPUTS = False
        print(f"\n=== Generating author baseline: {group} ({group_count} rows) ===", flush=True)
        baseline.main()


if __name__ == "__main__":
    main()
