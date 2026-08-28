"""Generate Stable-Hair outputs for pose-group manifests (both mode only).

Stable-Hair counterpart of HairFast's
scripts/celeba_hq_generate_baseline_pose_groups.py.  Drives the validated
Stable-Hair evaluator (celeba_hq_generate_eval_stable_hair.py, loaded from
next to this file) once per pose group, so inference, real-source export and
all safety checks stay identical to the ordinary 3000-row run.  The pipeline
is cached at module level and loaded only once for both groups.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "input").is_dir():
            return candidate
    return start.parents[1]

REPO_ROOT = find_repo_root(SCRIPT_DIR)

def load_sibling(module_name: str, filename: str):
    path = SCRIPT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing sibling module: {path}. Keep this wrapper next to "
            "celeba_hq_generate_eval_stable_hair.py."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

stable_hair = load_sibling(
    "celeba_hq_generate_eval_stable_hair",
    "celeba_hq_generate_eval_stable_hair.py",
)

# ========================= User config: edit here only ========================
USER_MODE = "both"  # Stable-Hair is two-input; only "both" is fair
USER_GROUPS = ("medium", "hard")
USER_GROUP_MANIFEST_ROOT = REPO_ROOT / "input" / "eval_pairs_pose_v1"
USER_OUTPUT_ROOT = REPO_ROOT / "output" / "celeba_hq_eval_external"
USER_RUN_PREFIX = "stable_hair_pose"
USER_EXPECTED_GROUP_COUNT = None  # None: accept each group manifest's row count
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
    if mode != "both":
        raise ValueError(
            "Stable-Hair accepts only source + one reference; pose-group FID "
            "is only fair in 'both' mode. Set USER_MODE='both'."
        )
    if not USER_GROUPS:
        raise ValueError("USER_GROUPS cannot be empty.")

    for group in USER_GROUPS:
        group = str(group).strip().lower()
        if group not in {"easy", "medium", "hard"}:
            raise ValueError(f"Unsupported pose group: {group!r}")
        manifest_path = (
            USER_GROUP_MANIFEST_ROOT / f"celeba_hq_{mode}_seed3407_pose_{group}.jsonl"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing group manifest: {manifest_path}. Run "
                "celeba_hq_group_pose_from_star.py (USER_MODE='both') first."
            )
        group_count = count_rows(manifest_path)
        if group_count <= 0:
            raise RuntimeError(f"Pose group manifest is empty: {manifest_path}")
        if USER_EXPECTED_GROUP_COUNT is not None and group_count != USER_EXPECTED_GROUP_COUNT:
            raise RuntimeError(
                f"{manifest_path} has {group_count} rows, expected "
                f"{USER_EXPECTED_GROUP_COUNT}."
            )

        # Absolute paths bypass the evaluator's CWD anchoring; checks adapt.
        stable_hair.USER_MODE = mode
        stable_hair.USER_MANIFEST_PATH = manifest_path
        stable_hair.USER_EXPECTED_SAMPLE_COUNT = group_count
        stable_hair.USER_OUTPUT_ROOT = USER_OUTPUT_ROOT
        stable_hair.USER_METHOD_RUN_NAME = f"{USER_RUN_PREFIX}_{group}"
        print(
            f"\n=== Generating Stable-Hair evaluation: {group} "
            f"({group_count} rows) ===",
            flush=True,
        )
        stable_hair.main()

if __name__ == "__main__":
    main()