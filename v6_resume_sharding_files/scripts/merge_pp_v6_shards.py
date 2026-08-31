"""Merge pp_gen_v6 shard directories without loading dataset tensors."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path


def policy_fingerprint(config: dict) -> str:
    """Canonicalize metadata while ignoring per-shard output coordinates."""

    comparable = copy.deepcopy(config)
    comparable.pop("identity_sha256", None)
    generator_args = comparable.get("generator_args")
    if isinstance(generator_args, dict):
        generator_args.pop("output", None)
        generator_args.pop("shard_index", None)
    return json.dumps(comparable, ensure_ascii=True, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = [path.resolve() for path in args.inputs]
    output = args.output.resolve()
    if output in inputs:
        raise ValueError("--output must be different from every input shard")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    first_config = None
    first_policy = None
    part_number = 1
    for input_dir in inputs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Shard directory does not exist: {input_dir}")
        config_path = input_dir / "dataset_config.json"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                shard_config = json.load(handle)
            if first_config is None:
                first_config = shard_config
                first_policy = policy_fingerprint(shard_config)
            elif policy_fingerprint(shard_config) != first_policy:
                raise RuntimeError(
                    "Shard dataset policies differ; refusing to merge parts generated "
                    "with different code, checkpoints, or settings."
                )
        elif first_config is not None:
            raise RuntimeError(
                f"Missing dataset_config.json in shard directory: {input_dir}"
            )

        def part_number_from_path(path: Path) -> int:
            match = re.fullmatch(r"pp_part_(\d+)\.dataset", path.name)
            if match is None:
                raise ValueError(f"Unexpected dataset part name: {path.name}")
            return int(match.group(1))

        parts = sorted(input_dir.glob("pp_part_*.dataset"), key=part_number_from_path)
        if not parts:
            raise FileNotFoundError(f"No dataset parts found under {input_dir}")
        for part in parts:
            shutil.copyfile(part, output / f"pp_part_{part_number}.dataset")
            part_number += 1

    if first_config is not None:
        first_config["merged_shards"] = len(inputs)
        first_config["merged_part_count"] = part_number - 1
        with (output / "dataset_config.json").open("w", encoding="utf-8") as handle:
            json.dump(first_config, handle, ensure_ascii=True, indent=2, sort_keys=True)
    print(f"Merged {part_number - 1} dataset parts into {output}")


if __name__ == "__main__":
    main()
