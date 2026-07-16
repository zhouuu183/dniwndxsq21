from __future__ import annotations

import random
from pathlib import Path

from torch.utils.data import Dataset

import torch

from utils.spsa_precompute_v13 import load_prior_npz, read_image_tensor, read_manifest


def split_manifest_records(
    records: list[dict[str, str]],
    *,
    val_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not records:
        raise ValueError("Manifest is empty.")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_size = max(1, min(int(val_size), len(shuffled)))
    return shuffled[val_size:], shuffled[:val_size]


class SPSAAlignmentDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        color_from_reference: bool = True,
    ):
        super().__init__()
        self.records = read_manifest(manifest_path)
        self.color_from_reference = bool(color_from_reference)
        self._prior_cache_path: str | None = None
        self._prior_cache: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _load_prior(self, path: str | Path) -> dict[str, torch.Tensor]:
        path = str(path)
        if self._prior_cache_path != path:
            self._prior_cache_path = path
            self._prior_cache = load_prior_npz(path)
        return {key: value.clone() for key, value in self._prior_cache.items()}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        source = read_image_tensor(record["source_path"])
        reference = read_image_tensor(record["reference_path"])
        color_path = record["reference_path"] if self.color_from_reference else record["color_path"]
        color = read_image_tensor(color_path)
        prior = self._load_prior(record["prior_path"])
        sample: dict[str, torch.Tensor | str] = {
            "sample_id": record["sample_id"],
            "source_path": record["source_path"],
            "reference_path": record["reference_path"],
            "color_path": color_path,
            "source": source,
            "reference": reference,
            "color": color,
        }
        sample.update(prior)
        return sample
