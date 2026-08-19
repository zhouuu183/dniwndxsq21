import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap import HairFast, get_parser


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"
USER_EXPORT_SPLIT = "val"  # "train", "val", "all", or "fixed"

USER_DATASET_DIR_FULL = Path("images/shape_dataset_v14_full")
USER_OUTPUT_DIR_FULL = Path("output/author_export_v14_full")
USER_VAL_SIZE_FULL = 256

USER_DATASET_DIR_SMALL = Path("images/shape_dataset_v14_small")
USER_OUTPUT_DIR_SMALL = Path("output/author_export_v14_small")
USER_VAL_SIZE_SMALL = 16

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_MAX_PAIRS = 0  # 0 means export all selected pairs
USER_SAVE_PANEL = True
USER_USE_SHADOW_CLEANUP = False
# ============================================================================


def resolve_profile_defaults():
    if USER_DATASET_PROFILE == "full":
        return {
            "dataset_dir": USER_DATASET_DIR_FULL,
            "output_dir": USER_OUTPUT_DIR_FULL,
            "val_size": USER_VAL_SIZE_FULL,
        }
    if USER_DATASET_PROFILE == "small":
        return {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "val_size": USER_VAL_SIZE_SMALL,
        }
    raise ValueError(f"Unsupported USER_DATASET_PROFILE: {USER_DATASET_PROFILE}")


PROFILE = resolve_profile_defaults()
ACTIVE_DATASET_DIR = PROFILE["dataset_dir"]
ACTIVE_OUTPUT_DIR = PROFILE["output_dir"]
ACTIVE_VAL_SIZE = PROFILE["val_size"]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_records() -> list[dict]:
    export_split = USER_EXPORT_SPLIT.lower()
    if export_split == "fixed":
        records = read_jsonl(ACTIVE_DATASET_DIR / "fixed_pairs.jsonl")
    else:
        records = read_jsonl(ACTIVE_DATASET_DIR / "manifest.jsonl")
        if export_split != "all":
            if not records:
                raise RuntimeError("manifest.jsonl is empty.")
            indices = list(range(len(records)))
            random.Random(USER_RANDOM_SEED).shuffle(indices)
            val_size = min(len(indices) - 1, max(1, int(ACTIVE_VAL_SIZE))) if len(indices) > 1 else 1
            train_records = [records[idx] for idx in indices[val_size:]] if len(indices) > val_size else list(records)
            val_records = [records[idx] for idx in indices[:val_size]]
            if export_split == "train":
                records = train_records
            elif export_split == "val":
                records = val_records
            else:
                raise ValueError(f"Unsupported USER_EXPORT_SPLIT: {USER_EXPORT_SPLIT}")

    max_pairs = int(USER_MAX_PAIRS)
    if max_pairs > 0:
        records = records[:max_pairs]
    return records


def load_tensor(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        return T.ToTensor()(image.convert("RGB"))


def ensure_chw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 4:
        image = image[0]
    return image.detach().cpu().clamp(0.0, 1.0)


def resize_like(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(image.shape[-2:]) == (height, width):
        return image
    return F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]


def save_panel(path: Path, source: torch.Tensor, reference: torch.Tensor, result: torch.Tensor) -> None:
    result = ensure_chw(result)
    height, width = result.shape[-2:]
    source = resize_like(ensure_chw(source), height, width)
    reference = resize_like(ensure_chw(reference), height, width)
    panel = torch.cat([source, reference, result], dim=2)
    save_image(panel, path)


def build_author_model() -> HairFast:
    model_args = get_parser().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    if USER_USE_SHADOW_CLEANUP:
        model_args.use_shadow_cleanup = True
    return HairFast(model_args)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    records = select_records()
    if not records:
        raise RuntimeError("No records selected for export.")

    split_tag = USER_EXPORT_SPLIT.lower()
    result_dir = ACTIVE_OUTPUT_DIR / split_tag / "results"
    panel_dir = ACTIVE_OUTPUT_DIR / split_tag / "panels"
    result_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_PANEL:
        panel_dir.mkdir(parents=True, exist_ok=True)

    model = build_author_model()
    manifest_lines = []

    for index, record in enumerate(tqdm(records, desc=f"Author export {split_tag}"), start=1):
        source_path = Path(record["source_path"])
        reference_path = Path(record["reference_path"])
        sample_id = record.get("sample_id", f"{source_path.stem}__{reference_path.stem}")
        result_name = f"{index:04d}__{sample_id}.png"
        result_path = result_dir / result_name

        result = model.swap(
            source_path,
            reference_path,
            reference_path,
            use_shadow_cleanup=USER_USE_SHADOW_CLEANUP,
        )
        save_image(ensure_chw(result), result_path)

        if USER_SAVE_PANEL:
            save_panel(panel_dir / result_name, load_tensor(source_path), load_tensor(reference_path), result)

        manifest_lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "source_path": str(source_path),
                    "reference_path": str(reference_path),
                    "result_path": result_path.as_posix(),
                },
                ensure_ascii=True,
            )
            + "\n"
        )

    with open(ACTIVE_OUTPUT_DIR / split_tag / "export_manifest.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(manifest_lines)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"export split: {split_tag}")
    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"exported records: {len(records)}")
    print(f"results dir: {result_dir}")
    if USER_SAVE_PANEL:
        print(f"panels dir: {panel_dir}")


if __name__ == "__main__":
    main()
