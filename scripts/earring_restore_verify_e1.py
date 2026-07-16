from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def list_required_images(root: Path, label: str) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"Cannot find {label}: {root}")
    files = sorted(
        path.name for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise RuntimeError(f"No jpg/jpeg/png images found under {label}: {root}")
    return files


def resolve_pairing_mode(mode: str, donor_dir: Path | None) -> str:
    if mode != "auto":
        return mode
    return "random" if donor_dir is not None else "self"


def build_triplets(
    source_dir: Path,
    source_files: list[str],
    donor_dir: Path | None,
    donor_files: list[str],
    mode: str,
    limit: int,
    seed: int,
) -> list[tuple[Path, Path, Path, str]]:
    count = len(source_files) if limit <= 0 else min(limit, len(source_files))
    rng = random.Random(seed)
    triplets: list[tuple[Path, Path, Path, str]] = []

    for index, source_name in enumerate(source_files[:count]):
        source_path = source_dir / source_name
        if mode == "self" or donor_dir is None:
            shape_path = source_path
            color_path = source_path
            donor_tag = "self"
        elif mode == "by_index":
            donor_name = donor_files[index % len(donor_files)]
            shape_path = donor_dir / donor_name
            color_path = shape_path
            donor_tag = Path(donor_name).stem
        else:
            donor_name = rng.choice(donor_files)
            shape_path = donor_dir / donor_name
            color_path = shape_path
            donor_tag = Path(donor_name).stem

        exp_name = f"{source_path.stem}__{donor_tag}"
        triplets.append((source_path, shape_path, color_path, exp_name))

    return triplets


def unwrap_result(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick validation for the _e1 direct earring restore path.",
    )
    parser.add_argument("--source_dir", type=Path, default=Path("images/ear"))
    parser.add_argument("--donor_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("output/earring_restore_e1"))
    parser.add_argument("--limit", type=int, default=8, help="0 means all source images.")
    parser.add_argument("--pairing", choices=("auto", "self", "by_index", "random"), default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--align", action="store_true", help="Use face alignment for uncropped raw photos.")
    parser.add_argument("--no_debug", action="store_true", help="Only save final images, not intermediate masks.")

    args, model_unknown = parser.parse_known_args()

    from hair_swap_e1 import HairFast, get_parser
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import functional as TVF
    from torchvision.utils import save_image

    model_parser = get_parser()
    model_args = model_parser.parse_args(model_unknown)

    source_files = list_required_images(args.source_dir, "source_dir")
    donor_files: list[str] = []
    if args.donor_dir is not None:
        donor_files = list_required_images(args.donor_dir, "donor_dir")

    pairing = resolve_pairing_mode(args.pairing, args.donor_dir)
    triplets = build_triplets(
        args.source_dir,
        source_files,
        args.donor_dir,
        donor_files,
        pairing,
        args.limit,
        args.seed,
    )

    model_args.use_earring_direct_restore = True
    model_args.save_all = not args.no_debug
    model_args.earring_restore_debug = not args.no_debug
    model_args.save_all_dir = args.output_dir / "debug"

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / "triplets.txt", "w", encoding="utf-8") as handle:
        for source_path, shape_path, color_path, exp_name in triplets:
            handle.write(f"{exp_name}\t{source_path}\t{shape_path}\t{color_path}\n")

    hair_fast = HairFast(model_args)

    def tensor_to_image_01(tensor):
        tensor = unwrap_result(tensor)
        if tensor.dim() == 4:
            tensor = tensor[0]
        tensor = tensor.detach().float().cpu()
        if tensor.max().item() > 2.0:
            tensor = tensor / 255.0
        if tensor.min().item() < -0.05:
            tensor = (tensor + 1.0) / 2.0
        return tensor.clamp(0, 1)

    def load_panel_input(path: Path, size: tuple[int, int]):
        image = Image.open(path).convert("RGB")
        tensor = TVF.to_tensor(image).unsqueeze(0)
        tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        return tensor[0].clamp(0, 1)

    def build_panel(source_path: Path, donor_path: Path, author_result, restored_result):
        author_image = tensor_to_image_01(author_result)
        restored_image = tensor_to_image_01(restored_result)
        panel_size = tuple(restored_image.shape[-2:])
        source_image = load_panel_input(source_path, panel_size)
        donor_image = load_panel_input(donor_path, panel_size)
        author_image = F.interpolate(author_image.unsqueeze(0), size=panel_size, mode="bilinear", align_corners=False)[0]
        return torch.cat(
            [
                source_image,
                donor_image,
                author_image.clamp(0, 1),
                restored_image.clamp(0, 1),
            ],
            dim=2,
        )

    for source_path, shape_path, color_path, exp_name in tqdm(triplets, desc="Validate earring restore e1"):
        model_args.save_all = False
        author_result = hair_fast.swap(
            source_path,
            shape_path,
            color_path,
            align=args.align,
            exp_name=None,
            use_earring_direct_restore=False,
        )

        model_args.save_all = not args.no_debug
        restored_result = hair_fast.swap(
            source_path,
            shape_path,
            color_path,
            align=args.align,
            exp_name=exp_name,
            use_earring_direct_restore=True,
        )
        panel = build_panel(source_path, shape_path, author_result, restored_result)
        save_image(panel, final_dir / f"{exp_name}.png")


if __name__ == "__main__":
    main()
