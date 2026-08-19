import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build M_remove = source_hair * (1 - target_hair).")
    parser.add_argument("source_mask", type=Path, help="Black/white source hair mask.")
    parser.add_argument("target_mask", type=Path, help="Black/white target/rotated-reference hair mask.")
    parser.add_argument("--output", type=Path, default=Path("output/M_remove.png"))
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Pixel threshold for binarizing masks. Values > threshold are treated as hair.",
    )
    return parser


def load_binary_mask(path: Path, size: tuple[int, int] | None, threshold: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")

    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.NEAREST)
    return (np.asarray(image) > threshold).astype(np.uint8)


def main() -> None:
    args = build_parser().parse_args()

    source_image = Image.open(args.source_mask).convert("L")
    source_size = source_image.size
    source = (np.asarray(source_image) > args.threshold).astype(np.uint8)
    target = load_binary_mask(args.target_mask, source_size, args.threshold)

    m_remove = source * (1 - target)
    output = (m_remove * 255).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="L").save(args.output)
    print(f"Saved M_remove: {args.output}")


if __name__ == "__main__":
    main()
