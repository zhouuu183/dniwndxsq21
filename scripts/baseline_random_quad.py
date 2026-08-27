"""Run one untouched author-baseline HairFast example and save its four-image quad.

This script imports only the no-suffix HairFast entry point.  It does not read
any PP dataset and does not import V5/V6/SATD/earring modules.
"""

import argparse
import random
import sys
from pathlib import Path

from PIL import Image
from torchvision.transforms import functional as transform

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hair_swap import HairFast, get_parser as get_hairfast_parser


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def choose_images(image_dir: Path, selection_seed: int) -> tuple[Path, Path, Path]:
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) < 3:
        raise RuntimeError(f"Expected at least three images in {image_dir}, found {len(paths)}.")
    return tuple(random.Random(selection_seed).sample(paths, 3))


def load_for_quad(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def main(args: argparse.Namespace) -> None:
    face_path, shape_path, color_path = choose_images(args.image_dir, args.selection_seed)
    model_args = get_hairfast_parser().parse_args([])
    model = HairFast(model_args)
    final = model.swap(face_path, shape_path, color_path, seed=args.inference_seed)
    final_image = transform.to_pil_image(final.detach().cpu().clamp(0, 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "final.png"
    quad_path = args.output_dir / "quad.png"
    final_image.save(final_path)

    face = load_for_quad(face_path, final_image.size)
    shape = load_for_quad(shape_path, final_image.size)
    color = load_for_quad(color_path, final_image.size)
    quad = Image.new("RGB", (final_image.width * 4, final_image.height))
    for index, image in enumerate((face, shape, color, final_image)):
        quad.paste(image, (index * final_image.width, 0))
    quad.save(quad_path)

    print(f"face={face_path}")
    print(f"shape={shape_path}")
    print(f"color={color_path}")
    print(f"final={final_path}")
    print(f"quad={quad_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save one official HairFast baseline quad.")
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=Path("/data/coding/HairFastGAN/HairFastGAN-main/images/mix_ear"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("output/baseline_random_quad"))
    parser.add_argument("--selection_seed", type=int, default=3407)
    parser.add_argument("--inference_seed", type=int, default=3407)
    main(parser.parse_args())
