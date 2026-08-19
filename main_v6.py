import argparse
import os
import sys
from pathlib import Path

from torchvision.utils import save_image
from tqdm.auto import tqdm

from hair_swap_v6 import HairFast_v6, get_parser_v6


def main(model_args, args):
    hair_fast = HairFast_v6(model_args)

    experiments: list[str | tuple[str, str, str]] = []
    if args.file_path is not None:
        with open(args.file_path, "r", encoding="utf-8") as file:
            experiments.extend(file.readlines())

    if all(path is not None for path in (args.face_path, args.shape_path, args.color_path)):
        experiments.append((args.face_path, args.shape_path, args.color_path))

    for exp in tqdm(experiments):
        if isinstance(exp, str):
            file_1, file_2, file_3 = exp.split()
        else:
            file_1, file_2, file_3 = exp

        face_path = args.input_dir / file_1
        shape_path = args.input_dir / file_2
        color_path = args.input_dir / file_3

        base_name = "_".join([path.stem for path in (face_path, shape_path, color_path)])
        exp_name = base_name if model_args.save_all else None

        if isinstance(exp, str) or args.result_path is None:
            os.makedirs(args.output_dir, exist_ok=True)
            output_image_path = args.output_dir / f"{base_name}.png"
        else:
            os.makedirs(args.result_path.parent, exist_ok=True)
            output_image_path = args.result_path

        final_image = hair_fast.swap(face_path, shape_path, color_path, benchmark=args.benchmark, exp_name=exp_name)
        save_image(final_image, output_image_path)


if __name__ == "__main__":
    model_parser = get_parser_v6()
    parser = argparse.ArgumentParser(description="HairFast v6 evaluate")
    parser.add_argument("--input_dir", type=Path, default=Path(""), help="Directory that contains the input images")
    parser.add_argument("--benchmark", action="store_true", help="Measure runtime during the session")
    parser.add_argument("--file_path", type=Path, default=None, help='Text file with "face shape color" triplets')
    parser.add_argument("--output_dir", type=Path, default=Path("output_v6"), help="Directory for final images")
    parser.add_argument("--face_path", type=Path, default=None, help="Single face/source image path")
    parser.add_argument("--shape_path", type=Path, default=None, help="Single shape/reference image path")
    parser.add_argument("--color_path", type=Path, default=None, help="Single color/reference image path")
    parser.add_argument("--result_path", type=Path, default=None, help="Optional exact output path")

    args, unknown1 = parser.parse_known_args()
    model_args, unknown2 = model_parser.parse_known_args()

    unknown_args = set(unknown1) & set(unknown2)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)
        print("\nExpected arguments for the model:", file=file_)
        model_parser.print_help(file=file_)
        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)
        sys.exit(1)

    main(model_args, args)
