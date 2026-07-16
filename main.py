import argparse
import os
import sys
from pathlib import Path

from torchvision.utils import save_image
from tqdm.auto import tqdm

from hair_swap import HairFast, get_parser


# User config area:
# 1. Set `enabled=True` to run without a long command line.
# 2. Fill `experiments` with one or more sample groups.
# 3. Each sample group is a Python dict.
# 4. Different sample groups are separated by commas.
# 5. Inside one sample group, fill `face_name`, `shape_name`, `color_name` separately.
# 6. Each image can come from a different directory.
#
# Example:
# 'experiments': [
#     {
#         'face_dir': Path('input/faces'),
#         'shape_dir': Path('input/shapes'),
#         'color_dir': Path('input/colors'),
#         'face_name': 'face_01.png',
#         'shape_name': 'shape_01.png',
#         'color_name': 'color_01.png',
#         'result_path': Path('output/result_01.png'),
#     },
#     {
#         'face_dir': Path('input/faces'),
#         'shape_dir': Path('input/shapes'),
#         'color_dir': Path('input/colors'),
#         'face_name': 'face_02.png',
#         'shape_name': 'shape_02.png',
#         'color_name': 'color_02.png',
#         'result_path': Path('output/result_02.png'),
#     },
# ]
USER_CONFIG = {
    'enabled': False,
    'experiments': [
        {
            'face_dir': Path('input'),
            'shape_dir': Path('input'),
            'color_dir': Path('input'),
            'face_name': '6.png',
            'shape_name': '7.png',
            'color_name': '8.png',
            'result_path': Path('output/result_01.png'),
        },
    ],
    'save_all': True,
    'save_all_dir': Path('output'),
    'benchmark': False,
}


def _resolve_config_path(directory, file_name):
    if file_name in (None, ''):
        return None

    path = Path(file_name)
    if path.is_absolute() or directory in (None, ''):
        return path
    return Path(directory) / path


def _normalize_user_experiments():
    experiments = USER_CONFIG.get('experiments', [])
    if not experiments:
        raise ValueError("USER_CONFIG['experiments'] is empty.")

    normalized = []
    for idx, experiment in enumerate(experiments, start=1):
        if not isinstance(experiment, dict):
            raise TypeError(
                f"USER_CONFIG['experiments'][{idx - 1}] must be a dict. "
                "Different sample groups should be separated by commas."
            )

        face_path = _resolve_config_path(experiment.get('face_dir'), experiment.get('face_name'))
        shape_path = _resolve_config_path(experiment.get('shape_dir'), experiment.get('shape_name'))
        color_path = _resolve_config_path(experiment.get('color_dir'), experiment.get('color_name'))
        result_path = _resolve_config_path(None, experiment.get('result_path'))

        missing = [name for name, path in (
            ('face_name', face_path),
            ('shape_name', shape_path),
            ('color_name', color_path),
            ('result_path', result_path),
        ) if path is None]
        if missing:
            raise ValueError(
                f"USER_CONFIG['experiments'][{idx - 1}] is missing values for: {missing}"
            )

        normalized.append({
            'face_path': face_path,
            'shape_path': shape_path,
            'color_path': color_path,
            'result_path': result_path,
        })

    return normalized


def apply_user_config(args, model_args):
    if not USER_CONFIG.get('enabled', False):
        return args, model_args

    args.file_path = None
    args.input_dir = Path('')
    args.face_path = None
    args.shape_path = None
    args.color_path = None
    args.result_path = None
    args.user_experiments = _normalize_user_experiments()
    args.benchmark = bool(USER_CONFIG.get('benchmark', args.benchmark))

    model_args.save_all = bool(USER_CONFIG.get('save_all', model_args.save_all))
    model_args.save_all_dir = _resolve_config_path(None, USER_CONFIG.get('save_all_dir')) or model_args.save_all_dir

    return args, model_args


def main(model_args, args):
    hair_fast = HairFast(model_args)

    experiments: list[str | tuple[str, str, str]] = []
    configured_outputs: list[Path | None] = []
    if args.file_path is not None:
        with open(args.file_path, 'r') as file:
            experiments.extend(file.readlines())
        configured_outputs.extend([None] * len(experiments))

    if all(path is not None for path in (args.face_path, args.shape_path, args.color_path)):
        experiments.append((args.face_path, args.shape_path, args.color_path))
        configured_outputs.append(args.result_path)

    for experiment in getattr(args, 'user_experiments', []):
        experiments.append((experiment['face_path'], experiment['shape_path'], experiment['color_path']))
        configured_outputs.append(experiment['result_path'])

    for exp, configured_output in tqdm(list(zip(experiments, configured_outputs))):
        if isinstance(exp, str):
            file_1, file_2, file_3 = exp.split()
        else:
            file_1, file_2, file_3 = exp

        face_path = args.input_dir / file_1
        shape_path = args.input_dir / file_2
        color_path = args.input_dir / file_3

        base_name = '_'.join([path.stem for path in (face_path, shape_path, color_path)])
        exp_name = base_name if model_args.save_all else None

        if configured_output is None:
            os.makedirs(args.output_dir, exist_ok=True)
            output_image_path = args.output_dir / f'{base_name}.png'
        else:
            os.makedirs(configured_output.parent, exist_ok=True)
            output_image_path = configured_output

        final_image = hair_fast.swap(face_path, shape_path, color_path, benchmark=args.benchmark, exp_name=exp_name)
        save_image(final_image, output_image_path)


if __name__ == "__main__":
    model_parser = get_parser()
    parser = argparse.ArgumentParser(description='HairFast evaluate')
    parser.add_argument('--input_dir', type=Path, default='', help='The directory of the images to be inverted')
    parser.add_argument('--benchmark', action='store_true', help='Calculates the speed of the method during the session')

    # Arguments for a set of experiments
    parser.add_argument('--file_path', type=Path, default=None,
                        help='File with experiments with the format "face_path.png shape_path.png color_path.png"')
    parser.add_argument('--output_dir', type=Path, default=Path('output'), help='The directory for final results')

    # Arguments for single experiment
    parser.add_argument('--face_path', type=Path, default=None, help='Path to the face image')
    parser.add_argument('--shape_path', type=Path, default=None, help='Path to the shape image')
    parser.add_argument('--color_path', type=Path, default=None, help='Path to the color image')
    parser.add_argument('--result_path', type=Path, default=None, help='Path to save the result')

    args, unknown1 = parser.parse_known_args()
    model_args, unknown2 = model_parser.parse_known_args()
    args, model_args = apply_user_config(args, model_args)

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
