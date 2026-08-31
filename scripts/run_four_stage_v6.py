"""Save the four visible stages of one V6 hair-transfer run.

The stages are:

1. author hairstyle transfer;
2. author hairstyle + hair-colour transfer before SATD;
3. author transfer after SATD cleanup;
4. V6 PP decode with the optional source-earring recall.

This is a diagnostic runner only.  It does not alter the V5/V6 training or
inference paths.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Resolve defaults from the project containing this script.  This keeps the
# diagnostic runner independent of the shell's current working directory.
DEFAULT_STYLEGAN_CHECKPOINT = ROOT / "pretrained_models" / "StyleGAN" / "ffhq.pt"
DEFAULT_ROTATE_CHECKPOINT = ROOT / "pretrained_models" / "Rotate" / "rotate_best.pth"
DEFAULT_BLENDING_CHECKPOINT = ROOT / "pretrained_models" / "Blending" / "checkpoint.pth"
DEFAULT_PP_CHECKPOINT = ROOT / "pretrained_models" / "PostProcess" / "pp_model.pth"
DEFAULT_SATD_CHECKPOINT = ROOT / "checkpoints" / "satd_3000_best.pth"


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def to_tensor(path: Path) -> torch.Tensor:
    from torchvision.transforms import functional as TF

    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB"))


def save_image(value: torch.Tensor, path: Path) -> None:
    if value.ndim == 4:
        value = value[0]
    image = (
        value.detach().float().clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .add(0.5)
        .byte()
        .cpu()
        .numpy()
    )
    Image.fromarray(image, mode="RGB").save(path)


def decode(net, latent_s: torch.Tensor, latent_f: torch.Tensor) -> torch.Tensor:
    image, _ = net.generator(
        [latent_s],
        input_is_latent=True,
        return_latents=False,
        start_layer=4,
        end_layer=8,
        layer_in=latent_f,
    )
    return ((image + 1.0) * 0.5).clamp(0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original face image")
    parser.add_argument("--shape", type=Path, required=True, help="Hairstyle reference image")
    parser.add_argument("--color", type=Path, required=True, help="Hair-colour reference image")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stylegan_checkpoint", type=Path, default=DEFAULT_STYLEGAN_CHECKPOINT)
    parser.add_argument("--rotate_checkpoint", type=Path, default=DEFAULT_ROTATE_CHECKPOINT)
    parser.add_argument("--blending_checkpoint", type=Path, default=DEFAULT_BLENDING_CHECKPOINT)
    parser.add_argument("--pp_checkpoint", type=Path, default=DEFAULT_PP_CHECKPOINT)
    parser.add_argument("--pp_v6_checkpoint", type=Path, default=DEFAULT_PP_CHECKPOINT)
    parser.add_argument("--satd_checkpoint", type=Path, default=DEFAULT_SATD_CHECKPOINT)
    parser.add_argument(
        "--satd_blend",
        type=float,
        default=0.34,
        help="SATD F residual strength; matches the original SATD calibration.",
    )
    parser.add_argument("--use_satd", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--direct_satd_pp_input",
        type=int,
        choices=(0, 1),
        default=1,
        help="Feed I_satd_blend_256 directly to PP when SATD is enabled.",
    )
    parser.add_argument(
        "--satd_background_reference_weight",
        type=float,
        default=1.0,
        help="Weight of clean unoccluded background in the SATD cleanup field.",
    )
    parser.add_argument(
        "--satd_background_reference_size",
        type=int,
        default=256,
        help="Working resolution for clean-background colour propagation.",
    )
    parser.add_argument(
        "--satd_background_reference_kernel",
        type=int,
        default=51,
        help="Kernel size for clean-background colour propagation.",
    )
    parser.add_argument(
        "--satd_background_reference_sigma",
        type=float,
        default=15.0,
        help="Gaussian sigma for clean-background colour propagation.",
    )
    parser.add_argument(
        "--satd_background_residual_gate_floor",
        type=float,
        default=0.01,
        help="Suppress background writes below this correction magnitude.",
    )
    parser.add_argument(
        "--satd_background_residual_gate_ceiling",
        type=float,
        default=0.06,
        help="Full background-write strength at this correction magnitude.",
    )
    parser.add_argument("--enable_earring_recall", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("source", "shape", "color", "stylegan_checkpoint", "rotate_checkpoint",
                 "blending_checkpoint", "pp_checkpoint", "pp_v6_checkpoint", "satd_checkpoint"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output = args.output.expanduser().resolve()
    for path, label in (
        (args.source, "Source image"),
        (args.shape, "Hairstyle reference image"),
        (args.color, "Hair-colour reference image"),
        (args.stylegan_checkpoint, "StyleGAN checkpoint"),
        (args.rotate_checkpoint, "Rotate checkpoint"),
        (args.blending_checkpoint, "Blending checkpoint"),
        (args.pp_checkpoint, "Author PP checkpoint"),
        (args.pp_v6_checkpoint, "V6 PP checkpoint"),
    ):
        require_file(path, label)
    if args.use_satd:
        require_file(args.satd_checkpoint, "SATD checkpoint")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    from hair_swap import HairFast, get_parser as get_baseline_parser
    from hair_swap_v6 import HairFastV6, get_parser
    from utils.image_utils import equal_replacer

    source, shape, color = (to_tensor(path) for path in (args.source, args.shape, args.color))
    source, shape, color = equal_replacer([source, shape, color])
    images_to_name = defaultdict(list)
    for image, name in ((source, "face"), (shape, "shape"), (color, "color")):
        images_to_name[image].append(name)

    # Stage 1 must come from the actual no-suffix author pipeline.  Do not
    # obtain it from HairFastV6's shared author submodules: that would still
    # construct and route through the innovation wrapper.
    baseline_args = get_baseline_parser().parse_args([])
    baseline_args.device = args.device
    baseline_args.ckpt = str(args.stylegan_checkpoint)
    baseline_args.rotate_checkpoint = str(args.rotate_checkpoint)
    baseline_args.blending_checkpoint = str(args.blending_checkpoint)
    baseline_args.pp_checkpoint = str(args.pp_checkpoint)
    baseline_args.save_all = False
    torch.manual_seed(args.seed)
    baseline_model = HairFast(baseline_args)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        baseline_embeddings = baseline_model.embed.embedding_images(images_to_name)
        baseline_alignment = baseline_model.align.align_images(
            "face", "shape", baseline_embeddings
        )
        stage1 = decode(
            baseline_model.net,
            baseline_embeddings["face"]["S"],
            baseline_alignment["latent_F_align"],
        )
    del baseline_model, baseline_embeddings, baseline_alignment
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_args = get_parser().parse_args([])
    model_args.device = args.device
    model_args.ckpt = str(args.stylegan_checkpoint)
    model_args.rotate_checkpoint = str(args.rotate_checkpoint)
    model_args.blending_checkpoint = str(args.blending_checkpoint)
    model_args.pp_checkpoint = str(args.pp_checkpoint)
    model_args.pp_v6_checkpoint = str(args.pp_v6_checkpoint)
    model_args.use_satd_v8 = bool(args.use_satd)
    model_args.satd_checkpoint_v8 = str(args.satd_checkpoint) if args.use_satd else ""
    model_args.satd_blend_v8 = float(args.satd_blend)
    model_args.direct_satd_pp_input = bool(args.direct_satd_pp_input and args.use_satd)
    model_args.satd_background_reference_weight = float(args.satd_background_reference_weight)
    model_args.satd_background_reference_size = int(args.satd_background_reference_size)
    model_args.satd_background_reference_kernel = int(args.satd_background_reference_kernel)
    model_args.satd_background_reference_sigma = float(args.satd_background_reference_sigma)
    model_args.satd_background_residual_gate_floor = float(args.satd_background_residual_gate_floor)
    model_args.satd_background_residual_gate_ceiling = float(args.satd_background_residual_gate_ceiling)
    model_args.enable_earring_recall = bool(args.enable_earring_recall)
    model_args.enable_earring_query_recall = bool(args.enable_earring_recall)
    # Keep diagnostics on for this four-stage evaluator.  The trace reports
    # the actual SATD cleanup alpha, and the runtime debug directory contains
    # the clean-background reference used by the new restoration branch.
    model_args.save_all = True
    model_args.save_all_dir = args.output
    model_args.v6_runtime_diagnostics = True

    torch.manual_seed(args.seed)
    model = HairFastV6(model_args)
    # HairFastV6 constructs additional SATD/PP modules after the author
    # modules.  Reset the process seed after construction so the author stage
    # starts from the same RNG state as an independent baseline run.
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        name_to_embed = model.embed.embedding_images(images_to_name)
        align_shape = model.align.align_images("face", "shape", name_to_embed)

        align_color = model.align.shape_module("face", "color", name_to_embed, only_target=True)

        # Keep the exact RNG point before the author render.  The final PP
        # run below resets to this point so its author/SATD canvas matches the
        # two intermediate images saved by this diagnostic script.
        author_rng_cpu = torch.random.get_rng_state()
        author_rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        # Prepare the V8 SATD correction from the already aligned F_author
        # before invoking the author's colour blending encoder.  The helper
        # restores RNG state because this branch must not perturb the exact
        # author baseline render below.
        satd_alignment = model.blend._resolve_satd_alignment(
            {
                "satd_alignment_factory": lambda: model.satd_align.build_satd_background_candidate(
                    "face",
                    "shape",
                    name_to_embed,
                    align_shape,
                    use_satd_v8=bool(args.use_satd),
                    satd_blend_v8=float(args.satd_blend),
                )
            }
        )

        # Now run the author's colour blend.  The baseline render uses
        # F_author, while the paired SATD render below uses the prepared
        # F_satd with the same S_blend and replayed StyleGAN noise.
        I_1, HM_3E, HM_X, target_mask, S_blend, author_image = model.blend._author_transfer(
            align_shape,
            align_color,
            name_to_embed,
        )

        # Stage 2: exact author hairstyle + hair-colour result, before SATD.
        stage2 = ((author_image + 1.0) * 0.5).clamp(0, 1)

        # Stage 3: exact SATD candidate from the production renderer.  This
        # also replays the author's StyleGAN noise schedule for a clean
        # before/after comparison.
        satd_image, _ = model.blend._render_satd_candidate(
            S_blend,
            author_image,
            {"satd_alignment": satd_alignment},
        )
        stage3 = ((satd_image + 1.0) * 0.5).clamp(0, 1)

        # Stage 4 must use the production V6 compositor.  It supplies the
        # complete PP auxiliary contract (target visibility, source earring
        # evidence, duplicate suppression, and SATD continuity) before the
        # final native-earring write-back.  Reimplementing this call locally
        # can silently omit one of those guards and lose the earring.
        torch.random.set_rng_state(author_rng_cpu)
        if author_rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(author_rng_cuda)
        stage4 = model.blend.blend_images(
            align_shape,
            align_color,
            name_to_embed,
            satd_alignment=satd_alignment,
            use_satd_v8=bool(args.use_satd),
            satd_blend_v8=float(args.satd_blend),
            direct_satd_pp_input=bool(args.direct_satd_pp_input and args.use_satd),
            exp_name="four_stage_final",
        )
        if stage4.ndim == 3:
            stage4 = stage4.unsqueeze(0)

    outputs = (
        ("01_hair_transfer.png", stage1),
        ("02_hair_color_before_satd.png", stage2),
        ("03_hair_color_satd.png", stage3),
        ("04_pp_earring.png", stage4),
    )
    for filename, image in outputs:
        save_image(image, args.output / filename)
    print(f"Wrote 4 stages to: {args.output}")


if __name__ == "__main__":
    main()
