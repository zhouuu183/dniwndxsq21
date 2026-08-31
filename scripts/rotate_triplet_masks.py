"""Rotate two reference images to a source face and export binary masks.

This is a standalone diagnostic/photo utility.  It does not modify or import
the PP, SATD, V5, or V6 post-processing paths.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original face image")
    parser.add_argument("--shape", type=Path, required=True, help="Reference hairstyle image")
    parser.add_argument("--color", type=Path, required=True, help="Reference hair-colour image")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stylegan_checkpoint", type=Path, default=Path("pretrained_models/StyleGAN/ffhq.pt"))
    parser.add_argument("--rotate_checkpoint", type=Path, default=Path("pretrained_models/Rotate/rotate_best.pth"))
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (
        tensor.detach().cpu().float().clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .add(0.5)
        .byte()
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def save_binary(mask: torch.Tensor, path: Path, size: tuple[int, int] | None = None) -> None:
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]
    mask = mask.float().clamp(0, 1)
    if size is not None and tuple(mask.shape[-2:]) != tuple(size):
        mask = torch.nn.functional.interpolate(
            mask[None, None], size=size, mode="nearest"
        )[0, 0]
    image = mask.mul(255).byte().cpu().numpy()
    Image.fromarray(image, mode="L").save(path)


def rotate_reference(net, rotate_model, source_w, reference_w):
    rotated_w = rotate_model(reference_w[:, :6], source_w[:, :6])
    rotated_w = torch.cat((rotated_w, reference_w[:, 6:]), dim=1)
    image, _ = net.generator(
        [rotated_w], input_is_latent=True, return_latents=False
    )
    return ((image + 1) / 2).clamp(0, 1)


def main() -> None:
    args = parse_args()
    import torch
    from torchvision import transforms as T
    from models.Embedding import Embedding
    from models.Encoders import RotateModel
    from models.Net import Net, get_segmentation
    from models.CtrlHair.external_code.face_parsing.my_parsing_util import (
        FaceParsing,
        FaceParsing_tensor,
    )

    args.source = args.source.expanduser().resolve()
    args.shape = args.shape.expanduser().resolve()
    args.color = args.color.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    # This utility is often launched from ``/data/coding`` while its code is
    # mounted under ``/root/shared-nvme``.  Resolve relative checkpoints and
    # legacy hard-coded asset paths against the author project that actually
    # contains pretrained_models, without copying those large files.
    project_root = ROOT
    asset_root = project_root
    for candidate in (
        project_root,
        project_root.parent / "HairFastGAN-main",
        Path("/data/coding/HairFastGAN/HairFastGAN-main"),
    ):
        if (candidate / "pretrained_models" / "StyleGAN" / "ffhq.pt").is_file():
            asset_root = candidate.resolve()
            break

    def resolve_asset_path(value: Path) -> Path:
        value = value.expanduser()
        if value.is_absolute():
            return value.resolve()
        project_candidate = project_root / value
        asset_candidate = asset_root / value
        return (project_candidate if project_candidate.exists() else asset_candidate).resolve()

    args.stylegan_checkpoint = resolve_asset_path(args.stylegan_checkpoint)
    args.rotate_checkpoint = resolve_asset_path(args.rotate_checkpoint)
    os.chdir(asset_root)
    for path, label in (
        (args.source, "Source image"),
        (args.shape, "Shape reference image"),
        (args.color, "Colour reference image"),
        (args.stylegan_checkpoint, "StyleGAN checkpoint"),
        (args.rotate_checkpoint, "Rotate checkpoint"),
    ):
        require_file(path, label)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    tensorize = T.ToTensor()
    # Embedding.embedding_images builds the batch dimension itself through its
    # DataLoader.  Keep each image as CHW here; adding a batch dimension would
    # make the collated input 5-D and break the bicubic downsampler's padding.
    source = tensorize(Image.open(args.source).convert("RGB"))
    shape = tensorize(Image.open(args.shape).convert("RGB"))
    color = tensorize(Image.open(args.color).convert("RGB"))

    model_args = argparse.Namespace(
        device=args.device,
        size=1024,
        latent=512,
        n_mlp=8,
        channel_multiplier=2,
        ckpt=str(args.stylegan_checkpoint),
        batch_size=3,
        mixing=0.95,
        save_all=False,
        save_all_dir=args.output,
    )
    net = Net(model_args)
    embed = Embedding(model_args, net=net)
    # FeatureStyleEncoder keeps ``dlatent_avg`` and ``noise_inputs`` as plain
    # attributes rather than registered buffers.  ``Trainer.to(device)``
    # therefore does not move them, which otherwise leaves CPU diagnostics
    # with a CUDA latent average during reconstruction.
    requested_device = torch.device(args.device)
    embed.encoder.to(requested_device)
    embed.encoder.device = requested_device
    if hasattr(embed.encoder, "opts"):
        embed.encoder.opts.device = str(requested_device)
    if hasattr(embed.encoder, "dlatent_avg"):
        embed.encoder.dlatent_avg = embed.encoder.dlatent_avg.to(requested_device)
    if hasattr(embed.encoder, "noise_inputs"):
        embed.encoder.noise_inputs = [
            noise.to(requested_device) for noise in embed.encoder.noise_inputs
        ]
    # Net.__init__ eagerly initializes the shared tensor parser and its legacy
    # implementation places BiSeNet on CUDA unconditionally.  Move that
    # singleton as well so get_segmentation() accepts CPU inputs.
    # The legacy implementation accidentally stores the tensor parser model
    # on FaceParsing.bise_net (rather than FaceParsing_tensor.bise_net), so
    # handle both aliases and avoid leaving the actual singleton on CUDA.
    parser_nets = (
        getattr(FaceParsing, "bise_net", None),
        getattr(FaceParsing_tensor, "bise_net", None),
    )
    for parser_net in parser_nets:
        if parser_net is not None:
            parser_net.to(requested_device).eval()
    # BicubicDownSample predates the device-aware pipeline and defaults to
    # constructing CUDA convolution weights.  The standalone utility also
    # supports CPU diagnostics, so make its two fixed filters follow the
    # requested device explicitly.
    use_cuda_filters = str(args.device).startswith("cuda")
    for downsample in (embed.downsample_512, embed.downsample_256):
        downsample.cuda = ".cuda" if use_cuda_filters else ""
        if use_cuda_filters:
            downsample.k1 = downsample.k1.to(args.device)
            downsample.k2 = downsample.k2.to(args.device)
        else:
            downsample.k1 = downsample.k1.cpu()
            downsample.k2 = downsample.k2.cpu()
    rotate_model = RotateModel()
    checkpoint = torch.load(args.rotate_checkpoint, map_location="cpu")
    rotate_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    rotate_model.to(args.device).eval()

    images_to_name = {source: ["source"], shape: ["shape"], color: ["color"]}
    with torch.inference_mode():
        name_to_embed = embed.embedding_images(images_to_name)
        source_entry = name_to_embed["source"]
        shape_entry = name_to_embed["shape"]
        color_entry = name_to_embed["color"]
        source_w = source_entry["W"]
        shape_rotated = rotate_reference(net, rotate_model, source_w, shape_entry["W"])
        color_rotated = rotate_reference(net, rotate_model, source_w, color_entry["W"])

        # BiSeNet parsing is converted to the project's CelebA label order by
        # get_segmentation().  In that order, hair is label 13 (not the raw
        # BiSeNet class id 17).
        source_parsing = source_entry["mask"]
        shape_parsing = get_segmentation(
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(shape_rotated)
        )
        color_parsing = get_segmentation(
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(color_rotated)
        )
        hair_label = 13
        source_hair_mask = (source_parsing == hair_label).float()
        shape_hair_mask = (shape_parsing == hair_label).float()
        color_hair_mask = (color_parsing == hair_label).float()
        # SATD cleanup is the source-hair area not covered by the rotated
        # reference hairstyle.  Keep it binary so it can be used directly as
        # a region mask.
        satd_cleanup_mask = (source_hair_mask * (1.0 - shape_hair_mask)).clamp(0, 1)

    save_rgb(shape_rotated, args.output / "shape_rotated_to_source.png")
    save_rgb(color_rotated, args.output / "color_rotated_to_source.png")
    mask_size = tuple(shape_rotated.shape[-2:])
    save_binary(source_hair_mask, args.output / "source_hair_mask.png", mask_size)
    save_binary(shape_hair_mask, args.output / "shape_rotated_hair_mask.png", mask_size)
    save_binary(color_hair_mask, args.output / "color_rotated_hair_mask.png", mask_size)
    save_binary(satd_cleanup_mask, args.output / "satd_cleanup_mask.png", mask_size)
    print(f"Wrote rotated images and binary masks to: {args.output}")


if __name__ == "__main__":
    main()
