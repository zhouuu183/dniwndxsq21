from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Encoders import FeatureEncoderMult, FeatureiResnet, ModulationModule
from models.postprocess_v5 import (
    FrozenPPBackbone,
    checkpoint_sha256,
    load_checkpoint_compat,
    normalize_checkpoint_key,
)
from models.stylegan2.model import Generator, PixelNorm


class OfficialPPReference(nn.Module):
    """Literal reference implementation from the original PP model."""

    def __init__(self, latent_avg_path):
        super().__init__()
        self.encoder_face = FeatureEncoderMult(
            fs_layers=[9],
            opts=argparse.Namespace(arcface_model_path="pretrained_models/ArcFace/backbone_ir50.pth"),
        )
        self.register_buffer("latent_avg", torch.load(latent_avg_path, map_location="cpu"), persistent=False)
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])
        self.to_latent_1 = nn.ModuleList([ModulationModule(18, index == 4) for index in range(5)])
        self.to_latent_2 = nn.ModuleList([ModulationModule(18, index == 4) for index in range(5)])
        self.pixelnorm = PixelNorm()

    def load_official_checkpoint(self, path):
        checkpoint = load_checkpoint_compat(path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        state = dict((normalize_checkpoint_key(key), value) for key, value in state.items())
        self.load_state_dict(state, strict=True)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, source, target):
        s_face, face_features = self.encoder_face(source)
        s_hair, hair_features = self.encoder_face(target)
        dt_face = self.pixelnorm(s_face)
        dt_hair = self.pixelnorm(s_hair)
        for module in self.to_latent_1:
            dt_face = module(dt_face, s_hair)
        for module in self.to_latent_2:
            dt_hair = module(dt_hair, s_face)
        latent_s = self.latent_avg.to(source.device) + 0.1 * (dt_face + dt_hair)
        latent_f = self.to_feature(torch.cat((face_features[0], hair_features[0]), dim=1))
        return latent_s, latent_f


def official_render(generator, latent_s, latent_f):
    raw_pp_norm, _ = generator(
        [latent_s],
        input_is_latent=True,
        return_latents=False,
        start_layer=5,
        end_layer=8,
        layer_in=latent_f,
    )
    return raw_pp_norm


@torch.no_grad()
def deterministic_render(generator, latent_s, latent_f):
    """Validation-only render using StyleGAN's fixed registered noise.

    Production keeps the exact official call above.  Equivalence validation
    must not compare two independently sampled noise realizations.
    """
    raw_pp_norm, _ = generator(
        [latent_s],
        input_is_latent=True,
        return_latents=False,
        start_layer=5,
        end_layer=8,
        layer_in=latent_f,
        randomize_noise=False,
    )
    return raw_pp_norm


def configure_deterministic_cuda(device):
    """Disable CUDA math modes that can create false equivalence failures."""
    settings = {
        "enabled": device.type == "cuda",
        "cublas_workspace_config": None,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "deterministic_algorithms": False,
    }
    if device.type != "cuda":
        return settings

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    settings["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        settings["deterministic_algorithms"] = True
    except TypeError:
        # PyTorch 1.9 has no warn_only argument.  Inference operations used by
        # this script support its strict deterministic mode.
        torch.use_deterministic_algorithms(True)
        settings["deterministic_algorithms"] = True
    return settings


def load_image_256(path, device):
    with Image.open(path) as image:
        tensor = T.functional.to_tensor(image.convert("RGB")).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    return (tensor.to(device) * 2.0 - 1.0).clamp(-1, 1)


def list_images(directory):
    if directory is None or not Path(directory).is_dir():
        return []
    extensions = (".png", ".jpg", ".jpeg")
    return [path for path in sorted(Path(directory).iterdir()) if path.suffix.lower() in extensions]


def build_generator(args, device):
    generator = Generator(1024, 512, 8, channel_multiplier=2)
    checkpoint = load_checkpoint_compat(args.stylegan_checkpoint, map_location="cpu")
    generator.load_state_dict(checkpoint["g_ema"], strict=True)
    generator.to(device).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator


def build_pairs(args):
    sources = list_images(args.source_gallery_dir)
    targets = list_images(args.target_gallery_dir) or sources
    if not sources or not targets:
        return []
    return [(sources[index % len(sources)], targets[index % len(targets)]) for index in range(args.count)]


def validate(args):
    device = torch.device(args.device)
    deterministic_settings = configure_deterministic_cuda(device)
    pairs = build_pairs(args)
    report = {
        "pp_checkpoint": str(Path(args.pp_checkpoint).resolve()),
        "pp_checkpoint_sha256": checkpoint_sha256(args.pp_checkpoint),
        "requested_sample_count": int(args.count),
        "real_sample_count": len(pairs),
        "strict_load_statistics": None,
        "official_max_error": None,
        "official_mean_error": None,
        "latent_s_max_error": None,
        "latent_f_max_error": None,
        "non_finite_outputs": 0,
        "render_validation_mode": "fixed_registered_noise",
        "deterministic_settings": deterministic_settings,
        "passed": False,
    }

    strict_probe = FrozenPPBackbone(args.latent_avg_path)
    report["strict_load_statistics"] = strict_probe.load_checkpoint_strict(args.pp_checkpoint)
    del strict_probe
    if not pairs:
        return report

    output_dir = args.output.parent / "frozen_pp_validation_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = build_generator(args, device)
    official = OfficialPPReference(args.latent_avg_path)
    official.load_official_checkpoint(args.pp_checkpoint)
    official.to(device)
    official_latents = []
    with torch.no_grad():
        for index, (source_path, target_path) in enumerate(pairs):
            source = load_image_256(source_path, device)
            target = load_image_256(target_path, device)
            latent_s, latent_f = official(source, target)
            official_latents.append((latent_s.cpu(), latent_f.cpu()))
    del official
    if device.type == "cuda":
        torch.cuda.empty_cache()

    frozen = FrozenPPBackbone(args.latent_avg_path)
    frozen.load_checkpoint_strict(args.pp_checkpoint)
    frozen.to(device)
    max_errors = []
    mean_errors = []
    latent_s_errors = []
    latent_f_errors = []
    with torch.no_grad():
        for index, (source_path, target_path) in enumerate(pairs):
            source = load_image_256(source_path, device)
            target = load_image_256(target_path, device)
            latent_s, latent_f = frozen(source, target)
            official_latent_s = official_latents[index][0].to(device)
            official_latent_f = official_latents[index][1].to(device)

            # Render back-to-back with identical fixed noise and CUDA math
            # settings.  This isolates PP equivalence from generator RNG and
            # workspace scheduling differences.
            official_raw = deterministic_render(generator, official_latent_s, official_latent_f)
            new_raw = deterministic_render(generator, latent_s, latent_f)
            difference = (new_raw - official_raw).abs()
            max_errors.append(float(difference.max().item()))
            mean_errors.append(float(difference.mean().item()))
            latent_s_errors.append(float((latent_s - official_latents[index][0].to(device)).abs().max().item()))
            latent_f_errors.append(float((latent_f - official_latents[index][1].to(device)).abs().max().item()))
            if not torch.isfinite(new_raw).all() or not torch.isfinite(official_raw).all():
                report["non_finite_outputs"] += 1
            preview_columns = (
                (target + 1.0) / 2.0,
                (official_raw + 1.0) / 2.0,
                (new_raw + 1.0) / 2.0,
                difference.clamp(0, 1),
            )
            preview = torch.cat(
                tuple(
                    F.interpolate(column, size=(256, 256), mode="bilinear", align_corners=False)
                    for column in preview_columns
                ),
                dim=-1,
            )
            save_image(preview.cpu(), str(output_dir / ("sample_%03d.png" % index)))

    report["official_max_error"] = max(max_errors)
    report["official_mean_error"] = sum(mean_errors) / len(mean_errors)
    report["latent_s_max_error"] = max(latent_s_errors)
    report["latent_f_max_error"] = max(latent_f_errors)
    report["passed"] = (
        len(pairs) >= 20
        and report["non_finite_outputs"] == 0
        and report["official_max_error"] < 1e-5
        and report["latent_s_max_error"] < 1e-6
        and report["latent_f_max_error"] < 1e-6
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Validate exact frozen PP equivalence")
    parser.add_argument("--source_gallery_dir", type=Path)
    parser.add_argument("--target_gallery_dir", type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--pp_checkpoint", type=str, default="pretrained_models/PostProcess/pp_model.pth")
    parser.add_argument("--latent_avg_path", type=str, default="pretrained_models/PostProcess/latent_avg.pt")
    parser.add_argument("--stylegan_checkpoint", type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("output/frozen_pp_validation_v2.json"))
    return parser


def main(args):
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(build_parser().parse_args())
