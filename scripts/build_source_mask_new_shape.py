import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.CtrlHair.global_value_utils import PARSING_LABEL_LIST
from models.CtrlHair.shape_branch.config import cfg as cfg_mask
from models.CtrlHair.shape_branch.model import Generator as ShapeGenerator
from models.CtrlHair.shape_branch.shape_util import mask_label_to_one_hot, mask_one_hot_to_label, split_hair_face
from models.CtrlHair.util.mask_color_util import mask_to_rgb
from models.encoder4editing.utils.model_utils import get_latents, setup_model
from models.stylegan2.model import Generator as StyleGANGenerator
from models.stylegan2.model import PixelNorm
from scripts.face_parsing_color_mask import LABELS as RAW_LABELS
from scripts.face_parsing_color_mask import PALETTE as RAW_PALETTE
from scripts.face_parsing_color_mask import load_model as load_face_parser
from scripts.face_parsing_color_mask import parse_image


CELEBA_PALETTE = np.array(
    [
        [0, 128, 64],
        [204, 0, 0],
        [76, 153, 0],
        [204, 204, 0],
        [51, 51, 255],
        [204, 0, 204],
        [0, 255, 255],
        [51, 255, 255],
        [102, 51, 0],
        [255, 0, 0],
        [102, 204, 0],
        [255, 255, 0],
        [0, 0, 153],
        [0, 0, 204],
        [255, 51, 153],
        [0, 204, 204],
        [0, 51, 0],
        [255, 153, 51],
        [0, 204, 0],
    ],
    dtype=np.uint8,
)


class ModulationModule(nn.Module):
    def __init__(self, layernum: int, last: bool = False, inp: int = 512, middle: int = 512):
        super().__init__()
        self.layernum = layernum
        self.last = last
        self.fc = nn.Linear(512, 512)
        self.norm = nn.LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = nn.Sequential(
            nn.Linear(inp, middle),
            nn.LayerNorm([middle]),
            nn.LeakyReLU(),
            nn.Linear(middle, 512),
        )
        self.beta_function = nn.Sequential(
            nn.Linear(inp, middle),
            nn.LayerNorm([middle]),
            nn.LeakyReLU(),
            nn.Linear(middle, 512),
        )
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.norm(x)
        gamma = self.gamma_function(embedding)
        beta = self.beta_function(embedding)
        out = x * (1 + gamma) + beta
        if not self.last:
            out = self.leakyrelu(out)
        return out


class RotateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(6, i == 4) for i in range(5)])

    def forward(self, latent_from: torch.Tensor, latent_to: torch.Tensor) -> torch.Tensor:
        dt_latent = self.pixelnorm(latent_from)
        for modulation_module in self.modulation_module_list:
            dt_latent = modulation_module(dt_latent, latent_to)
        return latent_from + 0.1 * dt_latent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build baseline source mask with new hairstyle shape from a source image, "
            "a hairstyle reference, and optional source mask."
        )
    )
    parser.add_argument("source", type=Path, help="Original source face image.")
    parser.add_argument("shape_reference", type=Path, help="Hairstyle reference image before rotation.")
    parser.add_argument("--source_mask", type=Path, default=None, help="Optional source mask, raw label or color mask.")
    parser.add_argument("--output", type=Path, default=Path("output/source_new_shape_mask.png"))
    parser.add_argument("--label_output", type=Path, default=Path("output/source_new_shape_label.png"))
    parser.add_argument("--rotated_output", type=Path, default=Path("output/shape_rotate_to_source.png"))
    parser.add_argument("--source_mask_output", type=Path, default=None)
    parser.add_argument("--rotated_shape_mask_output", type=Path, default=None)
    parser.add_argument(
        "--source_mask_type",
        choices=("auto", "celeba_label", "raw_label", "celeba_color", "raw_color"),
        default="auto",
        help=(
            "Mask format. Use celeba_label for baseline/CtrlHair raw labels where hair=13; "
            "raw_label for original BiSeNet labels where hair=17."
        ),
    )
    parser.add_argument(
        "--face_checkpoint",
        type=Path,
        default=Path("pretrained_models/BiSeNet/face_parsing_79999_iter.pth"),
    )
    parser.add_argument(
        "--e4e_checkpoint",
        type=Path,
        default=Path("pretrained_models/encoder4editing/e4e_ffhq_encode.pt"),
    )
    parser.add_argument(
        "--rotate_checkpoint",
        type=Path,
        default=Path("pretrained_models/Rotate/rotate_best.pth"),
    )
    parser.add_argument("--stylegan_checkpoint", type=Path, default=Path("pretrained_models/StyleGAN/ffhq.pt"))
    parser.add_argument(
        "--shape_checkpoint",
        type=Path,
        default=Path("pretrained_models/ShapeAdaptor/mask_generator.pth"),
    )
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--output_size", choices=("256", "source"), default="256")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def raw_to_celeba_label(raw_label: np.ndarray) -> np.ndarray:
    raw_names = [RAW_LABELS[idx] for idx in sorted(RAW_LABELS)]
    celeba_label = np.zeros_like(raw_label, dtype=np.uint8)
    for celeba_idx, label_name in enumerate(PARSING_LABEL_LIST):
        celeba_label[raw_label == raw_names.index(label_name)] = celeba_idx
    return celeba_label


def color_to_label(rgb: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, float]:
    pixels = rgb.reshape(-1, 3).astype(np.int16)
    palette_i16 = palette.astype(np.int16)
    distances = ((pixels[:, None, :] - palette_i16[None, :, :]) ** 2).sum(axis=2)
    label = distances.argmin(axis=1).astype(np.uint8).reshape(rgb.shape[:2])
    mean_distance = np.sqrt(distances.min(axis=1)).mean().item()
    return label, mean_distance


def load_source_label(mask_path: Path, mask_type: str) -> np.ndarray:
    image = Image.open(mask_path)
    image_np = np.asarray(image)

    if mask_type == "auto":
        if image_np.ndim == 2:
            mask_type = "celeba_label"
        else:
            rgb = np.asarray(image.convert("RGB"))
            celeba_label, celeba_distance = color_to_label(rgb, CELEBA_PALETTE)
            raw_label, raw_distance = color_to_label(rgb, RAW_PALETTE)
            if celeba_distance <= raw_distance:
                print(f"Detected source mask as celeba_color, mean color distance={celeba_distance:.3f}")
                return celeba_label
            print(f"Detected source mask as raw_color, mean color distance={raw_distance:.3f}")
            return raw_to_celeba_label(raw_label)

    if mask_type in {"celeba_color", "raw_color"}:
        rgb = np.asarray(image.convert("RGB"))
        palette = CELEBA_PALETTE if mask_type == "celeba_color" else RAW_PALETTE
        label, mean_distance = color_to_label(rgb, palette)
        print(f"Loaded source mask as {mask_type}, mean color distance={mean_distance:.3f}")
    else:
        label = np.asarray(image.convert("L"), dtype=np.uint8)
        if label.max() > 18 and label.max() != 255:
            raise ValueError(
                f"Unsupported label values in {mask_path}: max={int(label.max())}. "
                "Expected label ids in 0..18 or 255."
            )

    if mask_type in {"raw_label", "raw_color"}:
        label = raw_to_celeba_label(label)
    return label.astype(np.uint8)


def parse_celeba_mask(face_parser, image_path: Path, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return parse_celeba_image(face_parser, image, device)


def parse_celeba_image(face_parser, image: Image.Image, device: torch.device) -> np.ndarray:
    raw_label = parse_image(face_parser, image, device)
    return raw_to_celeba_label(raw_label)


def image_to_tensor_01(image: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_01_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)[0]
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def load_stylegan_generator(args, device: torch.device) -> StyleGANGenerator:
    if not args.stylegan_checkpoint.exists():
        raise FileNotFoundError(f"StyleGAN checkpoint not found: {args.stylegan_checkpoint}")
    generator = StyleGANGenerator(
        args.size,
        args.latent,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier,
    ).to(device).eval()
    checkpoint = torch.load(args.stylegan_checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["g_ema"])
    for param in generator.parameters():
        param.requires_grad = False
    return generator


def load_rotate_model(checkpoint: Path, device: torch.device) -> RotateModel:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Rotate checkpoint not found: {checkpoint}")
    model = RotateModel().to(device).eval()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model


def rotate_shape_to_source(
    source_path: Path,
    shape_path: Path,
    args,
    device: torch.device,
) -> Image.Image:
    e4e, _ = setup_model(str(args.e4e_checkpoint), device)
    generator = load_stylegan_generator(args, device)
    rotate_model = load_rotate_model(args.rotate_checkpoint, device)

    source = image_to_tensor_01(Image.open(source_path), 256, device)
    shape = image_to_tensor_01(Image.open(shape_path), 256, device)
    source_w = get_latents(e4e, (source - 0.5) / 0.5)
    shape_w = get_latents(e4e, (shape - 0.5) / 0.5)

    with torch.inference_mode():
        rotate_to = rotate_model(shape_w[:, :6], source_w[:, :6])
        rotate_to = torch.cat((rotate_to, shape_w[:, 6:]), dim=1)
        rotated, _ = generator([rotate_to], input_is_latent=True, return_latents=False)
    return tensor_01_to_image((rotated + 1) / 2)


def resize_label_tensor(label: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    label = label[None, None].float()
    return F.interpolate(label, size=size, mode="nearest").long()[0, 0]


def get_hair_face_code(mask_generator: ShapeGenerator, label: torch.Tensor):
    label = resize_label_tensor(label, (256, 256))
    one_hot = mask_label_to_one_hot(label[None, None])
    hair, face = split_hair_face(one_hot)
    hair_code = mask_generator.forward_hair_encoder(hair, testing=True)
    face_code = mask_generator.forward_face_encoder(face)
    return face_code, hair_code


def build_new_shape_mask(
    mask_generator: ShapeGenerator,
    source_label: np.ndarray,
    rotated_shape_label: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    source = torch.from_numpy(source_label).to(device=device, dtype=torch.long)
    rotated_shape = torch.from_numpy(rotated_shape_label).to(device=device, dtype=torch.long)
    source_face_code, _ = get_hair_face_code(mask_generator, source)
    _, rotated_hair_code = get_hair_face_code(mask_generator, rotated_shape)
    out_mask = mask_generator.forward_decode_by_code(rotated_hair_code, source_face_code)
    return mask_one_hot_to_label(out_mask)[0].detach().cpu().byte()


def load_shape_generator(checkpoint: Path, device: torch.device) -> ShapeGenerator:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Shape checkpoint not found: {checkpoint}")
    model = ShapeGenerator(cfg_mask).to(device).eval()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model


def save_label_and_color(label: np.ndarray, label_path: Path | None, color_path: Path | None) -> None:
    if label_path is not None:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label, mode="L").save(label_path)
    if color_path is not None:
        color_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask_to_rgb(label, draw_type=0)).save(color_path)


def main() -> None:
    args = build_parser().parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Source image not found: {args.source}")
    if args.source_mask is not None and not args.source_mask.exists():
        raise FileNotFoundError(f"Source mask not found: {args.source_mask}")
    if not args.shape_reference.exists():
        raise FileNotFoundError(f"Shape reference image not found: {args.shape_reference}")

    device = resolve_device(args.device)
    face_parser = load_face_parser(args.face_checkpoint, device)
    mask_generator = load_shape_generator(args.shape_checkpoint, device)

    rotated_shape_image = rotate_shape_to_source(args.source, args.shape_reference, args, device)
    args.rotated_output.parent.mkdir(parents=True, exist_ok=True)
    rotated_shape_image.save(args.rotated_output)

    if args.source_mask is None:
        source_label = parse_celeba_mask(face_parser, args.source, device)
    else:
        source_label = load_source_label(args.source_mask, args.source_mask_type)
    rotated_shape_label = parse_celeba_image(face_parser, rotated_shape_image, device)
    new_shape_label = build_new_shape_mask(mask_generator, source_label, rotated_shape_label, device).numpy()

    if args.output_size == "source":
        source_size = Image.open(args.source).size
        new_shape_label = np.array(
            Image.fromarray(new_shape_label, mode="L").resize(source_size, Image.NEAREST),
            dtype=np.uint8,
        )

    save_label_and_color(new_shape_label, args.label_output, args.output)
    save_label_and_color(source_label, None, args.source_mask_output)
    save_label_and_color(rotated_shape_label, None, args.rotated_shape_mask_output)

    print(f"Saved source mask with new shape: {args.output}")
    print(f"Saved raw label mask: {args.label_output}")
    print(f"Saved rotated hairstyle reference: {args.rotated_output}")


if __name__ == "__main__":
    main()
