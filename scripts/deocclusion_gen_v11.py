from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.CtrlHair.external_code.face_parsing.my_parsing_util import FaceParsing_tensor
from utils.deocclusion_masks_v11 import build_deocclusion_masks_v11, hair_mask_from_parsing
from utils.image_utils import list_image_files
from utils.train import seed_everything


# ========================= user config: edit only here =========================
USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407

USER_CLEAN_ROOT = Path("images/FFHQ_short")
USER_LONG_DONOR_ROOT = Path("images/FFHQ_long")
USER_OUTPUT_DIR = Path("input/deocclusion_dataset_v11_dgrr_fill")
USER_DATASET_SIZE = 300
USER_SAVE_EVERY = 100
USER_IMAGE_SIZE = 1024
USER_RENDER_HAIRFAST_BASELINE = True
USER_MASK_PREVIEW_COUNT = 24

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"

USER_HAIR_ALPHA_BLUR = 19
USER_MIN_DONOR_HAIR_AREA = 0.08
# ============================================================================ 


TO_BISENET = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TO_TENSOR = T.ToTensor()


def load_image(path: Path, size: int, device: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = T.functional.resize(image, [size, size], interpolation=T.InterpolationMode.BICUBIC)
    return TO_TENSOR(image).unsqueeze(0).to(device)


@torch.inference_mode()
def parse_image_256(image: torch.Tensor) -> torch.Tensor:
    image_512 = F.interpolate(image, size=(512, 512), mode="bilinear", align_corners=False)
    parsing, _ = FaceParsing_tensor.parsing_img(TO_BISENET(image_512[0]).unsqueeze(0))
    parsing = FaceParsing_tensor.swap_parsing_label_to_celeba_mask(parsing)
    parsing = parsing.long()[None, None, ...].to(image.device)
    return F.interpolate(parsing.float(), size=(256, 256), mode="nearest").long()


def blur_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.avg_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).clamp(0, 1)


@torch.inference_mode()
def make_synthetic_occlusion(clean: torch.Tensor, donor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_parsing = parse_image_256(clean)
    donor_parsing = parse_image_256(donor)
    donor_hair_256 = hair_mask_from_parsing(donor_parsing)
    if float(donor_hair_256.mean().detach().cpu()) < USER_MIN_DONOR_HAIR_AREA:
        raise RuntimeError("Donor hair mask is too small.")

    donor_hair = F.interpolate(donor_hair_256, size=clean.shape[-2:], mode="nearest")
    alpha = blur_mask(donor_hair, USER_HAIR_ALPHA_BLUR)
    source = (clean * (1.0 - alpha) + donor * alpha).clamp(0, 1)

    source_parsing = clean_parsing.clone()
    donor_hair_small = donor_hair_256 > 0.5
    source_parsing = torch.where(donor_hair_small, torch.full_like(source_parsing, 13), source_parsing)
    return source, source_parsing, clean_parsing


def flush_chunk(output_dir: Path, items: list[dict], chunk_idx: int):
    if not items:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(items, output_dir / f"deocclusion_part_{chunk_idx:03d}.dataset")


def save_mask_preview(output_dir: Path, sample_idx: int, item: dict):
    if sample_idx >= USER_MASK_PREVIEW_COUNT:
        return
    preview_dir = output_dir / "mask_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    masks = item["masks"]
    panels = [
        item["source"],
        item["reference"],
        item["base"],
        item["target"],
        masks["M_removed"].repeat(3, 1, 1),
        masks["M_halo"].repeat(3, 1, 1),
        masks["M_skin_soft"].repeat(3, 1, 1),
        masks["M_struct_soft"].repeat(3, 1, 1),
        masks["M_bg_soft"].repeat(3, 1, 1),
        masks["M_fill_soft"].repeat(3, 1, 1),
        masks["M_clean_soft"].repeat(3, 1, 1),
        masks["M_face_preserve"].repeat(3, 1, 1),
        masks["R_ctx"].repeat(3, 1, 1),
        masks["R_bg_ctx"].repeat(3, 1, 1),
        masks["E_struct"].repeat(3, 1, 1),
        masks["D_t_norm"].repeat(3, 1, 1),
    ]
    row = torch.cat([panel.detach().cpu().float().clamp(0, 1) for panel in panels], dim=2)
    T.ToPILImage()(row).save(preview_dir / f"sample_{sample_idx:03d}.png")
    if sample_idx == 0:
        with open(preview_dir / "columns.txt", "w", encoding="utf-8") as file:
            file.write(
                "source | reference | base | target | M_removed | M_halo | M_skin | M_struct | "
                "M_bg | M_fill | M_clean | M_face_preserve | R_ctx | R_bg_ctx | E_struct | D_t_norm\n"
            )


def build_hairfast_runner():
    from hair_swap_v11 import HairFast_v11, get_parser_v11

    model_args = get_parser_v11().parse_args([])
    model_args.device = USER_DEVICE
    model_args.ckpt = USER_STYLEGAN_CKPT
    model_args.rotate_checkpoint = USER_ROTATE_CKPT
    model_args.blending_checkpoint = USER_BLENDING_CKPT
    model_args.pp_checkpoint = USER_PP_CKPT
    model_args.use_deocclusion_v11 = False
    return HairFast_v11(model_args)


def main():
    seed_everything(USER_RANDOM_SEED)
    rng = random.Random(USER_RANDOM_SEED)
    clean_files = list_image_files(USER_CLEAN_ROOT)
    donor_files = list_image_files(USER_LONG_DONOR_ROOT)
    if not clean_files:
        raise RuntimeError(f"No clean images found under {USER_CLEAN_ROOT}")
    if not donor_files:
        raise RuntimeError(f"No donor images found under {USER_LONG_DONOR_ROOT}")

    hair_fast = build_hairfast_runner() if USER_RENDER_HAIRFAST_BASELINE else None

    USER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    chunk_idx = 1
    with open(USER_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for sample_idx in tqdm(range(USER_DATASET_SIZE), desc="build deocclusion v11"):
            clean_name = rng.choice(clean_files)
            donor_name = rng.choice(donor_files)
            clean = load_image(USER_CLEAN_ROOT / clean_name, USER_IMAGE_SIZE, USER_DEVICE)
            donor = load_image(USER_LONG_DONOR_ROOT / donor_name, USER_IMAGE_SIZE, USER_DEVICE)
            try:
                source, source_parsing, clean_parsing = make_synthetic_occlusion(clean, donor)
            except RuntimeError:
                continue

            target_hair = hair_mask_from_parsing(clean_parsing)
            masks = build_deocclusion_masks_v11(
                source_parsing=source_parsing,
                target_hair_mask=target_hair,
                target_parsing=clean_parsing,
            )
            if USER_RENDER_HAIRFAST_BASELINE and hair_fast is not None:
                with torch.inference_mode():
                    base = hair_fast.swap(source[0].detach().cpu(), clean[0].detach().cpu(), clean[0].detach().cpu())
                base = base.unsqueeze(0).to(USER_DEVICE)
            else:
                base = source

            item = {
                "source": F.interpolate(source, size=(256, 256), mode="bilinear", align_corners=False)[0].cpu(),
                "reference": F.interpolate(clean, size=(256, 256), mode="bilinear", align_corners=False)[0].cpu(),
                "base": F.interpolate(base, size=(256, 256), mode="bilinear", align_corners=False)[0].cpu(),
                "target": F.interpolate(clean, size=(256, 256), mode="bilinear", align_corners=False)[0].cpu(),
                "masks": {key: value[0].cpu() for key, value in masks.items()},
                "clean_name": clean_name,
                "donor_name": donor_name,
            }
            save_mask_preview(USER_OUTPUT_DIR, sample_idx, item)
            items.append(item)
            print(f"{sample_idx:06d} {clean_name} {donor_name}", file=f_exps, flush=True)

            if len(items) >= USER_SAVE_EVERY:
                flush_chunk(USER_OUTPUT_DIR, items, chunk_idx)
                items = []
                chunk_idx += 1

    flush_chunk(USER_OUTPUT_DIR, items, chunk_idx)
    print(f"Saved deocclusion v11 dataset to {USER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
