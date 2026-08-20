import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.SG_IDCT_v16 import SG_IDCT_v16, masked_mean_std, rgb_to_lab


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("input/sg_idct_dataset_v16")
USER_OUTPUT_DIR_FFHQ = Path("output/sg_idct_train_v16")
USER_VAL_SIZE_FFHQ = 512

USER_DATASET_DIR_SMALL = Path("input/sg_idct_dataset_v16_small")
USER_OUTPUT_DIR_SMALL = Path("output/sg_idct_train_v16_small")
USER_VAL_SIZE_SMALL = 32

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 16
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_EPOCHS = 20
USER_LR = 5e-5
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 1.0

USER_INIT_SG_IDCT_CKPT_V16 = ""
USER_BETA_INIT_V16 = 0.05
USER_CHROMA_STRENGTH_INIT_V16 = 0.55
USER_CHROMA_STD_STRENGTH_INIT_V16 = 0.0
USER_CHROMA_DELTA_LIMIT_INIT_V16 = 12.0
USER_TEXTURE_STRENGTH_INIT_V16 = 0.75
USER_ALPHA_STRENGTH_INIT_V16 = 0.90
USER_USE_GAMUT_MAP_V16 = False

USER_LAMBDA_COLOR_AB_V16 = 4.0


USER_LAMBDA_COLOR_RGB_V16 = 0.0
USER_LAMBDA_HAIR_LUMA_KEEP_V16 = 2.0
USER_LAMBDA_LOCK_RGB_KEEP_V16 = 8.0
USER_LAMBDA_LOCK_NON_DARK_V16 = 4.0



USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_LOG_IMAGE_COUNT = 16
# ============================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        },
        "small": {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "val_size": USER_VAL_SIZE_SMALL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(
            f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}. "
            f"Choose one of: {', '.join(sorted(profiles))}."
        )
    return profiles[USER_DATASET_PROFILE]


PROFILE = resolve_dataset_profile()
ACTIVE_DATASET_DIR = PROFILE["dataset_dir"]
ACTIVE_OUTPUT_DIR = PROFILE["output_dir"]
ACTIVE_VAL_SIZE = PROFILE["val_size"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


def masked_non_darker(pred: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return (torch.relu(baseline - pred) * mask).sum() / denom


def masked_stats_loss(pred: torch.Tensor, target: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    pred_mean, pred_std = masked_mean_std(pred, pred_mask)
    target_mean, target_std = masked_mean_std(target, target_mask)
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def read_manifest(dataset_dir: Path) -> list[Path]:
    manifest = dataset_dir / "manifest.tsv"
    if manifest.exists():
        paths = []
        with open(manifest, "r", encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                items = line.strip().split("\t")
                if items and items[0]:
                    paths.append(dataset_dir / "Tensors" / items[0])
        return [path for path in paths if path.exists()]
    return sorted((dataset_dir / "Tensors").glob("*.npz"))


class SGIDCTTensorDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        return {
            "I_satd": torch.from_numpy(data["I_satd"]).float(),
            "I_color": torch.from_numpy(data["I_color"]).float(),
            "H_align": torch.from_numpy(data["H_align"]).float(),
            "H_color": torch.from_numpy(data["H_color"]).float(),
            "M_remove": torch.from_numpy(data["M_remove"]).float(),
            "M_face": torch.from_numpy(data["M_face"]).float(),
            "M_neck": torch.from_numpy(data["M_neck"]).float(),
            "M_ear": torch.from_numpy(data["M_ear"]).float(),
        }


def calc_loss(model: SG_IDCT_v16, batch: dict[str, torch.Tensor], device: torch.device):
    I_satd = batch["I_satd"].to(device, non_blocking=True)
    I_color = batch["I_color"].to(device, non_blocking=True)
    H_align = batch["H_align"].to(device, non_blocking=True)
    H_color = batch["H_color"].to(device, non_blocking=True)
    M_remove = batch["M_remove"].to(device, non_blocking=True)
    M_face = batch["M_face"].to(device, non_blocking=True)
    M_neck = batch["M_neck"].to(device, non_blocking=True)
    M_ear = batch["M_ear"].to(device, non_blocking=True)

    I_pred, aux = model(
        I_satd,
        I_color,
        H_align,
        M_remove,
        M_face,
        M_neck,
        M_ear,
        H_color=H_color,
        return_aux=True,
    )

    lab_pred = rgb_to_lab(I_pred)
    lab_color = rgb_to_lab(I_color)
    color_ab_loss = masked_stats_loss(lab_pred[:, 1:3], lab_color[:, 1:3], aux["M_safe"], H_color)
    color_rgb_loss = masked_stats_loss(I_pred, I_color, aux["M_safe"], H_color)
    hair_luma_keep_loss = masked_l1(gray(I_pred), gray(I_satd), H_align)

    lock_mask = (M_remove * (1.0 - H_align) + M_face + M_neck + (1.0 - H_align) * (1.0 - M_ear)).clamp(0, 1)
    lock_rgb_keep_loss = masked_l1(I_pred, I_satd, lock_mask)
    lock_non_dark_loss = masked_non_darker(gray(I_pred), gray(I_satd), lock_mask)

    total_loss = (
        USER_LAMBDA_COLOR_AB_V16 * color_ab_loss
        + USER_LAMBDA_COLOR_RGB_V16 * color_rgb_loss
        + USER_LAMBDA_HAIR_LUMA_KEEP_V16 * hair_luma_keep_loss
        + USER_LAMBDA_LOCK_RGB_KEEP_V16 * lock_rgb_keep_loss
        + USER_LAMBDA_LOCK_NON_DARK_V16 * lock_non_dark_loss
    )
    return total_loss, {
        "loss": total_loss.detach(),
        "color_ab_loss": color_ab_loss.detach(),
        "color_rgb_loss": color_rgb_loss.detach(),
        "hair_luma_keep_loss": hair_luma_keep_loss.detach(),
        "lock_rgb_keep_loss": lock_rgb_keep_loss.detach(),
        "lock_non_dark_loss": lock_non_dark_loss.detach(),
        "I_pred": I_pred.detach(),
        "I_satd": I_satd.detach(),
        "I_color": I_color.detach(),
    }


def save_preview(path: Path, rows: list[list[torch.Tensor]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    for row in rows:
        panels.append(torch.cat([item.detach().cpu().clamp(0, 1) for item in row], dim=2))
    save_image(torch.cat(panels, dim=1), path)


def save_checkpoint(path: Path, model: SG_IDCT_v16, optimizer: torch.optim.Optimizer, epoch: int, best_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_loss": best_loss,
            "sg_idct_v16_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main():
    set_seed(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = read_manifest(ACTIVE_DATASET_DIR)
    if not paths:
        raise RuntimeError(f"No SG-IDCT tensor cache found under {ACTIVE_DATASET_DIR / 'Tensors'}")
    if len(paths) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"Dataset has {len(paths)} samples, smaller than validation split size ({ACTIVE_VAL_SIZE})."
        )

    train_paths, val_paths = train_test_split(paths, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        SGIDCTTensorDataset(train_paths),
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        SGIDCTTensorDataset(val_paths),
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
    )

    model = SG_IDCT_v16(
        beta=USER_BETA_INIT_V16,
        chroma_strength=USER_CHROMA_STRENGTH_INIT_V16,
        chroma_std_strength=USER_CHROMA_STD_STRENGTH_INIT_V16,
        chroma_delta_limit=USER_CHROMA_DELTA_LIMIT_INIT_V16,
        texture_strength=USER_TEXTURE_STRENGTH_INIT_V16,
        alpha_strength=USER_ALPHA_STRENGTH_INIT_V16,
        learnable=True,
        use_gamut_map=USER_USE_GAMUT_MAP_V16,
    ).to(device)
    if USER_INIT_SG_IDCT_CKPT_V16:
        ckpt = torch.load(USER_INIT_SG_IDCT_CKPT_V16, map_location=device)
        model.load_state_dict(ckpt.get("sg_idct_v16_state_dict", ckpt.get("model_state_dict", ckpt)), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
    best_loss = float("inf")

    for epoch in range(USER_EPOCHS):
        model.train()
        running_loss = 0.0
        running_steps = 0
        progress = tqdm(train_loader, desc=f"SG-IDCT train {epoch + 1}/{USER_EPOCHS}", leave=False)
        for batch in progress:
            loss, info = calc_loss(model, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), USER_GRAD_CLIP)
            optimizer.step()
            with torch.no_grad():
                model.beta.clamp_(0.0, 0.20)
                model.chroma_strength.clamp_(0.20, 1.15)
                model.chroma_std_strength.clamp_(0.0, 0.20)
                model.chroma_delta_limit.clamp_(8.0, 24.0)
                model.texture_strength.clamp_(0.20, 1.0)
                model.alpha_strength.clamp_(0.50, 1.10)
            running_loss += float(loss.item())
            running_steps += 1
            progress.set_postfix(loss=float(loss.item()), grad=float(grad_norm))

        model.eval()
        val_totals = {
            "loss": 0.0,
            "color_ab_loss": 0.0,
            "color_rgb_loss": 0.0,
            "hair_luma_keep_loss": 0.0,
            "lock_rgb_keep_loss": 0.0,
            "lock_non_dark_loss": 0.0,
        }
        total_steps = 0
        preview_rows = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"SG-IDCT val {epoch + 1}/{USER_EPOCHS}", leave=False):
                _, info = calc_loss(model, batch, device)
                for key in val_totals:
                    val_totals[key] += float(info[key].item())
                total_steps += 1
                if len(preview_rows) < USER_LOG_IMAGE_COUNT:
                    bsz = info["I_pred"].shape[0]
                    for idx in range(bsz):
                        preview_rows.append([
                            info["I_color"][idx],
                            info["I_satd"][idx],
                            info["I_pred"][idx],
                        ])
                        if len(preview_rows) >= USER_LOG_IMAGE_COUNT:
                            break

        avg = {key: value / max(total_steps, 1) for key, value in val_totals.items()}
        train_loss = running_loss / max(running_steps, 1)
        print(
            f"[sg_idct_v16] epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={avg['loss']:.6f} "
            f"val_ab={avg['color_ab_loss']:.6f} "
            f"val_rgb={avg['color_rgb_loss']:.6f} "
            f"val_hair_luma={avg['hair_luma_keep_loss']:.6f} "
            f"val_lock={avg['lock_rgb_keep_loss']:.6f} "
            f"val_lock_non_dark={avg['lock_non_dark_loss']:.6f}"
        )

        if epoch % USER_SAVE_PREVIEW_EVERY == 0 and preview_rows:
            save_preview(ACTIVE_OUTPUT_DIR / "val_images" / f"epoch_{epoch + 1:03d}.png", preview_rows)
        if (epoch + 1) % USER_SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(ACTIVE_OUTPUT_DIR / "checkpoints" / "last.pth", model, optimizer, epoch + 1, best_loss)
        if avg["loss"] <= best_loss:
            best_loss = avg["loss"]
            save_checkpoint(ACTIVE_OUTPUT_DIR / "checkpoints" / "best.pth", model, optimizer, epoch + 1, best_loss)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
