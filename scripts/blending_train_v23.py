import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ColorTransfer_v23 import V23ColorTransfer, masked_mean_std, rgb_to_lab


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("input/blending_dataset_v23")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_train_v23")
USER_VAL_SIZE_FFHQ = 512

USER_DATASET_DIR_SMALL = Path("input/blending_dataset_v23_small")
USER_OUTPUT_DIR_SMALL = Path("output/blending_train_v23_small")
USER_VAL_SIZE_SMALL = 64

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 16
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_EPOCHS = 12
USER_LR = 3e-3
USER_WEIGHT_DECAY = 0.0
USER_GRAD_CLIP = 1.0

USER_V23_WORK_SIZE = 256
USER_V23_BINS = 7
USER_V23_AB_STRENGTH = 1.0
USER_V23_AB_STD_STRENGTH = 0.12
USER_V23_MAIN_COLOR_STRENGTH = 0.18
USER_V23_CHROMA_FLOOR_STRENGTH = 0.72
USER_V23_CHROMA_FLOOR_MIN_REF = 8.0
USER_V23_LUMA_BOOST_SCALE = 0.18
USER_V23_DARKEN_SCALE = 0.16
USER_V23_MAX_LUMA_SHIFT = 8.0
USER_V23_MIN_LUMA_SHIFT = -4.0
USER_V23_SHADOW_THRESHOLD = 18.0
USER_V23_HIGHLIGHT_THRESHOLD = 84.0
USER_V23_EXTREME_LUMA_CHROMA_SCALE = 0.68
USER_V23_EXTREME_LUMA_SHIFT_SCALE = 0.20
USER_V23_TARGET_ERODE = 1
USER_V23_TARGET_BAND_DILATE = 1
USER_V23_BAND_ALPHA = 0.45
USER_V23_SOFT_LOCK_STRENGTH = 0.65
USER_V23_ALPHA_BLUR_RADIUS = 3
USER_V23_REF_ERODE = 2
USER_V23_REF_L_MIN = 5.0
USER_V23_REF_L_MAX = 95.0
USER_V23_REF_MIN_SAT = 0.025

USER_LAMBDA_AB_MEAN = 1.00
USER_LAMBDA_AB_STD = 0.35
USER_LAMBDA_AB_HIST = 0.35
USER_LAMBDA_AB_DIRECTION = 0.70
USER_LAMBDA_LUMA_KEEP = 1.40
USER_LAMBDA_LUMA_GRAD = 0.80
USER_LAMBDA_PRESERVE = 5.00
USER_LAMBDA_PARAM_REG = 0.04

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_LOG_IMAGE_COUNT = 24
USER_RESUME_CHECKPOINT = ""
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def role_key(role: str, stem: str) -> str:
    return f"{role}__{stem}"


def cache_name(face_name: str, shape_name: str, color_name: str) -> str:
    return f"{role_key('face', face_name)}_{role_key('shape', shape_name)}_{role_key('color', color_name)}.npz"


def read_triplets(dataset_dir: Path) -> list[tuple[str, str, str]]:
    path = dataset_dir / "dataset.exps"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Run scripts/blending_gen_v23.py first.")
    triplets = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            items = line.strip().split()
            if len(items) == 3:
                triplets.append((items[0], items[1], items[2]))
    return triplets


def masked_l1(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().clamp(0, 1)
    if mask.size(1) == 1 and source.size(1) != 1:
        mask = mask.expand(-1, source.size(1), -1, -1)
    return ((source - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def masked_soft_histogram(values: torch.Tensor, mask: torch.Tensor, bins: int, value_min: float, value_max: float) -> torch.Tensor:
    values = values.flatten(2)
    mask = mask.float().clamp(0, 1)
    if mask.size(1) == 1 and values.size(1) != 1:
        mask = mask.expand(-1, values.size(1), -1, -1)
    mask = mask.flatten(2)
    centers = torch.linspace(value_min, value_max, steps=bins, device=values.device, dtype=values.dtype)
    centers = centers.view(1, 1, 1, bins)
    sigma = (value_max - value_min) / max(bins - 1, 1)
    weights = torch.exp(-0.5 * ((values.unsqueeze(-1) - centers) / max(sigma, 1e-6)).square())
    weights = weights * mask.unsqueeze(-1)
    hist = weights.sum(dim=2)
    return hist / hist.sum(dim=2, keepdim=True).clamp_min(1e-6)


def masked_hist_loss(
    pred_features: torch.Tensor,
    pred_mask: torch.Tensor,
    ref_features: torch.Tensor,
    ref_mask: torch.Tensor,
    bins: int,
    value_min: float,
    value_max: float,
) -> torch.Tensor:
    pred_hist = masked_soft_histogram(pred_features, pred_mask, bins, value_min, value_max)
    ref_hist = masked_soft_histogram(ref_features, ref_mask, bins, value_min, value_max)
    return F.l1_loss(pred_hist, ref_hist)


def masked_gradient_l1(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().clamp(0, 1)
    if mask.size(1) == 1 and source.size(1) != 1:
        mask = mask.expand(-1, source.size(1), -1, -1)

    source_dx = source[:, :, :, 1:] - source[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    source_dy = source[:, :, 1:, :] - source[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    loss_x = ((source_dx - target_dx).abs() * mask_x).sum() / mask_x.sum().clamp_min(1.0)
    loss_y = ((source_dy - target_dy).abs() * mask_y).sum() / mask_y.sum().clamp_min(1.0)
    return loss_x + loss_y


def ab_direction_loss(out_ab_mean: torch.Tensor, ref_ab_mean: torch.Tensor) -> torch.Tensor:
    out_vec = out_ab_mean.flatten(1)
    ref_vec = ref_ab_mean.flatten(1)
    out_norm = out_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ref_norm = ref_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
    active = (ref_norm >= 5.0).float()
    cosine = (out_vec * ref_vec).sum(dim=1, keepdim=True) / (out_norm * ref_norm)
    cosine_loss = (1.0 - cosine.clamp(-1, 1)) * active
    ref_dir = ref_vec / ref_norm
    projection = (out_vec * ref_dir).sum(dim=1, keepdim=True)
    projection_loss = torch.relu(0.82 * ref_norm - projection) / ref_norm
    loss = cosine_loss + projection_loss * active
    return loss.sum() / active.sum().clamp_min(1.0)


def mask_to_preview(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float().clamp(0, 1)
    if mask.size(1) == 1:
        mask = mask.repeat(1, 3, 1, 1)
    return mask


def save_preview(path: Path, rows: list[list[torch.Tensor]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    for row in rows:
        tiles = []
        for tensor in row:
            if tensor.dim() == 4:
                tensor = tensor[0]
            tiles.append(tensor.detach().cpu().clamp(0, 1))
        panels.append(torch.cat(tiles, dim=2))
    save_image(torch.cat(panels, dim=1), path)


class V23Dataset(Dataset):
    def __init__(self, triplets: list[tuple[str, str, str]], dataset_dir: Path):
        self.triplets = triplets
        self.cache_dir = dataset_dir / "V23"

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index: int):
        face_name, shape_name, color_name = self.triplets[index]
        path = self.cache_dir / cache_name(face_name, shape_name, color_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing v23 cache: {path}. Run scripts/blending_gen_v23.py again.")
        with np.load(path) as data:
            satd_256 = torch.from_numpy(data["satd_256"]).float()
            color_256 = torch.from_numpy(data["color_256"]).float()
            target_hair_mask = torch.from_numpy(data["target_hair_mask"]).float()
            reference_hair_mask = torch.from_numpy(data["reference_hair_mask"]).float()
            hard_lock_mask = torch.from_numpy(data["hard_lock_mask"]).float()
            soft_lock_mask = torch.from_numpy(data["soft_lock_mask"]).float()
        return satd_256, color_256, target_hair_mask, reference_hair_mask, hard_lock_mask, soft_lock_mask


def v23_config() -> dict[str, object]:
    return {
        "work_size": USER_V23_WORK_SIZE,
        "bins": USER_V23_BINS,
        "ab_strength": USER_V23_AB_STRENGTH,
        "ab_std_strength": USER_V23_AB_STD_STRENGTH,
        "main_color_strength": USER_V23_MAIN_COLOR_STRENGTH,
        "chroma_floor_strength": USER_V23_CHROMA_FLOOR_STRENGTH,
        "chroma_floor_min_ref": USER_V23_CHROMA_FLOOR_MIN_REF,
        "luma_boost_scale": USER_V23_LUMA_BOOST_SCALE,
        "darken_scale": USER_V23_DARKEN_SCALE,
        "max_luma_shift": USER_V23_MAX_LUMA_SHIFT,
        "min_luma_shift": USER_V23_MIN_LUMA_SHIFT,
        "shadow_threshold": USER_V23_SHADOW_THRESHOLD,
        "highlight_threshold": USER_V23_HIGHLIGHT_THRESHOLD,
        "extreme_luma_chroma_scale": USER_V23_EXTREME_LUMA_CHROMA_SCALE,
        "extreme_luma_shift_scale": USER_V23_EXTREME_LUMA_SHIFT_SCALE,
        "target_erode": USER_V23_TARGET_ERODE,
        "target_band_dilate": USER_V23_TARGET_BAND_DILATE,
        "band_alpha": USER_V23_BAND_ALPHA,
        "soft_lock_strength": USER_V23_SOFT_LOCK_STRENGTH,
        "alpha_blur_radius": USER_V23_ALPHA_BLUR_RADIUS,
        "ref_erode": USER_V23_REF_ERODE,
        "ref_l_min": USER_V23_REF_L_MIN,
        "ref_l_max": USER_V23_REF_L_MAX,
        "ref_min_sat": USER_V23_REF_MIN_SAT,
    }


def build_model(learnable: bool = True) -> V23ColorTransfer:
    return V23ColorTransfer(**v23_config(), learnable=learnable)


class V23Trainer:
    def __init__(self, model: V23ColorTransfer, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
        self.output_ckpt_dir = ACTIVE_OUTPUT_DIR / "checkpoints"
        self.output_val_dir = ACTIVE_OUTPUT_DIR / "val_images"
        self.output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.initial_params = {name: value.detach().clone() for name, value in self.model.named_parameters()}

    def clamp_parameters(self) -> None:
        ranges = {
            "ab_strength": (0.55, 1.35),
            "ab_std_strength": (0.00, 0.50),
            "main_color_strength": (0.00, 0.55),
            "chroma_floor_strength": (0.35, 1.05),
            "luma_boost_scale": (0.00, 0.40),
            "darken_scale": (0.00, 0.40),
            "max_luma_shift": (2.00, 12.0),
            "min_luma_shift": (-8.0, 0.0),
            "extreme_luma_chroma_scale": (0.45, 1.00),
            "extreme_luma_shift_scale": (0.00, 0.60),
            "band_alpha": (0.15, 0.75),
            "soft_lock_strength": (0.25, 0.95),
        }
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in ranges:
                    lo, hi = ranges[name]
                    parameter.clamp_(lo, hi)

    def parameter_reg_loss(self) -> torch.Tensor:
        reg = torch.zeros((), device=self.device)
        for name, parameter in self.model.named_parameters():
            reg = reg + (parameter - self.initial_params[name].to(self.device)).square()
        return reg

    def prepare_batch(self, batch):
        return [item.to(self.device, non_blocking=True) for item in batch]

    def calc_loss(self, satd_256, color_256, target_hair_mask, reference_hair_mask, hard_lock_mask, soft_lock_mask):
        output, aux = self.model(
            satd_256,
            color_256,
            target_hair_mask,
            reference_hair_mask,
            hard_lock_mask,
            soft_lock_mask,
            return_aux=True,
        )
        core = aux["core"]
        ref_mask = aux["ref_mask"]
        alpha = aux["alpha"]
        hard_lock = aux["hard_lock"]

        out_lab = rgb_to_lab(output)
        ref_lab = rgb_to_lab(color_256)
        satd_lab = rgb_to_lab(satd_256)
        out_l, out_ab = out_lab[:, 0:1], out_lab[:, 1:3]
        ref_ab = ref_lab[:, 1:3]
        satd_l = satd_lab[:, 0:1]

        out_ab_mean, out_ab_std = masked_mean_std(out_ab, core)
        ref_ab_mean, ref_ab_std = masked_mean_std(ref_ab, ref_mask)
        ab_mean_loss = F.l1_loss(out_ab_mean, ref_ab_mean)
        ab_std_loss = F.l1_loss(out_ab_std, ref_ab_std)
        ab_hist_loss = masked_hist_loss(out_ab, core, ref_ab, ref_mask, bins=14, value_min=-110.0, value_max=110.0)
        ab_dir_loss = ab_direction_loss(out_ab_mean, ref_ab_mean)

        luma_keep_loss = masked_l1(out_l, satd_l, core)
        luma_grad_loss = masked_gradient_l1(out_l, satd_l, core)
        preserve_mask = ((1.0 - alpha) + hard_lock).clamp(0, 1)
        preserve_loss = masked_l1(output, satd_256, preserve_mask)
        reg_loss = self.parameter_reg_loss()

        total = (
            USER_LAMBDA_AB_MEAN * ab_mean_loss
            + USER_LAMBDA_AB_STD * ab_std_loss
            + USER_LAMBDA_AB_HIST * ab_hist_loss
            + USER_LAMBDA_AB_DIRECTION * ab_dir_loss
            + USER_LAMBDA_LUMA_KEEP * luma_keep_loss
            + USER_LAMBDA_LUMA_GRAD * luma_grad_loss
            + USER_LAMBDA_PRESERVE * preserve_loss
            + USER_LAMBDA_PARAM_REG * reg_loss
        )
        return total, {
            "loss": total,
            "ab_mean": ab_mean_loss,
            "ab_std": ab_std_loss,
            "ab_hist": ab_hist_loss,
            "ab_dir": ab_dir_loss,
            "luma_keep": luma_keep_loss,
            "luma_grad": luma_grad_loss,
            "preserve": preserve_loss,
            "reg": reg_loss,
        }, output, aux

    def save_checkpoint(self, epoch: int, name: str) -> None:
        torch.save(
            {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "v23_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": v23_config(),
            },
            self.output_ckpt_dir / f"{name}.pth",
        )

    def load_resume_checkpoint(self) -> int:
        if not USER_RESUME_CHECKPOINT:
            return 0
        path = Path(USER_RESUME_CHECKPOINT)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find USER_RESUME_CHECKPOINT: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get("v23_state_dict", checkpoint.get("model_state_dict", checkpoint)), strict=False)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = float(checkpoint.get("best_loss", self.best_loss))
        return int(checkpoint.get("epoch", 0))

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        progress = tqdm(self.train_loader, desc=f"V23 train {epoch + 1}/{USER_EPOCHS}", leave=False)
        for batch in progress:
            satd_256, color_256, target_hair_mask, reference_hair_mask, hard_lock_mask, soft_lock_mask = self.prepare_batch(batch)
            loss, loss_info, _output, _aux = self.calc_loss(
                satd_256,
                color_256,
                target_hair_mask,
                reference_hair_mask,
                hard_lock_mask,
                soft_lock_mask,
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
            self.optimizer.step()
            self.clamp_parameters()

            total_loss += float(loss.item())
            steps += 1
            progress.set_postfix(
                loss=float(loss.item()),
                ab=float(loss_info["ab_mean"].item()),
                direction=float(loss_info["ab_dir"].item()),
                luma=float(loss_info["luma_keep"].item()),
                grad=float(grad_norm),
            )
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        totals = {
            "loss": 0.0,
            "ab_mean": 0.0,
            "ab_std": 0.0,
            "ab_hist": 0.0,
            "ab_dir": 0.0,
            "luma_keep": 0.0,
            "luma_grad": 0.0,
            "preserve": 0.0,
            "reg": 0.0,
        }
        steps = 0
        preview_rows = []
        for batch in tqdm(self.val_loader, desc=f"V23 val {epoch + 1}/{USER_EPOCHS}", leave=False):
            satd_256, color_256, target_hair_mask, reference_hair_mask, hard_lock_mask, soft_lock_mask = self.prepare_batch(batch)
            _loss, loss_info, output, aux = self.calc_loss(
                satd_256,
                color_256,
                target_hair_mask,
                reference_hair_mask,
                hard_lock_mask,
                soft_lock_mask,
            )
            for key in totals:
                totals[key] += float(loss_info[key].item())
            steps += 1

            if len(preview_rows) < USER_LOG_IMAGE_COUNT:
                bsz = satd_256.size(0)
                for idx in range(bsz):
                    preview_rows.append(
                        [
                            satd_256[idx : idx + 1],
                            color_256[idx : idx + 1],
                            output[idx : idx + 1],
                            mask_to_preview(aux["alpha"][idx : idx + 1]),
                            mask_to_preview(aux["hard_lock"][idx : idx + 1]),
                            mask_to_preview(aux["soft_lock"][idx : idx + 1]),
                        ]
                    )
                    if len(preview_rows) >= USER_LOG_IMAGE_COUNT:
                        break

        avg = {key: value / max(steps, 1) for key, value in totals.items()}
        if epoch % USER_SAVE_PREVIEW_EVERY == 0 and preview_rows:
            save_preview(self.output_val_dir / f"epoch_{epoch + 1:03d}.png", preview_rows)
        print(
            f"[v23] epoch={epoch + 1} val_loss={avg['loss']:.6f} "
            f"val_ab={avg['ab_mean']:.6f} val_dir={avg['ab_dir']:.6f} "
            f"val_luma_keep={avg['luma_keep']:.6f} val_preserve={avg['preserve']:.6f}"
        )
        return avg["loss"]

    def train_loop(self) -> None:
        start_epoch = self.load_resume_checkpoint()
        for epoch in range(start_epoch, USER_EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            print(f"[v23] epoch={epoch + 1} train_loss={train_loss:.6f}")
            if (epoch + 1) % USER_SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(epoch + 1, "last")
            if val_loss <= self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch + 1, "best")


def main() -> None:
    set_seed(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    triplets = read_triplets(ACTIVE_DATASET_DIR)
    if len(triplets) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(f"dataset.exps has {len(triplets)} rows, but val_size={ACTIVE_VAL_SIZE}.")

    train_triplets, val_triplets = train_test_split(triplets, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_loader = DataLoader(
        V23Dataset(train_triplets, ACTIVE_DATASET_DIR),
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        V23Dataset(val_triplets, ACTIVE_DATASET_DIR),
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
    )
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    model = build_model(learnable=True)
    trainer = V23Trainer(model, train_loader, val_loader, device)
    print(
        f"[v23] dataset={ACTIVE_DATASET_DIR} batch_size={USER_BATCH_SIZE} lr={USER_LR} "
        f"epochs={USER_EPOCHS} output={ACTIVE_OUTPUT_DIR}"
    )
    trainer.train_loop()


if __name__ == "__main__":
    main()
