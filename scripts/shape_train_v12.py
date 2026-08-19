from __future__ import annotations

import os
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/shape_dataset_v12_3000")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/shape_train_v12_3000")
USER_VAL_SIZE_FFHQ = 256

USER_DATASET_DIR_SMALL = Path("images/shape_dataset_v12_small")
USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_OUTPUT_DIR_SMALL = Path("output/shape_train_v12_small")
USER_VAL_SIZE_SMALL = 30

USER_RANDOM_SEED = 3407
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_INIT_CKPT = ""
USER_RESUME_CHECKPOINT = ""

USER_BATCH_SIZE = 4
USER_EFFECTIVE_BATCH_SIZE = 16
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_CPU_THREADS = 1
USER_DISABLE_CUDNN_BENCHMARK = True

USER_EPOCHS = 30
USER_LR = 1e-4
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 1.0

USER_SHAPE_ADAPTER_HIDDEN_V12 = 512
USER_SHAPE_ADAPTER_STRENGTH_V12 = 1.0
USER_SHAPE_PRIOR_STRENGTH_V12 = 0.0

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_PREVIEW_COUNT = 24

USER_LAMBDA_LATENT_SHAPE = 0.45
USER_LAMBDA_LATENT_PRESERVE = 2.00
USER_LAMBDA_LATENT_REMOVE_PRESERVE = 1.50
USER_LAMBDA_IMAGE_SHAPE = 0.0
USER_LAMBDA_IMAGE_EDGE = 0.25
USER_LAMBDA_IMAGE_PRESERVE = 1.60
USER_LAMBDA_IMAGE_REMOVE_PRESERVE = 1.20
USER_LAMBDA_BOUNDARY_TONE_PRESERVE = 1.20
USER_LAMBDA_NO_WHITE_BOUNDARY = 1.50
USER_LAMBDA_DELTA = 0.04
USER_NO_WHITE_MARGIN_V12 = 0.035
# ========================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Net import Net
from models.ShapeAdapter_v12 import TopologyF32Adapter_v12, V12_MASK_KEYS, load_shape_adapter_v12, masked_l1
from utils.bicubic import BicubicDownSample
from utils.train import seed_everything, toggle_grad


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        },
        "small": {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
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


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_DATASET_DIR = _DATASET_CFG["dataset_dir"]
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_VAL_SIZE = _DATASET_CFG["val_size"]


torch.set_num_threads(max(1, USER_CPU_THREADS))
if USER_DISABLE_CUDNN_BENCHMARK and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
if USER_EFFECTIVE_BATCH_SIZE < USER_BATCH_SIZE or USER_EFFECTIVE_BATCH_SIZE % USER_BATCH_SIZE != 0:
    raise RuntimeError("USER_EFFECTIVE_BATCH_SIZE must be >= USER_BATCH_SIZE and divisible by it.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_opts(device: torch.device) -> Namespace:
    return Namespace(
        size=1024,
        ckpt=USER_STYLEGAN_CKPT,
        channel_multiplier=2,
        latent=512,
        n_mlp=8,
        device=str(device),
        batch_size=1,
        save_all=False,
        save_all_dir=Path("output"),
        mixing=0.95,
        smooth=5,
        rotate_checkpoint=USER_ROTATE_CKPT,
    )


def resolve_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find image for {stem} under {root}")


def load_image_256(root: Path, stem: str) -> torch.Tensor:
    with Image.open(resolve_image_path(root, stem)) as image:
        image = image.convert("RGB").resize((256, 256), Image.BICUBIC)
        return T.functional.to_tensor(image) * 2.0 - 1.0


def load_npz(path: Path, key: str) -> torch.Tensor:
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain key {key!r}")
        return torch.from_numpy(data[key]).float().squeeze(0)


def load_exps(dataset_dir: Path) -> list[tuple[str, str, str]]:
    exps = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as file:
        for line in file:
            items = line.strip().split()
            if len(items) == 3:
                exps.append((items[0], items[1], items[2]))
    return exps


def gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    g = gray(image)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(g, kernel_x, padding=1)
    grad_y = F.conv2d(g, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


class ShapeDatasetV12(Dataset):
    def __init__(self, exps: list[tuple[str, str, str]], dataset_dir: Path, face_root: Path, shape_root: Path):
        self.exps = exps
        self.dataset_dir = dataset_dir
        self.face_root = face_root
        self.shape_root = shape_root

    def __len__(self):
        return len(self.exps)

    def _align_shape_path(self, face_stem: str, shape_stem: str) -> Path:
        return self.dataset_dir / "AlignShape" / f"face_{face_stem}__alignshape_{shape_stem}.npz"

    def __getitem__(self, idx):
        face_stem, shape_stem, color_stem = self.exps[idx]
        face_fs = self.dataset_dir / "FS" / f"face_{face_stem}.npz"
        shape_fs = self.dataset_dir / "FS" / f"shape_{shape_stem}.npz"
        align_path = self._align_shape_path(face_stem, shape_stem)

        masks = {key: load_npz(align_path, key) for key in V12_MASK_KEYS}
        return {
            "face_stem": face_stem,
            "shape_stem": shape_stem,
            "color_stem": color_stem,
            "face_s": load_npz(face_fs, "latent_S"),
            "F_base": load_npz(align_path, "latent_F_base"),
            "F_src": load_npz(align_path, "latent_F_src"),
            "F_shape": load_npz(align_path, "latent_F_shape"),
            "face_i": load_image_256(self.face_root, face_stem),
            "shape_i": load_image_256(self.shape_root, shape_stem),
            "masks": masks,
        }


def collate_shape_v12(batch):
    masks = {key: torch.stack([item["masks"][key] for item in batch], dim=0) for key in V12_MASK_KEYS}
    return {
        "face_stem": [item["face_stem"] for item in batch],
        "shape_stem": [item["shape_stem"] for item in batch],
        "color_stem": [item["color_stem"] for item in batch],
        "face_s": torch.stack([item["face_s"] for item in batch], dim=0),
        "F_base": torch.stack([item["F_base"] for item in batch], dim=0),
        "F_src": torch.stack([item["F_src"] for item in batch], dim=0),
        "F_shape": torch.stack([item["F_shape"] for item in batch], dim=0),
        "face_i": torch.stack([item["face_i"] for item in batch], dim=0),
        "shape_i": torch.stack([item["shape_i"] for item in batch], dim=0),
        "masks": masks,
    }


class TrainerV12:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader):
        self.device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = ACTIVE_OUTPUT_DIR
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.preview_dir = self.output_dir / "previews"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        opts = build_opts(self.device)
        self.net = Net(opts)
        toggle_grad(self.net.generator, False)
        self.net.generator.eval()
        self.downsample_256 = BicubicDownSample(factor=4, cuda="cuda" in str(self.device))

        self.model = TopologyF32Adapter_v12(hidden_channels=USER_SHAPE_ADAPTER_HIDDEN_V12).to(self.device)
        if USER_INIT_CKPT:
            checkpoint = torch.load(USER_INIT_CKPT, map_location=self.device)
            loaded, skipped = load_shape_adapter_v12(self.model, checkpoint)
            print(f"loaded init adapter keys: {len(loaded)} | skipped incompatible: {len(skipped)}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
        self.grad_accum_steps = max(1, USER_EFFECTIVE_BATCH_SIZE // USER_BATCH_SIZE)
        self.best_loss = float("inf")
        self.cur_iter = 0

    def _to_device(self, batch: dict) -> dict:
        return {
            "face_s": batch["face_s"].to(self.device),
            "F_base": batch["F_base"].to(self.device),
            "F_src": batch["F_src"].to(self.device),
            "F_shape": batch["F_shape"].to(self.device),
            "face_i": batch["face_i"].to(self.device),
            "shape_i": batch["shape_i"].to(self.device),
            "masks": {key: value.to(self.device) for key, value in batch["masks"].items()},
        }

    def render_256(self, latent_s: torch.Tensor, latent_f: torch.Tensor) -> torch.Tensor:
        image, _ = self.net.generator(
            [latent_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_f,
        )
        return self.downsample_256(image)

    def forward_batch(self, batch: dict) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        out = self.model(
            F_base=batch["F_base"],
            F_src=batch["F_src"],
            F_shape=batch["F_shape"],
            masks_256=batch["masks"],
            strength=USER_SHAPE_ADAPTER_STRENGTH_V12,
            shape_prior_strength=USER_SHAPE_PRIOR_STRENGTH_V12,
        )
        pred_256 = self.render_256(batch["face_s"], out["latent_F_refined"])
        with torch.no_grad():
            baseline_256 = self.render_256(batch["face_s"], batch["F_base"])
        out.update({"pred_256": pred_256, "baseline_256": baseline_256})
        return out

    def calc_losses(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks = batch["masks"]
        masks32 = outputs["masks32"]
        pred = outputs["pred_256"]
        baseline = outputs["baseline_256"]
        shape = batch["shape_i"]

        shape32 = (masks32["M_add"] + masks32["M_boundary"] + masks32["M_bang"] + 0.25 * masks32["M_keep"]).clamp(0, 1)
        remove32 = masks32["M_remove"].clamp(0, 1)
        preserve32 = (1.0 - (shape32 + remove32).clamp(0, 1)).clamp(0, 1)
        edit256 = (masks["M_add"] + masks["M_boundary"] + masks["M_bang"]).clamp(0, 1)
        remove256 = masks["M_remove"].clamp(0, 1)
        preserve256 = (masks["M_face_protect"] + (1.0 - masks["M_edit"]).clamp(0, 1)).clamp(0, 1)
        boundary256 = (masks["M_boundary"] + masks["M_bang"]).clamp(0, 1)
        boundary_tone256 = (masks["M_boundary"] + masks["M_bang"] + 0.35 * masks["M_keep"]).clamp(0, 1)

        latent_shape = masked_l1(outputs["latent_F_refined"], batch["F_shape"], shape32)
        latent_preserve = masked_l1(outputs["latent_F_refined"], batch["F_base"], preserve32)
        latent_remove_preserve = masked_l1(outputs["latent_F_refined"], batch["F_base"], remove32)
        image_shape = masked_l1(pred, shape, edit256)
        image_edge = masked_l1(sobel_edges(pred), sobel_edges(shape), boundary256)
        image_preserve = masked_l1(pred, baseline.detach(), preserve256)
        image_remove_preserve = masked_l1(pred, baseline.detach(), remove256)
        boundary_tone_preserve = masked_l1(gray(pred), gray(baseline.detach()), boundary_tone256)
        no_white_boundary = (
            F.relu(gray(pred) - gray(baseline.detach()) - USER_NO_WHITE_MARGIN_V12) * boundary_tone256
        ).sum() / boundary_tone256.sum().clamp(min=1.0)
        delta = outputs["learned_delta_F32"].abs().mean()

        losses = {
            "latent_shape": USER_LAMBDA_LATENT_SHAPE * latent_shape,
            "latent_preserve": USER_LAMBDA_LATENT_PRESERVE * latent_preserve,
            "latent_remove_preserve": USER_LAMBDA_LATENT_REMOVE_PRESERVE * latent_remove_preserve,
            "image_shape": USER_LAMBDA_IMAGE_SHAPE * image_shape,
            "image_edge": USER_LAMBDA_IMAGE_EDGE * image_edge,
            "image_preserve": USER_LAMBDA_IMAGE_PRESERVE * image_preserve,
            "image_remove_preserve": USER_LAMBDA_IMAGE_REMOVE_PRESERVE * image_remove_preserve,
            "boundary_tone_preserve": USER_LAMBDA_BOUNDARY_TONE_PRESERVE * boundary_tone_preserve,
            "no_white_boundary": USER_LAMBDA_NO_WHITE_BOUNDARY * no_white_boundary,
            "delta": USER_LAMBDA_DELTA * delta,
        }
        losses["loss"] = sum(losses.values())
        return losses["loss"], losses

    def save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "shape_adapter_v12_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "cur_iter": self.cur_iter,
        }
        torch.save(state, self.ckpt_dir / "last.pth")
        if epoch % USER_SAVE_CHECKPOINT_EVERY == 0:
            torch.save(state, self.ckpt_dir / "checkpoint.pth")
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
        torch.save({"shape_adapter_v12_state_dict": self.model.state_dict()}, self.ckpt_dir / "shape_adapter_v12_for_infer.pth")

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        return T.functional.to_pil_image(((image + 1) / 2).detach().cpu().clamp(0, 1))

    @staticmethod
    def _mask_pil(mask: torch.Tensor) -> Image.Image:
        mask = mask.detach().cpu().float().clamp(0, 1)
        if mask.dim() == 3:
            mask = mask[0]
        return T.functional.to_pil_image(mask)

    def save_preview(self, epoch: int, previews: list[dict]):
        if epoch % USER_SAVE_PREVIEW_EVERY != 0:
            return
        save_dir = self.preview_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(previews[:USER_PREVIEW_COUNT]):
            panels = [
                self._to_pil(row["face"]),
                self._to_pil(row["shape"]),
                self._to_pil(row["baseline"]),
                self._to_pil(row["pred"]),
                self._mask_pil(row["edit"]).convert("RGB"),
            ]
            width, height = panels[0].size
            canvas = Image.new("RGB", (width * len(panels), height), color=(255, 255, 255))
            for panel_idx, panel in enumerate(panels):
                canvas.paste(panel.convert("RGB"), (panel_idx * width, 0))
            canvas.save(save_dir / f"sample_{idx:03d}.png")

    def run_epoch(self, loader: DataLoader, training: bool, epoch: int) -> tuple[dict[str, float], list[dict]]:
        self.model.train(training)
        totals = defaultdict(float)
        previews = []
        steps = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for step, raw_batch in enumerate(tqdm(loader, desc="train" if training else "val")):
                steps += 1
                batch = self._to_device(raw_batch)
                outputs = self.forward_batch(batch)
                loss, losses = self.calc_losses(outputs, batch)

                if training:
                    (loss / self.grad_accum_steps).backward()
                    if (step + 1) % self.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.cur_iter += 1

                for key, value in losses.items():
                    totals[key] += float(value.detach().cpu())

                if not training and len(previews) < USER_PREVIEW_COUNT:
                    max_add = min(batch["face_i"].shape[0], USER_PREVIEW_COUNT - len(previews))
                    for idx in range(max_add):
                        previews.append(
                            {
                                "face": batch["face_i"][idx],
                                "shape": batch["shape_i"][idx],
                                "baseline": outputs["baseline_256"][idx],
                                "pred": outputs["pred_256"][idx],
                                "edit": batch["masks"]["M_edit"][idx],
                            }
                        )

                del batch, outputs, loss, losses

        if training and steps % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {key: value / max(steps, 1) for key, value in totals.items()}, previews

    def train_loop(self):
        start_epoch = 0
        if USER_RESUME_CHECKPOINT:
            checkpoint = torch.load(USER_RESUME_CHECKPOINT, map_location=self.device)
            loaded, skipped = load_shape_adapter_v12(self.model, checkpoint)
            print(f"loaded resume adapter keys: {len(loaded)} | skipped incompatible: {len(skipped)}")
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_loss = checkpoint.get("best_loss", self.best_loss)
            self.cur_iter = checkpoint.get("cur_iter", 0)
            start_epoch = checkpoint.get("epoch", -1) + 1

        for epoch in range(start_epoch, USER_EPOCHS):
            train_losses, _ = self.run_epoch(self.train_loader, training=True, epoch=epoch)
            val_losses, previews = self.run_epoch(self.val_loader, training=False, epoch=epoch)
            self.save_preview(epoch, previews)
            is_best = val_losses["loss"] <= self.best_loss
            if is_best:
                self.best_loss = val_losses["loss"]
            self.save_checkpoint(epoch, is_best=is_best)
            print(
                f"[epoch {epoch:03d}] "
                f"train={train_losses['loss']:.6f} val={val_losses['loss']:.6f} best={self.best_loss:.6f}"
            )


def main():
    seed_everything(USER_RANDOM_SEED)
    set_seed(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exps = load_exps(ACTIVE_DATASET_DIR)
    if len(exps) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(f"Not enough samples in {ACTIVE_DATASET_DIR}: {len(exps)}")
    train_exps, val_exps = train_test_split(exps, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = ShapeDatasetV12(train_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT)
    val_dataset = ShapeDatasetV12(val_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_shape_v12,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_shape_v12,
        drop_last=False,
    )

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"train: {len(train_dataset)} | val: {len(val_dataset)}")
    TrainerV12(train_loader, val_loader).train_loop()


if __name__ == "__main__":
    main()
