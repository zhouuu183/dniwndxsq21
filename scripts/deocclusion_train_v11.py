from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.DeocclusionRepair_v11 import DeocclusionRepair_v11, load_deocclusion_repair_v11
from utils.deocclusion_masks_v11 import DEOCCLUSION_MASK_KEYS_V11, ensure_mask_4d
from utils.repair_losses_v11 import (
    alpha_tv_v11,
    chroma_delta_l1_v11,
    dark_residual_v11,
    masked_grad_l1_v11,
    masked_l1_v11,
    non_dark_v11,
)
from utils.train import seed_everything


# ========================= user config: edit only here =========================
USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_DATASET_DIR = Path("input/deocclusion_dataset_v11_dgrr_fill")
USER_OUTPUT_DIR = Path("output/deocclusion_train_v11_dgrr_fill")
USER_RESUME_CHECKPOINT = ""

USER_BATCH_SIZE = 8
USER_NUM_WORKERS = 0
USER_VAL_SIZE = 32
USER_EPOCHS = 40
USER_LR = 1e-4
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 1.0

USER_BASE_CHANNELS = 48
USER_DELTA_SCALE = 1.0
USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_PREVIEW_COUNT = 12

USER_DISABLE_SKIN = False
USER_DISABLE_STRUCT = False
USER_DISABLE_BG = False
USER_DISABLE_FILL = False
USER_DISABLE_CLEAN = True

USER_LAMBDA_REVEAL = 6.0
USER_LAMBDA_SKIN = 5.0
USER_LAMBDA_STRUCT = 5.0
USER_LAMBDA_FILL = 7.0
USER_LAMBDA_BG = 8.0
USER_LAMBDA_FILL_NONDARK = 1.0
USER_LAMBDA_FILL_DARK = 0.8
USER_LAMBDA_CLEAN = 1.5
USER_LAMBDA_CLEAN_CHROMA = 2.0
USER_LAMBDA_PRESERVE = 8.0
USER_LAMBDA_FACE_PRESERVE = 18.0
USER_LAMBDA_NON_DARK = 0.2
USER_LAMBDA_ALPHA_OUTSIDE = 0.2
USER_LAMBDA_ALPHA_TV = 0.05
USER_LAMBDA_FAR_CLEAN = 1.0
USER_LAMBDA_DELTA = 0.02
# ============================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_items(dataset_dir: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(dataset_dir.glob("*.dataset")):
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, list):
            items.extend(loaded)
        else:
            raise RuntimeError(f"Unsupported dataset chunk format: {path}")
    return items


class DeocclusionDatasetV11(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _mask_dict(raw_masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ref = next(iter(raw_masks.values()))
        masks = {}
        for key in DEOCCLUSION_MASK_KEYS_V11:
            value = raw_masks.get(key)
            if value is None:
                value = torch.zeros_like(ref)
            masks[key] = ensure_mask_4d(value).squeeze(0).float()
        return masks

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "source": item["source"].float(),
            "reference": item.get("reference", item["target"]).float(),
            "base": item["base"].float(),
            "target": item["target"].float(),
            "masks": self._mask_dict(item["masks"]),
            "clean_name": item.get("clean_name", ""),
            "donor_name": item.get("donor_name", ""),
        }


def collate_deocclusion_v11(batch: list[dict]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    masks = {key: torch.stack([sample["masks"][key] for sample in batch], dim=0) for key in DEOCCLUSION_MASK_KEYS_V11}
    return {
        "source": torch.stack([sample["source"] for sample in batch], dim=0),
        "reference": torch.stack([sample["reference"] for sample in batch], dim=0),
        "base": torch.stack([sample["base"] for sample in batch], dim=0),
        "target": torch.stack([sample["target"] for sample in batch], dim=0),
        "masks": masks,
    }


def resize_masks_to_image_v11(masks: dict[str, torch.Tensor], size: tuple[int, int]) -> dict[str, torch.Tensor]:
    resized = {}
    for key, value in masks.items():
        if value.shape[-2:] == size:
            resized[key] = value.float()
            continue
        mode = "bilinear" if (key.endswith("_soft") or key.startswith("E_")) else "nearest"
        align_corners = False if mode == "bilinear" else None
        resized[key] = F.interpolate(value.float(), size=size, mode=mode, align_corners=align_corners).clamp(0, 1)
    return resized


def calc_losses(pred: torch.Tensor, aux: dict[str, torch.Tensor], batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base = batch["base"]
    target = batch["target"]
    masks = batch["masks"]

    skin = masks["M_skin_soft"]
    struct = masks["M_struct_soft"]
    bg = masks["M_bg_soft"]
    fill = masks["M_fill_soft"]
    clean = masks["M_clean_soft"]
    safe = masks["M_safe"]
    face_preserve = masks["M_face_preserve"]
    reveal = (masks["M_reveal_soft"] * (1.0 - face_preserve)).clamp(0, 1)
    shadow = masks["M_shadow"]
    clean_far = masks["M_clean_far"]

    losses = {
        "reveal": masked_l1_v11(pred, target, reveal),
        "skin": masked_l1_v11(pred, target, skin) + 0.5 * masked_l1_v11(aux["skin_rgb"], target, skin),
        "struct": (
            masked_l1_v11(pred, target, struct)
            + 0.5 * masked_l1_v11(aux["struct_rgb"], target, struct)
            + 0.5 * masked_grad_l1_v11(pred, target, struct)
        ),
        "bg": masked_l1_v11(pred, target, bg) + 0.75 * masked_l1_v11(aux["bg_rgb"], target, bg),
        "fill": masked_l1_v11(pred, target, fill) + 0.5 * masked_l1_v11(aux["fill_rgb"], target, fill),
        "fill_nondark": non_dark_v11(base, pred, fill),
        "fill_dark": dark_residual_v11(pred, fill),
        "clean": masked_l1_v11(pred, target, clean) + 0.5 * masked_l1_v11(aux["clean_rgb"], target, clean),
        "clean_chroma": chroma_delta_l1_v11(pred, base, clean),
        "preserve": masked_l1_v11(pred, base, safe),
        "face_preserve": masked_l1_v11(pred, base, face_preserve),
        "non_dark": non_dark_v11(base, pred, shadow),
        "alpha_outside": (
            (aux["alpha_skin"] * (1.0 - skin)).mean()
            + (aux["alpha_struct"] * (1.0 - struct)).mean()
            + (aux["alpha_bg"] * (1.0 - bg)).mean()
            + (aux["alpha_fill"] * (1.0 - fill)).mean()
            + (aux["alpha_clean"] * (1.0 - clean)).mean()
            + (aux["alpha"] * safe).sum() / safe.sum().clamp(min=1.0)
            + (aux["alpha"] * face_preserve).sum() / face_preserve.sum().clamp(min=1.0)
        ),
        "alpha_tv": (
            alpha_tv_v11(aux["alpha_skin"], skin)
            + alpha_tv_v11(aux["alpha_struct"], struct)
            + alpha_tv_v11(aux["alpha_bg"], bg)
            + alpha_tv_v11(aux["alpha_fill"], fill)
            + alpha_tv_v11(aux["alpha_clean"], clean)
        ),
        "far_clean": (aux["alpha_clean"] * clean_far).sum() / clean_far.sum().clamp(min=1.0),
        "delta": (
            aux["delta_skin"].abs().mean()
            + aux["delta_struct"].abs().mean()
            + aux["delta_bg"].abs().mean()
            + aux["delta_fill"].abs().mean()
            + aux["delta_clean"].abs().mean()
        ) / 5.0,
    }
    w_skin = 0.0 if USER_DISABLE_SKIN else USER_LAMBDA_SKIN
    w_struct = 0.0 if USER_DISABLE_STRUCT else USER_LAMBDA_STRUCT
    w_bg = 0.0 if USER_DISABLE_BG else USER_LAMBDA_BG
    w_fill = 0.0 if USER_DISABLE_FILL else USER_LAMBDA_FILL
    w_fill_nondark = 0.0 if USER_DISABLE_FILL else USER_LAMBDA_FILL_NONDARK
    w_fill_dark = 0.0 if USER_DISABLE_FILL else USER_LAMBDA_FILL_DARK
    w_clean = 0.0 if USER_DISABLE_CLEAN else USER_LAMBDA_CLEAN
    w_clean_chroma = 0.0 if USER_DISABLE_CLEAN else USER_LAMBDA_CLEAN_CHROMA
    w_far_clean = 0.0 if USER_DISABLE_CLEAN else USER_LAMBDA_FAR_CLEAN

    total = (
        USER_LAMBDA_REVEAL * losses["reveal"]
        + w_skin * losses["skin"]
        + w_struct * losses["struct"]
        + w_bg * losses["bg"]
        + w_fill * losses["fill"]
        + w_fill_nondark * losses["fill_nondark"]
        + w_fill_dark * losses["fill_dark"]
        + w_clean * losses["clean"]
        + w_clean_chroma * losses["clean_chroma"]
        + USER_LAMBDA_PRESERVE * losses["preserve"]
        + USER_LAMBDA_FACE_PRESERVE * losses["face_preserve"]
        + USER_LAMBDA_NON_DARK * losses["non_dark"]
        + USER_LAMBDA_ALPHA_OUTSIDE * losses["alpha_outside"]
        + USER_LAMBDA_ALPHA_TV * losses["alpha_tv"]
        + w_far_clean * losses["far_clean"]
        + USER_LAMBDA_DELTA * losses["delta"]
    )
    losses["loss"] = total
    return total, losses


class TrainerV11:
    def __init__(self, train_loader, val_loader):
        self.device = USER_DEVICE if torch.cuda.is_available() else "cpu"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = DeocclusionRepair_v11(base_channels=USER_BASE_CHANNELS, delta_scale=USER_DELTA_SCALE).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
        self.ckpt_dir = USER_OUTPUT_DIR / "checkpoints"
        self.preview_dir = USER_OUTPUT_DIR / "previews"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def load_checkpoint(self, path: str) -> int:
        if not path:
            return 0
        checkpoint = torch.load(path, map_location=self.device)
        loaded, skipped = load_deocclusion_repair_v11(self.model, checkpoint, map_location=self.device)
        if skipped:
            print(f"skipped incompatible repair_v11 keys: {len(skipped)}")
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError:
                print("skipped incompatible optimizer state")
        self.best_loss = checkpoint.get("best_loss", self.best_loss)
        print(f"loaded repair_v11 keys: {len(loaded)}")
        return checkpoint.get("epoch", -1) + 1

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "repair_v11_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if epoch % USER_SAVE_CHECKPOINT_EVERY == 0:
            torch.save(state, self.ckpt_dir / "checkpoint.pth")
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")

    def _to_device(self, batch: dict) -> dict:
        moved = {
            "source": batch["source"].to(self.device),
            "reference": batch["reference"].to(self.device),
            "base": batch["base"].to(self.device),
            "target": batch["target"].to(self.device),
            "masks": {key: value.to(self.device) for key, value in batch["masks"].items()},
        }
        moved["masks"] = resize_masks_to_image_v11(moved["masks"], moved["base"].shape[-2:])
        return moved

    def run_epoch(self, loader, training: bool, epoch: int) -> dict[str, float]:
        self.model.train(training)
        totals = defaultdict(float)
        steps = 0
        previews = []
        context = torch.enable_grad() if training else torch.no_grad()
        to_pil = T.ToPILImage()

        with context:
            for batch in tqdm(loader, desc="train" if training else "val"):
                steps += 1
                batch = self._to_device(batch)
                pred, aux = self.model(
                    batch["source"],
                    batch["base"],
                    batch["masks"],
                    enable_skin=not USER_DISABLE_SKIN,
                    enable_struct=not USER_DISABLE_STRUCT,
                    enable_bg=not USER_DISABLE_BG,
                    enable_fill=not USER_DISABLE_FILL,
                    enable_clean=not USER_DISABLE_CLEAN,
                )
                loss, losses = calc_losses(pred, aux, batch)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
                    self.optimizer.step()

                for key, value in losses.items():
                    totals[key] += float(value.detach().cpu())

                if not training and len(previews) < USER_PREVIEW_COUNT:
                    max_add = min(batch["source"].shape[0], USER_PREVIEW_COUNT - len(previews))
                    for idx in range(max_add):
                        diff_rgb = (pred[idx] - batch["base"][idx]).abs().detach().cpu().mul(4.0).clamp(0, 1)
                        row = torch.cat(
                            [
                                batch["source"][idx].detach().cpu(),
                                batch["reference"][idx].detach().cpu(),
                                batch["base"][idx].detach().cpu(),
                                pred[idx].detach().cpu(),
                                diff_rgb,
                                batch["target"][idx].detach().cpu(),
                                aux["alpha_skin"][idx].detach().cpu().repeat(3, 1, 1),
                                aux["alpha_struct"][idx].detach().cpu().repeat(3, 1, 1),
                                aux["alpha_bg"][idx].detach().cpu().repeat(3, 1, 1),
                                aux["alpha_fill"][idx].detach().cpu().repeat(3, 1, 1),
                                aux["alpha_clean"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["M_skin_soft"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["M_struct_soft"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["M_bg_soft"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["M_fill_soft"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["M_clean_soft"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["M_face_preserve"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["R_ctx"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["R_bg_ctx"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["E_struct"][idx].detach().cpu().repeat(3, 1, 1),
                                batch["masks"]["D_t_norm"][idx].detach().cpu().repeat(3, 1, 1),
                            ],
                            dim=2,
                        ).clamp(0, 1)
                        previews.append(row)

        if (not training) and epoch % USER_SAVE_PREVIEW_EVERY == 0 and previews:
            save_dir = self.preview_dir / f"epoch_{epoch:03d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "columns.txt", "w", encoding="utf-8") as file:
                file.write(
                    "source | reference | base | pred | abs(pred-base)*4 | target | "
                    "alpha_skin | alpha_struct | alpha_bg | alpha_fill | alpha_clean | "
                    "M_skin | M_struct | M_bg | M_fill | M_clean | M_face_preserve | R_ctx | R_bg_ctx | E_struct | D_t_norm\n"
                )
            for idx, row in enumerate(previews):
                to_pil(row).save(save_dir / f"sample_{idx:03d}.png")

        return {key: value / max(steps, 1) for key, value in totals.items()}

    def train_loop(self):
        start_epoch = self.load_checkpoint(USER_RESUME_CHECKPOINT)
        for epoch in range(start_epoch, USER_EPOCHS):
            train_losses = self.run_epoch(self.train_loader, training=True, epoch=epoch)
            val_losses = self.run_epoch(self.val_loader, training=False, epoch=epoch)
            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_losses.get('loss', 0):.4f} "
                f"val_loss={val_losses.get('loss', 0):.4f}"
            )
            is_best = val_losses.get("loss", float("inf")) <= self.best_loss
            if is_best:
                self.best_loss = val_losses["loss"]
            self.save_checkpoint(epoch, is_best=is_best)


def main():
    seed_everything(USER_RANDOM_SEED)
    set_seed(USER_RANDOM_SEED)
    items = load_dataset_items(USER_DATASET_DIR)
    if len(items) <= USER_VAL_SIZE:
        raise RuntimeError(f"Not enough dataset items in {USER_DATASET_DIR}: {len(items)}")
    train_items, val_items = train_test_split(items, test_size=USER_VAL_SIZE, random_state=USER_RANDOM_SEED)

    train_loader = DataLoader(
        DeocclusionDatasetV11(train_items),
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        collate_fn=collate_deocclusion_v11,
        drop_last=True,
    )
    val_loader = DataLoader(
        DeocclusionDatasetV11(val_items),
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        collate_fn=collate_deocclusion_v11,
        drop_last=False,
    )
    print(f"dataset dir: {USER_DATASET_DIR}")
    print(f"train items: {len(train_items)} | val items: {len(val_items)}")
    TrainerV11(train_loader, val_loader).train_loop()


if __name__ == "__main__":
    main()
