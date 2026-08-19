import os
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from losses.pp_losses_v6 import LossBuilderV6
from models.Net import Net
from models.postprocess_v6 import PostProcessModelV6
from models.stylegan2 import dnnlib
from utils.bicubic import BicubicDownSample
from utils.train import _LegacyUnpickler, seed_everything, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
# 可选 "ffhq" 或 "small"。
USER_DATASET_PROFILE = "small"

# FFHQ 大数据集配置。
USER_DATASET_DIR_FFHQ = Path("input/pp_dataset_v6")
USER_OUTPUT_DIR_FFHQ = Path("output/pp_train_v6")
USER_VAL_SIZE_FFHQ = 512

# small 小数据集配置。
USER_DATASET_DIR_SMALL = Path("input/pp_dataset_v6_small")
USER_OUTPUT_DIR_SMALL = Path("output/pp_train_v6_small")
USER_VAL_SIZE_SMALL = 16

# 通用配置。
USER_BASE_CHECKPOINT = "pretrained_models/PostProcess/pp_model.pth"
USER_RESUME_CHECKPOINT = ""
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 8
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_EPOCHS = 50
USER_LR = 1e-4
USER_WEIGHT_DECAY = 0.0
USER_GRAD_CLIP = 0.5

USER_USE_ADV = False
USER_ADV_COEF = 0.05
USER_D_REG_EVERY = 16

USER_USE_MOD = True
USER_PRETRAIN = False
USER_FINETUNE = False
USER_INPAINT = 0.15
USER_DIFF_MASK_DILATE = 5
USER_DIFF_MASK_BLUR_KERNEL = 11
USER_DIFF_MASK_BLUR_SIGMA = 0.0
USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_PREVIEW_COUNT = 16
# ======================================================================


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


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_DATASET_DIR = _DATASET_CFG["dataset_dir"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_VAL_SIZE = _DATASET_CFG["val_size"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PPDatasetV6(Dataset):
    def __init__(self, items: list[dict], is_test: bool = False):
        self.items = items
        self.is_test = is_test
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.items)

    def _load_source(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            return self.to_tensor(image.convert("RGB"))

    def _flip(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.is_test or random.random() <= 0.5:
            return item
        flipped = {}
        for key, value in item.items():
            if torch.is_tensor(value):
                flipped[key] = T.functional.hflip(value)
            else:
                flipped[key] = value
        return flipped

    def __getitem__(self, idx):
        raw = self.items[idx]
        item = {
            "source": self._load_source(raw["source_path"]),
            "target": raw["target"].clone(),
            "target_mask": raw["target_mask"].clone(),
            "HT_E": raw["HT_E"].clone(),
            "diff_mask": raw["diff_mask"].clone() if raw["diff_mask"] is not None else torch.zeros_like(raw["target_mask"]),
            "valid_mask": raw["valid_mask"].clone() if raw["valid_mask"] is not None else raw["target_mask"].clone(),
            "source_hair_mask": raw["source_hair_mask"].clone() if raw["source_hair_mask"] is not None else torch.zeros_like(raw["target_mask"]),
            "target_hair_mask": raw["target_hair_mask"].clone() if raw["target_hair_mask"] is not None else torch.zeros_like(raw["target_mask"]),
        }
        return self._flip(item)


def collate_pp_v6(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}


def load_dataset_items(dataset_dir: Path) -> list[dict]:
    items = []
    idx = 1
    while (dataset_dir / f"pp_part_{idx:03d}.dataset").exists():
        items.extend(torch.load(dataset_dir / f"pp_part_{idx:03d}.dataset"))
        idx += 1
    if not items:
        raise RuntimeError(f"No pp_part_XXX.dataset files found under {dataset_dir}")
    return items


class TrainerV6:
    def __init__(self, model, optimizer, train_loader, val_loader):
        self.device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.downsample_256 = BicubicDownSample(factor=4)

        self.net = Net(
            Namespace(
                size=1024,
                ckpt=USER_STYLEGAN_CKPT,
                channel_multiplier=2,
                latent=512,
                n_mlp=8,
                device=str(self.device),
            )
        )
        toggle_grad(self.net.generator, False)

        self.discriminator = None
        self.disc_optim = None
        if USER_USE_ADV:
            with dnnlib.util.open_url("pretrained_models/StyleGAN/ffhq.pkl") as f:
                data = _LegacyUnpickler(f).load()
            self.discriminator = data["D"].to(self.device).eval()
            self.disc_optim = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=3e-4,
                betas=(0.9, 0.999),
                amsgrad=False,
                weight_decay=0.0,
            )
            toggle_grad(self.discriminator, False)

        self.loss_builder = LossBuilderV6(
            {
                "lpips_scale": 0.8,
                "id": 0.1,
                "landmark": 0.1,
                "feat_rec": 0.01,
                "adv": USER_ADV_COEF,
                "inpaint": USER_INPAINT,
            },
            device=str(self.device),
        )
        self.best_loss = float("inf")
        self.cur_iter = 0
        self.ckpt_dir = ACTIVE_OUTPUT_DIR / "checkpoints"
        self.preview_dir = ACTIVE_OUTPUT_DIR / "previews"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "cur_iter": self.cur_iter,
        }
        if self.discriminator is not None and self.disc_optim is not None:
            state["D"] = self.discriminator.state_dict()
            state["disc_optimizer_state_dict"] = self.disc_optim.state_dict()

        torch.save(state, self.ckpt_dir / "last.pth")
        if epoch % USER_SAVE_CHECKPOINT_EVERY == 0:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.discriminator is not None and "D" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["D"], strict=False)
        if self.disc_optim is not None and "disc_optimizer_state_dict" in checkpoint:
            self.disc_optim.load_state_dict(checkpoint["disc_optimizer_state_dict"])
        self.best_loss = checkpoint.get("best_loss", self.best_loss)
        self.cur_iter = checkpoint.get("cur_iter", self.cur_iter)
        return checkpoint.get("epoch", -1) + 1

    def _run_model(self, batch):
        source = batch["source"].to(self.device)
        target = batch["target"].to(self.device)
        target_mask = batch["target_mask"].to(self.device)
        HT_E = batch["HT_E"].to(self.device)
        diff_mask = batch["diff_mask"].to(self.device)
        valid_mask = batch["valid_mask"].to(self.device)
        source_hair_mask = batch["source_hair_mask"].to(self.device)
        target_hair_mask = batch["target_hair_mask"].to(self.device)

        latent_s, latent_f, aux = self.model(
            self.normalize(source),
            self.normalize(target),
            target_mask=target_mask,
            HT_E=HT_E,
            diff_mask=diff_mask,
            valid_mask=valid_mask,
            source_hair_mask=source_hair_mask,
            target_hair_mask=target_hair_mask,
        )

        gen_im_W, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False)
        F_w, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False, start_layer=0, end_layer=4)
        gen_im_F, _ = self.net.generator(
            [latent_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=latent_f,
        )
        return source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux

    def _maybe_train_discriminator(self, source, gen_im_F):
        if self.discriminator is None or self.disc_optim is None:
            return {}

        toggle_grad(self.discriminator, True)
        self.discriminator.train()

        real_1024 = self.normalize(source)
        disc_loss = self.loss_builder.CalcDisLoss(self.discriminator, real_1024, gen_im_F.detach())
        if self.cur_iter % USER_D_REG_EVERY == 0:
            disc_loss.update(self.loss_builder.CalcR1Loss(self.discriminator, real_1024))

        total_loss = sum(disc_loss.values())
        self.disc_optim.zero_grad()
        total_loss.backward()
        self.disc_optim.step()

        toggle_grad(self.discriminator, False)
        self.discriminator.eval()
        return disc_loss

    def train_one_epoch(self):
        self.model.train()
        total = defaultdict(float)
        steps = 0
        for batch in tqdm(self.train_loader):
            steps += 1
            source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux = self._run_model(batch)
            losses = self.loss_builder(source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux=aux)
            if self.discriminator is not None:
                losses.update(self.loss_builder.CalcAdvLoss(self.discriminator, gen_im_F))
            losses["loss"] = sum(losses.values())

            self.optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
            self.optimizer.step()

            if self.discriminator is not None:
                losses.update(self._maybe_train_discriminator(source, gen_im_F))

            self.cur_iter += 1
            for key, value in losses.items():
                total[key] += float(value.detach().cpu())

        return {key: value / max(steps, 1) for key, value in total.items()}

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total = defaultdict(float)
        previews = []
        steps = 0
        to_pil = T.ToPILImage()

        for batch in tqdm(self.val_loader):
            steps += 1
            source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux = self._run_model(batch)
            losses = self.loss_builder(source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux=aux)
            losses["loss"] = sum(losses.values())

            gen_w_256 = self.downsample_256((gen_im_W + 1) / 2).clamp(0, 1)
            gen_f_256 = self.downsample_256((gen_im_F + 1) / 2).clamp(0, 1)

            for key, value in losses.items():
                total[key] += float(value.detach().cpu())

            if len(previews) < USER_PREVIEW_COUNT:
                bsz = source.size(0)
                for idx in range(bsz):
                    if len(previews) >= USER_PREVIEW_COUNT:
                        break
                    previews.append(torch.cat([source[idx], target[idx], gen_w_256[idx], gen_f_256[idx]], dim=2).cpu())

        if epoch % USER_SAVE_PREVIEW_EVERY == 0 and previews:
            preview_epoch_dir = self.preview_dir / f"epoch_{epoch:03d}"
            preview_epoch_dir.mkdir(parents=True, exist_ok=True)
            for idx, image in enumerate(previews):
                to_pil(image).save(preview_epoch_dir / f"sample_{idx:03d}.png")

        return {key: value / max(steps, 1) for key, value in total.items()}

    def train_loop(self):
        start_epoch = 0
        if USER_RESUME_CHECKPOINT:
            start_epoch = self.load_checkpoint(USER_RESUME_CHECKPOINT)

        for epoch in range(start_epoch, USER_EPOCHS):
            train_losses = self.train_one_epoch()
            val_losses = self.validate(epoch)

            print(f"[epoch {epoch:03d}] train_loss={train_losses.get('loss', 0):.4f} val_loss={val_losses.get('loss', 0):.4f}")
            is_best = val_losses.get("loss", float("inf")) <= self.best_loss
            if is_best:
                self.best_loss = val_losses["loss"]
            self.save_checkpoint(epoch, is_best=is_best)


def build_model_and_optimizer(device: torch.device):
    model_args = Namespace(
        use_mod=USER_USE_MOD,
        pretrain=USER_PRETRAIN,
        finetune=USER_FINETUNE,
        diff_mask_dilate=USER_DIFF_MASK_DILATE,
        diff_mask_blur_kernel=USER_DIFF_MASK_BLUR_KERNEL,
        diff_mask_blur_sigma=USER_DIFF_MASK_BLUR_SIGMA,
    )
    model = PostProcessModelV6(model_args).to(device)
    if USER_BASE_CHECKPOINT:
        model.load_base_checkpoint(USER_BASE_CHECKPOINT)
    optimizer = torch.optim.Adam(model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
    return model, optimizer


def main():
    seed_everything(USER_RANDOM_SEED)
    set_seed(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    items = load_dataset_items(ACTIVE_DATASET_DIR)
    if len(items) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"Dataset is smaller than the validation split size ({ACTIVE_VAL_SIZE}) "
            f"for profile {USER_DATASET_PROFILE!r}."
        )

    train_items, val_items = train_test_split(items, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = PPDatasetV6(train_items, is_test=False)
    val_dataset = PPDatasetV6(val_items, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_pp_v6,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_pp_v6,
    )

    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    model, optimizer = build_model_and_optimizer(device)
    trainer = TrainerV6(model, optimizer, train_loader, val_loader)
    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"output dir: {ACTIVE_OUTPUT_DIR}")
    trainer.train_loop()


if __name__ == "__main__":
    main()
