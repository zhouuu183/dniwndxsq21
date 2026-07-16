import argparse
import os
import random
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# v1 bugfix:
# Allow repo-root package imports when running the script from the scripts
# directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from losses.pp_losses_v1 import LossBuilder, LossBuilderMultiV1
from models.Encoders_v1 import DeltaAwarePostProcessModelV1
from models.Net import Net
from models.stylegan2 import dnnlib
from utils.bicubic import BicubicDownSample
from utils.train import _LegacyUnpickler, WandbLogger, get_fid_calc, image_grid, seed_everything, toggle_grad


# ============================================================
# 用户配置区域：只改这里
# ------------------------------------------------------------
# 这里填：第三阶段数据集目录
USER_DATASET_DIR = Path("input/pp_dataset_v1")
#
# 这里填：FID 数据集目录字符串
USER_FID_DATASET = "input"
#
# 这里填：训练 batch size
USER_BATCH_SIZE = 16
#
# 这里填：训练轮数
USER_EPOCHS = 200
#
# 这里填：pp 对抗训练开始前的迭代数
USER_ITER_BEFORE = 10000
#
# 这里填：判别器正则频率
USER_D_REG_EVERY = 16
#
# 这里填：inpaint loss 系数
USER_INPAINT = 0.0
#
# 这里填：adv loss 系数
USER_ADV_COEF = 0.05
#
# 这里填：如果要从旧 checkpoint 继续训练，就写路径；否则保持 None
USER_CHECKPOINT = None
# ============================================================


# v1 modification:
# pp training now consumes remove/boundary masks from the new blending stage
# and uses a lightly adapted refinement model with delta-aware losses.


class TrainerV1:
    def __init__(self, model, args, optimizer, train_dataloader, test_dataloader, logger):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.net = Net(
            argparse.Namespace(
                size=1024,
                ckpt="pretrained_models/StyleGAN/ffhq.pt",
                channel_multiplier=2,
                latent=512,
                n_mlp=8,
                device=self.device,
            )
        )

        with dnnlib.util.open_url("pretrained_models/StyleGAN/ffhq.pkl") as f:
            data = _LegacyUnpickler(f).load()
        # v1 bugfix:
        # Follow the trainer device instead of forcing CUDA.
        self.discriminator = data["D"].to(self.device).eval()
        self.disc_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            amsgrad=False,
            weight_decay=0,
        )

        toggle_grad(self.discriminator, False)
        toggle_grad(self.net.generator, False)

        self.downsample_256 = BicubicDownSample(factor=4)
        self.best_loss = float("+inf")

        if self.args.pretrain:
            self.loss_builder = LossBuilder(
                {"lpips_scale": 0.8, "id": 0.1, "landmark": 0.0, "feat_rec": 0.01, "adv": self.args.adv_coef}
            )
        else:
            self.loss_builder = LossBuilderMultiV1(
                {"lpips_scale": 0.8, "id": 0.1, "landmark": 0.1, "feat_rec": 0.01, "adv": self.args.adv_coef, "inpaint": self.args.inpaint}
            )

        self.cur_iter = 1
        self.fid_calc = get_fid_calc("input/fid_v1.pkl", args.fid_dataset) if args.fid_dataset else None

    def save_model(self, name):
        with TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, f"{name}.pth")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "D": self.discriminator.state_dict(),
                    "cur_iter": self.cur_iter,
                },
                path,
            )
            self.logger.save(path, save_online=False)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if "D" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["D"], strict=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    def _forward_model(self, source, target, remove_mask, boundary_mask):
        latent_s, latent_f = self.model(
            self.normalize(source),
            self.normalize(target),
            remove_mask=remove_mask,
            boundary_mask=boundary_mask,
        )

        gen_im_W, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False)
        F_w, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False, start_layer=0, end_layer=4)
        gen_im_F, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False, start_layer=5, end_layer=8, layer_in=latent_f)
        return latent_f, gen_im_W, F_w, gen_im_F

    def train_one_epoch(self):
        self.model.to(self.device).train()
        for batch in tqdm(self.train_dataloader):
            source, target, target_mask, HT_E, remove_mask, boundary_mask = map(lambda x: x.to(self.device), batch)
            source, source_1024 = self.downsample_256(source).clip(0, 1), self.normalize(source)

            latent_f, gen_im_W, F_w, gen_im_F = self._forward_model(source, target, remove_mask, boundary_mask)
            losses = self.loss_builder(
                source,
                target,
                target_mask,
                HT_E,
                gen_im_W,
                F_w,
                gen_im_F,
                latent_f,
                remove_mask=remove_mask,
                boundary_mask=boundary_mask,
            )

            if self.args.use_adv and self.cur_iter >= self.args.iter_before:
                losses.update(self.loss_builder.CalcAdvLoss(self.discriminator, gen_im_F))

            losses["loss"] = sum(losses.values())

            self.optimizer.zero_grad()
            losses["loss"].backward()
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            if self.args.use_adv and self.cur_iter >= self.args.iter_before:
                toggle_grad(self.discriminator, True)
                self.discriminator.train()
                disc_loss = self.loss_builder.CalcDisLoss(self.discriminator, source_1024, gen_im_F.detach())
                if self.cur_iter % self.args.d_reg_every:
                    disc_loss.update(self.loss_builder.CalcR1Loss(self.discriminator, source_1024))
                total_disc = sum(disc_loss.values())
                self.disc_optim.zero_grad()
                total_disc.backward()
                total_norm_d = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                self.disc_optim.step()
                toggle_grad(self.discriminator, False)
                self.discriminator.eval()
                disc_loss["grad disc"] = total_norm_d
                losses.update(disc_loss)

            losses["pp grad"] = total_norm
            self.logger.next_step()
            self.logger.log_scalars({key: val for key, val in losses.items()})
            self.cur_iter += 1

    @torch.no_grad()
    def validate(self):
        self.model.to(self.device).eval()
        summed = {}
        files = []
        images_to_fid = []
        to_299 = T.Resize((299, 299))

        for batch in tqdm(self.test_dataloader):
            source, target, target_mask, HT_E, remove_mask, boundary_mask = map(lambda x: x.to(self.device), batch)
            source = self.downsample_256(source).clip(0, 1)

            latent_f, gen_im_W, F_w, gen_im_F = self._forward_model(source, target, remove_mask, boundary_mask)
            losses = self.loss_builder(
                source,
                target,
                target_mask,
                HT_E,
                gen_im_W,
                F_w,
                gen_im_F,
                latent_f,
                remove_mask=remove_mask,
                boundary_mask=boundary_mask,
            )
            losses["loss"] = sum(losses.values())

            for key, val in losses.items():
                summed[key] = summed.get(key, 0.0) + val.item()

            images_to_fid.append(to_299(((gen_im_F + 1) / 2).clip(0, 1)))
            gen_w_256 = self.downsample_256((gen_im_W + 1) / 2).clip(0, 1)
            gen_f_256 = self.downsample_256((gen_im_F + 1) / 2).clip(0, 1)
            for idx in range(source.size(0)):
                files.append([source[idx].cpu(), target[idx].cpu(), gen_w_256[idx].cpu(), gen_f_256[idx].cpu()])

        if self.fid_calc is not None:
            summed["FID CLIP"] = self.fid_calc(torch.cat(images_to_fid)).item()

        for key, val in summed.items():
            if key != "FID CLIP":
                val /= len(self.test_dataloader)
            self.logger.log_scalars({f"val {key}": val})

        np.random.seed(1927)
        idxs = np.random.choice(len(files), size=min(len(files), 32), replace=False)
        self.logger.log_scalars(
            {"val images": [wandb.Image(image_grid(list(map(T.functional.to_pil_image, files[idx])), 1, 4)) for idx in idxs]}
        )
        return summed["loss"] / len(self.test_dataloader)

    def train_loop(self, epochs):
        self.validate()
        for _ in range(epochs):
            self.train_one_epoch()
            loss = self.validate()
            self.save_model("last")
            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model("best")


class PPDatasetV1(Dataset):
    def __init__(self, items, is_test=False):
        super().__init__()
        self.items = items
        self.is_test = is_test

    def __len__(self):
        return len(self.items)

    def _load_image(self, path):
        return T.functional.to_tensor(Image.open(path).convert("RGB"))

    def _transform(self, img1, img2, mask1, mask2, remove_mask, boundary_mask):
        if self.is_test:
            return img1, img2, mask1, mask2, remove_mask, boundary_mask
        if random.random() > 0.5:
            img1 = T.functional.hflip(img1)
            img2 = T.functional.hflip(img2)
            mask1 = T.functional.hflip(mask1)
            mask2 = T.functional.hflip(mask2)
            remove_mask = T.functional.hflip(remove_mask)
            boundary_mask = T.functional.hflip(boundary_mask)
        return img1, img2, mask1, mask2, remove_mask, boundary_mask

    def __getitem__(self, idx):
        source_path, target, target_mask, HT_E, remove_mask, boundary_mask = self.items[idx]
        return self._transform(
            self._load_image(source_path),
            target,
            target_mask,
            HT_E,
            remove_mask,
            boundary_mask,
        )


def main(args):
    seed_everything()
    dataset = torch.load(args.dataset / "pp_delta_v1.dataset")
    test_size = max(1, min(len(dataset) // 10, 1024))
    x_train, x_test = train_test_split(dataset, test_size=test_size, random_state=42)

    train_dataset = PPDatasetV1(x_train)
    test_dataset = PPDatasetV1(x_test, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)

    logger = WandbLogger(name=args.name_run, project="HairFast-PostProcess-v1")
    logger.start_logging()
    logger.save(__file__)

    # v1 bugfix:
    # Keep the original pretrain/full-train semantics aligned with the v1 model.
    model = DeltaAwarePostProcessModelV1(pretrain=args.pretrain)
    if not args.finetune:
        toggle_grad(model.encoder_face, False)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4 if args.pretrain else 1e-4, weight_decay=0)
    trainer = TrainerV1(model, args, optimizer, train_loader, test_loader, logger)
    if args.checkpoint:
        trainer.load_model(args.checkpoint)
    trainer.train_loop(args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta-aware post-process trainer")
    parser.add_argument("--name_run", type=str, default="pp_v1")
    # v1 modification:
    # Local user-config block above keeps this script self-contained.
    parser.add_argument("--dataset", type=Path, default=USER_DATASET_DIR)
    parser.add_argument("--fid_dataset", type=str, default=USER_FID_DATASET)
    parser.add_argument("--batch_size", type=int, default=USER_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=USER_EPOCHS)
    parser.add_argument("--iter_before", type=int, default=USER_ITER_BEFORE)
    parser.add_argument("--d_reg_every", type=int, default=USER_D_REG_EVERY)
    parser.add_argument("--inpaint", type=float, default=USER_INPAINT)
    parser.add_argument("--use_adv", action="store_true")
    parser.add_argument("--adv_coef", type=float, default=USER_ADV_COEF)
    parser.add_argument("--checkpoint", type=str, default=USER_CHECKPOINT)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    main(args)
