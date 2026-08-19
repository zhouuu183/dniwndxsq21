import argparse
import os
import sys
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

from hair_swap import HairFast, get_parser
from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from models.postprocess_v6 import PostProcessModelV6
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.train import seed_everything, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
# 可选 "ffhq" 或 "small"。
USER_DATASET_PROFILE = "small"

# FFHQ 大数据集配置。
USER_DATASET_DIR_FFHQ = Path("images/blending_dataset_v6")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_train_v6")
USER_VAL_SIZE_FFHQ = 512

# small 小数据集配置。
USER_DATASET_DIR_SMALL = Path("images/blending_dataset_v6_small")
USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_short")
USER_OUTPUT_DIR_SMALL = Path("output/blending_train_v6_small")
USER_VAL_SIZE_SMALL = 100

# 通用配置。
USER_RANDOM_SEED = 3407
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_BASELINE_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_INIT_FROM_BASELINE_BLENDING = True
USER_BASELINE_PREVIEW_MODE = "original_full"
USER_BASELINE_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BASELINE_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_V6_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_BATCH_SIZE = 32
USER_NUM_WORKERS = 0
USER_EPOCHS = 50
USER_LR = 1e-4
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 5.0
USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_VAL_IMAGES_EVERY = 1
USER_LOG_IMAGE_COUNT = 16
# ======================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_COLOR_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        },
        "small": {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_COLOR_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
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
ACTIVE_COLOR_ROOT = _DATASET_CFG["color_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_VAL_SIZE = _DATASET_CFG["val_size"]


class BlendingDatasetV6(Dataset):
    def __init__(self, exps: list[list[str]], dataset_dir: Path, face_root: Path, shape_root: Path, color_root: Path):
        self.items = []
        self.dataset_dir = dataset_dir
        self.face_root = face_root
        self.shape_root = shape_root
        self.color_root = color_root
        for im1, im2, im3 in exps:
            item = self._prepare_item(im1, im2, im3)
            if item is not None:
                self.items.append(item)

    def _load_image(self, root: Path, stem: str) -> torch.Tensor:
        for ext in (".png", ".jpg", ".jpeg"):
            path = root / f"{stem}{ext}"
            if path.exists():
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image = T.functional.resize(
                        image,
                        [256, 256],
                        interpolation=T.InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                    return T.functional.normalize(T.functional.to_tensor(image), [0.5], [0.5])
        raise FileNotFoundError(f"Cannot find image for {stem} under {root}")

    def _prepare_item(self, face_stem: str, shape_stem: str, color_stem: str):
        try:
            color_s = torch.from_numpy(np.load(self.dataset_dir / "FS" / f"{color_stem}.npz")["latent_in"]).squeeze(0).float()
            align_s = torch.from_numpy(np.load(self.dataset_dir / "FS" / f"{face_stem}.npz")["latent_in"]).squeeze(0).float()

            shape_align_npz = np.load(self.dataset_dir / "Align" / f"{face_stem}_{shape_stem}.npz")
            color_align_npz = np.load(self.dataset_dir / "Align" / f"{face_stem}_{color_stem}.npz")
            align_f = torch.from_numpy(shape_align_npz["latent_F"]).squeeze(0).float()
            valid_mask = (
                torch.from_numpy(shape_align_npz["valid_mask"]).squeeze(0).float()
                if "valid_mask" in shape_align_npz
                else torch.ones((1, 256, 256), dtype=torch.float32)
            )
            diff_mask = (
                torch.from_numpy(shape_align_npz["diff_mask"]).squeeze(0).float()
                if "diff_mask" in shape_align_npz
                else torch.zeros((1, 256, 256), dtype=torch.float32)
            )
            source_hair_mask = (
                torch.from_numpy(shape_align_npz["source_hair_mask"]).squeeze(0).float()
                if "source_hair_mask" in shape_align_npz
                else torch.zeros((1, 256, 256), dtype=torch.float32)
            )
            shape_target_hair_mask = (
                torch.from_numpy(shape_align_npz["target_hair_mask"]).squeeze(0).float()
                if "target_hair_mask" in shape_align_npz
                else torch.zeros((1, 256, 256), dtype=torch.float32)
            )
            color_target_hair_mask = (
                torch.from_numpy(color_align_npz["target_hair_mask"]).squeeze(0).float()
                if "target_hair_mask" in color_align_npz
                else torch.zeros((1, 256, 256), dtype=torch.float32)
            )

            shape_i = self._load_image(self.shape_root, shape_stem)
            color_i = self._load_image(self.color_root, color_stem)
            face_i = self._load_image(self.face_root, face_stem)
            return {
                "color_s": color_s,
                "align_s": align_s,
                "align_f": align_f,
                "color_i": color_i,
                "face_i": face_i,
                "shape_i": shape_i,
                "valid_mask": valid_mask,
                "diff_mask": diff_mask,
                "source_hair_mask": source_hair_mask,
                "shape_target_hair_mask": shape_target_hair_mask,
                "color_target_hair_mask": color_target_hair_mask,
                "face_stem": face_stem,
                "shape_stem": shape_stem,
                "color_stem": color_stem,
            }
        except Exception as exc:
            print(exc, file=sys.stderr)
            return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TrainerV6:
    def __init__(self, model, optimizer, train_loader, val_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = ACTIVE_OUTPUT_DIR / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir = ACTIVE_OUTPUT_DIR / "val_images"
        self.val_dir.mkdir(parents=True, exist_ok=True)

        self.net = Net(
            type("Args", (), {
                "size": 1024,
                "ckpt": USER_STYLEGAN_CKPT,
                "channel_multiplier": 2,
                "latent": 512,
                "n_mlp": 8,
                "device": str(self.device),
            })()
        )
        self.seg = BiSeNet(n_classes=16).to(self.device).eval()
        self.seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth"))
        self.dilate_erosion = DilateErosion(device=self.device)
        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)
        toggle_grad(self.seg, False)
        toggle_grad(self.net.generator, False)
        self.baseline_model = None
        self.true_baseline_runner = None
        if USER_BASELINE_PREVIEW_MODE != "original_full" and USER_BASELINE_BLENDING_CKPT:
            checkpoint = torch.load(USER_BASELINE_BLENDING_CKPT, map_location="cpu")
            self.baseline_model = ClipBlendingModel()
            self.baseline_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.baseline_model.to(self.device).eval()
            toggle_grad(self.baseline_model, False)
        self.pp_v6 = PostProcessModelV6(
            argparse.Namespace(
                use_mod=True,
                pretrain=False,
                finetune=False,
                diff_mask_dilate=5,
                diff_mask_blur_kernel=11,
                diff_mask_blur_sigma=0.0,
            )
        ).to(self.device).eval()
        self.pp_v6.load_base_checkpoint(USER_V6_PP_CKPT)
        toggle_grad(self.pp_v6, False)
        self.best_loss = float("inf")

    @staticmethod
    def _stem_to_path(root: Path, stem: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg"):
            path = root / f"{stem}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Cannot find image for {stem} under {root}")

    def _get_true_baseline_runner(self):
        if self.true_baseline_runner is None:
            model_args = get_parser().parse_args([])
            model_args.device = str(self.device)
            model_args.ckpt = USER_STYLEGAN_CKPT
            model_args.rotate_checkpoint = USER_BASELINE_ROTATE_CKPT
            model_args.blending_checkpoint = USER_BASELINE_BLENDING_CKPT
            model_args.pp_checkpoint = USER_BASELINE_PP_CKPT
            self.true_baseline_runner = HairFast(model_args)
        return self.true_baseline_runner

    @torch.no_grad()
    def _render_true_baseline_sample(self, face_stem: str, shape_stem: str, color_stem: str) -> torch.Tensor:
        hair_fast = self._get_true_baseline_runner()
        face_path = self._stem_to_path(ACTIVE_FACE_ROOT, face_stem)
        shape_path = self._stem_to_path(ACTIVE_SHAPE_ROOT, shape_stem)
        color_path = self._stem_to_path(ACTIVE_COLOR_ROOT, color_stem)
        baseline_01 = hair_fast.swap(face_path, shape_path, color_path)
        baseline_256 = T.functional.resize(
            baseline_01,
            [256, 256],
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )
        return baseline_256 * 2.0 - 1.0

    @torch.no_grad()
    def generate_mask(self, image):
        im = (self.downsample_512((image + 1) / 2) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(im)
        current_mask = torch.argmax(down_seg, dim=1).long().float()
        hair = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        hair = F.interpolate(hair.unsqueeze(1), size=(256, 256), mode="nearest")
        hair_d, hair_e = self.dilate_erosion.mask(hair)
        return hair_d, hair_e

    def calc_loss(self, gen_256, face_i, color_i, valid_mask, hair_mask):
        face_embed = self.model.get_image_embed(gen_256 * valid_mask)
        gt_face_embed = self.model.get_image_embed(face_i * valid_mask)
        face_loss = (1.0 - F.cosine_similarity(face_embed, gt_face_embed)).mean()

        hair_embed = self.model.get_image_embed(gen_256 * hair_mask)
        gt_hair_embed = self.model.get_image_embed(color_i * hair_mask)
        hair_loss = (1.0 - F.cosine_similarity(hair_embed, gt_hair_embed)).mean()
        return face_loss + hair_loss, {"face_loss": face_loss, "hair_loss": hair_loss}

    def _render_output(self, model, align_s, color_s, face_i, face_mask, color_i, hair_mask, align_f):
        blend_s = model(align_s[:, 6:], color_s[:, 6:], face_i * face_mask, color_i * hair_mask)
        latent_in = torch.cat((torch.zeros(color_s.size(0), 6, 512, device=self.device), blend_s), dim=1)
        gen_im, _ = self.net.generator(
            [latent_in],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=align_f,
        )
        return self.downsample_256(gen_im)

    @torch.no_grad()
    def _render_v6_final_output(
        self,
        face_i,
        pre_pp_256,
        color_hair_e,
        diff_mask,
        valid_mask,
        source_hair_mask,
        target_hair_mask,
    ):
        s_final, f_final, _ = self.pp_v6(
            face_i,
            pre_pp_256,
            target_mask=valid_mask,
            HT_E=color_hair_e,
            diff_mask=diff_mask,
            valid_mask=valid_mask,
            source_hair_mask=source_hair_mask,
            target_hair_mask=target_hair_mask,
        )
        gen_im, _ = self.net.generator(
            [s_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=f_final,
        )
        return self.downsample_256(gen_im)

    def _build_reference_panel(self, shape_i, color_i):
        return torch.cat([shape_i, color_i], dim=3)

    def _save_preview_rows(self, epoch: int, preview_rows: list[torch.Tensor]):
        if epoch % USER_SAVE_VAL_IMAGES_EVERY != 0 or not preview_rows:
            return
        save_dir = self.val_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        to_pil = T.ToPILImage()
        for idx, row in enumerate(preview_rows[:USER_LOG_IMAGE_COUNT]):
            to_pil(row).save(save_dir / f"val_sample_{idx:03d}.png")

    def run_epoch(self, loader, training: bool, epoch: int | None = None):
        self.model.train(training)
        total = {"loss": 0.0, "face_loss": 0.0, "hair_loss": 0.0}
        steps = 0
        preview_rows: list[torch.Tensor] = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in tqdm(loader):
                steps += 1
                color_s = batch["color_s"].to(self.device)
                align_s = batch["align_s"].to(self.device)
                align_f = batch["align_f"].to(self.device)
                color_i = batch["color_i"].to(self.device)
                face_i = batch["face_i"].to(self.device)
                shape_i = batch["shape_i"].to(self.device)
                valid_mask = batch["valid_mask"].to(self.device)
                diff_mask = batch["diff_mask"].to(self.device)
                source_hair_mask = batch["source_hair_mask"].to(self.device)
                shape_target_hair_mask = batch["shape_target_hair_mask"].to(self.device)
                color_target_hair_mask = batch["color_target_hair_mask"].to(self.device)
                color_hair_d, color_hair_e = self.generate_mask(color_i)
                face_hair_d, _ = self.generate_mask(face_i)
                color_target_hair_d, _ = self.dilate_erosion.mask(color_target_hair_mask)

                gen_256 = self._render_output(
                    self.model,
                    align_s,
                    color_s,
                    face_i,
                    valid_mask,
                    color_i,
                    color_hair_e,
                    align_f,
                )
                loss, info = self.calc_loss(gen_256, face_i, color_i, valid_mask, color_hair_e)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
                    self.optimizer.step()

                total["loss"] += float(loss.detach().cpu())
                total["face_loss"] += float(info["face_loss"].detach().cpu())
                total["hair_loss"] += float(info["hair_loss"].detach().cpu())

                if (not training) and epoch is not None and len(preview_rows) < USER_LOG_IMAGE_COUNT:
                    gen_final_256 = self._render_v6_final_output(
                        face_i,
                        gen_256,
                        color_hair_e,
                        diff_mask,
                        valid_mask,
                        source_hair_mask,
                        shape_target_hair_mask,
                    )
                    ref_panel = self._build_reference_panel(shape_i, color_i)
                    max_add = min(face_i.size(0), USER_LOG_IMAGE_COUNT - len(preview_rows))
                    for idx in range(max_add):
                        if USER_BASELINE_PREVIEW_MODE == "original_full":
                            baseline_sample = self._render_true_baseline_sample(
                                batch["face_stem"][idx],
                                batch["shape_stem"][idx],
                                batch["color_stem"][idx],
                            ).cpu()
                        else:
                            baseline_face_mask = ((1.0 - face_hair_d) * (1.0 - color_hair_d) * (1.0 - color_target_hair_d)).clamp(0, 1)
                            if self.baseline_model is not None:
                                baseline_sample = self._render_output(
                                    self.baseline_model,
                                    align_s[idx : idx + 1],
                                    color_s[idx : idx + 1],
                                    face_i[idx : idx + 1],
                                    baseline_face_mask[idx : idx + 1],
                                    color_i[idx : idx + 1],
                                    color_hair_e[idx : idx + 1],
                                    align_f[idx : idx + 1],
                                )[0].cpu()
                            else:
                                baseline_sample = gen_final_256[idx].detach().cpu()
                        row = torch.cat(
                            [
                                ((face_i[idx] + 1) / 2).clamp(0, 1).cpu(),
                                ((ref_panel[idx] + 1) / 2).clamp(0, 1).cpu(),
                                ((baseline_sample + 1) / 2).clamp(0, 1).cpu(),
                                ((gen_final_256[idx] + 1) / 2).clamp(0, 1).cpu(),
                            ],
                            dim=2,
                        )
                        preview_rows.append(row)

        for key in total:
            total[key] /= max(steps, 1)
        if (not training) and epoch is not None:
            self._save_preview_rows(epoch, preview_rows)
        return total

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }
        torch.save(state, self.output_dir / "last.pth")
        if epoch % USER_SAVE_CHECKPOINT_EVERY == 0:
            torch.save(state, self.output_dir / f"epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, self.output_dir / "best.pth")

    def train_loop(self):
        for epoch in range(USER_EPOCHS):
            train_losses = self.run_epoch(self.train_loader, training=True)
            val_losses = self.run_epoch(self.val_loader, training=False, epoch=epoch)
            print(f"[epoch {epoch:03d}] train_loss={train_losses['loss']:.4f} val_loss={val_losses['loss']:.4f}")

            is_best = val_losses["loss"] <= self.best_loss
            if is_best:
                self.best_loss = val_losses["loss"]
            self.save_checkpoint(epoch, is_best=is_best)


def load_exps(dataset_dir: Path):
    exps = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as file:
        for line in file:
            exps.append(line.strip().split())
    return exps


def main():
    seed_everything(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exps = load_exps(ACTIVE_DATASET_DIR)
    if len(exps) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"dataset.exps is smaller than the validation split size ({ACTIVE_VAL_SIZE}) "
            f"for profile {USER_DATASET_PROFILE!r}."
        )
    train_exps, val_exps = train_test_split(exps, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = BlendingDatasetV6(train_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT, ACTIVE_COLOR_ROOT)
    val_dataset = BlendingDatasetV6(val_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT, ACTIVE_COLOR_ROOT)

    train_loader = DataLoader(train_dataset, batch_size=USER_BATCH_SIZE, shuffle=True, num_workers=USER_NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=USER_BATCH_SIZE, shuffle=False, num_workers=USER_NUM_WORKERS, drop_last=False)

    model = ClipBlendingModel()
    if USER_INIT_FROM_BASELINE_BLENDING and USER_BASELINE_BLENDING_CKPT:
        checkpoint = torch.load(USER_BASELINE_BLENDING_CKPT, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
    trainer = TrainerV6(model, optimizer, train_loader, val_loader)
    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"face/source root: {ACTIVE_FACE_ROOT}")
    print(f"shape/reference root: {ACTIVE_SHAPE_ROOT}")
    print(f"color/reference root: {ACTIVE_COLOR_ROOT}")
    trainer.train_loop()


if __name__ == "__main__":
    main()
