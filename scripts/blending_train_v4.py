import os
import sys
import hashlib
import shutil
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Encoders import ClipBlendingModel as BlendingModel
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v4 import filter_parsing_to_primary_subject
from utils.train import WandbLogger, get_fid_calc, image_grid, seed_everything, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
# 指定当前进程可见的物理 GPU。
USER_CUDA_VISIBLE_DEVICES = "0"

# PyTorch 实际使用的设备。通常写 "cuda"。
USER_DEVICE = "cuda"

# blending v4 数据集目录。这里应指向 blending_gen_v4.py 生成出的 dataset.exps 目录。
USER_DATASET_DIR = Path("images/blending_dataset_v4")

# FFHQ 原图目录。
USER_FFHQ_ROOT = Path("images/FFHQ")

# 输出目录。会保存 checkpoint、验证图等内容。
USER_OUTPUT_DIR = Path("output/blending_train_v4")

# StyleGAN2 权重路径。
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"

# 真实 batch size。
USER_BATCH_SIZE = 8
USER_VAL_BATCH_SIZE = 2

# 目标有效 batch size。比如 8 -> 16 表示累计 2 次梯度再更新。
USER_EFFECTIVE_BATCH_SIZE = 16

# DataLoader worker 数量。优先稳定建议先用 0。
USER_NUM_WORKERS = 0

# 是否启用 pin_memory。默认先关。
USER_PIN_MEMORY = False

# 总 epoch 数。
USER_EPOCHS = 100

# 学习率。
USER_LR = 1e-4

# 权重衰减。
USER_WEIGHT_DECAY = 1e-6

# 验证集样本数。
USER_VAL_SIZE = 512

# 随机种子。
USER_RANDOM_SEED = 3407

# 每隔多少个 epoch 保存一次 checkpoint.pth 和 epoch_xxx.pth。
USER_SAVE_CHECKPOINT_EVERY = 1

# 每隔多少个 epoch 保存一次本地验证图。
USER_SAVE_VAL_IMAGES_EVERY = 1

# 每轮最多记录多少组可视化。
USER_LOG_IMAGE_COUNT = 100

# 是否启用 wandb。
USER_USE_WANDB = False
USER_WANDB_RUN_NAME = "blending_train_v4"
USER_WANDB_PROJECT = "HairFast-Blending-v4"

# FID 的真实图目录。不需要时保持 None。
USER_FID_IMAGE_DIR = None

# 是否从现有 checkpoint 恢复。
USER_RESUME_CHECKPOINT = ""
USER_CACHE_BUILD_BATCH_SIZE = 2
USER_LAMBDA_PRESERVE_RGB = 1.25
USER_LAMBDA_REMOVE_RGB = 1.35
USER_LAMBDA_REMOVE_GRAY = 0.60
USER_LAMBDA_BODY_PRESERVE = 1.25
USER_LAMBDA_TAIL_GRAY = 0.80
# ========================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

if USER_EFFECTIVE_BATCH_SIZE < USER_BATCH_SIZE or USER_EFFECTIVE_BATCH_SIZE % USER_BATCH_SIZE != 0:
    raise RuntimeError("USER_EFFECTIVE_BATCH_SIZE 必须大于等于 USER_BATCH_SIZE，且能被 USER_BATCH_SIZE 整除。")


CACHE_BUILD_BATCH_SIZE = USER_CACHE_BUILD_BATCH_SIZE
BLEND_FACE_LABELS_V4 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14)
BLEND_CONTEXT_LABELS_V4 = BLEND_FACE_LABELS_V4 + (10, 11, 15)


def gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


class NullLogger:
    def __init__(self):
        self.train_step = 0

    def start_logging(self):
        return None

    def log(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None

    def next_step(self):
        self.train_step += 1

    @property
    def wandb(self):
        class _Dummy:
            @staticmethod
            def finish():
                return None

        return _Dummy()


class TrainerV4:
    def __init__(self, model=None, optimizer=None, train_dataloader=None, test_dataloader=None):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = self._build_logger() if self.model is not None else NullLogger()
        self.logger.start_logging()

        self.device = USER_DEVICE if torch.cuda.is_available() else "cpu"
        self.dilate_erosion = DilateErosion(device=self.device)

        self.fid_calc = None
        if USER_FID_IMAGE_DIR:
            self.fid_calc = get_fid_calc(str(USER_OUTPUT_DIR / "fid_clip_v4.pkl"), str(USER_FID_IMAGE_DIR), device=torch.device(self.device))

        self.net = Net(
            Namespace(size=1024, ckpt=USER_STYLEGAN_CKPT, channel_multiplier=2, latent=512, n_mlp=8, device=self.device)
        )
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.device).eval()
        self.seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth"))

        toggle_grad(self.seg, False)
        toggle_grad(self.net.generator, False)

        self.downsample_512 = BicubicDownSample(factor=2, cuda=torch.cuda.is_available())
        self.downsample_256 = BicubicDownSample(factor=4, cuda=torch.cuda.is_available())
        self.best_loss = float("+inf")
        self.cur_iter = 0
        self.grad_accum_steps = max(1, USER_EFFECTIVE_BATCH_SIZE // USER_BATCH_SIZE)

        self.output_ckpt_dir = USER_OUTPUT_DIR / "checkpoints"
        self.output_val_dir = USER_OUTPUT_DIR / "val_images"
        self.output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)

    def _build_logger(self):
        if USER_USE_WANDB:
            return WandbLogger(name=USER_WANDB_RUN_NAME, project=USER_WANDB_PROJECT)
        return NullLogger()

    @torch.no_grad()
    def generate_mask(self, I):
        IM = (self.downsample_512((I + 1) / 2) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(IM)
        current_mask = torch.argmax(down_seg, dim=1).long().unsqueeze(1)
        current_mask, _ = filter_parsing_to_primary_subject(
            current_mask,
            face_labels=BLEND_FACE_LABELS_V4,
            context_labels=BLEND_CONTEXT_LABELS_V4,
        )
        HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask)).float()
        HM_X = F.interpolate(HM_X, size=(256, 256), mode="nearest")
        HM_XD, HM_XE = self.dilate_erosion.mask(HM_X)
        return HM_XD, HM_XE

    def save_model(self, name, save_online=True):
        with TemporaryDirectory() as tmp_dir:
            model_state_dict = self.model.state_dict()
            for key in list(model_state_dict.keys()):
                if key.startswith("clip_model."):
                    del model_state_dict[key]
            torch.save({"model_state_dict": model_state_dict}, f"{tmp_dir}/{name}.pth")
            self.logger.save(f"{tmp_dir}/{name}.pth", save_online)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "cur_iter": self.cur_iter,
        }
        torch.save(state, self.output_ckpt_dir / "last.pth")
        if epoch % USER_SAVE_CHECKPOINT_EVERY == 0:
            torch.save(state, self.output_ckpt_dir / "checkpoint.pth")
            torch.save(state, self.output_ckpt_dir / f"epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, self.output_ckpt_dir / "best.pth")
        infer_state = {"model_state_dict": self.model.state_dict(), "clip": "ViT-B/32"}
        for key in list(infer_state["model_state_dict"].keys()):
            if key.startswith("clip_model."):
                del infer_state["model_state_dict"][key]
        torch.save(infer_state, self.output_ckpt_dir / "blending_for_infer.pth")

    def calc_loss(
        self,
        I_gen,
        I_face,
        I_color,
        mask_face,
        mask_hair,
        source_inpaint,
        remove_mask,
        remove_halo,
        body_preserve,
        remove_tail,
    ):
        gen_embed = self.model.get_image_embed(I_gen * mask_face)
        gt_embed = self.model.get_image_embed(I_face * mask_face)
        face_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        gen_embed = self.model.get_image_embed(I_gen * mask_hair)
        gt_embed = self.model.get_image_embed(I_color * mask_hair)
        hair_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        preserve_rgb = masked_l1(I_gen, I_face, mask_face)
        body_rgb = masked_l1(I_gen, I_face, body_preserve)
        remove_region = (remove_mask + 0.55 * remove_halo + 0.35 * remove_tail).clamp(0, 1)
        remove_region = (remove_region * (1 - 0.75 * body_preserve)).clamp(0, 1)
        remove_rgb = masked_l1(I_gen, source_inpaint, remove_region)
        remove_gray = masked_l1(gray(I_gen), gray(source_inpaint), remove_region)
        tail_gray = masked_l1(gray(I_gen), gray(source_inpaint), (remove_tail * (1 - body_preserve)).clamp(0, 1))

        total_loss = (
            face_loss
            + hair_loss
            + USER_LAMBDA_PRESERVE_RGB * preserve_rgb
            + USER_LAMBDA_REMOVE_RGB * remove_rgb
            + USER_LAMBDA_REMOVE_GRAY * remove_gray
            + USER_LAMBDA_BODY_PRESERVE * body_rgb
            + USER_LAMBDA_TAIL_GRAY * tail_gray
        )
        return total_loss, {
            "face loss": face_loss,
            "hair loss": hair_loss,
            "preserve rgb": preserve_rgb,
            "body preserve": body_rgb,
            "remove rgb": remove_rgb,
            "remove gray": remove_gray,
            "tail gray": tail_gray,
            "loss": total_loss,
        }

    def train_one_epoch(self):
        self.model.to(self.device).train()
        self.optimizer.zero_grad(set_to_none=True)
        steps_in_epoch = 0
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
            steps_in_epoch += 1
            color_s, align_s, align_f, color_i, face_i, target_mask, HM_3E, source_inpaint, remove_mask, remove_halo, body_preserve, remove_tail = map(lambda x: x.to(self.device), batch)
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E)
            latent_in = torch.cat((torch.zeros(bsz, 6, 512, device=self.device), blend_s), dim=1)
            I_G, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=align_f)

            loss, info = self.calc_loss(
                self.downsample_256(I_G),
                face_i,
                color_i,
                target_mask,
                HM_3E,
                source_inpaint,
                remove_mask,
                remove_halo,
                body_preserve,
                remove_tail,
            )
            (loss / self.grad_accum_steps).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.logger.log("grad", float(total_norm))

            self.logger.next_step()
            for key, val in info.items():
                self.logger.log(key, val.item())
            self.cur_iter += 1

        if steps_in_epoch % self.grad_accum_steps != 0:
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.logger.log("grad", float(total_norm))

    @torch.no_grad()
    def validate(self, epoch=None):
        self.model.to(self.device).eval()

        total = defaultdict(float)
        files = []
        seen_files = 0
        rng = np.random.RandomState(1927)
        images_to_fid = []
        to_299 = T.Resize((299, 299))

        for batch in tqdm(self.test_dataloader):
            color_s, align_s, align_f, color_i, face_i, target_mask, HM_3E, source_inpaint, remove_mask, remove_halo, body_preserve, remove_tail = map(lambda x: x.to(self.device), batch)
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E)
            latent_in = torch.cat((torch.zeros(bsz, 6, 512, device=self.device), blend_s), dim=1)
            I_G, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=align_f)

            _, info = self.calc_loss(
                self.downsample_256(I_G),
                face_i,
                color_i,
                target_mask,
                HM_3E,
                source_inpaint,
                remove_mask,
                remove_halo,
                body_preserve,
                remove_tail,
            )
            for key, val in info.items():
                total[key] += float(val.detach().cpu())

            out_256 = self.downsample_256(I_G)
            for k in range(bsz):
                item = [color_i[k].cpu(), face_i[k].cpu(), out_256[k].cpu()]
                seen_files += 1
                if len(files) < USER_LOG_IMAGE_COUNT:
                    files.append(item)
                else:
                    replace_idx = rng.randint(0, seen_files)
                    if replace_idx < USER_LOG_IMAGE_COUNT:
                        files[replace_idx] = item
            if self.fid_calc is not None:
                images_to_fid.append(to_299((I_G + 1) / 2).clip(0, 1))

        denom = max(len(self.test_dataloader), 1)
        for key in list(total.keys()):
            total[key] /= denom
            self.logger.log(f"val {key}", total[key])

        if self.fid_calc is not None and images_to_fid:
            total["FID CLIP"] = float(self.fid_calc(torch.cat(images_to_fid)))
            self.logger.log("val FID CLIP", total["FID CLIP"])

        images_to_log = [
            image_grid([T.functional.to_pil_image(((img + 1) / 2).clamp(0, 1)) for img in row], 1, 3)
            for row in files
        ]

        if epoch is not None and epoch % USER_SAVE_VAL_IMAGES_EVERY == 0:
            save_dir = self.output_val_dir / f"epoch_{epoch:03d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(images_to_log):
                img.save(save_dir / f"val_sample_{idx:03d}.png")

        self.logger.log("val images", images_to_log)
        return total["loss"]

    def train_loop(self):
        start_epoch = 0
        if USER_RESUME_CHECKPOINT:
            checkpoint = torch.load(USER_RESUME_CHECKPOINT, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_loss = checkpoint.get("best_loss", self.best_loss)
            self.cur_iter = checkpoint.get("cur_iter", 0)
            start_epoch = checkpoint.get("epoch", -1) + 1

        self.validate()
        for epoch in range(start_epoch, USER_EPOCHS):
            self.train_one_epoch()
            loss = self.validate(epoch=epoch)
            is_best = loss <= self.best_loss
            if is_best:
                self.best_loss = loss
            self.save_checkpoint(epoch, is_best=is_best)


def resolve_image_path(ffhq_root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = ffhq_root / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem}.png/.jpg/.jpeg in {ffhq_root}")


def save_cached_sample(path: Path, sample):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(sample, tmp_path)
    tmp_path.replace(path)


def prepare_item(face_name: str, color_name: str, dataset_path: Path, ffhq_root: Path):
    try:
        color_path = dataset_path / "FS" / f"{color_name}.npz"
        color_s = torch.from_numpy(np.load(color_path)["latent_in"]).squeeze(0)

        face_path = dataset_path / "FS" / f"{face_name}.npz"
        align_s = torch.from_numpy(np.load(face_path)["latent_in"]).squeeze(0)

        color_image_path = resolve_image_path(ffhq_root, color_name)
        face_image_path = resolve_image_path(ffhq_root, face_name)

        with Image.open(color_image_path) as image:
            color_i = T.functional.normalize(T.functional.to_tensor(image.convert("RGB")), [0.5], [0.5])
        with Image.open(face_image_path) as image:
            face_i = T.functional.normalize(T.functional.to_tensor(image.convert("RGB")), [0.5], [0.5])

        align_path = dataset_path / "Align" / f"{face_name}_{color_name}.npz"
        data = np.load(align_path)
        align_f = torch.from_numpy(data["latent_F"]).squeeze(0)
        source_inpaint_256 = (
            torch.from_numpy(data["source_inpaint_256"]).squeeze(0).float()
            if "source_inpaint_256" in data
            else T.functional.resize(face_i, [256, 256])
        )
        remove_mask = torch.from_numpy(data["remove_mask"]).squeeze(0).float() if "remove_mask" in data else torch.zeros(1, 256, 256)
        remove_halo = torch.from_numpy(data["remove_halo"]).squeeze(0).float() if "remove_halo" in data else torch.zeros(1, 256, 256)
        body_preserve = torch.from_numpy(data["body_preserve"]).squeeze(0).float() if "body_preserve" in data else torch.zeros(1, 256, 256)
        remove_tail = torch.from_numpy(data["remove_tail"]).squeeze(0).float() if "remove_tail" in data else torch.zeros(1, 256, 256)
        return color_s, align_s, align_f, color_i, face_i, source_inpaint_256, remove_mask, remove_halo, body_preserve, remove_tail
    except Exception as e:
        print(e, file=sys.stderr)
        return None


class BlendingDatasetV4(Dataset):
    def __init__(self, exps, dataset_path: Path, ffhq_root: Path, net_trainer: TrainerV4, split_name: str):
        super().__init__()
        self.dataset_path = dataset_path
        self.ffhq_root = ffhq_root
        self.net_trainer = net_trainer
        self.split_name = split_name
        self.downsample_256_cpu = BicubicDownSample(factor=4, cuda=False)
        self.cache_dir = self._build_cache_dir(exps)
        self.samples_dir = self.cache_dir / "samples"
        self.meta_file = self.cache_dir / "metadata.pt"
        self.entries = self._prepare_cache(exps)
        print(f"dataset[{self.split_name}]: {len(self.entries)} samples", file=sys.stderr)

    def _build_cache_dir(self, exps) -> Path:
        hasher = hashlib.sha1()
        hasher.update(b"blend_v4_supervision_v3")
        hasher.update(str(self.dataset_path.resolve()).encode("utf-8"))
        hasher.update(self.split_name.encode("utf-8"))
        for exp in exps:
            hasher.update((" ".join(exp) + "\n").encode("utf-8"))
        return USER_OUTPUT_DIR / "dataset_cache_v4" / f"{self.split_name}_{hasher.hexdigest()[:12]}"

    @staticmethod
    def _expand_specs(exps) -> list[tuple[str, str]]:
        specs: list[tuple[str, str]] = []
        for face_name, shape_name, color_name in exps:
            specs.append((face_name, color_name))
            specs.append((face_name, shape_name))
        return specs

    def _flush_pending(
        self,
        pending_specs: list[tuple[str, str]],
        pending_raw: list[tuple[torch.Tensor, ...]],
    ) -> list[dict[str, object]]:
        if not pending_raw:
            return []

        valid_entries: list[dict[str, object]] = []
        with torch.no_grad():
            align_f = torch.stack([item[2] for item in pending_raw], dim=0)
            color_i = torch.stack([item[3] for item in pending_raw], dim=0).to(self.net_trainer.device)
            face_i = torch.stack([item[4] for item in pending_raw], dim=0).to(self.net_trainer.device)
            align_s_gpu = torch.stack([item[1] for item in pending_raw], dim=0).to(self.net_trainer.device)
            align_f_gpu = align_f.to(self.net_trainer.device)

            hm_3d, hm_3e = self.net_trainer.generate_mask(color_i)
            hm_1d, _ = self.net_trainer.generate_mask(face_i)
            i_x, _ = self.net_trainer.net.generator(
                [align_s_gpu],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=align_f_gpu,
            )
            hm_xd, _ = self.net_trainer.generate_mask(i_x)

            target_mask = ((1 - hm_1d) * (1 - hm_xd)).cpu()
            hm_3e = hm_3e.cpu()

            for idx, (face_name, color_name) in enumerate(pending_specs):
                if not hm_3e[idx].any():
                    continue

                valid_entries.append(
                    {
                        "face_name": face_name,
                        "color_name": color_name,
                        "target_mask": target_mask[idx].to(torch.bool),
                        "hm_3e": hm_3e[idx].to(torch.bool),
                    }
                )

        return valid_entries

    def _prepare_cache(self, exps) -> list[dict[str, object]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.meta_file.exists():
            metadata = torch.load(self.meta_file, map_location="cpu")
            if isinstance(metadata, dict) and "entries" in metadata:
                return metadata["entries"]
            return metadata

        if self.samples_dir.exists():
            shutil.rmtree(self.samples_dir, ignore_errors=True)

        sample_specs = self._expand_specs(exps)
        entries: list[dict[str, object]] = []
        pending_specs: list[tuple[str, str]] = []
        pending_raw: list[tuple[torch.Tensor, ...]] = []

        for face_name, color_name in tqdm(sample_specs, desc=f"build {self.split_name} cache"):
            raw_item = prepare_item(face_name, color_name, self.dataset_path, self.ffhq_root)
            if raw_item is None:
                continue

            pending_specs.append((face_name, color_name))
            pending_raw.append(raw_item)

            if len(pending_raw) >= CACHE_BUILD_BATCH_SIZE:
                entries.extend(self._flush_pending(pending_specs, pending_raw))
                pending_specs.clear()
                pending_raw.clear()

        if pending_raw:
            entries.extend(self._flush_pending(pending_specs, pending_raw))

        save_cached_sample(self.meta_file, {"entries": entries})
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        sample = self.entries[idx]
        raw_item = prepare_item(sample["face_name"], sample["color_name"], self.dataset_path, self.ffhq_root)
        if raw_item is None:
            raise RuntimeError(f"Failed to load raw sample: {sample['face_name']} -> {sample['color_name']}")
        color_s, align_s, align_f, color_i, face_i, source_inpaint_256, remove_mask, remove_halo, body_preserve, remove_tail = raw_item
        color_i_256 = self.downsample_256_cpu(color_i.unsqueeze(0))[0]
        face_i_256 = self.downsample_256_cpu(face_i.unsqueeze(0))[0]
        return (
            color_s,
            align_s,
            align_f,
            color_i_256,
            face_i_256,
            sample["target_mask"].float(),
            sample["hm_3e"].float(),
            source_inpaint_256,
            remove_mask,
            remove_halo,
            body_preserve,
            remove_tail,
        )


def main():
    seed_everything(USER_RANDOM_SEED)

    exps = []
    with open(USER_DATASET_DIR / "dataset.exps", "r", encoding="utf-8") as file:
        for exp in file.readlines():
            exps.append(list(map(lambda x: x.replace(".png", ""), exp.split())))

    X_train, X_test = train_test_split(exps, test_size=USER_VAL_SIZE, random_state=42)

    net_trainer = TrainerV4()
    train_dataset = BlendingDatasetV4(X_train, USER_DATASET_DIR, USER_FFHQ_ROOT, net_trainer, split_name="train")
    test_dataset = BlendingDatasetV4(X_test, USER_DATASET_DIR, USER_FFHQ_ROOT, net_trainer, split_name="val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=USER_VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
    )

    model = BlendingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
    trainer = TrainerV4(model=model, optimizer=optimizer, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    trainer.train_loop()
    trainer.logger.wandb.finish()


if __name__ == "__main__":
    main()
