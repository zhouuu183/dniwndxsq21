import gc
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Alignment_v4 import Alignment_v4
from models.Embedding import Embedding
from models.Net import Net
from models.SATD_v4 import SATD_v4
from utils.bicubic import BicubicDownSample
from utils.image_utils import equal_replacer
from utils.train import WandbLogger, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
# 指定当前进程可见的物理 GPU。
USER_CUDA_VISIBLE_DEVICES = "0"

# PyTorch 实际使用的设备。通常写 "cuda"。
USER_DEVICE = "cuda"

# SATD 配对数据集目录。这里应指向 satd_gen_v4.py 生成的 dataset.exps 目录。
USER_DATASET_DIR = Path("images/satd_dataset_v4_long2short")

# FFHQ 原图目录。
USER_FFHQ_ROOT = Path("images/FFHQ")

# 输出目录。会保存 SATD_v4 的 checkpoint 和验证图。
USER_OUTPUT_DIR = Path("output/satd_train_v4_long2short3")

# StyleGAN2 权重路径。
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"

# rotate 阶段最佳权重。
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"

# SATD_v4 初始化权重。第一次训练保持空字符串即可。
USER_SATD_INIT_CKPT = ""

# 单次真实 batch size。
USER_BATCH_SIZE = 1

# 目标有效 batch size。比如 batch=1, effective=4 表示累计 4 次梯度后更新。
USER_EFFECTIVE_BATCH_SIZE = 4

# Embedding 阶段的内部 batch。公共服务器内存紧张时建议 1。
USER_ENCODER_BATCH_SIZE = 1

# CPU 线程上限。共享服务器上建议先压到 1，减少线程栈和并行算子的内存峰值。
USER_CPU_THREADS = 1

# DataLoader worker 数量。为稳妥建议先用 0。
USER_NUM_WORKERS = 0

# 是否启用 pin_memory。当前更建议关掉，优先稳定。
USER_PIN_MEMORY = False

# 是否关闭 cudnn benchmark。当前更建议关掉，优先稳定。
USER_DISABLE_CUDNN_BENCHMARK = True

# 总 epoch 数。
USER_EPOCHS = 30

# 学习率。
USER_LR = 1e-4

# 权重衰减。
USER_WEIGHT_DECAY = 1e-6

# 验证集样本数。
USER_VAL_SIZE = 30

# 随机种子。
USER_RANDOM_SEED = 3407

# 每隔多少个 epoch 保存一次 checkpoint.pth 和 epoch_xxx.pth。
USER_SAVE_CHECKPOINT_EVERY = 1

# 每隔多少个 epoch 保存一次本地验证图。
USER_SAVE_VAL_IMAGES_EVERY = 1

# 每轮最多保存多少组验证可视化。
USER_LOG_IMAGE_COUNT = 20

# 是否启用 wandb。服务器没配好时建议 False。
USER_USE_WANDB = False
USER_WANDB_RUN_NAME = "satd_train_v4"
USER_WANDB_PROJECT = "HairFast-SATD-v4"

# SATD 输出和原始 Eq.(8) 对齐结果的融合比例。
USER_SATD_BLEND_V4 = 0.40

# 显式差值边界带宽度。
USER_SATD_BOUNDARY_V4 = 8

# 各损失项权重。
USER_LAMBDA_PRESERVE = 1.5
USER_LAMBDA_REMOVE = 2.25
USER_LAMBDA_REMOVE_HALO = 0.75
USER_LAMBDA_REMOVE_GRAY = 0.60
USER_LAMBDA_SHAPE_EDGE = 2.60
USER_LAMBDA_SHAPE_GRAY = 1.35
USER_LAMBDA_LATENT_KEEP = 0.06
USER_LAMBDA_SHAPE_RGB = 1.35
USER_LAMBDA_BODY_PRESERVE = 1.40
USER_LAMBDA_TAIL_GRAY = 0.90

# 断点恢复。为空表示不恢复。
USER_RESUME_CHECKPOINT = ""
# ========================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

if USER_DISABLE_CUDNN_BENCHMARK and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False

torch.set_num_threads(max(1, USER_CPU_THREADS))
try:
    torch.set_num_interop_threads(max(1, USER_CPU_THREADS))
except RuntimeError:
    pass

if USER_EFFECTIVE_BATCH_SIZE < USER_BATCH_SIZE or USER_EFFECTIVE_BATCH_SIZE % USER_BATCH_SIZE != 0:
    raise RuntimeError("USER_EFFECTIVE_BATCH_SIZE 必须大于等于 USER_BATCH_SIZE，且能被 USER_BATCH_SIZE 整除。")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    g = gray(image)
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    edge_x = F.conv2d(g, kernel_x, padding=1)
    edge_y = F.conv2d(g, kernel_y, padding=1)
    return torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-6)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


def materialize_tensor_tree(obj):
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {key: materialize_tensor_tree(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [materialize_tensor_tree(val) for val in obj]
    if isinstance(obj, tuple):
        return tuple(materialize_tensor_tree(val) for val in obj)
    return obj


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


class SATDPairDataset_v4(Dataset):
    def __init__(self, entries: list[tuple[str, str]], ffhq_root: Path):
        self.entries = entries
        self.ffhq_root = ffhq_root
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.entries)

    def _read_image(self, name: str) -> torch.Tensor:
        path = self.ffhq_root / f"{name}.png"
        if not path.exists():
            path = self.ffhq_root / f"{name}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {name}.png/.jpg in {self.ffhq_root}")
        with Image.open(path) as image:
            return self.to_tensor(image.convert("RGB"))

    def __getitem__(self, idx):
        face, shape = self.entries[idx]
        return {
            "face_name": face,
            "shape_name": shape,
            "face": self._read_image(face),
            "shape": self._read_image(shape),
        }


def collate_pairs(batch):
    return {
        "face_name": [item["face_name"] for item in batch],
        "shape_name": [item["shape_name"] for item in batch],
        "face": torch.stack([item["face"] for item in batch]),
        "shape": torch.stack([item["shape"] for item in batch]),
    }


def load_pairs(dataset_dir: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split()
            if len(items) == 2:
                pairs.append((items[0], items[1]))
    return pairs


class TrainerSATD_v4:
    def __init__(self, train_loader, val_loader):
        self.device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
        self.logger = self._build_logger()
        self.logger.start_logging()

        self.opts = Namespace(
            size=1024,
            ckpt=USER_STYLEGAN_CKPT,
            channel_multiplier=2,
            latent=512,
            n_mlp=8,
            device=str(self.device),
            batch_size=USER_ENCODER_BATCH_SIZE,
            save_all=False,
            save_all_dir=Path("output"),
            mixing=0.95,
            smooth=5,
            rotate_checkpoint=USER_ROTATE_CKPT,
            use_satd_v4=False,
            satd_checkpoint_v4="",
            satd_blend_v4=USER_SATD_BLEND_V4,
            satd_boundary_v4=USER_SATD_BOUNDARY_V4,
        )

        self.net = Net(self.opts)
        self.embed = Embedding(self.opts, net=self.net).eval()
        self.align = Alignment_v4(self.opts, latent_encoder=self.embed.get_e4e_embed, net=self.net, satd_model=None)
        self.satd = SATD_v4().to(self.device)
        if USER_SATD_INIT_CKPT:
            ckpt = torch.load(USER_SATD_INIT_CKPT, map_location=self.device)
            self.satd.load_state_dict(ckpt.get("satd_state_dict", ckpt.get("model_state_dict", ckpt)), strict=False)

        self.downsample_256 = BicubicDownSample(factor=4)
        toggle_grad(self.net.generator, False)
        toggle_grad(self.satd, True)

        self.optimizer = torch.optim.Adam(self.satd.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
        self.grad_accum_steps = max(1, USER_EFFECTIVE_BATCH_SIZE // USER_BATCH_SIZE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_loss = float("inf")
        self.cur_iter = 0
        self.output_ckpt_dir = USER_OUTPUT_DIR / "checkpoints"
        self.output_val_dir = USER_OUTPUT_DIR / "val_images"
        self.output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)

    def _build_logger(self):
        if USER_USE_WANDB:
            return WandbLogger(name=USER_WANDB_RUN_NAME, project=USER_WANDB_PROJECT)
        return NullLogger()

    def build_name_to_embed(self, batch):
        images_to_name = defaultdict(list)
        sample_names = []
        bsz = batch["face"].shape[0]
        for idx in range(bsz):
            face, shape = equal_replacer([
                batch["face"][idx],
                batch["shape"][idx],
            ])
            names = {"face": f"face_{idx}", "shape": f"shape_{idx}"}
            images_to_name[face].append(names["face"])
            images_to_name[shape].append(names["shape"])
            sample_names.append(names)
        with torch.no_grad():
            name_to_embed = self.embed.embedding_images(images_to_name)
        return name_to_embed, sample_names

    def forward_batch(self, batch):
        name_to_embed, sample_names = self.build_name_to_embed(batch)

        pred_images = []
        latent_list = []
        info_list = []
        for names in sample_names:
            with torch.no_grad():
                info = self.align.prepare_satd_features(names["face"], names["shape"], name_to_embed)
            info = materialize_tensor_tree(info)

            satd_out, satd_aux = self.satd(
                F_src=info["latent_F_src"],
                F_src_inpaint=info["latent_F_src_inpaint"],
                F_shape_inpaint=info["latent_F_shape_inpaint"],
                F_ref=info["latent_F_ref"],
                masks_256=info["satd_masks_256"],
                source_rgb_256=info["source_image_256"],
            )
            transfer_mask, cleanup_mask = self.align._satd_residual_masks(
                info["delta_masks"],
                out_hw=info["latent_F_eq8"].shape[-2:],
            )
            satd_delta = satd_out - info["latent_F_eq8"]
            latent_F = info["latent_F_eq8"] + USER_SATD_BLEND_V4 * transfer_mask * satd_delta
            latent_F = latent_F + (0.60 * USER_SATD_BLEND_V4) * cleanup_mask * satd_delta
            pred_image, _ = self.net.generator(
                [info["latent_S_src"]],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F,
            )
            pred_images.append(self.downsample_256(pred_image))
            latent_list.append(latent_F)
            info_list.append((info, satd_aux))

        return torch.cat(pred_images, dim=0), latent_list, info_list

    def calc_losses(self, pred_images, latent_list, info_list):
        losses = defaultdict(float)
        for idx, (latent_F, (info, _)) in enumerate(zip(latent_list, info_list)):
            pred = pred_images[idx:idx + 1]
            delta_masks = info["delta_masks"]
            M_add = delta_masks["M_add"]
            M_remove = delta_masks["M_remove"]
            M_keep = delta_masks["M_keep"]
            M_boundary = delta_masks["M_boundary"]
            M_remove_halo = delta_masks.get("M_remove_halo", torch.zeros_like(M_remove))
            M_ref_overlap = delta_masks.get("M_ref_overlap", torch.zeros_like(M_add))
            M_body_preserve = delta_masks.get("M_body_preserve", torch.zeros_like(M_add))
            M_remove_tail = delta_masks.get("M_remove_tail", torch.zeros_like(M_add))

            preserve_mask = (1 - (M_add + M_remove + 0.75 * M_remove_halo + 0.50 * M_boundary).clamp(0, 1)).clamp(0, 1)
            shape_region = (M_add + 0.45 * M_keep + M_ref_overlap + 0.75 * M_boundary).clamp(0, 1)
            remove_region = (M_remove + 0.55 * M_remove_halo + 0.35 * M_remove_tail).clamp(0, 1)
            remove_region = (remove_region * (1 - 0.75 * M_body_preserve)).clamp(0, 1)
            tail_region = (M_remove_tail * (1 - M_body_preserve)).clamp(0, 1)

            losses["preserve"] += masked_l1(pred, info["source_image_256"], preserve_mask)
            losses["body_preserve"] += masked_l1(pred, info["source_image_256"], M_body_preserve)
            losses["remove"] += masked_l1(pred, info["source_inpaint_256"], M_remove)
            losses["remove_halo"] += masked_l1(pred, info["source_inpaint_256"], M_remove_halo)
            losses["remove_gray"] += masked_l1(gray(pred), gray(info["source_inpaint_256"]), remove_region)
            losses["tail_gray"] += masked_l1(gray(pred), gray(info["source_inpaint_256"]), tail_region)

            pred_edges = sobel_edges(pred)
            shape_edges = sobel_edges(info["shape_inpaint_256"])
            losses["shape_edge"] += masked_l1(pred_edges, shape_edges, shape_region)

            losses["shape_gray"] += masked_l1(gray(pred), gray(info["shape_inpaint_256"]), shape_region)
            losses["shape_rgb"] += masked_l1(
                pred,
                info["shape_inpaint_256"],
                (M_add + 0.45 * M_keep + 0.60 * M_ref_overlap + 0.75 * M_boundary).clamp(0, 1),
            )

            keep32 = F.interpolate((M_keep * (1 - 0.55 * M_ref_overlap)).clamp(0, 1), size=(32, 32), mode="nearest")
            losses["latent_keep"] += masked_l1(latent_F, info["latent_F_eq8"], keep32)

        count = max(len(info_list), 1)
        for key in list(losses.keys()):
            losses[key] = losses[key] / count

        total_loss = (
            USER_LAMBDA_PRESERVE * losses["preserve"]
            + USER_LAMBDA_BODY_PRESERVE * losses["body_preserve"]
            + USER_LAMBDA_REMOVE * losses["remove"]
            + USER_LAMBDA_REMOVE_HALO * losses["remove_halo"]
            + USER_LAMBDA_REMOVE_GRAY * losses["remove_gray"]
            + USER_LAMBDA_TAIL_GRAY * losses["tail_gray"]
            + USER_LAMBDA_SHAPE_EDGE * losses["shape_edge"]
            + USER_LAMBDA_SHAPE_GRAY * losses["shape_gray"]
            + USER_LAMBDA_LATENT_KEEP * losses["latent_keep"]
            + USER_LAMBDA_SHAPE_RGB * losses["shape_rgb"]
        )
        losses["loss"] = total_loss
        return total_loss, losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "satd_state_dict": self.satd.state_dict(),
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

        torch.save({"satd_state_dict": self.satd.state_dict()}, self.output_ckpt_dir / "satd_for_infer.pth")

    def save_preview(self, epoch: int, preview_rows: list[list[torch.Tensor]]):
        if epoch % USER_SAVE_VAL_IMAGES_EVERY != 0:
            return
        save_dir = self.output_val_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        to_pil = T.ToPILImage()
        for idx, row in enumerate(preview_rows[:USER_LOG_IMAGE_COUNT]):
            grid = torch.cat([((img[0] + 1) / 2).clamp(0, 1) for img in row], dim=2)
            to_pil(grid.cpu()).save(save_dir / f"sample_{idx:03d}.png")

    def run_epoch(self, loader, training: bool):
        self.satd.train(training)
        if not training:
            self.satd.eval()

        total = defaultdict(float)
        preview_rows = []
        collect_preview = (not training) and USER_LOG_IMAGE_COUNT > 0
        steps_in_epoch = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for step, batch in enumerate(tqdm(loader)):
                steps_in_epoch += 1
                pred_images, latent_list, info_list = self.forward_batch(batch)
                loss, losses = self.calc_losses(pred_images, latent_list, info_list)

                if training:
                    (loss / self.grad_accum_steps).backward()
                    if (step + 1) % self.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.satd.parameters(), 5.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.logger.next_step()
                    self.cur_iter += 1

                for key, val in losses.items():
                    total[key] += float(val.detach().cpu())

                if collect_preview and len(preview_rows) < USER_LOG_IMAGE_COUNT:
                    for idx, (info, _) in enumerate(info_list):
                        preview_rows.append(
                            [
                                info["source_image_256"],
                                info["shape_image_256"],
                                info["source_inpaint_256"],
                                pred_images[idx:idx + 1],
                            ]
                        )

                del pred_images, latent_list, info_list, loss, losses
                if (step + 1) % 20 == 0:
                    gc.collect()

        if training and steps_in_epoch % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.satd.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        for key in list(total.keys()):
            total[key] /= max(steps_in_epoch, 1)
        return total, preview_rows

    def train_loop(self):
        start_epoch = 0
        if USER_RESUME_CHECKPOINT:
            ckpt = torch.load(USER_RESUME_CHECKPOINT, map_location=self.device)
            self.satd.load_state_dict(ckpt["satd_state_dict"], strict=False)
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.best_loss = ckpt.get("best_loss", self.best_loss)
            self.cur_iter = ckpt.get("cur_iter", 0)
            start_epoch = ckpt.get("epoch", -1) + 1

        for epoch in range(start_epoch, USER_EPOCHS):
            train_losses, _ = self.run_epoch(self.train_loader, training=True)
            val_losses, preview_rows = self.run_epoch(self.val_loader, training=False)

            self.logger.log("epoch", epoch)
            for key, val in train_losses.items():
                self.logger.log(f"train/{key}", val)
            for key, val in val_losses.items():
                self.logger.log(f"val/{key}", val)
            self.save_preview(epoch, preview_rows)

            is_best = val_losses["loss"] <= self.best_loss
            if is_best:
                self.best_loss = val_losses["loss"]
            self.save_checkpoint(epoch, is_best=is_best)


def main():
    set_seed(USER_RANDOM_SEED)
    pairs = load_pairs(USER_DATASET_DIR)
    if len(pairs) <= USER_VAL_SIZE:
        raise RuntimeError("dataset.exps is smaller than USER_VAL_SIZE; please generate more pairs.")

    train_pairs, val_pairs = train_test_split(pairs, test_size=USER_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = SATDPairDataset_v4(train_pairs, USER_FFHQ_ROOT)
    val_dataset = SATDPairDataset_v4(val_pairs, USER_FFHQ_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_pairs,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_pairs,
        drop_last=False,
    )

    trainer = TrainerSATD_v4(train_loader, val_loader)
    trainer.train_loop()
    trainer.logger.wandb.finish()


if __name__ == "__main__":
    main()
