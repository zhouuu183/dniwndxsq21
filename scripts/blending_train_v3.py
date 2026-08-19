import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

from models.Alignment_v3 import Alignment_v3
from models.Embedding import Embedding
from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.SATD_v3 import SATD_v3
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion, equal_replacer
from utils.train import WandbLogger, get_fid_calc, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
# 指定当前进程可见的物理 GPU 编号。
# 例如只想用第 3 张卡，就写 "3"；想用第 0、1 两张卡就写 "0,1"。
USER_CUDA_VISIBLE_DEVICES = "2"

# PyTorch 实际使用的设备。
# 如果上面已经设置了 CUDA_VISIBLE_DEVICES="3"，这里通常写 "cuda" 就够了。
# 不要写成 "3" 这种字符串，torch 不认。
USER_DEVICE = "cuda"

# blending v3 三元组数据集目录。
# 这里应该指向 blending_gen_v3.py 生成出 dataset.exps 的目录。
# 说明：
# 1. v3 训练默认按论文通用 triples 设定读取 (face, shape, color)；
# 2. both / full 主要是推理、测试、算指标时的使用模式区分；
# 3. 所以训练时通常不需要专门为了 both/full 改训练脚本，只需要保证 dataset.exps 的采样策略和你的实验目标一致。
USER_DATASET_DIR = Path("images/blending_dataset_v3")

# FFHQ 原图目录。
# 训练时会按 dataset.exps 去这里读取 face / shape / color 三张原图。
USER_FFHQ_ROOT = Path("images/FFHQ")

# 当前这次训练的输出目录。
# 会在这里保存 checkpoints、验证图片、FID 缓存等内容。
USER_OUTPUT_DIR = Path("output/blending_train_v3")

# StyleGAN2 预训练权重路径。
# 一般保持仓库默认即可，除非你服务器上放在别的位置。
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"

# rotate 阶段训练好的最佳权重。
# v3 的 Alignment 仍然依赖这个权重做姿态/形状对齐。
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"

# SATD_v3 初始化权重。
# 第一次正式训练时保持空字符串 ""，表示从随机初始化开始。
# 如果你已经训练过一次 v3，想继续微调，可以填上次导出的 satd_for_infer.pth 路径。
USER_SATD_INIT_CKPT = ""

# 单次真实 batch size。
# 这是一次 forward/backward 实际喂进 GPU 的样本数，直接决定显存占用。
USER_BATCH_SIZE = 2

# 目标“有效 batch size”。
# 如果显存不够，可以用梯度累计来模拟更大的 batch。
# 例如：
#   USER_BATCH_SIZE = 4
#   USER_EFFECTIVE_BATCH_SIZE = 16
# 就表示累计 4 次梯度后再 optimizer.step() 一次，
# 等价于“近似” batch=16 的训练效果。
# 注意：这里必须能被 USER_BATCH_SIZE 整除。
USER_EFFECTIVE_BATCH_SIZE = 4

# Embedding 阶段内部一次编码多少张图。
# 这个值只影响前面的图像编码吞吐，不等于训练 batch。
# 显存紧张时可以适当调小，比如 3 或 4。
USER_ENCODER_BATCH_SIZE = 3

# DataLoader 的 worker 数量。
# Windows/某些环境下先用 0 最稳；Linux 服务器通常可以改大，比如 4 或 8。
USER_NUM_WORKERS = 0

# DataLoader 是否启用 pin_memory。
# 当前 v3 在线跑 e4e + SEAN + StyleGAN，链路比较重；某些环境下 pin_memory 反而更不稳定。
# 如果你遇到 segmentation fault，建议先保持 False。
USER_PIN_MEMORY = False

# 总训练 epoch 数。
USER_EPOCHS = 50

# 学习率。
# 这是 SATD_v3 + ClipBlendingModel 联合训练的主学习率。
USER_LR = 1e-4

# 权重衰减。
USER_WEIGHT_DECAY = 1e-6

# 验证集样本数。
# 会从 dataset.exps 里固定切出这么多三元组做验证。
USER_VAL_SIZE = 64

# 随机种子。
USER_RANDOM_SEED = 3407

# 每隔多少个 epoch 保存一次 checkpoint.pth 和 epoch_xxx.pth。
# 同时 last.pth 每个 epoch 都会更新。
USER_SAVE_CHECKPOINT_EVERY = 1

# 每隔多少个 epoch 保存一次本地验证图片。
USER_SAVE_VAL_IMAGES_EVERY = 1

# 每轮最多保存多少组验证可视化。
# 每组可视化会包含：face / shape / color / result。
USER_LOG_IMAGE_COUNT = 24

# 是否启用 wandb 日志。
# 如果服务器没有配置好 WANDB_KEY，建议先设为 False。
USER_USE_WANDB = True

# wandb 的 run 名称。
USER_WANDB_RUN_NAME = "blending_train_v3"

# wandb 的 project 名称。
USER_WANDB_PROJECT = "HairFast-Blending-v3"

# 用于计算 FID 的“真实图片目录”。
# 不需要算 FID 时保持 None。
# 如果要算，就填一个真实图片文件夹路径，例如：
# Path("fid_images")
USER_FID_IMAGE_DIR = None

# SATD 输出和原始 Eq.(8) 对齐结果的融合比例。
# 值越大，SATD 对最终 F_align 的影响越强。
# 建议先从 0.25~0.35 开始，太大容易把原模型已有能力带崩。
USER_SATD_BLEND_V3 = 0.15

# 显式差值边界带宽度。
# 越大表示 M_boundary 更宽，阴影和边缘会被更大范围地当成过渡区处理。
USER_SATD_BOUNDARY_V3 = 5

# 以下是各个损失项的权重。
# face_clip：非头发目标区的人脸/身份一致性
USER_LAMBDA_FACE_CLIP = 1.0
# hair_clip：目标头发区域和参考发色/外观的一致性
USER_LAMBDA_HAIR_CLIP = 1.0
# preserve：非编辑区尽量保留原图，避免背景和五官被误改
USER_LAMBDA_PRESERVE = 2.0
# remove：显式约束 remove 区参考原图背景/去遮挡恢复
USER_LAMBDA_REMOVE = 1.5
# shape_edge：目标发型轮廓/边缘更贴近 shape 参考
USER_LAMBDA_SHAPE_EDGE = 1.5
# shape_gray：目标发型的整体灰度结构更贴近 shape 参考
USER_LAMBDA_SHAPE_GRAY = 1.0
# color_stats：目标头发区域的颜色统计更贴近 color 参考
USER_LAMBDA_COLOR_STATS = 0.5
# latent_keep：keep 区尽量别偏离原始 Eq.(8) 太多，降低训练发散风险
USER_LAMBDA_LATENT_KEEP = 0.5

# 是否关闭 cudnn benchmark。
# benchmark=True 在某些固定 shape 任务上会更快，但当前这条生成链更建议优先稳定。
USER_DISABLE_CUDNN_BENCHMARK = True
# ========================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

if USER_DISABLE_CUDNN_BENCHMARK and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False

if USER_EFFECTIVE_BATCH_SIZE < USER_BATCH_SIZE or USER_EFFECTIVE_BATCH_SIZE % USER_BATCH_SIZE != 0:
    raise RuntimeError("USER_EFFECTIVE_BATCH_SIZE 必须大于等于 USER_BATCH_SIZE，且能被 USER_BATCH_SIZE 整除。")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_tensor_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {key: clone_tensor_tree(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [clone_tensor_tree(val) for val in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tensor_tree(val) for val in obj)
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


class V3TripletDataset(Dataset):
    def __init__(self, entries: list[tuple[str, str, str]], ffhq_root: Path):
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
        return self.to_tensor(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        face, shape, color = self.entries[idx]
        return {
            "face_name": face,
            "shape_name": shape,
            "color_name": color,
            "face": self._read_image(face),
            "shape": self._read_image(shape),
            "color": self._read_image(color),
        }


def collate_triplets(batch):
    return {
        "face_name": [item["face_name"] for item in batch],
        "shape_name": [item["shape_name"] for item in batch],
        "color_name": [item["color_name"] for item in batch],
        "face": torch.stack([item["face"] for item in batch]),
        "shape": torch.stack([item["shape"] for item in batch]),
        "color": torch.stack([item["color"] for item in batch]),
    }


def load_experiments(dataset_dir: Path) -> list[tuple[str, str, str]]:
    exps = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split()
            if len(items) == 3:
                exps.append(tuple(items))
    return exps


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    edge_x = F.conv2d(gray, kernel_x, padding=1)
    edge_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-6)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


def masked_stats(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.shape[1] == 1:
        mask = mask.expand(-1, image.shape[1], -1, -1)
    denom = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    mean = (image * mask).sum(dim=(2, 3), keepdim=True) / denom
    var = (((image - mean) * mask) ** 2).sum(dim=(2, 3), keepdim=True) / denom
    std = torch.sqrt(var + 1e-6)
    return mean, std


class TrainerV3:
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
            satd_boundary_v3=USER_SATD_BOUNDARY_V3,
            satd_blend_v3=USER_SATD_BLEND_V3,
            use_satd_v3=False,
            satd_checkpoint_v3="",
        )

        self.net = Net(self.opts)
        self.embed = Embedding(self.opts, net=self.net).eval()
        self.satd = SATD_v3().to(self.device)
        if USER_SATD_INIT_CKPT:
            ckpt = torch.load(USER_SATD_INIT_CKPT, map_location=self.device)
            self.satd.load_state_dict(ckpt.get("satd_state_dict", ckpt.get("model_state_dict", ckpt)), strict=False)

        self.align = Alignment_v3(self.opts, latent_encoder=self.embed.get_e4e_embed, net=self.net, satd_model=self.satd)
        self.blending = ClipBlendingModel().to(self.device)

        self.downsample_256 = BicubicDownSample(factor=4)
        self.dilate_erosion = DilateErosion(device=str(self.device))

        toggle_grad(self.net.generator, False)
        toggle_grad(self.satd, True)
        toggle_grad(self.blending, True)

        self.optimizer = torch.optim.Adam(
            list(self.satd.parameters()) + list(self.blending.parameters()),
            lr=USER_LR,
            weight_decay=USER_WEIGHT_DECAY,
        )
        self.grad_accum_steps = max(1, USER_EFFECTIVE_BATCH_SIZE // USER_BATCH_SIZE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_loss = float("inf")
        self.cur_iter = 0
        self.output_ckpt_dir = USER_OUTPUT_DIR / "checkpoints"
        self.output_val_dir = USER_OUTPUT_DIR / "val_images"
        self.output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)

        self.fid_calc = None
        if USER_FID_IMAGE_DIR:
            self.fid_calc = get_fid_calc(str(USER_OUTPUT_DIR / "fid_clip_v3.pkl"), str(USER_FID_IMAGE_DIR), device=self.device)

    def _build_logger(self):
        if USER_USE_WANDB:
            return WandbLogger(name=USER_WANDB_RUN_NAME, project=USER_WANDB_PROJECT)
        return NullLogger()

    def build_name_to_embed(self, batch):
        images_to_name = defaultdict(list)
        sample_names = []
        bsz = batch["face"].shape[0]
        for idx in range(bsz):
            face, shape, color = equal_replacer([
                batch["face"][idx].clone(),
                batch["shape"][idx].clone(),
                batch["color"][idx].clone(),
            ])
            names = {"face": f"face_{idx}", "shape": f"shape_{idx}", "color": f"color_{idx}"}
            images_to_name[face].append(names["face"])
            images_to_name[shape].append(names["shape"])
            images_to_name[color].append(names["color"])
            sample_names.append(names)
        with torch.no_grad():
            name_to_embed = self.embed.embedding_images(images_to_name)
        # embedding_images runs under inference_mode in the original repo.
        # Clone tensors here so they can safely be consumed by trainable modules.
        return clone_tensor_tree(name_to_embed), sample_names

    def build_batch_features(self, name_to_embed, sample_names):
        satd_inputs = []
        blend_inputs = []
        for names in sample_names:
            with torch.no_grad():
                shape_info = self.align.prepare_satd_features(names["face"], names["shape"], name_to_embed)
                face_mask = name_to_embed[names["face"]]["mask"]
                color_mask = name_to_embed[names["color"]]["mask"]
                HM_1D, _ = self.dilate_erosion.hair_from_mask(face_mask)
                HM_3D, HM_3E = self.dilate_erosion.hair_from_mask(color_mask)
                HM_X = shape_info["HM_X"]
                HM_XD, _ = self.dilate_erosion.mask(HM_X.float())
                target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

            satd_inputs.append(clone_tensor_tree(shape_info))
            blend_inputs.append(
                {
                    "target_mask": target_mask.detach().clone(),
                    "HM_3E": HM_3E.detach().clone(),
                    "latent_S_face": name_to_embed[names["face"]]["S"].detach().clone(),
                    "latent_S_color": name_to_embed[names["color"]]["S"].detach().clone(),
                    "face_image_norm": name_to_embed[names["face"]]["image_norm_256"].detach().clone(),
                    "color_image_norm": name_to_embed[names["color"]]["image_norm_256"].detach().clone(),
                    "face_image_256": (name_to_embed[names["face"]]["image_256"] * 2 - 1).detach().clone(),
                    "color_image_256": (name_to_embed[names["color"]]["image_256"] * 2 - 1).detach().clone(),
                }
            )
        return satd_inputs, blend_inputs

    def forward_batch(self, batch):
        name_to_embed, sample_names = self.build_name_to_embed(batch)
        satd_inputs, blend_inputs = self.build_batch_features(name_to_embed, sample_names)

        latent_F_align_list = []
        S_blend_list = []
        aux_list = []
        for satd_info, blend_info in zip(satd_inputs, blend_inputs):
            satd_out, satd_aux = self.satd(
                F_src=satd_info["latent_F_src"],
                F_src_inpaint=satd_info["latent_F_src_inpaint"],
                F_shape_inpaint=satd_info["latent_F_shape_inpaint"],
                F_ref=satd_info["latent_F_ref"],
                masks_256=satd_info["satd_masks_256"],
                source_rgb_256=satd_info["source_image_256"] * 2 - 1,
            )
            latent_F_align = (1 - USER_SATD_BLEND_V3) * satd_info["latent_F_eq8"] + USER_SATD_BLEND_V3 * satd_out
            latent_F_align_list.append(latent_F_align)
            aux_list.append((satd_info, blend_info, satd_aux))

            S_blend_6_18 = self.blending(
                blend_info["latent_S_face"][:, 6:],
                blend_info["latent_S_color"][:, 6:],
                blend_info["face_image_norm"] * blend_info["target_mask"],
                blend_info["color_image_norm"] * blend_info["HM_3E"],
            )
            # v3 training keeps the original blending_train convention:
            # only the 6:18 style layers are learned here.
            S_blend_list.append(S_blend_6_18)

        latent_F_align = torch.cat(latent_F_align_list, dim=0)
        S_blend = torch.cat(S_blend_list, dim=0)
        latent_in = torch.cat((torch.zeros(S_blend.shape[0], 6, 512, device=self.device), S_blend), dim=1)
        I_G, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=latent_F_align)
        I_G_256 = self.downsample_256(I_G)
        return I_G_256, aux_list, latent_F_align_list

    def calc_losses(self, I_G_256, aux_list, latent_F_align_list):
        losses = defaultdict(float)
        for idx, ((satd_info, blend_info, _), latent_F_align) in enumerate(zip(aux_list, latent_F_align_list)):
            gen = I_G_256[idx:idx + 1]
            face = blend_info["face_image_256"]
            color = blend_info["color_image_256"]
            target_mask = blend_info["target_mask"]
            hair_mask = blend_info["HM_3E"]

            face_embed = self.blending.get_image_embed(gen * target_mask)
            gt_face_embed = self.blending.get_image_embed(face * target_mask)
            losses["face_clip"] += (1 - F.cosine_similarity(face_embed, gt_face_embed)).mean()

            hair_embed = self.blending.get_image_embed(gen * hair_mask)
            gt_hair_embed = self.blending.get_image_embed(color * hair_mask)
            losses["hair_clip"] += (1 - F.cosine_similarity(hair_embed, gt_hair_embed)).mean()

            delta_masks = satd_info["delta_masks"]
            M_add = delta_masks["M_add"]
            M_remove = delta_masks["M_remove"]
            M_keep = delta_masks["M_keep"]
            M_boundary = delta_masks["M_boundary"]
            preserve_mask = (1 - (M_add + M_remove + M_keep + 0.5 * M_boundary).clamp(0, 1)).clamp(0, 1)

            losses["preserve"] += masked_l1(gen, face, preserve_mask)
            losses["remove"] += masked_l1(gen, satd_info["source_inpaint_256"], M_remove)

            gen_edges = sobel_edges(gen)
            shape_edges = sobel_edges(satd_info["shape_inpaint_256"])
            losses["shape_edge"] += masked_l1(gen_edges, shape_edges, (M_add + M_keep + M_boundary).clamp(0, 1))

            gen_gray = 0.299 * gen[:, 0:1] + 0.587 * gen[:, 1:2] + 0.114 * gen[:, 2:3]
            shape_gray = 0.299 * satd_info["shape_inpaint_256"][:, 0:1] + 0.587 * satd_info["shape_inpaint_256"][:, 1:2] + 0.114 * satd_info["shape_inpaint_256"][:, 2:3]
            losses["shape_gray"] += masked_l1(gen_gray, shape_gray, M_add + 0.5 * M_boundary)

            gen_mean, gen_std = masked_stats(gen, delta_masks["M_tgt"])
            color_mean, color_std = masked_stats(color, hair_mask)
            losses["color_stats"] += F.l1_loss(gen_mean, color_mean) + F.l1_loss(gen_std, color_std)

            keep32 = F.interpolate(M_keep, size=(32, 32), mode="nearest")
            losses["latent_keep"] += masked_l1(latent_F_align, satd_info["latent_F_eq8"], keep32)

        count = max(len(aux_list), 1)
        for key in list(losses.keys()):
            losses[key] = losses[key] / count

        total_loss = (
            USER_LAMBDA_FACE_CLIP * losses["face_clip"]
            + USER_LAMBDA_HAIR_CLIP * losses["hair_clip"]
            + USER_LAMBDA_PRESERVE * losses["preserve"]
            + USER_LAMBDA_REMOVE * losses["remove"]
            + USER_LAMBDA_SHAPE_EDGE * losses["shape_edge"]
            + USER_LAMBDA_SHAPE_GRAY * losses["shape_gray"]
            + USER_LAMBDA_COLOR_STATS * losses["color_stats"]
            + USER_LAMBDA_LATENT_KEEP * losses["latent_keep"]
        )
        losses["loss"] = total_loss
        return total_loss, losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "satd_state_dict": self.satd.state_dict(),
            "blending_state_dict": self.blending.state_dict(),
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

        # Export inference-friendly checkpoints for hair_swap_v3.
        torch.save(
            {
                "model_state_dict": self.blending.state_dict(),
                "clip": "ViT-B/32",
            },
            self.output_ckpt_dir / "blending_for_infer.pth",
        )
        torch.save(
            {
                "satd_state_dict": self.satd.state_dict(),
            },
            self.output_ckpt_dir / "satd_for_infer.pth",
        )

    def save_preview(self, epoch: int, preview_triplets: list[list[torch.Tensor]]):
        if epoch % USER_SAVE_VAL_IMAGES_EVERY != 0:
            return
        save_dir = self.output_val_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        to_pil = T.ToPILImage()
        for idx, trio in enumerate(preview_triplets[:USER_LOG_IMAGE_COUNT]):
            grid = torch.cat([((img[0] + 1) / 2).clamp(0, 1) for img in trio], dim=2)
            to_pil(grid.cpu()).save(save_dir / f"sample_{idx:03d}.png")

    def run_epoch(self, loader, training: bool):
        self.satd.train(training)
        self.blending.train(training)
        if not training:
            self.satd.eval()
            self.blending.eval()

        total = defaultdict(float)
        preview_triplets = []
        fid_images = []
        steps_in_epoch = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for step, batch in enumerate(tqdm(loader)):
                steps_in_epoch += 1
                I_G_256, aux_list, latent_F_align_list = self.forward_batch(batch)
                loss, losses = self.calc_losses(I_G_256, aux_list, latent_F_align_list)

                if training:
                    (loss / self.grad_accum_steps).backward()
                    if (step + 1) % self.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(list(self.satd.parameters()) + list(self.blending.parameters()), 5.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.logger.next_step()
                    self.cur_iter += 1

                for key, value in losses.items():
                    total[key] += float(value.detach().cpu())

                for idx, (satd_info, blend_info, _) in enumerate(aux_list[: max(0, USER_LOG_IMAGE_COUNT - len(preview_triplets))]):
                    preview_triplets.append([
                        blend_info["face_image_256"].detach().cpu(),
                        satd_info["shape_image_256"].detach().cpu() * 2 - 1,
                        blend_info["color_image_256"].detach().cpu(),
                        I_G_256[idx:idx + 1].detach().cpu(),
                    ])

                if self.fid_calc is not None:
                    fid_images.append(T.Resize((299, 299))(((I_G_256 + 1) / 2).clamp(0, 1)).detach().cpu())

        if training and steps_in_epoch % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(list(self.satd.parameters()) + list(self.blending.parameters()), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        count = max(len(loader), 1)
        averaged = {key: value / count for key, value in total.items()}
        if self.fid_calc is not None and fid_images:
            averaged["fid_clip"] = float(self.fid_calc(torch.cat(fid_images)).cpu())
        return averaged, preview_triplets

    def log_losses(self, prefix: str, losses: dict):
        for key, value in losses.items():
            self.logger.log(f"{prefix}/{key}", value)

    def train_loop(self):
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, USER_EPOCHS + 1):
            train_losses, _ = self.run_epoch(self.train_loader, training=True)
            val_losses, previews = self.run_epoch(self.val_loader, training=False)

            self.log_losses("train", train_losses)
            self.log_losses("val", val_losses)
            self.save_preview(epoch, previews)

            current = val_losses["loss"]
            is_best = current <= self.best_loss
            if is_best:
                self.best_loss = current
            self.save_checkpoint(epoch, is_best=is_best)


def main():
    set_seed(USER_RANDOM_SEED)
    exps = load_experiments(USER_DATASET_DIR)
    if len(exps) <= USER_VAL_SIZE:
        raise RuntimeError("dataset.exps is smaller than USER_VAL_SIZE; please generate more triplets.")

    train_exps, val_exps = train_test_split(exps, test_size=USER_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = V3TripletDataset(train_exps, USER_FFHQ_ROOT)
    val_dataset = V3TripletDataset(val_exps, USER_FFHQ_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_triplets,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_triplets,
        drop_last=False,
    )

    trainer = TrainerV3(train_loader, val_loader)
    trainer.train_loop()
    trainer.logger.wandb.finish()


if __name__ == "__main__":
    main()
