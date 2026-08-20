import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)

for module_name in ("utils", "datasets", "models"):
    loaded = sys.modules.get(module_name)
    loaded_file = getattr(loaded, "__file__", "") if loaded is not None else ""
    if loaded is not None and loaded_file and not str(loaded_file).startswith(ROOT_DIR_STR):
        del sys.modules[module_name]

from datasets.spsa_dataset_v13 import SPSAAlignmentDataset, split_manifest_records
from hair_swap_v13 import HairFast_v13, get_parser_v13
from models.face_parsing.model import BiSeNet
from utils.image_utils import equal_replacer
from utils.spsa_eval_v13 import make_fixed_pair_panels, save_fixed_pair_panels
from utils.spsa_precompute_v13 import load_prior_npz, read_image_tensor, write_manifest
from utils.train import WandbLogger, image_grid, seed_everything, toggle_grad

# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/spsa_dataset_v13")
USER_OUTPUT_DIR_FFHQ = Path("output/spsa_train_v13")
USER_VAL_SIZE_FFHQ = 64

USER_SMALL_DATASET_DIR = Path("images/spsa_dataset_v13_small")
USER_SMALL_OUTPUT_DIR = Path("output/spsa_train_v13_small")
USER_SMALL_VAL_SIZE = 15

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_SPSA_INIT_CKPT = ""

USER_BATCH_SIZE = 1
USER_EFFECTIVE_BATCH_SIZE = 2
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_DISABLE_CUDNN_BENCHMARK = True

USER_EPOCHS = 20
USER_LR = 1e-4
USER_WEIGHT_DECAY = 1e-6
USER_RANDOM_SEED = 3407

USER_SAVE_CHECKPOINT_EVERY = 1
USER_LOG_IMAGE_COUNT = 8
USER_VISUAL_VAL_PAIR_COUNT = 15
USER_RUN_INITIAL_VAL = False

USER_USE_WANDB = False
USER_WANDB_RUN_NAME = "spsa_train_v13"
USER_WANDB_PROJECT = "HairFast-SPSA-v13"

USER_SPSA_HIDDEN_CHANNELS_V13 = 128
USER_SPSA_BANG_ROI_SIZE_V13 = 8
USER_SPSA_TAIL_ROI_SIZE_V13 = 8
USER_SPSA_ALPHA_INIT_V13 = 0.4
USER_SPSA_BETA_BANG_INIT_V13 = 0.15
USER_SPSA_BETA_TAIL_INIT_V13 = 0.15

USER_LAMBDA_MASK_V13 = 0.80
USER_LAMBDA_EDGE_V13 = 0.40
USER_LAMBDA_ORI_V13 = 0.30
USER_LAMBDA_ROI_V13 = 0.60
USER_LAMBDA_PRESERVE_V13 = 0.10
USER_LAMBDA_ANCHOR_V13 = 0.005
USER_ALIGNMENT_STAGE_WEIGHT_V13 = 0.50

USER_RESUME_CHECKPOINT = ""
# ========================================================================


def resolve_profile_defaults():
    if USER_DATASET_PROFILE == "ffhq":
        return {
            "dataset_profile": "ffhq",
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        }
    if USER_DATASET_PROFILE == "small":
        return {
            "dataset_profile": "small",
            "dataset_dir": USER_SMALL_DATASET_DIR,
            "output_dir": USER_SMALL_OUTPUT_DIR,
            "val_size": USER_SMALL_VAL_SIZE,
        }
    raise ValueError(f"Unsupported USER_DATASET_PROFILE: {USER_DATASET_PROFILE}")


PROFILE_DEFAULTS = resolve_profile_defaults()


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_parser():
    parser = argparse.ArgumentParser(description="Train SPSA v13")
    parser.add_argument("--dataset_profile", type=str, default=PROFILE_DEFAULTS["dataset_profile"])
    parser.add_argument("--dataset_dir", type=Path, default=PROFILE_DEFAULTS["dataset_dir"])
    parser.add_argument("--output_dir", type=Path, default=PROFILE_DEFAULTS["output_dir"])
    parser.add_argument("--val_size", type=int, default=PROFILE_DEFAULTS["val_size"])
    parser.add_argument("--device", type=str, default=USER_DEVICE)
    parser.add_argument("--batch_size", type=int, default=USER_BATCH_SIZE)
    parser.add_argument("--effective_batch_size", type=int, default=USER_EFFECTIVE_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=USER_NUM_WORKERS)
    parser.add_argument("--pin_memory", type=str2bool, default=USER_PIN_MEMORY)
    parser.add_argument("--epochs", type=int, default=USER_EPOCHS)
    parser.add_argument("--lr", type=float, default=USER_LR)
    parser.add_argument("--weight_decay", type=float, default=USER_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=USER_RANDOM_SEED)
    parser.add_argument("--save_checkpoint_every", type=int, default=USER_SAVE_CHECKPOINT_EVERY)
    parser.add_argument("--log_image_count", type=int, default=USER_LOG_IMAGE_COUNT)
    parser.add_argument("--visual_val_pair_count", type=int, default=USER_VISUAL_VAL_PAIR_COUNT)
    parser.add_argument("--run_initial_val", type=str2bool, default=USER_RUN_INITIAL_VAL)
    parser.add_argument("--use_wandb", type=str2bool, default=USER_USE_WANDB)
    parser.add_argument("--wandb_run_name", type=str, default=USER_WANDB_RUN_NAME)
    parser.add_argument("--wandb_project", type=str, default=USER_WANDB_PROJECT)
    parser.add_argument("--stylegan_ckpt", type=str, default=USER_STYLEGAN_CKPT)
    parser.add_argument("--rotate_ckpt", type=str, default=USER_ROTATE_CKPT)
    parser.add_argument("--blending_ckpt", type=str, default=USER_BLENDING_CKPT)
    parser.add_argument("--pp_ckpt", type=str, default=USER_PP_CKPT)
    parser.add_argument("--spsa_init_ckpt", type=str, default=USER_SPSA_INIT_CKPT)
    parser.add_argument("--resume_checkpoint", type=str, default=USER_RESUME_CHECKPOINT)
    parser.add_argument("--spsa_hidden_channels", type=int, default=USER_SPSA_HIDDEN_CHANNELS_V13)
    parser.add_argument("--spsa_bang_roi_size", type=int, default=USER_SPSA_BANG_ROI_SIZE_V13)
    parser.add_argument("--spsa_tail_roi_size", type=int, default=USER_SPSA_TAIL_ROI_SIZE_V13)
    parser.add_argument("--spsa_alpha_init", type=float, default=USER_SPSA_ALPHA_INIT_V13)
    parser.add_argument("--spsa_beta_bang_init", type=float, default=USER_SPSA_BETA_BANG_INIT_V13)
    parser.add_argument("--spsa_beta_tail_init", type=float, default=USER_SPSA_BETA_TAIL_INIT_V13)
    parser.add_argument("--lambda_mask", type=float, default=USER_LAMBDA_MASK_V13)
    parser.add_argument("--lambda_edge", type=float, default=USER_LAMBDA_EDGE_V13)
    parser.add_argument("--lambda_ori", type=float, default=USER_LAMBDA_ORI_V13)
    parser.add_argument("--lambda_roi", type=float, default=USER_LAMBDA_ROI_V13)
    parser.add_argument("--lambda_preserve", type=float, default=USER_LAMBDA_PRESERVE_V13)
    parser.add_argument("--lambda_anchor", type=float, default=USER_LAMBDA_ANCHOR_V13)
    parser.add_argument("--alignment_stage_weight", type=float, default=USER_ALIGNMENT_STAGE_WEIGHT_V13)
    return parser


class NullLoggerV13:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.train_step = 0

    def start_logging(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def next_step(self):
        self.train_step += 1

    def log_scalars(self, scalars):
        return scalars

    def save(self, file_path, save_online=False):
        target = self.output_dir / Path(file_path).name
        if Path(file_path).resolve() != target.resolve():
            shutil.copy2(file_path, target)


class SPSAStructuralLossV13(nn.Module):
    def __init__(self, args, device: str):
        super().__init__()
        self.args = args
        self.device = torch.device(device)
        self.seg = BiSeNet(n_classes=16).to(self.device).eval()
        self.seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth", map_location=self.device))
        toggle_grad(self.seg, False)
        self.seg_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.seg_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=self.device,
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)
        self.to(self.device)

    def _resize(self, tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tuple(tensor.shape[-2:]) == tuple(size):
            return tensor.float()
        return F.interpolate(tensor.float(), size=size, mode="bilinear", align_corners=False)

    def _hair_prob(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image_512 = F.interpolate(image, size=(512, 512), mode="bilinear", align_corners=False)
        logits, _, _ = self.seg((image_512 - self.seg_mean) / self.seg_std)
        return torch.softmax(logits, dim=1)[:, 10:11]

    def _boundary(self, probability: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(probability, self.sobel_x, padding=1)
        grad_y = F.conv2d(probability, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        return magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + 1e-6)

    def _orientation(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        jxx = F.avg_pool2d(grad_x * grad_x, kernel_size=5, stride=1, padding=2)
        jyy = F.avg_pool2d(grad_y * grad_y, kernel_size=5, stride=1, padding=2)
        jxy = F.avg_pool2d(grad_x * grad_y, kernel_size=5, stride=1, padding=2)
        angle = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy + 1e-6)
        return torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = 2.0 * (pred * target).sum(dim=(2, 3))
        denominator = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6
        return 1.0 - (numerator + 1e-6) / denominator

    def _masked_mean(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return (value * mask).sum() / denom

    def _orientation_loss(self, pred_ori: torch.Tensor, target_ori: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cosine = (pred_ori * target_ori).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        return self._masked_mean(1.0 - cosine, mask)

    def forward(
        self,
        *,
        source: torch.Tensor,
        prediction: torch.Tensor,
        priors: dict[str, torch.Tensor],
        latent_f_align: torch.Tensor,
        latent_f_base: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(0)
        if source.ndim == 3:
            source = source.unsqueeze(0)

        hair_prob = self._hair_prob(prediction)
        loss_hw = tuple(hair_prob.shape[-2:])
        prediction_resized = self._resize(prediction, loss_hw)
        target_mask = self._resize(priors["H_tgt"].to(self.device), loss_hw)
        target_edge = self._resize(priors["E_tgt"].to(self.device), loss_hw)
        target_ori = self._resize(priors["O_tgt"].to(self.device), loss_hw)
        bang_mask = self._resize(priors["bang_mask"].to(self.device), loss_hw)
        tail_mask = self._resize(priors["tail_mask"].to(self.device), loss_hw)
        pred_edge = self._boundary(hair_prob)
        pred_ori = self._orientation(prediction_resized)
        source_resized = self._resize(source, loss_hw)

        mask_loss = F.binary_cross_entropy(hair_prob, target_mask) + self._dice_loss(hair_prob, target_mask).mean()
        edge_loss = F.l1_loss(pred_edge, target_edge)
        ori_loss = self._orientation_loss(pred_ori, target_ori, target_mask)

        bang_roi = bang_mask.clamp(0, 1)
        tail_roi = tail_mask.clamp(0, 1)
        bang_loss = (
            self._masked_mean(F.binary_cross_entropy(hair_prob, target_mask, reduction="none"), bang_roi)
            + self._masked_mean(torch.abs(pred_edge - target_edge), bang_roi)
            + self._orientation_loss(pred_ori, target_ori, bang_roi)
        )
        tail_loss = (
            self._masked_mean(F.binary_cross_entropy(hair_prob, target_mask, reduction="none"), tail_roi)
            + self._masked_mean(torch.abs(pred_edge - target_edge), tail_roi)
            + self._orientation_loss(pred_ori, target_ori, tail_roi)
        )
        roi_loss = bang_loss + tail_loss

        preserve_mask = (1.0 - target_mask).clamp(0, 1)
        preserve_loss = self._masked_mean(
            torch.abs(prediction_resized - source_resized).mean(dim=1, keepdim=True),
            preserve_mask,
        )
        anchor_loss = F.l1_loss(latent_f_align, latent_f_base)

        total = (
            self.args.lambda_mask * mask_loss
            + self.args.lambda_edge * edge_loss
            + self.args.lambda_ori * ori_loss
            + self.args.lambda_roi * roi_loss
            + self.args.lambda_preserve * preserve_loss
            + self.args.lambda_anchor * anchor_loss
        )
        losses = {
            "mask": mask_loss,
            "edge": edge_loss,
            "ori": ori_loss,
            "roi": roi_loss,
            "preserve": preserve_loss,
            "anchor": anchor_loss,
            "loss": total,
        }
        aux = {
            "hair_prob": hair_prob.detach(),
            "pred_edge": pred_edge.detach(),
        }
        return losses, aux


class SPSATrainingPipelineV13(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_args = get_parser_v13().parse_args([])
        model_args.device = args.device
        model_args.ckpt = args.stylegan_ckpt
        model_args.rotate_checkpoint = args.rotate_ckpt
        model_args.blending_checkpoint = args.blending_ckpt
        model_args.pp_checkpoint = args.pp_ckpt
        model_args.save_all = False
        model_args.use_shadow_cleanup = False
        model_args.spsa_hidden_channels = args.spsa_hidden_channels
        model_args.spsa_bang_roi_size = args.spsa_bang_roi_size
        model_args.spsa_tail_roi_size = args.spsa_tail_roi_size
        model_args.spsa_alpha_init = args.spsa_alpha_init
        model_args.spsa_beta_bang_init = args.spsa_beta_bang_init
        model_args.spsa_beta_tail_init = args.spsa_beta_tail_init
        model_args.spsa_feature_channels = 512
        self.hair_fast = HairFast_v13(model_args)
        self.device = torch.device(args.device)
        self.to(self.device)
        self._freeze_everything_except_spsa()
        self._set_hairfast_eval()
        self.hair_fast.align.spsa.train()
        if args.spsa_init_ckpt:
            self.hair_fast.align.load_spsa_checkpoint(args.spsa_init_ckpt)

    def _hairfast_modules(self):
        return [
            self.hair_fast.net,
            self.hair_fast.net.generator,
            self.hair_fast.embed,
            self.hair_fast.embed.encoder,
            self.hair_fast.embed.e4e,
            self.hair_fast.align,
            self.hair_fast.align.sean_model,
            self.hair_fast.align.mask_generator,
            self.hair_fast.align.rotate_model,
            self.hair_fast.align.spsa,
            self.hair_fast.blend,
            self.hair_fast.blend.blending_encoder,
            self.hair_fast.blend.post_process,
            self.hair_fast.shadow_cleanup,
        ]

    def _set_hairfast_eval(self):
        for module in self._hairfast_modules():
            if module is not None and hasattr(module, "eval"):
                module.eval()

    def _freeze_everything_except_spsa(self):
        modules_to_freeze = [
            self.hair_fast.net.generator,
            self.hair_fast.embed.encoder,
            self.hair_fast.embed.e4e,
            self.hair_fast.align.sean_model,
            self.hair_fast.align.mask_generator,
            self.hair_fast.align.rotate_model,
            self.hair_fast.blend.blending_encoder,
            self.hair_fast.blend.post_process,
            self.hair_fast.shadow_cleanup,
        ]
        for module in modules_to_freeze:
            if module is not None:
                toggle_grad(module, False)
        toggle_grad(self.hair_fast.align.spsa, True)

    def trainable_parameters(self):
        return self.hair_fast.align.spsa.parameters()

    def train(self, mode: bool = True):
        # Keep the frozen HairFast backbone in eval mode permanently.
        # Only the SPSA branch is allowed to switch to train mode.
        nn.Module.train(self, False)
        self._set_hairfast_eval()
        self.hair_fast.align.spsa.train(mode)
        self.training = mode
        return self

    def eval(self):
        nn.Module.train(self, False)
        self._set_hairfast_eval()
        self.hair_fast.align.spsa.eval()
        return self

    def _embed_triplet(self, source: torch.Tensor, reference: torch.Tensor, color: torch.Tensor):
        source, reference, color = equal_replacer([source, reference, color])
        images_to_name = defaultdict(list)
        for image, name in zip((source, reference, color), ("face", "shape", "color")):
            images_to_name[image].append(name)
        with torch.no_grad():
            return self.hair_fast.embed.embedding_images(images_to_name)

    def _render_final_image(self, align_shape, align_color, name_to_embed):
        blend = self.hair_fast.blend
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_2 = name_to_embed["shape"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        mask_de = blend.dilate_erosion.hair_from_mask(
            torch.cat([name_to_embed[x]["mask"] for x in ["face", "color"]], dim=0)
        )
        HM_1D, _ = mask_de[0][0].unsqueeze(0), mask_de[1][0].unsqueeze(0)
        HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)

        latent_S_1 = name_to_embed["face"]["S"]
        latent_F_align = align_shape["latent_F_align"]
        HM_X = align_color["HM_X"]
        latent_S_3 = name_to_embed["color"]["S"]

        HM_XD, _ = blend.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        if I_1 is not I_3 or I_1 is not I_2:
            S_blend_6_18 = blend.blending_encoder(latent_S_1[:, 6:], latent_S_3[:, 6:], I_1 * target_mask, I_3 * HM_3E)
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        I_blend, _ = self.hair_fast.net.generator(
            [S_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_align,
        )
        I_blend_256 = blend.downsample_256(I_blend)
        S_final, F_final = blend.post_process(I_1, I_blend_256)
        I_final, _ = self.hair_fast.net.generator(
            [S_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=F_final,
        )
        return ((I_final + 1) / 2).clamp(0, 1)

    def _render_alignment_image(self, align_shape):
        align_prediction = self.hair_fast.align.decode_alignment_image(
            align_shape["latent_S_face"],
            align_shape["latent_F_align"],
        )
        return self.hair_fast.blend.downsample_256(align_prediction)

    def forward_sample(self, sample):
        source = sample["source"].to(self.device)
        reference = sample["reference"].to(self.device)
        color = sample["color"].to(self.device)
        priors = {
            key: sample[key].to(self.device)
            for key in ["H_tgt", "E_tgt", "D_tgt", "O_tgt", "bang_box", "tail_box", "bang_mask", "tail_mask"]
        }

        name_to_embed = self._embed_triplet(source, reference, color)
        align_shape = self.hair_fast.align.forward_train("face", "shape", name_to_embed, spsa_prior=priors)
        align_color = align_shape
        align_prediction = self._render_alignment_image(align_shape)
        prediction = self._render_final_image(align_shape, align_color, name_to_embed)
        return {
            "align_prediction": align_prediction,
            "prediction": prediction,
            "source": source,
            "reference": reference,
            "priors": priors,
            "latent_F_align": align_shape["latent_F_align"],
            "latent_F_base": align_shape["latent_F_base"],
        }


class FixedPairEvaluatorV13:
    def __init__(self, args, pipeline: SPSATrainingPipelineV13, records: list[dict[str, str]]):
        self.args = args
        self.pipeline = pipeline
        self.records = list(records)

    @torch.inference_mode()
    def run(self, epoch_tag: str):
        if not self.records:
            return
        output_dir = self.args.output_dir / "fixed_pairs" / epoch_tag
        output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline.eval()
        for record in tqdm(self.records, desc=f"Fixed pairs {epoch_tag}", leave=False):
            source_path = Path(record["source_path"])
            reference_path = Path(record["reference_path"])
            spsa = self.pipeline.hair_fast.swap(
                source_path,
                reference_path,
                reference_path,
                spsa_prior_path=record["prior_path"],
                use_shadow_cleanup=False,
            )
            prior = load_prior_npz(record["prior_path"])
            panels = make_fixed_pair_panels(
                source=read_image_tensor(source_path),
                reference=read_image_tensor(reference_path),
                spsa=spsa,
                bang_box=prior["bang_box"],
                tail_box=prior["tail_box"],
            )
            save_fixed_pair_panels(output_dir / record["sample_id"], panels)


def collate_as_list(batch):
    return batch


def resolve_grad_accum_steps(args) -> int:
    return max(1, int(args.effective_batch_size) // max(1, int(args.batch_size)))


def save_checkpoint(args, pipeline, optimizer, epoch, best_val_loss, name):
    checkpoint_path = args.output_dir / f"{name}.pth"
    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "spsa_state_dict": pipeline.hair_fast.align.spsa.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path, pipeline, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    pipeline.hair_fast.align.spsa.load_state_dict(checkpoint["spsa_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def compute_alignment_stage_loss(args, align_losses: dict[str, torch.Tensor]) -> torch.Tensor:
    structure_only = (
        args.lambda_mask * align_losses["mask"]
        + args.lambda_edge * align_losses["edge"]
        + args.lambda_ori * align_losses["ori"]
        + args.lambda_roi * align_losses["roi"]
    )
    return args.alignment_stage_weight * structure_only


def compute_sample_losses(args, outputs, loss_helper):
    final_losses, _ = loss_helper(
        source=outputs["source"],
        prediction=outputs["prediction"],
        priors=outputs["priors"],
        latent_f_align=outputs["latent_F_align"],
        latent_f_base=outputs["latent_F_base"],
    )
    align_losses, _ = loss_helper(
        source=outputs["source"],
        prediction=outputs["align_prediction"],
        priors=outputs["priors"],
        latent_f_align=outputs["latent_F_align"],
        latent_f_base=outputs["latent_F_base"],
    )
    align_stage_loss = compute_alignment_stage_loss(args, align_losses)
    total_loss = final_losses["loss"] + align_stage_loss
    logs = {
        "loss": total_loss,
        "align_stage": align_stage_loss,
        **{f"final_{key}": value for key, value in final_losses.items()},
        **{f"align_{key}": value for key, value in align_losses.items()},
    }
    return total_loss, logs


def validate(args, pipeline, dataloader, loss_helper, logger, epoch_tag):
    pipeline.eval()
    total_losses: dict[str, float] = {}
    preview_rows = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Val {epoch_tag}", leave=False):
            batch_loss = {}
            for sample in batch:
                outputs = pipeline.forward_sample(sample)
                _, losses = compute_sample_losses(args, outputs, loss_helper)
                for key, value in losses.items():
                    batch_loss[key] = batch_loss.get(key, 0.0) + float(value.item())
                if len(preview_rows) < args.log_image_count:
                    preview_rows.append(
                        [
                            outputs["source"].cpu(),
                            outputs["reference"].cpu(),
                            outputs["prediction"][0].cpu(),
                        ]
                    )
            for key, value in batch_loss.items():
                total_losses[key] = total_losses.get(key, 0.0) + value / max(1, len(batch))

    denom = max(1, len(dataloader))
    log_payload = {f"val {key}": value / denom for key, value in total_losses.items()}
    logger.log_scalars(log_payload)

    if preview_rows:
        preview_dir = args.output_dir / "val_images" / epoch_tag
        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(preview_rows):
            image = image_grid(list(map(T.functional.to_pil_image, row)), 1, len(row))
            image.save(preview_dir / f"val_{idx:03d}.png")

    return log_payload.get("val loss", float("inf"))


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES
    if USER_DISABLE_CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = False

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.dataset_dir / "manifest.jsonl"
    dataset = SPSAAlignmentDataset(manifest_path)
    train_records, val_records = split_manifest_records(dataset.records, val_size=args.val_size, seed=args.seed)
    visual_val_pair_count = max(1, min(int(args.visual_val_pair_count), len(val_records)))
    visual_val_records = list(val_records[:visual_val_pair_count])
    write_manifest(visual_val_records, args.output_dir / "val_pairs_v13.jsonl")

    train_dataset = SPSAAlignmentDataset(manifest_path)
    val_dataset = SPSAAlignmentDataset(manifest_path)
    train_dataset.records = train_records
    val_dataset.records = val_records

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        collate_fn=collate_as_list,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        collate_fn=collate_as_list,
    )

    logger = WandbLogger(name=args.wandb_run_name, project=args.wandb_project) if args.use_wandb else NullLoggerV13(args.output_dir)
    logger.start_logging()

    pipeline = SPSATrainingPipelineV13(args)
    optimizer = torch.optim.Adam(pipeline.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_helper = SPSAStructuralLossV13(args, device=args.device)
    fixed_pair_evaluator = FixedPairEvaluatorV13(args, pipeline, visual_val_records)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint, pipeline, optimizer=optimizer)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    grad_accum_steps = resolve_grad_accum_steps(args)
    if args.run_initial_val:
        initial_val = validate(args, pipeline, val_loader, loss_helper, logger, "epoch_0000")
        best_val_loss = min(best_val_loss, initial_val)

    for epoch in range(start_epoch, args.epochs):
        pipeline.train(True)
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        step_in_group = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch + 1}", leave=False)):
            losses_group = {}
            total_loss = 0.0
            for sample in batch:
                outputs = pipeline.forward_sample(sample)
                sample_loss, losses = compute_sample_losses(args, outputs, loss_helper)
                total_loss = total_loss + sample_loss / max(1, len(batch))
                for key, value in losses.items():
                    losses_group[key] = losses_group.get(key, 0.0) + float(value.item()) / max(1, len(batch))

            (total_loss / grad_accum_steps).backward()
            step_in_group += 1

            for key, value in losses_group.items():
                running[key] = running.get(key, 0.0) + value

            should_step = step_in_group >= grad_accum_steps or batch_idx == len(train_loader) - 1
            if not should_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(list(pipeline.trainable_parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            logger.next_step()
            logger.log_scalars(
                {
                    **{f"train {key}": value / step_in_group for key, value in running.items()},
                    "train grad": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                }
            )
            running = {}
            step_in_group = 0

        epoch_tag = f"epoch_{epoch + 1:04d}"
        val_loss = validate(args, pipeline, val_loader, loss_helper, logger, epoch_tag)
        saved_checkpoint = False
        if (epoch + 1) % max(1, args.save_checkpoint_every) == 0:
            save_checkpoint(args, pipeline, optimizer, epoch, best_val_loss, "last")
            saved_checkpoint = True
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args, pipeline, optimizer, epoch, best_val_loss, "best")
            saved_checkpoint = True
        if saved_checkpoint:
            fixed_pair_evaluator.run(epoch_tag)


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
