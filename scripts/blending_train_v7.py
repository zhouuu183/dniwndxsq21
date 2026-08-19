import os
import random
import sys
from argparse import Namespace
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

from models.Blending_v7 import Blending_v7
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.train import seed_everything, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/blending_dataset_v7")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_train_v7")
USER_VAL_SIZE_FFHQ = 512

USER_DATASET_DIR_SMALL = Path("images/blending_dataset_v7_small")
USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_OUTPUT_DIR_SMALL = Path("output/blending_train_v7_small")
USER_VAL_SIZE_SMALL = 35

USER_RANDOM_SEED = 3407
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"

USER_BATCH_SIZE = 8
USER_NUM_WORKERS = 0
USER_EPOCHS = 20
USER_LR = 2e-5
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 1.0
USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_VAL_IMAGES_EVERY = 1
USER_LOG_IMAGE_COUNT = 20

USER_LAMBDA_FACE_CLIP = 1.0
USER_LAMBDA_HAIR_CLIP = 1.0
USER_LAMBDA_FRINGE_COLOR = 0.5
USER_LAMBDA_FRINGE_EDGE = 0.1
USER_LAMBDA_PRESERVE = 1.0
USER_LAMBDA_RESIDUAL_REG = 0.05
USER_LAMBDA_ALPHA_REG = 0.01
USER_LAMBDA_FUSION_REG = 0.01
USER_LAMBDA_TV_REG = 0.05
# ========================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        },
        "small": {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
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


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find image for {stem} under {root}")


def to_tensor_256(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = T.functional.resize(
            image,
            [256, 256],
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )
        return T.functional.normalize(T.functional.to_tensor(image), [0.5], [0.5])


def gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device)
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    gx = F.conv2d(gray(image), kernel_x, padding=1)
    gy = F.conv2d(gray(image), kernel_y, padding=1)
    return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)


def total_variation(image: torch.Tensor) -> torch.Tensor:
    diff_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    diff_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    return diff_y.abs().mean() + diff_x.abs().mean()


class BlendingDatasetV7(Dataset):
    def __init__(self, exps: list[list[str]], dataset_dir: Path, face_root: Path, shape_root: Path, color_root: Path):
        self.items = []
        self.dataset_dir = dataset_dir
        self.face_root = face_root
        self.shape_root = shape_root
        self.color_root = color_root

        for face_stem, shape_stem, color_stem in exps:
            item = self._prepare_item(face_stem, shape_stem, color_stem)
            if item is not None:
                self.items.append(item)

    def _prepare_item(self, face_stem: str, shape_stem: str, color_stem: str):
        try:
            color_s = torch.from_numpy(np.load(self.dataset_dir / "FS" / f"{color_stem}.npz")["latent_in"]).squeeze(0).float()
            face_s = torch.from_numpy(np.load(self.dataset_dir / "FS" / f"{face_stem}.npz")["latent_in"]).squeeze(0).float()

            shape_align_npz = np.load(self.dataset_dir / "Align" / f"{face_stem}_{shape_stem}.npz")
            color_align_npz = np.load(self.dataset_dir / "Align" / f"{face_stem}_{color_stem}.npz")

            align_f = torch.from_numpy(shape_align_npz["latent_F"]).squeeze(0).float()
            target_hair_mask = torch.from_numpy(color_align_npz["target_hair_mask"]).squeeze(0).float()

            face_i = to_tensor_256(resolve_image_path(self.face_root, face_stem))
            shape_i = to_tensor_256(resolve_image_path(self.shape_root, shape_stem))
            color_i = to_tensor_256(resolve_image_path(self.color_root, color_stem))
            return {
                "face_stem": face_stem,
                "shape_stem": shape_stem,
                "color_stem": color_stem,
                "color_s": color_s,
                "face_s": face_s,
                "align_f": align_f,
                "target_hair_mask": target_hair_mask,
                "face_i": face_i,
                "shape_i": shape_i,
                "color_i": color_i,
            }
        except Exception as exc:
            print(exc, file=sys.stderr)
            return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TrainerV7:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader):
        self.device = USER_DEVICE if torch.cuda.is_available() else "cpu"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = ACTIVE_OUTPUT_DIR
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.preview_dir = self.output_dir / "val_images"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        net_opts = Namespace(
            size=1024,
            ckpt=USER_STYLEGAN_CKPT,
            channel_multiplier=2,
            latent=512,
            n_mlp=8,
            device=self.device,
        )
        self.net = Net(net_opts)

        blend_opts = Namespace(
            size=1024,
            ckpt=USER_STYLEGAN_CKPT,
            channel_multiplier=2,
            latent=512,
            n_mlp=8,
            device=self.device,
            smooth=5,
            blending_checkpoint=USER_BLENDING_CKPT,
            pp_checkpoint=USER_PP_CKPT,
            save_all=False,
            fringe_crop_size=128,
            fringe_crop_padding=16,
            fringe_mask_channels=32,
            fringe_mask_prior_weight=0.8,
            fringe_boundary_kernel=9,
            fringe_roi_kernel=25,
            fringe_encoder_channels=32,
            fringe_attention_channels=128,
            fringe_decoder_channels=64,
        )
        self.model = Blending_v7(blend_opts, net=self.net).to(self.device)
        toggle_grad(self.model.blending_encoder, False)
        toggle_grad(self.model.post_process, False)
        toggle_grad(self.net.generator, False)
        self.model.blending_encoder.eval()
        self.model.post_process.eval()
        self.net.generator.eval()
        self.model.fringe_refiner.train()

        self.optimizer = torch.optim.Adam(
            self.model.fringe_refiner.parameters(),
            lr=USER_LR,
            weight_decay=USER_WEIGHT_DECAY,
        )
        self.dilate_erosion = DilateErosion(device=self.device)
        self.downsample_512 = BicubicDownSample(factor=2, cuda="cuda" in str(self.device))
        self.downsample_256 = BicubicDownSample(factor=4, cuda="cuda" in str(self.device))

        self.seg = BiSeNet(n_classes=16).to(self.device).eval()
        self.seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth", map_location=self.device))
        toggle_grad(self.seg, False)

        self.best_loss = float("inf")

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        image = ((image + 1) / 2).clamp(0, 1)
        return T.functional.to_pil_image(image.cpu())

    def _build_preview_strip(
        self,
        face_image: torch.Tensor,
        provider_image: torch.Tensor,
        baseline_image: torch.Tensor,
        v7_image: torch.Tensor,
    ) -> Image.Image:
        panels = [
            self._tensor_to_pil(face_image),
            self._tensor_to_pil(provider_image),
            self._tensor_to_pil(baseline_image),
            self._tensor_to_pil(v7_image),
        ]

        width, height = panels[0].size
        canvas = Image.new("RGB", (width * len(panels), height), color=(255, 255, 255))
        for idx, panel in enumerate(panels):
            canvas.paste(panel, (idx * width, 0))
        return canvas

    @staticmethod
    def _select_provider_image(
        shape_image: torch.Tensor,
        color_image: torch.Tensor,
        shape_stem: str,
        color_stem: str,
    ) -> torch.Tensor:
        # Validation preview is fixed to a 1x4 strip. When shape/color come from
        # different people, we keep the shape provider as the single reference panel.
        if shape_stem == color_stem:
            return shape_image
        return shape_image

    @torch.no_grad()
    def parsing_mask(self, image: torch.Tensor) -> torch.Tensor:
        image_512 = (self.downsample_512((image + 1) / 2) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(image_512)
        current_mask = torch.argmax(down_seg, dim=1).long().float()
        return F.interpolate(current_mask.unsqueeze(1), size=(256, 256), mode="nearest")

    def _forward_batch(self, batch: dict[str, torch.Tensor]):
        face_i = batch["face_i"].to(self.device)
        shape_i = batch["shape_i"].to(self.device)
        color_i = batch["color_i"].to(self.device)
        color_s = batch["color_s"].to(self.device)
        face_s = batch["face_s"].to(self.device)
        align_f = batch["align_f"].to(self.device)
        target_hair_mask = batch["target_hair_mask"].to(self.device)
        if target_hair_mask.dim() == 3:
            target_hair_mask = target_hair_mask.unsqueeze(1)

        with torch.no_grad():
            face_parsing = self.parsing_mask(face_i)
            color_parsing = self.parsing_mask(color_i)
            mask_de = self.dilate_erosion.hair_from_mask(torch.cat([face_parsing, color_parsing], dim=0))
            batch_size = face_i.size(0)
            hm_1d = mask_de[0][:batch_size]
            hm_3d = mask_de[0][batch_size:]
            hm_3e = mask_de[1][batch_size:]
            hm_xd, _ = self.dilate_erosion.mask(target_hair_mask)
            target_mask = (1 - hm_1d) * (1 - hm_3d) * (1 - hm_xd)

            blend_s_6_18 = self.model.blending_encoder(
                face_s[:, 6:],
                color_s[:, 6:],
                face_i * target_mask,
                color_i * hm_3e,
            )
            blend_s = torch.cat((face_s[:, :6], blend_s_6_18), dim=1)
            i_blend, _ = self.net.generator(
                [blend_s],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=align_f,
            )
            i_blend_256 = self.downsample_256(i_blend)
            s_final, f_final = self.model.post_process(face_i, i_blend_256)
            baseline_image, _ = self.net.generator(
                [s_final],
                input_is_latent=True,
                return_latents=False,
                start_layer=5,
                end_layer=8,
                layer_in=f_final,
            )
            baseline_256 = self.downsample_256(baseline_image)

        fringe_outputs = self.model.fringe_refiner(
            source_mask=face_parsing,
            target_hair_mask=target_hair_mask,
            source_image=face_i,
            reference_image=0.5 * (shape_i + color_i),
            blending_image=i_blend_256,
            global_style=s_final,
            global_feature=f_final,
            generator=self.net.generator,
        )
        i_final = fringe_outputs["final_image"]
        i_final_256 = self.downsample_256(i_final)
        return {
            "i_final": i_final,
            "i_final_256": i_final_256,
            "baseline_256": baseline_256,
            "face_i": face_i,
            "shape_i": shape_i,
            "color_i": color_i,
            "target_mask": target_mask,
            "hm_3e": hm_3e,
            "fringe_mask": fringe_outputs["fringe_mask_256"],
            "boundary_mask": fringe_outputs["boundary_mask"],
            "residual_canvas_256": fringe_outputs["residual_canvas_256"],
            "alpha_canvas_256": fringe_outputs["alpha_canvas_256"],
            "fused_feature": fringe_outputs["fused_feature"],
            "global_feature": f_final,
        }

    def calc_loss(self, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        i_final_256 = outputs["i_final_256"]
        face_i = outputs["face_i"]
        shape_i = outputs["shape_i"]
        color_i = outputs["color_i"]
        target_mask = outputs["target_mask"]
        hm_3e = outputs["hm_3e"]
        fringe_mask = outputs["fringe_mask"]
        boundary_mask = outputs["boundary_mask"]
        baseline_256 = outputs["baseline_256"]
        residual_canvas_256 = outputs["residual_canvas_256"]
        alpha_canvas_256 = outputs["alpha_canvas_256"]
        fused_feature = outputs["fused_feature"]
        global_feature = outputs["global_feature"]

        face_embed_gen = self.model.blending_encoder.get_image_embed(i_final_256 * target_mask)
        face_embed_gt = self.model.blending_encoder.get_image_embed(face_i * target_mask)
        face_loss = (1 - F.cosine_similarity(face_embed_gen, face_embed_gt)).mean()

        hair_embed_gen = self.model.blending_encoder.get_image_embed(i_final_256 * hm_3e)
        hair_embed_gt = self.model.blending_encoder.get_image_embed(color_i * hm_3e)
        hair_loss = (1 - F.cosine_similarity(hair_embed_gen, hair_embed_gt)).mean()

        fringe_support = torch.maximum(fringe_mask, boundary_mask)
        fringe_color = F.l1_loss(i_final_256 * fringe_support, color_i * fringe_support)
        fringe_edge = F.l1_loss(sobel_edges(i_final_256) * fringe_support, sobel_edges(shape_i) * fringe_support)
        preserve = F.l1_loss(i_final_256 * (1 - fringe_support), baseline_256.detach() * (1 - fringe_support))
        residual_reg = (residual_canvas_256 * fringe_support).abs().mean()
        alpha_reg = (alpha_canvas_256 * fringe_support).mean()
        fusion_reg = (fused_feature - global_feature.detach()).abs().mean()
        tv_reg = total_variation(residual_canvas_256 * fringe_support)

        losses = {
            "face_clip": USER_LAMBDA_FACE_CLIP * face_loss,
            "hair_clip": USER_LAMBDA_HAIR_CLIP * hair_loss,
            "fringe_color": USER_LAMBDA_FRINGE_COLOR * fringe_color,
            "fringe_edge": USER_LAMBDA_FRINGE_EDGE * fringe_edge,
            "preserve": USER_LAMBDA_PRESERVE * preserve,
            "residual_reg": USER_LAMBDA_RESIDUAL_REG * residual_reg,
            "alpha_reg": USER_LAMBDA_ALPHA_REG * alpha_reg,
            "fusion_reg": USER_LAMBDA_FUSION_REG * fusion_reg,
            "tv_reg": USER_LAMBDA_TV_REG * tv_reg,
        }
        losses["loss"] = sum(losses.values())
        return losses["loss"], losses

    def save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "fringe_refiner_state_dict": self.model.fringe_refiner.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, self.ckpt_dir / "last.pth")
        if epoch % USER_SAVE_CHECKPOINT_EVERY == 0:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")

    @torch.no_grad()
    def save_val_images(self, epoch: int):
        if epoch % USER_SAVE_VAL_IMAGES_EVERY != 0:
            return

        self.model.fringe_refiner.eval()
        epoch_dir = self.preview_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for batch in self.val_loader:
            outputs = self._forward_batch(batch)
            for idx in range(outputs["i_final_256"].shape[0]):
                if saved >= USER_LOG_IMAGE_COUNT:
                    self.model.fringe_refiner.train()
                    return
                face_stem = batch["face_stem"][idx]
                shape_stem = batch["shape_stem"][idx]
                color_stem = batch["color_stem"][idx]
                provider_image = self._select_provider_image(
                    shape_image=outputs["shape_i"][idx],
                    color_image=outputs["color_i"][idx],
                    shape_stem=shape_stem,
                    color_stem=color_stem,
                )

                strip = self._build_preview_strip(
                    face_image=outputs["face_i"][idx],
                    provider_image=provider_image,
                    baseline_image=outputs["baseline_256"][idx],
                    v7_image=outputs["i_final_256"][idx],
                )
                preview_name = f"{saved:03d}_{face_stem}__{shape_stem}__{color_stem}__1x4.png"
                strip.save(epoch_dir / preview_name)
                saved += 1
        self.model.fringe_refiner.train()

    def run_epoch(self, train: bool) -> float:
        loader = self.train_loader if train else self.val_loader
        self.model.fringe_refiner.train(mode=train)

        total = 0.0
        count = 0
        iterator = tqdm(loader, desc="train" if train else "val")
        for batch in iterator:
            outputs = self._forward_batch(batch)
            loss, losses = self.calc_loss(outputs)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.fringe_refiner.parameters(), USER_GRAD_CLIP)
                self.optimizer.step()

            total += loss.item()
            count += 1
            iterator.set_postfix({key: f"{val.item():.4f}" for key, val in losses.items() if key != "loss"})

        return total / max(count, 1)

    def train(self):
        for epoch in range(1, USER_EPOCHS + 1):
            train_loss = self.run_epoch(train=True)
            val_loss = self.run_epoch(train=False)
            is_best = val_loss <= self.best_loss
            if is_best:
                self.best_loss = val_loss
            self.save_checkpoint(epoch, is_best)
            self.save_val_images(epoch)
            print(f"[epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} best={self.best_loss:.6f}")


def main():
    seed_everything(USER_RANDOM_SEED)
    set_seed(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exps = []
    with open(ACTIVE_DATASET_DIR / "dataset.exps", "r", encoding="utf-8") as file:
        for exp in file.readlines():
            exps.append(list(map(lambda x: x.replace(".png", ""), exp.split())))

    if len(exps) <= ACTIVE_VAL_SIZE:
        raise RuntimeError("Validation split is larger than the dataset. Reduce USER_*_VAL_SIZE.")

    train_exps, val_exps = train_test_split(exps, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = BlendingDatasetV7(train_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT, ACTIVE_COLOR_ROOT)
    val_dataset = BlendingDatasetV7(val_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT, ACTIVE_COLOR_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    trainer = TrainerV7(train_loader, val_loader)
    trainer.train()


if __name__ == "__main__":
    main()
