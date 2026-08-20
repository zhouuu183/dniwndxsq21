# -*- coding: utf-8 -*-
import gc
import inspect
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from datasets.nhr_dataset_v14 import (
    NHRPairDatasetV14,
    NHRSourceDatasetV14,
    build_synthetic_pretrain_batch_v14,
    collate_dict,
    load_rgb,
    read_jsonl,
)
from hair_swap_v14 import HairFastV14, get_parser_v14
from losses.nhr_losses_v14 import NHRLossBuilderV14
from models.RepairNet_v14 import RepairNetV14
from utils.nhr_utils_v14 import dilate


# ========================= User Config: edit here only =========================
USER_STAGE = "joint_alignment"  # "repair_pretrain" or "joint_alignment"
USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FULL = Path("images/shape_dataset_v14_full")
USER_OUTPUT_DIR_FULL = Path("output/shape_train_v14_full")
USER_VAL_SIZE_FULL = 256

USER_DATASET_DIR_SMALL = Path("images/shape_dataset_v14_small")
USER_OUTPUT_DIR_SMALL = Path("output/shape_train_v14_small")
USER_VAL_SIZE_SMALL = 16

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 1
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_EPOCHS = 20
USER_LR = 1e-4
USER_WEIGHT_DECAY = 0.0
USER_GRAD_CLIP = 1.0

USER_REPAIR_INIT_CHECKPOINT = "output/shape_train_v14_small/checkpoints/best.pth"
USER_RESUME_CHECKPOINT = ""
USER_PP_V14_CHECKPOINT = ""

USER_NHR_BASE_CHANNELS = 32
USER_OCC_DILATE = 9
USER_RING_DILATE = 9
USER_RING_ERODE = 1
USER_USE_RING = True
USER_DISABLE_LOCAL_BYPASS = True
USER_NHR_BYPASS_SCALE = 0.15
USER_BG0_BLUR_KERNEL = 11
USER_BG0_SEAN_BLEND = 0.15
USER_USE_RAW_SOURCE_FOR_REFINE = False
USER_REAL_RECON_SAFE_WEIGHT = 0.10
USER_REAL_RECON_RING_WEIGHT = 1.00
USER_REAL_RECON_OCC_WEIGHT = 0.0

USER_SYNTHETIC_AUX_WEIGHT = 0.50
USER_TRAIN_POSTPROCESS = False
USER_VAL_MAX_STEPS = 16
USER_VAL_PREVIEW_COUNT = 15
USER_SAVE_PREVIEW_EVERY = 100
USER_SAVE_CHECKPOINT_EVERY = 1
USER_TRAIN_RECORD_CHUNK_SIZE = 256
USER_VAL_RECORD_CHUNK_SIZE = 64
USER_EMPTY_CACHE_BETWEEN_CHUNKS = True
USER_EMBED_IMAGE_SIZE = 1024
# ========================================================================


def resolve_profile_defaults():
    if USER_DATASET_PROFILE == "full":
        return {
            "dataset_dir": USER_DATASET_DIR_FULL,
            "output_dir": USER_OUTPUT_DIR_FULL,
            "val_size": USER_VAL_SIZE_FULL,
        }
    if USER_DATASET_PROFILE == "small":
        return {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "val_size": USER_VAL_SIZE_SMALL,
        }
    raise ValueError(f"Unsupported USER_DATASET_PROFILE: {USER_DATASET_PROFILE}")


PROFILE = resolve_profile_defaults()
DATASET_DIR = PROFILE["dataset_dir"]
OUTPUT_DIR = PROFILE["output_dir"]
VAL_SIZE = PROFILE["val_size"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_module(module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def to_unit(image_tanh: torch.Tensor) -> torch.Tensor:
    return ((image_tanh + 1.0) / 2.0).clamp(0.0, 1.0)


def build_source_records(records: list[dict]) -> list[dict]:
    unique = {}
    for record in records:
        unique.setdefault(
            record["source_path"],
            {
                "sample_id": Path(record["source_path"]).stem,
                "source_path": record["source_path"],
                "source_parsing_path": record["source_parsing_path"],
            },
        )
    return list(unique.values())


def split_stage_records(stage: str) -> tuple[list[dict], list[dict]]:
    manifest_path = DATASET_DIR / "manifest.jsonl"
    records = read_jsonl(manifest_path)
    if stage == "repair_pretrain":
        records = build_source_records(records)

    if not records:
        raise ValueError(f"No records found for stage={stage}.")

    indices = list(range(len(records)))
    random.Random(USER_RANDOM_SEED).shuffle(indices)
    val_size = min(len(indices) - 1, max(1, int(VAL_SIZE))) if len(indices) > 1 else 1
    train_records = [records[idx] for idx in indices[val_size:]] if len(indices) > val_size else list(records)
    val_records = [records[idx] for idx in indices[:val_size]]
    return train_records, val_records


def build_dataset_from_records(stage: str, records: list[dict], augment_flip: bool):
    if stage == "repair_pretrain":
        return NHRSourceDatasetV14(records=records, augment_flip=augment_flip, include_full_images=False)
    return NHRPairDatasetV14(records=records, augment_flip=augment_flip, include_full_images=False)


def build_loader_from_records(stage: str, records: list[dict], *, batch_size: int, augment_flip: bool, shuffle: bool):
    dataset = build_dataset_from_records(stage, records, augment_flip=augment_flip)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY,
        collate_fn=collate_dict,
    )


def chunk_records(records: list[dict], chunk_size: int) -> list[list[dict]]:
    chunk_size = max(1, int(chunk_size))
    return [records[idx : idx + chunk_size] for idx in range(0, len(records), chunk_size)]


def build_epoch_train_chunks(train_records: list[dict], epoch: int) -> list[list[dict]]:
    shuffled = list(train_records)
    random.Random(USER_RANDOM_SEED + epoch).shuffle(shuffled)
    return chunk_records(shuffled, max(USER_BATCH_SIZE, int(USER_TRAIN_RECORD_CHUNK_SIZE)))


def build_val_chunks(val_records: list[dict]) -> list[list[dict]]:
    return chunk_records(list(val_records), max(1, int(USER_VAL_RECORD_CHUNK_SIZE)))


def maybe_release_memory() -> None:
    gc.collect()
    if USER_EMPTY_CACHE_BETWEEN_CHUNKS and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_embed_image(path: str | Path) -> torch.Tensor:
    image_size = None if USER_EMBED_IMAGE_SIZE in (None, 0) else (int(USER_EMBED_IMAGE_SIZE), int(USER_EMBED_IMAGE_SIZE))
    return load_rgb(path, size=image_size, tanh=False).cpu()


def save_preview(path: Path, tiles: list[torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid([tile.detach().cpu() for tile in tiles], nrow=len(tiles))
    save_image(grid, path)


def build_nonhair_pseudo_target(stage: dict[str, torch.Tensor]) -> torch.Tensor:
    target = stage["source_mask"].clone().long()
    occ = stage["M_occ_d"] > 0.5
    face = (dilate(stage["M_face"], 5) > 0.5) & occ
    neck = (dilate(stage["M_neck"], 7) > 0.5) & occ & (~face)
    cloth = (dilate(stage["M_cloth"], 7) > 0.5) & occ & (~face) & (~neck)
    rest = occ & (~face) & (~neck) & (~cloth)

    target[face] = 1
    target[neck] = 17
    target[cloth] = 18
    target[rest] = 1
    return target


def build_real_reconstruction_mask(stage: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        USER_REAL_RECON_SAFE_WEIGHT * stage["M_safe"]
        + USER_REAL_RECON_RING_WEIGHT * stage["M_ring"]
    ).clamp(0.0, 1.0)


def call_nhr_loss_v14(loss_builder: NHRLossBuilderV14, *, recon_mask: torch.Tensor | None = None, **kwargs):
    forward_parameters = inspect.signature(loss_builder.forward).parameters
    supports_i_bg0 = "I_bg0" in forward_parameters
    supports_recon_mask = "recon_mask" in forward_parameters
    supports_erase_mask = "erase_mask" in forward_parameters

    if "I_bg0" in kwargs and not supports_i_bg0:
        kwargs.pop("I_bg0")
    if "erase_mask" in kwargs and not supports_erase_mask:
        kwargs.pop("erase_mask")
    if recon_mask is None or supports_recon_mask:
        if recon_mask is not None:
            kwargs["recon_mask"] = recon_mask
        return loss_builder(**kwargs)

    # Fallback for stale/foreign NHRLossBuilderV14 imports that do not yet expose recon_mask.
    if not getattr(call_nhr_loss_v14, "_warned_missing_recon_mask", False):
        print(
            "[shape_train_v14] Warning: imported NHRLossBuilderV14.forward does not accept recon_mask; "
            "falling back to M_occ_d=recon_mask compatibility mode."
        )
        call_nhr_loss_v14._warned_missing_recon_mask = True
    kwargs["M_occ_d"] = recon_mask.clamp(0.0, 1.0)
    return loss_builder(**kwargs)


def build_repair_model() -> RepairNetV14:
    model = RepairNetV14(in_channels=17, base_channels=USER_NHR_BASE_CHANNELS)
    if USER_REPAIR_INIT_CHECKPOINT:
        checkpoint = torch.load(USER_REPAIR_INIT_CHECKPOINT, map_location=USER_DEVICE)
        state_dict = checkpoint.get("repair_net_state_dict", checkpoint.get("repair_net", checkpoint))
        model.load_state_dict(state_dict, strict=False)
    return model.to(USER_DEVICE)


def build_hairfast_v14() -> HairFastV14:
    args = get_parser_v14().parse_args([])
    args.device = USER_DEVICE
    args.save_all = False
    args.use_nhr_branch = 1
    args.nhr_base_channels = USER_NHR_BASE_CHANNELS
    args.nhr_occ_dilate = USER_OCC_DILATE
    args.nhr_ring_dilate = USER_RING_DILATE
    args.nhr_ring_erode = USER_RING_ERODE
    args.nhr_use_ring = int(USER_USE_RING)
    args.nhr_disable_local_bypass = int(USER_DISABLE_LOCAL_BYPASS)
    args.nhr_bypass_scale = USER_NHR_BYPASS_SCALE
    args.nhr_bg0_blur_kernel = USER_BG0_BLUR_KERNEL
    args.nhr_bg0_sean_blend = USER_BG0_SEAN_BLEND
    args.nhr_use_raw_source_for_refine = int(USER_USE_RAW_SOURCE_FOR_REFINE)
    args.pp_v14_checkpoint = USER_PP_V14_CHECKPOINT
    args.repair_checkpoint = USER_REPAIR_INIT_CHECKPOINT
    hair_fast = HairFastV14(args)

    freeze_module(hair_fast.embed)
    freeze_module(hair_fast.net)
    freeze_module(hair_fast.align.sean_model)
    freeze_module(hair_fast.align.mask_generator)
    freeze_module(hair_fast.align.rotate_model)
    freeze_module(hair_fast.blend.blending_encoder)
    freeze_module(hair_fast.blend.post_process)

    unfreeze_module(hair_fast.align.repair_net)
    if USER_DISABLE_LOCAL_BYPASS:
        freeze_module(hair_fast.align.repair_adapter)
        freeze_module(hair_fast.align.repair_gate)
    else:
        unfreeze_module(hair_fast.align.repair_adapter)
        unfreeze_module(hair_fast.align.repair_gate)
    if USER_TRAIN_POSTPROCESS:
        unfreeze_module(hair_fast.blend.post_process)

    hair_fast.align.repair_net.train()
    if USER_DISABLE_LOCAL_BYPASS:
        hair_fast.align.repair_adapter.eval()
        hair_fast.align.repair_gate.eval()
    else:
        hair_fast.align.repair_adapter.train()
        hair_fast.align.repair_gate.train()
    if USER_TRAIN_POSTPROCESS:
        hair_fast.blend.post_process.train()
    else:
        hair_fast.blend.post_process.eval()
    return hair_fast


def save_repair_checkpoint(model, optimizer, epoch: int, step: int, best_loss: float, name: str) -> None:
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "best_loss": best_loss,
            "repair_net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_dir / f"{name}.pth",
    )


def save_joint_checkpoint(hair_fast: HairFastV14, optimizer, epoch: int, step: int, best_loss: float, name: str) -> None:
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "repair_net_state_dict": hair_fast.align.repair_net.state_dict(),
        "repair_adapter_state_dict": hair_fast.align.repair_adapter.state_dict(),
        "repair_gate_state_dict": hair_fast.align.repair_gate.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if USER_TRAIN_POSTPROCESS:
        state["post_process_state_dict"] = hair_fast.blend.post_process.state_dict()
    torch.save(state, ckpt_dir / f"{name}.pth")


def maybe_resume_repair(model, optimizer):
    if not USER_RESUME_CHECKPOINT:
        return 0, 0, float("inf")
    checkpoint = torch.load(USER_RESUME_CHECKPOINT, map_location=USER_DEVICE)
    model.load_state_dict(checkpoint["repair_net_state_dict"], strict=False)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0), checkpoint.get("best_loss", float("inf"))


def maybe_resume_joint(hair_fast: HairFastV14, optimizer):
    if not USER_RESUME_CHECKPOINT:
        return 0, 0, float("inf")
    checkpoint = torch.load(USER_RESUME_CHECKPOINT, map_location=USER_DEVICE)
    hair_fast.align.repair_net.load_state_dict(checkpoint["repair_net_state_dict"], strict=False)
    hair_fast.align.repair_adapter.load_state_dict(checkpoint["repair_adapter_state_dict"], strict=False)
    hair_fast.align.repair_gate.load_state_dict(checkpoint["repair_gate_state_dict"], strict=False)
    if USER_TRAIN_POSTPROCESS and "post_process_state_dict" in checkpoint:
        hair_fast.blend.post_process.load_state_dict(checkpoint["post_process_state_dict"], strict=False)
    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            print("[joint_alignment] skip optimizer state restore because parameter groups changed.")
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0), checkpoint.get("best_loss", float("inf"))


def synthesize_batch(batch: dict) -> dict[str, torch.Tensor]:
    items = []
    batch_size = len(batch["sample_id"])
    for idx in range(batch_size):
        items.append(
            build_synthetic_pretrain_batch_v14(
                batch["source_image_tanh"][idx].to(USER_DEVICE),
                batch["source_parsing"][idx].to(USER_DEVICE),
                occ_dilate=USER_OCC_DILATE,
            )
        )
    merged = {}
    for key in items[0]:
        merged[key] = torch.cat([item[key] for item in items], dim=0)
    return merged


def run_pretrain_forward(model: RepairNetV14, synth: dict) -> dict[str, torch.Tensor]:
    repair_input = torch.cat(
        [
            synth["I_source"],
            synth["I_bg0"],
            synth["I_hair_coarse"],
            synth["H_source"],
            synth["H_align"],
            synth["M_occ"],
            synth["M_occ_d"],
            synth["M_face"],
            synth["M_neck"],
            synth["M_cloth"],
            synth["M_ring"],
        ],
        dim=1,
    )
    outputs = model(repair_input)
    A = torch.sigmoid(outputs["A_fill"]) * synth["M_occ_d"]
    I_clean_bg = (1.0 - A) * synth["I_bg0"] + A * outputs["I_fill"]
    I_align_clean = (1.0 - synth["H_align"]) * I_clean_bg + synth["H_align"] * synth["I_hair_coarse"]
    return {
        **outputs,
        "A": A,
        "I_clean_bg": I_clean_bg,
        "I_align_clean": I_align_clean,
    }


def validate_pretrain(model, val_records: list[dict], loss_builder, step: int) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    preview_saved = False
    with torch.no_grad():
        val_chunks = build_val_chunks(val_records)
        for chunk_idx, chunk_records_batch in enumerate(val_chunks):
            val_loader = build_loader_from_records(
                "repair_pretrain",
                chunk_records_batch,
                batch_size=1,
                augment_flip=False,
                shuffle=False,
            )
            for batch in tqdm(val_loader, desc=f"Val pretrain {chunk_idx + 1}/{len(val_chunks)}", leave=False):
                synth = synthesize_batch(batch)
                outputs = run_pretrain_forward(model, synth)
                loss, _ = call_nhr_loss_v14(
                    loss_builder,
                    I_clean_bg=outputs["I_clean_bg"],
                    I_gt=synth["I_gt"],
                    I_source=synth["I_source"],
                    I_align_clean=outputs["I_align_clean"],
                    I_hair_coarse=synth["I_hair_coarse"],
                    H_align=synth["H_align"],
                    A=outputs["A"],
                    M_occ_d=synth["M_occ_d"],
                    M_ring=synth["M_ring"],
                    M_safe=synth["M_safe"],
                    semantic_target=synth["source_parsing"],
                )
                total_loss += loss.item()
                count += 1
                if not preview_saved:
                    preview_saved = True
                    save_preview(
                        OUTPUT_DIR / "previews" / f"pretrain_val_step_{step:06d}.png",
                        [
                            to_unit(synth["I_source"][0]),
                            to_unit(synth["I_bg0"][0]),
                            to_unit(outputs["I_fill"][0]),
                            to_unit(outputs["I_clean_bg"][0]),
                            to_unit(synth["I_gt"][0]),
                            synth["M_occ_d"][0].repeat(3, 1, 1),
                        ],
                    )
            del val_loader
            maybe_release_memory()
    model.train()
    return total_loss / max(count, 1)


def train_repair_pretrain():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_records, val_records = split_stage_records("repair_pretrain")
    model = build_repair_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
    loss_builder = NHRLossBuilderV14()
    start_epoch, step, best_loss = maybe_resume_repair(model, optimizer)

    for epoch in range(start_epoch, USER_EPOCHS):
        train_chunks = build_epoch_train_chunks(train_records, epoch)
        for chunk_idx, train_chunk in enumerate(train_chunks):
            train_loader = build_loader_from_records(
                "repair_pretrain",
                train_chunk,
                batch_size=USER_BATCH_SIZE,
                augment_flip=True,
                shuffle=False,
            )
            progress = tqdm(train_loader, desc=f"Pretrain epoch {epoch + 1}/{USER_EPOCHS} chunk {chunk_idx + 1}/{len(train_chunks)}", leave=False)
            for batch in progress:
                synth = synthesize_batch(batch)
                outputs = run_pretrain_forward(model, synth)
                loss, _ = call_nhr_loss_v14(
                    loss_builder,
                    I_clean_bg=outputs["I_clean_bg"],
                    I_gt=synth["I_gt"],
                    I_source=synth["I_source"],
                    I_align_clean=outputs["I_align_clean"],
                    I_hair_coarse=synth["I_hair_coarse"],
                    H_align=synth["H_align"],
                    A=outputs["A"],
                    M_occ_d=synth["M_occ_d"],
                    M_ring=synth["M_ring"],
                    M_safe=synth["M_safe"],
                    semantic_target=synth["source_parsing"],
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), USER_GRAD_CLIP)
                optimizer.step()
                step += 1
                progress.set_postfix(loss=float(loss.item()))

                if step % USER_SAVE_PREVIEW_EVERY == 0:
                    save_preview(
                        OUTPUT_DIR / "previews" / f"pretrain_step_{step:06d}.png",
                        [
                            to_unit(synth["I_source"][0]),
                            to_unit(synth["I_bg0"][0]),
                            to_unit(outputs["I_fill"][0]),
                            to_unit(outputs["I_clean_bg"][0]),
                            to_unit(synth["I_gt"][0]),
                            synth["M_occ_d"][0].repeat(3, 1, 1),
                        ],
                    )
            del train_loader
            maybe_release_memory()

        val_loss = validate_pretrain(model, val_records, loss_builder, step)
        print(f"[repair_pretrain] epoch={epoch + 1} val_loss={val_loss:.6f}")
        if (epoch + 1) % USER_SAVE_CHECKPOINT_EVERY == 0:
            save_repair_checkpoint(model, optimizer, epoch + 1, step, best_loss, "last")
        if val_loss <= best_loss:
            best_loss = val_loss
            save_repair_checkpoint(model, optimizer, epoch + 1, step, best_loss, "best")


def run_joint_sample(hair_fast: HairFastV14, sample: dict, loss_builder: NHRLossBuilderV14, *, need_preview: bool = False):
    source_cpu = load_embed_image(sample["source_path"])
    reference_cpu = load_embed_image(sample["reference_path"])
    source_tanh = sample["source_image_tanh"].unsqueeze(0).to(USER_DEVICE)
    source_parsing = sample["source_parsing"].unsqueeze(0).to(USER_DEVICE)

    images_to_name = defaultdict(list)
    images_to_name[source_cpu].append("face")
    images_to_name[reference_cpu].extend(["shape", "color"])
    name_to_embed = hair_fast.embed.embedding_images(images_to_name)

    align_shape = hair_fast.align.align_images("face", "shape", name_to_embed)
    align_color = align_shape

    pseudo_target = build_nonhair_pseudo_target(align_shape)
    real_recon_mask = build_real_reconstruction_mask(align_shape)
    erase_mask = align_shape["M_occ_d"]
    real_loss, real_info = call_nhr_loss_v14(
        loss_builder,
        I_clean_bg=align_shape["I_clean_bg"],
        I_gt=name_to_embed["face"]["image_norm_256"],
        I_bg0=align_shape["I_bg0"],
        recon_mask=real_recon_mask,
        erase_mask=erase_mask,
        I_source=name_to_embed["face"]["image_norm_256"],
        I_align_clean=align_shape["I_align_clean"],
        I_hair_coarse=align_shape["I_hair_coarse"],
        H_align=align_shape["H_align"],
        A=align_shape["A"],
        M_occ_d=align_shape["M_occ_d"],
        M_ring=align_shape["M_ring"],
        M_safe=align_shape["M_safe"],
        semantic_target=pseudo_target,
    )

    synth = build_synthetic_pretrain_batch_v14(source_tanh, source_parsing, occ_dilate=USER_OCC_DILATE)
    synth_outputs = run_pretrain_forward(hair_fast.align.repair_net, synth)
    aux_loss, aux_info = call_nhr_loss_v14(
        loss_builder,
        I_clean_bg=synth_outputs["I_clean_bg"],
        I_gt=synth["I_gt"],
        recon_mask=synth["M_occ_d"],
        I_source=synth["I_source"],
        I_align_clean=synth_outputs["I_align_clean"],
        I_hair_coarse=synth["I_hair_coarse"],
        H_align=synth["H_align"],
        A=synth_outputs["A"],
        M_occ_d=synth["M_occ_d"],
        M_ring=synth["M_ring"],
        M_safe=synth["M_safe"],
        semantic_target=synth["source_parsing"],
    )

    total_loss = real_loss + USER_SYNTHETIC_AUX_WEIGHT * aux_loss
    logs = {
        "pair_total": real_loss.detach(),
        "aux_total": aux_loss.detach(),
        "total": total_loss.detach(),
        **{f"pair_{key}": value.detach() for key, value in real_info.items()},
        **{f"aux_{key}": value.detach() for key, value in aux_info.items()},
    }
    final_preview = None
    if need_preview:
        with torch.no_grad():
            blend = hair_fast.blend.blend_images(align_shape, align_color, name_to_embed, return_pipeline_info=True)
        final_preview = F.interpolate(blend["final_image"].unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)
    preview = {
        "source": name_to_embed["face"]["image_norm_256"],
        "reference": name_to_embed["shape"]["image_norm_256"],
        "bg0": align_shape["I_bg0"],
        "clean_bg": align_shape["I_clean_bg"],
        "real_recon_mask": real_recon_mask,
        "source_clean": align_shape["I_source_clean"],
        "align_clean": align_shape["I_align_clean"],
        "final": final_preview,
        "occ": align_shape["M_occ_d"],
    }
    return total_loss, logs, preview


def save_joint_validation_preview(preview: dict[str, torch.Tensor | None], *, epoch: int, index: int, sample_id: str) -> None:
    sample_tag = str(sample_id).replace("\\", "_").replace("/", "_").replace(":", "_")
    final_tile = preview["final"][0].clamp(0.0, 1.0) if preview["final"] is not None else to_unit(preview["align_clean"][0])
    save_preview(
        OUTPUT_DIR / "previews" / f"joint_val_epoch_{epoch:03d}_{index:02d}_{sample_tag}.png",
        [
            to_unit(preview["source"][0]),
            to_unit(preview["reference"][0]),
            to_unit(preview["bg0"][0]),
            to_unit(preview["clean_bg"][0]),
            to_unit(preview["source_clean"][0]),
            to_unit(preview["align_clean"][0]),
            final_tile,
            preview["occ"][0].repeat(3, 1, 1),
            preview["real_recon_mask"][0].repeat(3, 1, 1),
        ],
    )


def validate_joint(hair_fast: HairFastV14, val_records: list[dict], loss_builder, epoch: int) -> float:
    total = 0.0
    count = 0
    preview_count = 0
    val_step_limit = max(USER_VAL_MAX_STEPS, USER_VAL_PREVIEW_COUNT)
    with torch.no_grad():
        val_chunks = build_val_chunks(val_records)
        for chunk_idx, chunk_records_batch in enumerate(val_chunks):
            val_loader = build_loader_from_records(
                "joint_alignment",
                chunk_records_batch,
                batch_size=1,
                augment_flip=False,
                shuffle=False,
            )
            for batch in tqdm(val_loader, desc=f"Val joint {chunk_idx + 1}/{len(val_chunks)}", leave=False):
                sample = {
                    "sample_id": batch["sample_id"][0],
                    "source_path": batch["source_path"][0],
                    "reference_path": batch["reference_path"][0],
                    "source_image": batch["source_image"][0],
                    "reference_image": batch["reference_image"][0],
                    "source_image_tanh": batch["source_image_tanh"][0],
                    "source_parsing": batch["source_parsing"][0],
                }
                should_save_preview = preview_count < USER_VAL_PREVIEW_COUNT
                loss, _, preview = run_joint_sample(hair_fast, sample, loss_builder, need_preview=should_save_preview)
                total += loss.item()
                count += 1
                if should_save_preview:
                    preview_count += 1
                    save_joint_validation_preview(
                        preview,
                        epoch=epoch,
                        index=preview_count,
                        sample_id=sample["sample_id"],
                    )
                if count >= val_step_limit:
                    break
            del val_loader
            maybe_release_memory()
            if count >= val_step_limit:
                break
    return total / max(count, 1)


def train_joint_alignment():
    if not USER_REPAIR_INIT_CHECKPOINT and not USER_RESUME_CHECKPOINT:
        raise ValueError(
            "joint_alignment requires a pretrained repair checkpoint. "
            "Run repair_pretrain first, then set USER_REPAIR_INIT_CHECKPOINT to its best.pth "
            "(or set USER_RESUME_CHECKPOINT to an existing joint checkpoint)."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_records, val_records = split_stage_records("joint_alignment")
    hair_fast = build_hairfast_v14()
    params = list(hair_fast.align.repair_net.parameters())
    if not USER_DISABLE_LOCAL_BYPASS:
        params += list(hair_fast.align.repair_adapter.parameters()) + list(hair_fast.align.repair_gate.parameters())
    if USER_TRAIN_POSTPROCESS:
        params += list(hair_fast.blend.post_process.parameters())
    optimizer = torch.optim.Adam(params, lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
    loss_builder = NHRLossBuilderV14()
    start_epoch, step, best_loss = maybe_resume_joint(hair_fast, optimizer)

    for epoch in range(start_epoch, USER_EPOCHS):
        train_chunks = build_epoch_train_chunks(train_records, epoch)
        for chunk_idx, train_chunk in enumerate(train_chunks):
            train_loader = build_loader_from_records(
                "joint_alignment",
                train_chunk,
                batch_size=USER_BATCH_SIZE,
                augment_flip=True,
                shuffle=False,
            )
            progress = tqdm(train_loader, desc=f"Joint epoch {epoch + 1}/{USER_EPOCHS} chunk {chunk_idx + 1}/{len(train_chunks)}", leave=False)
            for batch in progress:
                batch_size = len(batch["sample_id"])
                for idx in range(batch_size):
                    need_preview = step > 0 and step % USER_SAVE_PREVIEW_EVERY == 0
                    sample = {
                        "sample_id": batch["sample_id"][idx],
                        "source_path": batch["source_path"][idx],
                        "reference_path": batch["reference_path"][idx],
                        "source_image": batch["source_image"][idx],
                        "reference_image": batch["reference_image"][idx],
                        "source_image_tanh": batch["source_image_tanh"][idx],
                        "source_parsing": batch["source_parsing"][idx],
                    }
                    optimizer.zero_grad()
                    loss, logs, preview = run_joint_sample(hair_fast, sample, loss_builder, need_preview=need_preview)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, USER_GRAD_CLIP)
                    optimizer.step()
                    step += 1
                    progress.set_postfix(loss=float(logs["total"].item()))

                    if need_preview and preview["final"] is not None:
                        save_preview(
                            OUTPUT_DIR / "previews" / f"joint_step_{step:06d}.png",
                            [
                                to_unit(preview["source"][0]),
                                to_unit(preview["reference"][0]),
                                to_unit(preview["bg0"][0]),
                                to_unit(preview["clean_bg"][0]),
                                to_unit(preview["source_clean"][0]),
                                to_unit(preview["align_clean"][0]),
                                preview["final"][0].clamp(0.0, 1.0),
                                preview["occ"][0].repeat(3, 1, 1),
                            ],
                        )
            del train_loader
            maybe_release_memory()

        val_loss = validate_joint(hair_fast, val_records, loss_builder, epoch + 1)
        print(f"[joint_alignment] epoch={epoch + 1} val_loss={val_loss:.6f}")
        if (epoch + 1) % USER_SAVE_CHECKPOINT_EVERY == 0:
            save_joint_checkpoint(hair_fast, optimizer, epoch + 1, step, best_loss, "last")
        if val_loss <= best_loss:
            best_loss = val_loss
            save_joint_checkpoint(hair_fast, optimizer, epoch + 1, step, best_loss, "best")


def main():
    set_seed(USER_RANDOM_SEED)
    if USER_STAGE == "repair_pretrain":
        train_repair_pretrain()
        return
    if USER_STAGE == "joint_alignment":
        train_joint_alignment()
        return
    raise ValueError(f"Unsupported USER_STAGE: {USER_STAGE}")


if __name__ == "__main__":
    main()
