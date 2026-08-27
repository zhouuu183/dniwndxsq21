"""Trace the live V6 PP path against the no-suffix author implementation.

This script is diagnostic only. It does not train, write dataset parts, or
change checkpoints.  Its numbered PNGs keep the formal author pre-PP image
separate from the cached direct-SATD PP target.
"""

import argparse
import gc
import gzip
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap import HairFast, get_parser as get_author_parser
from hair_swap_v6 import HairFastV6, get_parser as get_v6_parser
from models.Net import Net
from models.postprocess_v6 import PostProcessModelV6
from scripts.pp_gen_v6 import RESOLVED_USER_CONFIG as GEN_CONFIG
from scripts.pp_train_v6 import (
    RESOLVED_USER_CONFIG as TRAIN_CONFIG,
    build_parser as build_train_parser,
)
from utils.bicubic import BicubicDownSample


def normalize(image):
    return (image - 0.5) / 0.5


def load_rgb(path):
    with Image.open(path) as image:
        return T.functional.to_tensor(image.convert("RGB"))


def as_rgb_01(value, name):
    if not torch.is_tensor(value) or value.ndim != 3 or value.size(0) != 3:
        raise RuntimeError(f"{name} must be RGB [3,H,W], got {type(value).__name__}.")
    if value.dtype == torch.uint8:
        return value.float().div(255)
    return value.float().clamp(0, 1)


def as_mask_4d(value, name):
    """Normalize one dataset/debug mask to ``[1, 1, H, W]`` on CPU."""

    if not torch.is_tensor(value):
        raise RuntimeError(f"{name} is missing or is not a tensor.")
    value = value.detach().cpu()
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        if value.size(0) == 1:
            value = value.unsqueeze(0)
        else:
            # Dataset items are single samples.  A multi-channel mask is not
            # a valid alpha contract; retain only the first channel so the
            # diagnostic cannot silently display an RGB image as a mask.
            value = value[:1].unsqueeze(0)
    elif value.ndim == 4 and value.size(0) == 1:
        value = value[:, :1]
    else:
        raise RuntimeError(f"{name} must have [H,W], [1,H,W], or [1,1,H,W], got {tuple(value.shape)}.")
    if value.dtype == torch.uint8:
        value = value.float().div(255.0)
    else:
        value = value.float()
    return value.clamp(0, 1)


def save_mask(value, path, name):
    """Save a mask as a high-contrast RGB PNG and return its area."""

    mask = as_mask_4d(value, name)
    save_image(mask.repeat(1, 3, 1, 1), path)
    return float(mask.sum().item()), tuple(mask.shape[-2:])


def batch_tensor(value, device):
    """Add a batch dimension to a single dataset item tensor."""

    if value is None or not torch.is_tensor(value):
        return None
    if value.ndim in (2, 3):
        value = value.unsqueeze(0)
    return value.to(device)


def batch_image_01(value, device):
    """Normalize a single stored RGB image to batched ``[0, 1]`` tensor."""

    value = batch_tensor(value, device)
    if value is None:
        return None
    if value.dtype == torch.uint8:
        value = value.float().div(255.0)
    else:
        value = value.float()
    return value.clamp(0, 1)


def capture_author_pre_pp_and_final(author_model, source_path, shape_path, color_path):
    """Run the complete author pipeline while recording its actual PP input."""

    original_downsample = author_model.blend.downsample_256

    class CapturePrePPDownsample(nn.Module):
        def __init__(self):
            super().__init__()
            self.image = None

        def forward(self, image):
            self.image = ((image[0] + 1.0) * 0.5).clamp(0, 1).detach().cpu()
            return original_downsample(image)

    capture = CapturePrePPDownsample()
    author_model.blend.downsample_256 = capture
    try:
        final_image = author_model(source_path, shape_path, color_path).detach().cpu()
    finally:
        author_model.blend.downsample_256 = original_downsample
    if capture.image is None:
        raise RuntimeError("The author pipeline did not reach its pre-PP stage.")
    return capture.image, final_image


def decode_pp(generator, latent_s, latent_f):
    """Decode with fixed StyleGAN noise for a reproducible pixel comparison."""

    image, _ = generator(
        [latent_s],
        input_is_latent=True,
        return_latents=False,
        randomize_noise=False,
        start_layer=5,
        end_layer=8,
        layer_in=latent_f,
    )
    return ((image[0] + 1.0) * 0.5).clamp(0, 1).cpu()


def error_metrics(left, right):
    difference = (left - right).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
    }


def load_dataset_part(path: Path):
    """Load V6 dataset parts in plain or gzip-compressed format."""

    with path.open("rb") as probe:
        is_gzip = probe.read(2) == b"\x1f\x8b"
    if is_gzip:
        with gzip.open(path, mode="rb") as handle:
            return torch.load(handle, map_location="cpu")
    return torch.load(path, map_location="cpu")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = Path(args.dataset)
    part_path = dataset_path / f"pp_part_{args.part}.dataset"
    if not part_path.is_file():
        raise FileNotFoundError(f"Dataset part does not exist: {part_path}")

    items = load_dataset_part(part_path)
    if not 0 <= args.item < len(items):
        raise IndexError(f"--item must be in [0, {len(items) - 1}], got {args.item}")
    item = items[args.item]
    required = (
        "source_path",
        "shape_reference_path",
        "color_reference_path",
        "target",
        "direct_satd_pp_input",
    )
    missing = [key for key in required if key not in item]
    if missing:
        raise RuntimeError(f"Dataset item is missing required keys: {', '.join(missing)}")
    if not bool(item["direct_satd_pp_input"]):
        raise RuntimeError(
            "This trace expects a schema-35 direct-SATD dataset item. "
            "Regenerate with pp_gen_v6.py --direct_satd_pp_input true."
        )

    source_path = Path(item["source_path"])
    shape_path = Path(item["shape_reference_path"])
    color_path = Path(item["color_reference_path"])
    for label, path in (("source", source_path), ("shape", shape_path), ("color", color_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} reference is unavailable: {path}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset_target = as_rgb_01(item["target"], "dataset target")
    source_full = load_rgb(source_path)
    downsample_256 = BicubicDownSample(factor=4)
    # BicubicDownSample creates its filters on the active CUDA device. Keep
    # every input to it on that same device, then retain CPU copies for PNGs.
    source_256 = (
        downsample_256(source_full.unsqueeze(0).to(device))
        .squeeze(0)
        .clamp(0, 1)
        .cpu()
    )

    # A: run the unmodified no-suffix pipeline, including its formal PP decode.
    author_args = get_author_parser().parse_args([])
    author_args.blending_checkpoint = GEN_CONFIG["blending_checkpoint"]
    author_model = HairFast(author_args)
    author_pre_pp_1024, author_baseline_final_captured = capture_author_pre_pp_and_final(
        author_model,
        source_path,
        shape_path,
        color_path,
    )
    # The preceding invocation temporarily wrapped only the downsampler to
    # observe I_blend.  Run the same author object again after restoration so
    # image 05 is a completely untouched author call, not an inferred result.
    author_baseline_final = author_model(source_path, shape_path, color_path).detach().cpu()
    author_pre_pp_256 = (
        downsample_256(author_pre_pp_1024.unsqueeze(0).to(device))
        .squeeze(0)
        .clamp(0, 1)
        .cpu()
    )

    with torch.no_grad():
        author_s_from_author_target, author_f_from_author_target = author_model.blend.post_process(
            normalize(source_256.unsqueeze(0).to(device)),
            normalize(author_pre_pp_256.unsqueeze(0).to(device)),
        )
        author_pp_from_author_target = decode_pp(
            author_model.net.generator,
            author_s_from_author_target,
            author_f_from_author_target,
        )
        author_s_from_dataset_target, author_f_from_dataset_target = author_model.blend.post_process(
            normalize(source_256.unsqueeze(0).to(device)),
            normalize(dataset_target.unsqueeze(0).to(device)),
        )
        author_pp_from_dataset_target = decode_pp(
            author_model.net.generator,
            author_s_from_dataset_target,
            author_f_from_dataset_target,
        )
        author_s_cpu = author_s_from_dataset_target.cpu()
        author_f_cpu = author_f_from_dataset_target.cpu()

    del author_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # B: regenerate the V6 pre-PP image with the current code.  The cached
    # dataset may have been made before a subsequent code change, so it is an
    # independent comparison rather than an assumed ground truth.
    v6_args = get_v6_parser().parse_args([])
    v6_args.blending_checkpoint = GEN_CONFIG["blending_checkpoint"]
    v6_args.allow_legacy_blending_checkpoint_v8 = GEN_CONFIG[
        "allow_legacy_blending_checkpoint_v8"
    ]
    # Recreate the direct-SATD stage used by pp_gen_v6 so the regenerated
    # target is comparable to the cached dataset target.
    v6_args.use_satd_v8 = bool(GEN_CONFIG.get("use_satd_v8", True))
    v6_args.satd_checkpoint_v8 = GEN_CONFIG.get("satd_checkpoint_v8", "")
    v6_args.direct_satd_pp_input = True
    v6_current = HairFastV6(v6_args)
    current_stage = v6_current(
        source_path,
        shape_path,
        color_path,
        return_stage="color_before_pp",
        stop_before_pp=True,
    )
    current_target = as_rgb_01(current_stage["color_before_pp"], "current V6 target")
    del v6_current
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # C: the exact V6 PP model/config currently used by pp_train_v6.py.
    train_args = build_train_parser(TRAIN_CONFIG).parse_args([])
    v6_model = PostProcessModelV6(train_args).to(device).eval()
    v6_model.load_base_checkpoint(train_args.base_checkpoint)
    source_full_batch = source_full.unsqueeze(0).to(device)
    target_mask = batch_tensor(item.get("target_mask"), device)
    target_hair = batch_tensor(item.get("HT_E"), device)
    source_native_alpha_item = item.get("source_native_earring_alpha")
    source_structured_alpha_item = item.get("source_earring_structured_alpha")
    earring_learning_mask_item = item.get("earring_learning_mask")
    source_native_alpha = batch_tensor(source_native_alpha_item, device)
    earring_learning_mask = batch_tensor(earring_learning_mask_item, device)
    completed_hair = batch_image_01(item.get("completed_hair_highres"), device)
    satd_background = batch_image_01(item.get("satd_background_highres"), device)
    # Forward the same dataset evidence that pp_train_v6 uses.  This keeps the
    # final alpha diagnostic tied to the actual V6 compositor contract instead
    # of running a reduced, mask-free approximation.
    common_v6_kwargs = {
        "source_parsing": batch_tensor(item.get("source_parsing"), device),
        "target_parsing": batch_tensor(item.get("target_parsing"), device),
        "source_hair_mask": batch_tensor(item.get("source_hair_mask"), device),
        "target_hair_mask": batch_tensor(item.get("target_hair_mask"), device),
        "source_face_reference": source_full_batch,
        "authoritative_hair_highres": None if completed_hair is None else normalize(completed_hair),
        "authoritative_target_highres": None if completed_hair is None else normalize(completed_hair),
        # In direct mode this compatibility field is deliberately not passed
        # as a post-decode residual candidate.
        "satd_background_highres": None,
        "direct_satd_pp_input": torch.ones(1, 1, 1, 1, device=device),
        "query_mask": batch_tensor(item.get("query_mask"), device),
        "source_ear_mask": source_native_alpha,
        "source_earring_object_mask": source_native_alpha,
        "source_instance_verified_left": batch_tensor(item.get("source_instance_verified_left"), device),
        "source_instance_verified_right": batch_tensor(item.get("source_instance_verified_right"), device),
        "source_native_left_alpha": batch_tensor(item.get("source_native_left_alpha"), device),
        "source_native_right_alpha": batch_tensor(item.get("source_native_right_alpha"), device),
        "earring_confident_mask": earring_learning_mask,
        "earring_supervision_mask": earring_learning_mask,
        "earring_reference": batch_image_01(item.get("earring_learning_reference"), device),
        "earring_mask_is_dataset": torch.tensor([1.0], device=device) if earring_learning_mask is not None else None,
        "earring_reference_is_dataset": torch.tensor([1.0], device=device) if item.get("earring_learning_reference") is not None else None,
        # ``presence_target`` is stored as one vector ``[3]`` per dataset
        # item, whereas image/mask tensors use ``[C,H,W]``.  Give the model
        # the same batch shape that DataLoader collation supplies.
        "presence_target": (
            item.get("presence_target").unsqueeze(0).to(device)
            if torch.is_tensor(item.get("presence_target"))
            and item.get("presence_target").ndim == 1
            else batch_tensor(item.get("presence_target"), device)
        ),
        "cleanup_masks": {
            key: batch_tensor(item[key], device)
            for key in ("M_remove", "M_remove_halo", "M_remove_tail", "M_remove_face", "M_remove_neck")
            if key in item
        },
        "revealed_skin_mask": batch_tensor(item.get("revealed_skin_mask"), device),
        "revealed_skin_seam_mask": batch_tensor(item.get("revealed_skin_seam_mask"), device),
        "source_visible_skin_reference_mask": batch_tensor(item.get("source_visible_skin_reference_mask"), device),
        "source_skin_valid_mask": batch_tensor(item.get("source_skin_valid_mask"), device),
    }
    with torch.no_grad():
        v6_s, v6_f, v6_aux = v6_model(
            normalize(source_256.unsqueeze(0).to(device)),
            normalize(dataset_target.unsqueeze(0).to(device)),
            target_mask,
            target_hair,
            **common_v6_kwargs,
        )
        # The model's 256px query resolver also stores a gated diagnostic mask;
        # restore the serialized SOURCE_NATIVE object alpha for the final
        # compositor without granting it a global verification flag.
        if source_native_alpha is not None:
            v6_aux["source_earring_object_mask"] = source_native_alpha
        for key in ("source_instance_verified_left", "source_instance_verified_right"):
            if common_v6_kwargs.get(key) is not None:
                v6_aux[key] = common_v6_kwargs[key]
        for key in ("source_native_left_alpha", "source_native_right_alpha"):
            if common_v6_kwargs.get(key) is not None:
                v6_aux[key] = common_v6_kwargs[key]
        for key, value in common_v6_kwargs["cleanup_masks"].items():
            v6_aux[key] = value
        # The author HairFast instance has been released above; instantiate
        # only its frozen StyleGAN owner for the live V6 decode.  Calling the
        # V6 renderer is important: it populates output_v19_source_alpha from
        # the real post-decode earring compositor.
        stylegan = Net(author_args)
        v6_decoded, v6_aux = v6_model.render_refined(
            stylegan.generator,
            v6_s,
            v6_f,
            v6_aux,
        )
        v6_pp_from_dataset_target = ((v6_decoded[0] + 1.0) * 0.5).clamp(0, 1).cpu()

    metrics = {
        "author_captured_vs_unpatched_final": error_metrics(
            author_baseline_final_captured,
            author_baseline_final,
        ),
        "author_pre_pp_vs_dataset_target": error_metrics(author_pre_pp_256, dataset_target),
        "author_pre_pp_vs_current_v6_target": error_metrics(author_pre_pp_256, current_target),
        "current_v6_target_vs_dataset_target": error_metrics(current_target, dataset_target),
        "author_vs_v6_S": error_metrics(author_s_cpu, v6_s.cpu()),
        "author_vs_v6_F": error_metrics(author_f_cpu, v6_f.cpu()),
        "author_vs_v6_pp_from_dataset_target": error_metrics(
            author_pp_from_dataset_target,
            v6_pp_from_dataset_target,
        ),
    }
    for label, image in (
        ("01_source_256", source_256),
        ("02_dataset_i_blend_256", dataset_target),
        ("03_author_i_blend_256", author_pre_pp_256),
        ("04_current_v6_i_blend_256", current_target),
        ("05_author_baseline_final", author_baseline_final),
        ("06_author_baseline_final_captured", author_baseline_final_captured),
        ("07_author_pp_from_author_i_blend", author_pp_from_author_target),
        ("08_author_pp_from_dataset_i_blend", author_pp_from_dataset_target),
        ("09_v6_pp_from_dataset_i_blend", v6_pp_from_dataset_target),
    ):
        save_image(image, output / f"{label}.png")

    # D: explicit earring diagnostics.  These are deliberately separate from
    # the RGB panels above: an all-black alpha means "no accepted object", not
    # a dark earring.  The three requested fields are kept in their own
    # coordinate spaces so a misplaced source alpha cannot be mistaken for a
    # failed target alignment.
    mask_stats = {}
    if source_native_alpha_item is not None:
        area, shape = save_mask(
            source_native_alpha_item,
            output / "10_source_native_earring_alpha.png",
            "source_native_earring_alpha",
        )
        mask_stats["source_native_earring_alpha"] = {"area_px": area, "shape": shape, "space": "SOURCE_NATIVE"}
    else:
        mask_stats["source_native_earring_alpha"] = {"present": False}
    if earring_learning_mask_item is not None:
        area, shape = save_mask(
            earring_learning_mask_item,
            output / "11_earring_learning_mask.png",
            "earring_learning_mask",
        )
        mask_stats["earring_learning_mask"] = {"area_px": area, "shape": shape, "space": "TARGET_CANONICAL"}
    else:
        mask_stats["earring_learning_mask"] = {"present": False}
    if source_structured_alpha_item is not None:
        area, shape = save_mask(
            source_structured_alpha_item,
            output / "10b_source_earring_structured_alpha.png",
            "source_earring_structured_alpha",
        )
        mask_stats["source_earring_structured_alpha"] = {
            "area_px": area,
            "shape": shape,
            "space": "SOURCE_NATIVE",
        }
    output_alpha = v6_aux.get("output_v19_source_alpha") if isinstance(v6_aux, dict) else None
    if torch.is_tensor(output_alpha):
        output_alpha_cpu = output_alpha.detach().cpu()
        area = float(output_alpha_cpu.sum().item())
        save_image(output_alpha_cpu.repeat(1, 3, 1, 1).clamp(0, 1), output / "12_output_v19_source_alpha.png")
        mask_stats["output_v19_source_alpha"] = {
            "area_px": area,
            "shape": tuple(output_alpha_cpu.shape[-2:]),
            "space": "TARGET_OUTPUT",
        }
    else:
        mask_stats["output_v19_source_alpha"] = {"present": False}
    # Per-side masks explain the two failure modes directly: the same object
    # in both source masks would be aligned twice, while a short source mask
    # means the long pendant was lost before target visibility was considered.
    for key, filename, space in (
        ("v6_source_selected_left_alpha", "13_v6_source_selected_left_alpha.png", "SOURCE_NATIVE"),
        ("v6_source_selected_right_alpha", "14_v6_source_selected_right_alpha.png", "SOURCE_NATIVE"),
        ("v6_target_aligned_left_alpha", "15_v6_target_aligned_left_alpha.png", "TARGET_NATIVE"),
        ("v6_target_aligned_right_alpha", "16_v6_target_aligned_right_alpha.png", "TARGET_NATIVE"),
        ("v6_source_left_label9_chain", "17_v6_source_left_label9_chain.png", "SOURCE_NATIVE"),
        ("v6_source_right_label9_chain", "18_v6_source_right_label9_chain.png", "SOURCE_NATIVE"),
        ("source_component_labels", "19_source_component_ids.png", "SOURCE_NATIVE"),
        ("source_selected_components", "20_source_accepted_component_ids.png", "SOURCE_NATIVE"),
        ("final_hole_alpha", "21_final_hole_alpha.png", "TARGET_OUTPUT"),
    ):
        value = v6_aux.get(key) if isinstance(v6_aux, dict) else None
        if torch.is_tensor(value):
            value_cpu = value.detach().cpu()
            save_image(value_cpu.repeat(1, 3, 1, 1).clamp(0, 1), output / filename)
            mask_stats[key] = {
                "area_px": float(value_cpu.sum().item()),
                "shape": tuple(value_cpu.shape[-2:]),
                "space": space,
            }
    for key in ("v6_source_duplicate_left_removed", "v6_source_duplicate_right_removed"):
        value = v6_aux.get(key) if isinstance(v6_aux, dict) else None
        if torch.is_tensor(value):
            mask_stats[key] = {"value": float(value.detach().cpu().amax().item())}
    for key in (
        "v6_target_left_shift_y",
        "v6_target_left_shift_x",
        "v6_target_right_shift_y",
        "v6_target_right_shift_x",
    ):
        value = v6_aux.get(key) if isinstance(v6_aux, dict) else None
        if torch.is_tensor(value):
            mask_stats[key] = {"value": float(value.detach().cpu().flatten()[0].item())}
    for key in (
        "alignment_raw_left_dx",
        "alignment_raw_left_dy",
        "alignment_raw_right_dx",
        "alignment_raw_right_dy",
        "alignment_valid_left",
        "alignment_valid_right",
        "fallback_zero_shift_used_left",
        "fallback_zero_shift_used_right",
        "target_lobe_hair_cover_ratio_left",
        "target_lobe_hair_cover_ratio_right",
        "source_root_component_id",
        "source_accepted_component_count",
        "source_rejected_component_count",
        "source_reject_reason",
    ):
        value = v6_aux.get(key) if isinstance(v6_aux, dict) else None
        if torch.is_tensor(value):
            mask_stats[key] = {
                "value": [float(item) for item in value.detach().cpu().flatten().tolist()]
            }
    for key in ("output_v19_target_left_visible_ear", "output_v19_target_right_visible_ear"):
        value = v6_aux.get(key) if isinstance(v6_aux, dict) else None
        if torch.is_tensor(value):
            save_image(
                as_mask_4d(value[0] if value.ndim == 4 else value, key).repeat(1, 3, 1, 1),
                output / f"{key}.png",
            )

    manifest = """01_source_256.png: source image after the author PP input resize.
02_dataset_i_blend_256.png: target cached in the current dataset part.
03_author_i_blend_256.png: formal author I_blend_256 for this exact triplet.
04_current_v6_i_blend_256.png: current V6 I_blend_256 for this exact triplet.
05_author_baseline_final.png: complete formal output of an untouched author call.
06_author_baseline_final_captured.png: author output from the call that recorded image 03.
07_author_pp_from_author_i_blend.png: author PP from image 03, fixed StyleGAN noise.
08_author_pp_from_dataset_i_blend.png: author PP from image 02, fixed StyleGAN noise.
09_v6_pp_from_dataset_i_blend.png: V6 PP from image 02, fixed StyleGAN noise.
10_source_native_earring_alpha.png: source-native extracted earring object alpha (white = source pixels eligible for recall).
11_earring_learning_mask.png: target-canonical training mask (white = target-frame earring supervision).
12_output_v19_source_alpha.png: actual final V6 compositor alpha after target-ear visibility gating and alignment.
10b_source_earring_structured_alpha.png: validated source-lobe structured instance used to complete native alpha.
13/14_v6_source_selected_*_alpha.png: final per-side source masks before target anchoring.
15/16_v6_target_aligned_*_alpha.png: each source side after its one target-lobe alignment.
output_v19_target_left_visible_ear.png / output_v19_target_right_visible_ear.png: per-side target-ear visibility gates.
metrics.txt also records raw/accepted alignment, source component counts/rejection
reason, target lobe hair-cover ratios, and zero-shift fallback diagnostics.
"""
    (output / "README.txt").write_text(manifest, encoding="utf-8")
    (output / "metrics.txt").write_text(
        "\n".join(f"{name}: {values}" for name, values in metrics.items())
        + "\n\nmask_stats:\n"
        + "\n".join(f"{name}: {values}" for name, values in mask_stats.items())
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved trace images and metrics to: {output}")
    for name, values in metrics.items():
        print(f"{name}: max_abs={values['max_abs']:.9g}, mean_abs={values['mean_abs']:.9g}")
    for name, values in mask_stats.items():
        print(f"{name}: {values}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace author and V6 PP outputs for one V6 dataset item")
    parser.add_argument("--dataset", type=Path, default=TRAIN_CONFIG["dataset"])
    parser.add_argument("--part", type=int, default=1)
    parser.add_argument("--item", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("output/pp_v6_trace/item7_direct_satd"))
    main(parser.parse_args())
