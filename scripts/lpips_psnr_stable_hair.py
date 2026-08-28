"""Compute LPIPS and PSNR for the Stable-Hair frozen 3000-image reconstruction set.

Metric-protocol twin of ``scripts/lpips_psnr.py`` (same 256x256 resize, same
alex-net LPIPS weight, same per-row CSV + summary schema), so Stable-Hair and
HairFast recon numbers are directly comparable.  Only the User config section
differs: it points at the Stable-Hair reconstruction results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


# ========================= User config: edit here only ========================
USER_MANIFEST = Path("input/eval_pairs_v5/celeba_hq_recon_seed3407_3000.jsonl")
USER_IMAGE_ROOT = Path("/root/shared-nvme/HairFastGAN/celeba-1024")
USER_RESULT_DIR = Path("output/celeba_hq_stable_hair_recon/stable_hair_fp16_512/results")
USER_OUTPUT_CSV = Path("output/lpips_psnr_stable_hair_recon_3000.csv")
USER_IMAGE_SIZE = (256, 256)
USER_DEVICE = "cuda"
USER_CUDA_VISIBLE_DEVICES = "0"
USER_LPIPS_NET = "alex"
# LPIPS metric's official learned calibration weight. This is not a model
# checkpoint. Download it once; every comparison reuses it.
USER_LPIPS_WEIGHT = Path("losses/lpips/weights/v0.1/alex.pth")
USER_BATCH_SIZE = 16
USER_EXPECTED_SAMPLE_COUNT = 3000
USER_SKIP_INVALID_ROWS = True
# ===============================================================================

def _find_repo_root(start: Path) -> Path:
    # Works no matter which scripts subdirectory the file lives in: walk up to
    # the first ancestor that looks like the project root.
    for candidate in (start, *start.parents):
        if (candidate / "input").is_dir():
            return candidate
    return start.parents[1] if len(start.parents) > 1 else start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def check_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing reconstruction manifest: {path}. "
            "Run scripts/make_recon_manifest_stable_hair.py first."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"Manifest row {line_number} is not an object.")
            rows.append(row)
    if len(rows) != USER_EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"Expected {USER_EXPECTED_SAMPLE_COUNT} rows, found {len(rows)}.")

    seen_inputs: set[str] = set()
    seen_outputs: set[str] = set()
    for expected_index, row in enumerate(rows, start=1):
        if int(row.get("index", -1)) != expected_index:
            raise RuntimeError(f"Manifest index is not contiguous at row {expected_index}.")
        if str(row.get("mode", "")).lower() != "recon":
            raise RuntimeError(f"Manifest row {expected_index} is not mode=recon.")
        source = str(row.get("source_file", ""))
        if not source or source != str(row.get("shape_file", "")) or source != str(row.get("color_file", "")):
            raise RuntimeError(f"Row {expected_index} must use one image for source/shape/color.")
        if source in seen_inputs:
            raise RuntimeError(f"Duplicate source image in manifest: {source}")
        seen_inputs.add(source)
        output_file = Path(str(row.get("output_file", f"{expected_index:06d}.png")))
        if output_file.name != str(output_file) or output_file.suffix.lower() != ".png":
            raise RuntimeError(f"Unsafe output filename at row {expected_index}: {output_file}")
        if str(output_file) in seen_outputs:
            raise RuntimeError(f"Duplicate output filename: {output_file}")
        seen_outputs.add(str(output_file))
        row["source_file"] = source
        row["output_file"] = str(output_file)
    return rows


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(USER_IMAGE_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def psnr_values(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return [float("inf") if x == 0.0 else float(10.0 * math.log10(1.0 / x)) for x in mse.tolist()]


def make_lpips(device: torch.device, weight_path: Path):
    sys.path.insert(0, str(REPO_ROOT))
    from losses.lpips import PerceptualLoss  # type: ignore[import-not-found]

    options = {
        "model": "net-lin",
        "net": USER_LPIPS_NET,
        "use_gpu": device.type == "cuda",
        "gpu_ids": [0],
    }
    try:
        # Newer local LPIPS wrapper: support any explicitly configured weight.
        return PerceptualLoss(model_path=str(weight_path), **options).eval()
    except TypeError as error:
        # Original project wrapper: it always reads this standard location.
        standard_path = (
            REPO_ROOT / "losses/lpips/weights/v0.1" / f"{USER_LPIPS_NET}.pth"
        ).resolve()
        if weight_path != standard_path:
            raise RuntimeError(
                "This repository's LPIPS wrapper cannot accept a custom weight path. "
                f"Place the {USER_LPIPS_NET} weight at {standard_path}."
            ) from error
        return PerceptualLoss(**options).eval()


def add_rows(
    destination: list[dict[str, Any]],
    batch: list[tuple[dict[str, Any], Path, Path]],
    lpips: list[float],
    psnr: list[float],
) -> None:
    for (row, _result, _target), lpips_value, psnr_value in zip(batch, lpips, psnr):
        destination.append({
            "index": int(row["index"]),
            "output_file": str(row["output_file"]),
            "target_file": str(row["source_file"]),
            "lpips": lpips_value,
            "psnr": psnr_value,
        })


def main() -> None:
    if USER_CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES
    if USER_BATCH_SIZE <= 0:
        raise ValueError("USER_BATCH_SIZE must be positive.")
    device = torch.device(USER_DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("USER_DEVICE is CUDA, but PyTorch cannot see a CUDA device.")

    manifest = repo_path(USER_MANIFEST).resolve()
    image_root = USER_IMAGE_ROOT.expanduser().resolve()
    result_dir = repo_path(USER_RESULT_DIR).resolve()
    output_csv = repo_path(USER_OUTPUT_CSV).resolve()
    lpips_weight = repo_path(USER_LPIPS_WEIGHT).resolve()
    if not lpips_weight.is_file():
        raise FileNotFoundError(
            f"Missing LPIPS {USER_LPIPS_NET} weight: {lpips_weight}\n"
            "Download the official LPIPS weight once with:\n"
            "mkdir -p losses/lpips/weights/v0.1\n"
            "curl -L --retry 5 -o losses/lpips/weights/v0.1/alex.pth "
            "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest)

    valid: list[tuple[dict[str, Any], Path, Path]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        result_path = result_dir / str(row["output_file"])
        target_path = resolve_image(image_root, str(row["source_file"]))
        reason: str | None = None
        for label, path in (("result", result_path), ("original", target_path)):
            if not path.is_file():
                reason = f"{label} image is missing: {path}"
                break
            try:
                check_image(path)
            except Exception as error:  # noqa: BLE001
                reason = f"{label} image is unreadable: {path}: {error}"
                break
        if reason:
            if not USER_SKIP_INVALID_ROWS:
                raise FileNotFoundError(f"Row {row['index']}: {reason}")
            skipped.append({"index": int(row["index"]), "output_file": str(row["output_file"]), "reason": reason})
        else:
            valid.append((row, result_path, target_path))
    if not valid:
        raise RuntimeError("No valid reconstruction/result pairs remain.")

    print(f"Frozen rows: {len(rows)}; valid: {len(valid)}; skipped: {len(skipped)}", flush=True)
    print(f"LPIPS weight: {lpips_weight}", flush=True)
    metric = make_lpips(device, lpips_weight)
    metrics: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(valid), USER_BATCH_SIZE), desc="LPIPS + PSNR"):
            batch = valid[start : start + USER_BATCH_SIZE]
            try:
                pred = torch.stack([image_tensor(item[1]) for item in batch])
                target = torch.stack([image_tensor(item[2]) for item in batch])
                value = metric(pred.to(device) * 2.0 - 1.0, target.to(device) * 2.0 - 1.0)
                lpips = [float(x) for x in value.detach().reshape(len(batch), -1).mean(dim=1).cpu().tolist()]
                add_rows(metrics, batch, lpips, psnr_values(pred, target))
            except Exception as error:  # noqa: BLE001
                if not USER_SKIP_INVALID_ROWS:
                    raise RuntimeError(f"Metric batch failed: {error}") from error
                for row, result_path, target_path in batch:
                    try:
                        pred = image_tensor(result_path).unsqueeze(0)
                        target = image_tensor(target_path).unsqueeze(0)
                        value = metric(pred.to(device) * 2.0 - 1.0, target.to(device) * 2.0 - 1.0)
                        add_rows(metrics, [(row, result_path, target_path)], [float(value.detach().mean().cpu().item())], psnr_values(pred, target))
                    except Exception as row_error:  # noqa: BLE001
                        skipped.append({"index": int(row["index"]), "output_file": str(row["output_file"]), "reason": f"metric failed: {row_error}"})

    metrics.sort(key=lambda item: int(item["index"]))
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "output_file", "target_file", "lpips", "psnr"])
        writer.writeheader()
        writer.writerows(metrics)
    lpips_values = [float(item["lpips"]) for item in metrics]
    finite_psnr = [float(item["psnr"]) for item in metrics if math.isfinite(float(item["psnr"]))]
    summary = {
        "task": "reconstruction",
        "model": "stable_hair",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "image_root": str(image_root),
        "result_dir": str(result_dir),
        "image_size": list(USER_IMAGE_SIZE),
        "lpips_net": USER_LPIPS_NET,
        "lpips_weight": str(lpips_weight),
        "lpips_weight_sha256": sha256(lpips_weight),
        "frozen_rows": len(rows),
        "valid_rows": len(metrics),
        "skipped_rows": len(skipped),
        "mean_lpips": float(np.mean(lpips_values)),
        "std_lpips": float(np.std(lpips_values)),
        "mean_psnr": float(np.mean(finite_psnr)) if finite_psnr else None,
        "std_psnr": float(np.std(finite_psnr)) if finite_psnr else None,
    }
    summary_path = output_csv.with_name(output_csv.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if skipped:
        output_csv.with_name(output_csv.stem + "_skipped.json").write_text(json.dumps(skipped, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Valid samples: {len(metrics)}")
    print(f"Skipped samples: {len(skipped)}")
    print(f"Mean LPIPS: {summary['mean_lpips']:.6f}")
    print("Mean PSNR: n/a" if summary["mean_psnr"] is None else f"Mean PSNR: {summary['mean_psnr']:.6f} dB")
    print(f"CSV: {output_csv}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
