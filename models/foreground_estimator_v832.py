"""Frozen PyMatting multi-level foreground/background estimation for V2.32."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch


class V232ForegroundError(RuntimeError):
    pass


class ForegroundEstimatorV832:
    def __init__(self, *, roi_padding: int = 64, cache_dir: str | Path | None = None,
                 estimator=None):
        self.roi_padding = int(roi_padding)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.estimator = estimator
        self.backend_name = "injected" if estimator is not None else "PYMATTING_MULTILEVEL"
        if self.estimator is None:
            try:
                from pymatting import estimate_foreground_ml
            except ImportError as exc:
                raise V232ForegroundError("V232_FOREGROUND_DEPENDENCY_MISSING") from exc
            self.estimator = estimate_foreground_ml

    def config_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "roi": True,
            "roi_padding": self.roi_padding,
            "regularization": 1e-5,
            "n_small_iterations": 10,
            "n_big_iterations": 2,
            "small_size": 32,
            "return_background": True,
            "gradient_weight": 1.0,
            "training": False,
        }

    def _bbox(self, alpha: torch.Tensor) -> tuple[int, int, int, int]:
        points = (alpha > 0.01).nonzero(as_tuple=False)
        if points.numel() == 0:
            return 0, alpha.shape[-2], 0, alpha.shape[-1]
        y0 = max(0, int(points[:, -2].min().item()) - self.roi_padding)
        y1 = min(alpha.shape[-2], int(points[:, -2].max().item()) + self.roi_padding + 1)
        x0 = max(0, int(points[:, -1].min().item()) - self.roi_padding)
        x1 = min(alpha.shape[-1], int(points[:, -1].max().item()) + self.roi_padding + 1)
        return y0, y1, x0, x1

    @staticmethod
    def _key(image: torch.Tensor, alpha: torch.Tensor) -> str:
        digest = hashlib.sha1()
        digest.update(image.detach().float().cpu().numpy().tobytes())
        digest.update(alpha.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()

    def _run_one(self, image: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        started = time.perf_counter()
        height, width = image.shape[-2:]
        y0, y1, x0, x1 = self._bbox(alpha)
        key = self._key(image, alpha)
        cache_path = self.cache_dir / f"{key}.npz" if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            foreground = torch.from_numpy(cached["foreground"]).to(device=image.device, dtype=image.dtype)
            background = torch.from_numpy(cached["background"]).to(device=image.device, dtype=image.dtype)
            return foreground, background, {
                "roi": [y0, y1, x0, x1], "cache_hit": True,
                "seconds": time.perf_counter() - started, "key": key,
            }
        image_np = image[..., y0:y1, x0:x1].detach().float().cpu().numpy()[0]
        alpha_np = alpha[..., y0:y1, x0:x1].detach().float().cpu().numpy()[0, 0]
        image_np = np.transpose(image_np, (1, 2, 0)).astype(np.float64, copy=False)
        alpha_np = alpha_np.astype(np.float64, copy=False)
        try:
            foreground_np, background_np = self.estimator(
                image_np, alpha_np, regularization=1e-5,
                n_small_iterations=10, n_big_iterations=2, small_size=32,
                return_background=True, gradient_weight=1.0,
            )
        except TypeError:
            foreground_np, background_np = self.estimator(
                image_np, alpha_np, regularization=1e-5,
                n_small_iterations=10, n_big_iterations=2, small_size=32,
                return_background=True,
            )
        foreground_np = np.asarray(foreground_np, dtype=np.float32)
        background_np = np.asarray(background_np, dtype=np.float32)
        if foreground_np.shape != image_np.shape or background_np.shape != image_np.shape:
            raise V232ForegroundError("V232_FOREGROUND_BACKEND_FAIL: invalid F/B shape")
        raw_foreground = foreground_np.copy()
        raw_background = background_np.copy()
        clip_fraction = float(
            (np.logical_or(raw_foreground < 0, raw_foreground > 1).mean()
             + np.logical_or(raw_background < 0, raw_background > 1).mean()) / 2.0
        )
        foreground = torch.zeros_like(image)
        background = image.clone()
        foreground_roi = torch.from_numpy(
            np.transpose(foreground_np, (2, 0, 1)).copy()
        ).unsqueeze(0).to(device=image.device, dtype=image.dtype)
        background_roi = torch.from_numpy(
            np.transpose(background_np, (2, 0, 1)).copy()
        ).unsqueeze(0).to(device=image.device, dtype=image.dtype)
        foreground[..., y0:y1, x0:x1] = foreground_roi
        background[..., y0:y1, x0:x1] = background_roi
        foreground = foreground.clamp(0, 1)
        background = background.clamp(0, 1)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                foreground=foreground.detach().cpu().numpy(),
                background=background.detach().cpu().numpy(),
            )
        return foreground, background, {
            "roi": [y0, y1, x0, x1], "cache_hit": False,
            "seconds": time.perf_counter() - started, "key": key,
            "raw_clip_fraction": clip_fraction,
        }

    @torch.inference_mode()
    def __call__(self, *, image_rgb_1024: torch.Tensor, alpha_hr: torch.Tensor,
                 return_aux: bool = False):
        if image_rgb_1024.size(0) != 1:
            foregrounds, backgrounds, aux_rows = [], [], []
            for image, alpha in zip(image_rgb_1024, alpha_hr):
                foreground, background, aux = self._run_one(image[None], alpha[None])
                foregrounds.append(foreground); backgrounds.append(background); aux_rows.append(aux)
            foreground = torch.cat(foregrounds); background = torch.cat(backgrounds)
            aux = {"rows": aux_rows, "roi": [row["roi"] for row in aux_rows],
                   "cache_hit": all(row["cache_hit"] for row in aux_rows),
                   "seconds": sum(float(row["seconds"]) for row in aux_rows)}
        else:
            foreground, background, aux = self._run_one(image_rgb_1024, alpha_hr)
        recon = alpha_hr * foreground + (1.0 - alpha_hr) * background
        if not torch.isfinite(foreground).all() or not torch.isfinite(background).all():
            raise V232ForegroundError("V232_FOREGROUND_BACKEND_FAIL: NaN or Inf")
        if not return_aux:
            return foreground, background
        aux.update({
            "foreground_rgb": foreground, "background_rgb": background,
            "reconstruction_rgb": recon,
            "reconstruction_error": (recon - image_rgb_1024).abs(),
            "foreground_min": foreground.amin(dim=(1, 2, 3)),
            "foreground_max": foreground.amax(dim=(1, 2, 3)),
            "background_min": background.amin(dim=(1, 2, 3)),
            "background_max": background.amax(dim=(1, 2, 3)),
        })
        return foreground, background, aux


__all__ = ["ForegroundEstimatorV832", "V232ForegroundError"]
