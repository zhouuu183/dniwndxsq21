"""Frozen local ViTMatte-S backend for V2.31."""

from __future__ import annotations

from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from models.trimap_builder_v831 import TrimapBuilderV831


class V231MattingError(RuntimeError):
    pass


class HairMattingV831(nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device,
        inner_width: int = 8,
        outer_width: int = 8,
        face_contact_extra_inner: int = 4,
        max_trimap_hole_area: int = 16,
        processor=None,
        model=None,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.trimap_builder = TrimapBuilderV831(
            inner_width=inner_width,
            outer_width=outer_width,
            face_contact_extra_inner=face_contact_extra_inner,
            max_trimap_hole_area=max_trimap_hole_area,
        )
        self.processor = processor
        self.model = model
        self.backend_name = "injected"
        if self.processor is None or self.model is None:
            self._load_local_backend()
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)

    def _load_local_backend(self) -> None:
        try:
            from transformers import AutoImageProcessor, VitMatteForImageMatting
            from transformers.utils import is_torch_available
        except ImportError as exc:
            raise V231MattingError("V231_MATTING_DEPENDENCY_MISSING") from exc
        if not is_torch_available():
            raise V231MattingError(
                "V231_MATTING_BACKEND_FAIL: Transformers disabled PyTorch; "
                "install requirements_v231_matting.txt (Transformers 4.x)"
            )
        if not self.model_path.exists():
            raise V231MattingError("V231_MATTING_CHECKPOINT_MISSING")
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model = VitMatteForImageMatting.from_pretrained(
                str(self.model_path), local_files_only=True
            )
        except Exception as exc:
            raise V231MattingError(f"V231_MATTING_BACKEND_FAIL: {exc}") from exc
        self.backend_name = "hustvl/vitmatte-small-composition-1k"

    def config_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "model_path": str(self.model_path),
            "local_files_only": True,
            "training": False,
            "requires_grad": False,
            **self.trimap_builder.config_dict(),
        }

    def _processor_inputs(self, image: torch.Tensor, trimap: torch.Tensor) -> dict:
        images = [
            item.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            for item in image
        ]
        trimaps = [
            item[0].detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
            for item in trimap
        ]
        try:
            values = self.processor(images=images, trimaps=trimaps, return_tensors="pt")
        except TypeError:
            values = self.processor(images, trimaps=trimaps, return_tensors="pt")
        return {key: value.to(self.device) for key, value in values.items()}

    def _predict_alpha(self, image: torch.Tensor, trimap: torch.Tensor) -> torch.Tensor:
        outputs = self.model(**self._processor_inputs(image, trimap))
        alpha_pred = getattr(outputs, "alphas", None)
        if alpha_pred is None:
            alpha_pred = getattr(outputs, "alpha", None)
        if alpha_pred is None:
            raise V231MattingError("V231_MATTING_BACKEND_FAIL: missing alpha output")
        if alpha_pred.dim() == 3:
            alpha_pred = alpha_pred.unsqueeze(1)
        return F.interpolate(
            alpha_pred.float(), size=image.shape[-2:], mode="bilinear", align_corners=False
        ).to(device=image.device, dtype=image.dtype).clamp(0, 1)

    def _predict_roi_after_oom(
        self,
        image: torch.Tensor,
        trimap: torch.Tensor,
        coarse_hard: torch.Tensor,
        padding: int = 64,
    ) -> torch.Tensor:
        if image.size(0) != 1:
            raise V231MattingError(
                "V231_MATTING_BACKEND_FAIL: ROI OOM fallback requires batch size 1"
            )
        points = (coarse_hard[0, 0] > 0.5).nonzero(as_tuple=False)
        if points.numel() == 0:
            raise V231MattingError("V231_MATTING_BACKEND_FAIL: empty hair ROI")
        height, width = image.shape[-2:]
        y0 = max(0, int(points[:, 0].min().item()) - padding)
        y1 = min(height, int(points[:, 0].max().item()) + padding + 1)
        x0 = max(0, int(points[:, 1].min().item()) - padding)
        x1 = min(width, int(points[:, 1].max().item()) + padding + 1)
        roi_alpha = self._predict_alpha(
            image[..., y0:y1, x0:x1], trimap[..., y0:y1, x0:x1]
        )
        alpha = torch.zeros_like(image[:, :1])
        alpha[..., y0:y1, x0:x1] = roi_alpha
        return alpha

    @torch.inference_mode()
    def forward(
        self,
        *,
        pp_rgb_1024: torch.Tensor,
        target_hair_mask_256: torch.Tensor,
        source_face_mask_256: torch.Tensor,
        source_skin_mask_256: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        del source_skin_mask_256
        output_size = pp_rgb_1024.shape[-2:]
        trimap, trimap_aux = self.trimap_builder(
            target_hair_mask_256,
            source_face_mask_256,
            output_size=output_size,
            return_aux=True,
        )
        started = time.perf_counter()
        inference_mode = "FULL_1024"
        try:
            alpha_pred = self._predict_alpha(pp_rgb_1024.clamp(0, 1), trimap)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            alpha_pred = self._predict_roi_after_oom(
                pp_rgb_1024.clamp(0, 1), trimap, trimap_aux["coarse_hard"]
            )
            inference_mode = "HAIR_ROI_PLUS_64_AFTER_OOM"
        sure_fg = trimap_aux["sure_fg"].to(alpha_pred)
        sure_bg = trimap_aux["sure_bg"].to(alpha_pred)
        alpha_hr = torch.where(
            sure_fg > 0.5,
            torch.ones_like(alpha_pred),
            torch.where(sure_bg > 0.5, torch.zeros_like(alpha_pred), alpha_pred),
        )
        if alpha_hr.shape != pp_rgb_1024[:, :1].shape or not torch.isfinite(alpha_hr).all():
            raise V231MattingError("V231_MATTING_BACKEND_FAIL: invalid alpha")
        if not return_aux:
            return alpha_hr
        unknown = trimap_aux["unknown"].to(alpha_hr)
        fractional = ((alpha_hr > 0.01) & (alpha_hr < 0.99)).to(alpha_hr.dtype)
        return alpha_hr, {
            **{key: value.to(alpha_hr) for key, value in trimap_aux.items()},
            "alpha_pred": alpha_pred,
            "alpha_hr": alpha_hr,
            "inference_mode": inference_mode,
            "fractional_alpha": fractional,
            "inference_seconds": torch.tensor(
                time.perf_counter() - started, device=alpha_hr.device
            ),
            "sure_fg_error": ((alpha_hr - 1.0).abs() * sure_fg).amax(),
            "sure_bg_error": (alpha_hr.abs() * sure_bg).amax(),
            "unknown_fractional_fraction": (
                (fractional * unknown).flatten(1).sum(dim=1)
                / unknown.flatten(1).sum(dim=1).clamp_min(1.0)
            ),
        }


__all__ = ["HairMattingV831", "V231MattingError"]
