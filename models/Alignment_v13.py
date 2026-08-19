from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from models.Alignment import Alignment
from models.sean_codes.models.pix2pix_model import decode_sean, encode_sean
from utils.save_utils import save_gen_image, save_latents
from utils.spsa_precompute_v13 import load_prior_npz

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)


def _build_inline_spsa_wrapper():
    def _sigmoid_logit(value: float) -> float:
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

    def _ensure_4d_tensor(tensor: torch.Tensor | None, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _resize_map(tensor: torch.Tensor | None, size: tuple[int, int], *, mode: str = "bilinear") -> torch.Tensor | None:
        if tensor is None:
            return None
        if tuple(tensor.shape[-2:]) == tuple(size):
            return tensor
        align_corners = False if mode in {"bilinear", "bicubic"} else None
        kwargs = {"mode": mode}
        if align_corners is not None:
            kwargs["align_corners"] = align_corners
        return F.interpolate(tensor, size=size, **kwargs)

    def _make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
        groups = min(max_groups, int(channels))
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    def _ensure_boxes(
        boxes: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if boxes is None:
            return None
        if not torch.is_tensor(boxes):
            boxes = torch.as_tensor(boxes)
        boxes = boxes.to(device=device, dtype=torch.float32)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if boxes.ndim == 3 and boxes.size(1) == 1:
            boxes = boxes[:, 0]
        if boxes.ndim != 2 or boxes.size(-1) != 4:
            raise ValueError(f"ROI boxes must have shape [B, 4], got {tuple(boxes.shape)}")
        if boxes.size(0) == 1 and batch_size > 1:
            boxes = boxes.expand(batch_size, -1)
        if boxes.size(0) != batch_size:
            raise ValueError(f"Expected {batch_size} ROI boxes, got {boxes.size(0)}")
        return boxes

    def _box_from_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[0]
        active = torch.nonzero(mask > 0.5, as_tuple=False)
        if active.numel() == 0:
            return None
        y1 = active[:, 0].min()
        y2 = active[:, 0].max() + 1
        x1 = active[:, 1].min()
        x2 = active[:, 1].max() + 1
        return torch.stack([x1, y1, x2, y2]).float()

    def _scale_box_to_feature(
        box: torch.Tensor,
        *,
        full_hw: tuple[int, int],
        feat_hw: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        full_h, full_w = full_hw
        feat_h, feat_w = feat_hw
        if full_h <= 0 or full_w <= 0:
            return None

        x1, y1, x2, y2 = box.tolist()
        x1 = max(0.0, min(x1, full_w - 1))
        y1 = max(0.0, min(y1, full_h - 1))
        x2 = max(x1 + 1.0, min(x2, full_w))
        y2 = max(y1 + 1.0, min(y2, full_h))

        scale_x = feat_w / float(full_w)
        scale_y = feat_h / float(full_h)

        fx1 = int(math.floor(x1 * scale_x))
        fy1 = int(math.floor(y1 * scale_y))
        fx2 = int(math.ceil(x2 * scale_x))
        fy2 = int(math.ceil(y2 * scale_y))

        fx1 = max(0, min(fx1, feat_w - 1))
        fy1 = max(0, min(fy1, feat_h - 1))
        fx2 = max(fx1 + 1, min(fx2, feat_w))
        fy2 = max(fy1 + 1, min(fy2, feat_h))
        if fx2 <= fx1 or fy2 <= fy1:
            return None
        return fx1, fy1, fx2, fy2

    class ResidualConvBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                _make_group_norm(channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                _make_group_norm(channels),
            )
            self.act = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(x + self.block(x))

    class ShapePriorEncoder(nn.Module):
        def __init__(self, out_channels: int, in_channels: int = 5, hidden_channels: int = 128):
            super().__init__()
            hidden_channels = max(hidden_channels, out_channels // 2)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                _make_group_norm(hidden_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.res_blocks = nn.Sequential(
                ResidualConvBlock(hidden_channels),
                ResidualConvBlock(hidden_channels),
                ResidualConvBlock(hidden_channels),
            )
            self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        def forward(self, prior_maps: torch.Tensor) -> torch.Tensor:
            x = self.stem(prior_maps)
            x = self.res_blocks(x)
            return self.proj(x)

    class GlobalGateNet(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            )
            last_conv = self.net[-1]
            nn.init.constant_(last_conv.bias, -1.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.net(x))

    class _ROIDetailHead(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                _make_group_norm(hidden_channels),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualConvBlock(hidden_channels),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            )
            last_conv = self.net[-1]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class BangDetailHead(_ROIDetailHead):
        def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128):
            super().__init__(in_channels, out_channels, hidden_channels)

    class TailDetailHead(_ROIDetailHead):
        def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128):
            super().__init__(in_channels, out_channels, hidden_channels)

    class InlineSPSAWrapper(nn.Module):
        def __init__(
            self,
            feature_channels: int,
            prior_in_channels: int = 5,
            hidden_channels: int = 128,
            bang_roi_size: int = 8,
            tail_roi_size: int = 8,
            alpha_init: float = 0.4,
            beta_bang_init: float = 0.15,
            beta_tail_init: float = 0.15,
        ):
            super().__init__()
            self.feature_channels = feature_channels
            self.bang_roi_size = int(bang_roi_size)
            self.tail_roi_size = int(tail_roi_size)
            self.max_global_residual_scale = 0.60
            self.max_local_residual_scale = 0.15

            self.prior_encoder = ShapePriorEncoder(
                out_channels=feature_channels,
                in_channels=prior_in_channels,
                hidden_channels=hidden_channels,
            )
            self.global_gate = GlobalGateNet(
                in_channels=2 * feature_channels + 2,
                hidden_channels=hidden_channels,
            )
            detail_channels = 2 * feature_channels + 1
            self.bang_head = BangDetailHead(detail_channels, feature_channels, hidden_channels)
            self.tail_head = TailDetailHead(detail_channels, feature_channels, hidden_channels)

            self.alpha_logit = nn.Parameter(torch.tensor(_sigmoid_logit(alpha_init), dtype=torch.float32))
            self.beta_bang_logit = nn.Parameter(torch.tensor(_sigmoid_logit(beta_bang_init), dtype=torch.float32))
            self.beta_tail_logit = nn.Parameter(torch.tensor(_sigmoid_logit(beta_tail_init), dtype=torch.float32))

        def _roi_delta(
            self,
            *,
            feature_map: torch.Tensor,
            prior_map: torch.Tensor,
            roi_mask: torch.Tensor | None,
            roi_boxes: torch.Tensor | None,
            full_hw: tuple[int, int],
            head: nn.Module,
            output_size: int,
        ) -> torch.Tensor:
            batch_size, _, feat_h, feat_w = feature_map.shape
            delta_full = torch.zeros_like(feature_map)
            feat_hw = (feat_h, feat_w)

            mask_feature = _resize_map(roi_mask, feat_hw, mode="bilinear")
            if mask_feature is None:
                mask_feature = torch.zeros(batch_size, 1, feat_h, feat_w, device=feature_map.device, dtype=feature_map.dtype)

            for idx in range(batch_size):
                box = None if roi_boxes is None else roi_boxes[idx]
                if box is None or bool((box[2:] <= box[:2]).any()):
                    box = _box_from_mask(roi_mask[idx] if roi_mask is not None else None)
                if box is None:
                    continue
                feat_box = _scale_box_to_feature(box, full_hw=full_hw, feat_hw=feat_hw)
                if feat_box is None:
                    continue
                x1, y1, x2, y2 = feat_box
                roi_spec = torch.tensor([[0.0, float(x1), float(y1), float(x2), float(y2)]], device=feature_map.device)
                roi_feature = roi_align(
                    torch.cat(
                        [
                            feature_map[idx : idx + 1],
                            prior_map[idx : idx + 1],
                            mask_feature[idx : idx + 1],
                        ],
                        dim=1,
                    ),
                    roi_spec,
                    output_size=output_size,
                    spatial_scale=1.0,
                    aligned=True,
                )
                roi_mask_feature = roi_align(
                    mask_feature[idx : idx + 1],
                    roi_spec,
                    output_size=output_size,
                    spatial_scale=1.0,
                    aligned=True,
                )
                roi_delta = head(roi_feature) * roi_mask_feature
                region_h = max(1, y2 - y1)
                region_w = max(1, x2 - x1)
                roi_delta = F.interpolate(roi_delta, size=(region_h, region_w), mode="bilinear", align_corners=False)
                roi_mask_resized = F.interpolate(
                    roi_mask_feature,
                    size=(region_h, region_w),
                    mode="bilinear",
                    align_corners=False,
                )
                delta_full[idx : idx + 1, :, y1:y2, x1:x2] += roi_delta * roi_mask_resized
            return delta_full

        def forward(
            self,
            *,
            F_base: torch.Tensor,
            H_tgt: torch.Tensor | None,
            E_tgt: torch.Tensor | None,
            D_tgt: torch.Tensor | None,
            O_tgt: torch.Tensor | None,
            bang_box: torch.Tensor | None = None,
            tail_box: torch.Tensor | None = None,
            bang_mask: torch.Tensor | None = None,
            tail_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            batch_size, _, feat_h, feat_w = F_base.shape
            device = F_base.device
            dtype = F_base.dtype

            H_tgt = _ensure_4d_tensor(H_tgt, device=device, dtype=dtype)
            E_tgt = _ensure_4d_tensor(E_tgt, device=device, dtype=dtype)
            D_tgt = _ensure_4d_tensor(D_tgt, device=device, dtype=dtype)
            O_tgt = _ensure_4d_tensor(O_tgt, device=device, dtype=dtype)
            bang_mask = _ensure_4d_tensor(bang_mask, device=device, dtype=dtype)
            tail_mask = _ensure_4d_tensor(tail_mask, device=device, dtype=dtype)

            if H_tgt is None or E_tgt is None or D_tgt is None or O_tgt is None:
                zero_gate = torch.zeros(batch_size, 1, feat_h, feat_w, device=device, dtype=dtype)
                return F_base, {
                    "applied": False,
                    "F_prior": F_base.detach(),
                    "G_global": zero_gate,
                    "delta_bang": torch.zeros_like(F_base),
                    "delta_tail": torch.zeros_like(F_base),
                    "bang_mask_ds": zero_gate,
                    "tail_mask_ds": zero_gate,
                }

            if O_tgt.size(1) != 2:
                raise ValueError(f"O_tgt must have 2 channels [dx, dy], got {O_tgt.size(1)}")

            full_hw = tuple(H_tgt.shape[-2:])
            bang_box = _ensure_boxes(bang_box, batch_size=batch_size, device=device)
            tail_box = _ensure_boxes(tail_box, batch_size=batch_size, device=device)

            feat_hw = (feat_h, feat_w)
            prior_maps = torch.cat([H_tgt, E_tgt, D_tgt, O_tgt], dim=1)
            prior_maps = _resize_map(prior_maps, feat_hw, mode="bilinear")
            bang_mask_ds = _resize_map(
                bang_mask if bang_mask is not None else torch.zeros(batch_size, 1, *full_hw, device=device, dtype=dtype),
                feat_hw,
                mode="bilinear",
            )
            tail_mask_ds = _resize_map(
                tail_mask if tail_mask is not None else torch.zeros(batch_size, 1, *full_hw, device=device, dtype=dtype),
                feat_hw,
                mode="bilinear",
            )

            F_prior = self.prior_encoder(prior_maps)
            gate_input = torch.cat([F_base, F_prior, bang_mask_ds, tail_mask_ds], dim=1)
            G_global = self.global_gate(gate_input)

            alpha = torch.sigmoid(self.alpha_logit)
            beta_bang = torch.sigmoid(self.beta_bang_logit)
            beta_tail = torch.sigmoid(self.beta_tail_logit)

            global_residual = torch.tanh(F_prior - F_base)
            F_global = F_base + self.max_global_residual_scale * alpha * G_global * global_residual
            delta_bang = torch.tanh(self._roi_delta(
                feature_map=F_global,
                prior_map=F_prior,
                roi_mask=bang_mask,
                roi_boxes=bang_box,
                full_hw=full_hw,
                head=self.bang_head,
                output_size=self.bang_roi_size,
            ))
            delta_tail = torch.tanh(self._roi_delta(
                feature_map=F_global,
                prior_map=F_prior,
                roi_mask=tail_mask,
                roi_boxes=tail_box,
                full_hw=full_hw,
                head=self.tail_head,
                output_size=self.tail_roi_size,
            ))
            F_align_new = (
                F_global
                + self.max_local_residual_scale * beta_bang * delta_bang
                + self.max_local_residual_scale * beta_tail * delta_tail
            )

            aux = {
                "applied": True,
                "F_prior": F_prior,
                "G_global": G_global,
                "delta_bang": delta_bang,
                "delta_tail": delta_tail,
                "bang_mask_ds": bang_mask_ds,
                "tail_mask_ds": tail_mask_ds,
                "alpha": alpha.detach(),
                "beta_bang": beta_bang.detach(),
                "beta_tail": beta_tail.detach(),
            }
            return F_align_new, aux

    return InlineSPSAWrapper


try:
    from models.spsa_modules_v13 import SPSAWrapper
except (ModuleNotFoundError, ImportError):
    helper_path = ROOT_DIR / "models" / "spsa_modules_v13.py"
    if helper_path.is_file():
        spec = importlib.util.spec_from_file_location("models.spsa_modules_v13_local", helper_path)
        helper_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(helper_module)
        SPSAWrapper = helper_module.SPSAWrapper
    else:
        SPSAWrapper = _build_inline_spsa_wrapper()


class Alignment_v13(Alignment):
    def __init__(self, opts, latent_encoder=None, net=None):
        super().__init__(opts, latent_encoder=latent_encoder, net=net)
        self.spsa = SPSAWrapper(
            feature_channels=getattr(opts, "spsa_feature_channels", 512),
            hidden_channels=getattr(opts, "spsa_hidden_channels", 128),
            bang_roi_size=getattr(opts, "spsa_bang_roi_size", 8),
            tail_roi_size=getattr(opts, "spsa_tail_roi_size", 8),
            alpha_init=getattr(opts, "spsa_alpha_init", 0.4),
            beta_bang_init=getattr(opts, "spsa_beta_bang_init", 0.15),
            beta_tail_init=getattr(opts, "spsa_beta_tail_init", 0.15),
        ).to(self.opts.device)
        prior_dir = getattr(opts, "spsa_prior_dir", None)
        self.spsa_prior_dir = Path(prior_dir) if prior_dir else None
        self.spsa_require_priors = bool(getattr(opts, "spsa_require_priors", False))

    def load_spsa_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("spsa_state_dict", checkpoint)
        self.spsa.load_state_dict(state_dict, strict=False)

    def decode_alignment_image(self, latent_s_face: torch.Tensor, latent_f_align: torch.Tensor) -> torch.Tensor:
        image_tanh, _ = self.net.generator(
            [latent_s_face],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_f_align,
        )
        return ((image_tanh + 1) / 2).clamp(0, 1)

    def _prepare_prior_inputs(self, prior_like: Any) -> dict[str, torch.Tensor] | None:
        if prior_like is None:
            return None
        if isinstance(prior_like, (str, Path)):
            prior_like = load_prior_npz(prior_like)
        if not isinstance(prior_like, dict):
            raise TypeError(f"Unsupported SPSA prior payload: {type(prior_like)}")

        device = torch.device(self.opts.device)
        prepared: dict[str, torch.Tensor] = {}
        for key in ["H_tgt", "E_tgt", "D_tgt", "O_tgt", "bang_mask", "tail_mask", "bang_box", "tail_box"]:
            value = prior_like.get(key)
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            prepared[key] = value.to(device=device, dtype=torch.float32)
        return prepared or None

    def _resolve_prior_inputs(self, im_name: str, name_to_embed, **kwargs) -> dict[str, torch.Tensor] | None:
        candidates = [
            kwargs.get("spsa_prior"),
            kwargs.get("spsa_prior_path"),
            kwargs.get("spsa_priors", {}).get(im_name) if isinstance(kwargs.get("spsa_priors"), dict) else None,
            name_to_embed.get(im_name, {}).get("spsa_prior"),
            name_to_embed.get(im_name, {}).get("spsa_prior_path"),
        ]

        for candidate in candidates:
            prepared = self._prepare_prior_inputs(candidate)
            if prepared is not None:
                return prepared

        if self.spsa_prior_dir is not None:
            for stem in [im_name, Path(im_name).stem]:
                prior_path = self.spsa_prior_dir / f"{stem}.npz"
                if prior_path.is_file():
                    return self._prepare_prior_inputs(prior_path)

        if self.spsa_require_priors:
            raise FileNotFoundError(f"No SPSA priors were found for '{im_name}'.")
        return None

    def _compute_author_state(self, im_name1, im_name2, name_to_embed, **kwargs) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            img1_in = name_to_embed[im_name1]["image_256"]
            img2_in = name_to_embed[im_name2]["image_256"]
            latent_s_1 = name_to_embed[im_name1]["S"]
            latent_f_1 = name_to_embed[im_name1]["F"]
            latent_f_2 = name_to_embed[im_name2]["F"]

            if img1_in is img2_in:
                hair_mask_target = super().shape_module(
                    im_name1,
                    im_name2,
                    name_to_embed,
                    only_target=True,
                    **kwargs,
                )["HM_X"]
                return {
                    "latent_F_base": latent_f_1.detach(),
                    "latent_S_face": latent_s_1.detach(),
                    "HM_X": hair_mask_target.detach(),
                }

            inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = super().shape_module(
                im_name1,
                im_name2,
                name_to_embed,
                only_target=False,
                **kwargs,
            )

            images = torch.cat([img1_in, img2_in], dim=0)
            labels = torch.cat([inp_mask1, inp_mask2], dim=0)
            img1_code, img2_code = encode_sean(self.sean_model, images, labels)

            gen1_sean = decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask)
            gen2_sean = decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask)
            enc_imgs = self.latent_encoder([gen1_sean, gen2_sean])

            intermediate_align = enc_imgs["F"][0].unsqueeze(0)
            latent_inter = enc_imgs["W"][0].unsqueeze(0)
            latent_f_out_new = enc_imgs["F"][1].unsqueeze(0)
            latent_out = enc_imgs["W"][1].unsqueeze(0)

            masks = torch.cat(
                [
                    1 - (1 - hair_mask1) * (1 - hair_mask_target),
                    hair_mask_target,
                    hair_mask2 * hair_mask_target,
                ],
                dim=0,
            )
            dilate, erosion = self.dilate_erosion.mask(masks)
            free_mask = torch.stack([dilate[0], erosion[1], erosion[2]], dim=0)
            free_mask_down_32 = torch.nn.functional.interpolate(free_mask.float(), size=(32, 32), mode="bicubic")
            interpolation_low = 1 - free_mask_down_32

            latent_f_base = intermediate_align + interpolation_low[0] * (latent_f_1 - intermediate_align)
            latent_f_base = latent_f_out_new + interpolation_low[1] * (latent_f_base - latent_f_out_new)
            latent_f_base = latent_f_2 + interpolation_low[2] * (latent_f_base - latent_f_2)

            return {
                "latent_F_base": latent_f_base.detach(),
                "latent_S_face": latent_s_1.detach(),
                "latent_inter": latent_inter.detach(),
                "latent_out": latent_out.detach(),
                "latent_intermediate_align": intermediate_align.detach(),
                "latent_F_out_new": latent_f_out_new.detach(),
                "HM_X": hair_mask_target.detach(),
                "target_mask": target_mask.detach(),
                "gen1_sean": gen1_sean.detach(),
                "gen2_sean": gen2_sean.detach(),
            }

    def forward_train(self, im_name1, im_name2, name_to_embed, **kwargs) -> dict[str, Any]:
        author_state = self._compute_author_state(im_name1, im_name2, name_to_embed, **kwargs)
        prior_inputs = self._resolve_prior_inputs(im_name2, name_to_embed, **kwargs)
        if prior_inputs is None:
            latent_f_align = author_state["latent_F_base"]
            spsa_aux = {"applied": False}
        else:
            latent_f_align, spsa_aux = self.spsa(F_base=author_state["latent_F_base"], **prior_inputs)
        return author_state | {
            "latent_F_align": latent_f_align,
            "spsa_aux": spsa_aux,
            "spsa_priors": prior_inputs,
        }

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        outputs = self.forward_train(im_name1, im_name2, name_to_embed, **kwargs)
        latent_f_align = outputs["latent_F_align"]

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name

            if "gen1_sean" in outputs:
                save_gen_image(output_dir, "Align_v13", f"{im_name1}_{im_name2}_SEAN.png", outputs["gen1_sean"])
            if "gen2_sean" in outputs:
                save_gen_image(output_dir, "Align_v13", f"{im_name2}_{im_name1}_SEAN.png", outputs["gen2_sean"])

            if "latent_inter" in outputs and "latent_intermediate_align" in outputs:
                img1_e4e = self.net.generator(
                    [outputs["latent_inter"]],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=outputs["latent_intermediate_align"],
                )[0]
                save_gen_image(output_dir, "Align_v13", f"{im_name1}_{im_name2}_e4e.png", img1_e4e)

            if "latent_out" in outputs and "latent_F_out_new" in outputs:
                img2_e4e = self.net.generator(
                    [outputs["latent_out"]],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=outputs["latent_F_out_new"],
                )[0]
                save_gen_image(output_dir, "Align_v13", f"{im_name2}_{im_name1}_e4e.png", img2_e4e)

            gen_im, _ = self.net.generator(
                [name_to_embed[im_name1]["S"]],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_f_align,
            )
            save_gen_image(output_dir, "Align_v13", f"{im_name1}_{im_name2}_output.png", gen_im)
            save_latents(
                output_dir,
                "Align_v13",
                f"{im_name1}_{im_name2}_F.npz",
                latent_F_align=latent_f_align,
                latent_F_base=outputs["latent_F_base"],
            )

        return {
            "latent_F_align": latent_f_align,
            "latent_F_base": outputs["latent_F_base"],
            "HM_X": outputs["HM_X"],
            "spsa_aux": outputs["spsa_aux"],
        }
