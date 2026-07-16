from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bicubic import BicubicDownSample


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class FringeMaskHead(nn.Module):
    def __init__(self, hidden_channels: int = 32, prior_weight: float = 0.8,
                 boundary_kernel: int = 9, roi_kernel: int = 25):
        super().__init__()
        self.prior_weight = prior_weight
        self.boundary_kernel = boundary_kernel
        self.roi_kernel = roi_kernel
        self.learned_head = nn.Sequential(
            ConvBlock(8, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            _zero_module(nn.Conv2d(hidden_channels, 2, kernel_size=1, stride=1, padding=0)),
        )

    @staticmethod
    def _to_binary_face_mask(face_mask: torch.Tensor) -> torch.Tensor:
        if face_mask.dim() == 3:
            face_mask = face_mask.unsqueeze(1)
        if torch.max(face_mask) > 1:
            return ((face_mask > 0) & (face_mask != 13)).float()
        return (face_mask > 0.5).float()

    @staticmethod
    def _to_binary_hair_mask(hair_mask: torch.Tensor) -> torch.Tensor:
        if hair_mask.dim() == 3:
            hair_mask = hair_mask.unsqueeze(1)
        return (hair_mask > 0.5).float()

    @staticmethod
    def _build_upper_face_band(face_region: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = face_region.shape
        upper_band = torch.zeros_like(face_region)

        for idx in range(batch):
            coords = torch.nonzero(face_region[idx, 0] > 0.5, as_tuple=False)
            if coords.numel() == 0:
                upper_band[idx, 0, : height // 2, width // 4: width - width // 4] = 1.0
                continue

            y_min = int(coords[:, 0].min().item())
            y_max = int(coords[:, 0].max().item()) + 1
            x_min = int(coords[:, 1].min().item())
            x_max = int(coords[:, 1].max().item()) + 1
            face_height = max(1, y_max - y_min)
            band_bottom = min(height, y_min + max(1, int(face_height * 0.58)))
            upper_band[idx, 0, y_min:band_bottom, x_min:x_max] = 1.0

        return upper_band

    @staticmethod
    def _morphological_ring(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, face_mask: torch.Tensor, target_hair_mask: torch.Tensor,
                source_image: torch.Tensor, blending_image: torch.Tensor) -> dict[str, torch.Tensor]:
        face_region = self._to_binary_face_mask(face_mask)
        target_hair = self._to_binary_hair_mask(target_hair_mask)

        upper_face_band = self._build_upper_face_band(face_region)
        boundary_prior = self._morphological_ring(target_hair, self.boundary_kernel) * upper_face_band
        roi_prior = F.max_pool2d(boundary_prior, kernel_size=self.roi_kernel, stride=1,
                                 padding=self.roi_kernel // 2)
        roi_prior = (roi_prior * face_region).clamp(0.0, 1.0)

        learned = self.learned_head(torch.cat([face_region, target_hair, source_image, blending_image], dim=1))
        learned_roi = torch.sigmoid(learned[:, 0:1])
        learned_boundary = torch.sigmoid(learned[:, 1:2])

        roi_mask = (self.prior_weight * roi_prior + (1.0 - self.prior_weight) * learned_roi).clamp(0.0, 1.0)
        boundary_mask = (self.prior_weight * boundary_prior +
                         (1.0 - self.prior_weight) * learned_boundary).clamp(0.0, 1.0)
        fringe_mask = torch.maximum(roi_mask, boundary_mask)

        return {
            'face_region': face_region,
            'upper_face_band': upper_face_band,
            'roi_prior': roi_prior,
            'boundary_prior': boundary_prior,
            'roi_mask': roi_mask,
            'boundary_mask': boundary_mask,
            'fringe_mask': fringe_mask,
        }


class FringeEncoder(nn.Module):
    def __init__(self, in_channels: int = 11, base_channels: int = 32):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels)
        self.down1 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up2 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.low_projection = nn.Conv2d(base_channels * 4, 128, kernel_size=1, stride=1, padding=0)
        self.high_projection = nn.Conv2d(base_channels, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, source_crop: torch.Tensor, reference_crop: torch.Tensor, blending_crop: torch.Tensor,
                roi_mask_crop: torch.Tensor, boundary_mask_crop: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([source_crop, reference_crop, blending_crop, roi_mask_crop, boundary_mask_crop], dim=1)
        stem = self.stem(x)
        down1 = self.down1(stem)
        down2 = self.down2(down1)
        bottleneck = self.bottleneck(down2)
        up1 = self.up1(bottleneck, down1)
        up2 = self.up2(up1, stem)

        return {
            'low_feature': self.low_projection(bottleneck),
            'high_feature': self.high_projection(up2),
        }


class FringeFusionBlock(nn.Module):
    def __init__(self, global_channels: int = 512, local_channels: int = 128, attn_channels: int = 128):
        super().__init__()
        self.query_projection = nn.Conv2d(global_channels, attn_channels, kernel_size=1, stride=1, padding=0)
        self.key_projection = nn.Conv2d(local_channels, attn_channels, kernel_size=1, stride=1, padding=0)
        self.value_projection = nn.Conv2d(local_channels, attn_channels, kernel_size=1, stride=1, padding=0)
        self.delta_projection = _zero_module(
            nn.Conv2d(attn_channels, global_channels, kernel_size=1, stride=1, padding=0)
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(global_channels + attn_channels + 1, global_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(global_channels // 2, global_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.scale = math.sqrt(attn_channels)

    def forward(self, global_feature: torch.Tensor, local_feature: torch.Tensor,
                fringe_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        target_size = local_feature.shape[-2:]
        global_low = F.adaptive_avg_pool2d(global_feature, output_size=target_size)

        query = self.query_projection(global_low).flatten(2).transpose(1, 2)
        key = self.key_projection(local_feature).flatten(2)
        value = self.value_projection(local_feature).flatten(2).transpose(1, 2)

        attention = torch.softmax(torch.bmm(query, key) / self.scale, dim=-1)
        attended = torch.bmm(attention, value).transpose(1, 2).reshape_as(self.value_projection(local_feature))

        fringe_mask_low = F.interpolate(fringe_mask, size=target_size, mode='bilinear', align_corners=False)
        gate = self.gate_head(torch.cat([global_low, attended, fringe_mask_low], dim=1))
        delta_low = self.delta_projection(attended) * gate * fringe_mask_low
        delta = F.interpolate(delta_low, size=global_feature.shape[-2:], mode='bilinear', align_corners=False)
        fused_feature = global_feature + delta

        return {
            'fused_feature': fused_feature,
            'attention_map': attention,
            'gate_map_low': gate,
            'delta_low': delta_low,
        }


class FringeResidualHead(nn.Module):
    def __init__(self, low_channels: int = 128, high_channels: int = 64, hidden_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            ConvBlock(low_channels + high_channels + 11, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            _zero_module(nn.Conv2d(hidden_channels, 4, kernel_size=1, stride=1, padding=0)),
        )

    def forward(self, low_feature: torch.Tensor, high_feature: torch.Tensor, source_crop: torch.Tensor,
                reference_crop: torch.Tensor, global_crop: torch.Tensor, roi_mask_crop: torch.Tensor,
                boundary_mask_crop: torch.Tensor) -> dict[str, torch.Tensor]:
        low_upsampled = F.interpolate(low_feature, size=high_feature.shape[-2:], mode='bilinear', align_corners=False)
        residual_inputs = torch.cat([
            low_upsampled,
            high_feature,
            source_crop,
            reference_crop,
            global_crop,
            roi_mask_crop,
            boundary_mask_crop,
        ], dim=1)
        residual_logits = self.head(residual_inputs)
        residual = 0.25 * torch.tanh(residual_logits[:, :3])
        alpha = torch.sigmoid(residual_logits[:, 3:4]) * torch.maximum(roi_mask_crop, boundary_mask_crop)
        return {
            'residual_crop': residual,
            'alpha_crop': alpha,
        }


class FringeRefinerV7(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.crop_size = int(getattr(opts, 'fringe_crop_size', 128))
        self.crop_padding = int(getattr(opts, 'fringe_crop_padding', 16))
        self.mask_head = FringeMaskHead(
            hidden_channels=int(getattr(opts, 'fringe_mask_channels', 32)),
            prior_weight=float(getattr(opts, 'fringe_mask_prior_weight', 0.8)),
            boundary_kernel=int(getattr(opts, 'fringe_boundary_kernel', 9)),
            roi_kernel=int(getattr(opts, 'fringe_roi_kernel', 25)),
        )
        self.encoder = FringeEncoder(
            in_channels=11,
            base_channels=int(getattr(opts, 'fringe_encoder_channels', 32)),
        )
        self.fusion = FringeFusionBlock(
            global_channels=512,
            local_channels=128,
            attn_channels=int(getattr(opts, 'fringe_attention_channels', 128)),
        )
        self.residual_head = FringeResidualHead(
            low_channels=128,
            high_channels=64,
            hidden_channels=int(getattr(opts, 'fringe_decoder_channels', 64)),
        )
        use_cuda = 'cuda' in str(getattr(opts, 'device', 'cuda')).lower()
        self.downsample_256 = BicubicDownSample(factor=4, cuda=use_cuda)

    @staticmethod
    def _boxes_from_mask(mask: torch.Tensor, padding: int) -> list[tuple[int, int, int, int]]:
        _, _, height, width = mask.shape
        boxes: list[tuple[int, int, int, int]] = []

        for idx in range(mask.shape[0]):
            coords = torch.nonzero(mask[idx, 0] > 0.05, as_tuple=False)
            if coords.numel() == 0:
                y0 = 0
                y1 = max(1, height // 2)
                x0 = width // 4
                x1 = width - width // 4
            else:
                y0 = max(0, int(coords[:, 0].min().item()) - padding)
                y1 = min(height, int(coords[:, 0].max().item()) + 1 + padding)
                x0 = max(0, int(coords[:, 1].min().item()) - padding)
                x1 = min(width, int(coords[:, 1].max().item()) + 1 + padding)

            if y1 <= y0:
                y1 = min(height, y0 + 1)
            if x1 <= x0:
                x1 = min(width, x0 + 1)
            boxes.append((y0, y1, x0, x1))

        return boxes

    def _crop_tensor(self, tensor: torch.Tensor, boxes: list[tuple[int, int, int, int]]) -> torch.Tensor:
        crops = []
        for idx, (y0, y1, x0, x1) in enumerate(boxes):
            crop = tensor[idx:idx + 1, :, y0:y1, x0:x1]
            crop = F.interpolate(crop, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
            crops.append(crop)
        return torch.cat(crops, dim=0)

    @staticmethod
    def _paste_tensor(crops: torch.Tensor, boxes: list[tuple[int, int, int, int]],
                      canvas_size: tuple[int, int]) -> torch.Tensor:
        batch, channels = crops.shape[:2]
        canvas = crops.new_zeros((batch, channels, canvas_size[0], canvas_size[1]))

        for idx, (y0, y1, x0, x1) in enumerate(boxes):
            crop = F.interpolate(crops[idx:idx + 1], size=(max(1, y1 - y0), max(1, x1 - x0)),
                                 mode='bilinear', align_corners=False)
            canvas[idx:idx + 1, :, y0:y1, x0:x1] = crop
        return canvas

    def forward(self, source_mask: torch.Tensor, target_hair_mask: torch.Tensor, source_image: torch.Tensor,
                reference_image: torch.Tensor, blending_image: torch.Tensor, global_style: torch.Tensor,
                global_feature: torch.Tensor, generator: nn.Module) -> dict[str, torch.Tensor]:
        mask_outputs = self.mask_head(source_mask, target_hair_mask, source_image, blending_image)
        fringe_mask = mask_outputs['fringe_mask']
        boxes = self._boxes_from_mask(fringe_mask, padding=self.crop_padding)

        source_crop = self._crop_tensor(source_image, boxes)
        reference_crop = self._crop_tensor(reference_image, boxes)
        blending_crop = self._crop_tensor(blending_image, boxes)
        roi_mask_crop = self._crop_tensor(mask_outputs['roi_mask'], boxes)
        boundary_mask_crop = self._crop_tensor(mask_outputs['boundary_mask'], boxes)

        encoder_outputs = self.encoder(
            source_crop=source_crop,
            reference_crop=reference_crop,
            blending_crop=blending_crop,
            roi_mask_crop=roi_mask_crop,
            boundary_mask_crop=boundary_mask_crop,
        )
        fusion_outputs = self.fusion(global_feature, encoder_outputs['low_feature'], fringe_mask)

        global_image, _ = generator(
            [global_style],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=fusion_outputs['fused_feature'],
        )
        global_image_256 = self.downsample_256(global_image)
        global_crop = self._crop_tensor(global_image_256, boxes)

        residual_outputs = self.residual_head(
            low_feature=encoder_outputs['low_feature'],
            high_feature=encoder_outputs['high_feature'],
            source_crop=source_crop,
            reference_crop=reference_crop,
            global_crop=global_crop,
            roi_mask_crop=roi_mask_crop,
            boundary_mask_crop=boundary_mask_crop,
        )
        residual_canvas_256 = self._paste_tensor(
            residual_outputs['residual_crop'],
            boxes,
            canvas_size=blending_image.shape[-2:],
        )
        alpha_canvas_256 = self._paste_tensor(
            residual_outputs['alpha_crop'],
            boxes,
            canvas_size=blending_image.shape[-2:],
        ).clamp(0.0, 1.0)

        refined_image_256 = (global_image_256 + residual_canvas_256 * alpha_canvas_256).clamp(-1.0, 1.0)
        refined_image_1024 = F.interpolate(refined_image_256, size=global_image.shape[-2:],
                                           mode='bilinear', align_corners=False)
        fringe_mask_1024 = F.interpolate(fringe_mask, size=global_image.shape[-2:],
                                         mode='bilinear', align_corners=False).clamp(0.0, 1.0)
        final_image = global_image * (1.0 - fringe_mask_1024) + refined_image_1024 * fringe_mask_1024

        return {
            'global_image': global_image,
            'final_image': final_image,
            'global_feature': global_feature,
            'fused_feature': fusion_outputs['fused_feature'],
            'global_image_256': global_image_256,
            'refined_image_256': refined_image_256,
            'fringe_mask_256': fringe_mask,
            'fringe_mask_1024': fringe_mask_1024,
            'residual_canvas_256': residual_canvas_256,
            'alpha_canvas_256': alpha_canvas_256,
            'boxes': boxes,
            **mask_outputs,
            **encoder_outputs,
            **fusion_outputs,
            **residual_outputs,
        }
