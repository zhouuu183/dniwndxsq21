import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T


def rgb01(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(0.0, 1.0)


def image_from_norm(image: torch.Tensor) -> torch.Tensor:
    return ((image + 1.0) * 0.5).clamp(0.0, 1.0)


def gray01(image: torch.Tensor) -> torch.Tensor:
    rgb = rgb01(image)
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def rgb01_to_lab_ab(image: torch.Tensor) -> torch.Tensor:
    rgb = rgb01(image)
    rgb = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )

    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    x = x / 0.95047
    z = z / 1.08883

    delta = 6.0 / 29.0

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta**3, t.pow(1.0 / 3.0), t / (3.0 * delta**2) + 4.0 / 29.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([a, b], dim=1)


def rgb01_to_hsv(image: torch.Tensor) -> torch.Tensor:
    rgb = rgb01(image)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)
    delta = maxc - minc

    sat = delta / maxc.clamp(min=1e-6)
    sat = torch.where(maxc > 0, sat, torch.zeros_like(sat))

    hue = torch.zeros_like(maxc)
    red_mask = (maxc == r) & (delta > 1e-6)
    green_mask = (maxc == g) & (delta > 1e-6)
    blue_mask = (maxc == b) & (delta > 1e-6)

    hue = torch.where(red_mask, ((g - b) / delta.clamp(min=1e-6)) % 6.0, hue)
    hue = torch.where(green_mask, ((b - r) / delta.clamp(min=1e-6)) + 2.0, hue)
    hue = torch.where(blue_mask, ((r - g) / delta.clamp(min=1e-6)) + 4.0, hue)
    hue = hue / 6.0
    return torch.cat([hue, sat, maxc], dim=1)


def _masked_histogram(values: torch.Tensor, mask: torch.Tensor, bins: int, value_min: float, value_max: float) -> torch.Tensor:
    values = values.view(values.size(0), -1)
    mask = mask.view(mask.size(0), -1) > 0.5
    histograms = []
    for batch_idx in range(values.size(0)):
        current = values[batch_idx][mask[batch_idx]]
        if current.numel() == 0:
            hist = torch.zeros(bins, device=values.device, dtype=values.dtype)
        else:
            hist = torch.histc(current, bins=bins, min=value_min, max=value_max)
            hist = hist / hist.sum().clamp(min=1.0)
        histograms.append(hist)
    return torch.stack(histograms, dim=0)


def _masked_quantiles(values: torch.Tensor, mask: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    values = values.view(values.size(0), -1)
    mask = mask.view(mask.size(0), -1) > 0.5
    outputs = []
    for batch_idx in range(values.size(0)):
        current = values[batch_idx][mask[batch_idx]]
        if current.numel() == 0:
            result = torch.zeros_like(quantiles, device=values.device, dtype=values.dtype)
        else:
            result = torch.quantile(current, quantiles)
        outputs.append(result)
    return torch.stack(outputs, dim=0)


class ChromaTokenEncoder_v17(nn.Module):
    def __init__(self, token_count: int = 12, hist_bins: int = 8, token_dim: int = 512):
        super().__init__()
        self.token_count = token_count
        self.hist_bins = hist_bins
        self.quantiles = torch.tensor([0.25, 0.50, 0.75], dtype=torch.float32)

        # feature_vector = [ab histograms, hue histogram, sat quantiles, hue quantiles, dark/mid/highlight stats]
        feature_dim = hist_bins * 2 + 6 + self.quantiles.numel() * 2 + 9
        self.summary_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, token_dim),
        )
        self.layer_mlp = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, token_count * token_dim),
        )

    def _segment_stats(
        self,
        ab: torch.Tensor,
        sat: torch.Tensor,
        luma: torch.Tensor,
        hair_mask: torch.Tensor,
    ) -> torch.Tensor:
        segments = []
        thresholds = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]
        for low, high in thresholds:
            segment_mask = (hair_mask > 0.5) & (luma >= low) & (luma < high)
            segment_mask = segment_mask.float()
            denom = segment_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            mean_a = (ab[:, 0:1] * segment_mask).sum(dim=(-2, -1), keepdim=True) / denom
            mean_b = (ab[:, 1:2] * segment_mask).sum(dim=(-2, -1), keepdim=True) / denom
            mean_s = (sat * segment_mask).sum(dim=(-2, -1), keepdim=True) / denom
            segments.append(torch.cat([mean_a, mean_b, mean_s], dim=1).flatten(1))
        return torch.cat(segments, dim=1)

    def forward(self, hair_image_norm: torch.Tensor, hair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hair_rgb = image_from_norm(hair_image_norm)
        hair_mask = hair_mask.float().clamp(0, 1)
        lab_ab = rgb01_to_lab_ab(hair_rgb)
        hsv = rgb01_to_hsv(hair_rgb)
        hue = hsv[:, 0:1]
        sat = hsv[:, 1:2]
        luma = gray01(hair_rgb)

        quantiles = self.quantiles.to(hair_rgb.device, hair_rgb.dtype)
        hist_a = _masked_histogram(lab_ab[:, 0:1], hair_mask, self.hist_bins, -110.0, 110.0)
        hist_b = _masked_histogram(lab_ab[:, 1:2], hair_mask, self.hist_bins, -110.0, 110.0)
        hist_h = _masked_histogram(hue, hair_mask, 6, 0.0, 1.0)
        sat_q = _masked_quantiles(sat, hair_mask, quantiles)
        hue_q = _masked_quantiles(hue, hair_mask, quantiles)
        segment_stats = self._segment_stats(lab_ab, sat, luma, hair_mask)

        feature_vector = torch.cat([hist_a, hist_b, hist_h, sat_q, hue_q, segment_stats], dim=1)
        summary_token = self.summary_mlp(feature_vector)
        layer_tokens = self.layer_mlp(feature_vector).view(hair_rgb.size(0), self.token_count, -1)

        return summary_token, layer_tokens, {
            "feature_vector": feature_vector,
            "hist_a": hist_a,
            "hist_b": hist_b,
            "hist_h": hist_h,
            "sat_q": sat_q,
            "hue_q": hue_q,
            "segment_stats": segment_stats,
        }


class MaskGate_v17(nn.Module):
    def __init__(self, token_count: int = 12, token_dim: int = 512):
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.mask_context = nn.Sequential(
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, token_dim),
        )
        self.feature_gate = nn.Conv2d(64, 1, kernel_size=1)
        nn.init.zeros_(self.feature_gate.weight)
        nn.init.constant_(self.feature_gate.bias, -1.0)

    def forward(
        self,
        safe_mask: torch.Tensor,
        lock_mask: torch.Tensor,
        align_hair_mask: torch.Tensor,
        satd_luma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        safe_32 = F.interpolate(safe_mask.float(), size=(32, 32), mode="nearest")
        lock_32 = F.interpolate(lock_mask.float(), size=(32, 32), mode="nearest")
        hair_32 = F.interpolate(align_hair_mask.float(), size=(32, 32), mode="nearest")
        luma_32 = F.interpolate(satd_luma.float(), size=(32, 32), mode="bilinear", align_corners=False)

        gate_feat = self.encoder(torch.cat([safe_32, lock_32, hair_32, luma_32], dim=1))
        pooled = gate_feat.mean(dim=(-2, -1))
        mask_context = self.mask_context(pooled)

        color_region_32 = torch.maximum(safe_32, 0.35 * hair_32 * (1.0 - lock_32)).clamp(0, 1)
        safe_ratio = color_region_32.mean(dim=(-2, -1), keepdim=False).sqrt().clamp(min=0.30, max=1.0).unsqueeze(1)
        feature_gate = (0.30 + 0.70 * torch.sigmoid(self.feature_gate(gate_feat))) * color_region_32
        return feature_gate, mask_context, {
            "safe_32": safe_32,
            "lock_32": lock_32,
            "hair_32": hair_32,
            "luma_32": luma_32,
            "color_region_32": color_region_32,
            "safe_ratio": safe_ratio,
        }


class FeatureDeltaPredictor_v17(nn.Module):
    def __init__(self, context_dim: int = 512):
        super().__init__()
        self.align_proj = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(128 + 128 + 4, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Conv2d(256, 512, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        align_f: torch.Tensor,
        context_token: torch.Tensor,
        safe_mask: torch.Tensor,
        lock_mask: torch.Tensor,
        align_hair_mask: torch.Tensor,
        satd_luma: torch.Tensor,
    ) -> torch.Tensor:
        safe_32 = F.interpolate(safe_mask.float(), size=align_f.shape[-2:], mode="nearest")
        lock_32 = F.interpolate(lock_mask.float(), size=align_f.shape[-2:], mode="nearest")
        hair_32 = F.interpolate(align_hair_mask.float(), size=align_f.shape[-2:], mode="nearest")
        luma_32 = F.interpolate(satd_luma.float(), size=align_f.shape[-2:], mode="bilinear", align_corners=False)

        align_feat = self.align_proj(align_f)
        context_feat = self.context_proj(context_token).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, align_f.size(2), align_f.size(3))
        feat = self.body(torch.cat([align_feat, context_feat, safe_32, lock_32, hair_32, luma_32], dim=1))
        return 0.35 * torch.tanh(self.output(feat))


class ChromaFeatureAffine_v17(nn.Module):
    def __init__(self, context_dim: int = 512, channels: int = 512):
        super().__init__()
        self.channels = channels
        self.to_affine = nn.Sequential(
            nn.Linear(context_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, channels * 2),
        )
        nn.init.normal_(self.to_affine[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.to_affine[-1].bias)

    def forward(self, align_f: torch.Tensor, context_token: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_affine(context_token).view(align_f.size(0), 2, self.channels, 1, 1).unbind(dim=1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        mean = align_f.mean(dim=(-2, -1), keepdim=True)
        std = align_f.std(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
        align_norm = (align_f - mean) / std
        return 0.18 * (gamma * align_norm + beta)


class SIDBlendingModel_v17(nn.Module):
    def __init__(self, clip_model: str = "ViT-B/32", token_count: int = 12):
        super().__init__()
        self.token_count = token_count
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(clip_model, device=clip_device)
        self.transform = T.Compose(
            [T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
        )
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.chroma_encoder = ChromaTokenEncoder_v17(token_count=token_count, token_dim=512)
        self.mask_gate = MaskGate_v17(token_count=token_count, token_dim=512)
        self.feature_predictor = FeatureDeltaPredictor_v17(context_dim=512)
        self.feature_affine = ChromaFeatureAffine_v17(context_dim=512, channels=512)

    def get_image_embed(self, image_tensor: torch.Tensor) -> torch.Tensor:
        resized_tensor = self.face_pool(image_tensor)
        renormed_tensor = self.transform(resized_tensor * 0.5 + 0.5)
        return self.clip_model.encode_image(renormed_tensor)

    def forward(
        self,
        latent_face: torch.Tensor,
        latent_color: torch.Tensor,
        target_face: torch.Tensor,
        reference_hair_image: torch.Tensor,
        reference_hair_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        lock_mask: torch.Tensor,
        align_hair_mask: torch.Tensor,
        satd_luma: torch.Tensor,
        align_f: torch.Tensor,
        feature_strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        chroma_summary, chroma_tokens, chroma_aux = self.chroma_encoder(reference_hair_image, reference_hair_mask)
        feature_gate, mask_context, gate_aux = self.mask_gate(
            safe_mask=safe_mask,
            lock_mask=lock_mask,
            align_hair_mask=align_hair_mask,
            satd_luma=satd_luma,
        )

        del latent_color, target_face
        s_blend = latent_face

        feature_context = chroma_summary + mask_context
        delta_f = self.feature_predictor(
            align_f=align_f,
            context_token=feature_context,
            safe_mask=safe_mask,
            lock_mask=lock_mask,
            align_hair_mask=align_hair_mask,
            satd_luma=satd_luma,
        )
        affine_delta_f = self.feature_affine(align_f, feature_context)
        f_blend = align_f + feature_strength * feature_gate * (delta_f + affine_delta_f)

        return s_blend, f_blend, {
            "feature_gate": feature_gate,
            "style_delta": torch.zeros_like(latent_face),
            "delta_f": delta_f,
            "affine_delta_f": affine_delta_f,
            "chroma_summary": chroma_summary,
            "chroma_tokens": chroma_tokens,
            "mask_context": mask_context,
            "feature_only": torch.ones((), device=latent_face.device, dtype=latent_face.dtype),
            **chroma_aux,
            **gate_aux,
        }
