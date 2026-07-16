from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=max(1, min(8, out_channels // 8)), num_channels=out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        gate = torch.sigmoid(self.gate(x))
        return self.act(self.norm(feat * gate))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block1 = GatedConvBlock(channels, channels)
        self.block2 = GatedConvBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block2(self.block1(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            GatedConvBlock(in_channels, out_channels, stride=2),
            GatedConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            GatedConvBlock(in_channels + skip_channels, out_channels),
            GatedConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class RepairNetV14(nn.Module):
    def __init__(self, in_channels: int = 16, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Sequential(
            GatedConvBlock(in_channels, c1),
            GatedConvBlock(c1, c1),
        )
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)

        self.bottleneck = nn.Sequential(
            ResidualBlock(c4),
            ResidualBlock(c4),
            ResidualBlock(c4),
        )

        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.rgb_head = nn.Sequential(
            GatedConvBlock(c1, c1),
            nn.Conv2d(c1, 3, kernel_size=1),
            nn.Tanh(),
        )
        self.alpha_head = nn.Sequential(
            GatedConvBlock(c1, c1),
            nn.Conv2d(c1, 1, kernel_size=1),
        )
        self.feature_channels = c1
        self._init_safe_alpha_head()

    def _init_safe_alpha_head(self) -> None:
        # Start with a near-zero alpha so the model initially preserves I_bg0
        # instead of aggressively mixing in an untrained I_fill.
        alpha_conv = self.alpha_head[-1]
        nn.init.zeros_(alpha_conv.weight)
        if alpha_conv.bias is not None:
            nn.init.constant_(alpha_conv.bias, -4.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)

        bottleneck = self.bottleneck(s4)
        d3 = self.up3(bottleneck, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        return {
            "I_fill": self.rgb_head(d1),
            "A_fill": self.alpha_head(d1),
            "R_feat": d1,
        }
