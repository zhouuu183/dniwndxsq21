from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from models.Net import FeatureEncoderMult, IBasicBlock, conv1x1
from models.stylegan2.model import PixelNorm


class ModulationModuleV14(nn.Module):
    def __init__(self, layernum, last=False, inp=512, middle=512):
        super().__init__()
        self.layernum = layernum
        self.last = last
        self.fc = nn.Linear(512, 512)
        self.norm = nn.LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = nn.Sequential(
            nn.Linear(inp, middle),
            nn.LayerNorm([middle]),
            nn.LeakyReLU(),
            nn.Linear(middle, 512),
        )
        self.beta_function = nn.Sequential(
            nn.Linear(inp, middle),
            nn.LayerNorm([middle]),
            nn.LeakyReLU(),
            nn.Linear(middle, 512),
        )
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, embedding):
        x = self.fc(x)
        x = self.norm(x)
        gamma = self.gamma_function(embedding)
        beta = self.beta_function(embedding)
        out = x * (1 + gamma) + beta
        if not self.last:
            out = self.leakyrelu(out)
        return out


class FeatureiResnetV14(nn.Module):
    def __init__(self, blocks, inplanes=1024):
        super().__init__()
        self.res_blocks = {}
        for n, block in enumerate(blocks, start=1):
            planes, num_blocks = block
            for k in range(1, num_blocks + 1):
                downsample = None
                if inplanes != planes:
                    downsample = nn.Sequential(
                        conv1x1(inplanes, planes, 1),
                        nn.BatchNorm2d(planes, eps=1e-05),
                    )
                self.res_blocks[f"res_block_{n}_{k}"] = IBasicBlock(inplanes, planes, 1, downsample, 1, 64, 1)
                inplanes = planes
        self.res_blocks = nn.ModuleDict(self.res_blocks)

    def forward(self, x):
        for module in self.res_blocks.values():
            x = module(x)
        return x


class PostProcessModelV14(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_face = FeatureEncoderMult(
            fs_layers=[9],
            opts=argparse.Namespace(**{"arcface_model_path": "pretrained_models/ArcFace/backbone_ir50.pth"}),
        )
        self.latent_avg = torch.load("pretrained_models/PostProcess/latent_avg.pt", map_location=torch.device("cuda"))
        self.to_feature = FeatureiResnetV14([[1024, 2], [768, 2], [512, 2]])
        self.to_latent_1 = nn.ModuleList([ModulationModuleV14(18, i == 4) for i in range(5)])
        self.to_latent_2 = nn.ModuleList([ModulationModuleV14(18, i == 4) for i in range(5)])
        self.pixelnorm = PixelNorm()

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_face, [f_face] = self.encoder_face(source)
        s_hair, [f_hair] = self.encoder_face(target)

        dt_latent_face = self.pixelnorm(s_face)
        dt_latent_hair = self.pixelnorm(s_hair)

        for mod_module in self.to_latent_1:
            dt_latent_face = mod_module(dt_latent_face, s_hair)

        for mod_module in self.to_latent_2:
            dt_latent_hair = mod_module(dt_latent_hair, s_face)

        finall_s = self.latent_avg + 0.1 * (dt_latent_face + dt_latent_hair)
        finall_f = self.to_feature(torch.cat((f_face, f_hair), dim=1))
        return finall_s, finall_f
