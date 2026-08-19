from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader

from datasets.image_dataset import ImagesDataset, image_collate
from models.FeatureStyleEncoder import FSencoder
from models.Net import Net, get_segmentation
from models.encoder4editing.utils.model_utils import get_latents, setup_model
from utils.bicubic import BicubicDownSample
from utils.save_utils import save_gen_image, save_latents


class Embedding_v19(nn.Module):
    """
    v19 embedding with both 32x32 and 64x64 FS features for boundary-guided shape alignment.
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.encoder = FSencoder.get_trainer(self.opts.device)
        self.e4e, _ = setup_model("pretrained_models/encoder4editing/e4e_ffhq_encode.pt", self.opts.device)

        self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.to_bisenet = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)

    def setup_dataloader(self, images: dict[torch.Tensor, list[str]] | list[torch.Tensor], batch_size=None):
        dataset = ImagesDataset(images)
        return DataLoader(
            dataset,
            collate_fn=image_collate,
            shuffle=False,
            batch_size=batch_size or self.opts.batch_size,
        )

    @torch.inference_mode()
    def get_e4e_embed(self, images: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        device = self.opts.device
        batch_size = max(1, int(getattr(self.opts, "e4e_batch_size", getattr(self.opts, "batch_size", 1))))
        dataloader = self.setup_dataloader(images, batch_size=batch_size)
        all_latent_w = []
        all_latent_f32 = []
        all_latent_f64 = []
        for image, _ in dataloader:
            image = image.to(device)
            latent_W = get_latents(self.e4e, image)
            latent_F32, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
            latent_F64, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=4)
            all_latent_w.append(latent_W)
            all_latent_f32.append(latent_F32)
            all_latent_f64.append(latent_F64)
        if not all_latent_w:
            raise RuntimeError("No images were provided to get_e4e_embed().")
        latent_W = torch.cat(all_latent_w, dim=0)
        latent_F32 = torch.cat(all_latent_f32, dim=0)
        latent_F64 = torch.cat(all_latent_f64, dim=0)
        return {"F": latent_F32, "F32": latent_F32, "F64": latent_F64, "W": latent_W}

    @torch.inference_mode()
    def embedding_images(self, images_to_name: dict[torch.Tensor, list[str]], **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        device = self.opts.device
        dataloader = self.setup_dataloader(images_to_name)
        name_to_embed = defaultdict(dict)

        for image, names in dataloader:
            image = image.to(device)
            im_512 = self.downsample_512(image)
            im_256 = self.downsample_256(image)
            im_256_norm = self.normalize(im_256)

            latent_W = get_latents(self.e4e, im_256_norm)

            output = self.encoder.test(img=self.normalize(image), return_latent=True)
            feature_16 = output.pop()
            latent_S = output.pop()

            latent_F32, _ = self.net.generator(
                [latent_S],
                input_is_latent=True,
                return_latents=False,
                start_layer=3,
                end_layer=3,
                layer_in=feature_16,
            )
            latent_F64, _ = self.net.generator(
                [latent_S],
                input_is_latent=True,
                return_latents=False,
                start_layer=3,
                end_layer=4,
                layer_in=feature_16,
            )

            masks = torch.cat([get_segmentation(image_item.unsqueeze(0)) for image_item in self.to_bisenet(im_512)])

            if len(images_to_name) > 1:
                hair_mask_32 = F.interpolate((masks == 13).float(), size=(32, 32), mode="bicubic")
                hair_mask_64 = F.interpolate((masks == 13).float(), size=(64, 64), mode="bicubic")

                latent_F32_from_W, _ = self.net.generator(
                    [latent_W],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=0,
                    end_layer=3,
                )
                latent_F64_from_W, _ = self.net.generator(
                    [latent_W],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=0,
                    end_layer=4,
                )
                latent_F32 = latent_F32 + self.opts.mixing * hair_mask_32 * (latent_F32_from_W - latent_F32)
                latent_F64 = latent_F64 + self.opts.mixing * hair_mask_64 * (latent_F64_from_W - latent_F64)

            for batch_idx, batch_names in enumerate(names):
                for name in batch_names:
                    name_to_embed[name]["W"] = latent_W[batch_idx].unsqueeze(0)
                    name_to_embed[name]["S"] = latent_S[batch_idx].unsqueeze(0)
                    name_to_embed[name]["feature_16"] = feature_16[batch_idx].unsqueeze(0)
                    name_to_embed[name]["F"] = latent_F32[batch_idx].unsqueeze(0)
                    name_to_embed[name]["F32"] = latent_F32[batch_idx].unsqueeze(0)
                    name_to_embed[name]["F64"] = latent_F64[batch_idx].unsqueeze(0)
                    name_to_embed[name]["mask"] = masks[batch_idx].unsqueeze(0)
                    name_to_embed[name]["image_256"] = im_256[batch_idx].unsqueeze(0)
                    name_to_embed[name]["image_norm_256"] = im_256_norm[batch_idx].unsqueeze(0)

            if self.opts.save_all:
                exp_name = kwargs.get("exp_name") or ""
                output_dir = self.opts.save_all_dir / exp_name
                gen_W_im, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False)
                gen_FS_im, _ = self.net.generator(
                    [latent_S],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=latent_F32,
                )
                for batch_names, im_W, lat_W in zip(names, gen_W_im, latent_W):
                    for name in batch_names:
                        save_gen_image(output_dir, "W+", f"{name}.png", im_W)
                        save_latents(output_dir, "W+", f"{name}.npz", latent_W=lat_W)
                for batch_names, im_F, lat_S, lat_F32, lat_F64 in zip(names, gen_FS_im, latent_S, latent_F32, latent_F64):
                    for name in batch_names:
                        save_gen_image(output_dir, "FS", f"{name}.png", im_F)
                        save_latents(
                            output_dir,
                            "FS",
                            f"{name}.npz",
                            latent_S=lat_S,
                            latent_F32=lat_F32,
                            latent_F64=lat_F64,
                        )

        return name_to_embed
