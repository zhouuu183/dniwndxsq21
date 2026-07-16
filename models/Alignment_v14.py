from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from models.CtrlHair.shape_branch.config import cfg as cfg_mask
from models.CtrlHair.shape_branch.solver import Solver as SolverMask, get_hair_face_code, get_new_shape
from models.Net import Net, get_segmentation
from models.RepairNet_v14 import RepairNetV14
from models.sean_codes.models.pix2pix_model import Pix2PixModel, SEAN_OPT, decode_sean, encode_sean
from models.stylegan2.model import PixelNorm
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.nhr_utils_v14 import build_reveal_masks, resize_mask
from utils.save_utils import save_gen_image, save_latents, save_vis_mask

try:
    from utils.save_utils_v14 import save_feature_map, save_tensor_image, save_tensor_mask
except ImportError:
    _to_pil_v14 = T.ToPILImage()

    def _ensure_4d_v14(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _prepare_dir_v14(output_dir: Path | str, folder: str) -> Path:
        save_dir = Path(output_dir) / folder
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_tensor_image(output_dir: Path | str, folder: str, name: str, tensor: torch.Tensor, value_range: str = "tanh") -> None:
        tensor = _ensure_4d_v14(tensor)[0].detach().cpu().float()
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if value_range == "tanh":
            tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
        else:
            tensor = tensor.clamp(0.0, 1.0)
        _to_pil_v14(tensor).save(_prepare_dir_v14(output_dir, folder) / name)

    def save_tensor_mask(output_dir: Path | str, folder: str, name: str, tensor: torch.Tensor) -> None:
        tensor = _ensure_4d_v14(tensor)[0].detach().cpu().float()
        if tensor.shape[0] != 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        tensor = tensor.clamp(0.0, 1.0)
        _to_pil_v14(tensor).save(_prepare_dir_v14(output_dir, folder) / name)

    def save_feature_map(output_dir: Path | str, folder: str, name: str, feature: torch.Tensor) -> None:
        feature = _ensure_4d_v14(feature)[0].detach().cpu().float()
        if feature.shape[0] == 1:
            vis = feature
        else:
            vis = feature.abs().mean(dim=0, keepdim=True)
        vis = vis - vis.amin(dim=(1, 2), keepdim=True)
        vis = vis / vis.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        _to_pil_v14(vis).save(_prepare_dir_v14(output_dir, folder) / name)


class RepairFeatureAdapterV14(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
        return self.proj(feature)


class RepairGateV14(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, kernel_size=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, repair_feature: torch.Tensor, align_feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.expand(-1, 1, -1, -1)
        logits = self.net(torch.cat([repair_feature, align_feature, mask], dim=1))
        return torch.sigmoid(logits)


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


class RotateModelV14(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModuleV14(6, i == 4) for i in range(5)])

    def forward(self, latent_from, latent_to):
        dt_latent = self.pixelnorm(latent_from)
        for modulation_module in self.modulation_module_list:
            dt_latent = modulation_module(dt_latent, latent_to)
        return latent_from + 0.1 * dt_latent


class AlignmentV14(nn.Module):
    """
    Shape alignment with NHR-Branch for long-hair to short-hair reveal repair.
    """

    def __init__(self, opts, latent_encoder=None, net=None):
        super().__init__()
        self.opts = opts
        self.latent_encoder = latent_encoder
        self.net = Net(self.opts) if net is None else net

        self.sean_model = Pix2PixModel(SEAN_OPT)
        self.sean_model.eval()

        solver_mask = SolverMask(cfg_mask, device=self.opts.device, local_rank=-1, training=False)
        self.mask_generator = solver_mask.gen
        self.mask_generator.load_state_dict(torch.load("pretrained_models/ShapeAdaptor/mask_generator.pth"))

        self.rotate_model = RotateModelV14()
        self.rotate_model.load_state_dict(torch.load(self.opts.rotate_checkpoint)["model_state_dict"])
        self.rotate_model.to(self.opts.device).eval()

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.to_bisenet = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.downsample_256 = BicubicDownSample(factor=4)

        self.repair_net = RepairNetV14(
            in_channels=getattr(self.opts, "nhr_input_channels", 17),
            base_channels=getattr(self.opts, "nhr_base_channels", 32),
        ).to(self.opts.device)
        self.repair_adapter = RepairFeatureAdapterV14(self.repair_net.feature_channels, out_channels=512).to(self.opts.device)
        self.repair_gate = RepairGateV14(512).to(self.opts.device)

        repair_checkpoint = getattr(self.opts, "repair_checkpoint", "")
        if repair_checkpoint:
            self.load_repair_checkpoint(repair_checkpoint)

    def load_repair_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.opts.device)
        repair_state = checkpoint.get("repair_net_state_dict", checkpoint.get("repair_net", checkpoint))
        adapter_state = checkpoint.get("repair_adapter_state_dict", checkpoint.get("repair_adapter"))
        gate_state = checkpoint.get("repair_gate_state_dict", checkpoint.get("repair_gate"))

        self.repair_net.load_state_dict(repair_state, strict=False)
        if adapter_state is not None:
            self.repair_adapter.load_state_dict(adapter_state, strict=False)
        if gate_state is not None:
            self.repair_gate.load_state_dict(gate_state, strict=False)
        self.repair_net.to(self.opts.device)
        self.repair_adapter.to(self.opts.device)
        self.repair_gate.to(self.opts.device)

    def shape_module(self, im_name1: str, im_name2: str, name_to_embed, only_target: bool = True, **kwargs):
        with torch.no_grad():
            device = self.opts.device
            img1_in = name_to_embed[im_name1]["image_256"]
            img2_in = name_to_embed[im_name2]["image_256"]
            same_image = im_name1 == im_name2 or torch.allclose(img1_in, img2_in)
            latent_W_1 = name_to_embed[im_name1]["W"]
            latent_W_2 = name_to_embed[im_name2]["W"]
            inp_mask1 = name_to_embed[im_name1]["mask"]
            inp_mask2 = name_to_embed[im_name2]["mask"]

            if not same_image:
                rotate_to = self.rotate_model(latent_W_2[:, :6], latent_W_1[:, :6])
                rotate_to = torch.cat((rotate_to, latent_W_2[:, 6:]), dim=1)
                I_rot, _ = self.net.generator([rotate_to], input_is_latent=True, return_latents=False)
                I_rot_to_seg = self.to_bisenet(((I_rot + 1) / 2).clip(0, 1))
                rot_mask = get_segmentation(I_rot_to_seg)
            else:
                I_rot = None
                rot_mask = inp_mask2

            if not same_image:
                face_1, _ = get_hair_face_code(self.mask_generator, inp_mask1[0, 0, ...])
                _, hair_2 = get_hair_face_code(self.mask_generator, rot_mask[0, 0, ...])
                target_mask = get_new_shape(self.mask_generator, face_1, hair_2)[None, None]
            else:
                target_mask = inp_mask1

            hair_mask_target = torch.where(target_mask == 13, torch.ones_like(target_mask, device=device), torch.zeros_like(target_mask, device=device))

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get("exp_name")) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            if I_rot is not None:
                save_gen_image(output_dir, "Shape_v14", f"{im_name2}_rotate_to_{im_name1}.png", I_rot)
            save_vis_mask(output_dir, "Shape_v14", f"mask_{im_name1}.png", inp_mask1)
            save_vis_mask(output_dir, "Shape_v14", f"mask_{im_name2}.png", inp_mask2)
            save_vis_mask(output_dir, "Shape_v14", f"mask_{im_name2}_rotate_to_{im_name1}.png", rot_mask)
            save_vis_mask(output_dir, "Shape_v14", f"mask_{im_name1}_{im_name2}_target.png", target_mask)

        if only_target:
            return {"HM_X": hair_mask_target, "target_mask": target_mask}

        hair_mask1 = torch.where(inp_mask1 == 13, torch.ones_like(inp_mask1, device=device), torch.zeros_like(inp_mask1, device=device))
        hair_mask2 = torch.where(inp_mask2 == 13, torch.ones_like(inp_mask2, device=device), torch.zeros_like(inp_mask2, device=device))
        return inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target

    def _author_fusion(
        self,
        *,
        latent_F_1: torch.Tensor,
        latent_F_2: torch.Tensor,
        intermediate_align: torch.Tensor,
        latent_F_out_new: torch.Tensor,
        hair_mask1: torch.Tensor,
        hair_mask2: torch.Tensor,
        hair_mask_target: torch.Tensor,
    ) -> torch.Tensor:
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
        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode="bicubic")
        interpolation_low = 1 - free_mask_down_32

        latent_F_align = intermediate_align + interpolation_low[0] * (latent_F_1 - intermediate_align)
        latent_F_align = latent_F_out_new + interpolation_low[1] * (latent_F_align - latent_F_out_new)
        latent_F_align = latent_F_2 + interpolation_low[2] * (latent_F_align - latent_F_2)
        return latent_F_align

    def _build_coarse_bg0(
        self,
        *,
        I_source: torch.Tensor,
        gen1_sean: torch.Tensor,
        masks: dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        blur_kernel = int(kwargs.get("nhr_bg0_blur_kernel", getattr(self.opts, "nhr_bg0_blur_kernel", 11)))
        blur_kernel = max(1, blur_kernel)
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        blurred_source = F.avg_pool2d(I_source, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
        sean_blend = float(kwargs.get("nhr_bg0_sean_blend", getattr(self.opts, "nhr_bg0_sean_blend", 0.15)))
        sean_blend = min(max(sean_blend, 0.0), 1.0)

        sean_guidance = gen1_sean.unsqueeze(0)
        core_fill = torch.lerp(blurred_source, sean_guidance, sean_blend)
        core_mask = masks["M_occ"]
        shell_mask = (masks["M_occ_d"] - core_mask).clamp(0.0, 1.0)
        I_bg0 = I_source * (1.0 - masks["M_occ_d"]) + blurred_source * shell_mask + core_fill * core_mask
        return I_bg0.clamp(-1.0, 1.0)

    def _run_nhr_branch(
        self,
        *,
        im_name1: str,
        name_to_embed,
        source_parsing: torch.Tensor,
        hair_mask1: torch.Tensor,
        hair_mask_target: torch.Tensor,
        latent_F_align_old: torch.Tensor,
        gen1_sean: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        I_source = name_to_embed[im_name1]["image_norm_256"]
        latent_S_source = name_to_embed[im_name1]["S"]
        with torch.no_grad():
            I_align_old_1024, _ = self.net.generator(
                [latent_S_source],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_align_old,
            )
            I_hair_coarse = self.downsample_256(I_align_old_1024)

            masks = build_reveal_masks(
                source_parsing=source_parsing,
                target_hair_mask=hair_mask_target,
                occ_dilate=kwargs.get("nhr_occ_dilate", getattr(self.opts, "nhr_occ_dilate", 7)),
                ring_dilate=kwargs.get("nhr_ring_dilate", getattr(self.opts, "nhr_ring_dilate", 9)),
                ring_erode=kwargs.get("nhr_ring_erode", getattr(self.opts, "nhr_ring_erode", 1)),
                use_ring=kwargs.get("nhr_use_ring", getattr(self.opts, "nhr_use_ring", True)),
            )
            I_bg0 = self._build_coarse_bg0(I_source=I_source, gen1_sean=gen1_sean, masks=masks, **kwargs)

        repair_input = torch.cat(
            [
                I_source,
                I_bg0,
                I_hair_coarse,
                masks["H_source"],
                masks["H_align"],
                masks["M_occ"],
                masks["M_occ_d"],
                masks["M_face"],
                masks["M_neck"],
                masks["M_cloth"],
                masks["M_ring"],
            ],
            dim=1,
        )
        repair_outputs = self.repair_net(repair_input)
        A = torch.sigmoid(repair_outputs["A_fill"]) * masks["M_occ_d"]
        I_clean_bg = (1.0 - A) * I_bg0 + A * repair_outputs["I_fill"]
        I_align_clean = (1.0 - masks["H_align"]) * I_clean_bg + masks["H_align"] * I_hair_coarse
        I_source_clean = (1.0 - masks["M_occ_d"]) * I_source + masks["M_occ_d"] * I_clean_bg

        latent_F_align_new = latent_F_align_old
        M32 = resize_mask(masks["M_occ_d"], latent_F_align_old.shape[-2:], mode="bilinear")
        R32 = None
        bypass_enabled = not kwargs.get(
            "nhr_disable_local_bypass",
            getattr(self.opts, "nhr_disable_local_bypass", False),
        )
        bypass_mode = kwargs.get("nhr_bypass_mode", getattr(self.opts, "nhr_bypass_mode", "gated_residual"))
        bypass_scale = float(kwargs.get("nhr_bypass_scale", getattr(self.opts, "nhr_bypass_scale", 0.15)))
        bypass_scale = max(0.0, bypass_scale)
        gate = None
        if bypass_enabled and bypass_scale > 0.0:
            R32 = self.repair_adapter(repair_outputs["R_feat"], latent_F_align_old.shape[-2:])
            if bypass_mode == "replace":
                latent_F_align_new = latent_F_align_old + bypass_scale * M32 * (R32 - latent_F_align_old)
            else:
                gate = self.repair_gate(R32, latent_F_align_old, M32)
                latent_F_align_new = latent_F_align_old + bypass_scale * M32 * gate * R32

        return {
            "latent_F_align_old": latent_F_align_old,
            "latent_F_align": latent_F_align_new,
            "I_bg0": I_bg0,
            "I_hair_coarse": I_hair_coarse,
            "I_fill": repair_outputs["I_fill"],
            "A": A,
            "I_clean_bg": I_clean_bg,
            "I_align_clean": I_align_clean,
            "I_source_clean": I_source_clean,
            "R_feat": repair_outputs["R_feat"],
            "R32": R32,
            "M32": M32,
            "Gate": gate,
            **masks,
        }

    def _save_nhr_debug(self, output_dir, im_name1: str, im_name2: str, stage: dict[str, torch.Tensor]) -> None:
        stem = f"{im_name1}_{im_name2}"
        save_tensor_mask(output_dir, "NHR_v14", f"{stem}_M_occ.png", stage["M_occ"])
        save_tensor_mask(output_dir, "NHR_v14", f"{stem}_M_occ_d.png", stage["M_occ_d"])
        save_tensor_mask(output_dir, "NHR_v14", f"{stem}_M_ring.png", stage["M_ring"])
        save_tensor_image(output_dir, "NHR_v14", f"{stem}_I_bg0.png", stage["I_bg0"])
        save_tensor_image(output_dir, "NHR_v14", f"{stem}_I_hair_coarse.png", stage["I_hair_coarse"])
        save_tensor_image(output_dir, "NHR_v14", f"{stem}_I_fill.png", stage["I_fill"])
        save_tensor_mask(output_dir, "NHR_v14", f"{stem}_A.png", stage["A"])
        save_tensor_image(output_dir, "NHR_v14", f"{stem}_I_clean_bg.png", stage["I_clean_bg"])
        save_tensor_image(output_dir, "NHR_v14", f"{stem}_I_align_clean.png", stage["I_align_clean"])
        save_tensor_image(output_dir, "NHR_v14", f"{stem}_I_source_clean.png", stage["I_source_clean"])
        save_feature_map(output_dir, "NHR_v14", f"{stem}_F_align_old.png", stage["latent_F_align_old"])
        save_feature_map(output_dir, "NHR_v14", f"{stem}_F_align_new.png", stage["latent_F_align"])
        save_latents(
            output_dir,
            "NHR_v14",
            f"{stem}.npz",
            latent_F_align_old=stage["latent_F_align_old"],
            latent_F_align=stage["latent_F_align"],
            M_occ=stage["M_occ"],
            M_occ_d=stage["M_occ_d"],
            A=stage["A"],
        )

    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        same_image = im_name1 == im_name2 or torch.allclose(img1_in, img2_in)
        latent_S_1 = name_to_embed[im_name1]["S"]
        latent_F_1 = name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]

        if same_image:
            target = self.shape_module(im_name1, im_name2, name_to_embed, only_target=True, **kwargs)
            return {
                "latent_F_align_old": latent_F_1,
                "latent_F_align": latent_F_1,
                "HM_X": target["HM_X"],
                "I_source_clean": name_to_embed[im_name1]["image_norm_256"],
            }

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = self.shape_module(
            im_name1,
            im_name2,
            name_to_embed,
            only_target=False,
            **kwargs,
        )

        with torch.no_grad():
            images = torch.cat([img1_in, img2_in], dim=0)
            labels = torch.cat([inp_mask1, inp_mask2], dim=0)
            img1_code, img2_code = encode_sean(self.sean_model, images, labels)
            gen1_sean = decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask)
            gen2_sean = decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask)

            enc_imgs = self.latent_encoder([gen1_sean, gen2_sean])
            intermediate_align = enc_imgs["F"][0].unsqueeze(0)
            latent_inter = enc_imgs["W"][0].unsqueeze(0)
            latent_F_out_new = enc_imgs["F"][1].unsqueeze(0)
            latent_out = enc_imgs["W"][1].unsqueeze(0)

            latent_F_align_old = self._author_fusion(
                latent_F_1=latent_F_1,
                latent_F_2=latent_F_2,
                intermediate_align=intermediate_align,
                latent_F_out_new=latent_F_out_new,
                hair_mask1=hair_mask1,
                hair_mask2=hair_mask2,
                hair_mask_target=hair_mask_target,
            )

        nhr_enabled = kwargs.get("use_nhr_branch", getattr(self.opts, "use_nhr_branch", True))
        if nhr_enabled:
            stage = self._run_nhr_branch(
                im_name1=im_name1,
                name_to_embed=name_to_embed,
                source_parsing=inp_mask1,
                hair_mask1=hair_mask1,
                hair_mask_target=hair_mask_target,
                latent_F_align_old=latent_F_align_old,
                gen1_sean=gen1_sean,
                **kwargs,
            )
        else:
            stage = {
                "latent_F_align_old": latent_F_align_old,
                "latent_F_align": latent_F_align_old,
                "HM_X": hair_mask_target,
                "I_source_clean": name_to_embed[im_name1]["image_norm_256"],
            }

        stage["HM_X"] = hair_mask_target
        stage["target_mask"] = target_mask
        stage["source_mask"] = inp_mask1

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get("exp_name")) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Align_v14", f"{im_name1}_{im_name2}_SEAN_bg.png", gen1_sean)
            save_gen_image(output_dir, "Align_v14", f"{im_name2}_{im_name1}_SEAN_shape.png", gen2_sean)

            img1_e4e = self.net.generator(
                [latent_inter],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=intermediate_align,
            )[0]
            img2_e4e = self.net.generator(
                [latent_out],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_out_new,
            )[0]
            save_gen_image(output_dir, "Align_v14", f"{im_name1}_{im_name2}_e4e_bg.png", img1_e4e)
            save_gen_image(output_dir, "Align_v14", f"{im_name2}_{im_name1}_e4e_shape.png", img2_e4e)

            gen_im_old, _ = self.net.generator(
                [latent_S_1],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_align_old,
            )
            save_gen_image(output_dir, "Align_v14", f"{im_name1}_{im_name2}_author_alignment.png", gen_im_old)
            if nhr_enabled:
                self._save_nhr_debug(output_dir, im_name1, im_name2, stage)

            gen_im_new, _ = self.net.generator(
                [latent_S_1],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=stage["latent_F_align"],
            )
            save_gen_image(output_dir, "Align_v14", f"{im_name1}_{im_name2}_output.png", gen_im_new)

        return stage
