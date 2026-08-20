import os
import random
import sys
import argparse
from pathlib import Path
from argparse import Namespace
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_image  # 更快读取
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from losses.pp_losses import LossBuilder, LossBuilderMulti
from models.Encoders import ModulationModule, FeatureiResnet, FeatureEncoderMult
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from models.stylegan2 import dnnlib
from models.stylegan2.model import PixelNorm
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.train import image_grid, WandbLogger, toggle_grad, _LegacyUnpickler, seed_everything, get_fid_calc


# ---------- 配置类 ----------
class Config:
    dataset_path = Path('./images/pp')
    fid_dataset = './fid_images'
    batch_size = 10
    epochs = 120
    iter_before = 10000
    d_reg_every = 16
    inpaint = 0.0
    use_adv = False          # 默认为 False，相关代码已做条件判断
    adv_coef = 0.05
    use_mod = False
    use_full = False
    pretrain = False
    finetune = False
    checkpoint = None
    checkpoint_path = 'checkpoint_pp.pth'
    resume = False
    name_run = 'pp_train'
    save_ckpt_freq = 10
    save_val_freq = 5
    ckpt_dir = Path('pp_modles')
    val_dir = Path('pp_files')
# ------------------------------------------------


# 启用 cuDNN 自动调优（在 Trainer 初始化时设置）
torch.backends.cudnn.benchmark = True
# 可选：设置矩阵乘法精度（若GPU支持TensorCore）
torch.set_float32_matmul_precision('high')   # 或 'medium'


class Trainer:
    def __init__(self, model=None, config=None, optimizer=None, scheduler=None,
                 train_dataloader=None, test_dataloader=None, logger=None):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dilate_erosion = DilateErosion(device=self.device)
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if self.model is not None:
            self.fid_calc = get_fid_calc('input/fid.pkl', config.fid_dataset)

        self.net = Net(Namespace(size=1024, ckpt='pretrained_models/StyleGAN/ffhq.pt',
                                 channel_multiplier=2, latent=512, n_mlp=8, device=self.device))

        with dnnlib.util.open_url("pretrained_models/StyleGAN/ffhq.pkl") as f:
            data = _LegacyUnpickler(f).load()
        self.discriminator = data['D'].cuda().eval()
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(),
                                           lr=3e-4, betas=(0.9, 0.999), weight_decay=0)

        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.device)
        self.seg.load_state_dict(torch.load('pretrained_models/BiSeNet/seg.pth'))
        self.seg.eval()

        # 固定不需要训练的网络
        toggle_grad(self.discriminator, False)
        toggle_grad(self.net.generator, False)
        toggle_grad(self.seg, False)
        self.net.generator.eval()
        self.discriminator.eval()

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.downsample_128 = BicubicDownSample(factor=8)

        self.best_loss = float('+inf')
        if self.config is not None:
            if self.config.pretrain:
                self.LossBuilder = LossBuilder({
                    'lpips_scale': 0.8, 'id': 0.1, 'landmark': 0,
                    'feat_rec': 0.01, 'adv': self.config.adv_coef})
            else:
                self.LossBuilder = LossBuilderMulti({
                    'lpips_scale': 0.8, 'id': 0.1, 'landmark': 0.1,
                    'feat_rec': 0.01, 'adv': self.config.adv_coef, 'inpaint': self.config.inpaint})
        self.cur_iter = 1

    @torch.no_grad()
    def generate_mask(self, I):
        IM = (self.downsample_512(I) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(IM)
        current_mask = torch.argmax(down_seg, dim=1).long().float()
        HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        HM_X = F.interpolate(HM_X.unsqueeze(1), size=(256, 256), mode='nearest')
        HM_XD, HM_XE = self.dilate_erosion.mask(HM_X)
        return HM_XD, HM_XE

    def save_model(self, name, save_online=True):
        with TemporaryDirectory() as tmp_dir:
            model_state_dict = self.model.state_dict()
            for key in list(model_state_dict.keys()):
                if key.startswith("clip_model."):
                    del model_state_dict[key]
            torch.save(
                {'model_state_dict': model_state_dict, 'D': self.discriminator.state_dict(),
                 'cur_iter': self.cur_iter},
                f'{tmp_dir}/{name}.pth')
            self.logger.save(f'{tmp_dir}/{name}.pth', save_online)

    def save_model_local(self, name, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        model_state_dict = self.model.state_dict()
        for key in list(model_state_dict.keys()):
            if key.startswith("clip_model."):
                del model_state_dict[key]
        torch.save(
            {'model_state_dict': model_state_dict, 'D': self.discriminator.state_dict(),
             'cur_iter': self.cur_iter},
            save_dir / f'{name}.pth')
        if self.logger is not None:
            self.logger.save(str(save_dir / f'{name}.pth'), save_online=True)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'D' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['D'], strict=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def save_checkpoint(self, epoch, optimizer, best_loss, checkpoint_path='checkpoint_pp.pth'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'D': self.discriminator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'cur_iter': self.cur_iter,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path, optimizer):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(checkpoint['D'], strict=False)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        self.cur_iter = checkpoint.get('cur_iter', 1)
        print(f"Resuming from epoch {start_epoch}, best_loss: {best_loss}, cur_iter: {self.cur_iter}")
        return start_epoch, best_loss

    def train_one_epoch(self):
        self.model.to(self.device).train()
        # 缓存频繁使用的模块引用，减少属性查找开销
        normalize = self.normalize
        downsample_256 = self.downsample_256
        net_generator = self.net.generator
        loss_builder = self.LossBuilder
        device = self.device
        config = self.config

        for batch in tqdm(self.train_dataloader):
            source, target, target_mask, HT_E = [x.to(device, non_blocking=True) for x in batch]

            source_256 = downsample_256(source).clip(0, 1)                # 256x256
            source_norm = normalize(source)                               # 1024x1024 归一化

            latent_s, latent_f = self.model(
                normalize(source_256),          # 编码器输入：256x256 归一化
                normalize(target),              # target 使用原图归一化（1024x1024）
                target_mask, HT_E
            )

            # 优化生成器前向：先计算前4层特征
            F_w, _ = net_generator([latent_s], input_is_latent=True, return_latents=False,
                                   start_layer=0, end_layer=4)
            gen_im_W, _ = net_generator([latent_s], input_is_latent=True, return_latents=False,
                                        start_layer=5, end_layer=8, layer_in=F_w)

            if config.pretrain:
                alpha = min(1, self.cur_iter / config.iter_before)
                latent_f_gen = alpha * latent_f + (1 - alpha) * F_w
            else:
                latent_f_gen = latent_f

            gen_im_F, _ = net_generator([latent_s], input_is_latent=True, return_latents=False,
                                        start_layer=5, end_layer=8, layer_in=latent_f_gen)

            losses = loss_builder(source_256, target, target_mask, HT_E,
                                  gen_im_W, F_w, gen_im_F, latent_f)

            # 对抗损失（仅在启用且达到迭代阈值时计算）
            if config.use_adv and self.cur_iter >= config.iter_before:
                losses.update(loss_builder.CalcAdvLoss(self.discriminator, gen_im_F))

            losses['loss'] = sum(losses.values())

            self.optimizer.zero_grad(set_to_none=True)   # 比 zero_grad() 更快
            losses['loss'].backward()
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            # 判别器更新（仅当启用对抗训练时执行）
            if config.use_adv and self.cur_iter >= config.iter_before:
                if self.cur_iter == config.iter_before:
                    print('Start training discriminator')

                toggle_grad(self.discriminator, True)
                self.discriminator.train()

                source_1024 = source_norm if self.cur_iter % config.d_reg_every == 0 else None
                disc_loss = loss_builder.CalcDisLoss(self.discriminator, source_1024, gen_im_F.detach())
                if source_1024 is not None:
                    disc_loss.update(loss_builder.CalcR1Loss(self.discriminator, source_1024))

                total_disc_loss = sum(disc_loss.values())
                self.disc_optim.zero_grad(set_to_none=True)
                total_disc_loss.backward()
                total_norm_d = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                disc_loss['grad disc'] = total_norm_d
                self.disc_optim.step()

                toggle_grad(self.discriminator, False)
                self.discriminator.eval()
                losses.update(disc_loss)

            losses['scripts grad'] = total_norm
            self.logger.next_step()
            self.logger.log_scalars({f'scripts {key}': val for key, val in losses.items()})
            self.cur_iter += 1

    @torch.no_grad()
    def validate(self, epoch):
        self.model.to(self.device).eval()
        # 缓存模块引用
        normalize = self.normalize
        downsample_256 = self.downsample_256
        net_generator = self.net.generator
        loss_builder = self.LossBuilder
        device = self.device

        def sum_losses(x, y):
            return {key: y.get(key, 0) + x.get(key, 0) for key in set(x.keys()) | set(y.keys())}

        files = []
        val_losses = {}
        to_299 = T.Resize((299, 299))
        images_to_fid = []

        save_val_images = (self.config.save_val_freq > 0 and epoch % self.config.save_val_freq == 0)
        if save_val_images:
            val_img_dir = self.config.val_dir / f'pp_val_{epoch}'
            val_img_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
            source, target, target_mask, HT_E = [x.to(device, non_blocking=True) for x in batch]

            source_256 = downsample_256(source).clip(0, 1)

            latent_s, latent_f = self.model(
                normalize(source_256),
                normalize(target),
                target_mask, HT_E
            )

            F_w, _ = net_generator([latent_s], input_is_latent=True, return_latents=False,
                                   start_layer=0, end_layer=4)
            gen_im_W, _ = net_generator([latent_s], input_is_latent=True, return_latents=False,
                                        start_layer=5, end_layer=8, layer_in=F_w)
            gen_im_F, _ = net_generator([latent_s], input_is_latent=True, return_latents=False,
                                        start_layer=5, end_layer=8, layer_in=latent_f)

            losses = loss_builder(source_256, target, target_mask, HT_E,
                                  gen_im_W, F_w, gen_im_F, latent_f)
            losses['loss'] = sum(losses.values())

            # 为可视化生成下采样图像
            gen_w_256 = downsample_256((gen_im_W + 1) / 2).clip(0, 1)
            gen_f_256 = downsample_256((gen_im_F + 1) / 2).clip(0, 1)

            images_to_fid.append(to_299((gen_im_F + 1) / 2).clip(0, 1))

            val_losses = sum_losses(val_losses, losses)
            for k in range(source.size(0)):
                files.append([source_256[k].cpu(), target[k].cpu(), gen_w_256[k].cpu(), gen_f_256[k].cpu()])

        val_losses['FID CLIP'] = self.fid_calc(torch.cat(images_to_fid))
        for key, val in val_losses.items():
            if key != 'FID CLIP':
                val = val.item() / len(self.test_dataloader)
            self.logger.log_scalars({f'val {key}': val})

        # 记录验证图像
        np.random.seed(1927)
        idxs = np.random.choice(len(files), size=min(len(files), 100), replace=False)
        images_to_log = []
        for idx in idxs:
            grid_img = image_grid(list(map(T.functional.to_pil_image, files[idx])), 1, len(files[idx]))
            images_to_log.append(grid_img)
            if save_val_images:
                grid_img.save(val_img_dir / f'sample_{idx}.png')

        self.logger.log_scalars({'val images': [wandb.Image(image) for image in images_to_log]})

        return val_losses['loss']

    def train_loop(self, epochs, start_epoch=0, resume=False):
        if not resume:
            self.validate(epoch=0)

        for epoch in range(start_epoch, epochs):
            self.train_one_epoch()
            loss = self.validate(epoch + 1)

            self.save_checkpoint(epoch, self.optimizer, self.best_loss)

            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model(f'best_{epoch}', save_online=False)

            if self.config.save_ckpt_freq > 0 and (epoch + 1) % self.config.save_ckpt_freq == 0:
                self.save_model_local(f'pp_{epoch+1}', self.config.ckpt_dir)

            self.save_model('last', save_online=False)


# ---------- 数据集类 ----------
class PP_dataset(Dataset):
    def __init__(self, source, target, target_mask, HT_E, is_test=False):
        super().__init__()
        self.source = source
        self.target = target
        self.target_mask = target_mask
        self.HT_E = HT_E
        self.is_test = is_test

    def __len__(self):
        return len(self.source)

    def load_image(self, path):
        # 使用 torchvision.io.read_image 替代 PIL，返回 0-255 的 uint8 Tensor (C, H, W)
        img = read_image(str(path)) / 255.0   # 归一化到 [0,1]
        return img

    def __transform__(self, img1, img2, mask1, mask2):
        if self.is_test:
            return img1, img2, mask1, mask2

        if random.random() > 0.5:
            img1 = T.functional.hflip(img1)
            img2 = T.functional.hflip(img2)
            mask1 = T.functional.hflip(mask1)
            mask2 = T.functional.hflip(mask2)
        return img1, img2, mask1, mask2

    def __getitem__(self, idx):
        return self.__transform__(
            self.load_image(self.source[idx]),
            self.target[idx],          # 假设 target 已是 tensor，若也是路径需同样处理
            self.target_mask[idx],
            self.HT_E[idx]
        )


# ---------- 后处理模型 ----------
class PostProcessModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder_face = FeatureEncoderMult(fs_layers=[9], opts=argparse.Namespace(
            **{'arcface_model_path': "pretrained_models/ArcFace/backbone_ir50.pth"}))
        if not self.config.finetune:
            toggle_grad(self.encoder_face, False)

        self.latent_avg = torch.load('pretrained_models/PostProcess/latent_avg.pt', map_location=torch.device('cuda'))
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        if self.config.use_mod:
            self.to_latent_1 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
            self.to_latent_2 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
            self.pixelnorm = PixelNorm()
        else:
            self.to_latent = nn.Sequential(
                nn.Linear(1024, 1024), nn.LayerNorm([1024]), nn.LeakyReLU(),
                nn.Linear(1024, 512)
            )

    def forward(self, source, target, target_mask=None, *args, **kwargs):
        s_face, [f_face] = self.encoder_face(source)
        if self.config.pretrain:
            return self.latent_avg + s_face, f_face

        s_hair, [f_hair] = self.encoder_face(target)

        if self.config.use_mod:
            dt_latent_face = self.pixelnorm(s_face)
            dt_latent_hair = self.pixelnorm(s_hair)

            for mod_module in self.to_latent_1:
                dt_latent_face = mod_module(dt_latent_face, s_hair)

            for mod_module in self.to_latent_2:
                dt_latent_hair = mod_module(dt_latent_hair, s_face)
            finall_s = self.latent_avg + 0.1 * (dt_latent_face + dt_latent_hair)
        else:
            cat_s = torch.cat((s_face, s_hair), dim=-1)
            finall_s = self.latent_avg + self.to_latent(cat_s)

        if self.config.use_full:
            cat_f = torch.cat((f_face, f_hair), dim=1)
        else:
            t_mask = F.interpolate(target_mask, size=(64, 64), mode='nearest')
            cat_f = torch.cat((f_face * t_mask, f_hair * (1 - t_mask)), dim=1)

        finall_f = self.to_feature(cat_f)
        return finall_s, finall_f


# ---------- 主函数 ----------
def main(config):
    seed_everything()
    dataset = []

    idx = 1
    while os.path.isfile(config.dataset_path / f'pp_part_{idx}.dataset'):
        batch_data = torch.load(config.dataset_path / f'pp_part_{idx}.dataset')
        dataset.extend(batch_data)
        idx += 1

    X_train, X_test = train_test_split(dataset, test_size=1024, random_state=42)

    train_dataset = PP_dataset(*list(zip(*X_train)))
    test_dataset = PP_dataset(*list(zip(*X_test)), is_test=True)

    # 优化 DataLoader：增加 prefetch_factor，使用非阻塞传输
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=6,                # 根据 CPU 核心数调整
        prefetch_factor=4,              # 预取批次
        pin_memory=True,
        shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        num_workers=12,
        prefetch_factor=4,
        pin_memory=True,
        shuffle=False
    )

    logger = WandbLogger(name=config.name_run, project='HairFast-PostProcess')
    logger.start_logging()
    logger.save(__file__)

    model = PostProcessModel(config)
    # 尝试使用 torch.compile 加速（需要 PyTorch 2.0+，可选）
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model, mode='reduce-overhead')
    #     # 注意：编译需要时间，首次迭代较慢，后续加速

    if config.pretrain:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

    trainer = Trainer(model, config, optimizer, None, train_dataloader, test_dataloader, logger)

    start_epoch = 0
    resume = False

    if config.resume and os.path.exists(config.checkpoint_path):
        start_epoch, trainer.best_loss = trainer.load_checkpoint(config.checkpoint_path, optimizer)
        resume = True

    if not config.pretrain and not resume and config.checkpoint is not None:
        trainer.load_model(config.checkpoint)

    trainer.train_loop(config.epochs, start_epoch=start_epoch, resume=resume)


if __name__ == '__main__':
    config = Config()
    main(config)