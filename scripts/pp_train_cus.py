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


# ---------- 配置类：用户在此修改参数 ----------
class Config:
    # 数据集路径
    dataset_path = Path('./images/pp')
    fid_dataset = './fid_images'                     # FID 计算所需数据集路径

    # 训练参数
    batch_size = 8
    epochs = 120
    iter_before = 10000                        # 预训练迭代次数（用于 alpha 渐变）
    d_reg_every = 16                            # 判别器 R1 正则化间隔
    inpaint = 0.0                               # 修复损失系数
    use_adv = False                             # 是否使用对抗训练
    adv_coef = 0.05                             # 对抗损失系数
    use_mod = False                              # 是否使用调制模块（ModulationModule）
    use_full = False                             # 是否使用完整特征融合
    pretrain = False                             # 是否为预训练阶段（影响损失函数和输出）
    finetune = False                              # 是否微调编码器

    # 模型加载/恢复
    checkpoint = None                            # 初始模型权重路径（用于非预训练）
    checkpoint_path = 'checkpoint_pp.pth'        # 恢复训练用的检查点路径
    resume = False                                # 是否从检查点恢复训练

    # 日志与保存
    name_run = 'pp_train'                             # wandb 运行名称
    save_ckpt_freq = 10                           # 每多少 epoch 保存一次 pp_{epoch}.pth
    save_val_freq = 5                             # 每多少 epoch 保存一次验证图片
    ckpt_dir = Path('pp_modles')                # 保存 pp_n.pth 的目录
    val_dir = Path('pp_files')                     # 保存验证图片的根目录
# ------------------------------------------------


class Trainer:
    def __init__(self,
                 model=None,
                 config=None,
                 optimizer=None,
                 scheduler=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 logger=None
                 ):
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

        self.net = Net(Namespace(size=1024, ckpt='pretrained_models/StyleGAN/ffhq.pt', channel_multiplier=2, latent=512,
                                 n_mlp=8, device=self.device))

        with dnnlib.util.open_url("pretrained_models/StyleGAN/ffhq.pkl") as f:
            data = _LegacyUnpickler(f).load()
        self.discriminator = data['D'].cuda().eval()
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4, betas=(0.9, 0.999), amsgrad=False,
                                           weight_decay=0)

        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.device)
        self.seg.load_state_dict(torch.load('pretrained_models/BiSeNet/seg.pth'))
        self.seg.eval()

        toggle_grad(self.discriminator, False)
        toggle_grad(self.net.generator, False)
        toggle_grad(self.seg, False)

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.downsample_128 = BicubicDownSample(factor=8)

        self.best_loss = float('+inf')
        if self.config is not None:
            if self.config.pretrain:
                self.LossBuilder = LossBuilder(
                    {'lpips_scale': 0.8, 'id': 0.1, 'landmark': 0, 'feat_rec': 0.01, 'adv': self.config.adv_coef})
            else:
                self.LossBuilder = LossBuilderMulti(
                    {'lpips_scale': 0.8, 'id': 0.1, 'landmark': 0.1, 'feat_rec': 0.01, 'adv': self.config.adv_coef,
                     'inpaint': self.config.inpaint})
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
        """通过 logger 上传模型（wandb），不保留本地文件"""
        with TemporaryDirectory() as tmp_dir:
            model_state_dict = self.model.state_dict()
            # 删除预训练的 CLIP 权重
            for key in list(model_state_dict.keys()):
                if key.startswith("clip_model."):
                    del model_state_dict[key]
            torch.save(
                {'model_state_dict': model_state_dict, 'D': self.discriminator.state_dict(), 'cur_iter': self.cur_iter},
                f'{tmp_dir}/{name}.pth')
            self.logger.save(f'{tmp_dir}/{name}.pth', save_online)

    def save_model_local(self, name, save_dir):
        """保存模型权重到本地目录，并可选上传 wandb"""
        os.makedirs(save_dir, exist_ok=True)
        model_state_dict = self.model.state_dict()
        for key in list(model_state_dict.keys()):
            if key.startswith("clip_model."):
                del model_state_dict[key]
        torch.save(
            {'model_state_dict': model_state_dict, 'D': self.discriminator.state_dict(), 'cur_iter': self.cur_iter},
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
        """保存完整训练检查点（用于恢复训练）"""
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
        for batch in tqdm(self.train_dataloader):
            source, target, target_mask, HT_E = map(lambda x: x.to(self.device), batch)
            source, source_1024 = self.downsample_256(source).clip(0, 1), self.normalize(source)

            latent_s, latent_f = self.model(self.normalize(source), self.normalize(target), target_mask, HT_E)

            gen_im_W, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False)
            F_w, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False,
                                        start_layer=0, end_layer=4)

            if self.config.pretrain:
                alpha = min(1, self.cur_iter / self.config.iter_before)
                latent_f_gen = alpha * latent_f + (1 - alpha) * F_w
            else:
                latent_f_gen = latent_f

            gen_im_F, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False,
                                             start_layer=5, end_layer=8, layer_in=latent_f_gen)

            losses = self.LossBuilder(source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f)

            if self.config.use_adv and self.cur_iter >= self.config.iter_before:
                losses.update(self.LossBuilder.CalcAdvLoss(self.discriminator, gen_im_F))

            losses['loss'] = sum(losses.values())

            self.optimizer.zero_grad()
            losses['loss'].backward()

            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            if self.config.use_adv and self.cur_iter >= self.config.iter_before:
                if self.cur_iter == self.config.iter_before:
                    print('Start scripts discr')

                toggle_grad(self.discriminator, True)
                self.discriminator.train()

                disc_loss = self.LossBuilder.CalcDisLoss(self.discriminator, source_1024, gen_im_F.detach())
                if self.cur_iter % self.config.d_reg_every:
                    disc_loss.update(self.LossBuilder.CalcR1Loss(self.discriminator, source_1024))

                total_loss = sum(disc_loss.values())

                self.disc_optim.zero_grad()

                total_loss.backward()
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

        sum_losses = lambda x, y: {key: y.get(key, 0) + x.get(key, 0) for key in set(x.keys()) | set(y.keys())}
        files = []
        val_losses = {}
        to_299 = T.Resize((299, 299))
        images_to_fid = []

        # 判断是否保存验证图片到本地
        save_val_images = (self.config.save_val_freq > 0 and epoch % self.config.save_val_freq == 0)
        if save_val_images:
            val_img_dir = self.config.val_dir / f'pp_val_{epoch}'
            val_img_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
            source, target, target_mask, HT_E = map(lambda x: x.to(self.device), batch)
            source = self.downsample_256(source).clip(0, 1)
            bsz = source.size(0)

            latent_s, latent_f = self.model(self.normalize(source), self.normalize(target), target_mask, HT_E)

            gen_im_W, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False)
            F_w, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False,
                                        start_layer=0, end_layer=4)
            gen_im_F, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False,
                                             start_layer=5, end_layer=8, layer_in=latent_f)

            losses = self.LossBuilder(source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f)
            losses['loss'] = sum(losses.values())

            gen_w_256 = self.downsample_256((gen_im_W + 1) / 2).clip(0, 1)
            gen_f_256 = self.downsample_256((gen_im_F + 1) / 2).clip(0, 1)

            images_to_fid.append(to_299((gen_im_F + 1) / 2).clip(0, 1))

            val_losses = sum_losses(val_losses, losses)
            for k in range(bsz):
                files.append([source[k].cpu(), target[k].cpu(), gen_w_256[k].cpu(), gen_f_256[k].cpu()])

        val_losses['FID CLIP'] = self.fid_calc(torch.cat(images_to_fid))
        for key, val in val_losses.items():
            if key != 'FID CLIP':
                val = val.item() / len(self.test_dataloader)
            self.logger.log_scalars({f'val {key}': val})

        # 随机选择一部分样本生成拼接图并记录到 wandb
        np.random.seed(1927)
        idxs = np.random.choice(len(files), size=min(len(files), 100), replace=False)
        images_to_log = []
        for idx in idxs:
            grid_img = image_grid(list(map(T.functional.to_pil_image, files[idx])), 1, len(files[idx]))
            images_to_log.append(grid_img)
            # 如果需要保存到本地，则同时保存单个 grid 图片
            if save_val_images:
                grid_img.save(val_img_dir / f'sample_{idx}.png')

        self.logger.log_scalars({'val images': [wandb.Image(image) for image in images_to_log]})

        return val_losses['loss']

    def train_loop(self, epochs, start_epoch=0, resume=False):
        if not resume:
            self.validate(epoch=0)  # 初始验证，epoch=0 用于判断频率

        for epoch in range(start_epoch, epochs):
            self.train_one_epoch()
            loss = self.validate(epoch + 1)  # 传入当前 epoch（从1开始）

            # 保存完整检查点（用于恢复）
            self.save_checkpoint(epoch, self.optimizer, self.best_loss)

            # 保存最佳模型（wandb 上传）
            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model(f'best_{epoch}', save_online=False)

            # 按频率保存本地模型 pp_{epoch}.pth
            if self.config.save_ckpt_freq > 0 and (epoch + 1) % self.config.save_ckpt_freq == 0:
                self.save_model_local(f'pp_{epoch+1}', self.config.ckpt_dir)

            # 始终保存 last 模型（wandb 上传）
            self.save_model('last', save_online=False)


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
        return T.functional.to_tensor(Image.open(path))

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
        return self.__transform__(self.load_image(self.source[idx]), self.target[idx], self.target_mask[idx],
                                  self.HT_E[idx])


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
            self.to_latent = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm([1024]), nn.LeakyReLU(),
                                           nn.Linear(1024, 512))

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


def main(config):
    seed_everything()
    dataset = []

    # 加载预处理的数据集块
    idx = 1
    while os.path.isfile(config.dataset_path / f'pp_part_{idx}.dataset'):
        batch_data = torch.load(config.dataset_path / f'pp_part_{idx}.dataset')
        dataset.extend(batch_data)
        idx += 1

    X_train, X_test = train_test_split(dataset, test_size=1024, random_state=42)

    train_dataset = PP_dataset(*list(zip(*X_train)))
    test_dataset = PP_dataset(*list(zip(*X_test)), is_test=True)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=10, pin_memory=True,
                                  shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=10, pin_memory=True,
                                 shuffle=False)

    logger = WandbLogger(name=config.name_run, project='HairFast-PostProcess')
    logger.start_logging()
    logger.save(__file__)

    model = PostProcessModel(config)
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
    # 用户在此修改配置
    config = Config()
    main(config)