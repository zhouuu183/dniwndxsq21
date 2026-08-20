import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Encoders import ClipBlendingModel as BlendingModel
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.train import toggle_grad, WandbLogger, image_grid, seed_everything, get_fid_calc

# ================== 用户配置区域 ==================
config = {
    'name_run': 'blending_train_base_f',                         # wandb 运行名称
    'dataset': './images/blending',         # 数据集路径（包含 FS, Align 子文件夹）
    'FFHQ': './images/FFHQ',              # FFHQ 图像路径，需替换为实际路径
    'fid_dataset': './celeba-1024',                       # FID 计算所需的数据集路径
    'epochs': 50,                                # 训练总轮数
    'checkpoint_path': './checkpoints/checkpoint_base_f_blending.pth', # 断点保存路径
    'resume': True,                               # 是否从断点恢复
    'save_model_interval': 5,                     # 每多少轮保存一次 blending_n.pth
    'save_val_interval': 2,                         # 每多少轮保存一次验证图片
    'models_dir': 'blending_base_models_f',                # 模型权重保存目录
    'val_images_dir': 'blending_base_files_f',             # 验证图片保存根目录
}
# =================================================

MASK_CACHE_DIR = os.path.join(config['dataset'], 'mask_cache')


def precompute_masks(exps, dataset_path, ffhq_path, cache_dir):
    """
    预计算所有样本的 target_mask, HM_3E, HM_XE，并保存到 cache_dir。
    只依赖于 (im1, im3)，与 im2 无关。
    """
    os.makedirs(cache_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载必要的模型（与 Trainer 中一致）
    net = Net(Namespace(size=1024, ckpt='pretrained_models/StyleGAN/ffhq.pt',
                        channel_multiplier=2, latent=512, n_mlp=8, device=device))
    seg = BiSeNet(n_classes=16)
    seg.to(device)
    seg.eval()
    seg.load_state_dict(torch.load('pretrained_models/BiSeNet/seg.pth'))
    toggle_grad(seg, False)
    toggle_grad(net.generator, False)

    dilate_erosion = DilateErosion(device=device)
    downsample_512 = BicubicDownSample(factor=2)   # 用于 BiSeNet 输入

    def generate_mask(img_tensor):
        """与 Trainer.generate_mask 逻辑完全一致"""
        IM = (downsample_512((img_tensor + 1) / 2) - seg_mean) / seg_std
        down_seg, _, _ = seg(IM)
        current_mask = torch.argmax(down_seg, dim=1).long().float()
        HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        HM_X = F.interpolate(HM_X.unsqueeze(1), size=(256, 256), mode='nearest')
        HM_XD, HM_XE = dilate_erosion.mask(HM_X)
        return HM_XD, HM_XE

    # 收集所有唯一的 (im1, im3) 对
    pairs = set()
    for im1, im2, im3 in exps:
        pairs.add((im1, im3))
        # 反向方向也包含在内，但 im1 和 im3 互换？注意：反向是 (im1, im3, im2)，所以 im1 仍是人脸，im3 是颜色
        # 所以 (im1, im3) 对已经覆盖。无需额外添加。

    print(f"预计算 {len(pairs)} 个 (im1, im3) 对的 mask...")
    for im1, im3 in tqdm(pairs):
        cache_file = os.path.join(cache_dir, f"{im1}_{im3}.pt")
        if os.path.exists(cache_file):
            continue  # 已存在则跳过

        try:
            # 加载图像
            color_img = T.functional.normalize(
                T.functional.to_tensor(Image.open(os.path.join(ffhq_path, f'{im3}.png'))),
                [0.5], [0.5]
            ).unsqueeze(0).to(device)
            face_img = T.functional.normalize(
                T.functional.to_tensor(Image.open(os.path.join(ffhq_path, f'{im1}.png'))),
                [0.5], [0.5]
            ).unsqueeze(0).to(device)

            # 加载 latent
            align_s = torch.from_numpy(
                np.load(os.path.join(dataset_path, 'FS', f'{im1}.npz'))['latent_in']
            ).squeeze(0).unsqueeze(0).to(device)
            align_f = torch.from_numpy(
                np.load(os.path.join(dataset_path, 'Align', f'{im1}_{im3}.npz'))['latent_F']
            ).squeeze(0).unsqueeze(0).to(device)

            # 生成 I_X
            with torch.no_grad():
                I_X, _ = net.generator([align_s], input_is_latent=True, return_latents=False,
                                        start_layer=4, end_layer=8, layer_in=align_f)

            # 生成所需 mask
            HM_3D, HM_3E = generate_mask(color_img)
            HM_1D, _ = generate_mask(face_img)
            HM_XD, HM_XE = generate_mask(I_X)

            # 计算 target_mask
            target_mask = ((1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)).cpu().squeeze(0)

            # 检查有效性（至少有一个非零区域）
            if not (HM_3E.any() and HM_XE.any()):
                print(f"警告：样本 {im1}_{im3} 的 mask 为空，跳过缓存")
                continue

            # 保存为 .pt 文件
            torch.save({
                'target_mask': target_mask,
                'HM_3E': HM_3E.cpu().squeeze(0),
                'HM_XE': HM_XE.cpu().squeeze(0)
            }, cache_file)

        except Exception as e:
            print(f"处理 {im1}_{im3} 时出错：{e}，跳过")
            continue

    print("预计算完成。")


class BlendingDataset(Dataset):
    def __init__(self, exps, dataset_path, ffhq_path, mask_cache_dir):
        """
        Args:
            exps: list of [im1, im2, im3] triplets
            dataset_path: path containing FS/ and Align/ subfolders
            ffhq_path: path to FFHQ images
            mask_cache_dir: directory containing precomputed mask .pt files
        """
        self.dataset_path = dataset_path
        self.ffhq_path = ffhq_path
        # 不再使用 BicubicDownSample，避免 CUDA 多进程问题

        # 加载 mask 缓存
        self.mask_cache = {}
        self.valid_exps = []  # 只保留 mask 缓存存在的样本

        print("加载 mask 缓存并过滤有效样本...")
        for im1, im2, im3 in tqdm(exps):
            # 两个方向都加入
            for (p1, p2, p3) in [(im1, im2, im3), (im1, im3, im2)]:
                key = f"{p1}_{p3}"  # mask 依赖 (人脸, 颜色)
                if key in self.mask_cache:
                    # 已加载过，直接使用
                    self.valid_exps.append((p1, p2, p3))
                else:
                    cache_file = os.path.join(mask_cache_dir, f"{p1}_{p3}.pt")
                    if os.path.exists(cache_file):
                        data = torch.load(cache_file)
                        self.mask_cache[key] = (data['target_mask'], data['HM_3E'], data['HM_XE'])
                        self.valid_exps.append((p1, p2, p3))
                    else:
                        # 缓存不存在，说明该样本无效（预处理时跳过了），忽略
                        pass

        print(f"有效样本数：{len(self.valid_exps)}")

    def __len__(self):
        return len(self.valid_exps)

    def __getitem__(self, idx):
        im1, im2, im3 = self.valid_exps[idx]

        # 加载 latent
        color_path = os.path.join(self.dataset_path, 'FS', f'{im3}.npz')
        Color_S = torch.from_numpy(np.load(color_path)['latent_in']).squeeze(0)

        face_path = os.path.join(self.dataset_path, 'FS', f'{im1}.npz')
        Align_S = torch.from_numpy(np.load(face_path)['latent_in']).squeeze(0)

        # 加载图像（直接为 tensor，范围 [-1,1]）
        Color_I = T.functional.normalize(
            T.functional.to_tensor(Image.open(os.path.join(self.ffhq_path, f'{im3}.png'))),
            [0.5], [0.5]
        )
        Face_I = T.functional.normalize(
            T.functional.to_tensor(Image.open(os.path.join(self.ffhq_path, f'{im1}.png'))),
            [0.5], [0.5]
        )

        align_path = os.path.join(self.dataset_path, 'Align')
        data = np.load(os.path.join(align_path, f'{im1}_{im3}.npz'))
        Align_F = torch.from_numpy(data['latent_F']).squeeze(0)

        # 从缓存获取 mask（已在 CPU）
        target_mask, HM_3E, HM_XE = self.mask_cache[f"{im1}_{im3}"]

        # 使用纯 CPU 操作下采样图像到 256x256（避免 CUDA 多进程冲突）
        Color_I_256 = F.interpolate(Color_I.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)
        Face_I_256 = F.interpolate(Face_I.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)

        return (Color_S, Align_S, Align_F, Color_I_256, Face_I_256,
                target_mask, HM_3E, HM_XE)


class Trainer:
    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 logger=None,
                 fid_dataset='input',
                 save_model_interval=10,
                 save_val_interval=5,
                 val_images_dir='blending_files',
                 models_dir='blending_models'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dilate_erosion = DilateErosion(device=self.device)

        self.fid_dataset = fid_dataset
        self.save_model_interval = save_model_interval
        self.save_val_interval = save_val_interval
        self.val_images_dir = Path(val_images_dir)
        self.models_dir = Path(models_dir)
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.fid_calc = get_fid_calc('input/fid.pkl', self.fid_dataset)

        self.net = Net(Namespace(size=1024, ckpt='pretrained_models/StyleGAN/ffhq.pt', channel_multiplier=2, latent=512,
                                 n_mlp=8, device=self.device))
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.device)
        self.seg.eval()
        self.seg.load_state_dict(torch.load('pretrained_models/BiSeNet/seg.pth'))
        toggle_grad(self.seg, False)
        toggle_grad(self.net.generator, False)

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.downsample_128 = BicubicDownSample(factor=8)

        self.best_loss = float('+inf')
        self.cur_iter = 0

    @torch.no_grad()
    def generate_mask(self, I):
        IM = (self.downsample_512((I + 1) / 2) - seg_mean) / seg_std
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
            torch.save({'model_state_dict': model_state_dict}, f'{tmp_dir}/{name}.pth')
            self.logger.save(f'{tmp_dir}/{name}.pth', save_online)

    def save_checkpoint(self, epoch, optimizer, best_loss, checkpoint_path='checkpoint_blending.pth'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'cur_iter': self.cur_iter,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path, optimizer):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        self.cur_iter = checkpoint.get('cur_iter', 0)
        print(f"Resuming from epoch {start_epoch}, best_loss: {best_loss}, cur_iter: {self.cur_iter}")
        return start_epoch, best_loss

    def calc_loss(self, I_gen, I_face, I_color, mask_face, mask_hair, gen_hair):
        gen_embed = self.model.get_image_embed(I_gen * mask_face)
        gt_embed = self.model.get_image_embed(I_face * mask_face)
        face_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        gen_embed = self.model.get_image_embed(I_gen * mask_hair)
        gt_embed = self.model.get_image_embed(I_color * mask_hair)
        hair_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        losses = {'face loss': face_loss, 'hair loss': hair_loss, 'loss': face_loss + hair_loss}
        return losses['loss'], losses

    def train_one_epoch(self):
        self.model.to(self.device).train()
        for batch in tqdm(self.train_dataloader):
            color_s, align_s, align_f, color_i, face_i, target_mask, HM_3E, HM_XE = map(lambda x: x.to(self.device),
                                                                                        batch)
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E)
            latent_in = torch.cat((torch.zeros(bsz, 6, 512, device=self.device), blend_s), axis=1)
            I_G, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=align_f)

            loss, info = self.calc_loss(self.downsample_256(I_G), face_i, color_i, target_mask, HM_3E, HM_XE)

            self.optimizer.zero_grad()
            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.logger.next_step()
            for key, val in info.items():
                self.logger.log(key, val.item())
            self.logger.log('grad', total_norm.item())
            self.cur_iter += 1

    @torch.no_grad()
    def validate(self, epoch=None):
        self.model.to(self.device).eval()

        sum_losses = lambda x, y: {key: val + x.get(key, 0) for key, val in y.items()}
        files = []
        losses = {}
        to_299 = T.Resize((299, 299))
        images_to_fid = []

        for batch in tqdm(self.test_dataloader):
            color_s, align_s, align_f, color_i, face_i, target_mask, HM_3E, HM_XE = map(lambda x: x.to(self.device),
                                                                                        batch)
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E)
            latent_in = torch.cat((torch.zeros(bsz, 6, 512, device=self.device), blend_s), axis=1)
            I_G, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=align_f)

            _, info = self.calc_loss(self.downsample_256(I_G), face_i, color_i, target_mask, HM_3E, HM_XE)
            losses = sum_losses(losses, info)
            for k in range(bsz):
                files.append([color_i[k].cpu(), face_i[k].cpu(), self.downsample_256(I_G)[k].cpu()])

            images_to_fid.append(to_299((I_G + 1) / 2).clip(0, 1))

        losses['FID CLIP'] = self.fid_calc(torch.cat(images_to_fid))
        for key, val in losses.items():
            if key != 'FID CLIP':
                val = val.item() / len(self.test_dataloader)
            self.logger.log(f'val {key}', val)

        np.random.seed(1927)
        idxs = np.random.choice(len(files), size=100, replace=False)
        images_to_log = [
            image_grid([T.functional.to_pil_image(((img + 1) / 2).clamp(0, 1)) for img in files[idx]], 1, 3) for idx in
            idxs]

        if epoch is not None and epoch % self.save_val_interval == 0:
            save_dir = self.val_images_dir / f"blending_val_{epoch}"
            save_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(images_to_log):
                img.save(save_dir / f"val_sample_{idx}.png")
            print(f"Validation images saved to {save_dir}")

        self.logger.log('val images', [wandb.Image(image) for image in images_to_log])

        return losses['loss']

    def train_loop(self, epochs, start_epoch=0, resume=False):
        if not resume:
            self.validate()

        for epoch in range(start_epoch, epochs):
            self.train_one_epoch()
            loss = self.validate(epoch=epoch)

            self.save_model('last', save_online=False)
            self.save_checkpoint(epoch, self.optimizer, self.best_loss)

            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model('best', save_online=False)

            if epoch % self.save_model_interval == 0:
                model_state_dict = self.model.state_dict()
                for key in list(model_state_dict.keys()):
                    if key.startswith("clip_model."):
                        del model_state_dict[key]
                save_path = self.models_dir / f"blending_{epoch}.pth"
                torch.save({'model_state_dict': model_state_dict}, save_path)
                print(f"Model saved to {save_path}")


def main(cfg):
    seed_everything()

    # 读取三元组列表
    exps = []
    with open(os.path.join(cfg['dataset'], 'dataset.exps'), 'r') as file:
        for exp in file.readlines():
            exps.append(list(map(lambda x: x.replace('.png', ''), exp.split())))

    # 预计算 mask（如果缓存不存在）
    if not os.path.exists(MASK_CACHE_DIR) or len(os.listdir(MASK_CACHE_DIR)) == 0:
        print("未检测到 mask 缓存，开始预计算...")
        precompute_masks(exps, cfg['dataset'], cfg['FFHQ'], MASK_CACHE_DIR)
    else:
        print(f"使用现有 mask 缓存：{MASK_CACHE_DIR}")

    # 划分训练/验证集
    X_train, X_test = train_test_split(exps, test_size=512, random_state=42)

    # 创建数据集（使用预计算 mask）
    train_dataset = BlendingDataset(X_train, cfg['dataset'], cfg['FFHQ'], MASK_CACHE_DIR)
    test_dataset = BlendingDataset(X_test, cfg['dataset'], cfg['FFHQ'], MASK_CACHE_DIR)

    # 优化 DataLoader 配置
    num_workers = min(4, os.cpu_count())  # 根据 CPU 核心数调整
    train_dataloader = DataLoader(
        train_dataset, batch_size=10, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=10, shuffle=False,
        num_workers=num_workers // 2, pin_memory=True
    )

    # 初始化 logger 和模型
    logger = WandbLogger(name=cfg['name_run'], project='Barbershop-Blending')
    logger.start_logging()
    logger.save(__file__)

    model = BlendingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.000001)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        logger=logger,
        fid_dataset=cfg['fid_dataset'],
        save_model_interval=cfg['save_model_interval'],
        save_val_interval=cfg['save_val_interval'],
        val_images_dir=cfg['val_images_dir'],
        models_dir=cfg['models_dir']
    )

    start_epoch = 0
    resume = False
    if cfg['resume'] and os.path.exists(cfg['checkpoint_path']):
        start_epoch, trainer.best_loss = trainer.load_checkpoint(cfg['checkpoint_path'], optimizer)
        resume = True

    trainer.train_loop(cfg['epochs'], start_epoch=start_epoch, resume=resume)
    logger.wandb.finish()


if __name__ == '__main__':
    main(config)