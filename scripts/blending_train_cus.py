import os
import sys
from argparse import Namespace 
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from joblib import Parallel, delayed
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
    'name_run': 'blending_train',                         # wandb 运行名称
    'dataset': './images/blending',         # 数据集路径（包含 datasets）
    'FFHQ': './images/FFHQ',              # FFHQ 图像路径，需替换为实际路径
    'fid_dataset': './fid_images',                       # FID 计算所需的数据集路径
    'epochs': 181,                                # 训练总轮数
    'checkpoint_path': 'checkpoint_blending.pth', # 断点保存路径
    'resume': False,                               # 是否从断点恢复
    'save_model_interval': 10,                     # 每多少轮保存一次 blending_n.pth
    'save_val_interval': 1,                         # 每多少轮保存一次验证图片
    'models_dir': 'blending_models',                # 模型权重保存目录
    'val_images_dir': 'blending_files',             # 验证图片保存根目录
}
# =================================================


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

        # 新增配置
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

            # delete pretrained clip
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

        # 根据间隔保存验证图片到本地目录
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

            # 保存 last 和 best 模型（原有功能）
            self.save_model('last', save_online=False)
            self.save_checkpoint(epoch, self.optimizer, self.best_loss)
            
            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model(f'best', save_online=False)

            # 新增：按间隔保存模型权重
            if epoch % self.save_model_interval == 0:
                model_state_dict = self.model.state_dict()
                # 删除 pretrained clip 键
                for key in list(model_state_dict.keys()):
                    if key.startswith("clip_model."):
                        del model_state_dict[key]
                save_path = self.models_dir / f"blending_{epoch}.pth"
                torch.save({'model_state_dict': model_state_dict}, save_path)
                print(f"Model saved to {save_path}")


def prepare_item(exp, path, ffhq_path):
    im1, im2, im3 = exp

    try:
        color_path = os.path.join(path, 'FS', f'{im3}.npz')
        Color_S = torch.from_numpy(np.load(color_path)['latent_in']).squeeze(0)

        face_path = os.path.join(path, 'FS', f'{im1}.npz')
        Align_S = torch.from_numpy(np.load(face_path)['latent_in']).squeeze(0)

        Color_I = T.functional.normalize(T.functional.to_tensor(
            Image.open(os.path.join(ffhq_path, f'{im3}.png'))
        ), [0.5], [0.5])
        Face_I = T.functional.normalize(T.functional.to_tensor(
            Image.open(os.path.join(ffhq_path, f'{im1}.png'))
        ), [0.5], [0.5])

        align_path = os.path.join(path, 'Align')
        data = np.load(
            os.path.join(align_path, f'{im1}_{im3}.npz')
        )
        Align_F = torch.from_numpy(data['latent_F']).squeeze(0)

        return (Color_S, Align_S, Align_F, Color_I, Face_I)
    except Exception as e:
        print(e, file=sys.stderr)
        return None


class Blending_dataset(Dataset):
    def __init__(self, exps, path, net_trainer,ffhq_path):
        """
        Args:
            exps: list of [im1, im2, im3] triplets
            path: dataset path containing FS/ and Align/ subfolders
            ffhq_path: path to FFHQ images
            net_trainer: Trainer instance (used for mask generation)
        """
        self.exps = []
        for (p1, p2, p3) in tqdm(exps):
            # 每个三元组生成两个方向：(im1, im2, im3) 和 (im1, im3, im2)
            self.exps.append((p1, p2, p3))
            self.exps.append((p1, p3, p2))
        self.path = path
        self.ffhq_path = ffhq_path
        self.net_trainer = net_trainer
        self.downsample_256 = BicubicDownSample(factor=4)

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        im1, im2, im3 = self.exps[idx]

        # 加载 latent
        color_path = os.path.join(self.path, 'FS', f'{im3}.npz')
        Color_S = torch.from_numpy(np.load(color_path)['latent_in']).squeeze(0)

        face_path = os.path.join(self.path, 'FS', f'{im1}.npz')
        Align_S = torch.from_numpy(np.load(face_path)['latent_in']).squeeze(0)

        # 加载图像
        Color_I = T.functional.normalize(
            T.functional.to_tensor(Image.open(os.path.join(self.ffhq_path, f'{im3}.png'))),
            [0.5], [0.5]
        )
        Face_I = T.functional.normalize(
            T.functional.to_tensor(Image.open(os.path.join(self.ffhq_path, f'{im1}.png'))),
            [0.5], [0.5]
        )

        align_path = os.path.join(self.path, 'Align')
        data = np.load(os.path.join(align_path, f'{im1}_{im3}.npz'))
        Align_F = torch.from_numpy(data['latent_F']).squeeze(0)

        # 生成 mask（注意需要将图像移到 GPU）
        Color_I_gpu = Color_I.unsqueeze(0).to('cuda')
        Face_I_gpu = Face_I.unsqueeze(0).to('cuda')

        HM_3D, HM_3E = self.net_trainer.generate_mask(Color_I_gpu)
        HM_1D, _ = self.net_trainer.generate_mask(Face_I_gpu)

        I_X, _ = self.net_trainer.net.generator(
            [Align_S.unsqueeze(0).to('cuda')],
            input_is_latent=True, return_latents=False,
            start_layer=4, end_layer=8,
            layer_in=Align_F.unsqueeze(0).to('cuda')
        )
        HM_XD, HM_XE = self.net_trainer.generate_mask(I_X)

        target_mask = ((1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)).cpu().squeeze(0)
        HM_3E = HM_3E.cpu().squeeze(0)
        HM_XE = HM_XE.cpu().squeeze(0)

        # 检查 mask 是否有效（至少有一个非零区域）
        if not (HM_3E.any() and HM_XE.any()):
            # 如果无效，返回下一个有效样本（可自定义处理，这里简单跳过）
            # 实际训练中，可以通过自定义 collate_fn 过滤，或者这里递归查找下一个
            return self.__getitem__((idx + 1) % len(self))

        # 下采样图像到 256x256
        Color_I_256 = self.downsample_256(Color_I_gpu).cpu().squeeze(0)
        Face_I_256 = self.downsample_256(Face_I_gpu).cpu().squeeze(0)

        return (Color_S, Align_S, Align_F, Color_I_256, Face_I_256, target_mask, HM_3E, HM_XE)


def main(cfg):
    seed_everything()

    exps = []
    with open(os.path.join(cfg['dataset'], 'dataset.exps'), 'r') as file:
        for exp in file.readlines():
            exps.append(list(map(lambda x: x.replace('.png', ''), exp.split())))

    X_train, X_test = train_test_split(exps, test_size=512, random_state=42)

    net_trainer = Trainer()  # 临时用于生成 mask
    train_dataset = Blending_dataset(X_train, cfg['dataset'], net_trainer, cfg['FFHQ'])
    test_dataset = Blending_dataset(X_test, cfg['dataset'], net_trainer, cfg['FFHQ'])

    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=False)

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