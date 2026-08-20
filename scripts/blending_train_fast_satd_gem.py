import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

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
from models.SATD import SATD
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.train import toggle_grad, WandbLogger, image_grid, seed_everything, get_fid_calc

# ================== 用户配置区域 ==================
config = {
    'name_run': 'blending_train_satd_final_fix',
    'dataset': './images/blending',
    'FFHQ': './images/FFHQ',
    'fid_dataset': './celeba-1024',
    'epochs': 200,
    'checkpoint_path': './checkpoints/checkpoint_satd_blending.pth',
    'satd_checkpoint': 'pretrained_models/SATD/satd.pth',
    'resume': True,
    'save_model_interval': 10,
    'save_val_interval': 2,
    'models_dir': 'blending_satd__final_models',
    'val_images_dir': 'blending_satd_final_files',
    'batch_size': 12
}
# =================================================

MASK_CACHE_DIR = os.path.join(config['dataset'], 'mask_cache')

def precompute_masks(exps, dataset_path, ffhq_path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = Net(Namespace(size=1024, ckpt='pretrained_models/StyleGAN/ffhq.pt',
                        channel_multiplier=2, latent=512, n_mlp=8, device=device))
    seg = BiSeNet(n_classes=16).to(device).eval()
    seg.load_state_dict(torch.load('pretrained_models/BiSeNet/seg.pth'))
    toggle_grad(seg, False)
    toggle_grad(net.generator, False)

    dilate_erosion = DilateErosion(device=device)
    downsample_512 = BicubicDownSample(factor=2)

    def generate_mask(img_tensor):
        IM = (downsample_512((img_tensor + 1) / 2) - seg_mean) / seg_std
        down_seg, _, _ = seg(IM)
        current_mask = torch.argmax(down_seg, dim=1).long().float()
        # 10 在此特定 16-class BiSeNet 中是头发
        HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        HM_X_raw = F.interpolate(HM_X.unsqueeze(1), size=(256, 256), mode='nearest')
        HM_XD, HM_XE = dilate_erosion.mask(HM_X_raw)
        return HM_XD, HM_XE, HM_X_raw

    pairs = set((im1, im3) for im1, _, im3 in exps)

    print(f"预计算 {len(pairs)} 个 (im1, im3) 对的 mask...")
    for im1, im3 in tqdm(pairs):
        cache_file = os.path.join(cache_dir, f"{im1}_{im3}.pt")
        if os.path.exists(cache_file): continue

        try:
            color_img = T.functional.normalize(T.functional.to_tensor(Image.open(os.path.join(ffhq_path, f'{im3}.png'))), [0.5], [0.5]).unsqueeze(0).to(device)
            face_img = T.functional.normalize(T.functional.to_tensor(Image.open(os.path.join(ffhq_path, f'{im1}.png'))), [0.5], [0.5]).unsqueeze(0).to(device)

            npz_fs = np.load(os.path.join(dataset_path, 'FS', f'{im1}.npz'))
            align_s_key = 'S' if 'S' in npz_fs else 'latent_in'
            align_s = torch.from_numpy(npz_fs[align_s_key]).squeeze(0).unsqueeze(0).to(device)
            
            npz_data = np.load(os.path.join(dataset_path, 'Align', f'{im1}_{im3}.npz'))
            align_key = 'latent_F_align' if 'latent_F_align' in npz_data else 'latent_F'
            align_f = torch.from_numpy(npz_data[align_key]).squeeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                I_X, _ = net.generator([align_s], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=align_f)

            # 提取原图掩码，目标图掩码
            _, HM_3E, HM_3D = generate_mask(color_img)
            HM_1D, _, _ = generate_mask(face_img)
            HM_XD, HM_XE, _ = generate_mask(I_X)

            target_mask = ((1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)).cpu().squeeze(0)

            torch.save({
                'target_mask': target_mask,
                'HM_3E': HM_3E.cpu().squeeze(0),
                'HM_XE': HM_XE.cpu().squeeze(0),
                'HM_1D': HM_1D.cpu().squeeze(0),
                'HM_3D': HM_3D.cpu().squeeze(0) # [修复1] 保存纯净的目标头发Mask
            }, cache_file)
        except Exception as e:
            print(f"处理 {im1}_{im3} 时出错：{e}，跳过")
            continue


class BlendingDataset(Dataset):
    def __init__(self, exps, dataset_path, ffhq_path, mask_cache_dir):
        self.dataset_path = dataset_path
        self.ffhq_path = ffhq_path
        self.mask_cache = {}
        self.valid_exps = []

        for im1, im2, im3 in exps:
            for (p1, p2, p3) in [(im1, im2, im3), (im1, im3, im2)]:
                key = f"{p1}_{p3}"
                if key in self.mask_cache:
                    self.valid_exps.append((p1, p2, p3))
                else:
                    cache_file = os.path.join(mask_cache_dir, f"{p1}_{p3}.pt")
                    if os.path.exists(cache_file):
                        data = torch.load(cache_file)
                        self.mask_cache[key] = (data['target_mask'], data['HM_3E'], data['HM_XE'], data['HM_1D'], data.get('HM_3D', data['HM_3E']))
                        self.valid_exps.append((p1, p2, p3))

    def __len__(self): 
        return len(self.valid_exps)

    def __getitem__(self, idx):
        im1, im2, im3 = self.valid_exps[idx]

        face_path = os.path.join(self.dataset_path, 'FS', f'{im1}.npz')
        Align_S = torch.from_numpy(np.load(face_path)['latent_in']).squeeze(0)
        
        color_path = os.path.join(self.dataset_path, 'FS', f'{im3}.npz')
        Color_S = torch.from_numpy(np.load(color_path)['latent_in']).squeeze(0)

        align_npz = np.load(os.path.join(self.dataset_path, 'Align', f'{im1}_{im3}.npz'))
        F_sean1 = torch.from_numpy(align_npz['F_sean1']).squeeze(0)
        F_sean2 = torch.from_numpy(align_npz['F_sean2']).squeeze(0)

        Color_I = T.functional.normalize(T.functional.to_tensor(Image.open(os.path.join(self.ffhq_path, f'{im3}.png'))), [0.5], [0.5])
        Face_I = T.functional.normalize(T.functional.to_tensor(Image.open(os.path.join(self.ffhq_path, f'{im1}.png'))), [0.5], [0.5])

        target_mask, HM_3E, HM_XE, HM_1D, HM_3D = self.mask_cache[f"{im1}_{im3}"]

        Color_I_256 = F.interpolate(Color_I.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)
        Face_I_256 = F.interpolate(Face_I.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)

        return (Color_S, Align_S, F_sean1, F_sean2, Color_I_256, Face_I_256, target_mask, HM_3E, HM_XE, HM_1D, HM_3D, im1, im3)


class Trainer:
    def __init__(self, model=None, optimizer=None, scheduler=None, train_dataloader=None, test_dataloader=None, logger=None, fid_dataset='input', save_model_interval=10, save_val_interval=5, val_images_dir='blending_files', models_dir='blending_models', satd_checkpoint=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.save_model_interval = save_model_interval
        self.save_val_interval = save_val_interval
        self.val_images_dir = Path(val_images_dir)
        self.models_dir = Path(models_dir)
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.fid_calc = get_fid_calc('input/fid.pkl', fid_dataset)

        self.net = Net(Namespace(size=1024, ckpt='pretrained_models/StyleGAN/ffhq.pt', channel_multiplier=2, latent=512, n_mlp=8, device=self.device))
        self.seg = BiSeNet(n_classes=16).to(self.device).eval()
        self.seg.load_state_dict(torch.load('pretrained_models/BiSeNet/seg.pth'))

        self.satd = SATD(feat_ch=512, mask_ch=3, mid_ch=128).to(self.device)
        if satd_checkpoint and os.path.exists(satd_checkpoint):
            self.satd.load_state_dict(torch.load(satd_checkpoint, map_location=self.device))
            print(f"成功加载 SATD 权重: {satd_checkpoint}")

        toggle_grad(self.net.generator, False)
        toggle_grad(self.seg, False)
        if self.model is not None:
            self.model.to(self.device)
            toggle_grad(self.model, False)
        toggle_grad(self.satd, True)

        self.downsample_256 = BicubicDownSample(factor=4)
        self.best_loss = float('+inf')
        self.cur_iter = 0

    def save_model(self, name, save_online=True):
        with TemporaryDirectory() as tmp_dir:
            torch.save({'satd_state_dict': self.satd.state_dict()}, f'{tmp_dir}/{name}.pth')
            self.logger.save(f'{tmp_dir}/{name}.pth', save_online)

    def save_checkpoint(self, epoch, optimizer, best_loss, checkpoint_path='checkpoint_blending_satd.pth'):
        torch.save({'epoch': epoch, 'satd_state_dict': self.satd.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss, 'cur_iter': self.cur_iter}, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, optimizer):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.satd.load_state_dict(checkpoint['satd_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor): state[k] = v.to(self.device)
        self.cur_iter = checkpoint.get('cur_iter', 0)
        return checkpoint['epoch'] + 1, checkpoint['best_loss']

    def calc_loss(self, I_gen, I_face, I_color, target_mask, HM_3E, HM_1D, HM_3D, F_sean1, latent_F_align):
        gen_embed = self.model.get_image_embed(I_gen * target_mask)
        gt_embed = self.model.get_image_embed(I_face * target_mask)
        face_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        gen_embed = self.model.get_image_embed(I_gen * HM_3E)
        gt_embed_color = self.model.get_image_embed(I_color * HM_3E)
        hair_loss = (1 - F.cosine_similarity(gen_embed, gt_embed_color)).mean()

        safe_skin_mask = (1 - HM_1D) * (1 - HM_3D)
        l_skin_preserve = F.l1_loss(I_gen * safe_skin_mask, I_face * safe_skin_mask)

        # [修复2] 精准定位幽灵阴影区：源图有长发，但目标图没有长发的地方
        exposed_mask = HM_1D * (1 - HM_3D)
        if exposed_mask.dim() == 3: exposed_mask = exposed_mask.unsqueeze(1)
        exposed_mask_32 = F.interpolate(exposed_mask.float(), size=(32, 32), mode='nearest')
        l_exposed_guidance = F.mse_loss(latent_F_align * exposed_mask_32, F_sean1.detach() * exposed_mask_32)

        # [修复3] 增强像素级颜色对齐，解决颜色泄漏
        l_texture = F.l1_loss(I_gen * HM_3E, I_color * HM_3E)

        total_loss = face_loss + 1.2 * hair_loss + 0.3 * l_skin_preserve + 0.8 * l_exposed_guidance + 0.8 * l_texture
        losses = {'face loss': face_loss, 'hair loss': hair_loss, 'skin preserve': l_skin_preserve, 'exposed guide': l_exposed_guidance, 'texture loss': l_texture, 'loss': total_loss}
        return total_loss, losses

    def train_one_epoch(self):
        self.satd.train()
        self.model.eval()
        
        for batch in tqdm(self.train_dataloader):
            color_s, align_s, F_sean1, F_sean2, color_i, face_i, target_mask, HM_3E, HM_XE, HM_1D, HM_3D, im1, im3 = batch
            color_s, align_s, F_sean1, F_sean2 = color_s.to(self.device), align_s.to(self.device), F_sean1.to(self.device), F_sean2.to(self.device)
            color_i, face_i, target_mask, HM_3E, HM_XE, HM_1D, HM_3D = color_i.to(self.device), face_i.to(self.device), target_mask.to(self.device), HM_3E.to(self.device), HM_XE.to(self.device), HM_1D.to(self.device), HM_3D.to(self.device)
            bsz = color_s.size(0)

            with torch.no_grad():
                F_face, _ = self.net.generator([align_s], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
                F_hair, _ = self.net.generator([color_s], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E)
            
            # [修复4] 终极 Mask 逻辑修复，彻底解决鼓包和多发 Bug！
            mask_union = 1 - (1 - HM_1D) * (1 - HM_3D)
            mask_target = HM_3D
            mask_inter = HM_3D
            
            if mask_union.dim() == 3:
                masks_3ch = torch.cat([mask_union.unsqueeze(1), mask_target.unsqueeze(1), mask_inter.unsqueeze(1)], dim=1).float()
            else:
                masks_3ch = torch.cat([mask_union, mask_target, mask_inter], dim=1).float()
                
            masks_32 = F.interpolate(masks_3ch, size=(32, 32), mode='bicubic', align_corners=False)
            
            latent_F_align = self.satd(F_face=F_face, F_sean1=F_sean1, F_sean2=F_sean2, F_hair=F_hair, masks_hw=masks_32)
            
            latent_in = torch.cat((torch.zeros(bsz, 6, 512, device=self.device), blend_s), axis=1)
            I_G, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=latent_F_align)

            loss, info = self.calc_loss(self.downsample_256(I_G), face_i, color_i, target_mask, HM_3E, HM_1D, HM_3D, F_sean1, latent_F_align)

            self.optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(self.satd.parameters(), 5)
            self.optimizer.step()

            self.logger.next_step()
            for key, val in info.items(): self.logger.log(key, val.item())
            self.logger.log('grad', total_norm.item())
            self.cur_iter += 1

    @torch.no_grad()
    def validate(self, epoch=None):
        self.satd.eval()
        self.model.eval()

        sum_losses = lambda x, y: {key: val + x.get(key, 0) for key, val in y.items()}
        files, losses, images_to_fid = [], {}, []
        to_299 = T.Resize((299, 299))

        for batch in tqdm(self.test_dataloader):
            batch_on_device = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
            color_s, align_s, F_sean1, F_sean2, color_i, face_i, target_mask, HM_3E, HM_XE, HM_1D, HM_3D, im1, im3 = batch_on_device
            bsz = color_s.size(0)

            F_face, _ = self.net.generator([align_s], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
            F_hair, _ = self.net.generator([color_s], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E)
            
            mask_union = 1 - (1 - HM_1D) * (1 - HM_3D)
            mask_target = HM_3D
            mask_inter = HM_3D
            
            if mask_union.dim() == 3:
                masks_3ch = torch.cat([mask_union.unsqueeze(1), mask_target.unsqueeze(1), mask_inter.unsqueeze(1)], dim=1).float()
            else:
                masks_3ch = torch.cat([mask_union, mask_target, mask_inter], dim=1).float()
                
            masks_32 = F.interpolate(masks_3ch, size=(32, 32), mode='bicubic', align_corners=False)
            
            latent_F_align = self.satd(F_face=F_face, F_sean1=F_sean1, F_sean2=F_sean2, F_hair=F_hair, masks_hw=masks_32)
            
            latent_in = torch.cat((torch.zeros(bsz, 6, 512, device=self.device), blend_s), axis=1)
            I_G, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=latent_F_align)

            _, info = self.calc_loss(self.downsample_256(I_G), face_i, color_i, target_mask, HM_3E, HM_1D, HM_3D, F_sean1, latent_F_align)
            losses = sum_losses(losses, info)
            for k in range(bsz): files.append([color_i[k].cpu(), face_i[k].cpu(), self.downsample_256(I_G)[k].cpu()])
            images_to_fid.append(to_299((I_G + 1) / 2).clip(0, 1))

        losses['FID CLIP'] = self.fid_calc(torch.cat(images_to_fid))
        for key, val in losses.items():
            if key != 'FID CLIP': val = val.item() / len(self.test_dataloader)
            self.logger.log(f'val {key}', val)

        np.random.seed(1927)
        idxs = np.random.choice(len(files), size=min(100, len(files)), replace=False)
        images_to_log = [image_grid([T.functional.to_pil_image(((img + 1) / 2).clamp(0, 1)) for img in files[idx]], 1, 3) for idx in idxs]

        if epoch is not None and epoch % self.save_val_interval == 0:
            save_dir = self.val_images_dir / f"blending_val_{epoch}"
            save_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(images_to_log): img.save(save_dir / f"val_sample_{idx}.png")
            print(f"Validation images saved to {save_dir}")

        self.logger.log('val images', [wandb.Image(image) for image in images_to_log])
        return losses['loss']

    def train_loop(self, epochs, start_epoch=0, resume=False):
        if not resume: self.validate()
        for epoch in range(start_epoch, epochs):
            self.train_one_epoch()
            loss = self.validate(epoch=epoch)
            self.save_model('last', save_online=False)
            self.save_checkpoint(epoch, self.optimizer, self.best_loss)
            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model('best', save_online=False)
            if epoch % self.save_model_interval == 0:
                save_path = self.models_dir / f"satd_blending_{epoch}.pth"
                torch.save({'satd_state_dict': self.satd.state_dict()}, save_path)
                print(f"SATD model saved to {save_path}")

def main(cfg):
    seed_everything()
    exps = []
    with open(os.path.join(cfg['dataset'], 'dataset.exps'), 'r') as file:
        for exp in file.readlines(): exps.append(list(map(lambda x: x.replace('.png', ''), exp.split())))

    # 只要改了缓存逻辑，就必须删除老的缓存重新生成！
    if not os.path.exists(MASK_CACHE_DIR) or len(os.listdir(MASK_CACHE_DIR)) == 0:
        print("未检测到完整的 mask 缓存，开始预计算...")
        precompute_masks(exps, cfg['dataset'], cfg['FFHQ'], MASK_CACHE_DIR)
    
    X_train, X_test = train_test_split(exps, test_size=512, random_state=42)
    train_dataset = BlendingDataset(X_train, cfg['dataset'], cfg['FFHQ'], MASK_CACHE_DIR)
    test_dataset = BlendingDataset(X_test, cfg['dataset'], cfg['FFHQ'], MASK_CACHE_DIR)

    num_workers = min(8, os.cpu_count())
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=num_workers // 2, pin_memory=True)

    logger = WandbLogger(name=cfg['name_run'], project='Barbershop-Blending-SATD')
    logger.start_logging()
    logger.save(__file__)

    model = BlendingModel()
    trainer = Trainer(model=model, optimizer=None, scheduler=None, train_dataloader=train_dataloader, test_dataloader=test_dataloader, logger=logger, fid_dataset=cfg['fid_dataset'], save_model_interval=cfg['save_model_interval'], save_val_interval=cfg['save_val_interval'], val_images_dir=cfg['val_images_dir'], models_dir=cfg['models_dir'], satd_checkpoint=cfg['satd_checkpoint'])
    
    optimizer = torch.optim.Adam(trainer.satd.parameters(), lr=1e-4, weight_decay=1e-6)
    trainer.optimizer = optimizer

    start_epoch, resume = 0, False
    if cfg['resume'] and os.path.exists(cfg['checkpoint_path']):
        start_epoch, trainer.best_loss = trainer.load_checkpoint(cfg['checkpoint_path'], optimizer)
        resume = True

    trainer.train_loop(cfg['epochs'], start_epoch=start_epoch, resume=resume)
    logger.wandb.finish()

if __name__ == '__main__':
    main(config)