import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Encoders import ClipBlendingModel
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.image_utils import DilateErosion, equal_replacer
from utils.bicubic import BicubicDownSample
from utils.train import seed_everything
from models.Embedding import Embedding
from models.Alignment import Alignment
from hair_swap import get_parser

# 导入torchmetrics用于FID计算
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available. Only FID_CLIP will be calculated.")


# ================== 用户配置区域 ==================
config = {
    # 模型路径配置#
    'blending_checkpoint': "blending_satd_v5_models/satd_blending_35.pth",   # blending模型路径
    'stylegan_checkpoint': 'pretrained_models/StyleGAN/ffhq.pt',  # StyleGAN模型路径
    'bisenet_checkpoint': 'pretrained_models/BiSeNet/seg.pth',  # BiSeNet分割模型路径
    'e4e_checkpoint': 'pretrained_models/encoder4editing/e4e_ffhq_encode.pt',  # e4e编码器路径
    
    # 数据配置
    'fid_images_dir': "./celeba-1024",  # 原始图片目录
    'num_samples': 1000,  # 使用的样本数量（组成多少组三元组）
    
    # 随机种子配置
    'random_seed': 42,  # 随机种子，保证可重复性
    
    # 设备配置
    'device': 'cuda',  # 运行设备: 'cuda' 或 'cpu'
    
    # 输出配置
    'output_dir': 'fid_calculation_results_v5_20pth',  # 结果保存目录
    'save_generated_images': True,  # 是否保存生成的图像
    
    # 模型参数
    'image_size': 1024,
    'channel_multiplier': 2,
    'latent_dim': 512,
    'n_mlp': 8,
    'smooth': 5,  # 膨胀腐蚀参数
}
# =================================================


class ClipModel(torch.nn.Module):
    """CLIP模型用于FID_CLIP计算"""
    def __init__(self):
        super().__init__()
        import clip
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.model.eval()
        
    def forward(self, x):
        # x: [B, 3, 299, 299] in [0, 1] or [0, 255]
        # Ensure float type and same device
        x = x.float().to(self.model.visual.conv1.weight.device)
        # CLIP expects [B, 3, 224, 224]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize for CLIP
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.model.encode_image(x).float()


class FIDCalculator:
    """FID计算器，同时计算传统FID和FID_CLIP"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 初始化FID计算器
        self._init_fid_calculators()
        
        # 加载模型
        self._load_models()
        
        # 图像预处理
        self.to_299 = T.Resize((299, 299))
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
    def _init_fid_calculators(self):
        """初始化FID和FID_CLIP计算器"""
        print("Initializing FID calculators...")
        
        # FID_CLIP (基于CLIP)
        self.fid_clip = self._create_fid_clip_calculator()
        
        # 传统FID (基于InceptionV3)
        if TORCHMETRICS_AVAILABLE:
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
            self.fid.to(self.device).eval()
            print("Traditional FID calculator initialized")
        else:
            self.fid = None
            print("Traditional FID calculator not available (torchmetrics required)")
    
    def _create_fid_clip_calculator(self):
        """创建FID_CLIP计算器"""
        from torchmetrics.image.fid import FrechetInceptionDistance
        
        fid_clip = FrechetInceptionDistance(
            feature=ClipModel(), 
            reset_real_features=False, 
            normalize=True
        )
        fid_clip.to(self.device).eval()
        
        return fid_clip
    
    def _load_models(self):
        """加载所有需要的模型"""
        print("Loading models...")
        
        # 加载模型参数
        model_args = get_parser()
        model_args = model_args.parse_args([])
        model_args.device = self.device
        model_args.blending_checkpoint = self.config['blending_checkpoint']
        model_args.size = self.config['image_size']
        model_args.ckpt = self.config['stylegan_checkpoint']
        model_args.channel_multiplier = self.config['channel_multiplier']
        model_args.latent = self.config['latent_dim']
        model_args.n_mlp = self.config['n_mlp']
        model_args.smooth = self.config['smooth']
        model_args.batch_size = 1
        
        # 加载StyleGAN网络
        self.net = Net(model_args)
        
        # 加载blending编码器
        blending_checkpoint_data = torch.load(self.config['blending_checkpoint'], map_location=self.device)
        self.blending_encoder = ClipBlendingModel(blending_checkpoint_data.get('clip', "ViT-B/32"))
        if 'model_state_dict' in blending_checkpoint_data:
            self.blending_encoder.load_state_dict(blending_checkpoint_data['model_state_dict'], strict=False)
            print("✅ 成功加载微调后的颜色模型权重！")
        else:
            print("⚠️ 警告：当前 Checkpoint 中没有 model_state_dict，将使用默认颜色权重（因为这是修复颜色 Bug 之前的旧模型）。")
        self.blending_encoder.to(self.device).eval()
        
        # 加载embedding和对齐模块
        self.embedding_model = Embedding(model_args, net=self.net)
        self.alignment_model = Alignment(model_args, self.embedding_model.get_e4e_embed, net=self.net)
        
        # 初始化辅助工具
        self.dilate_erosion = DilateErosion(
            dilate_erosion=self.config['smooth'], 
            device=self.device
        )
        self.downsample_256 = BicubicDownSample(factor=4)
        self.downsample_512 = BicubicDownSample(factor=2)
        
        # 加载BiSeNet用于分割
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.device)
        self.seg.load_state_dict(torch.load(self.config['bisenet_checkpoint'], map_location=self.device))
        self.seg.eval()
        
        # 固定不需要训练的参数
        from utils.train import toggle_grad
        toggle_grad(self.seg, False)
        toggle_grad(self.net.generator, False)
        toggle_grad(self.blending_encoder, False)
        
        print("Models loaded successfully!")
    
    def update_real_features(self, real_images_dir):
        """分批读取真实图像并更新FID统计，避免一次性加载所有图像"""
        print(f"\nUpdating real features from: {real_images_dir}")
        
        image_paths = self._load_image_paths(real_images_dir)
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {real_images_dir}")
        
        # 限制图像数量（可选项，若目录太大可设置上限）
        max_real_images = 30000
        if len(image_paths) > max_real_images:
            print(f"Using first {max_real_images} images for FID calculation")
            image_paths = image_paths[:max_real_images]
        
        print(f"Found {len(image_paths)} real images")
        
        # 重置统计量
        self.fid_clip.reset()
        if self.fid is not None:
            self.fid.reset()
        
        # 分批处理
        batch_size = 32
        batch_buffer = []
        
        for img_path in tqdm(image_paths, desc="Processing real images"):
            img = self._load_image(img_path)                # [3, H, W] in [0,1]
            img_299 = self.to_299(img.unsqueeze(0))         # [1, 3, 299, 299]
            batch_buffer.append(img_299)
            
            if len(batch_buffer) >= batch_size:
                batch = torch.cat(batch_buffer, dim=0).to(self.device)
                # 更新FID_CLIP
                self.fid_clip.update(batch, real=True)
                # 更新传统FID
                if self.fid is not None:
                    self.fid.update(batch, real=True)
                batch_buffer = []      # 清空缓冲区，释放显存
        
        # 处理剩余不足一个批次的图像
        if batch_buffer:
            batch = torch.cat(batch_buffer, dim=0).to(self.device)
            self.fid_clip.update(batch, real=True)
            if self.fid is not None:
                self.fid.update(batch, real=True)
        
        print("Real features updated successfully!")
    
    def _load_image_paths(self, fid_dir):
        """加载目录中的所有图像路径"""
        fid_path = Path(fid_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(fid_path.glob(f'*{ext}')))
            image_paths.extend(list(fid_path.glob(f'*{ext.upper()}')))
        
        # 去重并排序
        image_paths = sorted(list(set([str(p) for p in image_paths])))
        
        return image_paths
    
    def _load_image(self, image_path):
        """加载并预处理单张图像"""
        from torchvision.io import read_image, ImageReadMode
        
        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        image = image.float() / 255.0  # 归一化到[0, 1]
        
        return image.to(self.device)
    
    def get_image_embedding(self, image_tensor, name):
        """获取单张图像的embedding"""
        from collections import defaultdict
        from models.Embedding import get_latents, get_segmentation
        
        images_to_name = defaultdict(list)
        images_to_name[image_tensor].append(name)
        
        self.embedding_model.setup_dataloader(images_to_name)
        
        name_to_embed = defaultdict(dict)
        for batch_image, names in self.embedding_model.dataloader:
            batch_image = batch_image.to(self.device)
            # 确保 img_name 是字符串
            img_name = names[0] if isinstance(names[0], str) else names[0][0]
            
            im_512 = self.embedding_model.downsample_512(batch_image)
            im_256 = self.embedding_model.downsample_256(batch_image)
            im_256_norm = self.embedding_model.normalize(im_256)
            
            # E4E
            latent_W = get_latents(self.embedding_model.e4e, im_256_norm)
            
            # FS encoder
            output = self.embedding_model.encoder.test(
                img=self.embedding_model.normalize(batch_image), 
                return_latent=True
            )
            latent = output.pop()
            latent_S = output.pop()
            
            latent_F, _ = self.net.generator(
                [latent_S], 
                input_is_latent=True, 
                return_latents=False,
                start_layer=3, 
                end_layer=3, 
                layer_in=latent
            )
            
            # BiSeNet分割
            masks = torch.cat([
                get_segmentation(batch_image.unsqueeze(0)) 
                for batch_image in self.embedding_model.to_bisenet(im_512)
            ])
            
            name_to_embed[img_name] = {
                'image': batch_image,
                'image_256': im_256,
                'image_norm_256': im_256_norm,
                'S': latent_S,
                'W': latent_W,
                'F': latent_F,
                'mask': masks
            }
        
        return name_to_embed[name]
    
    def blend_images(self, name_to_embed):
        """使用blending模型处理图像"""
        I_1 = name_to_embed['face']['image_norm_256']
        I_2 = name_to_embed['shape']['image_norm_256']
        I_3 = name_to_embed['color']['image_norm_256']
        
        mask_de = self.dilate_erosion.hair_from_mask(
            torch.cat([name_to_embed[x]['mask'] for x in ['face', 'color']], dim=0)
        )
        HM_1D, _ = mask_de[0][0].unsqueeze(0), mask_de[1][0].unsqueeze(0)
        HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)
        
        latent_S_1 = name_to_embed['face']['S']
        latent_F_align = name_to_embed['shape']['latent_F_align']
        HM_X = name_to_embed['color']['HM_X']
        
        latent_S_3 = name_to_embed['color']["S"]
        
        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)
        
        # Blending
        with torch.no_grad():
            if I_1 is not I_3 or I_1 is not I_2:
                S_blend_6_18 = self.blending_encoder(
                    latent_S_1[:, 6:], 
                    latent_S_3[:, 6:], 
                    I_1 * target_mask, 
                    I_3 * HM_3E
                )
                S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
            else:
                S_blend = latent_S_1
            
            I_blend, _ = self.net.generator(
                [S_blend], 
                input_is_latent=True, 
                return_latents=False, 
                start_layer=4,
                end_layer=8, 
                layer_in=latent_F_align
            )
        
        # 归一化到[0, 1]
        final_image = ((I_blend[0] + 1) / 2).clip(0, 1)
        
        return final_image
    
    def process_triplet(self, triplet, save_dir=None, idx=None):
        """处理单个三元组，生成结果图像"""
        # 加载图像
        face_img = self._load_image(triplet['face'])
        shape_img = self._load_image(triplet['shape'])
        color_img = self._load_image(triplet['color'])
        
        # 确保图像大小一致
        face_img, shape_img, color_img = equal_replacer([face_img, shape_img, color_img])
        
        # 获取embedding
        face_embed = self.get_image_embedding(face_img, 'face')
        shape_embed = self.get_image_embedding(shape_img, 'shape')
        color_embed = self.get_image_embedding(color_img, 'color')
        
        # 构建name_to_embed字典
        name_to_embed = {
            'face': face_embed,
            'shape': shape_embed,
            'color': color_embed
        }
        
        # 对齐阶段
        align_shape = self.alignment_model.align_images('face', 'shape', name_to_embed)
        
        # 形状模块阶段
        if shape_img is not color_img:
            align_color = self.alignment_model.shape_module('face', 'color', name_to_embed)
        else:
            align_color = align_shape
        
        # 更新name_to_embed中的对齐结果
        name_to_embed['shape']['latent_F_align'] = align_shape['latent_F_align']
        name_to_embed['color']['HM_X'] = align_color['HM_X']
        
        # 生成结果
        result = self.blend_images(name_to_embed)
        
        # 保存生成的图像（如果需要）
        if save_dir is not None and idx is not None:
            save_path = os.path.join(save_dir, f'generated_{idx:04d}.png')
            save_image(result, save_path)
        
        return result
    
    def calculate_fid_metrics(self, triplets):
        """生成图像并流式更新FID统计，避免累积所有生成图像"""
        print(f"\nProcessing {len(triplets)} triplets...")
        
        # 创建输出目录
        if self.config['save_generated_images']:
            generated_dir = os.path.join(self.config['output_dir'], 'generated_images')
            os.makedirs(generated_dir, exist_ok=True)
        else:
            generated_dir = None
        
        # 重置统计量（因为之前已更新过真实特征，现在只需重置虚假部分）
        # 注意：torchmetrics的FID类在调用update(real=False)前需先reset_fake？
        # 实际上我们只需要在开始计算生成图像前重置虚假部分的统计。
        # 但torchmetrics的FrechetInceptionDistance会累积real和fake，
        # 由于我们已经通过update_real_features更新了real部分，这里只需要继续更新fake即可。
        # 无需额外重置，直接开始更新fake。
        
        # 分批更新参数
        batch_size = 32
        batch_buffer = []
        
        for idx, triplet in enumerate(tqdm(triplets, desc="Generating images")):
            result = self.process_triplet(triplet, generated_dir, idx)
            result_299 = self.to_299(result.unsqueeze(0))   # [1,3,299,299]
            batch_buffer.append(result_299)
            
            if len(batch_buffer) >= batch_size:
                batch = torch.cat(batch_buffer, dim=0).to(self.device)
                # 更新FID_CLIP
                self.fid_clip.update(batch, real=False)
                # 更新传统FID
                if self.fid is not None:
                    self.fid.update(batch, real=False)
                batch_buffer = []      # 释放显存
        
        # 处理剩余批次
        if batch_buffer:
            batch = torch.cat(batch_buffer, dim=0).to(self.device)
            self.fid_clip.update(batch, real=False)
            if self.fid is not None:
                self.fid.update(batch, real=False)
        
        # 计算最终分数
        fid_clip_score = self.fid_clip.compute()
        fid_score = self.fid.compute() if self.fid is not None else None
        
        print(f"\n{'='*50}")
        print(f"Results:")
        print(f"{'='*50}")
        print(f"Number of samples: {len(triplets)}")
        if fid_score is not None:
            print(f"FID (InceptionV3): {fid_score:.4f}")
        print(f"FID_CLIP (CLIP): {fid_clip_score:.4f}")
        print(f"{'='*50}")
        
        # 保存结果到文件
        results_file = os.path.join(self.config['output_dir'], 'fid_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"FID Calculation Results\n")
            f.write(f"{'='*50}\n")
            f.write(f"Number of samples: {len(triplets)}\n")
            f.write(f"Random seed: {self.config['random_seed']}\n")
            if fid_score is not None:
                f.write(f"FID (InceptionV3): {fid_score:.4f}\n")
            f.write(f"FID_CLIP (CLIP): {fid_clip_score:.4f}\n")
            f.write(f"\nTriplets used:\n")
            for i, triplet in enumerate(triplets):
                f.write(f"{i+1}. Face: {Path(triplet['face']).name}, "
                       f"Shape: {Path(triplet['shape']).name}, "
                       f"Color: {Path(triplet['color']).name}\n")
        
        print(f"\nResults saved to: {results_file}")
        
        return {
            'fid': fid_score.item() if fid_score is not None else None,
            'fid_clip': fid_clip_score.item(),
            'num_samples': len(triplets)
        }


def set_random_seeds(seed):
    """设置所有随机种子，保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image_paths(fid_dir):
    """加载fid目录中的所有图像路径"""
    fid_path = Path(fid_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(fid_path.glob(f'*{ext}')))
        image_paths.extend(list(fid_path.glob(f'*{ext.upper()}')))
    
    # 去重并排序，保证一致性
    image_paths = sorted(list(set([str(p) for p in image_paths])))
    
    return image_paths


def generate_triplets(image_paths, num_samples, seed):
    """
    生成三元组 (face, shape, color)
    保证三元组不是同一张图片
    """
    set_random_seeds(seed)
    
    n = len(image_paths)
    if n < 3:
        raise ValueError(f"至少需要3张图片，当前只有{n}张")
    
    triplets = []
    available_indices = list(range(n))
    
    for i in range(num_samples):
        # 随机选择3个不同的索引
        selected = random.sample(available_indices, 3)
        face_idx, shape_idx, color_idx = selected
        
        triplets.append({
            'face': image_paths[face_idx],
            'shape': image_paths[shape_idx],
            'color': image_paths[color_idx],
            'face_idx': face_idx,
            'shape_idx': shape_idx,
            'color_idx': color_idx
        })
    
    return triplets


def main():
    """主函数"""
    # 设置随机种子
    set_random_seeds(config['random_seed'])
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 加载图像路径
    print(f"Loading images from: {config['fid_images_dir']}")
    image_paths = load_image_paths(config['fid_images_dir'])
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) < 3:
        raise ValueError(f"至少需要3张图片，当前只有{len(image_paths)}张")
    
    # 生成三元组
    print(f"\nGenerating {config['num_samples']} triplets with seed {config['random_seed']}...")
    triplets = generate_triplets(image_paths, config['num_samples'], config['random_seed'])
    
    # 打印前几个三元组信息
    print(f"\nFirst 3 triplets:")
    for i, triplet in enumerate(triplets[:3]):
        print(f"  {i+1}. Face: {Path(triplet['face']).name}, "
              f"Shape: {Path(triplet['shape']).name}, "
              f"Color: {Path(triplet['color']).name}")
    
    # 初始化FID计算器
    fid_calculator = FIDCalculator(config)
    
    # 更新真实图像特征
    fid_calculator.update_real_features(config['fid_images_dir'])
    
    # 计算FID指标
    results = fid_calculator.calculate_fid_metrics(triplets)
    
    print(f"\nCalculation completed!")


if __name__ == '__main__':
    main()
