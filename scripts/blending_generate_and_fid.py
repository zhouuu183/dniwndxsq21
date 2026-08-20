import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 导入你的模型

from models.Encoders import ClipBlendingModel, ClipModel
from models.Net import Net
from utils.seed import set_seed
from utils.image_utils import list_image_files, equal_replacer

# ------------------ 图像加载 ------------------
def load_images_as_tensor(folder: Path, max_images=None):
    image_paths = list_image_files(folder)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    images = []
    for p in tqdm(image_paths, desc=f"Loading images from {folder}"):
        img = Image.open(p).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        images.append(img_tensor)
    return torch.stack(images), image_paths[:len(images)]

# ------------------ 生成三元组 ------------------
def generate_triplets(image_paths, seed=42):
    """
    对每张真实图像生成三元组 (face, shape, color)
    保证一一对应
    """
    set_seed(seed)
    n = len(image_paths)
    triplets = []
    for i, face_path in enumerate(image_paths):
        # 随机选 shape/color，不能与 face 相同
        choices = list(range(n))
        choices.remove(i)
        shape_idx = random.choice(choices)
        color_idx = random.choice(choices)
        triplets.append({
            'face': face_path,
            'shape': image_paths[shape_idx],
            'color': image_paths[color_idx]
        })
    return triplets

# ------------------ FID 计算 ------------------
@torch.inference_mode()
def compute_fid(real_tensor, fake_tensor, device='cuda', use_clip=False, batch_size=128, seed=42):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    if use_clip:
        fid_metric = FrechetInceptionDistance(feature=ClipModel(), reset_real_features=False, normalize=False)
    else:
        fid_metric = FrechetInceptionDistance(reset_real_features=False, normalize=False)
    fid_metric.to(device).eval()

    # 分批更新 real
    real_loader = DataLoader(TensorDataset(real_tensor), batch_size=batch_size)
    for batch in real_loader:
        fid_metric.update(batch[0].to(device), real=True)

    # 分批更新 fake
    fake_loader = DataLoader(TensorDataset(fake_tensor), batch_size=batch_size)
    fid_metric.reset()
    for batch in fake_loader:
        fid_metric.update(batch[0].to(device), real=False)

    return fid_metric.compute().item()

# ------------------ 生成第二阶段图像 ------------------
@torch.inference_mode()
def generate_stage2_images(blending_model, stylegan_net, triplets, output_folder, device='cuda'):
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    generated_paths = []

    for idx, t in enumerate(tqdm(triplets, desc="Generating Stage2 images")):
        # 读取 face, shape, color
        imgs = []
        for key in ['face','shape','color']:
            img = Image.open(t[key]).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
            imgs.append(img_tensor.unsqueeze(0).to(device))
        face_tensor, shape_tensor, color_tensor = imgs

        # ---- 生成 Blending 图像 ----
        # 这里用 blending_model.forward(face, shape, color)
        # 注意根据你的 blending_train.py 调整
        output = blending_model(face_tensor, shape_tensor, color_tensor)
        output_img = ((output[0]+1)/2).clamp(0,1)
        save_path = os.path.join(output_folder, f"generated_{idx:04d}.png")
        save_image(output_img, save_path)
        generated_paths.append(save_path)

    return generated_paths

# ------------------ 主函数 ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dataset', type=Path, required=True, help="真实图片文件夹")
    parser.add_argument('--checkpoint', type=Path, required=True, help="第二阶段 Blending 模型权重")
    parser.add_argument('--output_dir', type=Path, default="./generated_stage2", help="生成图像保存路径")
    parser.add_argument('--fid_output', type=Path, default="./logs/fid_stage2.csv", help="FID CSV 输出")
    parser.add_argument('--num_images', type=int, default=3000, help="用于计算的图片数量")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # ---- 加载真实图片 ----
    print("Loading real images...")
    real_tensor, image_paths = load_images_as_tensor(args.real_dataset, max_images=args.num_images)

    # ---- 生成三元组 ----
    print("Generating triplets...")
    triplets = generate_triplets(image_paths)

    # ---- 加载模型 ----
    blending_ckpt = torch.load(args.checkpoint, map_location=device)
    blending_model = ClipBlendingModel()
    blending_model.load_state_dict(blending_ckpt['model_state_dict'], strict=False)
    blending_model.to(device).eval()

    stylegan_net = Net()  # 如果 blending 需要
    stylegan_net.to(device).eval()

    # ---- 生成第二阶段图像 ----
    print("Generating Stage2 images...")
    generate_stage2_images(blending_model, stylegan_net, triplets, args.output_dir, device=device)

    # ---- 计算 FID/FID_CLIP ----
    print("Loading generated images...")
    fake_tensor, _ = load_images_as_tensor(args.output_dir, max_images=args.num_images)

    print("Computing FID (InceptionV3)...")
    fid_score = compute_fid(real_tensor, fake_tensor, device=device, use_clip=False, batch_size=args.batch_size)
    print("Computing FID_CLIP...")
    fid_clip_score = compute_fid(real_tensor, fake_tensor, device=device, use_clip=True, batch_size=args.batch_size)

    print(f"FID: {fid_score:.2f}, FID_CLIP: {fid_clip_score:.2f}")

    # ---- 保存结果 ----
    os.makedirs(args.fid_output.parent, exist_ok=True)
    pd.DataFrame({"FID":[fid_score], "FID_CLIP":[fid_clip_score]}).to_csv(args.fid_output, index=False)
    print(f"Results saved to {args.fid_output}")

if __name__ == "__main__":
    main()