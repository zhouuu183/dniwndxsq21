import argparse
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.rotate_train import Trainer
from utils.train import seed_everything
from utils.image_utils import list_image_files

toTensor = T.ToTensor()

class DatasetProcessor:
    def __init__(self, ffhq_path, batch_size=4):
        self.batch_size = batch_size
        self.ffhq_path = Path(ffhq_path)
        self.net_trainer = Trainer()
        self.device = self.net_trainer.device
        self.checkpoint_interval = 50  # 每50个批次保存一次检查点
        
    def load_image_batch(self, image_names):
        """批量加载图像"""
        batch_images = []
        for img_name in image_names:
            try:
                # 构建完整路径
                img_path = self.ffhq_path / img_name
                if not img_path.exists():
                    print(f"Warning: Image not found: {img_path}")
                    continue
                    
                img = Image.open(img_path)
                # 转换为RGB，确保3通道
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                batch_images.append(toTensor(img))
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                continue
        return batch_images
    
    def process_batch(self, image_batch):
        """处理单个批次"""
        if not image_batch:
            return [], [], []
            
        batch_tensor = torch.stack(image_batch).to(self.device)
        
        # 使用更小的chunk size进行内部处理
        chunk_size = 4
        images_256_chunks = []
        latents_chunks = []
        key_points_chunks = []
        
        for i in range(0, len(batch_tensor), chunk_size):
            chunk = batch_tensor[i:i+chunk_size]
            
            # 下采样到256x256
            images_256 = self.net_trainer.downsample_256(chunk).clip(0, 1)
            images_256_chunks.append(images_256.cpu())
            
            # 生成潜在向量
            latents = self.net_trainer.generate_latents(images_256 * 2 - 1)
            latents_chunks.append(latents.cpu())
            
            # 生成关键点
            key_points = self.net_trainer.generate_key_points(chunk)
            key_points_chunks.append(key_points.cpu())
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                del chunk, images_256, latents, key_points
        
        # 合并结果
        if images_256_chunks:
            images_256 = torch.cat(images_256_chunks, dim=0)
            latents = torch.cat(latents_chunks, dim=0)
            key_points = torch.cat(key_points_chunks, dim=0)
            return images_256, key_points, latents
        else:
            return [], [], []
    
    def save_checkpoint(self, data, checkpoint_path):
        """保存检查点"""
        torch.save(data, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            return torch.load(checkpoint_path)
        return None

def main(args):
    seed_everything(args.seed)
    
    # 获取所有图像文件
    images = list_image_files(args.FFHQ)
    print(f"Found {len(images)} images in dataset")
    random.shuffle(images)
    
    # 只取指定数量的图像
    total_images = min(args.size, len(images))
    images = images[:total_images]
    
    print(f"Processing {total_images} images...")
    
    # 创建处理器
    processor = DatasetProcessor(ffhq_path=args.FFHQ, batch_size=args.batch_size)
    
    # 检查点路径
    checkpoint_path = Path(str(args.output) + '.checkpoint')
    final_output = args.output
    
    # 尝试加载检查点
    checkpoint_data = processor.load_checkpoint(checkpoint_path)
    if checkpoint_data:
        print(f"Resuming from checkpoint with {len(checkpoint_data['images'])} processed images")
        all_images = checkpoint_data['images']
        all_key_points = checkpoint_data['key_points']
        all_latents = checkpoint_data['latents']
        start_idx = len(all_images)
    else:
        all_images, all_key_points, all_latents = [], [], []
        start_idx = 0
    
    # 分批处理图像
    batch_size = processor.batch_size
    checkpoint_counter = 0
    
    for i in tqdm(range(start_idx, total_images, batch_size), desc="Processing batches"):
        batch_names = images[i:i+batch_size]
        
        try:
            # 加载图像批次
            batch_images = processor.load_image_batch(batch_names)
            
            if not batch_images:
                print(f"Warning: No images loaded for batch starting at index {i}")
                continue
                
            # 处理批次
            images_batch, key_points_batch, latents_batch = processor.process_batch(batch_images)
            
            if len(images_batch) > 0:
                # 添加到结果列表
                all_images.extend(images_batch)
                all_key_points.extend(key_points_batch)
                all_latents.extend(latents_batch)
                
                checkpoint_counter += 1
                
                # 按间隔保存检查点
                if checkpoint_counter % processor.checkpoint_interval == 0:
                    checkpoint_data = {
                        'images': all_images,
                        'key_points': all_key_points,
                        'latents': all_latents
                    }
                    processor.save_checkpoint(checkpoint_data, checkpoint_path)
                    print(f"Progress: {len(all_images)}/{total_images} images processed")
            else:
                print(f"Warning: No images processed for batch starting at index {i}")
                
            # 清理内存
            del batch_images
            if 'images_batch' in locals():
                del images_batch, key_points_batch, latents_batch
            
            # 定期清理Python内存
            if checkpoint_counter % 10 == 0:
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存最终结果
    if all_images:
        final_data = {
            'images': all_images,
            'key_points': all_key_points,
            'latents': all_latents
        }
        
        torch.save(final_data, final_output)
        print(f"Saved final dataset to {final_output} with {len(all_images)} images")
    else:
        print("Warning: No images were processed!")
    
    # 删除检查点文件
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint file removed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotate dataset generator')
    parser.add_argument('--FFHQ', type=Path, required=True, help='Path to FFHQ dataset')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--size', type=int, default=10_000, help='Number of images to process')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing (lower to reduce memory usage)')
    parser.add_argument('--output', type=Path, default='input/rotate_dataset.pkl')
    args = parser.parse_args()
    
    # 创建输出目录
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    main(args)