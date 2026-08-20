import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Encoders import ClipModel
from utils.seed import set_seed
from utils.image_utils import list_image_files

def name_path(pair):
    name, path = pair.split(',')
    return name, Path(path)

# =========================================================================
# [修改标注 1]：新增 LazyImageDataset 类 (懒加载数据集)
# 作用：彻底取代原本会把内存撑爆的 parallel_load_images。
# 机制：初始化时只存文件路径列表（几 MB），真正训练到那一个 Batch 时才去硬盘读图。
# =========================================================================
class LazyImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        # 仅获取文件名列表
        self.image_paths = list_image_files(self.folder_path)
        # torchmetrics FID 需要 uint8 格式的 Tensor，PILToTensor 速度极快且正好符合要求
        self.transform = transforms.PILToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # [修改标注 2]：修复 FileNotFoundError 路径问题
        # 提取纯文件名，并与文件夹路径拼接成完整的绝对/相对路径
        img_name = os.path.basename(str(self.image_paths[idx]))
        full_path = os.path.join(self.folder_path, img_name)
        
        # 打开图片并转换为 RGB
        img = Image.open(full_path).convert('RGB')
        return self.transform(img)


@torch.inference_mode()
def compute_fid_datasets(datasets, target='celeba', device=torch.device('cuda'), CLIP=False, seed=3407, batch_size=16, num_workers=4):
    set_seed(seed)
    result = {}

    if CLIP:
        fid = FrechetInceptionDistance(feature=ClipModel(), reset_real_features=False, normalize=False)
    else:
        fid = FrechetInceptionDistance(reset_real_features=False, normalize=False)
    fid.to(device).eval()

    # =========================================================================
    # [修改标注 3]：使用 DataLoader 进行多进程、分批次读取
    # 作用：num_workers 负责去硬盘搬图，pin_memory=True 加速数据放入显卡
    # =========================================================================
    real_dataloader = DataLoader(
        datasets[target], 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True
    )
    for batch in tqdm(real_dataloader, desc=f"Processing Real ({target})"):
        # 取消了 batch[0]，因为我们自定义的 Dataset 直接返回的就是 Tensor
        batch = batch.to(device)
        fid.update(batch, real=True)

    for key, dataset in datasets.items():
        if key == target:
            continue
        fid.reset()

        fake_dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True
        )
        for batch in tqdm(fake_dataloader, desc=f"Processing Fake ({key})"):
            batch = batch.to(device)
            fid.update(batch, real=False)
            
        result[key] = fid.compute().item()
    return result


def main(args):
    datasets = {}
    source = args.source_dataset.name

    # [修改标注 4]：弃用 parallel_load_images，改用懒加载类实例化数据集
    print(f"📦 注册真实数据集: {source} (硬盘懒加载模式)...")
    datasets[source] = LazyImageDataset(args.source_dataset)

    for method, path_dataset in args.methods_dataset:
        print(f"📦 注册生成数据集: {method} (硬盘懒加载模式)...")
        datasets[method] = LazyImageDataset(path_dataset)

    # [修改标注 5]：将外部传入的 batch_size 和 num_workers 传递给计算函数
    print("\n[1/2] 开始计算标准 FID...")
    FIDs = compute_fid_datasets(
        datasets, target=source, CLIP=False, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    df_fid = pd.DataFrame.from_dict(FIDs, orient='index', columns=['FID'])

    print("\n[2/2] 开始计算 CLIP-FID...")
    FIDs_CLIP = compute_fid_datasets(
        datasets, target=source, CLIP=True, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    df_clip = pd.DataFrame.from_dict(FIDs_CLIP, orient='index', columns=['FID_CLIP'])

    df_result = pd.concat([df_fid, df_clip], axis=1).round(2)
    print("\n📊 最终评估结果:")
    print(df_result)

    os.makedirs(args.output.parent, exist_ok=True)
    df_result.to_csv(args.output, index=True)
    print(f"\n✅ 结果已保存至: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics')
    parser.add_argument('--source_dataset', type=Path, required=True, help='Dataset with real faces')
    parser.add_argument('--methods_dataset', type=name_path, nargs='+', required=True, help='Datasets after applying the method')
    parser.add_argument('--output', type=Path, default='logs/metric.csv', help='Folder for saving logs')
    
    # [修改标注 6]：新增控制显存和多进程的命令行参数
    parser.add_argument('--batch_size', type=int, default=16, help='GPU 显存批次大小 (默认: 16)')
    parser.add_argument('--num_workers', type=int, default=4, help='用来从硬盘读图的 CPU 进程数 (默认: 4)')
    
    args = parser.parse_args()
    main(args)