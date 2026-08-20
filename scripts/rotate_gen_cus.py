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

# 添加项目根目录到系统路径，以便导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.rotate_train import Trainer
from utils.train import seed_everything
from utils.image_utils import list_image_files

# 图像预处理转换：将PIL图像转换为PyTorch张量
toTensor = T.ToTensor()

# 初始化训练器，用于特征提取
net_trainer = Trainer()


def load_image(path):
    """
    加载单个图像文件并转换为张量
    
    参数:
        path (str): 图像文件路径
        
    返回:
        torch.Tensor: 图像张量
    """
    try:
        return toTensor(Image.open(path))
    except Exception as e:
        print(f"加载图像失败: {path}, 错误: {str(e)}")
        return None


@torch.no_grad()
def load_dataset_images(imgs, ffhq_path, batch_size=32):
    """
    批量加载和处理图像数据集
    
    参数:
        imgs (list): 图像文件名列表
        ffhq_path (Path): FFHQ数据集根目录
        batch_size (int): 批处理大小，默认32
        
    返回:
        tuple: (图像列表, 关键点列表, 潜在编码列表)
    """
    print(f"开始加载和预处理 {len(imgs)} 张图像...")
    
    # 并行加载所有图像到内存
    tensors_images = Parallel(n_jobs=-1)(
        delayed(load_image)(os.path.join(ffhq_path, str(img))) 
        for img in tqdm(imgs, desc="加载图像")
    )
    
    # 过滤掉加载失败的图像
    tensors_images = [img for img in tensors_images if img is not None]
    print(f"成功加载 {len(tensors_images)} 张图像")
    
    # 创建数据加载器
    tensors_dataloader = DataLoader(
        tensors_images, 
        batch_size=batch_size, 
        pin_memory=False, 
        shuffle=False, 
        drop_last=False
    )
    
    images, key_points, latents = [], [], []
    
    # 批量处理图像
    for batch in tqdm(tensors_dataloader, desc="提取特征"):
        batch = batch.to(net_trainer.device)
        
        # 下采样到256x256分辨率
        images_256 = net_trainer.downsample_256(batch).clip(0, 1)
        images.extend(images_256.cpu())
        
        # 生成潜在编码（特征向量）
        latents.extend(net_trainer.generate_latents(images_256 * 2 - 1).cpu())
        
        # 生成关键点
        key_points.extend(net_trainer.generate_key_points(batch).cpu())
    
    return images, key_points, latents


def main(ffhq_path, seed, size, output_path, batch_size=32):
    """
    主函数：准备旋转增强数据集
    
    参数:
        ffhq_path (Path): FFHQ数据集目录路径
        seed (int): 随机种子，用于确保结果可复现
        size (int): 要处理的数据集大小（图像数量）
        output_path (Path): 输出文件路径
        batch_size (int): 批处理大小，默认32
    """
    # 设置随机种子
    seed_everything(seed)
    
    # 列出所有图像文件
    images = list_image_files(ffhq_path)
    print(f"FFHQ数据集共包含 {len(images)} 张图像")
    
    # 随机打乱图像顺序
    random.shuffle(images)
    
    # 限制处理数量
    if size > len(images):
        print(f"警告: 指定的size({size})大于数据集大小({len(images)})，将使用所有图像")
        selected_images = images
    else:
        selected_images = images[:size]
    
    print(f"将处理 {len(selected_images)} 张图像")
    
    # 加载并处理图像数据集
    images_data, key_points_data, latents_data = load_dataset_images(
        selected_images, ffhq_path, batch_size
    )
    
    # 保存处理后的数据集
    torch.save({
        'images': images_data,
        'key_points': key_points_data,
        'latents': latents_data
    }, output_path)
    
    print(f"数据集已保存到: {output_path}")
    print(f"保存了 {len(images_data)} 张图像的特征数据")


if __name__ == '__main__':
    # ============================================================================
    # 用户可在此处修改参数（无需通过命令行）
    # ============================================================================
    
    # 输入参数配置
    # FFHQ: FFHQ数据集目录路径，包含人脸图像
    # 注意：此目录应包含用于旋转训练的原始图像
    FFHQ = Path('./images/FFHQ')  # 请修改为您的FFHQ数据集路径
    
    # seed: 随机种子，用于确保结果可复现
    # 固定种子可以使每次运行结果一致，便于调试和比较
    seed = 3407  # 默认值: 3407
    
    # size: 要处理的数据集大小（图像数量）
    # 建议值: 1000-10000，根据可用内存和计算资源调整
    size = 10_00  # 默认值: 10_000
    
    # output: 输出文件路径，保存处理后的数据集
    # 文件格式为PyTorch的.pkl格式
    output = Path('./images/FFHQ_rotate_dataset/rotate_dataset.pkl')  # 默认值: 'input/rotate_dataset.pkl'
    
    # batch_size: 批处理大小，影响内存使用
    # 较大值可以提高处理速度，但需要更多内存
    # 较小值可以减少内存使用，但处理速度较慢
    batch_size = 64  # 默认值: 32
    
    # ============================================================================
    # 参数验证和提示
    # ============================================================================
    
    print("=" * 60)
    print("旋转增强数据集准备工具")
    print("=" * 60)
    print(f"FFHQ数据集目录: {FFHQ}")
    print(f"随机种子: {seed}")
    print(f"处理图像数量: {size}")
    print(f"批处理大小: {batch_size}")
    print(f"输出文件: {output}")
    print("=" * 60)
    
    # 检查数据集目录是否存在
    if not FFHQ.exists():
        print(f"错误: FFHQ数据集目录不存在: {FFHQ}")
        print("请检查FFHQ路径是否正确，或下载FFHQ数据集")
        sys.exit(1)
    
    # 检查数据集目录是否为空
    image_files = list_image_files(FFHQ)
    if len(image_files) == 0:
        print(f"错误: FFHQ数据集目录为空: {FFHQ}")
        print("请确保目录中包含图像文件")
        sys.exit(1)
    
    # 检查GPU是否可用
    if torch.cuda.is_available():
        print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 未检测到GPU，将使用CPU处理（速度较慢）")
    
    # 创建输出目录（如果不存在）
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # 执行主函数
    main(FFHQ, seed, size, output, batch_size)