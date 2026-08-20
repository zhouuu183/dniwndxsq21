import os
import sys
from pathlib import Path
from PIL import Image

# 添加项目根目录到系统路径，以便导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.seed import set_seed
from utils.image_utils import list_image_files
from utils.shape_predictor import align_face


def alignment_from_path(input_path, output_path, replace_cropped=False):
    """
    对输入目录中的所有人脸图像进行对齐处理
    
    参数:
        input_path (Path): 输入目录路径，包含待处理的原始图像
        output_path (Path): 输出目录路径，用于保存对齐后的人脸图像
        replace_cropped (bool): 是否替换已存在的对齐图像（默认False）
    """
    # 创建输出目录（如果不存在）
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取输入目录中所有图像文件
    image_files = list_image_files(input_path)
    
    for img_path in image_files:
        # 如果输出文件已存在且不替换，则跳过
        if (output_path / img_path).is_file() and not replace_cropped:
            print(f"跳过已存在的文件: {img_path}")
            continue
        
        try:
            # 打开图像文件
            image = Image.open(input_path / img_path)
            
            # 对齐人脸，返回裁剪后的人脸图像列表
            crop_image = align_face(image, return_tensors=False)
            
            # 处理检测到的人脸
            if len(crop_image) == 0:
                print(f"警告: 未检测到人脸，跳过文件: {img_path}")
                continue
            elif len(crop_image) == 1:
                # 保存对齐后的人脸图像
                crop_image[0].save(output_path / img_path)
                print(f"已保存对齐后的人脸: {img_path}")
            elif len(crop_image) > 1:
                # 检测到多个人脸的情况（可以按需修改处理逻辑）
                print(f"警告: 检测到多个人脸({len(crop_image)}个)，仅保存第一个: {img_path}")
                # 保存第一个人脸（可以根据需要修改为保存所有人脸）
                crop_image[0].save(output_path / img_path)
                
        except Exception as e:
            print(f"处理图像 {img_path} 时发生错误: {str(e)}")
            continue


if __name__ == "__main__":
    # ============================================================================
    # 用户可在此处修改参数（无需通过命令行）
    # ============================================================================
    
    # 输入参数配置
    # unprocessed_dir: 未处理的原始图像目录路径
    unprocessed_dir = Path('./images/FFHQ')  # 默认值: 'unprocessed'
    
    # output_dir: 输出目录路径，保存对齐后的人脸图像
    output_dir = Path('./images/aligen_face')  # 默认值: 'input'
    
    # replace_cropped: 是否替换已存在的对齐图像
    # True: 重新处理并覆盖已存在的对齐图像
    # False: 跳过已存在的对齐图像（默认）
    replace_cropped = False  # 默认值: False
    
    # seed: 随机种子，用于确保结果可复现
    seed_value = 3407  # 默认值: 3407
    
    # ============================================================================
    # 参数验证和提示
    # ============================================================================
    
    print("=" * 50)
    print("人脸对齐处理配置")
    print("=" * 50)
    print(f"输入目录: {unprocessed_dir}")
    print(f"输出目录: {output_dir}")
    print(f"替换已存在文件: {replace_cropped}")
    print(f"随机种子: {seed_value}")
    print("=" * 50)
    
    # 检查输入目录是否存在
    if not unprocessed_dir.exists():
        print(f"错误: 输入目录不存在: {unprocessed_dir}")
        print("请检查 unprocessed_dir 路径是否正确")
        sys.exit(1)
    
    # 统计输入目录中的图像数量
    image_files = list_image_files(unprocessed_dir)
    print(f"发现 {len(image_files)} 个图像文件")
    
    # 设置随机种子以确保结果可复现
    set_seed(seed_value)
    
    # 执行人脸对齐处理
    print("\n开始人脸对齐处理...")
    alignment_from_path(unprocessed_dir, output_dir, replace_cropped)
    print("\n处理完成！")
    
    # 统计输出结果
    output_files = list(list_image_files(output_dir))
    print(f"成功处理 {len(output_files)} 个图像")