import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image

# 1. 强制路径对齐：确保能看到根目录
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 2. 严格导入原版模块：不带任何后缀 
try:
    from hair_swap import HairFast, get_parser
    print("✅ 成功载入【原版复现】逻辑 (hair_swap.py)")
except ImportError as e:
    print(f"❌ 导入失败，请检查根目录下是否存在 hair_swap.py。错误: {e}")
    sys.exit(1)

# ======================= 用户配置区域 =======================
DATASET_DIR = "celeba-1024"                 # 原始图片目录
INPUT_CSV = "fixed_both_test_3000.csv"          # 3000组固定清单
OUTPUT_DIR = "results_both_reproduce_stage2" # 复现结果保存目录

# 填写你复现出来的第二阶段权重路径 
# 注意：这通常是你在训练作者原版代码时产生的 .pth 文件
MY_CHECKPOINT = "blending/blending_base_models/blending_20.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 3407
# ===========================================================

@torch.no_grad()
def run_reproduction():
    # 3. 初始化作者原版参数 [cite: 72]
    model_parser = get_parser()
    args = model_parser.parse_args([]) 
    
    # 将权重路径喂给原版模型 [cite: 72]
    args.device = DEVICE
    args.blending_checkpoint = MY_CHECKPOINT 
    
    # 4. 实例化模型
    # 这会触发导入 models.Alignment 和 models.Blending (原版) 
    print(f"🏗️ 正在初始化原版模型，加载复现权重: {MY_CHECKPOINT}...")
    hair_fast = HairFast(args)
    
    # 5. 读取清单并开始批量生成
    df = pd.read_csv(INPUT_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🚀 开始生成 3000 张复现照片 (Both模式)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        face_path = os.path.join(DATASET_DIR, row['source_face'])
        ref_path = os.path.join(DATASET_DIR, row['ref_image'])
        save_path = os.path.join(OUTPUT_DIR, row['source_face'])

        if os.path.exists(save_path):
            continue

        try:
            # Both 模式：形状和颜色均来自 ref_path [cite: 100, 101]
            final_image = hair_fast.swap(face_path, ref_path, ref_path, align=False, seed=SEED)
            save_image(final_image, save_path)
        except Exception as e:
            print(f"❌ 处理 {row['source_face']} 出错: {e}")

    print(f"✨ 复现图片生成完毕！结果在: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_reproduction()