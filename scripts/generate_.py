import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ======================= 用户配置区域 =======================
DATASET_DIR = "celeba-1024"          # 原始 CelebA-HQ 1024 目录
CHECKPOINT_PATH = "blending/blending_base_models/blending_20.pth" # 你的第二阶段权重路径
OUTPUT_CSV = "fixed_both_test_3000.csv"   # 固定的配对清单文件
GEN_OUTPUT_DIR = "results_both_stage2"    # 生成图片存放目录
SEED = 3407                               # 建议使用作者常用的随机种子
NUM_PAIRS = 3000                          # 固定 3000 组
# ===========================================================

def generate_fixed_pairs():
    """从数据集选取6000张不重复照片，构造3000个固定的Both配对"""
    print(f"🔍 正在扫描数据集: {DATASET_DIR}")
    # 获取所有图片并排序，确保跨平台顺序一致
    all_images = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(all_images) < NUM_PAIRS * 2:
        raise ValueError(f"错误：数据集图片不足 {NUM_PAIRS * 2} 张，无法构造不重复配对。")

    # 固定随机种子
    random.seed(SEED)
    
    # 随机选取 6000 张照片
    selected_images = random.sample(all_images, NUM_PAIRS * 2)
    
    pairs = []
    # 每两张照片组成一个 Both 二元组 (Face, Reference)
    for i in range(0, len(selected_images), 2):
        pairs.append({
            'source_face': selected_images[i],
            'ref_image': selected_images[i+1] # Both模式：Ref提供形状和颜色 [cite: 276]
        })

    df = pd.DataFrame(pairs)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 已成功构造并保存 3000 组固定配对清单至: {OUTPUT_CSV}")

def run_inference():
    """批量推理骨架：调用你训练的第二阶段模型"""
    if not os.path.exists(OUTPUT_CSV):
        generate_fixed_pairs()

    df = pd.read_csv(OUTPUT_CSV)
    os.makedirs(GEN_OUTPUT_DIR, exist_ok=True)

    print(f"🏗️ 正在加载 Checkpoint: {CHECKPOINT_PATH}")
    # 示例代码：加载你的模型
    # model = YourModelClass.load_from_checkpoint(CHECKPOINT_PATH).to('cuda').eval()

    print(f"🚀 开始为 Both 模式生成 {NUM_PAIRS} 张图片...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        face_path = os.path.join(DATASET_DIR, row['source_face'])
        ref_path = os.path.join(DATASET_DIR, row['ref_image'])
        
        # 保存文件名建议包含原脸信息，方便定位
        output_filename = f"gen_{row['source_face']}"
        save_path = os.path.join(GEN_OUTPUT_DIR, output_filename)

        if os.path.exists(save_path):
            continue

        # 推理逻辑（需替换为你实际的模型调用接口）
        # with torch.no_grad():
        #     # Both 模式下，Shape 和 Color 的参考源是同一个 [cite: 276]
        #     result = model.generate(face_path, shape_ref=ref_path, color_ref=ref_path)
        #     result.save(save_path)

    print(f"✨ 所有生成图片已保存至: {GEN_OUTPUT_DIR}")

if __name__ == "__main__":
    # 生成固定清单
    generate_fixed_pairs()
    # 执行推理
    # run_inference()