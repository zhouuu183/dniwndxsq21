import os
from typing import List, Tuple

import torch
from torchvision.utils import save_image, make_grid
from tqdm.auto import tqdm

from models.Encoders import ClipBlendingModel
from models.Net import Net
from utils.image_utils import equal_replacer, DilateErosion
from utils.bicubic import BicubicDownSample
from hair_swap import get_parser


def load_images(origin_paths: List[str], shape_paths: List[str], color_paths: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    加载图像并进行预处理
    """
    from torchvision.io import read_image, ImageReadMode
    
    image_tuples = []
    
    for origin_path, shape_path, color_path in zip(origin_paths, shape_paths, color_paths):
        # 加载图像
        origin = read_image(origin_path, mode=ImageReadMode.RGB)
        shape = read_image(shape_path, mode=ImageReadMode.RGB)
        color = read_image(color_path, mode=ImageReadMode.RGB)
        
        # 确保图像大小一致
        images = equal_replacer([origin, shape, color])
        image_tuples.append(tuple(images))
    
    return image_tuples


def get_single_image_embedding(embedding_model: Embedding, image: torch.Tensor, device: torch.device, name: str) -> dict:
    """
    获取单张图像的embedding
    """
    from collections import defaultdict
    
    # 创建单张图像的输入字典
    images_to_name = defaultdict(list)
    images_to_name[image].append(name)
    
    # 设置dataloader
    embedding_model.setup_dataloader(images_to_name)
    
    # 获取embedding
    name_to_embed = defaultdict(dict)
    for batch_image, names in embedding_model.dataloader:
        batch_image = batch_image.to(device)
        img_name = names[0]
        
        im_512 = embedding_model.downsample_512(batch_image)
        im_256 = embedding_model.downsample_256(batch_image)
        im_256_norm = embedding_model.normalize(im_256)

        # E4E
        from models.Embedding import get_latents
        latent_W = get_latents(embedding_model.e4e, im_256_norm)

        # FS encoder
        output = embedding_model.encoder.test(img=embedding_model.normalize(batch_image), return_latent=True)
        latent = output.pop()  # [bs, 512, 16, 16]
        latent_S = output.pop()  # [bs, 18, 512]

        latent_F, _ = embedding_model.net.generator([latent_S], input_is_latent=True, return_latents=False,
                                                 start_layer=3, end_layer=3, layer_in=latent)  # [bs, 512, 32, 32]

        # BiSeNet
        from models.Embedding import get_segmentation
        masks = torch.cat([get_segmentation(batch_image.unsqueeze(0)) for batch_image in embedding_model.to_bisenet(im_512)])

        name_to_embed[img_name] = {
            'image': batch_image,
            'image_256': im_256,
            'image_norm_256': im_256_norm,
            'S': latent_S,
            'W': latent_W,
            'F': latent_F,
            'mask': masks
        }
    
    return name_to_embed[img_name]


def blend_images_without_postprocess(blending_encoder: ClipBlendingModel, net: Net, dilate_erosion: DilateErosion, 
                                     downsample_256: BicubicDownSample,
                                     name_to_embed: dict) -> torch.Tensor:
    """
    使用blending模型处理图像（不使用后处理）
    """
    I_1 = name_to_embed['face']['image_norm_256']
    I_2 = name_to_embed['shape']['image_norm_256']
    I_3 = name_to_embed['color']['image_norm_256']

    mask_de = dilate_erosion.hair_from_mask(
        torch.cat([name_to_embed[x]['mask'] for x in ['face', 'color']], dim=0)
    )
    HM_1D, _ = mask_de[0][0].unsqueeze(0), mask_de[1][0].unsqueeze(0)
    HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)

    latent_S_1 = name_to_embed['face']['S']
    latent_F_align = name_to_embed['shape']['latent_F_align']
    HM_X = name_to_embed['color']['HM_X']

    latent_S_3 = name_to_embed['color']["S"]

    HM_XD, _ = dilate_erosion.mask(HM_X)
    target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

    # Blending
    if I_1 is not I_3 or I_1 is not I_2:
        S_blend_6_18 = blending_encoder(latent_S_1[:, 6:], latent_S_3[:, 6:], I_1 * target_mask, I_3 * HM_3E)
        S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
    else:
        S_blend = latent_S_1

    I_blend, _ = net.generator([S_blend], input_is_latent=True, return_latents=False, start_layer=4,
                              end_layer=8, layer_in=latent_F_align)

    # 直接返回blending结果，不进行后处理
    final_image = ((I_blend[0] + 1) / 2).clip(0, 1)
    return final_image


def create_grid(origin: torch.Tensor, shape: torch.Tensor, color: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """
    创建包含四张图片的网格
    """
    # 归一化图像到[0, 1]范围
    def normalize(img):
        return img.float() / 255.0
    
    # 准备图像
    origin = normalize(origin)
    shape = normalize(shape)
    color = normalize(color)
    result = result  # result已经是[0, 1]范围
    
    # 创建网格：1x4布局
    # origin, shape, color, result
    grid = make_grid([origin, shape, color, result], nrow=4, padding=2, normalize=False)
    
    return grid


def main():
    """
    主函数
    """
    
    # ========== 用户配置区域 ==========
    
    # 模型配置
    blending_checkpoint = "pretrained_models/Blending/checkpoint.pth"  # blending模型路径
    device = "cuda"  # 运行设备: "cuda" 或 "cpu"
    
    # 图像路径配置
    origin_images = [
        "path/to/origin1.jpg",
        "path/to/origin2.jpg",
        # 添加更多原始图像路径...
    ]
    
    shape_images = [
        "path/to/shape1.jpg",
        "path/to/shape2.jpg",
        # 添加更多参考发型图像路径...
    ]
    
    color_images = [
        "path/to/color1.jpg",
        "path/to/color2.jpg",
        # 添加更多参考颜色图像路径...
    ]
    
    # 输出配置
    output_dir = "validation_output"  # 结果保存目录
    
    # ========== 配置区域结束 ==========
    
    # 验证输入长度
    if not (len(origin_images) == len(shape_images) == len(color_images)):
        raise ValueError("origin_images, shape_images, and color_images must have the same length")
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型参数
    model_args = get_parser()
    model_args = model_args.parse_args([])
    model_args.device = device
    model_args.blending_checkpoint = blending_checkpoint
    model_args.size = 1024
    model_args.ckpt = "pretrained_models/StyleGAN/ffhq.pt"
    model_args.channel_multiplier = 2
    model_args.latent = 512
    model_args.n_mlp = 8
    model_args.smooth = 5
    model_args.batch_size = 1  # 设置batch size为1，避免特征尺寸不匹配
    
    # 加载网络
    print("Loading models...")
    net = Net(model_args)
    
    # 加载blending编码器
    blending_checkpoint_data = torch.load(blending_checkpoint)
    blending_encoder = ClipBlendingModel(blending_checkpoint_data.get('clip', "ViT-B/32"))
    blending_encoder.load_state_dict(blending_checkpoint_data['model_state_dict'], strict=False)
    blending_encoder.to(device).eval()
    
    # 加载embedding和对齐模块
    from models.Embedding import Embedding
    from models.Alignment import Alignment
    embedding_model = Embedding(model_args, net=net)
    alignment_model = Alignment(model_args, embedding_model.get_e4e_embed, net=net)
    
    # 初始化辅助工具
    dilate_erosion = DilateErosion(dilate_erosion=model_args.smooth, device=device)
    downsample_256 = BicubicDownSample(factor=4)
    
    print("Models loaded successfully!")
    
    # 加载图像
    print("Loading images...")
    image_tuples = load_images(origin_images, shape_images, color_images)
    print(f"Loaded {len(image_tuples)} image tuples")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每张图像
    print("Processing images...")
    for i, (origin, shape, color) in enumerate(tqdm(image_tuples)):
        # 分别获取每张图像的embedding
        print(f"Processing image {i+1}...")
        
        # 获取单张图像的embedding
        origin_embed = get_single_image_embedding(embedding_model, origin, device, 'face')
        shape_embed = get_single_image_embedding(embedding_model, shape, device, 'shape')
        color_embed = get_single_image_embedding(embedding_model, color, device, 'color')
        
        # 构建name_to_embed字典
        name_to_embed = {
            'face': origin_embed,
            'shape': shape_embed,
            'color': color_embed
        }
        
        # 对齐阶段
        align_shape = alignment_model.align_images('face', 'shape', name_to_embed)
        
        # 形状模块阶段
        if shape is not color:
            align_color = alignment_model.shape_module('face', 'color', name_to_embed)
        else:
            align_color = align_shape
        
        # 更新name_to_embed中的对齐结果
        name_to_embed['shape']['latent_F_align'] = align_shape['latent_F_align']
        name_to_embed['color']['HM_X'] = align_color['HM_X']
        
        # 处理图像（不使用后处理）
        result = blend_images_without_postprocess(
            blending_encoder, net, dilate_erosion, downsample_256,
            name_to_embed
        )
        
        # 创建网格
        grid = create_grid(origin, shape, color, result)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"result_{i+1}.png")
        save_image(grid, output_path)
        print(f"Saved result to: {output_path}")
    
    print("All images processed successfully!")


if __name__ == "__main__":
    main()