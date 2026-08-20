# mini_train_satd.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Alignment import Alignment
# 假设你有原来的数据集和 lpips/arcface
from utils.metrics import lpips_fn, arcface_fn   # 如果没有，自己 import

opts = ...  # 你的 opts 对象（和原来一样）
model = Alignment(opts)
model.satd.train()
model.satd.to(opts.device)

# 只训练 SATD，其余全部冻结
for p in model.parameters():
    if "satd" not in str(p):
        p.requires_grad = False

optimizer = torch.optim.Adam(model.satd.parameters(), lr=2e-4, weight_decay=1e-6)

# 只用 500 张 failure 相关数据
dataset = YourDataset(root="data", split="train", num_samples=500)  # 重点采样 long-to-short + braid
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for epoch in range(2):  # 2 个 epoch 就够看初步效果
    total_loss = 0
    for batch in loader:
        # 走完整 align_images（里面已经调用 SATD）
        result = model.align_images(
            im_name1=batch['source'], im_name2=batch['shape'],
            name_to_embed=batch['embed_dict']
        )
        
        # 生成最终图像（用你的 generator）
        I_final = model.net.generator([batch['S']], input_is_latent=True,
                                      return_latents=False, start_layer=4, end_layer=18,
                                      layer_in=result['latent_F_align'])[0]
        
        hair_mask = result['HM_X']
        skin_mask = 1 - hair_mask
        I_gt = batch['gt_image']   # ground truth

        # 专治你两个问题的 loss
        L_shadow = F.l1_loss(I_final * skin_mask, I_gt * skin_mask)          # 皮肤阴影
        L_texture = lpips_fn(I_final * hair_mask, I_gt * hair_mask)          # 头发纹理
        loss = L_shadow * 0.4 + L_texture * 0.4   # 可根据效果微调权重

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/2 完成 | Loss: {total_loss/len(loader):.4f}")

torch.save(model.satd.state_dict(), "satd_mini_trained.pth")
print("? Mini-training 完成！")