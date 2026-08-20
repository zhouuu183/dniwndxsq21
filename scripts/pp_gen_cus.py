import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hair_swap import get_parser, HairFast
from scripts.pp_train import Trainer
from utils.train import seed_everything
from utils.image_utils import list_image_files


class ImageException(Exception):
    def __init__(self, image, message="Return image before PP"):
        self.image = image
        self.message = message
        super().__init__(self.message)


def hairfast_wo_pp(hair_fast):
    """Hijack downsampling step to obtain image before post-processing."""
    class RaiseDownsample(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, image):
            image = ((image[0] + 1) / 2).clip(0, 1)
            raise ImageException(image)

    def blend_images(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImageException as e:
                return e.image
        return wrapper

    hair_fast.blend.downsample_256 = RaiseDownsample()
    hair_fast.blend.blend_images = blend_images(hair_fast.blend.blend_images)


def load_image(path):
    img = Image.open(path).convert('RGB')
    return T.functional.to_tensor(img)   # [0,1], CxHxW


def get_max_processed_index(output_dir):
    """Scan output directory for saved dataset parts and return largest right index."""
    pattern = str(output_dir / 'pp_part_*_*.dataset')
    max_right = 0
    for fname in glob.glob(pattern):
        basename = os.path.basename(fname)
        parts = basename.split('_')
        if len(parts) >= 4:
            try:
                right = int(parts[3].split('.')[0])
                if right > max_right:
                    max_right = right
            except ValueError:
                continue
    return max_right


def main(args):
    seed_everything(args.seed)

    # ---------- 1. init models ----------
    model_parser = get_parser()
    model_args = model_parser.parse_args([])
    hair_fast = HairFast(model_args)
    hairfast_wo_pp(hair_fast)

    net_trainer = Trainer()
    # Trainer already sets models to eval mode internally

    # ---------- 2. prepare experiment list ----------
    os.makedirs(args.output, exist_ok=True)
    images = list_image_files(args.FFHQ)
    # random split (deterministic due to seed)
    face, shape, color = np.array_split(
        np.random.choice(images, size=3 * args.size), 3
    )

    exps = []
    for f, s, c in zip(face, shape, color):
        imgs = [Path(f).stem, Path(s).stem, Path(c).stem]
        target_name = f"{'_'.join(imgs)}.png"
        exps.append([f, target_name, (f, s, c)])

    total_exps = len(exps)

    # ---------- 3. resume from interrupted state ----------
    start_idx = get_max_processed_index(args.output)
    if start_idx >= total_exps:
        print(f"Already processed all {total_exps} samples. Exiting.")
        return

    print(f"Resuming from index {start_idx} (total samples: {total_exps})")

    # ---------- 4. main processing loop ----------
    left = start_idx
    right = min(left + args.save_batch_size, total_exps)

    while left < total_exps:
        current_batch_data = []          # to be saved as one .dataset file
        batch_sources = []              # accumulate for batched inference
        batch_targets = []
        batch_meta = []                # (source_path, down_target, target_mask, HT_E) after batch

        # process samples in current save block
        for idx in tqdm(range(left, right), desc=f"Processing [{left}, {right})"):
            exp = exps[idx]
            src_path = args.FFHQ / exp[0]
            target_path = exp[1]        # only used for filename, not needed here

            # --- load source image ---
            src_tensor = load_image(src_path).unsqueeze(0).cuda()   # 1,3,H,W

            # --- generate swapped hair (without PP) ---
            # hair_fast returns tensor in [0,1], possibly 3D (C,H,W) or 4D (1,C,H,W)
            tgt_tensor = hair_fast(args.FFHQ / exp[2][0],
                                   args.FFHQ / exp[2][1],
                                   args.FFHQ / exp[2][2])
            if tgt_tensor.dim() == 3:
                tgt_tensor = tgt_tensor.unsqueeze(0)
            tgt_tensor = tgt_tensor.cuda()

            # store for batched mask generation
            batch_sources.append(src_tensor)
            batch_targets.append(tgt_tensor)
            batch_meta.append((str(src_path), tgt_tensor.cpu()))   # keep cpu copy for later

            # when batch is full or this is the last sample in save block
            if len(batch_sources) == args.inference_batch_size or idx == right - 1:
                # --- batch inference on Trainer ---
                src_batch = torch.cat(batch_sources, dim=0)
                tgt_batch = torch.cat(batch_targets, dim=0)

                with torch.no_grad():
                    HS_D, _ = net_trainer.generate_mask(src_batch)
                    HT_D, HT_E = net_trainer.generate_mask(tgt_batch)
                    target_mask = (1 - HS_D) * (1 - HT_D)   # B,1,H,W
                    down_target = net_trainer.downsample_256(tgt_batch).clip(0, 1).cpu()

                # --- process each sample in the batch ---
                for i, (src_path, _) in enumerate(batch_meta):
                    current_batch_data.append((
                        src_path,
                        down_target[i],
                        target_mask[i].cpu(),
                        HT_E[i].cpu()
                    ))

                # --- cleanup ---
                del src_batch, tgt_batch, HS_D, HT_D, HT_E, target_mask, down_target
                torch.cuda.empty_cache()
                batch_sources.clear()
                batch_targets.clear()
                batch_meta.clear()

        # --- save current block ---
        part_file = args.output / f'pp_part_{left}_{right}.dataset'
        torch.save(current_batch_data, part_file)
        print(f"Saved {len(current_batch_data)} samples to {part_file}")

        # --- advance to next block ---
        left = right
        right = min(left + args.save_batch_size, total_exps)

    print("Dataset generation finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blending dataset generation with resume support')
    parser.add_argument('--FFHQ', type=Path, required=True,
                        help='Path to FFHQ dataset')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed')
    parser.add_argument('--size', type=int, default=10_000,
                        help='Number of triplets to generate')
    parser.add_argument('--output', type=Path, default='input/pp_dataset',
                        help='Directory to save dataset parts')
    parser.add_argument('--save_batch_size', type=int, default=5000,
                        help='Number of samples per saved .dataset file')
    parser.add_argument('--inference_batch_size', type=int, default=8,
                        help='Batch size for Trainer.generate_mask (GPU memory trade-off)')
    args = parser.parse_args()

    main(args)