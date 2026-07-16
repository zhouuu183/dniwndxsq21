import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v6 import HairFast_v6, get_parser_v6
from utils.image_utils import list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything
from utils.triplet_sampling_v6 import assert_disjoint_stems, sample_triplets


# ========================= 用户配置区域：只改这里 =========================
# 可选 "ffhq" 或 "small"。
USER_DATASET_PROFILE = "small"

# FFHQ 大数据集配置。
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("images/blending_dataset_v6")
USER_DATASET_SIZE_FFHQ = 3000

# small 小数据集配置。
USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_short")
USER_OUTPUT_DIR_SMALL = Path("images/blending_dataset_v6_small")
USER_DATASET_SIZE_SMALL = 100

# 通用配置。
USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_TRIPLETS = True
USER_COLOR_EQUALS_SHAPE = False

USER_DEVICE = "cuda"
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_V6_BASE_CKPT = "pretrained_models/PostProcess/pp_model.pth"

USER_ALIGN_V6_MODE = "author_cleanup"
USER_ALIGN_CLEANUP_STRENGTH = 0.70
USER_ALIGN_FILL_ITERATIONS = 3
USER_ALIGN_FILL_DILATION = 2
USER_ALIGN_FACE_STRUCTURE_WEIGHT = 0.90
USER_ALIGN_EAR_STRUCTURE_WEIGHT = 0.96
USER_ALIGN_BODY_STRUCTURE_WEIGHT = 0.88
USER_ALIGN_BACKGROUND_FILL_WEIGHT = 0.82
USER_ALIGN_OTHER_STRUCTURE_WEIGHT = 1.00
USER_DIFF_MASK_DILATE = 5
USER_DIFF_MASK_BLUR_KERNEL = 11
USER_DIFF_MASK_BLUR_SIGMA = 0.0
# ======================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "dataset_size": USER_DATASET_SIZE_FFHQ,
        },
        "small": {
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "dataset_size": USER_DATASET_SIZE_SMALL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(
            f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}. "
            f"Choose one of: {', '.join(sorted(profiles))}."
        )
    return profiles[USER_DATASET_PROFILE]


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_COLOR_ROOT = _DATASET_CFG["color_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_DATASET_SIZE = _DATASET_CFG["dataset_size"]


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


def align_instead_shape(hair_fast):
    def shape_module(func):
        def wrapper(*args, **kwargs):
            if kwargs.get("align_flag", False):
                return hair_fast.align.align_images(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    def align_module(func):
        def wrapper(*args, **kwargs):
            if "align_flag" in kwargs:
                kwargs = kwargs.copy()
                kwargs.pop("align_flag")
            return func(*args, **kwargs)

        return wrapper

    hair_fast.align.shape_module = shape_module(hair_fast.align.shape_module)
    hair_fast.align.align_images = align_module(hair_fast.align.align_images)


def save_align_bundle(output_dir: Path, file_name: str, align_result: dict):
    payload = {
        "latent_F": align_result["latent_F_align"],
        "target_hair_mask": align_result["HM_X"],
        "source_hair_mask": align_result["source_hair_mask"],
        "diff_mask": align_result["M_diff"],
        "valid_mask": align_result["M_valid"],
        "sample_mask": align_result["M_sample"],
        "diff_soft_mask": align_result["M_diff_soft"],
        "source_image_256": align_result["source_image_256"],
        "source_inpaint_256": align_result["source_inpaint_256"],
        "shape_inpaint_256": align_result["shape_inpaint_256"],
    }
    optional_keys = (
        "latent_F_align_author",
        "latent_F_align_cleanup",
        "author_free_mask_32",
        "author_interpolation_low_32",
        "author_cleanup_mask_32",
        "M_sample_face",
        "M_sample_ear",
        "M_sample_body",
        "M_sample_skin",
        "M_sample_background",
        "M_route_face",
        "M_route_ear",
        "M_route_body",
        "M_route_skin",
        "M_route_background",
        "M_route_other",
    )
    for key in optional_keys:
        if key in align_result:
            payload[key.lower()] = align_result[key]

    save_latents(output_dir, "Align", file_name, **payload)


def main():
    seed_everything(USER_RANDOM_SEED)

    model_args = get_parser_v6().parse_args([])
    model_args.device = USER_DEVICE
    model_args.ckpt = USER_STYLEGAN_CKPT
    model_args.rotate_checkpoint = USER_ROTATE_CKPT
    model_args.blending_checkpoint = USER_BLENDING_CKPT
    model_args.pp_v6_checkpoint = USER_PP_V6_BASE_CKPT
    model_args.align_v6_mode = USER_ALIGN_V6_MODE
    model_args.align_v6_cleanup_strength = USER_ALIGN_CLEANUP_STRENGTH
    model_args.align_v6_fill_iterations = USER_ALIGN_FILL_ITERATIONS
    model_args.align_v6_fill_dilation = USER_ALIGN_FILL_DILATION
    model_args.align_v6_face_structure_weight = USER_ALIGN_FACE_STRUCTURE_WEIGHT
    model_args.align_v6_ear_structure_weight = USER_ALIGN_EAR_STRUCTURE_WEIGHT
    model_args.align_v6_body_structure_weight = USER_ALIGN_BODY_STRUCTURE_WEIGHT
    model_args.align_v6_background_fill_weight = USER_ALIGN_BACKGROUND_FILL_WEIGHT
    model_args.align_v6_other_structure_weight = USER_ALIGN_OTHER_STRUCTURE_WEIGHT
    model_args.diff_mask_dilate = USER_DIFF_MASK_DILATE
    model_args.diff_mask_blur_kernel = USER_DIFF_MASK_BLUR_KERNEL
    model_args.diff_mask_blur_sigma = USER_DIFF_MASK_BLUR_SIGMA

    hair_fast = HairFast_v6(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)

    face_images = list_image_files(ACTIVE_FACE_ROOT)
    shape_images = list_image_files(ACTIVE_SHAPE_ROOT)
    color_images = shape_images if USER_COLOR_EQUALS_SHAPE else list_image_files(ACTIVE_COLOR_ROOT)

    if ACTIVE_FACE_ROOT.resolve() != ACTIVE_SHAPE_ROOT.resolve():
        assert_disjoint_stems(face_images, shape_images, "ACTIVE_FACE_ROOT", "ACTIVE_SHAPE_ROOT")
    if not USER_COLOR_EQUALS_SHAPE and ACTIVE_COLOR_ROOT.resolve() != ACTIVE_FACE_ROOT.resolve():
        assert_disjoint_stems(face_images, color_images, "ACTIVE_FACE_ROOT", "ACTIVE_COLOR_ROOT")

    triplets = sample_triplets(
        face_images,
        shape_images,
        color_images,
        ACTIVE_DATASET_SIZE,
        USER_ALLOW_REUSE_ACROSS_TRIPLETS,
        USER_RANDOM_SEED,
        USER_COLOR_EQUALS_SHAPE,
    )

    os.makedirs(ACTIVE_OUTPUT_DIR, exist_ok=True)
    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for file_1, file_2, file_3 in tqdm(triplets):
            im1, im2, im3 = map(lambda im: Path(im).stem, (file_1, file_2, file_3))
            print(im1, im2, im3, file=f_exps, flush=True)

            pt1 = ACTIVE_FACE_ROOT / file_1
            pt2 = ACTIVE_SHAPE_ROOT / file_2
            pt3 = ACTIVE_SHAPE_ROOT / file_3 if USER_COLOR_EQUALS_SHAPE else ACTIVE_COLOR_ROOT / file_3
            align_shape, align_color, name_to_embed = hair_fast(
                pt1,
                pt2,
                pt3,
                align_flag=True,
            )

            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{im1}.npz", latent_in=name_to_embed["face"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{im2}.npz", latent_in=name_to_embed["shape"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{im3}.npz", latent_in=name_to_embed["color"]["S"])
            save_align_bundle(ACTIVE_OUTPUT_DIR, f"{im1}_{im2}.npz", align_shape)
            save_align_bundle(ACTIVE_OUTPUT_DIR, f"{im1}_{im3}.npz", align_color)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"Saved {len(triplets)} blending triplets to {ACTIVE_OUTPUT_DIR}")
    print(f"face/source root: {ACTIVE_FACE_ROOT}")
    print(f"shape/reference root: {ACTIVE_SHAPE_ROOT}")
    print(f"color/reference root: {ACTIVE_SHAPE_ROOT if USER_COLOR_EQUALS_SHAPE else ACTIVE_COLOR_ROOT}")


if __name__ == "__main__":
    main()
