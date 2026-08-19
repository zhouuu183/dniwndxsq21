import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)

for module_name in ("utils", "datasets", "models"):
    loaded = sys.modules.get(module_name)
    loaded_file = getattr(loaded, "__file__", "") if loaded is not None else ""
    if loaded is not None and loaded_file and not str(loaded_file).startswith(ROOT_DIR_STR):
        del sys.modules[module_name]

from hair_swap_v13 import HairFast_v13, get_parser_v13
from utils.spsa_eval_v13 import make_fixed_pair_panels, save_fixed_pair_panels
from utils.spsa_precompute_v13 import load_prior_npz, read_manifest, read_image_tensor

# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_FIXED_PAIR_MANIFEST = Path("output/spsa_train_v13_small/val_pairs_v13.jsonl")
USER_OUTPUT_DIR = Path("output/spsa_eval_v13")
USER_SPSA_CHECKPOINT = ""

USER_USE_SHADOW_CLEANUP = False
USER_SAVE_ALL = False
# ========================================================================


def build_parser():
    parser = argparse.ArgumentParser(description="SPSA v13 fixed-pair evaluator")
    parser.add_argument("--fixed_pair_manifest", type=Path, default=USER_FIXED_PAIR_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=USER_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=USER_DEVICE)
    parser.add_argument("--spsa_checkpoint", type=str, default=USER_SPSA_CHECKPOINT)
    parser.add_argument("--use_shadow_cleanup", type=int, default=int(USER_USE_SHADOW_CLEANUP))
    parser.add_argument("--save_all", type=int, default=int(USER_SAVE_ALL))
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spsa_args = get_parser_v13().parse_args([])
    spsa_args.device = args.device
    spsa_args.save_all = bool(args.save_all)
    spsa_args.use_shadow_cleanup = bool(args.use_shadow_cleanup)
    spsa_args.spsa_checkpoint = args.spsa_checkpoint
    spsa_model = HairFast_v13(spsa_args)

    records = read_manifest(args.fixed_pair_manifest)
    for record in tqdm(records, desc="Eval fixed SPSA pairs"):
        source_path = Path(record["source_path"])
        reference_path = Path(record["reference_path"])
        prior = load_prior_npz(record["prior_path"])

        spsa = spsa_model.swap(
            source_path,
            reference_path,
            reference_path,
            spsa_prior_path=record["prior_path"],
        )

        panels = make_fixed_pair_panels(
            source=read_image_tensor(source_path),
            reference=read_image_tensor(reference_path),
            spsa=spsa,
            bang_box=prior["bang_box"],
            tail_box=prior["tail_box"],
        )
        pair_output_dir = args.output_dir / record["sample_id"]
        save_fixed_pair_panels(pair_output_dir, panels)

    print(f"SPSA fixed-pair outputs saved to {args.output_dir}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
