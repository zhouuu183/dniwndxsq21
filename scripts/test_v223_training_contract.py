import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def function_source(path: Path, class_name: str, function_name: str) -> str:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function_node = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return ast.get_source_segment(source, function_node)


def main():
    trainer_path = ROOT / "scripts" / "blending_train_v8.py"
    inference_path = ROOT / "models" / "Blending_v8.py"
    swap_path = ROOT / "hair_swap_v8.py"

    trainer_source = trainer_path.read_text(encoding="utf-8-sig")
    trainer_tree = ast.parse(trainer_source)
    v830_runtime_imports = {
        alias.name
        for node in trainer_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "models.v830_runtime_inputs"
        for alias in node.names
    }
    assert "build_v830_runtime_inputs" in v830_runtime_imports

    train_loop = function_source(trainer_path, "BlendingTrainerV8", "train_loop")
    cache_builder = next(
        node for node in trainer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_cache_model"
    )
    cache_builder_source = ast.get_source_segment(trainer_source, cache_builder)
    assert "model_args.v231_enabled = False" in cache_builder_source
    assert "model_args.v232_enabled = False" in cache_builder_source
    assert "model_args.v233_enabled = False" in cache_builder_source
    v228_diagnostic = function_source(
        trainer_path, "BlendingTrainerV8", "run_v228_diagnostic"
    )
    v229_diagnostic = function_source(
        trainer_path, "BlendingTrainerV8", "run_v229_diagnostic"
    )
    v230_diagnostic = function_source(
        trainer_path, "BlendingTrainerV8", "run_v230_diagnostic"
    )
    assert "run_v231_diagnostic" in train_loop
    assert "USER_V231_DIAGNOSTIC_ONLY" in train_loop
    assert "run_v230_diagnostic" not in train_loop
    assert "run_v229_diagnostic" not in train_loop
    assert "run_v228_diagnostic" not in train_loop
    assert "run_v227_diagnostic" not in train_loop
    assert "run_v223_pretrain_diagnostic" not in train_loop
    assert "run_v222_pretrain_diagnostic" not in train_loop
    assert "train_one_epoch_v222" not in train_loop
    assert "torch.save" not in train_loop
    assert "phase_a_baseline = None" in v229_diagnostic
    assert "phase_a_summary=phase_a_baseline" in v229_diagnostic
    assert "phase_a_baseline" not in v228_diagnostic
    assert "F." not in v230_diagnostic
    assert "tnf.interpolate" in v230_diagnostic

    v223_loss = function_source(trainer_path, "BlendingTrainerV8", "calc_loss_v223")
    assert "pseudo_l_low" in v223_loss
    assert "ref_median_l" in v223_loss
    assert "hf_preserve" in v223_loss
    assert "core_luma_keep" not in v223_loss

    checkpoint = function_source(
        trainer_path, "BlendingTrainerV8", "_save_v225_checkpoint"
    )
    assert "FULL_COLOR_ARCH_V8_8" in checkpoint
    assert '"v2.25"' in checkpoint

    inference_source = inference_path.read_text(encoding="utf-8-sig")
    assert "FullColorToneSelectiveProjectorV823" in inference_source
    assert 'color_bundle["metrics"]["delta_l_global"]' in inference_source
    assert "FULL_COLOR_ARCH_V8_6" in inference_source
    assert "FULL_COLOR_ARCH_V8_7" in inference_source
    assert "FULL_COLOR_ARCH_V8_8" in inference_source
    assert "FULL_COLOR_ARCH_V8_9" in inference_source
    assert "run_v226_projector_debug" in inference_source
    assert "StrongAnchorAppearanceCompositorV828" in inference_source
    assert "build_v828_runtime_inputs" in inference_source
    assert "apply_pp_hair_lock_v828" in inference_source
    assert "HybridHairCarrierV829" in inference_source
    assert "BoundaryRecompositionV829" in inference_source
    assert "build_v829_runtime_inputs" in inference_source
    assert "apply_pp_hair_ownership_lock_v829" in inference_source
    assert "AchromaticHairCarrierV830" in inference_source
    assert "HairTopologyRepairV830" in inference_source
    assert "PPGuidedFinalV830" in inference_source
    assert "build_v830_runtime_inputs" in inference_source
    assert "HairMattingV831" in inference_source
    assert "PPUnifiedFinalV831" in inference_source
    assert "build_v831_runtime_inputs" in inference_source
    assert "FBConfidenceV833" in inference_source
    assert "ReliableHairForegroundTargetV833" in inference_source
    assert "FaceSideAlphaCalibratorV833" in inference_source
    assert "BackgroundTargetV833" in inference_source
    assert "build_v833_runtime_inputs" in inference_source

    blend_images = function_source(inference_path, "Blending_v8", "blend_images")
    v233_branch = blend_images.split("if self.v233 and needs_blend:", 1)[1].split(
        "elif self.v232 and needs_blend:", 1
    )[0]
    assert "run_v833_pipeline" in v233_branch
    assert "self.v231_matting" in v233_branch
    assert "self.v231_finalizer" not in v233_branch
    assert "alpha_eff" not in v233_branch or "final_aux" in v233_branch
    v231_branch = blend_images.split("if self.v231 and needs_blend:", 1)[1].split(
        "elif self.v230 and needs_blend:", 1
    )[0]
    assert "self.v231_matting" in v231_branch
    assert "self.v231_finalizer" in v231_branch
    assert "target_hair_mask=HM_X" in v231_branch
    assert "apply_pp_hair_ownership_lock_v829" not in v231_branch
    assert "soft_core_weight" not in v231_branch
    assert "achromatic_core_rgb" not in v231_branch
    assert "I_anchor" not in v231_branch
    assert "alpha_hr >" not in v231_branch

    v230_branch = blend_images.split("if self.v230 and needs_blend:", 1)[1].split(
        "elif self.v229 and needs_blend:", 1
    )[0]
    assert "self.v230_finalizer" in v230_branch
    assert "apply_pp_hair_ownership_lock_v829" not in v230_branch

    swap_source = swap_path.read_text(encoding="utf-8-sig")
    assert "v2_26_boundary_target_metric_aligned_small/checkpoints/v226_boundary_target_metric_aligned_pass.pth" in swap_source
    assert "--disable-v228" in swap_source
    assert "--disable-v229" in swap_source
    assert "--disable-v230" in swap_source
    assert "--disable-v231" in swap_source
    assert "--disable-v232" in swap_source
    assert "--disable-v233" in swap_source
    assert "BLENDING_V8_VITMATTE_PATH" in swap_source
    print("v2.31-v2.33 diagnostic/inference contract tests passed")


if __name__ == "__main__":
    main()
