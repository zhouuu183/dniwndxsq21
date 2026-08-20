import ast
import copy
from pathlib import Path

import torch

from v8_adapter_test_utils import build_adapter, run_adapter, synthetic_inputs
from utils.v221_feasibility import (
    aggregate_fixed_alpha_metrics,
    assert_finite_json,
    build_normal_color_manifest,
    choose_direct_anchor_feasibility,
    evaluate_v221_checkpoint_gate,
    should_start_phase1,
)


def build_sweep_records():
    records_by_alpha = {}
    for alpha in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        records = []
        for index in range(20):
            progress = alpha * 0.9
            improvement = alpha * 0.8
            records.append({
                "sample_id": f"sample-{index:02d}",
                "safe_fraction": 0.8,
                "ref_base_ab_distance": 4.0 + 0.1 * index,
                "reference_chroma_magnitude": 10.0 + 0.1 * index,
                "pseudo_to_reference_ab_error": 1.0,
                "pseudo_to_reference_hue_error": 3.0,
                "final_to_reference_ab_error": 8.0 * (1.0 - improvement),
                "final_to_reference_hue_error": 10.0 * (1.0 - improvement),
                "final_to_reference_chroma_error": 6.0 * (1.0 - improvement),
                "reference_progress": progress,
                "color_improvement_ratio": improvement,
                "edge_luma_excess_mean": 0.1 + 0.1 * alpha,
                "edge_luma_excess_fraction": 0.02 + 0.01 * alpha,
                "outer_bg_keep_l1": 0.01 + 0.002 * alpha,
                "face_keep_l1": 0.01 + 0.002 * alpha,
                "global_frac_l_excess_gt12": 0.01,
            })
        records_by_alpha[alpha] = records
    return records_by_alpha


def assert_failed_pretrain_returns_before_training() -> None:
    source = Path(__file__).with_name("blending_train_v8.py").read_text(
        encoding="utf-8-sig"
    )
    module = ast.parse(source)
    trainer = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "BlendingTrainerV8"
    )
    train_loop = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_loop"
    )
    train_loop_text = ast.unparse(train_loop)
    assert "run_v225_pretrain_diagnostic" in train_loop_text
    assert "V225_REFERENCE_BOUNDARY_PASS" in train_loop_text
    diagnostic_call = next(
        node
        for node in ast.walk(train_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_v225_pretrain_diagnostic"
    )
    loop_nodes = [node for node in ast.walk(train_loop) if isinstance(node, ast.For)]
    assert loop_nodes
    assert diagnostic_call.lineno < min(node.lineno for node in loop_nodes)
    assert "train_one_epoch_v222" not in train_loop_text


def assert_cpu_half_preview_is_promoted() -> None:
    source = Path(__file__).with_name("blending_train_v8.py").read_text(
        encoding="utf-8-sig"
    )
    module = ast.parse(source)
    save_preview_node = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_preview"
    )
    captured = {}

    def capture_image(tensor, path):
        captured["tensor"] = tensor
        captured["path"] = path

    namespace = {"Path": Path, "torch": torch, "save_image": capture_image}
    exec(
        compile(
            ast.Module(body=[save_preview_node], type_ignores=[]),
            "<save_preview>",
            "exec",
        ),
        namespace,
    )
    namespace["save_preview"](
        Path("preview-test.png"),
        [torch.ones(1, 3, 2, 2, dtype=torch.float16)],
    )
    assert captured["tensor"].dtype == torch.float32
    assert captured["tensor"].device.type == "cpu"


def main():
    assert_failed_pretrain_returns_before_training()
    assert_cpu_half_preview_is_promoted()
    torch.manual_seed(221)
    model = build_adapter(alpha_init=0.75, layer_offset_max=0.06)
    latent_face, latent_color, descriptor = synthetic_inputs(batch=3)
    ones = torch.ones(3)

    state_before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        _, fixed_aux = run_adapter(
            model,
            latent_face,
            latent_color,
            descriptor,
            ones,
            ones,
            ones,
            correction_enabled=False,
            layer_mix_override=0.6,
        )
    assert torch.allclose(fixed_aux["layer_mix"], torch.full_like(fixed_aux["layer_mix"], 0.6))
    assert all(torch.equal(state_before[key], model.state_dict()[key]) for key in state_before)

    model.configure_v221_layer_adaptation()
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names
    assert all(
        name.startswith(("descriptor_encoder.", "layer_offset_head."))
        for name in trainable_names
    )
    output, adaptive_aux = run_adapter(
        model,
        latent_face,
        latent_color,
        descriptor,
        ones,
        ones,
        ones,
        correction_enabled=False,
        base_alpha_override=0.85,
    )
    assert torch.allclose(
        adaptive_aux["base_alpha"], torch.full_like(adaptive_aux["base_alpha"], 0.85)
    )
    assert torch.allclose(
        adaptive_aux["layer_mix"], torch.full_like(adaptive_aux["layer_mix"], 0.85)
    )
    output.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.strength_head.parameters())
    assert all(parameter.grad is None for parameter in model.correction_mlp.parameters())
    assert any(parameter.grad is not None for parameter in model.layer_offset_head.parameters())

    records_by_alpha = build_sweep_records()
    manifest = build_normal_color_manifest(records_by_alpha[0.0], 0.35)
    assert manifest["status"] == "OK"
    assert manifest["normal_fraction"] >= 0.30
    summary = aggregate_fixed_alpha_metrics(records_by_alpha, manifest)
    decision = choose_direct_anchor_feasibility(summary, manifest)
    assert decision["decision"] == "PASS_STRONG"
    assert should_start_phase1(decision)
    assert decision["alpha_star"] in (0.7, 0.8, 0.9, 1.0)
    assert summary["monotonic_progress_fraction"] == 1.0
    assert_finite_json({"manifest": manifest, "summary": summary, "decision": decision})

    failed_records = build_sweep_records()
    for records in failed_records.values():
        for record in records:
            record["reference_progress"] = 0.2
            record["color_improvement_ratio"] = 0.1
    failed_summary = aggregate_fixed_alpha_metrics(failed_records, manifest)
    failed_decision = choose_direct_anchor_feasibility(failed_summary, manifest)
    assert failed_decision["decision"] == "FAIL_DIRECTION"
    assert not should_start_phase1(failed_decision)

    phase0_metrics = {
        "median_reference_progress": 0.70,
        "median_color_improvement_ratio": 0.60,
    }
    accepted_gate = evaluate_v221_checkpoint_gate(
        median_reference_progress=0.65,
        median_color_improvement_ratio=0.58,
        effective_alpha_mean=0.87,
        alpha_star=0.90,
        phase0_alpha_star_metrics=phase0_metrics,
    )
    assert accepted_gate["color_gate_pass"]
    assert not accepted_gate["training_regressed_from_fixed_anchor"]
    regressed_gate = evaluate_v221_checkpoint_gate(
        median_reference_progress=0.65,
        median_color_improvement_ratio=0.56,
        effective_alpha_mean=0.87,
        alpha_star=0.90,
        phase0_alpha_star_metrics=phase0_metrics,
    )
    assert regressed_gate["training_regressed_from_fixed_anchor"]
    assert not regressed_gate["color_gate_pass"]
    drifted_gate = evaluate_v221_checkpoint_gate(
        median_reference_progress=0.65,
        median_color_improvement_ratio=0.58,
        effective_alpha_mean=0.85,
        alpha_star=0.90,
        phase0_alpha_star_metrics=phase0_metrics,
    )
    assert drifted_gate["anchor_drift_warning"]
    assert not drifted_gate["color_gate_pass"]
    print(
        "v2.21 feasibility fixed-anchor tests passed: "
        f"normal_fraction={manifest['normal_fraction']:.2f} "
        f"alpha_star={decision['alpha_star']}"
    )


if __name__ == "__main__":
    main()
