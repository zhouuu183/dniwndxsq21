"""Read-only V2.31 ViTMatte environment audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def main() -> None:
    model_path = Path(os.environ.get(
        "BLENDING_V8_VITMATTE_PATH",
        "pretrained_models/ViTMatte/vitmatte-small-composition-1k",
    ))
    audit = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_path": str(model_path),
        "model_path_exists": model_path.exists(),
        "transformers_available": False,
        "transformers_version": None,
        "transformers_torch_backend_available": False,
        "vitmatte_api_available": False,
        "failure_reason": None,
    }
    try:
        import transformers
        from transformers import AutoImageProcessor, VitMatteForImageMatting
        from transformers.utils import is_torch_available
        del AutoImageProcessor, VitMatteForImageMatting
        audit["transformers_available"] = True
        audit["transformers_version"] = transformers.__version__
        audit["vitmatte_api_available"] = True
        audit["transformers_torch_backend_available"] = bool(is_torch_available())
        if not audit["transformers_torch_backend_available"]:
            audit["decision"] = "V231_MATTING_BACKEND_FAIL"
            audit["failure_reason"] = (
                "Transformers disabled its PyTorch backend. Keep torch 1.13.1 "
                "and install the pinned Transformers 4.x requirements."
            )
    except ImportError as exc:
        audit["decision"] = "V231_MATTING_DEPENDENCY_MISSING"
        audit["failure_reason"] = str(exc)
    if audit.get("decision") is None and not model_path.exists():
        audit["decision"] = "V231_MATTING_CHECKPOINT_MISSING"
    if audit.get("decision") is None:
        audit["decision"] = "V231_MATTING_BACKEND_READY"
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
