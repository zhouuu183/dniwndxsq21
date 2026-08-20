"""Audit the explicit V2.32 PyMatting foreground backend without installing anything."""

from __future__ import annotations

import json
import platform
from pathlib import Path


def main() -> int:
    payload: dict[str, object] = {
        "python": platform.python_version(),
        "requirements": "requirements_v232_foreground.txt",
        "training": False,
        "auto_install": False,
    }
    try:
        import pymatting
        from pymatting import estimate_foreground_ml
    except Exception as exc:
        payload.update({
            "pymatting_available": False,
            "backend_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "decision": "V232_FOREGROUND_DEPENDENCY_MISSING",
        })
    else:
        payload.update({
            "pymatting_available": True,
            "pymatting_version": getattr(pymatting, "__version__", "unknown"),
            "backend_available": callable(estimate_foreground_ml),
            "backend": "pymatting.estimate_foreground_ml",
            "decision": "V232_FOREGROUND_BACKEND_READY" if callable(estimate_foreground_ml) else "V232_FOREGROUND_DEPENDENCY_MISSING",
        })
    output = Path("res") / "v232_diagnostic" / "foreground_env_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["decision"] == "V232_FOREGROUND_BACKEND_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
