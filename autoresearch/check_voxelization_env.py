#!/usr/bin/env python3
"""
check_voxelization_env.py

Preflight dependency check for voxelization training on RunPod.
"""

import importlib
import json
import sys

REQUIRED_MODULES = [
    "torch",
    "tensorboard",
    "open3d",
    "trimesh",
    "pytorch3d",
    "igl",
    "diplib",
]


def main() -> int:
    missing = []
    versions = {}

    for mod in REQUIRED_MODULES:
        try:
            imported = importlib.import_module(mod)
            versions[mod] = getattr(imported, "__version__", "unknown")
        except Exception as exc:
            missing.append({"module": mod, "error": str(exc)})

    payload = {
        "ok": len(missing) == 0,
        "versions": versions,
        "missing": missing,
    }
    print(json.dumps(payload, indent=2))

    if missing:
        print("\nRemediation hints:")
        print("- Install base requirements: `pip install -r requirements.txt`")
        print("- Install tensorboard explicitly if needed: `pip install tensorboard`")
        print("- Install open3d explicitly if needed: `pip install open3d`")
        print("- For pytorch3d, use a compatible wheel/index or build from source for your torch/cuda.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
