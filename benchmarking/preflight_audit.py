#!/usr/bin/env python3
"""Workspace preflight audit for pcdiff-implant baseline reproduction.

This script is intentionally stdlib-first so it can run before the ML stack is
installed. It validates the artifacts that make a benchmark run reproducible:
environment, datasets, CSV integrity, configs, and checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REQUIRED_IMPORTS = [
    "torch",
    "torchvision",
    "numpy",
    "nrrd",
    "mcubes",
    "trimesh",
]

OPTIONAL_IMPORTS = [
    "open3d",
    "pytorch3d",
    "torch_scatter",
]

CHECKPOINTS = [
    "pcdiff/checkpoints/model_best.pth",
    "pcdiff/checkpoints/model_latest.pth",
    "pcdiff/checkpoints/pcdiff_model_best.pth",
    "voxelization/checkpoints/model_best.pt",
]

CSV_FILES = [
    "pcdiff/datasets/SkullBreak/train.csv",
    "pcdiff/datasets/SkullBreak/test.csv",
]

CONFIG_FILES = [
    "voxelization/configs/gen_skullbreak.yaml",
    "voxelization/configs/train_skullbreak.yaml",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_spec_status(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    return {
        "module": module_name,
        "available": spec is not None,
        "origin": None if spec is None else spec.origin,
    }


def summarize_csv(path: Path, repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "rows": 0,
        "sample_rows": [],
        "absolute_rows": 0,
        "rows_inside_repo": 0,
        "rows_missing": 0,
        "missing_examples": [],
    }
    if not path.is_file():
        return result

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]

    result["rows"] = len(rows)
    result["sample_rows"] = [row[0] for row in rows[:3]]

    for row in rows:
        entry = Path(row[0])
        if entry.is_absolute():
            result["absolute_rows"] += 1
        try:
            entry.relative_to(repo_root)
            result["rows_inside_repo"] += 1
        except ValueError:
            pass
        if not entry.exists():
            result["rows_missing"] += 1
            if len(result["missing_examples"]) < 5:
                result["missing_examples"].append(str(entry))

    return result


def parse_simple_yaml(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    if not path.is_file():
        return items

    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*(?:#.*)?$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        key, value = match.groups()
        if value:
            items[key] = value.strip("'\"")
    return items


def summarize_config(path: Path, repo_root: Path) -> dict[str, Any]:
    values = parse_simple_yaml(path)
    interesting_keys = ["path", "model_file", "train_path", "eval_path", "out_dir", "generation_dir"]
    output_keys = {"out_dir", "generation_dir"}
    refs: dict[str, dict[str, Any]] = {}
    for key in interesting_keys:
        if key not in values:
            continue
        raw = values[key]
        resolved = (repo_root / raw).resolve() if not os.path.isabs(raw) else Path(raw)
        refs[key] = {
            "raw": raw,
            "kind": "output" if key in output_keys else "input",
            "exists": resolved.exists(),
            "resolved": str(resolved),
        }
    return {
        "path": str(path),
        "exists": path.is_file(),
        "refs": refs,
    }


def summarize_checkpoint(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": None,
        "sha256": None,
    }
    if not path.is_file():
        return result
    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit baseline reproducibility prerequisites.")
    parser.add_argument("--repo-root", default=".", help="Repository root to audit")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    report = {
        "repo_root": str(repo_root),
        "python": {
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "tools": {
            "uv": shutil.which("uv"),
            "git": shutil.which("git"),
            "nvidia_smi": shutil.which("nvidia-smi"),
        },
        "imports": {
            "required": [find_spec_status(name) for name in REQUIRED_IMPORTS],
            "optional": [find_spec_status(name) for name in OPTIONAL_IMPORTS],
        },
        "datasets": {
            "skullbreak_dir_exists": (repo_root / "pcdiff/datasets/SkullBreak").is_dir(),
            "csvs": [summarize_csv(repo_root / rel_path, repo_root) for rel_path in CSV_FILES],
        },
        "configs": [summarize_config(repo_root / rel_path, repo_root) for rel_path in CONFIG_FILES],
        "checkpoints": [summarize_checkpoint(repo_root / rel_path) for rel_path in CHECKPOINTS],
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("== Environment ==")
    print(f"repo_root: {report['repo_root']}")
    print(f"python: {report['python']['version']} ({report['python']['platform']})")
    for tool_name, tool_path in report["tools"].items():
        print(f"{tool_name}: {'OK' if tool_path else 'MISSING'}{f' ({tool_path})' if tool_path else ''}")

    print("\n== Imports ==")
    for group_name in ("required", "optional"):
        print(group_name + ":")
        for item in report["imports"][group_name]:
            state = "OK" if item["available"] else "MISSING"
            print(f"  - {item['module']}: {state}")

    print("\n== Dataset CSVs ==")
    for item in report["datasets"]["csvs"]:
        print(f"{item['path']}: {'OK' if item['exists'] else 'MISSING'}")
        if not item["exists"]:
            continue
        print(
            f"  rows={item['rows']} absolute_rows={item['absolute_rows']} "
            f"rows_inside_repo={item['rows_inside_repo']} rows_missing={item['rows_missing']}"
        )
        for example in item["missing_examples"]:
            print(f"  missing_example: {example}")

    print("\n== Config References ==")
    for item in report["configs"]:
        print(f"{item['path']}: {'OK' if item['exists'] else 'MISSING'}")
        for key, ref in item["refs"].items():
            if ref["kind"] == "output":
                print(f"  {key}: OUTPUT -> {ref['raw']}")
            else:
                print(f"  {key}: {'OK' if ref['exists'] else 'MISSING'} -> {ref['raw']}")

    print("\n== Checkpoints ==")
    for item in report["checkpoints"]:
        if item["exists"]:
            print(f"{item['path']}: OK size={item['size_bytes']} sha256={item['sha256']}")
        else:
            print(f"{item['path']}: MISSING")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
