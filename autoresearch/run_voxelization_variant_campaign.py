#!/usr/bin/env python3
"""
run_voxelization_variant_campaign.py

Minimal campaign harness for voxelization ablations on SkullBreak.
It generates per-variant config files, launches voxelization training,
and persists audit artifacts.
"""

import argparse
import json
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BASE_CONFIG = REPO_ROOT / "voxelization" / "configs" / "train_skullbreak.yaml"
TRAIN_SCRIPT = REPO_ROOT / "voxelization" / "train.py"
RESULTS_ROOT = SCRIPT_DIR / "results" / "voxelization_campaigns"
DEFAULT_TRAIN_MANIFEST = REPO_ROOT / "pcdiff" / "datasets" / "SkullBreak" / "voxelization" / "train.csv"
DEFAULT_EVAL_MANIFEST = REPO_ROOT / "pcdiff" / "datasets" / "SkullBreak" / "voxelization" / "eval.csv"


CAMPAIGNS: Dict[str, Dict[str, Dict[str, str]]] = {
    "vox_skullbreak_ablation_v1": {
        "baseline_short": {},
        "lr_3e4": {
            "lr": "3e-4",
        },
        "batch1_lr5e4": {
            "batch_size": "1",
            "lr": "5e-4",
        },
        "batch1_lr3e4": {
            "batch_size": "1",
            "lr": "3e-4",
        },
        "batch1_cdim48_lr5e4": {
            "batch_size": "1",
            "c_dim": "48",
            "lr": "5e-4",
        },
        "cdim_48": {
            "c_dim": "48",
        },
        "grid_res_384": {
            "grid_res": "384",
        },
    }
}


def replace_yaml_key_line(src: str, key: str, value: str, required: bool = True) -> str:
    pattern = rf"(?m)^(\s*{re.escape(key)}\s*:\s*).*$"
    replaced, count = re.subn(pattern, lambda m: f"{m.group(1)}{value}", src, count=1)
    if count != 1 and required:
        raise ValueError(f"Could not find unique key '{key}' in config for override.")
    return replaced


def make_variant_config(
    base_text: str,
    variant_name: str,
    overrides: Dict[str, str],
    epochs: int,
    validate_every: int,
    out_dir: Path,
    train_manifest: Path,
    eval_manifest: Path,
) -> str:
    cfg = base_text

    # Force deterministic output location under autoresearch audit tree.
    cfg = replace_yaml_key_line(cfg, "train_path", train_manifest.as_posix())
    cfg = replace_yaml_key_line(cfg, "eval_path", eval_manifest.as_posix())
    cfg = replace_yaml_key_line(cfg, "out_dir", out_dir.as_posix())
    cfg = replace_yaml_key_line(cfg, "total_epochs", str(epochs))
    cfg = replace_yaml_key_line(cfg, "validate_every", str(validate_every))
    cfg = replace_yaml_key_line(cfg, "timestamp", "False", required=False)

    for key, value in overrides.items():
        cfg = replace_yaml_key_line(cfg, key, value)

    # Tag variant name into generation dir for traceability if generation is run later.
    if re.search(r"(?m)^\s*generation_dir\s*:", cfg):
        cfg = replace_yaml_key_line(cfg, "generation_dir", f"generation_{variant_name}")

    return cfg


def run_variant(config_path: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TRAIN_SCRIPT), str(config_path)]
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )


def sample_manifest(src: Path, dst: Path, max_samples: int, seed: int) -> Path:
    rows = [line for line in src.read_text().splitlines() if line.strip()]
    if max_samples <= 0 or len(rows) <= max_samples:
        dst.write_text("\n".join(rows) + ("\n" if rows else ""))
        return dst

    rng = random.Random(seed)
    sampled = rng.sample(rows, max_samples)
    dst.write_text("\n".join(sampled) + "\n")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Run voxelization variant campaign.")
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS.keys()), default="vox_skullbreak_ablation_v1")
    parser.add_argument("--epochs", type=int, default=80, help="Epochs written into each generated config.")
    parser.add_argument("--validate-every", type=int, default=10, help="Validation cadence per variant config.")
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated subset of variant names to run (default: all variants in campaign).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, sample up to this many rows from train/eval manifests per variant for faster smoke/ablation runs.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed used when --max-samples > 0.")
    args = parser.parse_args()

    missing_manifests = [p for p in (DEFAULT_TRAIN_MANIFEST, DEFAULT_EVAL_MANIFEST) if not p.exists()]
    if missing_manifests:
        missing_str = ", ".join(str(p) for p in missing_manifests)
        raise FileNotFoundError(
            f"Missing voxelization manifests: {missing_str}. "
            "Generate them first with voxelization preprocessing + split scripts."
        )

    base_text = BASE_CONFIG.read_text()
    campaign = CAMPAIGNS[args.campaign]
    selected_variants = None
    if args.variants.strip():
        selected_variants = {v.strip() for v in args.variants.split(",") if v.strip()}
        unknown = selected_variants.difference(campaign.keys())
        if unknown:
            raise ValueError(f"Unknown variants requested: {sorted(unknown)}")

    campaign_dir = RESULTS_ROOT / args.campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)

    for variant_name, overrides in campaign.items():
        if selected_variants is not None and variant_name not in selected_variants:
            continue

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        variant_dir = campaign_dir / f"{variant_name}_{ts}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        out_dir = variant_dir / "train_out"
        train_manifest = DEFAULT_TRAIN_MANIFEST
        eval_manifest = DEFAULT_EVAL_MANIFEST
        if args.max_samples > 0:
            train_manifest = sample_manifest(
                DEFAULT_TRAIN_MANIFEST,
                variant_dir / "train.sampled.csv",
                args.max_samples,
                args.seed,
            )
            eval_manifest = sample_manifest(
                DEFAULT_EVAL_MANIFEST,
                variant_dir / "eval.sampled.csv",
                args.max_samples,
                args.seed + 1,
            )

        config_text = make_variant_config(
            base_text=base_text,
            variant_name=variant_name,
            overrides=overrides,
            epochs=args.epochs,
            validate_every=args.validate_every,
            out_dir=out_dir,
            train_manifest=train_manifest,
            eval_manifest=eval_manifest,
        )
        cfg_path = variant_dir / "train_skullbreak.variant.yaml"
        cfg_path.write_text(config_text)
        (variant_dir / "overrides.json").write_text(json.dumps(overrides, indent=2))

        print(f"\n=== {variant_name} ===")
        print(f"config: {cfg_path}")
        print(json.dumps(overrides, indent=2))

        if args.dry_run:
            continue

        started_at = datetime.now(timezone.utc)
        proc = run_variant(cfg_path)
        finished_at = datetime.now(timezone.utc)

        (variant_dir / "stdout.log").write_text(proc.stdout or "")
        (variant_dir / "stderr.log").write_text(proc.stderr or "")
        (variant_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "returncode": proc.returncode,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_sec": (finished_at - started_at).total_seconds(),
                    "command": [sys.executable, str(TRAIN_SCRIPT), str(cfg_path)],
                },
                indent=2,
            )
        )

        summary_line = {
            "variant": variant_name,
            "artifact_dir": str(variant_dir),
            "returncode": proc.returncode,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        }
        with open(campaign_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps(summary_line) + "\n")

        print(f"returncode={proc.returncode} artifacts={variant_dir}")


if __name__ == "__main__":
    main()
