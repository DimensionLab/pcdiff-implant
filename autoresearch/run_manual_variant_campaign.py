#!/usr/bin/env python3
"""
run_manual_variant_campaign.py

Execute a fixed family of manual PCDiff variants by temporarily overriding
top-level hyperparameters in train_pcdiff.py, running one training budget per
variant, and saving full audit artifacts.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = SCRIPT_DIR / "train_pcdiff.py"
RESULTS_DIR = SCRIPT_DIR / "results" / "manual_campaigns"


CAMPAIGNS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "attn_dropout_ablation_long": {
        "attn_off_long": {
            "USE_AMP": False,
            "LR_WARMUP_EPOCHS": 0,
            "USE_ATTENTION": False,
        },
        "dropout_zero_long": {
            "USE_AMP": False,
            "LR_WARMUP_EPOCHS": 0,
            "DROPOUT": 0.0,
        },
        "attn_off_dropout_zero_long": {
            "USE_AMP": False,
            "LR_WARMUP_EPOCHS": 0,
            "USE_ATTENTION": False,
            "DROPOUT": 0.0,
        },
    }
}


def py_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def apply_overrides(src: str, overrides: Dict[str, Any]) -> str:
    out = src
    for key, value in overrides.items():
        pattern = rf"(?m)^({re.escape(key)}\s*=\s*).*$"
        replacement = rf"\1{py_literal(value)}"
        out, count = re.subn(pattern, replacement, out, count=1)
        if count != 1:
            raise ValueError(f"Could not apply override for key: {key}")
    return out


def parse_result_from_stdout(stdout: str) -> Dict[str, Any]:
    lines = [line.rstrip() for line in stdout.splitlines()]
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    marker = "=== RESULT ==="
    if marker in stdout:
        tail = stdout.split(marker, 1)[1].strip()
        try:
            return json.loads(tail)
        except json.JSONDecodeError:
            return {"error": "Could not parse JSON result block", "result_block": tail[-4000:]}

    return {"error": "No JSON result found in stdout", "stdout_tail": stdout[-4000:]}


def run_variant(
    variant_name: str, overrides: Dict[str, Any], time_budget: int, checkpoint: str | None
) -> Dict[str, Any]:
    original_src = TRAIN_FILE.read_text()
    variant_src = apply_overrides(original_src, overrides)
    TRAIN_FILE.write_text(variant_src)

    cmd = [sys.executable, str(TRAIN_FILE), "--time-budget", str(time_budget)]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])

    started_at = datetime.now(timezone.utc)
    result = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
    )
    finished_at = datetime.now(timezone.utc)

    parsed = parse_result_from_stdout(result.stdout)
    payload = {
        "variant": variant_name,
        "overrides": overrides,
        "command": cmd,
        "returncode": result.returncode,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_sec": (finished_at - started_at).total_seconds(),
        "parsed_result": parsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    return payload


def save_artifacts(
    campaign_name: str, variant_name: str, overrides: Dict[str, Any], run_payload: Dict[str, Any]
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_DIR / campaign_name / f"{variant_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "overrides.json").write_text(json.dumps(overrides, indent=2))
    (out_dir / "result.json").write_text(json.dumps(run_payload, indent=2))
    (out_dir / "stdout.log").write_text(run_payload.get("stdout", ""))
    (out_dir / "stderr.log").write_text(run_payload.get("stderr", ""))
    return out_dir


def maybe_commit_logs(campaign_name: str, variant_name: str) -> None:
    repo_root = SCRIPT_DIR.parent
    if not (repo_root / ".git").exists():
        print("Skipping commit: git repository not found.")
        return

    add = subprocess.run(
        ["git", "add", "autoresearch/results/manual_campaigns", "autoresearch/train_pcdiff.py"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        print(f"Skipping commit: git add failed ({add.stderr.strip()})")
        return

    message = f"autoresearch: manual campaign {campaign_name} {variant_name}"
    commit = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if commit.returncode == 0:
        print(f"Committed: {message}")
        return

    out = (commit.stderr or commit.stdout or "").strip().lower()
    if "nothing to commit" in out:
        print(f"No changes to commit for {variant_name}.")
    else:
        print(f"Warning: commit failed for {variant_name}: {(commit.stderr or commit.stdout).strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed manual variant campaign for PCDiff autoresearch.")
    parser.add_argument(
        "--campaign",
        choices=sorted(CAMPAIGNS.keys()),
        default="attn_dropout_ablation_long",
        help="Campaign family to run.",
    )
    parser.add_argument("--time-budget", type=int, default=5400, help="Time budget per variant in seconds.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(SCRIPT_DIR / "results" / "checkpoints" / "latest.pth"),
        help="Checkpoint path to resume from. Set empty string to disable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned runs.")
    parser.add_argument("--commit-logs", action="store_true", help="Best-effort git commit after each variant.")
    args = parser.parse_args()

    campaign = CAMPAIGNS[args.campaign]
    original_src = TRAIN_FILE.read_text()

    checkpoint = args.checkpoint.strip()
    checkpoint = checkpoint if checkpoint else None
    if checkpoint and not Path(checkpoint).exists():
        print(f"Warning: checkpoint not found at {checkpoint}; variants will start without checkpoint resume.")
        checkpoint = None

    try:
        for variant_name, overrides in campaign.items():
            print(f"\n=== Variant: {variant_name} ===")
            print(json.dumps(overrides, indent=2))
            if args.dry_run:
                continue

            run_payload = run_variant(variant_name, overrides, args.time_budget, checkpoint)
            out_dir = save_artifacts(args.campaign, variant_name, overrides, run_payload)
            print(f"Saved artifacts: {out_dir}")

            summary_line = {
                "campaign": args.campaign,
                "variant": variant_name,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "artifact_dir": str(out_dir),
                "returncode": run_payload.get("returncode"),
                "val_loss_mean": run_payload.get("parsed_result", {}).get("metrics", {}).get("val_loss_mean"),
                "error": run_payload.get("parsed_result", {}).get("error"),
            }
            summary_file = RESULTS_DIR / args.campaign / "summary.jsonl"
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_file, "a") as f:
                f.write(json.dumps(summary_line) + "\n")

            if args.commit_logs:
                maybe_commit_logs(args.campaign, variant_name)

    finally:
        TRAIN_FILE.write_text(original_src)
        print("\nRestored train_pcdiff.py to original content.")


if __name__ == "__main__":
    main()
