#!/usr/bin/env python3
"""
run_experiments.py — Autonomous experiment orchestrator for PCDiff autoresearch.

This script runs the autoresearch loop:
  1. Read program.md for research directions
  2. Read current train_pcdiff.py and experiment history
  3. Ask LLM (via OpenRouter) to propose a modification
  4. Apply the modification to train_pcdiff.py
  5. Run training with fixed time budget
  6. Evaluate and accept/reject based on validation loss improvement
  7. If rejected, revert train_pcdiff.py; if accepted, keep changes
  8. Repeat

Usage:
    python run_experiments.py                          # Run experiment loop
    python run_experiments.py --max-experiments 10     # Run 10 experiments
    python run_experiments.py --dry-run                # Show what would be changed

Environment variables:
    OPENROUTER_API_KEY  — Required for LLM access
"""

import argparse
import copy
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = SCRIPT_DIR / "train_pcdiff.py"
PROGRAM_FILE = SCRIPT_DIR / "program_pcdiff.md"
RESULTS_DIR = SCRIPT_DIR / "results"

# LLM config
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "anthropic/claude-sonnet-4"  # Good balance of cost and capability

# Experiment config
DEFAULT_TIME_BUDGET = 900    # 15 minutes per experiment
MAX_EXPERIMENTS = 100

# ---------------------------------------------------------------------------
# LLM Client (OpenRouter)
# ---------------------------------------------------------------------------

def call_llm(messages: list, temperature: float = 0.7) -> str:
    """Call LLM via OpenRouter API. Returns the assistant response text."""
    import urllib.request
    import urllib.error

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_file = Path(__file__).resolve().parent.parent / "web_viewer" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Set it in environment or web_viewer/.env")

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 32768,
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dimensionlab/pcdiff-implant",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"OpenRouter API error {e.code}: {body}")


# ---------------------------------------------------------------------------
# Experiment History
# ---------------------------------------------------------------------------

def load_history() -> list:
    """Load experiment history from JSONL file."""
    history_file = RESULTS_DIR / "experiments.jsonl"
    if not history_file.exists():
        return []
    with open(history_file) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_history_summary(history: list, last_n: int = 10) -> str:
    """Format recent experiment history for the LLM prompt."""
    if not history:
        return "No experiments run yet."

    recent = history[-last_n:]
    lines = []
    for exp in recent:
        status = "ACCEPTED" if exp.get("accepted") else "REJECTED"
        cd = exp.get("metrics", {}).get("val_loss_mean", "N/A")
        lines.append(f"- [{status}] {exp.get('experiment_id', '?')}: val_loss={cd:.6f}" if isinstance(cd, float) else f"- [{status}] {exp.get('experiment_id', '?')}: val_loss={cd}")
        if exp.get("diff"):
            # Show first 5 lines of diff
            diff_lines = exp["diff"].split("\n")[:5]
            for dl in diff_lines:
                lines.append(f"    {dl}")
    return "\n".join(lines)


def get_best_cd(history: list) -> float:
    """Get best accepted validation loss."""
    best = float("inf")
    for exp in history:
        if exp.get("accepted"):
            cd = exp.get("metrics", {}).get("val_loss_mean", float("inf"))
            if isinstance(cd, (int, float)) and cd < best:
                best = cd
    return best


# ---------------------------------------------------------------------------
# Code Modification
# ---------------------------------------------------------------------------

def propose_modification(current_code: str, program: str, history_summary: str, best_cd: float) -> str:
    """Ask LLM to propose a modification to train_pcdiff.py."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an ML research agent running autonomous experiments on a Point Cloud Diffusion model. "
                "Your goal is to minimize validation loss (MSE on noise prediction) by modifying the training script. "
                "You make ONE focused change per experiment. Respond with ONLY the complete modified train_pcdiff.py file, "
                "nothing else — no explanations before or after the code. The code must be valid Python. "
                "CRITICAL: You MUST output the ENTIRE file — every class, function, and the if __name__ == '__main__' block. "
                "Do NOT truncate or omit any part of the file. The output must have at least as many lines as the input."
            ),
        },
        {
            "role": "user",
            "content": f"""## Research Program
{program}

## Current Best Validation Loss
{best_cd if best_cd < float('inf') else 'Not yet established (first run)'}

## Recent Experiment History
{history_summary}

## Current train_pcdiff.py
```python
{current_code}
```

Based on the research program and experiment history, propose your next modification.
Make ONE focused change (or a small coherent group of related changes).
Respond with the COMPLETE modified train_pcdiff.py file.""",
        },
    ]

    response = call_llm(messages, temperature=0.7)

    # Extract code from response (handle markdown code blocks)
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
    else:
        code = response

    code = code.strip()

    # Guard against truncated LLM output
    original_lines = len(current_code.splitlines())
    proposed_lines = len(code.splitlines())
    if proposed_lines < original_lines * 0.8:
        raise ValueError(
            f"Proposed code looks truncated ({proposed_lines} lines vs {original_lines} original). "
            f"Rejecting to protect the training script."
        )

    return code


def compute_diff(old_code: str, new_code: str) -> str:
    """Compute unified diff between old and new code."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="train_pcdiff.py.old", tofile="train_pcdiff.py.new")
    return "".join(diff)


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(time_budget: int, checkpoint: str = None) -> dict:
    """Run train_pcdiff.py and return results."""
    cmd = [sys.executable, str(TRAIN_FILE), "--time-budget", str(time_budget)]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=time_budget + 300,  # Extra 5 min buffer for eval
        )

        print("STDOUT (last 30 lines):")
        stdout_lines = result.stdout.strip().split("\n")
        for line in stdout_lines[-30:]:
            print(f"  {line}")

        if result.returncode != 0:
            print(f"\nSTDERR (last 20 lines):")
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-20:]:
                print(f"  {line}")
            return {"error": f"Process exited with code {result.returncode}", "stderr": result.stderr[-2000:]}

        # Parse result from output (last line should be JSON)
        for line in reversed(stdout_lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass

        # Try to find JSON block in output
        in_json = False
        json_lines = []
        for line in stdout_lines:
            if "=== RESULT ===" in line:
                in_json = True
                continue
            if in_json:
                json_lines.append(line)

        if json_lines:
            try:
                return json.loads("\n".join(json_lines))
            except json.JSONDecodeError:
                pass

        return {"error": "Could not parse results from output", "stdout": result.stdout[-2000:]}

    except subprocess.TimeoutExpired:
        return {"error": "Experiment timed out"}
    except Exception as e:
        return {"error": str(e)}


def run_autoresearch_loop(max_experiments: int = MAX_EXPERIMENTS, time_budget: int = DEFAULT_TIME_BUDGET,
                           dry_run: bool = False):
    """Main autoresearch loop."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Read program
    program = PROGRAM_FILE.read_text()
    print(f"Loaded research program from {PROGRAM_FILE}")

    # Load history
    history = load_history()
    print(f"Experiment history: {len(history)} previous experiments")

    # Backup original train file
    backup_path = TRAIN_FILE.with_suffix(".py.original")
    if not backup_path.exists():
        shutil.copy2(TRAIN_FILE, backup_path)
        print(f"Backed up original: {backup_path}")

    # Run baseline if no history
    if not history:
        print("\n=== RUNNING BASELINE ===")
        if not dry_run:
            baseline_result = run_experiment(time_budget)
            if "error" not in baseline_result:
                entry = {
                    "experiment_id": "baseline",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "metrics": baseline_result.get("metrics", {}),
                    "config_summary": baseline_result.get("config", {}),
                    "accepted": True,
                    "diff": "",
                }
                with open(RESULTS_DIR / "experiments.jsonl", "a") as f:
                    f.write(json.dumps(entry) + "\n")
                history.append(entry)
                print(f"Baseline val_loss: {baseline_result.get('metrics', {}).get('val_loss_mean', 'N/A')}")
            else:
                print(f"Baseline failed: {baseline_result['error']}")
                return
        else:
            print("[DRY RUN] Would run baseline experiment")

    for exp_idx in range(max_experiments):
        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT {exp_idx + 1}/{max_experiments}")
        print(f"{'#'*60}")

        # Read current code
        current_code = TRAIN_FILE.read_text()
        history_summary = format_history_summary(history)
        best_cd = get_best_cd(history)
        print(f"Current best CD: {best_cd:.6f}" if best_cd < float("inf") else "No best CD yet")

        # Propose modification
        print("\nAsking LLM for modification proposal...")
        try:
            proposed_code = propose_modification(current_code, program, history_summary, best_cd)
        except Exception as e:
            print(f"LLM call failed: {e}")
            time.sleep(30)  # Back off before retry
            continue

        # Compute diff
        diff = compute_diff(current_code, proposed_code)
        if not diff.strip():
            print("LLM proposed no changes. Skipping.")
            continue

        print(f"\n--- Proposed diff ({len(diff.splitlines())} lines) ---")
        for line in diff.splitlines()[:40]:
            print(f"  {line}")
        if len(diff.splitlines()) > 40:
            print(f"  ... ({len(diff.splitlines()) - 40} more lines)")

        if dry_run:
            print("[DRY RUN] Would apply this change and run experiment")
            continue

        # Apply modification
        TRAIN_FILE.write_text(proposed_code)
        print(f"\nApplied modification to {TRAIN_FILE}")

        # Verify syntax
        try:
            compile(proposed_code, str(TRAIN_FILE), "exec")
        except SyntaxError as e:
            print(f"Syntax error in proposed code: {e}")
            TRAIN_FILE.write_text(current_code)
            entry = {
                "experiment_id": f"exp_{exp_idx + 1:04d}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": {},
                "config_summary": {},
                "accepted": False,
                "diff": diff,
                "error": f"Syntax error: {e}",
            }
            with open(RESULTS_DIR / "experiments.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            history.append(entry)
            continue

        # Run experiment
        # Use latest checkpoint for incremental training
        latest_ckpt = RESULTS_DIR / "checkpoints" / "latest.pth"
        ckpt_arg = str(latest_ckpt) if latest_ckpt.exists() else None
        result = run_experiment(time_budget, checkpoint=ckpt_arg)

        if "error" in result:
            print(f"\nExperiment FAILED: {result['error']}")
            # Revert
            TRAIN_FILE.write_text(current_code)
            print("Reverted train_pcdiff.py")

            entry = {
                "experiment_id": f"exp_{exp_idx + 1:04d}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": {},
                "config_summary": {},
                "accepted": False,
                "diff": diff,
                "error": result["error"],
            }
        else:
            new_cd = result.get("metrics", {}).get("val_loss_mean", float("inf"))
            accepted = isinstance(new_cd, (int, float)) and new_cd < best_cd

            if accepted:
                print(f"\n*** ACCEPTED *** CD improved: {best_cd:.6f} -> {new_cd:.6f}")
            else:
                print(f"\nREJECTED: CD did not improve ({new_cd:.6f} >= {best_cd:.6f})")
                TRAIN_FILE.write_text(current_code)
                print("Reverted train_pcdiff.py")

            entry = {
                "experiment_id": f"exp_{exp_idx + 1:04d}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": result.get("metrics", {}),
                "config_summary": result.get("config", {}),
                "accepted": accepted,
                "diff": diff,
            }

        with open(RESULTS_DIR / "experiments.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        history.append(entry)

        print(f"\nExperiment {exp_idx + 1} complete. Total accepted: {sum(1 for h in history if h.get('accepted'))}")

    # Summary
    print(f"\n{'='*60}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {len(history)}")
    print(f"Accepted: {sum(1 for h in history if h.get('accepted'))}")
    print(f"Best CD: {get_best_cd(history):.6f}" if get_best_cd(history) < float("inf") else "No successful experiments")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCDiff autoresearch orchestrator")
    parser.add_argument("--max-experiments", type=int, default=MAX_EXPERIMENTS,
                        help=f"Maximum number of experiments (default: {MAX_EXPERIMENTS})")
    parser.add_argument("--time-budget", type=int, default=DEFAULT_TIME_BUDGET,
                        help=f"Time budget per experiment in seconds (default: {DEFAULT_TIME_BUDGET})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show proposed changes without running experiments")
    args = parser.parse_args()

    run_autoresearch_loop(
        max_experiments=args.max_experiments,
        time_budget=args.time_budget,
        dry_run=args.dry_run,
    )
