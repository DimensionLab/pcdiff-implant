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
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = SCRIPT_DIR / "train_pcdiff.py"
PROGRAM_FILE = SCRIPT_DIR / "program_pcdiff.md"
RESULTS_DIR = SCRIPT_DIR / "results"
AUDIT_DIR = RESULTS_DIR / "audit"

# LLM config
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-5.3-codex"  # Coding-optimized model, fewer syntax errors
MAX_RETRIES_PER_EXPERIMENT = 4  # Retry with error feedback on syntax/apply failures

# Experiment config
DEFAULT_TIME_BUDGET = 900  # 15 minutes per experiment
MAX_EXPERIMENTS = 100

# ---------------------------------------------------------------------------
# LLM Client (OpenRouter)
# ---------------------------------------------------------------------------


def call_llm(messages: list, temperature: float = 0.7) -> str:
    """Call LLM via OpenRouter API. Returns the assistant response text."""
    import urllib.error
    import urllib.request

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_file = Path(__file__).resolve().parent.parent / "crainial_app" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Set it in environment or crainial_app/.env")

    payload = json.dumps(
        {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 8192,
        }
    ).encode("utf-8")

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


def format_history_summary(history: list, last_n: int = 15) -> str:
    """Format recent experiment history for the LLM prompt."""
    if not history:
        return "No experiments run yet."

    # Track repeated failure patterns to warn LLM
    failure_patterns = {}
    for exp in history:
        err = exp.get("error", "")
        if err:
            # Extract what the experiment tried from diff
            diff = exp.get("diff", "")
            adds = [l.lstrip("+").strip() for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++")][
                :3
            ]
            key = " | ".join(adds)[:80] if adds else "unknown"
            failure_patterns[key] = failure_patterns.get(key, 0) + 1

    lines = []

    # Show repeated failures as a warning
    repeated = {k: v for k, v in failure_patterns.items() if v >= 3}
    if repeated:
        lines.append("⚠️ REPEATEDLY FAILED APPROACHES (DO NOT retry these):")
        for pattern, count in sorted(repeated.items(), key=lambda x: -x[1]):
            lines.append(f"  - Failed {count}x: {pattern}")
        lines.append("")

    recent = history[-last_n:]
    for exp in recent:
        err = exp.get("error", "")
        if err:
            status = f"ERROR: {err[:80]}"
        elif exp.get("accepted"):
            status = "ACCEPTED"
        else:
            status = "REJECTED"
        cd = exp.get("metrics", {}).get("val_loss_mean", "N/A")
        cd_str = f"{cd:.6f}" if isinstance(cd, float) else str(cd)
        lines.append(f"- [{status}] {exp.get('experiment_id', '?')}: val_loss={cd_str}")
        if exp.get("diff") and not err:
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


def apply_search_replace_blocks(original: str, blocks_text: str) -> str:
    """Apply SEARCH/REPLACE edit blocks to the original code.

    Format:
        <<<<<<< SEARCH
        exact lines to find
        =======
        replacement lines
        >>>>>>> REPLACE
    """
    result = original
    # Split into individual blocks
    parts = blocks_text.split("<<<<<<< SEARCH")
    for part in parts[1:]:  # skip text before first block
        if "=======" not in part or ">>>>>>> REPLACE" not in part:
            continue
        search_section, rest = part.split("=======", 1)
        replace_section = rest.split(">>>>>>> REPLACE", 1)[0]
        search_text = search_section.strip("\n")
        replace_text = replace_section.strip("\n")

        if search_text not in result:
            # Try with slightly relaxed whitespace matching
            import re

            # Normalize trailing whitespace per line for matching
            search_lines = [l.rstrip() for l in search_text.splitlines()]
            result_lines = [l.rstrip() for l in result.splitlines()]
            search_joined = "\n".join(search_lines)
            result_joined = "\n".join(result_lines)
            if search_joined in result_joined:
                result_joined = result_joined.replace(search_joined, replace_text.rstrip("\n"), 1)
                result = result_joined + "\n"
            else:
                raise ValueError(f"SEARCH block not found in file:\n{search_text[:200]}...")
        else:
            result = result.replace(search_text, replace_text, 1)

    return result


def propose_modification(
    current_code: str, program: str, history_summary: str, best_cd: float, previous_error: str = None
) -> str:
    """Ask LLM to propose a modification to train_pcdiff.py using SEARCH/REPLACE blocks."""

    error_context = ""
    if previous_error:
        error_context = (
            f"\n\n## IMPORTANT — Your Previous Attempt Failed\n"
            f"Error: {previous_error}\n"
            f"Fix this issue or try a completely different modification."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an ML research agent running autonomous experiments on a Point Cloud Diffusion model. "
                "Your goal is to minimize validation loss (MSE on noise prediction) by modifying the training script. "
                "You make ONE focused change per experiment.\n\n"
                "Respond with SEARCH/REPLACE edit blocks that describe your change. Use this exact format:\n\n"
                "<<<<<<< SEARCH\n"
                "exact lines from the current file to find\n"
                "=======\n"
                "replacement lines\n"
                ">>>>>>> REPLACE\n\n"
                "CRITICAL RULES:\n"
                "- The SEARCH block must match the current file EXACTLY — copy-paste the lines, preserving all whitespace and indentation\n"
                "- Include 3-5 context lines around the change in each SEARCH block so it matches unambiguously\n"
                "- You may use multiple SEARCH/REPLACE blocks for related changes across different parts of the file\n"
                "- Do NOT output the entire file — only the edit blocks\n"
                "- If you add new functions/classes, make sure any code that calls them is also updated in a separate SEARCH/REPLACE block\n"
                "- If you add imports, put them in their own SEARCH/REPLACE block targeting the import section\n"
                "- Before the edit blocks, write a one-line comment explaining your change\n"
                "- NEVER include ======= or <<<<<<< or >>>>>>> markers inside the code itself"
            ),
        },
        {
            "role": "user",
            "content": f"""## Research Program
{program}

## Current Best Validation Loss
{best_cd if best_cd < float("inf") else "Not yet established (first run)"}

## Recent Experiment History
{history_summary}
{error_context}

## Current train_pcdiff.py
```python
{current_code}
```

Based on the research program and experiment history, propose your next modification.
Make ONE focused change (or a small coherent group of related changes).
Respond with SEARCH/REPLACE edit blocks.""",
        },
    ]

    response = call_llm(messages, temperature=0.4)

    if "<<<<<<< SEARCH" not in response:
        raise ValueError("LLM response does not contain SEARCH/REPLACE blocks")

    modified_code = apply_search_replace_blocks(current_code, response)
    return modified_code


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

    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    started_at = datetime.now(timezone.utc)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=time_budget + 300,  # Extra 5 min buffer for eval
        )
        finished_at = datetime.now(timezone.utc)
        duration_sec = (finished_at - started_at).total_seconds()

        print("STDOUT (last 30 lines):")
        stdout_lines = result.stdout.strip().split("\n")
        for line in stdout_lines[-30:]:
            print(f"  {line}")

        if result.returncode != 0:
            print("\nSTDERR (last 20 lines):")
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-20:]:
                print(f"  {line}")
            return {
                "error": f"Process exited with code {result.returncode}",
                "stderr": result.stderr[-2000:],
                "stdout_full": result.stdout,
                "stderr_full": result.stderr,
                "returncode": result.returncode,
                "command": cmd,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_sec": duration_sec,
            }

        # Parse result from output (last line should be JSON)
        for line in reversed(stdout_lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    payload["_run_meta"] = {
                        "command": cmd,
                        "returncode": result.returncode,
                        "started_at": started_at.isoformat(),
                        "finished_at": finished_at.isoformat(),
                        "duration_sec": duration_sec,
                        "stdout_full": result.stdout,
                        "stderr_full": result.stderr,
                    }
                    return payload
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
                payload = json.loads("\n".join(json_lines))
                payload["_run_meta"] = {
                    "command": cmd,
                    "returncode": result.returncode,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_sec": duration_sec,
                    "stdout_full": result.stdout,
                    "stderr_full": result.stderr,
                }
                return payload
            except json.JSONDecodeError:
                pass

        return {
            "error": "Could not parse results from output",
            "stdout": result.stdout[-2000:],
            "stdout_full": result.stdout,
            "stderr_full": result.stderr,
            "returncode": result.returncode,
            "command": cmd,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_sec": duration_sec,
        }

    except subprocess.TimeoutExpired:
        finished_at = datetime.now(timezone.utc)
        duration_sec = (finished_at - started_at).total_seconds()
        return {
            "error": "Experiment timed out",
            "command": cmd,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_sec": duration_sec,
        }
    except Exception as e:
        finished_at = datetime.now(timezone.utc)
        duration_sec = (finished_at - started_at).total_seconds()
        return {
            "error": str(e),
            "command": cmd,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_sec": duration_sec,
        }


def persist_experiment_artifacts(
    experiment_id: str,
    train_before: str,
    train_after: str,
    diff: str,
    result: dict,
    accepted: bool,
) -> Path:
    """Persist full audit artifacts so every experiment is reproducible and reviewable."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_dir = AUDIT_DIR / f"{experiment_id}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "train_before.py").write_text(train_before)
    (exp_dir / "train_after.py").write_text(train_after)
    (exp_dir / "proposal.diff").write_text(diff or "")

    run_meta = result.get("_run_meta", {}) if isinstance(result, dict) else {}
    if run_meta.get("stdout_full") is not None:
        (exp_dir / "stdout.log").write_text(run_meta.get("stdout_full", ""))
    if run_meta.get("stderr_full") is not None:
        (exp_dir / "stderr.log").write_text(run_meta.get("stderr_full", ""))

    artifact_payload = {
        "experiment_id": experiment_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "result": result,
    }
    (exp_dir / "result.json").write_text(json.dumps(artifact_payload, indent=2))

    return exp_dir


def maybe_commit_audit_logs(experiment_id: str, accepted: bool) -> None:
    """Best-effort git commit of experiment artifacts for auditability."""
    repo_root = SCRIPT_DIR.parent
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        print("Skipping git commit: repository not found.")
        return

    add_result = subprocess.run(
        ["git", "add", "autoresearch/results", "autoresearch/train_pcdiff.py"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        print(f"Skipping git commit: git add failed ({add_result.stderr.strip()})")
        return

    status = "accepted" if accepted else "rejected"
    message = f"autoresearch: audit {experiment_id} ({status})"
    commit_result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if commit_result.returncode == 0:
        print(f"Committed experiment audit logs: {message}")
        return

    stderr = (commit_result.stderr or "").strip()
    stdout = (commit_result.stdout or "").strip()
    if "nothing to commit" in stderr.lower() or "nothing to commit" in stdout.lower():
        print(f"No changes to commit for {experiment_id}.")
    else:
        print(f"Warning: git commit failed for {experiment_id}: {stderr or stdout}")


def run_autoresearch_loop(
    max_experiments: int = MAX_EXPERIMENTS,
    time_budget: int = DEFAULT_TIME_BUDGET,
    dry_run: bool = False,
    commit_logs: bool = False,
):
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
        print(f"\n{'#' * 60}")
        print(f"# EXPERIMENT {exp_idx + 1}/{max_experiments}")
        print(f"{'#' * 60}")

        # Read current code
        current_code = TRAIN_FILE.read_text()
        history_summary = format_history_summary(history)
        best_cd = get_best_cd(history)
        print(f"Current best CD: {best_cd:.6f}" if best_cd < float("inf") else "No best CD yet")

        # Propose modification with retry on syntax/apply errors
        proposed_code = None
        diff = None
        last_error = None

        for attempt in range(MAX_RETRIES_PER_EXPERIMENT + 1):
            if attempt > 0:
                print(f"\n  Retry {attempt}/{MAX_RETRIES_PER_EXPERIMENT} (previous error: {last_error})")

            print("\nAsking LLM for modification proposal...")
            try:
                proposed_code = propose_modification(
                    current_code,
                    program,
                    history_summary,
                    best_cd,
                    previous_error=last_error if attempt > 0 else None,
                )
            except Exception as e:
                last_error = str(e)
                print(f"LLM/apply failed: {e}")
                time.sleep(10)
                continue

            # Compute diff
            diff = compute_diff(current_code, proposed_code)
            if not diff.strip():
                print("LLM proposed no changes. Skipping.")
                last_error = "No changes proposed"
                continue

            print(f"\n--- Proposed diff ({len(diff.splitlines())} lines) ---")
            for line in diff.splitlines()[:40]:
                print(f"  {line}")
            if len(diff.splitlines()) > 40:
                print(f"  ... ({len(diff.splitlines()) - 40} more lines)")

            # Verify syntax before writing
            try:
                compile(proposed_code, str(TRAIN_FILE), "exec")
            except SyntaxError as e:
                # Show context around the error line so LLM can fix it
                code_lines = proposed_code.splitlines()
                lineno = e.lineno or 0
                start = max(0, lineno - 3)
                end = min(len(code_lines), lineno + 2)
                context_lines = []
                for i in range(start, end):
                    marker = ">>>" if i == lineno - 1 else "   "
                    context_lines.append(f"{marker} {i + 1}: {code_lines[i]}")
                context_str = "\n".join(context_lines)
                last_error = f"Syntax error at line {lineno}: {e.msg}\n{context_str}"
                print(f"  Syntax error: {e}")
                proposed_code = None
                continue

            # Syntax OK — break out of retry loop
            break

        if proposed_code is None or diff is None or not diff.strip():
            print(f"All attempts failed for experiment {exp_idx + 1}. Logging and moving on.")
            experiment_id = f"exp_{exp_idx + 1:04d}"
            entry = {
                "experiment_id": experiment_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": {},
                "config_summary": {},
                "accepted": False,
                "diff": diff or "",
                "error": last_error or "All retries exhausted",
            }
            persist_experiment_artifacts(
                experiment_id=experiment_id,
                train_before=current_code,
                train_after=current_code,
                diff=diff or "",
                result={"error": last_error or "All retries exhausted"},
                accepted=False,
            )
            with open(RESULTS_DIR / "experiments.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            history.append(entry)
            if commit_logs:
                maybe_commit_audit_logs(experiment_id, accepted=False)
            continue

        if dry_run:
            print("[DRY RUN] Would apply this change and run experiment")
            continue

        # Apply verified modification
        TRAIN_FILE.write_text(proposed_code)
        print(f"\nApplied modification to {TRAIN_FILE}")

        # Run experiment
        # Use latest checkpoint for incremental training
        latest_ckpt = RESULTS_DIR / "checkpoints" / "latest.pth"
        ckpt_arg = str(latest_ckpt) if latest_ckpt.exists() else None
        result = run_experiment(time_budget, checkpoint=ckpt_arg)
        experiment_id = f"exp_{exp_idx + 1:04d}"

        if "error" in result:
            print(f"\nExperiment FAILED: {result['error']}")
            # Revert
            TRAIN_FILE.write_text(current_code)
            print("Reverted train_pcdiff.py")

            entry = {
                "experiment_id": experiment_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": {},
                "config_summary": {},
                "accepted": False,
                "diff": diff,
                "error": result["error"],
            }
            persist_experiment_artifacts(
                experiment_id=experiment_id,
                train_before=current_code,
                train_after=proposed_code,
                diff=diff,
                result=result,
                accepted=False,
            )
            if commit_logs:
                maybe_commit_audit_logs(experiment_id, accepted=False)
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
                "experiment_id": experiment_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": result.get("metrics", {}),
                "config_summary": result.get("config", {}),
                "accepted": accepted,
                "diff": diff,
            }
            persist_experiment_artifacts(
                experiment_id=experiment_id,
                train_before=current_code,
                train_after=proposed_code,
                diff=diff,
                result=result,
                accepted=accepted,
            )
            if commit_logs:
                maybe_commit_audit_logs(experiment_id, accepted=accepted)

        with open(RESULTS_DIR / "experiments.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        history.append(entry)

        print(f"\nExperiment {exp_idx + 1} complete. Total accepted: {sum(1 for h in history if h.get('accepted'))}")

    # Summary
    print(f"\n{'=' * 60}")
    print("AUTORESEARCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total experiments: {len(history)}")
    print(f"Accepted: {sum(1 for h in history if h.get('accepted'))}")
    print(
        f"Best CD: {get_best_cd(history):.6f}" if get_best_cd(history) < float("inf") else "No successful experiments"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCDiff autoresearch orchestrator")
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=MAX_EXPERIMENTS,
        help=f"Maximum number of experiments (default: {MAX_EXPERIMENTS})",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=DEFAULT_TIME_BUDGET,
        help=f"Time budget per experiment in seconds (default: {DEFAULT_TIME_BUDGET})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show proposed changes without running experiments")
    parser.add_argument(
        "--commit-logs",
        action="store_true",
        help="After each experiment, git-commit audit logs in autoresearch/results (best effort).",
    )
    args = parser.parse_args()

    run_autoresearch_loop(
        max_experiments=args.max_experiments,
        time_budget=args.time_budget,
        dry_run=args.dry_run,
        commit_logs=args.commit_logs,
    )
