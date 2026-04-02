#!/usr/bin/env python3
"""
run_single_experiment.py — Run a single PCDiff experiment with config overrides.

Reads a JSON config file, applies hyperparameter overrides to train_pcdiff.py
module-level variables, then runs training. No internet/LLM needed.

Usage:
    python run_single_experiment.py --config perun_experiments/perun_001.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
AUTORESEARCH_DIR = PROJECT_DIR / "autoresearch"
RESULTS_DIR = AUTORESEARCH_DIR / "results" / "perun"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    args = parser.parse_args()

    # Load config
    config = json.loads(Path(args.config).read_text())
    exp_id = config["id"]
    exp_name = config["name"]
    changes = config.get("changes", {})
    time_budget = config.get("time_budget", 3600)

    print(f"Experiment: {exp_id} ({exp_name})")
    print(f"Description: {config.get('desc', 'N/A')}")
    print(f"Time budget: {time_budget}s ({time_budget / 60:.0f} min)")
    print(f"Overrides: {json.dumps(changes, indent=2)}")
    print()

    # Set up results directory for this experiment
    exp_results_dir = RESULTS_DIR / exp_id
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    # We need to modify train_pcdiff.py's globals before importing/running it.
    # Strategy: read the file, apply overrides, write a temp copy, run it.

    train_file = AUTORESEARCH_DIR / "train_pcdiff.py"
    train_source = train_file.read_text()

    # Apply overrides by modifying the source
    modified_source = train_source
    for key, value in changes.items():
        # Find and replace the variable assignment
        import re

        if isinstance(value, str):
            replacement = f'{key} = "{value}"'
        elif isinstance(value, bool):
            replacement = f"{key} = {value}"
        elif isinstance(value, float):
            replacement = f"{key} = {value}"
        elif isinstance(value, int):
            replacement = f"{key} = {value}"
        else:
            replacement = f"{key} = {value}"

        # Match patterns like: KEY = <value>  # optional comment
        pattern = rf"^({key}\s*=\s*)(.+?)(\s*#.*)?$"
        new_source = re.sub(pattern, replacement, modified_source, count=1, flags=re.MULTILINE)
        if new_source == modified_source:
            print(f"WARNING: Could not apply override for {key}")
        else:
            modified_source = new_source
            print(f"  Applied: {replacement}")

    # Override RESULTS_DIR — it's imported from prepare_pcdiff, so we inject an override after imports
    import_override = f'\n# === PERUN EXPERIMENT OVERRIDE ===\nfrom pathlib import Path as _Path\nRESULTS_DIR = _Path("{exp_results_dir}")\nRESULTS_DIR.mkdir(parents=True, exist_ok=True)\n# === END OVERRIDE ===\n'

    # Insert after the prepare_pcdiff import block
    insert_marker = "NUM_POINTS, NUM_NN, SV_POINTS, RESULTS_DIR,"
    if insert_marker in modified_source:
        idx = modified_source.index(insert_marker)
        # Find end of the import statement (closing paren + newline)
        close_idx = modified_source.index(")", idx) + 1
        modified_source = modified_source[:close_idx] + "\n" + import_override + modified_source[close_idx:]

    # Write modified training script
    temp_train_file = exp_results_dir / "train_pcdiff_modified.py"
    temp_train_file.write_text(modified_source)

    print(f"\nModified training script: {temp_train_file}")
    print(f"Results directory: {exp_results_dir}")
    print(f"\n{'=' * 60}")
    print("Starting training...")
    print(f"{'=' * 60}\n")

    # Run training
    t_start = time.time()

    # Change to autoresearch dir (so imports work)
    os.chdir(str(AUTORESEARCH_DIR))

    # Import and modify the module
    sys.path.insert(0, str(AUTORESEARCH_DIR))

    # Override RESULTS_DIR env var so prepare_pcdiff picks it up
    os.environ["PCDIFF_RESULTS_DIR"] = str(exp_results_dir)

    # Run the modified script from the autoresearch directory so imports work
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(AUTORESEARCH_DIR) + ":" + str(PROJECT_DIR / "pcdiff") + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(temp_train_file), "--time-budget", str(time_budget)],
        cwd=str(AUTORESEARCH_DIR),
        env=env,
        timeout=time_budget + 600,  # 10 min grace period
    )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Experiment {exp_id} finished in {elapsed:.0f}s (exit code: {result.returncode})")
    print(f"{'=' * 60}")

    # Write summary
    summary = {
        "experiment_id": exp_id,
        "name": exp_name,
        "changes": changes,
        "time_budget": time_budget,
        "elapsed_seconds": elapsed,
        "exit_code": result.returncode,
    }

    # Try to read metrics from the results
    metrics_file = exp_results_dir / "checkpoints" / "best.pth"
    if not metrics_file.exists():
        metrics_file = exp_results_dir / "checkpoints" / "latest.pth"

    best_val_file = exp_results_dir / "best_val_loss.txt"
    if best_val_file.exists():
        summary["best_val_loss"] = float(best_val_file.read_text().strip())

    summary_path = exp_results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
