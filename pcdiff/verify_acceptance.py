#!/usr/bin/env python3
"""
Acceptance Criteria Verification Script for PCDiff.

This script verifies that a trained PCDiff checkpoint meets the acceptance criteria
defined in productize/acceptance-criteria.md:

    Minimum Thresholds:
    - DSC >= 0.85
    - bDSC >= 0.87
    - HD95 <= 2.60 mm

    Target Thresholds (paper baseline):
    - DSC >= 0.87
    - bDSC >= 0.89
    - HD95 <= 2.45 mm

Usage:
    # Verify from existing E2E evaluation results
    python pcdiff/verify_acceptance.py --eval-dir pcdiff/eval/e2e_run_name

    # Run full E2E evaluation and verify
    python pcdiff/verify_acceptance.py --checkpoint path/to/model.pth --run-eval

    # Generate frozen evaluation report
    python pcdiff/verify_acceptance.py --eval-dir pcdiff/eval/e2e_run_name --freeze-report

The frozen evaluation report is a self-contained artifact that includes:
- Git commit hash and checkpoint path
- All metrics (DDIM-50 and DDPM-1000)
- Acceptance criteria verification results
- Timestamp and environment info
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Acceptance criteria from PRD/acceptance-criteria.md
ACCEPTANCE_CRITERIA = {
    "minimum": {
        "dsc": 0.85,
        "bdsc": 0.87,
        "hd95": 2.60,
    },
    "target": {
        "dsc": 0.87,
        "bdsc": 0.89,
        "hd95": 2.45,
    },
}


@dataclass
class VerificationResult:
    """Result of acceptance criteria verification."""

    # Checkpoint info
    checkpoint_path: str
    checkpoint_hash: str
    git_commit: str
    git_branch: str

    # DDIM-50 results
    ddim_dsc: float
    ddim_bdsc: float
    ddim_hd95: float
    ddim_meets_minimum: Dict[str, bool]
    ddim_meets_target: Dict[str, bool]

    # DDPM-1000 results (optional, may not always be run)
    ddpm_dsc: Optional[float] = None
    ddpm_bdsc: Optional[float] = None
    ddpm_hd95: Optional[float] = None
    ddpm_meets_minimum: Optional[Dict[str, bool]] = None
    ddpm_meets_target: Optional[Dict[str, bool]] = None

    # Overall verification
    overall_minimum_pass: bool = False
    overall_target_pass: bool = False

    # Metadata
    timestamp: str = ""
    eval_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_git_info() -> tuple[str, str]:
    """Get current git commit hash and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        branch = "unknown"

    return commit, branch


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file (first 16 chars)."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
    except Exception:
        return "unknown"


def verify_metrics(
    dsc: float,
    bdsc: float,
    hd95: float,
    threshold_type: str = "minimum"
) -> Dict[str, bool]:
    """Verify metrics against thresholds."""
    thresholds = ACCEPTANCE_CRITERIA[threshold_type]
    return {
        "dsc": dsc >= thresholds["dsc"],
        "bdsc": bdsc >= thresholds["bdsc"],
        "hd95": hd95 <= thresholds["hd95"],  # HD95 is lower-is-better
    }


def load_eval_summary(eval_dir: Path) -> Optional[Dict[str, Any]]:
    """Load comparison_summary.json from an E2E evaluation directory."""
    summary_path = eval_dir / "comparison_summary.json"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return None

    with open(summary_path) as f:
        return json.load(f)


def verify_from_eval_dir(eval_dir: Path, checkpoint_path: Optional[Path] = None) -> VerificationResult:
    """Verify acceptance criteria from existing E2E evaluation results."""
    summary = load_eval_summary(eval_dir)
    if summary is None:
        raise FileNotFoundError(f"No comparison_summary.json found in {eval_dir}")

    # Get checkpoint info
    if checkpoint_path is None:
        # Try to infer from summary
        checkpoint_path = Path(summary.get("checkpoint", "unknown"))

    checkpoint_hash = compute_file_hash(checkpoint_path) if checkpoint_path.exists() else "unknown"
    git_commit, git_branch = get_git_info()

    # Extract DDIM results
    ddim = summary.get("ddim", {})
    ddim_dsc = ddim.get("mean_dsc", 0.0)
    ddim_bdsc = ddim.get("mean_bdsc", 0.0)
    ddim_hd95 = ddim.get("mean_hd95", 999.0)

    ddim_meets_min = verify_metrics(ddim_dsc, ddim_bdsc, ddim_hd95, "minimum")
    ddim_meets_tgt = verify_metrics(ddim_dsc, ddim_bdsc, ddim_hd95, "target")

    # Extract DDPM results (may be None if not run)
    ddpm = summary.get("ddpm", {})
    ddpm_dsc = ddpm.get("mean_dsc") if ddpm else None
    ddpm_bdsc = ddpm.get("mean_bdsc") if ddpm else None
    ddpm_hd95 = ddpm.get("mean_hd95") if ddpm else None

    ddpm_meets_min = None
    ddpm_meets_tgt = None
    if ddpm_dsc is not None:
        ddpm_meets_min = verify_metrics(ddpm_dsc, ddpm_bdsc, ddpm_hd95, "minimum")
        ddpm_meets_tgt = verify_metrics(ddpm_dsc, ddpm_bdsc, ddpm_hd95, "target")

    # Overall pass: DDIM must pass (DDPM is optional but should also pass if run)
    overall_min = all(ddim_meets_min.values())
    if ddpm_meets_min is not None:
        overall_min = overall_min and all(ddpm_meets_min.values())

    overall_tgt = all(ddim_meets_tgt.values())
    if ddpm_meets_tgt is not None:
        overall_tgt = overall_tgt and all(ddpm_meets_tgt.values())

    return VerificationResult(
        checkpoint_path=str(checkpoint_path),
        checkpoint_hash=checkpoint_hash,
        git_commit=git_commit,
        git_branch=git_branch,
        ddim_dsc=ddim_dsc,
        ddim_bdsc=ddim_bdsc,
        ddim_hd95=ddim_hd95,
        ddim_meets_minimum=ddim_meets_min,
        ddim_meets_target=ddim_meets_tgt,
        ddpm_dsc=ddpm_dsc,
        ddpm_bdsc=ddpm_bdsc,
        ddpm_hd95=ddpm_hd95,
        ddpm_meets_minimum=ddpm_meets_min,
        ddpm_meets_target=ddpm_meets_tgt,
        overall_minimum_pass=overall_min,
        overall_target_pass=overall_tgt,
        timestamp=datetime.now().isoformat(),
        eval_dir=str(eval_dir),
    )


def create_frozen_report(result: VerificationResult, output_path: Path) -> None:
    """Create a frozen evaluation report artifact."""
    report = {
        "report_type": "frozen_evaluation_report",
        "report_version": "1.0",
        "created_at": result.timestamp,
        "verification": result.to_dict(),
        "acceptance_criteria": ACCEPTANCE_CRITERIA,
        "status": {
            "minimum_thresholds": "PASS" if result.overall_minimum_pass else "FAIL",
            "target_thresholds": "PASS" if result.overall_target_pass else "FAIL",
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Frozen report saved to: {output_path}")


def print_verification_summary(result: VerificationResult) -> None:
    """Print a human-readable verification summary."""
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA VERIFICATION REPORT")
    print("=" * 60)

    print(f"\nCheckpoint: {result.checkpoint_path}")
    print(f"Checkpoint Hash: {result.checkpoint_hash}")
    print(f"Git Commit: {result.git_commit}")
    print(f"Git Branch: {result.git_branch}")
    print(f"Timestamp: {result.timestamp}")

    print("\n" + "-" * 60)
    print("DDIM-50 Results:")
    print("-" * 60)
    print(f"  DSC:   {result.ddim_dsc:.4f} (min: {ACCEPTANCE_CRITERIA['minimum']['dsc']:.2f}, "
          f"target: {ACCEPTANCE_CRITERIA['target']['dsc']:.2f}) "
          f"[{'PASS' if result.ddim_meets_minimum['dsc'] else 'FAIL'}]")
    print(f"  bDSC:  {result.ddim_bdsc:.4f} (min: {ACCEPTANCE_CRITERIA['minimum']['bdsc']:.2f}, "
          f"target: {ACCEPTANCE_CRITERIA['target']['bdsc']:.2f}) "
          f"[{'PASS' if result.ddim_meets_minimum['bdsc'] else 'FAIL'}]")
    print(f"  HD95:  {result.ddim_hd95:.4f} (min: {ACCEPTANCE_CRITERIA['minimum']['hd95']:.2f}, "
          f"target: {ACCEPTANCE_CRITERIA['target']['hd95']:.2f}) "
          f"[{'PASS' if result.ddim_meets_minimum['hd95'] else 'FAIL'}]")

    if result.ddpm_dsc is not None:
        print("\n" + "-" * 60)
        print("DDPM-1000 Results:")
        print("-" * 60)
        print(f"  DSC:   {result.ddpm_dsc:.4f} (min: {ACCEPTANCE_CRITERIA['minimum']['dsc']:.2f}, "
              f"target: {ACCEPTANCE_CRITERIA['target']['dsc']:.2f}) "
              f"[{'PASS' if result.ddpm_meets_minimum['dsc'] else 'FAIL'}]")
        print(f"  bDSC:  {result.ddpm_bdsc:.4f} (min: {ACCEPTANCE_CRITERIA['minimum']['bdsc']:.2f}, "
              f"target: {ACCEPTANCE_CRITERIA['target']['bdsc']:.2f}) "
              f"[{'PASS' if result.ddpm_meets_minimum['bdsc'] else 'FAIL'}]")
        print(f"  HD95:  {result.ddpm_hd95:.4f} (min: {ACCEPTANCE_CRITERIA['minimum']['hd95']:.2f}, "
              f"target: {ACCEPTANCE_CRITERIA['target']['hd95']:.2f}) "
              f"[{'PASS' if result.ddpm_meets_minimum['hd95'] else 'FAIL'}]")

    print("\n" + "=" * 60)
    print("OVERALL VERIFICATION STATUS")
    print("=" * 60)

    min_status = "PASS" if result.overall_minimum_pass else "FAIL"
    tgt_status = "PASS" if result.overall_target_pass else "FAIL"

    print(f"  Minimum Thresholds: {min_status}")
    print(f"  Target Thresholds:  {tgt_status}")
    print("=" * 60 + "\n")

    return result.overall_minimum_pass


def main():
    parser = argparse.ArgumentParser(
        description="Verify PCDiff checkpoint against acceptance criteria"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        help="Path to E2E evaluation output directory (contains comparison_summary.json)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to PCDiff checkpoint (.pth file)"
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run full E2E evaluation before verification (requires --checkpoint)"
    )
    parser.add_argument(
        "--freeze-report",
        action="store_true",
        help="Generate a frozen evaluation report artifact"
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        help="Output path for frozen report (default: <eval-dir>/frozen_evaluation_report.json)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="GPU IDs for E2E evaluation (comma-separated)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.run_eval and args.checkpoint is None:
        parser.error("--run-eval requires --checkpoint")

    if args.eval_dir is None and not args.run_eval:
        parser.error("Either --eval-dir or --run-eval must be specified")

    eval_dir = args.eval_dir

    # Run E2E evaluation if requested
    if args.run_eval:
        if args.checkpoint is None or not args.checkpoint.exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        # Generate eval directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = Path(f"pcdiff/eval/verification_{timestamp}")

        print(f"Running E2E evaluation...")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Output: {eval_dir}")
        print(f"  GPUs: {args.gpus}")

        # Run eval_e2e.py
        cmd = [
            sys.executable,
            "pcdiff/eval_e2e.py",
            "--pcdiff-checkpoint", str(args.checkpoint),
            "--output-dir", str(eval_dir),
            "--gpus", args.gpus,
        ]

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("E2E evaluation failed")
            sys.exit(1)

    # Verify from evaluation results
    if eval_dir is None or not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        sys.exit(1)

    try:
        result = verify_from_eval_dir(eval_dir, args.checkpoint)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print summary
    passed = print_verification_summary(result)

    # Generate frozen report if requested
    if args.freeze_report:
        report_path = args.output_report or (eval_dir / "frozen_evaluation_report.json")
        create_frozen_report(result, report_path)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
