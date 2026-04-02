#!/usr/bin/env python3
"""
Generate visualization plots for the Stage-1 ablation study.
Reads completed_results.json and manifest.json to show sampling speed/quality trade-offs.
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

ABLATION_DIR = Path(__file__).parent / "runs" / "stage1_ablation"
OUTPUT_DIR = Path(__file__).parent
BASELINE_RESULTS = Path(__file__).parent / "baseline_results.json"


def load_ablation_data():
    """Load ablation study data."""
    completed = {}
    completed_path = ABLATION_DIR / "completed_results.json"
    if completed_path.exists():
        with open(completed_path) as f:
            completed = json.load(f)

    manifest = {}
    manifest_path = ABLATION_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    return completed, manifest


def load_baseline():
    """Load baseline benchmark results."""
    if BASELINE_RESULTS.exists():
        with open(BASELINE_RESULTS) as f:
            return json.load(f)
    return None


def plot_ablation():
    completed, manifest = load_ablation_data()
    baseline_data = load_baseline()

    # Parse completed results into per-config stats
    configs = {}
    for key, val in completed.items():
        run_id = key.split("/")[0]
        case_name = "/".join(key.split("/")[1:])
        if val.get("status") == "success":
            meta = val.get("metadata", {})
            time_s = meta.get("processing_time_seconds", 0)
            steps = meta.get("sampling_steps", 0)
            ens = meta.get("num_ensemble", 1)
            configs.setdefault(
                run_id,
                {
                    "times": [],
                    "cases": [],
                    "sampling_steps": steps,
                    "ensemble_size": ens,
                    "method": run_id.split("-")[0],
                },
            )
            configs[run_id]["times"].append(time_s)
            configs[run_id]["cases"].append(case_name)

    # Parse manifest for planned runs
    planned_runs = manifest.get("runs", [])

    # =========================================================================
    # Figure: Ablation Study Overview
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("PCDiff Stage-1 Ablation Study — Sampling Speed & Quality", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: Planned ablation matrix ---
    ax = axes[0, 0]
    # Build matrix of planned configs
    methods_steps = {}
    for run in planned_runs:
        method = run["sampling_method"]
        steps = run["sampling_steps"]
        ens = run["ensemble_size"]
        run_id = run["run_id"]
        key = (method, steps)
        methods_steps.setdefault(key, {})[ens] = run_id

    # Check completion status
    matrix_data = []
    for run in planned_runs:
        run_id = run["run_id"]
        n_completed = len(configs.get(run_id, {}).get("cases", []))
        status = "completed" if n_completed >= 3 else ("partial" if n_completed > 0 else "pending")
        matrix_data.append(
            {
                "run_id": run_id,
                "method": run["sampling_method"],
                "steps": run["sampling_steps"],
                "ensemble": run["ensemble_size"],
                "n_completed": n_completed,
                "status": status,
            }
        )

    # Plot as a table-like visual
    methods = sorted(set(r["method"] for r in matrix_data))
    steps_list = sorted(set(r["steps"] for r in matrix_data), reverse=True)
    ens_list = sorted(set(r["ensemble"] for r in matrix_data))

    cell_text = []
    cell_colors = []
    for method in methods:
        for steps in steps_list:
            row = []
            row_colors = []
            for ens in ens_list:
                matches = [
                    r for r in matrix_data if r["method"] == method and r["steps"] == steps and r["ensemble"] == ens
                ]
                if matches:
                    r = matches[0]
                    row.append(f"{r['n_completed']}/3")
                    if r["status"] == "completed":
                        row_colors.append("#90EE90")
                    elif r["status"] == "partial":
                        row_colors.append("#FFD700")
                    else:
                        row_colors.append("#FFB6C1")
                else:
                    row.append("-")
                    row_colors.append("#F0F0F0")
            cell_text.append(row)
            cell_colors.append(row_colors)

    row_labels = [f"{m.upper()}\n{s} steps" for m in methods for s in steps_list]
    col_labels = [f"ens={e}" for e in ens_list]

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax.axis("off")
    ax.set_title(
        "Ablation Matrix — Completion Status\n(green=done, yellow=partial, pink=pending)",
        fontsize=11,
        fontweight="bold",
    )

    # --- Panel 2: Processing time by config (for completed) ---
    ax = axes[0, 1]
    if configs:
        sorted_configs = sorted(configs.items(), key=lambda x: np.mean(x[1]["times"]))
        names = [c[0] for c in sorted_configs]
        mean_times = [np.mean(c[1]["times"]) for c in sorted_configs]
        std_times = [np.std(c[1]["times"]) if len(c[1]["times"]) > 1 else 0 for c in sorted_configs]
        n_cases = [len(c[1]["cases"]) for c in sorted_configs]

        colors = ["#4CAF50" if "ddim" in n else "#2196F3" for n in names]
        bars = ax.barh(
            range(len(names)), mean_times, xerr=std_times, color=colors, edgecolor="white", linewidth=0.5, capsize=3
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"{n}\n(n={nc})" for n, nc in zip(names, n_cases)], fontsize=8)
        ax.set_xlabel("Processing Time (seconds)", fontsize=10)
        ax.set_title("Avg Processing Time per Config", fontsize=11, fontweight="bold")

        for bar, t in zip(bars, mean_times):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2, f"{t:.0f}s", va="center", fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
    else:
        ax.text(
            0.5, 0.5, "No completed ablation runs yet", ha="center", va="center", fontsize=14, transform=ax.transAxes
        )
        ax.axis("off")

    # --- Panel 3: Baseline results overview ---
    ax = axes[1, 0]
    if baseline_data and baseline_data.get("results"):
        results = baseline_data["results"]
        times = [r.get("processing_time", 0) for r in results if r.get("processing_time")]
        cases = [f"{r['case']}_{r['defect'][:8]}" for r in results]

        if times:
            ax.bar(range(len(times)), times, color="#FF9800", edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(times)))
            ax.set_xticklabels(cases, fontsize=7, rotation=45, ha="right")
            ax.set_ylabel("Processing Time (s)", fontsize=10)
            ax.set_title(f"Baseline Full Pipeline — {len(results)} Cases Completed", fontsize=11, fontweight="bold")
            ax.axhline(y=np.mean(times), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(times):.1f}s")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5, 0.5, "No timing data in baseline", ha="center", va="center", fontsize=14, transform=ax.transAxes
            )
    else:
        ax.text(0.5, 0.5, "No baseline results yet", ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.axis("off")

    # --- Panel 4: Summary statistics ---
    ax = axes[1, 1]
    ax.axis("off")

    total_planned = len(planned_runs)
    total_completed_configs = len([c for c in configs.values() if len(c["cases"]) >= 3])
    total_partial = len([c for c in configs.values() if 0 < len(c["cases"]) < 3])
    total_cases = sum(len(c["cases"]) for c in configs.values())

    summary_text = [
        "ABLATION STUDY SUMMARY",
        "=" * 40,
        "",
        f"Planned configurations:  {total_planned}",
        f"Completed configs:      {total_completed_configs}",
        f"Partial configs:        {total_partial}",
        f"Pending configs:        {total_planned - total_completed_configs - total_partial}",
        f"Total cases processed:  {total_cases}",
        "",
        "SAMPLING METHODS TESTED:",
        "  DDPM: 1000 steps × ens {1,3,5}",
        "  DDIM: {25,50,100,250} steps × ens {1,3,5}",
        "",
    ]

    if baseline_data:
        bl = baseline_data
        summary_text.extend(
            [
                "BASELINE:",
                f"  Completed: {bl.get('completed', '?')}/{bl.get('total_jobs', '?')} cases",
            ]
        )
        if bl.get("results"):
            times = [r.get("processing_time", 0) for r in bl["results"] if r.get("processing_time")]
            if times:
                summary_text.append(f"  Avg time:  {np.mean(times):.1f}s")

    ax.text(
        0.05,
        0.95,
        "\n".join(summary_text),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.text(
        0.5,
        0.005,
        "Stage-1 ablation: finding fastest sampling config that preserves reconstruction quality",
        ha="center",
        fontsize=9,
        style="italic",
    )

    outpath = OUTPUT_DIR / "ablation_study_plot.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    plot_ablation()
    print("Done!")
