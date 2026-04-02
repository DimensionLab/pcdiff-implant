#!/usr/bin/env python3
"""
Generate visualization plots for PCDiff autoresearch experiments on Perun HPC.
Reads summary.json from each experiment directory and produces comparison plots.
"""

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RESULTS_DIR = Path(__file__).parent / "results" / "perun"
OUTPUT_DIR = Path(__file__).parent


def load_all_summaries():
    """Load all experiment summaries from perun results."""
    summaries = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        sf = RESULTS_DIR / d / "summary.json"
        if sf.is_file():
            with open(sf) as f:
                data = json.load(f)
                data["dir"] = d
                summaries.append(data)
    return summaries


def get_wave(experiment_id):
    """Extract experiment wave (v1, v2, ..., v10) from experiment ID."""
    if "_v" in experiment_id:
        parts = experiment_id.split("_")
        for p in parts:
            if p.startswith("v") and p[1:].isdigit():
                return p
    return "v1"


def plot_experiments():
    summaries = load_all_summaries()
    successful = [s for s in summaries if s.get("exit_code") == 0 and s.get("best_val_loss") is not None]
    failed = [s for s in summaries if s.get("exit_code") != 0 or s.get("best_val_loss") is None]

    if not successful:
        print("No successful experiments found!")
        return

    # Sort by val loss
    successful.sort(key=lambda x: x["best_val_loss"])

    # Group by wave
    waves = {}
    for s in successful:
        w = get_wave(s["experiment_id"])
        waves.setdefault(w, []).append(s)

    wave_order = sorted(waves.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

    # Color map for waves
    cmap = plt.cm.tab10
    wave_colors = {w: cmap(i / max(len(wave_order) - 1, 1)) for i, w in enumerate(wave_order)}

    # =========================================================================
    # Figure 1: Overview - All experiments ranked by val loss
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("PCDiff Autoresearch — Perun HPC Experiment Results", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: Horizontal bar chart ranked by val loss ---
    ax = axes[0, 0]
    top_n = min(25, len(successful))
    top = successful[:top_n]
    names = [f"{s['experiment_id']}\n({s['name'][:25]})" for s in top]
    vals = [s["best_val_loss"] for s in top]
    colors = [wave_colors[get_wave(s["experiment_id"])] for s in top]

    y_pos = np.arange(top_n)
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Best Validation Loss", fontsize=10)
    ax.set_title(f"Top {top_n} Experiments by Val Loss (lower = better)", fontsize=11, fontweight="bold")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="val_loss = 1.0")

    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=7)
    ax.legend(fontsize=8)

    # --- Panel 2: Wave progression (best per wave over time) ---
    ax = axes[0, 1]
    wave_bests = []
    for w in wave_order:
        exps = waves[w]
        best = min(exps, key=lambda x: x["best_val_loss"])
        wave_bests.append((w, best["best_val_loss"], best["experiment_id"], best["name"], len(exps)))

    x_labels = [f"{wb[0]}\n({wb[4]} exp)" for wb in wave_bests]
    y_vals = [wb[1] for wb in wave_bests]
    colors = [wave_colors[wb[0]] for wb in wave_bests]

    ax.plot(range(len(wave_bests)), y_vals, "k-", alpha=0.3, zorder=1)
    scatter = ax.scatter(range(len(wave_bests)), y_vals, c=colors, s=100, zorder=2, edgecolors="black", linewidth=0.8)

    for i, wb in enumerate(wave_bests):
        ax.annotate(
            f"{wb[1]:.4f}\n{wb[3][:20]}",
            (i, wb[1]),
            textcoords="offset points",
            xytext=(0, 12),
            fontsize=7,
            ha="center",
        )

    ax.set_xticks(range(len(wave_bests)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Best Validation Loss", fontsize=10)
    ax.set_title("Best Val Loss per Experiment Wave", fontsize=11, fontweight="bold")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Time budget vs val loss ---
    ax = axes[1, 0]
    for w in wave_order:
        exps = waves[w]
        times = [s["time_budget"] / 3600 for s in exps]
        losses = [s["best_val_loss"] for s in exps]
        ax.scatter(
            times, losses, c=[wave_colors[w]] * len(exps), label=w, s=60, alpha=0.8, edgecolors="black", linewidth=0.5
        )

    ax.set_xlabel("Time Budget (hours)", fontsize=10)
    ax.set_ylabel("Best Validation Loss", fontsize=10)
    ax.set_title("Time Budget vs Val Loss", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Key hyperparameter impact ---
    ax = axes[1, 1]
    # Extract key changes
    hp_impact = {}
    for s in successful:
        changes = s.get("changes", {})
        for k, v in changes.items():
            hp_impact.setdefault(k, []).append(s["best_val_loss"])

    # Show top hyperparam keys by occurrence
    hp_keys = sorted(hp_impact.keys(), key=lambda k: len(hp_impact[k]), reverse=True)[:12]
    bp_data = [hp_impact[k] for k in hp_keys]
    bp_labels = [k.replace("_", "\n") for k in hp_keys]

    bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True, vert=True)
    for patch, color in zip(bp["boxes"], plt.cm.Set3(np.linspace(0, 1, len(bp_data)))):
        patch.set_facecolor(color)
    ax.set_ylabel("Val Loss Distribution", fontsize=10)
    ax.set_title("Hyperparameter Impact on Val Loss", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Summary text
    fig.text(
        0.5,
        0.005,
        f"Total: {len(summaries)} experiments ({len(successful)} successful, {len(failed)} failed) | "
        f"Best: {successful[0]['experiment_id']} ({successful[0]['name']}) = {successful[0]['best_val_loss']:.6f} | "
        f"Baseline RunPod: 1.0520",
        ha="center",
        fontsize=9,
        style="italic",
    )

    outpath = OUTPUT_DIR / "pcdiff_experiments_plot.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    plot_experiments()
    print("Done!")
