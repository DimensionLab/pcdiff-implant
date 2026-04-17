#!/usr/bin/env python3
"""2D quality / wall-clock Pareto plot for DIM-96.

Reads `pareto_cases.csv` from `pareto_bench.py`, aggregates by (method, nfe), and
emits a scatter plot with a Pareto frontier overlay. Quality on Y, wall-clock on X
(log-scale). Points labeled by method/NFE. Produces both `pareto.png` (end-to-end
timing, the product-relevant curve) and `pareto_model_only.png` (raw model time,
to isolate the NFE-scaling claim from voxelization post-processing).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_STYLE = {
    "pcdiff": {"marker": "o", "color": "#1f77b4", "label": "pcdiff + vox"},
    "wv3":    {"marker": "s", "color": "#2ca02c", "label": "wv3 (Wodzinski 2024)"},
    "rf":     {"marker": "^", "color": "#d62728", "label": "RF (Zhou 2025, epoch-3)"},
}

QUALITY_CHOICES = {
    "dice": ("Dice (DSC)", True),        # higher is better
    "bdice_10mm": ("Boundary Dice 10mm", True),
    "hd95_mm": ("HD95 (mm)", False),     # lower is better
    "assd_mm": ("ASSD (mm)", False),
}


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row.get("status") == "ok"]


def aggregate(rows: List[Dict[str, str]], quality: str, time_field: str):
    agg: Dict[Tuple[str, int, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"q": [], "t": []}
    )
    for r in rows:
        try:
            key = (r["method"], int(r["nfe"]), r["solver"])
            agg[key]["q"].append(float(r[quality]))
            agg[key]["t"].append(float(r[time_field]))
        except (ValueError, KeyError):
            continue
    out = []
    for (method, nfe, solver), vals in sorted(agg.items()):
        qs, ts = vals["q"], vals["t"]
        if not qs:
            continue
        out.append({
            "method": method, "nfe": nfe, "solver": solver,
            "q_mean": mean(qs), "q_std": stdev(qs) if len(qs) > 1 else 0.0,
            "t_mean": mean(ts), "t_std": stdev(ts) if len(ts) > 1 else 0.0,
            "n": len(qs),
        })
    return out


def pareto_front(points, higher_is_better: bool):
    """Return indices of non-dominated points. X=time (lower=better), Y=quality."""
    kept: List[int] = []
    for i, pi in enumerate(points):
        dominated = False
        for j, pj in enumerate(points):
            if i == j:
                continue
            q_better = pj["q_mean"] >= pi["q_mean"] if higher_is_better else pj["q_mean"] <= pi["q_mean"]
            q_strict = pj["q_mean"] > pi["q_mean"] if higher_is_better else pj["q_mean"] < pi["q_mean"]
            t_better = pj["t_mean"] <= pi["t_mean"]
            t_strict = pj["t_mean"] < pi["t_mean"]
            if q_better and t_better and (q_strict or t_strict):
                dominated = True
                break
        if not dominated:
            kept.append(i)
    return kept


def render(points, quality: str, time_field: str, out_path: Path, title: str) -> None:
    q_label, higher_is_better = QUALITY_CHOICES[quality]
    fig, ax = plt.subplots(figsize=(8.5, 6))

    by_method: Dict[str, List[Dict]] = defaultdict(list)
    for p in points:
        by_method[p["method"]].append(p)

    for method, pts in by_method.items():
        style = METHOD_STYLE.get(method, {"marker": "x", "color": "gray", "label": method})
        xs = [p["t_mean"] for p in pts]
        ys = [p["q_mean"] for p in pts]
        xerr = [p["t_std"] for p in pts]
        yerr = [p["q_std"] for p in pts]
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt=style["marker"], color=style["color"],
                    label=style["label"], markersize=9, alpha=0.85, capsize=3, linewidth=1)
        for p, x, y in zip(pts, xs, ys):
            tag = "" if p["nfe"] == 1 and method == "wv3" else f" {p['nfe']}"
            ax.annotate(tag, (x, y), textcoords="offset points", xytext=(7, 4),
                        fontsize=9, color=style["color"])

    front_idx = pareto_front(points, higher_is_better)
    if front_idx:
        fp = sorted((points[i] for i in front_idx), key=lambda p: p["t_mean"])
        ax.plot([p["t_mean"] for p in fp], [p["q_mean"] for p in fp],
                linestyle="--", color="black", alpha=0.4, linewidth=1.2, label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel(f"Wall-clock / case ({time_field}, s, log scale)")
    ax.set_ylabel(q_label)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cases-csv", required=True, help="Path to pareto_cases.csv.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--quality", default="dice", choices=list(QUALITY_CHOICES))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.cases_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No successful rows in {csv_path}.")

    e2e = aggregate(rows, args.quality, "t_end_to_end_sec")
    render(e2e, args.quality, "t_end_to_end_sec", out_dir / "pareto.png",
           title="Quality vs end-to-end wall-clock (SkullBreak, DIM-96)")

    model_only = aggregate(rows, args.quality, "t_model_sec")
    render(model_only, args.quality, "t_model_sec", out_dir / "pareto_model_only.png",
           title="Quality vs raw model time (SkullBreak, DIM-96)")
    print(f"Wrote {out_dir/'pareto.png'} and {out_dir/'pareto_model_only.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
