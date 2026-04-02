from __future__ import annotations

import csv
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def summarize_numeric_fields(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if not values:
            summary[field] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue
        summary[field] = {
            "count": len(values),
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def collect_git_info(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)

    def run_git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_short": status.splitlines()[:50] if status else [],
    }


def collect_system_info(device: Any = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda

        if torch.cuda.is_available():
            if device is None:
                device_index = torch.cuda.current_device()
            elif isinstance(device, torch.device):
                device_index = device.index if device.index is not None else torch.cuda.current_device()
            elif isinstance(device, int):
                device_index = device
            else:
                device_index = torch.cuda.current_device()

            props = torch.cuda.get_device_properties(device_index)
            info["gpu"] = {
                "index": device_index,
                "name": props.name,
                "total_memory_mb": round(props.total_memory / (1024**2), 2),
            }
    except Exception:
        pass

    return info


def build_stage_report(
    *,
    stage_name: str,
    dataset: str,
    repo_root: str | Path,
    started_at: str,
    finished_at: str,
    command: list[str],
    config: dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    device: Any = None,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "stage_name": stage_name,
        "dataset": dataset,
        "started_at": started_at,
        "finished_at": finished_at,
        "command": command,
        "git": collect_git_info(repo_root),
        "system": collect_system_info(device=device),
    }
    if config is not None:
        report["config"] = config
    if args is not None:
        report["args"] = args
    if outputs is not None:
        report["outputs"] = outputs
    if summary is not None:
        report["summary"] = summary
    if extra is not None:
        report["extra"] = extra
    return report
