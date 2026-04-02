from __future__ import annotations

import csv
from pathlib import Path


def load_manifest_records(path: str | Path) -> list[dict[str, object]]:
    manifest_path = Path(path).expanduser().resolve()
    records: list[dict[str, object]] = []

    with manifest_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            normalized = [cell.strip() for cell in row if cell.strip()]
            if not normalized or normalized[0].startswith("#"):
                continue
            if normalized[0] in {"base_path", "pointcloud_path", "sample_points_path"}:
                continue
            records.append(parse_manifest_row(normalized, manifest_path.parent))

    return records


def parse_manifest_row(row: list[str], manifest_dir: Path) -> dict[str, object]:
    if len(row) == 1:
        base_path = _resolve_path(row[0], manifest_dir)
        pointcloud_path = Path(str(base_path.with_suffix("")) + "_pc.npz")
        gt_psr_path = Path(str(base_path) + "_vox.npz")
        return {
            "format": "legacy",
            "pointcloud_path": pointcloud_path,
            "gt_psr_path": gt_psr_path,
        }

    if len(row) == 2:
        return {
            "format": "explicit_npz",
            "pointcloud_path": _resolve_path(row[0], manifest_dir),
            "gt_psr_path": _resolve_path(row[1], manifest_dir),
        }

    if len(row) in {4, 5}:
        return {
            "format": "generated_completion",
            "sample_points_path": _resolve_path(row[0], manifest_dir),
            "shift_path": _resolve_path(row[1], manifest_dir),
            "scale_path": _resolve_path(row[2], manifest_dir),
            "gt_psr_path": _resolve_path(row[3], manifest_dir),
            "ensemble_index": int(row[4]) if len(row) == 5 else 0,
        }

    raise ValueError(f"Unsupported voxelization manifest row. Expected 1, 2, 4, or 5 columns, got {len(row)}: {row}")


def _resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    return path
